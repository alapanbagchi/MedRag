"""Verifier agent: batched intent-relevance classification of structural units.

Chunks are only used to find candidate units. The verifier receives whole
structural units (paragraph / table / figure) plus the subquery intent and
classifies each as relevant / partially_relevant / not_relevant.

Key design points:
  - BATCHED: all candidate units for a (subquery, round) go through as few
    LLM calls as possible — requests are split only when they exceed
    VERIFY_BATCH_MAX_DOCS / VERIFY_BATCH_MAX_TOKENS.
  - BOUNDED: concurrent batch calls are capped by max_workers.
  - FAILURE != IRRELEVANCE: any infrastructure failure (quota, timeout, parse)
    yields relevance="unknown", which the orchestrator RETRIES in later
    rounds instead of silently dropping the unit like a judged rejection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("src.verifier_agent")

RELEVANCE_LEVELS = ("relevant", "partially_relevant", "not_relevant", "unknown")


class RelevanceVerdict(BaseModel):
    document_id: str
    relevance: str = "not_relevant"
    confidence: float = 0.0
    reason: str = ""


class VerificationResult(BaseModel):
    results: List[RelevanceVerdict] = Field(default_factory=list)
    overall: str = "none_relevant"


class VerifierAgent:
    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        from src.prompts.load import load_prompt

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="verifier")
        self._prompt = load_prompt("legacy", "verifier.txt")
        # ONE plain agent reused for every call; structured output is chosen
        # per-run by ask_structured().
        self.agent = Agent(self.model, system_prompt=self._prompt, name="verifier")
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------
    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, int(getattr(self.config, "max_workers", 4))))
        return self._semaphore

    def _batches(self, papers: List[dict]) -> List[List[dict]]:
        """Split papers into LLM-sized batches under doc/token budgets."""
        from src.llm.ratelimit import estimate_tokens

        max_docs = max(1, int(getattr(self.config, "verify_batch_max_docs", 6)))
        max_tokens = max(1, int(getattr(self.config, "verify_batch_max_tokens", 8000)))
        batches: List[List[dict]] = []
        current: List[dict] = []
        current_tokens = 0
        for p in papers:
            size = estimate_tokens(p.get("full_text", "")) + 64
            if current and (len(current) >= max_docs or current_tokens + size > max_tokens):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(p)
            current_tokens += size
        if current:
            batches.append(current)
        return batches

    def _prompt_for(self, papers: List[dict], query: str, evidence_required: List[str]) -> str:
        sep = "\n\n===DOCUMENT===\n\n"
        papers_text = sep.join(
            "DOCUMENT %d (id=%s):\n%s" % (i + 1, p.get("document_id", ""), p.get("full_text", ""))
            for i, p in enumerate(papers)
        )
        crit = "\n".join(f"- {c}" for c in evidence_required) if evidence_required else "(none)"
        return (
            f"ORIGINAL QUERY: {query}\n\n"
            f"SUBQUERY INTENT (what evidence we need):\n{crit}\n\n"
            f"DOCUMENTS TO CLASSIFY:\n{papers_text}\n\n"
            'Return EXACTLY ONE JSON object: {"results": [{"document_id": "<id>", '
            '"relevance": "relevant|partially_relevant|not_relevant", "confidence": 0.0-1.0, '
            '"reason": "max 20 words"}], "overall": "any_relevant|none_relevant"} '
            "with one entry per DOCUMENT id above."
        )

    async def _verify_batch(
        self, papers: List[dict], query: str, evidence_required: List[str]
    ) -> VerificationResult:
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        prompt = self._prompt_for(papers, query, evidence_required)

        async with self.semaphore:
            try:
                parsed = await ask_structured(
                    self.agent,
                    prompt,
                    VerificationResult,
                    label="verifier",
                    max_tokens=self.config.agent_max_tokens,
                    fallback_parser=lambda raw: self._parse(raw, papers),
                )
            except Exception as exc:
                # Infrastructure failure -> UNKNOWN (retryable), never a silent
                # "not_relevant" that drops the unit forever.
                logger.warning("verifier batch failed (%s); marking unknown", exc)
                trace.log("verifier_failed", error=str(exc)[:400])
                return self._unknown(papers, reason=f"verifier unavailable: {str(exc)[:120]}")
        return self._fill_missing(parsed, papers)

    async def verify_papers(
        self,
        papers: List[dict],
        query: str,
        evidence_required: List[str],
    ) -> VerificationResult:
        """Classify all ``papers`` using as few LLM calls as possible."""
        if not papers:
            return VerificationResult(results=[], overall="none_relevant")
        batches = self._batches(papers)
        results = await asyncio.gather(
            *(self._verify_batch(b, query, evidence_required) for b in batches)
        )
        merged: List[RelevanceVerdict] = []
        seen_ids: set = set()
        for res in results:
            for v in res.results:
                key = v.document_id or f"idx-{len(merged)}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    merged.append(v)
        has_relevant = any(v.relevance in ("relevant", "partially_relevant") for v in merged)
        return VerificationResult(
            results=merged,
            overall="any_relevant" if has_relevant else "none_relevant",
        )

    # ------------------------------------------------------------------
    def _fill_missing(self, parsed: VerificationResult, papers: List[dict]) -> VerificationResult:
        """Any document the model skipped gets an explicit unknown verdict."""
        known = {v.document_id for v in parsed.results}
        missing = [p for p in papers if str(p.get("document_id", "")) not in known]
        if missing:
            parsed = parsed.model_copy(deep=True)
            parsed.results.extend(self._unknown(missing).results)
        return parsed

    def _parse(self, raw: str, papers: List[dict]) -> VerificationResult:
        """Salvage verdict JSON from free text (arrays, bare objects, booleans)."""
        from src.lib import strip_think

        text = strip_think(raw or "")
        candidates = self._extract_lists(text) + self._extract_objects(text) + [text]
        for candidate in candidates:
            try:
                import json

                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict) and "results" in data:
                items, overall = data["results"], data.get("overall", "")
            elif isinstance(data, dict) and "document_id" in data:
                items, overall = [data], ""
            elif isinstance(data, list):
                items, overall = data, ""
            else:
                continue
            verdicts = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                rel = str(it.get("relevance", it.get("matches", "not_relevant")))
                rel = {"true": "relevant", "false": "not_relevant",
                       "1": "relevant", "0": "not_relevant"}.get(rel.lower(), rel)
                if rel not in ("relevant", "partially_relevant", "not_relevant"):
                    rel = "not_relevant"
                try:
                    conf = float(it.get("confidence", 0.0))
                except (TypeError, ValueError):
                    conf = 0.0
                verdicts.append(RelevanceVerdict(
                    document_id=str(it.get("document_id", it.get("id", ""))),
                    relevance=rel,
                    confidence=max(0.0, min(1.0, conf)),
                    reason=str(it.get("reason", it.get("reasons", ""))),
                ))
            if verdicts:
                overall = overall or (
                    "any_relevant"
                    if any(v.relevance in ("relevant", "partially_relevant") for v in verdicts)
                    else "none_relevant"
                )
                return VerificationResult(results=verdicts, overall=overall)
        logger.warning("could not parse verifier output; returning all unknown")
        return self._unknown(papers, reason="unparseable verifier output")

    @staticmethod
    def _extract_lists(text: str) -> List[str]:
        out, depth, start = [], 0, -1
        for i, ch in enumerate(text):
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
        return sorted(out, key=len, reverse=True)

    @staticmethod
    def _extract_objects(text: str) -> List[str]:
        out, depth, start = [], 0, -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
        return sorted(out, key=len, reverse=True)

    @staticmethod
    def _unknown(papers: List[dict], reason: str = "verifier unavailable") -> VerificationResult:
        return VerificationResult(
            results=[
                RelevanceVerdict(document_id=str(p.get("document_id", "")),
                                 relevance="unknown", confidence=0.0,
                                 reason=reason)
                for p in papers
            ],
            overall="none_relevant",
        )


if __name__ == "__main__":
    async def _run():
        agent = VerifierAgent()
        res = await agent.verify_papers(
            [{"document_id": "PMC1", "full_text": "Heart failure patients ejection fraction survival"}],
            "relation between EF and survival in heart failure",
            ["long-term survival", "EF correlation"],
        )
        print(res.model_dump())

    asyncio.run(_run())
