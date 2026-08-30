"""Agentic v3 - Stage 8: the verification gate / CRITIC.

Every candidate evidence passage passes through this gate. The CRITIC's
primary question (spec section 10):

    Does this passage actually answer the Worker's task / evidence
    requirement?

It assesses:
  * relevance - is the passage about the required topic at all?
  * whether the passage actually addresses the REQUESTED RELATIONSHIP
    (e.g. "vitamin D deficiency is common among patients with
    hypertension" does NOT establish whether supplementation lowers
    blood pressure),
  * whether the claim is SUPPORTED by the passage (and in which direction),
  * whether enough context is available to judge,
  * whether the passage is answering the assigned evidence requirement
    rather than merely mentioning the topic.

Example (from the spec):
    TASK: determine whether vitamin D supplementation lowers blood pressure
    PASSAGE: "Vitamin D deficiency is common among patients with hypertension."
    CRITIC -> RELEVANT: partially | ANSWERS TASK: NO   => REJECT

Acceptance rule (Stage 9): only passages judged ANSWERS_TASK=YES become
accepted evidence candidates and count toward the requirement's N threshold.
A YES judgement passes even when the passage CONTRADICTS the hypothesis -
contradictory evidence is real evidence and belongs to the Contradiction
Agent, not the rejection pile (spec section 19).

One passage per LLM call (full expanded-context text), matching the intent
verifier's design so no truncated snippet is ever judged.
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
from typing import Any, List

from pydantic import BaseModel

from src.agentic_v3.state import (
    AnswersTask,
    CriticRelevance,
    CriticVerdict,
    EvidenceRequirement,
    ResearchTask,
    RetrievedPaper,
    SupportDirection,
)

logger = logging.getLogger("src.agentic_v3.critic")


class CriticOutput(BaseModel):
    """Raw LLM verdict for ONE passage against ONE requirement."""
    relevance: str = "not_relevant"   # relevant | partially_relevant | not_relevant
    answers_task: str = "no"          # yes | no | partial
    support: str = "neutral"          # supports | contradicts | neutral
    confidence: float = 0.0
    note: str = ""


CRITIC_SYSTEM_PROMPT = load_prompt('agentic_v3', 'critic.txt')

_RELEVANCE = {r.value: r for r in CriticRelevance}
_ANSWERS = {a.value: a for a in AnswersTask}
_SUPPORT = {s.value: s for s in SupportDirection}

_MAX_PASSAGES_PER_ROUND = 6

# Process-wide "batch API unavailable" cache: after the FIRST failure (e.g.
# 402 billing-not-enabled) we stop trying to create batch jobs and fall back
# to sequential calls silently for the rest of the process.
_BATCH_UNAVAILABLE: set = set()


class CriticAgent:
    """The verification gate. One full-passage LLM call per candidate."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="verifier")
        self.agent = Agent(
            self.model,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            name="critic",
        )

    # ------------------------------------------------------------------
    # Shared helpers (used by both the sequential and the batch path)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_scope(task: ResearchTask, requirement: EvidenceRequirement,
                     paper: RetrievedPaper) -> None:
        """Hard isolation guard (requirement 3): scope must match the caller."""
        if (paper.task_id or "").strip() and paper.task_id != task.id:
            raise ValueError(
                f"critic scope mismatch: paper.task_id={paper.task_id!r} != "
                f"task.id={task.id!r}")
        if (paper.requirement_id or "").strip() and paper.requirement_id != requirement.id:
            raise ValueError(
                f"critic scope mismatch: paper.requirement_id="
                f"{paper.requirement_id!r} != requirement.id={requirement.id!r}")

    @staticmethod
    def _scope(task: ResearchTask, requirement: EvidenceRequirement,
               paper: RetrievedPaper, *, run_id: str, attempt_id: str) -> dict:
        return {
            "run_id": run_id or "",
            "task_id": task.id,
            "requirement_id": requirement.id,
            "attempt_id": attempt_id or paper.attempt_id or "",
            "evidence_id": paper.evidence_id or "",
            "document_id": paper.document_id,
            "chunk_id": paper.chunk_id,
            "section": paper.section,
        }

    @staticmethod
    def _prompt(task: ResearchTask, requirement: EvidenceRequirement,
                paper: RetrievedPaper) -> str:
        text = (paper.text or "").strip()
        return (
            f"WORKER TASK {task.id}: {task.title}\n"
            f"TASK OBJECTIVE: {task.objective}\n"
            f"EVIDENCE REQUIREMENT {requirement.id}: {requirement.text}\n"
            f"\nCANDIDATE PASSAGE (evidence={paper.evidence_id or '?'}, "
            f"document={paper.document_id}, chunk={paper.chunk_id}, "
            f"section={paper.section}, type={paper.unit_kind}):\n{text}\n"
            "\nJudge whether this passage ACTUALLY ANSWERS the evidence "
            "requirement. Return the JSON verdict."
        )

    @staticmethod
    def _apply_output(scope: dict, out: CriticOutput) -> CriticVerdict:
        """Map a raw LLM verdict onto a scoped CriticVerdict.

        The scope was fixed BEFORE the call (above) and is never overwritten;
        only the JUDGEMENT fields are read from the LLM output.
        """
        relevance = _RELEVANCE.get((out.relevance or "").strip().lower(),
                                   CriticRelevance.NOT_RELEVANT)
        answers = _ANSWERS.get((out.answers_task or "").strip().lower(),
                               AnswersTask.NO)
        support = _SUPPORT.get((out.support or "").strip().lower(),
                               SupportDirection.NEUTRAL)
        try:
            confidence = max(0.0, min(1.0, float(out.confidence or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        verdict = CriticVerdict(**scope)
        verdict.relevance = relevance
        verdict.answers_task = answers
        verdict.support = support
        verdict.confidence = confidence
        verdict.note = (out.note or "").strip()
        return verdict

    async def judge(
        self,
        task: ResearchTask,
        requirement: EvidenceRequirement,
        paper: RetrievedPaper,
        *,
        run_id: str = "",
        attempt_id: str = "",
    ) -> CriticVerdict:
        """Judge ONE candidate passage (expanded context) vs one requirement.

        STATE ISOLATION (requirement 3): the verdict's scope is derived from
        the candidate paper itself ({task_id, requirement_id, attempt_id,
        evidence_id} are stamped on the paper when the worker creates it),
        and the caller's task/requirement MUST match that scope - otherwise
        judgement raises and a misplaced critic call is impossible.

        The verdict is constructed with its scope BEFORE any LLM output is
        read, so no result can ever be attached to a different requirement.
        """
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        self._check_scope(task, requirement, paper)
        scope = self._scope(task, requirement, paper, run_id=run_id,
                            attempt_id=attempt_id)
        trace.agent("critic", output_type="CriticVerdict",
                    meta={"task": task.id, "requirement": requirement.id,
                          "attempt": scope["attempt_id"],
                          "evidence": scope["evidence_id"],
                          "document": paper.document_id})
        try:
            out = await ask_structured(
                self.agent,
                self._prompt(task, requirement, paper),
                CriticOutput,
                label=(f"critic:{task.id}:{requirement.id}:"
                       f"{attempt_id or paper.attempt_id or ''}:"
                       f"{paper.evidence_id or paper.document_id or paper.chunk_id}"),
                max_tokens=max(1024, getattr(self.config, "agent_max_tokens", 2048)),
                max_attempts=2,
            )
        except Exception as exc:
            # Failure != irrelevance: never record a failed call as a reject.
            logger.warning("critic call failed for %s (%s)",
                           paper.document_id or paper.chunk_id, exc)
            raise
        verdict = self._apply_output(scope, out)
        trace.log("critic_verdict", task=task.id, requirement=requirement.id,
                  document_id=paper.document_id, relevance=verdict.relevance.value,
                  answers_task=verdict.answers_task.value, support=verdict.support.value,
                  accepted=verdict.accepted)
        return verdict

    # ------------------------------------------------------------------
    # Batching: one Mistral batch job per worker round instead of one LLM
    # call per passage (falls back to sequential when unavailable).
    # ------------------------------------------------------------------

    def _batching_enabled(self) -> bool:
        from src.llm.client import role_provider_name

        mode = (getattr(self.config, "verifier_batch_enabled", "auto")
                or "auto").strip().lower()
        if mode in ("0", "off", "false", "no"):
            return False
        if mode in ("1", "on", "true", "yes"):
            return True
        return role_provider_name(self.config, "verifier") == "mistral"

    def _build_batch_requests(self, task: ResearchTask,
                              requirement: EvidenceRequirement,
                              papers: List[RetrievedPaper],
                              model: str) -> tuple[dict, List[tuple]]:
        """Build Mistral batch requests; returns (requests, [(paper, scope)]).

        The scope per paper is fixed here (before any output is read) and
        each request carries a unique custom_id derived from the evidence id.
        """
        import json

        requests: List[dict] = []
        pairs: List[tuple] = []
        for paper in papers:
            scope = self._scope(task, requirement, paper, run_id="",
                                attempt_id="")
            custom_id = scope["evidence_id"] or f"{scope['document_id']}:{scope['chunk_id']}"
            prompt = self._prompt(task, requirement, paper)
            messages = [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max(1024, getattr(self.config, "agent_max_tokens", 2048)),
                "temperature": 0.0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "critic_output",
                        "strict": True,
                        "schema": CriticOutput.model_json_schema(),
                    },
                },
            }
            requests.append({"custom_id": custom_id, "body": body})
            pairs.append((paper, scope))
        return requests, pairs

    def _parse_batch_text(self, raw: str) -> CriticOutput:
        """Parse one batch response text into CriticOutput (lenient)."""
        from src.lib import decode_structured

        try:
            return decode_structured(raw, CriticOutput)
        except Exception as exc:
            logger.warning("critic batch output unparseable (%s): %s",
                           exc, (raw or "")[:120])
            return CriticOutput()

    async def judge_papers(
        self,
        task: ResearchTask,
        requirement: EvidenceRequirement,
        papers: List[RetrievedPaper],
        *,
        run_id: str = "",
        attempt_id: str = "",
    ) -> List[tuple]:
        """Judge ALL candidate passages for one query and return
        ``[(paper, CriticVerdict | None)]`` in the same order.

        When the verifier provider is Mistral and batching is enabled, all
        judgements go into ONE Mistral batch job (one network round-trip per
        worker round instead of one LLM call per passage); otherwise it runs
        the sequential ``judge`` per passage. A verdict is ``None`` on a
        critic failure (treated as 'cannot verify', never as a rejection).
        """
        papers = list(papers)[:_MAX_PASSAGES_PER_ROUND]
        if not papers:
            return []

        pairs: List[tuple] = []
        if self._batching_enabled() and not _BATCH_UNAVAILABLE:
            try:
                pairs = await self._batch_judge_papers(
                    task, requirement, papers, run_id=run_id,
                    attempt_id=attempt_id)
                if pairs:
                    return pairs
            except Exception as exc:  # noqa: BLE001 - any batch failure falls back
                from src.trace import get_trace

                _BATCH_UNAVAILABLE.add(str(exc)[:120])
                logger.warning("mistral batch verification failed (%s); "
                               "falling back to sequential critic", exc)
                get_trace().bullet(
                    f"Mistral batch unavailable ({str(exc)[:120]}); "
                    f"verifying {len(papers)} passage(s) sequentially.",
                    agent="critic")

        # sequential fallback path
        for paper in papers:
            from src.trace import get_trace

            trace = get_trace()
            trace.bullet(
                f"Verifying {paper.document_id} ({paper.section}) against "
                f"{requirement.id}…",
                agent="critic")
            try:
                verdict = await self.judge(task, requirement, paper,
                                           run_id=run_id, attempt_id=attempt_id)
            except Exception as exc:
                logger.warning("critic skipped %s: %s",
                               paper.document_id or paper.chunk_id, exc)
                pairs.append((paper, None))
                continue
            pairs.append((paper, verdict))
        return pairs

    async def _batch_judge_papers(
        self,
        task: ResearchTask,
        requirement: EvidenceRequirement,
        papers: List[RetrievedPaper],
        *,
        run_id: str,
        attempt_id: str,
    ) -> List[tuple]:
        """One Mistral batch job for all papers; returns [(paper, verdict|None)]."""
        from src.llm.client import resolve_role_provider
        from src.llm.mistral_batch import MistralBatch
        from src.trace import get_trace

        trace = get_trace()
        base_url, api_key, model, _profile = resolve_role_provider(
            self.config, "verifier")
        requests, pairs = self._build_batch_requests(task, requirement,
                                                     papers, model)
        custom_ids = [p[1]["evidence_id"] or f"{p[0].document_id}:{p[0].chunk_id}"
                      for p in pairs]
        client = MistralBatch(self.config, base_url=base_url, api_key=api_key,
                              model=model)
        trace.bullet(
            f"Submitting {len(requests)} verification call(s) for "
            f"{requirement.id} as ONE Mistral batch job…",
            agent="critic")
        job = await client.run(
            requests,
            model=model,
            poll_seconds=float(getattr(self.config, "verifier_batch_poll_seconds", 5)),
            timeout_seconds=float(getattr(self.config, "verifier_batch_timeout_seconds", 900)),
            metadata={"task": task.id, "requirement": requirement.id,
                      "attempt": attempt_id or ""},
        )
        job_id = job.get("id", "")
        outputs = MistralBatch.extract_outputs(job, custom_ids)
        total = job.get("total_requests") or len(requests)
        succeeded = job.get("succeeded_requests") or sum(1 for v in outputs.values() if v)
        trace.bullet(
            f"Mistral batch {job_id} finished: {succeeded}/{total} "
            f"verdict(s) returned.",
            agent="critic")
        trace.log("mistral_batch", job_id=job_id, task=task.id,
                  requirement=requirement.id, total=total,
                  succeeded=succeeded, failed=job.get("failed_requests"))

        results: List[tuple] = []
        for paper, scope in pairs:
            cid = scope["evidence_id"] or f"{paper.document_id}:{paper.chunk_id}"
            raw = outputs.get(cid)
            out = self._parse_batch_text(raw) if raw else CriticOutput()
            verdict = self._apply_output(scope, out)
            trace.log("critic_verdict", task=task.id,
                      requirement=requirement.id, document_id=paper.document_id,
                      relevance=verdict.relevance.value,
                      answers_task=verdict.answers_task.value,
                      support=verdict.support.value, accepted=verdict.accepted,
                      batch_job=job_id)
            results.append((paper, verdict if raw else None))
        return results

    async def judge_batch(
        self,
        task: ResearchTask,
        requirement: EvidenceRequirement,
        papers: List[RetrievedPaper],
    ) -> List[CriticVerdict]:
        """Judge up to max-papers-per-round passages (batched or sequential).

        Returns the verdicts that succeeded (critic-failed verdicts are
        dropped, matching the worker's 'cannot verify' semantics).
        """
        pairs = await self.judge_papers(task, requirement, papers)
        return [v for _, v in pairs if v is not None]


# ---------------------------------------------------------------------------
# Deterministic helpers (unit-testable without the LLM)
# ---------------------------------------------------------------------------

def evidence_excerpt(text: str, max_chars: int = 1600) -> str:
    """Whitespace-collapsed bounded excerpt with an explicit marker."""
    flat = " ".join((text or "").split())
    if len(flat) <= max_chars:
        return flat
    return flat[:max_chars] + "... [excerpt truncated]"


def is_promising_for_deep_inspection(verdict: CriticVerdict) -> bool:
    """A paper "that might have something in it" (spec section 11/14).

    Triggered when the passage is at least partially on-topic but did NOT
    actually answer the requirement - typically because the search excerpt
    lacks the outcome (it may sit in a Results table or Discussion section
    that ordinary retrieval did not surface).
    """
    if verdict.accepted:
        return False
    return verdict.relevance in (CriticRelevance.PARTIALLY_RELEVANT,
                                  CriticRelevance.RELEVANT) and         verdict.answers_task in (AnswersTask.NO, AnswersTask.PARTIAL)
