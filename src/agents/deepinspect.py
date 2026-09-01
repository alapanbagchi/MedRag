"""Agentic v3 - Stage 11: deep paper inspection.

Sometimes a paper looks highly promising but the normal search excerpt does not
contain enough information (spec sections 11, 14, 15):

    * the search index only exposes certain passages,
    * relevant information may be in a TABLE,
    * the evidence may be several paragraphs away,
    * the finding may be buried in Results or Discussion,
    * the paper may contain useful material that never matched the
      search terms.

Instead of rejecting the paper, the Worker invokes deep inspection - "the
paper might have something in it":

    Potentially useful paper
        -> obtain the ACTUAL paper text (the corpus stores full PMC text)
        -> an inspection LLM finds candidate evidence relevant to the
           requirement (this is the vision-model role: locate the relevant
           Results table / passage that ordinary retrieval missed)
        -> TEXT/GREP verification: the claimed quote/finding must actually
           exist in the source - the inspection model is NOT trusted as the
           final authority (spec section 14)
        -> the extracted passage is expanded to its surrounding context and
           passed through the CRITIC gate (ACCEPT / REJECT)

In this corpus the "paper" is the full PMC text (no PDFs are stored), so
"PDF pages -> images -> vision LLM" is realized as
"full text -> section-aware LLM finder -> deterministic text verification".
The epistemic rule is preserved unchanged: nothing the finder claims is
accepted until it is verified against the actual source text, and no passage
counts as evidence until the CRITIC accepts it.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.agents.critic import CriticAgent, evidence_excerpt
from src.agentic.state import (
    EvidenceRequirement,
    EvidenceSource,
    ResearchTask,
    SupportDirection,
    VerifiedEvidence,
)
from src.lib.utils import normalize_for_match, plural, quote_grounded
from src.prompts.load import load_prompt

logger = logging.getLogger("src.agents.deepinspect")

_MAX_DOC_TOKENS = 12000      # bounded paper body fed to the inspection model
_MAX_FINDINGS = 4


class Finding(BaseModel):
    """One candidate location the inspection model surfaced."""
    section: str = ""
    quote: str = ""                    # claimed verbatim text (verified next)
    claim: str = ""                    # what the model believes the text says
    support: str = "supports"          # supports | contradicts (as claimed)


INSPECTOR_SYSTEM_PROMPT = load_prompt('agents', 'deep_inspector.txt')


class InspectorOutput(BaseModel):
    findings: list[Finding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full-text provider (lazily binds the shared corpus)
# ---------------------------------------------------------------------------

class _FullTextProvider:
    """Reads the actual paper text + section map from the shared corpus."""

    def __init__(self, config: Any = None):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self._service = None
        self._corpus = None

    def _corpus_obj(self) -> Any:
        if self._corpus is not None:
            return self._corpus
        try:
            from src.retrieval.retriever import get_retrieval_service
            service = get_retrieval_service(self.config)
            self._corpus = service._components()["corpus"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("corpus unavailable for deep inspection: %s", exc)
            self._corpus = None
        return self._corpus

    def split_sections_rows(self, document_id: str) -> list[tuple[str, str]]:
        """[(section, section_text)] ordered by document position, deduped."""

        corpus = self._corpus_obj()
        if corpus is None:
            return []
        df = getattr(corpus, "_df", None)
        if df is None or "document_id" not in df.columns:
            return []
        sub = df[df["document_id"] == document_id]
        if sub.empty:
            return []
        order_col = "document_position" if "document_position" in sub.columns else None
        if order_col is not None:
            sub = sub.sort_values(order_col)
        sections: dict[str, list[str]] = {}
        for _, row in sub.iterrows():
            sec = str(row.get("section") or "(untitled)")
            text = str(row.get("text") or "").strip()
            if text:
                sections.setdefault(sec, []).append(text)
        return [(sec, "\n".join(parts)) for sec, parts in sections.items()]

    def document_text(self, document_id: str, max_tokens: int = _MAX_DOC_TOKENS) -> str:
        """Concatenated full paper text (bounded); returns '' when unavailable.

        Runs synchronously (corpus DataFrame reads are not async); callers
        may wrap it in asyncio.to_thread if they need to keep the loop free.
        """
        try:
            rows = self.split_sections_rows(document_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cannot read document %s: %s", document_id, exc)
            return ""
        parts = []
        used = 0
        for sec, text in rows:
            header = f"[{sec}]"
            words_avail = max(0, max_tokens - used)
            if words_avail <= 0:
                break
            sec_tokens = text.split()
            if len(sec_tokens) > words_avail:
                text = " ".join(sec_tokens[:words_avail]) + " [...truncated]"
            parts.append(header)
            parts.append(text)
            used += len(text.split())
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic text verification ("GREP proves the claimed text exists")
# ---------------------------------------------------------------------------

def verify_quote(document_text: str, quote: str) -> tuple[int, int] | None:
    """Return (start, end) in the normalized document text, or None.

    Grounds the claim via src.lib.utils.quote_grounded (normalized ->
    squashed -> fuzzy ladder); the returned span lives in the same
    normalize_for_match form expand_passage consumes.
    """
    if not document_text or not quote:
        return None
    grounded, _ = quote_grounded(quote, document_text)
    if not grounded:
        return None
    doc = normalize_for_match(document_text)
    q = normalize_for_match(quote)
    start = doc.find(q)
    return (start, start + len(q)) if start >= 0 else None


def expand_passage(document_text: str, start: int, end: int,
                   lead: int = 400, tail: int = 700) -> str:
    """Expand the verified span to its surrounding paragraph context."""
    doc = normalize_for_match(document_text)
    lo = max(0, start - lead)
    hi = min(len(doc), end + tail)
    # snap to sentence boundaries
    lo = doc.rfind(". ", 0, lo) + 1 if doc.rfind(". ", 0, lo) >= 0 else lo
    nxt = doc.find(". ", hi)
    if nxt >= 0 and nxt - hi < 400:
        hi = nxt + 2
    return doc[lo:hi].strip()


# ---------------------------------------------------------------------------
# The deep inspector
# ---------------------------------------------------------------------------

class DeepInspector:
    """Deep paper inspection for one (promising) paper vs one requirement."""

    def __init__(self, config: Any = None, model: Any = None,
                 critic: Any = None, provider: Any = None):
        from pydantic_ai import Agent

        from src.config import AppConfig
        from src.llm import build_model_for

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="deep_inspector")
        self.agent = Agent(self.model, system_prompt=INSPECTOR_SYSTEM_PROMPT,
                           output_type=InspectorOutput, retries=2,
                           name="paper_inspector")
        self.critic = critic or CriticAgent(config=self.config)
        self.provider = provider or _FullTextProvider(self.config)

    async def inspect(
        self,
        task: ResearchTask,
        requirement: EvidenceRequirement,
        document_id: str,
        *,
        context_hint: str = "",
        run_id: str = "",
        attempt_id: str = "",
    ) -> list[VerifiedEvidence]:
        """Deep-inspect ONE paper; returns critic-ACCEPTED evidence (may be []).

        Steps (spec section 14):
          PDF/full text obtained  -> finder locates candidate findings
          -> TEXT/GREP verification (the claims must exist) -> CRITIC -> accept.

        Every finding is stamped with the caller's scope (run/task/
        requirement/attempt) so it stays isolated like any other evidence.
        """
        from src.lib.trace import get_trace

        trace = get_trace()
        doc_text = self._full_text(document_id)
        if not doc_text:
            return []

        findings: list[Finding] = []
        try:
            out = await self.agent.run(self._prompt(task, requirement, doc_text, context_hint))
            findings = [f for f in out.output.findings if (f.quote or "").strip()][:_MAX_FINDINGS]
        except Exception as exc:  # noqa: BLE001
            logger.warning("deep inspection find failed for %s (%s)", document_id, exc)
            return []

        accepted: list[VerifiedEvidence] = []
        for f in findings:
            match = verify_quote(doc_text, f.quote)
            if match is None:
                # The claimed text does NOT exist -> the finder is not trusted;
                # the finding is discarded (this is exactly the spec's rule).
                trace.log("deep_inspection_unverified", document=document_id,
                          claim=(f.claim or "")[:120])
                continue
            start, end = match
            passage = expand_passage(doc_text, start, end)
            if not passage:
                continue
            # The CRITIC is the final authority even here; the candidate
            # carries the full scope so the verdict is scoped like any other.
            from src.agentic.state import EvidenceStatus, RetrievedPaper
            paper = RetrievedPaper(
                evidence_id=f"{task.id}.{requirement.id}.{attempt_id or '?'}."
                           f"DI{len(accepted) + 1}",
                task_id=task.id,
                requirement_id=requirement.id,
                attempt_id=attempt_id,
                status=EvidenceStatus.RETRIEVED,
                document_id=document_id,
                chunk_id="",
                section=f.section or "",
                unit_kind="paragraph",
                score=0.0,
                retrieval_method="deep_inspection",
                rank=0,
                text=passage,
                source_query="deep_inspection",
            )
            try:
                verdict = await self.critic.judge(
                    task, requirement, paper, run_id=run_id, attempt_id=attempt_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("critic failed during deep inspection %s: %s",
                               document_id, exc)
                continue
            if not verdict.accepted:
                continue
            try:
                support = SupportDirection(verdict.support.value)
            except Exception:  # noqa: BLE001
                support = SupportDirection.SUPPORTS
            item = VerifiedEvidence(
                id=paper.evidence_id,
                run_id=run_id,
                task_id=task.id,
                requirement_id=requirement.id,
                attempt_id=attempt_id,
                document_id=document_id,
                chunk_id="",
                section=verdict.section or f.section or "",
                excerpt=evidence_excerpt(passage),
                claim=(f.claim or "").strip(),
                support=support,
                confidence=verdict.confidence,
                source=EvidenceSource.DEEP_INSPECTION,
                status=(EvidenceStatus.CONTRADICTORY
                        if support == SupportDirection.CONTRADICTS
                        else EvidenceStatus.ACCEPTED),
                retrieval_method="deep_inspection",
                rank=0,
                critic_verdict=verdict.model_dump(mode="json"),
                note=verdict.note,
                search_query="deep_inspection",
            )
            accepted.append(item)
        trace.bullet(
            f"Deep inspection of {document_id}: "
            f"{plural(len(findings), 'candidate passage')} located, "
            f"{plural(len(accepted), 'verified passage')} passed verbatim "
            f"verification and the critic.",
            agent="deep_inspector")
        return accepted

    def _full_text(self, document_id: str) -> str:
        try:
            return self.provider.document_text(document_id, max_tokens=_MAX_DOC_TOKENS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("full-text load failed for %s: %s", document_id, exc)
            return ""

    @staticmethod
    def _prompt(task: ResearchTask, requirement: EvidenceRequirement,
                doc_text: str, context_hint: str = "") -> str:
        hint = f"\nWHY THIS PAPER WAS FLAGGED: {context_hint[:300]}" if context_hint else ""
        return (
            f"WORKER TASK {task.id}: {task.title}\n"
            f"EVIDENCE REQUIREMENT {requirement.id}: {requirement.text}{hint}\n"
            f"\nFULL PAPER TEXT (document):\n{doc_text}\n"
            "\nLocate candidate passages in this text that answer the "
            "evidence requirement. Return the findings JSON."
        )
