"""Step 4 - LLM verification (keep/reject) of restored paragraphs, as a tool.

Maps retrieved + paragraph-restored units through the existing batched
VerifierAgent and returns a clean keep/reject/unknown verdict per unit, with a
human-readable rejection reason that later feeds query rewriting.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.agentic.planner import SubQueryPlan
from src.agentic.retriever_tool import RetrievalResult

logger = logging.getLogger("src.agentic.verify")

KEEP = ("relevant", "partially_relevant")


class VerifiedUnit(BaseModel):
    chunk_id: str = ""
    document_id: str = ""
    verdict: str = "reject"          # keep | reject | unknown
    relevance: str = "not_relevant"  # raw verifier relevance level
    confidence: float = 0.0
    reason: str = ""                 # why rejected (or why kept) - feeds rewrite
    paragraph_text: str = ""
    section: str = ""
    unit_kind: str = "paragraph"
    score: float = 0.0


class VerificationOutcome(BaseModel):
    subquery_id: str = ""
    kept: List[VerifiedUnit] = Field(default_factory=list)
    rejected: List[VerifiedUnit] = Field(default_factory=list)
    unknown: List[VerifiedUnit] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def distinct_papers(self) -> int:
        return len({u.document_id for u in self.kept if u.document_id})


class VerifyTool:
    """Wraps the VerifierAgent into a single keep/reject call."""

    def __init__(self, config: Any = None, verifier: Any = None):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self.verifier = verifier

    def _verifier(self):
        if self.verifier is None:
            from src.agents.verifier_new import VerifierAgent
            self.verifier = VerifierAgent(config=self.config)
        return self.verifier

    async def verify(
        self,
        sub: SubQueryPlan,
        results: List[RetrievalResult],
        base_query: Optional[str] = None,
    ) -> VerificationOutcome:
        """Classify restored paragraphs against the subquery intent."""
        if not results:
            return VerificationOutcome(subquery_id=sub.id)

        query = base_query or sub.query or sub.target
        papers = [
            {"document_id": r.chunk_id or r.document_id, "full_text": r.paragraph_text or ""}
            for r in results
        ]
        try:
            verdicts = await self._verifier().verify_papers(
                papers, query, list(sub.evidence_required or [])
            )
        except Exception as exc:
            logger.warning("verification failed for %s: %s", sub.id, exc)
            # total failure -> everything unknown (retryable), not rejected
            return VerificationOutcome(
                subquery_id=sub.id,
                unknown=[
                    self._to_unit(r, "unknown", "not_relevant", 0.0, str(exc)[:120])
                    for r in results
                ],
            )

        by_id = {v.document_id: v for v in verdicts.results}
        outcome = VerificationOutcome(subquery_id=sub.id)
        for r in results:
            cid = r.chunk_id or r.document_id
            v = by_id.get(cid)
            if v is None:
                outcome.unknown.append(self._to_unit(r, "unknown", "not_relevant", 0.0, "verifier missing verdict"))
                continue
            unit = self._to_unit(r, self._map_verdict(v.relevance), v.relevance,
                                 v.confidence or 0.0, v.reason or "")
            if unit.verdict == "keep":
                outcome.kept.append(unit)
            elif unit.verdict == "reject":
                outcome.rejected.append(unit)
                if unit.reason:
                    outcome.rejection_reasons.append(unit.reason)
            else:
                outcome.unknown.append(unit)
        return outcome

    @staticmethod
    def _map_verdict(relevance: str) -> str:
        if relevance in KEEP:
            return "keep"
        if relevance == "unknown":
            return "unknown"
        return "reject"

    @staticmethod
    def _to_unit(r: RetrievalResult, verdict: str, relevance: str,
                 confidence: float, reason: str) -> VerifiedUnit:
        return VerifiedUnit(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            verdict=verdict,
            relevance=relevance,
            confidence=confidence,
            reason=reason,
            paragraph_text=r.paragraph_text,
            section=r.section,
            unit_kind=r.unit_kind,
            score=r.rrf_score,
        )
