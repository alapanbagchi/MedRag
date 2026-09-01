"""Agentic v3 - Stage 14: the global Contradiction / Anomaly Agent.

After all Workers finish, THEIR accepted evidence is passed to this agent.
Its scope is DIFFERENT from the CRITIC:

    CRITIC  - one passage vs one requirement   (verification gate)
    THIS    - evidence A vs evidence B vs ...  (cross-evidence analysis)

It asks: do any of these VERIFIED excerpts contradict each other or contain
an important anomaly? Example (spec section 19):

    Paper A: vitamin D supplementation significantly reduced systolic BP.
    Paper B: vitamin D supplementation had NO significant effect on BP.

Both passed individual verification (each answers its requirement). The
Contradiction Agent identifies:

    CONTRADICTION DETECTED  Claim: vitamin D supplementation affects SBP
    Evidence A: positive effect | Evidence B: no significant effect

The output feeds the dedicated Resolution Agent (Stage 15) - this agent
NEVER picks one paper side.

A deterministic fallback drives the same logic offline: whenever one
evidence requirement has accepted SUPPORTS papers AND accepted CONTRADICTS
papers, a contradiction is flagged - no LLM needed (the LLM's richer reading
is merely layered on top).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.agentic.state import (
    Contradiction,
    ContradictionKind,
    EvidenceStatus,
    SupportDirection,
    V3RunState,
    VerifiedEvidence,
)
from src.prompts.load import load_prompt

_VERIFIED_STATUSES = (EvidenceStatus.ACCEPTED, EvidenceStatus.CONTRADICTORY)

logger = logging.getLogger("src.agents.contradiction")


class ContradictionItem(BaseModel):
    """One LLM-flagged contradiction (evidence ids must match the input)."""
    claim: str = ""
    requirement_id: str = ""
    evidence_a: list[str] = Field(default_factory=list)
    evidence_b: list[str] = Field(default_factory=list)
    kind: str = "direct_conflict"     # direct_conflict | context_dependent | anomaly
    description: str = ""


class ContradictionList(BaseModel):
    contradictions: list[ContradictionItem] = Field(default_factory=list)


CONTRADICTION_SYSTEM_PROMPT = load_prompt('agents', 'contradiction.txt')

_KIND = {k.value: k for k in ContradictionKind}


def _group_by_requirement(evidence: list[VerifiedEvidence]) -> dict[str, list[VerifiedEvidence]]:
    out: dict[str, list[VerifiedEvidence]] = {}
    for e in evidence:
        out.setdefault(e.requirement_id or "-", []).append(e)
    return out


def detect_contradictions_deterministic(
    evidence: list[VerifiedEvidence],
    existing_ids: int = 0,
) -> list[Contradiction]:
    """Offline fallback: requirement-level support/contradict clashes.

    Only VERIFIED items may be considered (requirements 5-6); items in any
    other state (REJECTED / UNDER_REVIEW) are silently ignored here and by
    the caller's filtering.
    """
    verified = [e for e in evidence
                if getattr(e, "status", None) in (_VERIFIED_STATUSES)]
    contradictions: list[Contradiction] = []
    for req_id, items in _group_by_requirement(verified).items():
        supports = [e for e in items if e.support == SupportDirection.SUPPORTS]
        contradicts = [e for e in items if e.support == SupportDirection.CONTRADICTS]
        if supports and contradicts:
            existing_ids += 1
            contradictions.append(Contradiction(
                id=f"C{existing_ids}",
                claim=f"conflicting findings for requirement {req_id}",
                requirement_id=req_id,
                evidence_a=[e.id for e in supports],
                evidence_b=[e.id for e in contradicts],
                kind=ContradictionKind.DIRECT_CONFLICT,
                description=(
                    f"{len(supports)} verified paper(s) support and "
                    f"{len(contradicts)} verified paper(s) contradict the "
                    "same evidence requirement"
                ),
            ))
    return contradictions


def _evidence_block(evidence: list[VerifiedEvidence]) -> str:
    lines = []
    for e in evidence:
        excerpt = " ".join((e.excerpt or "").split())[:400]
        lines.append(
            f"- id={e.id} | requirement={e.requirement_id} | document={e.document_id} "
            f"| support={e.support.value} | conf={e.confidence:.2f}\n"
            f"    excerpt: {excerpt}"
        )
    return "\n".join(lines)


class ContradictionAgent:
    """Looks across all workers' accepted evidence for conflicts/anomalies."""

    def __init__(self, config: Any = None, model: Any = None):
        from pydantic_ai import Agent

        from src.config import AppConfig
        from src.llm import build_model_for

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="contradiction")
        self.agent = Agent(self.model, system_prompt=CONTRADICTION_SYSTEM_PROMPT,
                           output_type=ContradictionList, retries=2,
                           name="contradiction_agent")

    async def detect(self, state: V3RunState) -> list[Contradiction]:
        """Detect contradictions across the run's VERIFIED evidence (LLM on top of
        the deterministic fallback; only CRITIC-passed evidence enters here)."""
        evidence = state.verified_evidence()
        fallback = detect_contradictions_deterministic(evidence)
        if len(evidence) < 2:
            return fallback
        try:
            out = (await self.agent.run(
                "VERIFIED EVIDENCE (all workers):\n" + _evidence_block(evidence)
                + "\n\nDetect genuine contradictions/anomalies. Use the exact "
                  "evidence ids above.")).output
        except Exception as exc:  # noqa: BLE001
            logger.warning("contradiction detection failed (%s); fallback used", exc)
            return fallback

        ids = {e.id for e in evidence}
        contradictions: list[Contradiction] = []
        n = 0
        for item in out.contradictions:
            evidence_a = [i for i in item.evidence_a if i in ids]
            evidence_b = [i for i in item.evidence_b if i in ids]
            if not evidence_a or not evidence_b:
                continue
            n += 1
            contradictions.append(Contradiction(
                id=f"C{n}",
                claim=(item.claim or "").strip()[:300],
                requirement_id=item.requirement_id or "",
                evidence_a=evidence_a, evidence_b=evidence_b,
                kind=_KIND.get((item.kind or "").strip().lower(),
                               ContradictionKind.DIRECT_CONFLICT),
                description=(item.description or "").strip()[:500],
            ))
        return contradictions or fallback
