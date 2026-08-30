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
from src.prompts.load import load_prompt

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.agentic_v3.state import (
    Contradiction,
    ContradictionKind,
    EvidenceStatus,
    SupportDirection,
    VerifiedEvidence,
    V3RunState,
)

_VERIFIED_STATUSES = (EvidenceStatus.ACCEPTED, EvidenceStatus.CONTRADICTORY)

logger = logging.getLogger("src.agentic_v3.contradiction")


class ContradictionItem(BaseModel):
    """One LLM-flagged contradiction (evidence ids must match the input)."""
    claim: str = ""
    requirement_id: str = ""
    evidence_a: List[str] = Field(default_factory=list)
    evidence_b: List[str] = Field(default_factory=list)
    kind: str = "direct_conflict"     # direct_conflict | context_dependent | anomaly
    description: str = ""


class ContradictionList(BaseModel):
    contradictions: List[ContradictionItem] = Field(default_factory=list)


CONTRADICTION_SYSTEM_PROMPT = load_prompt('agentic_v3', 'contradiction.txt')

_KIND = {k.value: k for k in ContradictionKind}


def _group_by_requirement(evidence: List[VerifiedEvidence]) -> Dict[str, List[VerifiedEvidence]]:
    out: Dict[str, List[VerifiedEvidence]] = {}
    for e in evidence:
        out.setdefault(e.requirement_id or "-", []).append(e)
    return out


def detect_contradictions_deterministic(
    evidence: List[VerifiedEvidence],
    existing_ids: int = 0,
) -> List[Contradiction]:
    """Offline fallback: requirement-level support/contradict clashes.

    Only VERIFIED items may be considered (requirements 5-6); items in any
    other state (REJECTED / UNDER_REVIEW) are silently ignored here and by
    the caller's filtering.
    """
    verified = [e for e in evidence
                if getattr(e, "status", None) in (_VERIFIED_STATUSES)]
    contradictions: List[Contradiction] = []
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


def _evidence_block(evidence: List[VerifiedEvidence]) -> str:
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
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="contradiction")
        self.agent = Agent(
            self.model,
            system_prompt=CONTRADICTION_SYSTEM_PROMPT,
            name="contradiction_agent",
        )

    async def detect(self, state: V3RunState) -> List[Contradiction]:
        """Detect contradictions across the run's VERIFIED evidence.

        Only evidence that passed the CRITIC (ACCEPTED + CONTRADICTORY
        states) enters this stage (requirement 5): REJECTED / UNDER_REVIEW
        material can never be reinterpreted here.
        """
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        evidence = state.verified_evidence()
        fallback = detect_contradictions_deterministic(evidence)

        if len(evidence) < 2:
            return fallback
        trace.agent("contradiction_agent", output_type="ContradictionList",
                    meta={"evidence": len(evidence)})
        prompt = (
            "VERIFIED EVIDENCE (all workers):\n"
            + _evidence_block(evidence)
            + "\n\nDetect genuine contradictions/anomalies. Use the exact "
            "evidence ids above."
        )
        try:
            out = await ask_structured(
                self.agent,
                prompt,
                ContradictionList,
                label="contradiction_agent",
                max_tokens=min(900, getattr(self.config, "agent_max_tokens", 2048)),
            )
        except Exception as exc:
            logger.warning("contradiction detection failed (%s); fallback used", exc)
            return fallback

        ids = {e.id for e in evidence}
        contradictions: List[Contradiction] = []
        n = 0
        for item in out.contradictions:
            evidence_a = [i for i in item.evidence_a if i in ids]
            evidence_b = [i for i in item.evidence_b if i in ids]
            if not evidence_a or not evidence_b:
                continue
            n += 1
            try:
                kind = _KIND.get((item.kind or "").strip().lower(),
                                 ContradictionKind.DIRECT_CONFLICT)
            except Exception:
                kind = ContradictionKind.DIRECT_CONFLICT
            contradictions.append(Contradiction(
                id=f"C{n}",
                claim=(item.claim or "").strip()[:300],
                requirement_id=item.requirement_id or "",
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                kind=kind,
                description=(item.description or "").strip()[:500],
            ))
        return contradictions or fallback
