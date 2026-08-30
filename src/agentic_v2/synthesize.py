"""Agentic v2 — final synthesis from verified evidence only.

This is the SYNTHESIZE action's agent. It receives the original question, the
research objectives (with their final statuses), the VERIFIED evidence, the
unresolved gaps, and the contradictions — and writes an honest answer.

It must NOT invent facts or citations: every citation must copy the document /
chunk ids that appear in the verified evidence, and unresolved objectives must
be stated as unresolved rather than papered over.
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
from typing import Any, List

from pydantic import BaseModel, Field

from src.agentic_v2.state import EvidenceQuality, ResearchState, VerifiedEvidence

logger = logging.getLogger("src.agentic_v2.synthesize")


class Citation(BaseModel):
    objective_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    section: str = ""


class AnswerSection(BaseModel):
    heading: str = ""
    body: str = ""
    citations: List[Citation] = Field(default_factory=list)


class SynthesisReport(BaseModel):
    summary: str = ""
    sections: List[AnswerSection] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    citations: List[Citation] = Field(default_factory=list)


SYNTHESIS_SYSTEM_PROMPT = load_prompt('agentic_v2', 'synthesize.txt')


# Evidence allowed to back the final answer: the verifier's relevant
# (DIRECT) and partially_relevant (INDIRECT) units. Rejected passages are
# BACKGROUND, failed verifications are UNKNOWN — neither may support claims.
ANSWER_QUALITIES = (EvidenceQuality.DIRECT, EvidenceQuality.INDIRECT)


def usable_evidence(state: ResearchState) -> list:
    """Relevant + partially relevant evidence — the only synthesis input."""
    return [e for e in state.evidence if e.quality in ANSWER_QUALITIES]


def _display_excerpt(text: str, max_chars: int = 1600) -> str:
    """Whitespace-collapsed excerpt with an explicit truncation marker.

    The evidence excerpt is already term-anchored upstream (actions._verify),
    so this cap is a safety bound, NOT a head slice: clipped text is marked,
    never silently dropped.
    """
    flat = " ".join((text or "").split())
    if len(flat) <= max_chars:
        return flat
    return flat[:max_chars] + "... [excerpt truncated]"


def _evidence_line(e) -> str:
    excerpt = _display_excerpt(e.excerpt or "")
    return (
        f"[{e.objective_id}] (doc={e.document_id}, chunk={e.chunk_id}, section={e.section}) "
        f"quality={e.quality.value} support={e.support} conf={e.confidence:.2f}\n"
        f'    excerpt: "{excerpt}"'
    )


def _synthesis_prompt(state: ResearchState) -> str:
    lines: List[str] = []
    lines.append(f"ORIGINAL QUESTION: {state.question}")

    lines.append("\nOBJECTIVES:")
    for o in state.objectives:
        lines.append(f"- {o.id} [{o.status.value}] {o.statement}"
                     f"{(' | gap: ' + o.gap) if o.gap else ''}")

    lines.append("\nVERIFIED EVIDENCE (relevant / partially relevant only):")
    usable = usable_evidence(state)
    if usable:
        for e in usable:
            lines.append(_evidence_line(e))
    else:
        lines.append("(none)")

    lines.append("\nUNRESOLVED GAPS:")
    lines.extend(f"- {g}" for g in state.gaps) if state.gaps else lines.append("(none)")

    lines.append("\nCONTRADICTIONS:")
    lines.extend(f"- {c}" for c in state.contradictions) if state.contradictions else lines.append("(none)")

    lines.append("\nWrite the final answer from the verified evidence above.")
    return "\n".join(lines)


class FinalSynthesizer:
    """Produces the final answer from a ResearchState."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model(self.config)
        self.agent = Agent(
            self.model,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            name="synthesizer_v2",
        )

    async def synthesize(self, state: ResearchState) -> SynthesisReport:
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        trace.agent("synthesizer_v2", output_type="SynthesisReport",
                    meta={"evidence": len(usable_evidence(state)),
                          "evidence_total": len(state.evidence),
                          "objectives": len(state.objectives)})
        prompt = _synthesis_prompt(state)
        report = await ask_structured(
            self.agent,
            prompt,
            SynthesisReport,
            label="synthesizer_v2",
            max_tokens=self.config.agent_max_tokens,
        )
        return report
