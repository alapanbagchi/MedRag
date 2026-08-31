"""Agentic v3 - final answer synthesis from verified evidence only.

The final evidence set (Stage 23) feeds this stage: the question, the tasks
with their requirements/coverage, the VERIFIED evidence, the gaps, the
contradictions and their resolutions. The synthesizer writes an honest
answer governed by the design principles (spec section 32):

    * the goal is the best-supported answer justified by the retrieved and
      verified evidence;
    * evidence gaps are reported, not papered over;
    * contradictions that could not be resolved are EXPLICITLY reported as
      unresolved - the system never manufactures certainty.

Every substantive claim must be backed by evidence ids that exist in the
verified evidence; after the LLM call a deterministic repair pass removes
any citation id that does not exist (the model is never the citation
authority).
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
from typing import Any, List

from pydantic import BaseModel, Field

from src.agents.state import ResolutionStatus, V3RunState

logger = logging.getLogger("src.agents.synthesize")


class Citation(BaseModel):
    requirement_id: str = ""
    document_id: str = ""
    support: str = ""


class AnswerSection(BaseModel):
    heading: str = ""
    body: str = ""
    citations: List[Citation] = Field(default_factory=list)


class SynthesisReport(BaseModel):
    summary: str = ""
    sections: List[AnswerSection] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    unresolved_gaps: List[str] = Field(default_factory=list)
    unresolved_contradictions: List[str] = Field(default_factory=list)
    resolved_contradictions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    citations: List[Citation] = Field(default_factory=list)


SYNTHESIS_SYSTEM_PROMPT = load_prompt('agents', 'synthesize.txt')


def _evidence_line(evidence: Any) -> str:
    excerpt = " ".join((evidence.excerpt or "").split())[:500]
    return (
        f"- id={evidence.id} | requirement={evidence.requirement_id} | "
        f"doc={evidence.document_id} | source={evidence.source.value} | "
        f"support={evidence.support.value} | conf={evidence.confidence:.2f}\n"
        f"    excerpt: {excerpt}"
    )


def _task_block(state: V3RunState) -> str:
    lines = []
    for task in state.tasks:
        lines.append(f"- {task.id} [{task.status.value}] {task.title}: {task.objective}")
        for r in task.evidence_requirements:
            lines.append(
                f"    req {r.id} [{r.status.value}] coverage={r.coverage()}/{r.target_n}"
                f"{(' | gap: ' + r.gap) if r.gap else ''}"
            )
    return "\n".join(lines)


def _synthesis_prompt(state: V3RunState) -> str:
    # EVIDENCE GATE (requirement 7): only CRITIC-verified material (ACCEPTED
    # + CONTRADICTORY) is ever presented to the synthesizer - arbitrary
    # retrieval results and REJECTED items are structurally excluded.
    evidence = state.verified_evidence()
    lines: List[str] = []
    lines.append(f"USER QUESTION: {state.question}")
    lines.append("\nTASKS:")
    lines.append(_task_block(state))
    lines.append("\nVERIFIED EVIDENCE (the ONLY citable material):")
    if evidence:
        lines.extend(_evidence_line(e) for e in evidence)
    else:
        lines.append("(none)")
    lines.append("\nUNRESOLVED GAPS:")
    gaps = [r.gap for t in state.tasks for r in t.evidence_requirements if r.gap]
    lines.extend(f"- {g}" for g in gaps) if gaps else lines.append("(none - all requirements satisfied)")
    lines.append("\nCONTRADICTIONS:")
    if state.contradictions:
        for c in state.contradictions:
            res = c.resolution
            if res is None:
                res_line = "not yet investigated"
            elif res.status == ResolutionStatus.RESOLVED:
                res_line = f"RESOLVED: {res.explanation}"
            elif res.status == ResolutionStatus.PARTIALLY_RESOLVED:
                res_line = f"PARTIALLY RESOLVED: {res.explanation}"
            else:
                res_line = f"UNRESOLVED: {res.explanation}"
            lines.append(
                f"- {c.id} [{c.kind.value}] {c.claim}\n"
                f"    side A ids: {', '.join(c.evidence_a)}\n"
                f"    side B ids: {', '.join(c.evidence_b)}\n"
                f"    resolution: {res_line}"
            )
    else:
        lines.append("(none)")
    lines.append("\nWrite the final answer from the verified evidence above.")
    return "\n".join(lines)


def _repair_citations(report: SynthesisReport, evidence_ids: set) -> SynthesisReport:
    """Deterministic citation repair: drop any id that is not in the verified
    evidence (the LLM is never the citation authority)."""
    def clean(cits: List[Citation]) -> List[Citation]:
        return [c for c in cits if c.requirement_id in evidence_ids or c.document_id in evidence_ids]
    sections = [s.model_copy(update={"citations": clean(s.citations)})
                for s in report.sections]
    return report.model_copy(
        update={"sections": sections, "citations": clean(report.citations)})


class FinalSynthesizer:
    """Writes the final answer from the final verified evidence set."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="synthesizer")
        self.agent = Agent(
            self.model,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            name="synthesizer_v3",
        )

    async def synthesize(self, state: V3RunState) -> SynthesisReport:
        from src.llm.run import ask_structured
        from src.lib.trace import get_trace

        trace = get_trace()
        prompt = _synthesis_prompt(state)
        trace.agent("synthesizer_v3", output_type="SynthesisReport",
                    meta={"evidence": len(state.verified_evidence()),
                          "contradictions": len(state.contradictions),
                          "tasks": len(state.tasks)})
        report = await ask_structured(
            self.agent,
            prompt,
            SynthesisReport,
            label="synthesizer_v3",
            max_tokens=min(2500, getattr(self.config, "agent_max_tokens", 2048)),
        )
        # citation repair binds every answer citation to a REAL verified id
        return _repair_citations(report, {e.id for e in state.verified_evidence()})
