"""Agentic v3 - Stage 15: the Contradiction Resolution Agent.

When a contradiction is detected, the system does NOT automatically choose
one paper. It opens a dedicated Resolution Agent whose narrow purpose is
(spec section 20):

    Use the paper search tool to investigate the contradiction and determine
    whether it can be resolved.

It searches for additional literature (larger studies, meta-analyses,
different populations/dosages/durations/designs/outcomes) and then compares
the new evidence against the conflicting sides.

Resolution paths (sections 21-22):
  RESOLVED       - the conflict is explained (e.g. effect seen only in
                   vitamin-D-deficient participants; findings are
                   context-dependent, not one study "wrong"), or
  UNRESOLVED     - the system could not determine why the evidence
                   conflicts -> it MUST preserve the uncertainty; the final
                   answer will explicitly report the unresolved conflict.

The paper search tool is the same retriever the workers use (candidate
papers are never trusted as verified evidence inside resolution either -
the original verified evidence remains the cited material; the searched
papers only inform the resolution reasoning and are recorded as
additional_papers, never as new citations).
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
from typing import Any, List

from pydantic import BaseModel, Field

from src.lib import plural
from src.agentic_v3.retriever import PaperRetrieverTool
from src.agentic_v3.state import (
    Contradiction,
    EvidenceRequirement,
    ResolutionOutcome,
    ResolutionStatus,
    ResearchTask,
    V3RunState,
)

logger = logging.getLogger("src.agentic_v3.resolution")


class ResolutionQueryPlan(BaseModel):
    """Search formulations the resolution agent wants to run."""
    rationale: str = ""
    queries: List[str] = Field(default_factory=list)


class StudyContext(BaseModel):
    """Context facets of ONE side of a contradiction (requirement 6)."""
    population: str = ""              # e.g. "hypertensive, vitamin-D-deficient adults"
    intervention_exposure: str = ""
    outcome: str = ""
    study_design: str = ""            # RCT / cohort / meta-analysis / ...
    dosage_duration: str = ""
    baseline_characteristics: str = ""
    measurement: str = ""
    research_context: str = ""        # journal/era/population setting, if visible


class ContradictionCharacterization(BaseModel):
    """The evidence-aware comparison performed BEFORE any judgement.

    The resolver compares the two sides on the factors the spec lists
    (population, intervention/exposure, outcome, study design, baseline
    characteristics, dosage, duration, measurement differences, publication
    context), decides whether the conflict is GENUINE, and only then reaches
    a resolution status. It never silently reinterprets irrelevant evidence.
    """
    side_a: StudyContext = Field(default_factory=StudyContext)
    side_b: StudyContext = Field(default_factory=StudyContext)
    differing_factors: List[str] = Field(default_factory=list)
    conflict_genuine: bool = True
    genuine_reason: str = ""
    explained_by: str = ""            # the factor that explains the difference


class ResolutionDecision(BaseModel):
    """The resolution agent's structured judgement after the searches."""
    status: str = "unresolved"        # resolved | partially_resolved | unresolved
    explanation: str = ""
    characterization: str = ""        # e.g. "context-dependent: baseline vitamin D status"


RESOLUTION_SYSTEM_PROMPT = load_prompt('agentic_v3', 'resolution.txt')


def _contradiction_block(contradiction: Contradiction, state: V3RunState) -> str:
    by_id = state.evidence_by_id()
    lines = [
        f"CLAIM: {contradiction.claim}",
        f"KIND: {contradiction.kind.value}",
        "SIDE A evidence:",
    ]
    for eid in contradiction.evidence_a:
        e = by_id.get(eid)
        if e:
            lines.append(f"  - {eid} (doc={e.document_id}, support={e.support.value}, "
                         f"conf={e.confidence:.2f}): {' '.join((e.excerpt or '').split())[:400]}")
    lines.append("SIDE B evidence:")
    for eid in contradiction.evidence_b:
        e = by_id.get(eid)
        if e:
            lines.append(f"  - {eid} (doc={e.document_id}, support={e.support.value}, "
                         f"conf={e.confidence:.2f}): {' '.join((e.excerpt or '').split())[:400]}")
    return "\n".join(lines)


class ResolutionAgent:
    """Resolves (or honestly fails to resolve) one contradiction."""

    def __init__(self, config: Any = None, model: Any = None,
                 retriever: Any = None, max_additional_papers: int = 5):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="resolution")
        self.agent = Agent(
            self.model,
            system_prompt=RESOLUTION_SYSTEM_PROMPT,
            name="resolution_agent",
        )
        self.retriever = retriever or PaperRetrieverTool(config=self.config)
        self.max_additional_papers = max_additional_papers

    async def resolve(self, contradiction: Contradiction,
                      state: V3RunState) -> ResolutionOutcome:
        """Investigate the contradiction with the paper search tool."""
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        block = _contradiction_block(contradiction, state)

        # 1. evidence-aware CHARACTERIZATION of both sides (requirement 6):
        #    compare population / exposure / outcome / design / baseline /
        #    dosage / duration / measurement / context BEFORE judging.
        characterization: Optional[ContradictionCharacterization] = None
        try:
            characterization = await ask_structured(
                self.agent,
                block
                + "\n\nCHARACTERISE both sides of this contradiction on the "
                  "study-context factors (population, intervention/exposure, "
                  "outcome, study design, baseline characteristics, dosage, "
                  "duration, measurement, research context). Decide whether "
                  "the conflict is GENUINE. Return the characterization JSON.",
                ContradictionCharacterization,
                label=f"resolution_characterize:{contradiction.id}",
                max_tokens=min(1100, getattr(self.config, "agent_max_tokens", 2048)),
            )
        except Exception as exc:
            logger.warning("resolution characterization failed (%s)", exc)

        # 2. the agent proposes search queries (query-planning stage).
        queries: List[str] = []
        try:
            plan = await ask_structured(
                self.agent,
                self._query_prompt(block, characterization),
                ResolutionQueryPlan,
                label=f"resolution_queries:{contradiction.id}",
                max_tokens=min(700, getattr(self.config, "agent_max_tokens", 2048)),
            )
            queries = [q for q in (plan.queries or []) if q and len(q) >= 7][:3]
        except Exception as exc:
            logger.warning("resolution query planning failed (%s)", exc)

        # 2. paper search tool: additional literature (candidate context only).
        additional_papers: List[str] = []
        seen_docs: set = set()
        if queries:
            for q in queries:
                papers = await self._search_tool(q)
                for p in papers:
                    if p and p not in seen_docs:
                        seen_docs.add(p)
                        additional_papers.append(p)
                if len(additional_papers) >= self.max_additional_papers:
                    break
        trace.bullet(
            f"Looking for a resolution of {contradiction.id}: ran "
            f"{plural(len(queries), 'search query')} and read "
            f"{plural(len(additional_papers), 'additional paper')} for "
            f"context (queries: {queries}).",
            agent="resolution"
        )

        # 3. final judgement; unresolved is always an available, honest exit.
        outcome = ResolutionOutcome(
            status=ResolutionStatus.UNRESOLVED,
            explanation="could not determine why the verified evidence conflicts",
            additional_queries=queries,
            additional_papers=additional_papers,
            characterization="unresolved",
        )
        try:
            decision = await ask_structured(
                self.agent,
                self._judgement_prompt(block, queries, additional_papers,
                                       characterization),
                ResolutionDecision,
                label=f"resolution_judgement:{contradiction.id}",
                max_tokens=min(900, getattr(self.config, "agent_max_tokens", 2048)),
            )
        except Exception as exc:
            logger.warning("resolution judgement failed (%s); unresolved kept", exc)
            return outcome

        status = ResolutionStatus.UNRESOLVED
        if (decision.status or "").strip().lower() == "resolved":
            status = ResolutionStatus.RESOLVED
        elif (decision.status or "").strip().lower() == "partially_resolved":
            status = ResolutionStatus.PARTIALLY_RESOLVED
        characterization_key = (
            (characterization.explained_by or "").strip()
            if characterization and not characterization.conflict_genuine
            else (decision.characterization or "").strip())
        outcome = ResolutionOutcome(
            status=status,
            explanation=(decision.explanation or "").strip()[:800],
            additional_queries=queries,
            additional_papers=additional_papers,
            characterization=characterization_key[:200],
        )
        # if the conflict was shown NON-genuine, carry the evidence-aware
        # explanation into the outcome
        if characterization and not characterization.conflict_genuine and \
                not outcome.explanation:
            outcome.explanation = characterization.genuine_reason[:800]
        return outcome

    async def _search_tool(self, query: str) -> List[str]:
        """Paper search tool: returns document ids of additional literature."""
        from src.agentic_v3.state import ResearchTask, EvidenceRequirement

        task = ResearchTask(id="RES", title="contradiction resolution",
                            objective=query[:200])
        req = EvidenceRequirement(id="RES.R1", text=query[:200], target_n=1)
        try:
            papers = await self.retriever.search(task, req, query,
                                                 top_k=3, round_no=0)
            return [p.document_id for p in papers if p.document_id]
        except Exception as exc:
            logger.warning("resolution search failed for %r: %s", query, exc)
            return []

    @staticmethod
    def _query_prompt(contradiction_block: str,
                      characterization: Optional[ContradictionCharacterization]) -> str:
        block = contradiction_block
        if characterization is not None:
            block += "\n\nCOMPARISON:"
            block += f"\n- side A: {characterization.side_a.model_dump(mode='json')}"
            block += f"\n- side B: {characterization.side_b.model_dump(mode='json')}"
            block += f"\n- differing factors: {', '.join(characterization.differing_factors)}"
            block += f"\n- conflict genuine: {characterization.conflict_genuine}"
            if not characterization.conflict_genuine:
                block += f"\n- not genuine because: {characterization.genuine_reason}"
        return (
            block
            + "\n\nDesign 1-3 targeted search queries to investigate whether "
            "this contradiction can be resolved (meta-analyses, larger "
            "studies, specific populations/dosages/designs/outcomes). Return "
            "the queries JSON."
        )

    @staticmethod
    def _judgement_prompt(contradiction_block: str, queries: List[str],
                          additional_papers: List[str],
                          characterization: Optional[ContradictionCharacterization]) -> str:
        block = contradiction_block
        if characterization is not None:
            block += "\n\nSTUDY-CONTEXT COMPARISON:"
            block += f"\n- side A: {characterization.side_a.model_dump(mode='json')}"
            block += f"\n- side B: {characterization.side_b.model_dump(mode='json')}"
            block += f"\n- differing factors: {', '.join(characterization.differing_factors)}"
            block += f"\n- conflict genuine: {characterization.conflict_genuine}"
            if not characterization.conflict_genuine:
                block += f"\n- not genuine because: {characterization.genuine_reason}"
        if queries:
            block += "\n\nSEARCH QUERIES USED: " + " | ".join(queries)
        if additional_papers:
            block += "\nADDITIONAL LITERATURE FOUND (documents): " + ", ".join(additional_papers)
        else:
            block += "\nADDITIONAL LITERATURE FOUND: (none)"
        return (
            block
            + "\n\nMake the final resolution judgement (resolved / "
            "partially_resolved / unresolved) with explanation."
        )
