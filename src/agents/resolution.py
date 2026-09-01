"""Agentic v3 - Stage 15: contradiction resolution agent (LLM + paper search tool)."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.tools.paper_retriever import PaperRetrieverTool
from src.agentic.state import (
    Contradiction,
    EvidenceRequirement,
    ResearchTask,
    ResolutionOutcome,
    ResolutionStatus,
    V3RunState,
)
from src.lib.trace import get_trace
from src.lib.utils import plural
from src.prompts.load import load_prompt

logger = logging.getLogger("src.agents.resolution")


class ResolutionQueryPlan(BaseModel):
    """Search formulations the resolution agent wants to run."""
    rationale: str = ""
    queries: list[str] = Field(default_factory=list)


class StudyContext(BaseModel):
    """Context facets of ONE side of a contradiction."""
    population: str = ""
    intervention_exposure: str = ""
    outcome: str = ""
    study_design: str = ""
    dosage_duration: str = ""
    baseline_characteristics: str = ""
    measurement: str = ""
    research_context: str = ""


class ContradictionCharacterization(BaseModel):
    """Evidence-aware comparison performed BEFORE any judgement."""
    side_a: StudyContext = Field(default_factory=StudyContext)
    side_b: StudyContext = Field(default_factory=StudyContext)
    differing_factors: list[str] = Field(default_factory=list)
    conflict_genuine: bool = True
    genuine_reason: str = ""
    explained_by: str = ""


class ResolutionDecision(BaseModel):
    """The resolution agent's structured judgement after the searches."""
    status: str = "unresolved"
    explanation: str = ""
    characterization: str = ""


RESOLUTION_SYSTEM_PROMPT = load_prompt('agents', 'resolution.txt')


def _contradiction_block(contradiction: Contradiction, state: V3RunState) -> str:
    by_id = state.evidence_by_id()
    lines = [f"CLAIM: {contradiction.claim}", f"KIND: {contradiction.kind.value}"]
    for side in ("SIDE A", "SIDE B"):
        ids = contradiction.evidence_a if side == "SIDE A" else contradiction.evidence_b
        lines.append(f"{side} evidence:")
        for eid in ids:
            e = by_id.get(eid)
            if e:
                lines.append(f"  - {eid} (doc={e.document_id}, support={e.support.value}, "
                             f"conf={e.confidence:.2f}): "
                             f"{' '.join((e.excerpt or '').split())[:400]}")
    return "\n".join(lines)


def _comparison(characterization: ContradictionCharacterization | None,
                label: str) -> str:
    if characterization is None:
        return ""
    c = characterization
    block = f"\n\n{label}:"
    block += f"\n- side A: {c.side_a.model_dump(mode='json')}"
    block += f"\n- side B: {c.side_b.model_dump(mode='json')}"
    block += f"\n- differing factors: {', '.join(c.differing_factors)}"
    block += f"\n- conflict genuine: {c.conflict_genuine}"
    if not c.conflict_genuine:
        block += f"\n- not genuine because: {c.genuine_reason}"
    return block


class ResolutionAgent:
    """Resolves (or honestly fails to resolve) one contradiction via 3 LLM
    steps + the paper search tool (searched papers are context only)."""

    def __init__(self, config: Any = None, model: Any = None,
                 retriever: Any = None, max_additional_papers: int = 5):
        from pydantic_ai import Agent

        from src.config import AppConfig
        from src.llm import build_model_for

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="resolution")
        self.agent = Agent(self.model, system_prompt=RESOLUTION_SYSTEM_PROMPT,
                           retries=2, name="resolution_agent")
        self.retriever = retriever or PaperRetrieverTool(config=self.config)
        self.max_additional_papers = max_additional_papers

    async def _llm(self, prompt: str, output_type) -> Any:
        return (await self.agent.run(prompt, output_type=output_type)).output

    async def resolve(self, contradiction: Contradiction,
                      state: V3RunState) -> ResolutionOutcome:
        """Characterize -> search -> judge; unresolved is always an honest exit."""
        block = _contradiction_block(contradiction, state)

        # 1. characterize both sides (study-context factors) BEFORE judging
        characterization: ContradictionCharacterization | None = None
        try:
            characterization = await self._llm(
                block + "\n\nCHARACTERISE both sides of this contradiction on the "
                        "study-context factors (population, intervention/exposure, "
                        "outcome, study design, baseline characteristics, dosage, "
                        "duration, measurement, research context). Decide whether "
                        "the conflict is GENUINE.",
                ContradictionCharacterization)
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolution characterization failed (%s)", exc)

        # 2. propose queries, then run the paper search tool
        queries: list[str] = []
        try:
            plan = await self._llm(
                block + _comparison(characterization, "COMPARISON")
                + "\n\nDesign 1-3 targeted search queries to investigate whether "
                  "this contradiction can be resolved (meta-analyses, larger "
                  "studies, specific populations/dosages/designs/outcomes).",
                ResolutionQueryPlan)
            queries = [q for q in (plan.queries or []) if q and len(q) >= 7][:3]
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolution query planning failed (%s)", exc)

        additional_papers: list[str] = []
        seen: set = set()
        if queries:
            for q in queries:
                for p in await self._search_tool(q):
                    if p and p not in seen:
                        seen.add(p)
                        additional_papers.append(p)
                if len(additional_papers) >= self.max_additional_papers:
                    break
        get_trace().bullet(
            f"Resolving {contradiction.id}: ran {plural(len(queries), 'search query')}, "
            f"read {plural(len(additional_papers), 'additional paper')} (queries: {queries}).",
            agent="resolution")

        # 3. final judgement
        outcome = ResolutionOutcome(
            status=ResolutionStatus.UNRESOLVED,
            explanation="could not determine why the verified evidence conflicts",
            additional_queries=queries, additional_papers=additional_papers,
            characterization="unresolved")
        try:
            decision = await self._llm(
                block + _comparison(characterization, "STUDY-CONTEXT COMPARISON")
                + ("\n\nSEARCH QUERIES USED: " + " | ".join(queries) if queries else "")
                + ("\nADDITIONAL LITERATURE FOUND (documents): "
                   + ", ".join(additional_papers) if additional_papers
                   else "\nADDITIONAL LITERATURE FOUND: (none)")
                + "\n\nMake the final resolution judgement (resolved / "
                  "partially_resolved / unresolved) with explanation.",
                ResolutionDecision)
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolution judgement failed (%s); unresolved kept", exc)
            return outcome

        status = ResolutionStatus.UNRESOLVED
        low = (decision.status or "").strip().lower()
        if low == "resolved":
            status = ResolutionStatus.RESOLVED
        elif low == "partially_resolved":
            status = ResolutionStatus.PARTIALLY_RESOLVED
        characterization_key = (
            (characterization.explained_by or "").strip()
            if characterization and not characterization.conflict_genuine
            else (decision.characterization or "").strip())
        outcome = ResolutionOutcome(
            status=status, explanation=(decision.explanation or "").strip()[:800],
            additional_queries=queries, additional_papers=additional_papers,
            characterization=characterization_key[:200])
        if characterization and not characterization.conflict_genuine and not outcome.explanation:
            outcome.explanation = characterization.genuine_reason[:800]
        return outcome

    async def _search_tool(self, query: str) -> list[str]:
        """Paper search tool: document ids of additional literature (context only)."""
        task = ResearchTask(id="RES", title="contradiction resolution", objective=query[:200])
        req = EvidenceRequirement(id="RES.R1", text=query[:200], target_n=1)
        try:
            papers = await self.retriever.search(task, req, query, top_k=3, round_no=0)
            return [p.document_id for p in papers if p.document_id]
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolution search failed for %r: %s", query, exc)
            return []