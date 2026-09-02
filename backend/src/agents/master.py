"""
Master Orchestrator – decomposes a query into retrieval tasks via structured output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.agentic.state import EvidenceRequirement, MasterPlan, ResearchTask, RunBudget
from src.config import AppConfig
from src.lib.trace import get_trace
from src.llm import build_model_for
from src.prompts.load import load_prompt

MASTER_SYSTEM_PROMPT = load_prompt("agents", "master.txt")


class PlannedTask(BaseModel):
    id: str = Field(..., min_length=1, max_length=10)
    title: str = Field(..., min_length=1, max_length=100)
    objective: str = Field(..., min_length=1, max_length=400)
    intent: str = Field(..., min_length=1, max_length=200)
    evidence_required: list[str] = Field(..., min_length=1, max_length=3)
    entities: list[str] = Field(default_factory=list, max_length=6)


class Decomposition(BaseModel):
    rationale: str = Field(default="", max_length=500)
    tasks: list[PlannedTask] = Field(..., min_length=1, max_length=4)


class MasterOrchestratorAgent:
    def __init__(self, config=None, model=None):
        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="orchestrator")
        self.agent = Agent(
            self.model,
            system_prompt=MASTER_SYSTEM_PROMPT,
            output_type=Decomposition,
            retries=3,
            name="master_orchestrator",
        )

    async def plan(self, query: str, budget: RunBudget | None = None,
                   memory_context: str = "") -> MasterPlan:
        trace = get_trace()
        budget = budget or RunBudget()
        trace.agent("master_orchestrator", meta={"budget": budget.model_dump()})

        result = await self.agent.run(self._plan_prompt(query, memory_context))
        decomp = result.output
        target = budget.evidence_target or 3

        plan = MasterPlan(
            question=query,
            global_stop_criteria=["all tasks satisfied", "budget exhausted", "insufficient evidence"],
            rationale=decomp.rationale or "master decomposition",
            tasks=[
                ResearchTask(
                    id=t.id,
                    title=t.title,
                    objective=t.objective,
                    intent=t.intent,
                    entities=t.entities,
                    stop_criteria=["required evidence satisfied", "retrieval budget exhausted"],
                    evidence_requirements=[
                        EvidenceRequirement(id=f"{t.id}.R{j}", text=e, target_n=target)
                        for j, e in enumerate(t.evidence_required, 1)
                    ],
                )
                for t in decomp.tasks[:4]
            ],
        )

        trace.bullet(f"Divided into {len(plan.tasks)} task(s)", agent="master")
        return plan

    @staticmethod
    def _plan_prompt(query: str, memory_context: str = "") -> str:
        """Build the planner prompt.

        ``memory_context`` is ADVISORY prior-research memory (session state,
        prior claims/contradictions/gaps). It may inform decomposition, but
        it is explicitly labeled as NOT evidence — the evidence boundary is
        the retrieval pipeline, and the planner can never cite memory as a
        source. When empty (no memory layer attached) the prompt is exactly
        the historical one.

        The planning rules are the key memory-utilization safeguard: prior
        questions marked [answered] / listed under ALREADY INVESTIGATED must
        not be re-derived — the new plan is ONLY for the new question, with
        prior findings used as background (this is what makes follow-ups
        build on earlier runs instead of re-running them).
        """
        prompt = (
            f"USER QUESTION:\n{query}\n===========\n"
            "Produce the retrieval plan in JSON format."
        )
        if memory_context:
            prompt += (
                "\n\nPRIOR RESEARCH MEMORY (advisory context only — prior "
                "sessions/claims from the memory layer. NOT medical evidence, "
                "NEVER citable as a source):\n"
                + memory_context
                + "\n\nPLANNING RULES (mandatory):\n"
                "1) Plan tasks ONLY for the NEW USER QUESTION above.\n"
                "2) Any prior question marked [answered] or listed under "
                "ALREADY INVESTIGATED was already investigated — do NOT "
                "recreate its tasks.\n"
                "3) You MAY use prior conclusions/findings as background to "
                "focus the new plan and avoid redundant work, but every task "
                "must be about the NEW question.\n"
                "4) Follow-ups build ON prior findings; they are not the "
                "prior question re-run."
            )
        return prompt

