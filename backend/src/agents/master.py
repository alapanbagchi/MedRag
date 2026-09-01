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

    async def plan(self, query: str, budget: RunBudget | None = None) -> MasterPlan:
        trace = get_trace()
        budget = budget or RunBudget()
        trace.agent("master_orchestrator", meta={"budget": budget.model_dump()})

        result = await self.agent.run(
            f"USER QUESTION:\n{query}\n===========\nProduce the retrieval plan in JSON format."
        )
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

