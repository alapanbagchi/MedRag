"""Agentic v2 — the Orchestrator Agent.

The orchestrator is the *controller* of the research process. It is not the
final answer generator and it is not a fixed pipeline. On every turn it:

  1. reads the persistent ``ResearchState`` (the external working memory),
  2. decides the SINGLE next action with the highest expected value,
  3. emits exactly one ``ActionDecision`` (one of the eight actions).

The decision loop itself lives in ``pipeline.py``; this module only produces
the decision from the current state. Keeping the decision as a typed, bounded
object makes the loop deterministic and testable (the LLM can be swapped for a
scripted sequence in tests).

The actions:
  DECOMPOSE, ENRICH, GLOBAL_RETRIEVE, READ_DOCUMENT, FIND_SECTIONS, VERIFY,
  SYNTHESIZE, STOP.
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
from typing import Any, List

from pydantic import BaseModel, Field

from src.agentic_v2.state import ActionType, ResearchState

logger = logging.getLogger("src.agentic_v2.orchestrator")


class ActionDecision(BaseModel):
    """Exactly one research decision, with the parameters the executor needs.

    The ``instructions`` field is the concise task for the selected agent/tool;
    the typed fields are hints the deterministic executor can fall back on when
    the model omits something.
    """
    action: ActionType
    objective_id: str = ""
    rationale: str = ""
    query: str = ""                 # for GLOBAL_RETRIEVE
    terms: List[str] = Field(default_factory=list)  # unused (kept for compatibility)
    document_id: str = ""           # for READ_DOCUMENT / FIND_SECTIONS
    chunk_id: str = ""              # for READ_DOCUMENT
    instructions: str = ""


ORCHESTRATOR_SYSTEM_PROMPT = load_prompt('agentic_v2', 'orchestrator.txt')


def _decision_prompt(
    state: ResearchState,
    phase: Any = None,
    legal_actions: Any = None,
) -> str:
    phase_line = f"CURRENT PHASE: {phase.value}" if phase is not None else ""
    legal_line = ""
    if legal_actions is not None:
        legal_line = "LEGAL ACTIONS: " + ", ".join(sorted(a.value for a in legal_actions))
    header = "\n".join(x for x in (phase_line, legal_line) if x)
    return (
        "CURRENT RESEARCH STATE\n"
        "======================\n"
        + (header + "\n" if header else "")
        + f"{state.summarize()}\n\n"
        "Decide the single next action that most reduces the most important "
        "evidence gap. Return EXACTLY ONE JSON decision object."
    )


class OrchestratorAgent:
    """Turns the current state into exactly one next action."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="orchestrator")
        self.agent = Agent(
            self.model,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            name="orchestrator",
        )

    async def decide(
        self,
        state: ResearchState,
        *,
        phase: Any = None,
        legal_actions: Any = None,
    ) -> ActionDecision:
        """Return the next action from the current state (policy-aware prompt)."""
        from src.llm.run import ask_structured
        from src.trace import get_trace

        if phase is None or legal_actions is None:
            from src.agentic_v2 import policy as policy_mod
            phase = phase or policy_mod.determine_phase(state)
            legal_actions = legal_actions or policy_mod.legal_actions(state)

        trace = get_trace()
        trace.agent("orchestrator", output_type="ActionDecision",
                    meta={"iteration": state.iteration, "objectives": len(state.objectives),
                          "phase": getattr(phase, "value", str(phase))})
        prompt = _decision_prompt(state, phase=phase, legal_actions=legal_actions)
        decision = await ask_structured(
            self.agent,
            prompt,
            ActionDecision,
            label="orchestrator",
            max_tokens=min(700, self.config.agent_max_tokens),
        )
        return self._repair(decision, state, phase=phase, legal_actions=legal_actions)

    @staticmethod
    def _repair(
        decision: ActionDecision,
        state: ResearchState,
        phase: Any = None,
        legal_actions: Any = None,
    ) -> ActionDecision:
        """Sanity-fix a decision (empty query + legality/exhaustion).

        * GLOBAL_RETRIEVE needs a query: fall back to the objective statement or
          the question.
        * Illegal / already-exhausted strategies are repaired deterministically
          by the policy layer (this is advisory hardening; the pipeline remains
          authoritative).
        """
        action = decision.action
        # Empty search query is unrecoverable for the executor; provide one.
        if action == ActionType.GLOBAL_RETRIEVE:
            if not (decision.query or "").strip():
                obj = state.objective(decision.objective_id) if decision.objective_id else None
                decision.query = (obj.statement if obj else state.question) or ""

        from src.agentic_v2 import policy as policy_mod
        decision, _repaired, _reason = policy_mod.validate_or_repair(
            decision, state, phase=phase, legal=legal_actions
        )
        return decision
