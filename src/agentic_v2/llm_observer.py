"""Agentic v2 — LLM observer bridge (prompt / thought / response -> events).

Installs itself into ``src.llm.run`` (the single choke point every agent uses)
and forwards every LLM call as an ``llm_call`` event tagged with the current
iteration and a canonical agent name. The n8n-style UI uses these events to
attach the question / model thought / response to each node.

The current iteration is tracked with a contextvar set by the pipeline, so the
observer needs no threading and no signature changes on the agents.
"""

from __future__ import annotations

import contextvars
from typing import Any

_iteration: contextvars.ContextVar[int] = contextvars.ContextVar("av2_iteration", default=0)

_emitter: Any = None

# Map the ``label`` passed to ask_structured -> the canonical agent name used by
# the agent_spawn / decision events (so the UI can join them).
_AGENT_ALIAS = {
    "orchestrator": "orchestrator",
    "decompose_planner": "planner",
    "objective_verifier": "verifier",
    "synthesizer_v2": "synthesizer",
    "planner": "planner",
    "verifier": "verifier",
    "synthesizer": "synthesizer",
    "rewriter": "rewriter",
}


def install(emitter: Any) -> None:
    """Route LLM calls to ``emitter`` (an EventEmitter or any .emit(type_, **))."""
    global _emitter
    _emitter = emitter
    from src.llm import run as llm_run
    llm_run.set_llm_observer(_handle)


def uninstall() -> None:
    global _emitter
    _emitter = None
    from src.llm import run as llm_run
    llm_run.reset_llm_observer()


def set_iteration(iteration: int) -> None:
    _iteration.set(iteration)


def _handle(label: str, input: str, thought: str, output: Any,
            phase: str = "end") -> None:
    """Forward one LLM call to the event stream.

    Emits "llm_call_start" when the call begins (live spinner + timing span)
    and "llm_call" when it finishes (prompt / thought / response). The agent is
    aliased from the label's PREFIX so labels like "objective_verifier:1:P" still
    resolve to the canonical agent while the full label stays available.
    """
    if _emitter is None:
        return
    base = (label or "").split(":")[0]
    agent = _AGENT_ALIAS.get(base, base)
    if phase == "start":
        _emitter.emit(
            "llm_call_start",
            iteration=_iteration.get(),
            agent=agent,
            label=label,
        )
        return
    _emitter.emit(
        "llm_call",
        iteration=_iteration.get(),
        agent=agent,
        label=label,
        input=input,
        thought=thought,
        output=output,
    )
