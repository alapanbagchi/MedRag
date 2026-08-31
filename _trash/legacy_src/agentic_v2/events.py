"""Agentic v2 — structured event stream (for the UI / observability).

The orchestrator loop can emit one JSON line per event to a file. A tiny web UI
(scripts/v2_ui.py + ui/agentic_v2.html) reads that file and renders the
research timeline: decisions, spawned agents, their inputs/outputs, retries and
failures, and the live state (objectives / evidence / gaps).

The emitter is deliberately optional and dependency-free (stdlib only). It is
disabled unless an events file path is provided (env ``AGENTIC_V2_EVENTS_FILE``
or ``config.agentic_v2_events_file``), so ordinary runs stay byte-for-byte
unchanged.

Event types (all carry ``ts``):
  run_start        {question, max_rounds, max_global_retrieves}
  decision         {iteration, action, objective_id, rationale, query, instructions}
  action_start     {iteration, action, objective_id}
  agent_spawn      {iteration, action, agent, input}
  agent_output     {iteration, action, agent, output, status}
  llm_call         {iteration, agent, label, input, thought, output}  (prompt/reasoning/response)
  step             {iteration, action, input, output, status}   (deterministic tools)
  action_done      {iteration, action, objective_id, status, outcome}
  failure          {iteration, action, error}
  state            {iteration, phase, objectives, documents, evidence, ...}
  phase_changed    {iteration, phase}
  policy_repair    {iteration, original_action, repaired_action, reason}
  no_progress      {iteration, action, strategy_key, reason}
  strategy_exhausted {iteration, action, strategy_key}
  timeout          {iteration, action, error}
  progress         {iteration, action, progress, summary, strategy_key}
  run_end          {stop_reason, iterations, terminal, confidence, phase, answer}
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional


class EventEmitter:
    """Appends one JSON object per line to ``path`` (thread-safe)."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def emit(self, type_: str, **fields: Any) -> None:
        record = {"ts": time.time(), "type": type_, **fields}
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
        except Exception:
            line = json.dumps({"ts": time.time(), "type": type_, "error": "unserializable"})
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def truncate(self) -> None:
        """Start a fresh event log (the UI calls this before a new run)."""
        with self._lock:
            open(self.path, "w", encoding="utf-8").close()


# ---------------------------------------------------------------------------
# Module-level helper (used by the UI server to (re)create one emitter per run).
# ---------------------------------------------------------------------------

_SINGLETON: Optional[EventEmitter] = None


def get_emitter(path: Optional[str] = None) -> Optional[EventEmitter]:
    global _SINGLETON
    if path:
        if _SINGLETON is None or _SINGLETON.path != path:
            _SINGLETON = EventEmitter(path)
        return _SINGLETON
    return _SINGLETON


def reset_emitter() -> None:
    global _SINGLETON
    _SINGLETON = None
