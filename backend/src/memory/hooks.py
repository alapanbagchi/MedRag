"""Memory + Context layer — AgenticV3Pipeline integration hooks.

Keeps the evidence boundary intact: the deep agents (master/workers/critic)
are NOT redesigned. This adapter only

  * BEFORE a run:  resumes the research session and assembles the bounded
                   memory context region (``prepare_run``);
  * AFTER a run:   persists the verified evidence, claims, contradictions,
                   gaps and the labeled conclusion (``record_run``).

The pipeline treats the memory layer as an external client: it passes an
optional ``memory`` object, and the hooks here do the rest. When no memory
object is attached, behavior is byte-identical to before (all existing tests
keep passing).
"""

from __future__ import annotations

from typing import Any

from src.memory.api import MemoryAPI, RunPrep, RunRecordStats

_LOG_PREFIX = "[memory]"


class MemoryRunHooks:
    """Synchronous helpers used by AgenticV3Pipeline around its run."""

    def __init__(self, memory: MemoryAPI):
        self.memory = memory
        self.last_prep: RunPrep | None = None

    def prepare(self, query: str, user_id: str = "",
                conversation_id: str | None = None) -> RunPrep:
        """Resume session + assemble memory context (sync; cheap local ops)."""
        prep = self.memory.prepare_run(query, user_id=user_id,
                                       conversation_id=conversation_id)
        self.last_prep = prep
        return prep

    def record(self, result: dict[str, Any],
               session_id: str = "",
               user_id: str = "",
               conversation_id: str | None = None) -> RunRecordStats:
        """Persist the finished run into persistent research memory."""
        stats = self.memory.record_run(
            result, session_id=session_id, user_id=user_id,
            conversation_id=conversation_id)
        return stats

    def context_text(self) -> str:
        return self.last_prep.context_text() if self.last_prep else ""


__all__ = ["MemoryRunHooks", "MemoryAPI", "RunPrep", "RunRecordStats"]