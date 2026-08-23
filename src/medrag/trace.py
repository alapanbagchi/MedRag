"""Structured execution tracer for MedRAG pipelines.

Captures every significant call with full request/response bodies,
timestamps, durations, and structured metadata.

Usage:
    from medrag.trace import TraceLogger

    trace = TraceLogger()
    trace.log("llm_call", params={"prompt": "...", "model": "deepseek"},
              result_text="expanded query...", duration_ms=1234)
    trace.save("trace.log")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _truncate(s: str, max_len: int = 500) -> str:
    """Truncate long strings for display, keeping head + tail."""
    if len(s) <= max_len:
        return s
    half = max_len // 2
    return s[:half] + f"\n    ... [{len(s) - max_len} chars truncated] ...\n    " + s[-half:]


def _fmt_list(items: Sequence[Any], max_items: int = 10) -> str:
    """Format a list for display, truncating if long."""
    if len(items) <= max_items:
        return str(list(items))
    return f"[{len(items)} items: {list(items[:3])} ... {list(items[-2:])}]"


@dataclass
class TraceEvent:
    """A single traced operation with full params and results."""
    timestamp: str          # ISO-8601 wall clock
    elapsed_ms: float       # ms since trace start
    duration_ms: float      # ms for this operation
    operation: str          # e.g. "llm_call", "pg_search", "rerank"
    params: Dict[str, Any] = field(default_factory=dict)   # full request/input
    result: Dict[str, Any] = field(default_factory=dict)    # full response/output
    result_text: str = ""   # short human-readable result summary
    detail: str = ""        # extra context
    error: str = ""         # error message if failed


class TraceLogger:
    """Append-only structured execution trace with full I/O capture."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._wall_start = time.time()
        self._events: List[TraceEvent] = []

    # ── logging API ───────────────────────────────────────────────

    def log(
        self,
        operation: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        result_text: str = "",
        detail: str = "",
        duration_ms: Optional[float] = None,
        error: str = "",
    ) -> None:
        """Record one trace event with full params and result.

        ``result`` may be a dict (structured response) or a str (short
        summary). Strings are routed to ``result_text`` for display.
        """
        now_perf = time.perf_counter()
        now_wall = time.time()
        elapsed = (now_perf - self._start) * 1000
        dur = duration_ms if duration_ms is not None else 0.0
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now_wall))
        frac = f"{now_wall:.3f}".split(".")[-1]
        ts = f"{ts}.{frac}"

        if result is not None and not isinstance(result, dict):
            # Legacy string result → keep as human-readable summary
            if not result_text:
                result_text = str(result)
            result = None

        self._events.append(TraceEvent(
            timestamp=ts,
            elapsed_ms=round(elapsed, 2),
            duration_ms=round(dur, 2),
            operation=operation,
            params=params or {},
            result=result or {},
            result_text=result_text,
            detail=detail,
            error=error,
        ))

    def step(self, label: str) -> _StepTimer:
        """Context manager that times a block and logs it automatically."""
        return _StepTimer(self, label)

    # ── serialization ─────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write the trace to a human-readable log file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []

        # Header
        lines.append("=" * 100)
        lines.append("MEDRAG EXECUTION TRACE")
        lines.append(f"Started:  {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self._wall_start))}")
        lines.append(f"Events:   {len(self._events)}")
        if self._events:
            total = self._events[-1].elapsed_ms
            lines.append(f"Duration: {total:.0f} ms ({total/1000:.1f}s)")
        lines.append("=" * 100)
        lines.append("")

        for i, ev in enumerate(self._events, 1):
            # ── Header line ──
            dur_str = f"{ev.duration_ms:>8.1f} ms" if ev.duration_ms > 0 else "       —"
            err_flag = " *** ERROR ***" if ev.error else ""
            lines.append(f"[{i:>4}] [{ev.timestamp}]  +{ev.elapsed_ms:>8.1f}ms  ({dur_str})  {ev.operation}{err_flag}")

            if ev.detail:
                lines.append(f"        DETAIL: {ev.detail}")

            # ── Params (full body) ──
            if ev.params:
                lines.append("        ┌─── REQUEST ───────────────────────────────────────────")
                for k, v in ev.params.items():
                    val_str = _format_value(v)
                    for line in val_str.split("\n"):
                        lines.append(f"        │ {k}: {line}")
                lines.append("        └──────────────────────────────────────────────────────")

            # ── Result (full body) ──
            if ev.result:
                lines.append("        ┌─── RESPONSE ──────────────────────────────────────────")
                for k, v in ev.result.items():
                    val_str = _format_value(v)
                    for line in val_str.split("\n"):
                        lines.append(f"        │ {k}: {line}")
                lines.append("        └──────────────────────────────────────────────────────")

            if ev.result_text:
                lines.append(f"        RESULT: {ev.result_text}")

            if ev.error:
                lines.append(f"        ERROR:  {ev.error}")

            lines.append("")

        lines.append("=" * 100)
        lines.append("END OF TRACE")
        lines.append("=" * 100)

        path.write_text("\n".join(lines), encoding="utf-8")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._wall_start)),
            "events": [
                {
                    "timestamp": ev.timestamp,
                    "elapsed_ms": ev.elapsed_ms,
                    "duration_ms": ev.duration_ms,
                    "operation": ev.operation,
                    "detail": ev.detail,
                    "result_text": ev.result_text,
                    "error": ev.error,
                    **({"params": ev.params} if ev.params else {}),
                    **({"result": ev.result} if ev.result else {}),
                }
                for ev in self._events
            ],
        }


def _format_value(v: Any) -> str:
    """Format a value for trace output, truncating long strings/lists."""
    if isinstance(v, str):
        return _truncate(v, 600)
    if isinstance(v, (list, tuple)):
        if len(v) > 15:
            return _fmt_list(v, 15)
        return _truncate(str(v), 600)
    if isinstance(v, dict):
        s = json.dumps(v, indent=0, ensure_ascii=False, default=str)
        return _truncate(s, 600)
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


class _StepTimer:
    """Context-manager helper for ``TraceLogger.step``."""

    def __init__(self, logger: TraceLogger, label: str) -> None:
        self._logger = logger
        self._label = label
        self._t0: float = 0.0

    def __enter__(self) -> _StepTimer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        dur = (time.perf_counter() - self._t0) * 1000
        status = "OK" if exc_info[0] is None else f"ERROR: {exc_info[0].__name__}"
        self._logger.log(
            self._label,
            result_text=status,
            duration_ms=dur,
            error=str(exc_info[1]) if exc_info[0] else "",
        )


# ── Global singleton ─────────────────────────────────────────────
_TRACE: Optional["TraceLogger"] = None

def get_trace() -> "TraceLogger":
    """Get the global trace logger (creates one if none set)."""
    global _TRACE
    if _TRACE is None:
        _TRACE = TraceLogger()
    return _TRACE

def set_trace(trace: "TraceLogger") -> None:
    """Set the global trace logger."""
    global _TRACE
    _TRACE = trace
