"""xdeep logging: live progress on stdout PLUS structured events to logs.txt.

The existing pipeline writes to backend/logs.txt through src.lib.trace. xdeep
reuses the SAME file (same format family) so one log file tells the whole
story, with events prefixed [xdeep] to separate them from the legacy run:

  progress(msg)        - stdout live line (already in graph.py) + log line
  xdeep_log(...)       - structured '[xdeep]' event (stage/agent/tool call)
  open_run_log(query)  - start a MEDRAG-style RUN header + query
  close_run_log()      - close the run block

Web searches are logged explicitly (query, engine, trust gate outcome) so the
log shows when the agent reaches outside the corpus.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

# The log target is resolved from XDEEP_LOG_FILE at EVERY call (not frozen at
# import), so tests / deployments can redirect the log without restarting the
# process. Default: backend/logs.txt anchored at this package's parent (the
# backend dir), so the CLI, the API server and tests all share one log file
# regardless of CWD.
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOG = _BACKEND_DIR / "logs.txt"
_OPEN = False


def log_path() -> Path:
    return Path(os.environ.get("XDEEP_LOG_FILE", str(_DEFAULT_LOG)))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append(line: str) -> None:
    global _OPEN
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def open_run_log(query: str = "", append: bool = True) -> None:
    """Start a MEDRAG-style run header in logs.txt."""
    global _OPEN
    _OPEN = True
    _append("=" * 78)
    _append(f"[xdeep] RUN  {_now()}")
    if query:
        _append(f"[xdeep] QUERY: {query}")
    _append("=" * 78)


def close_run_log() -> None:
    global _OPEN
    _OPEN = False
    _append(f"[xdeep] END RUN  {_now()}")


def xdeep_log(event: str, **fields: Any) -> None:
    """Structured [xdeep] event line: [xdeep] <event> | k=v | k=v ..."""
    parts = [f"[xdeep] {event}"]
    for k, v in fields.items():
        s = v if isinstance(v, str) else str(v)
        if len(s) > 2000:
            s = s[:2000] + "..."
        parts.append(f"{k}={s}")
    _append(" | ".join(parts))


def xdeep_stage(title: str, **fields: Any) -> None:
    """A stage boundary in the log (mirrors trace.stage)."""
    _append("")
    _append(f"==== [xdeep] {title}  ({_now()}) ====")
    if fields:
        xdeep_log("stage_fields", **fields)
