"""Incremental streaming trace -> logs.txt (+ Logfire when enabled).

Every event (stage, agent call, prompt, raw model response, retrieval ranks,
worker spawn) is written to the log file immediately and flushed, so logs.txt
is always current even if the process dies. When Logfire is enabled
(see src.logfire_obs) every event is also forwarded there as a span/log with
run/task context attributes, so the same call sites feed both sinks.

Usage:
    trace = get_trace()
    trace.open_stream("logs.txt", query="...")
    trace.stage("STAGE 1 - PLAN")
    trace.agent("planner", output_type="QueryPlan")
    trace.prompt("user_prompt", "...")
    trace.response("raw", "...")
    trace.close_stream()

Full untruncated chunk/unit texts are only logged when TRACE_FULL_TEXTS=1;
by default long texts are truncated (they previously produced multi-MB logs
that made runs slower and diffs impossible).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_TEXT_LIMIT = int(os.environ.get("TRACE_TEXT_CHARS", "1500"))


def _full_texts_enabled() -> bool:
    return os.environ.get("TRACE_FULL_TEXTS", "").strip().lower() in ("1", "true", "yes", "on")


class Trace:
    """Appends structured, human-readable lines to a log file, flushing each write."""

    def __init__(self) -> None:
        self._fh = None
        self._path: Optional[Path] = None

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._fh is not None and not self._fh.closed

    @staticmethod
    def _logfire():
        """The Logfire bridge when active (lazy import; never raises)."""
        try:
            from src import logfire_obs
            return logfire_obs if logfire_obs.enabled() else None
        except Exception:
            return None

    def open_stream(self, path: str | Path, query: str = "") -> None:
        """Open the log file for incremental writing (append+flush)."""
        self.close_stream()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w", encoding="utf-8")
        self._write("=" * 78)
        self._write(f"MEDRAG RUN  {self._now()}")
        if query:
            self._write(f"QUERY: {query}")
        self._write("=" * 78)

    def close_stream(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    # ------------------------------------------------------------------
    def _write(self, line: str) -> None:
        if self.active:
            self._fh.write(line + "\n")
            self._fh.flush()

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _truncate(text: Any, limit: int = 60000) -> str:
        s = text if isinstance(text, str) else str(text)
        if len(s) <= limit:
            return s
        return s[:limit] + f"\n... [truncated {len(s) - limit} chars]"

    # ------------------------------------------------------------------
    # Public event methods
    # ------------------------------------------------------------------
    def stage(self, title: str) -> None:
        lf = self._logfire()
        if lf is not None:
            lf.stage(title)
        self._write("")
        self._write(f"==== {title}  ({self._now()}) ====")

    def bullet(self, msg: str, agent: str = "") -> None:
        lf = self._logfire()
        if lf is not None:
            lf.bullet(msg, agent=agent)
        prefix = f"{agent.upper()}: " if agent else ""
        self._write(f"  > {prefix}{msg}")

    def agent(self, name: str, output_type: str = "", meta: dict | None = None) -> None:
        lf = self._logfire()
        if lf is not None:
            lf.agent(name, output_type=output_type, meta=meta)
        parts = [f"call {name}"]
        if output_type:
            parts.append(f"output_type={output_type}")
        if meta:
            parts.append(json.dumps(meta))
        self._write(f"[agent] {' | '.join(parts)}")

    def prompt(self, label: str, text: Any) -> None:
        lf = self._logfire()
        if lf is not None:
            lf.prompt(label, text)
        self._write(f"[prompt] {label}:")
        self._write(self._indent(self._truncate(text)))

    def response(self, label: str, text: Any) -> None:
        lf = self._logfire()
        if lf is not None:
            lf.response(label, text)
        self._write(f"[response] {label}:")
        self._write(self._indent(self._truncate(text)))

    def parsed(self, type_name: str, data: Any) -> None:
        lf = self._logfire()
        if lf is not None:
            lf.parsed(type_name, data)
        self._write(f"[parsed] {type_name}: {self._truncate(data, 4000)}")

    def log(self, event: str, **fields: Any) -> None:
        lf = self._logfire()
        if lf is not None:
            lf.log(event, **fields)
        parts = [f"[event] {event}"]
        for k, v in fields.items():
            parts.append(f"{k}={self._truncate(v, 2000)}")
        self._write(" | ".join(parts))

    def tool(self, name: str, args: dict, result: Any = None) -> None:
        """Log a tool call.

        Call once at invocation (result omitted) -> prints ``[tool]``.
        Passing ``result`` prints ONLY the result line — use it that way at
        completion if you didn't log the invocation, never both, or every
        call shows up twice in the trace.
        """
        if result is None:
            lf = self._logfire()
            if lf is not None:
                lf.tool(name, args)
            self._write(f"[tool] {name} args={json.dumps(args, default=str)}")
            return
        lf = self._logfire()
        if lf is not None:
            lf.tool(name, args, result)
        self._write(f"[tool-result] {name}: {self._truncate(result, 4000)}")

    def retrieved(self, method: str, query: str, ranks: list, texts=None) -> None:
        """Log raw per-method rankings: [(chunk_id, score), ...]."""
        lf = self._logfire()
        if lf is not None:
            lf.retrieved(method, query, ranks, texts)
        self._write(f"[retrieved] {method} | query={query[:80]!r} | n={len(ranks)}")
        texts = texts or {}
        full = _full_texts_enabled()
        for i, (cid, score) in enumerate(ranks, 1):
            self._write(f"    rank{i:>4}  score={score:.5f}  {cid}")
            if cid in texts:
                body = str(texts[cid])
                if not body.strip():
                    self._write("       | (empty text)")
                    continue
                if not full:
                    head = " ".join(body.split())[:_TEXT_LIMIT]
                    suffix = f" ... [+{len(body) - len(head)} chars]" if len(body) > len(head) else ""
                    self._write(f"       | {head}{suffix}")
                    continue
                self._write("    -------- FULL TEXT (untruncated) --------")
                for ln in body.splitlines():
                    self._write(f"       | {ln}")
                self._write("    -------- END TEXT --------")

    def wake(self, count: int, kind: str = "") -> None:
        lf = self._logfire()
        if lf is not None:
            lf.log("workers_spawned", count=count, kind=kind)
        self._write(f"[workers] spawned {count} {kind} task(s)")

    def chunk(self, doc: Any, rank: int = 0, label: str = "retrieved") -> None:
        """Log ONE retrieved chunk with FULL untruncated text + metadata."""
        lf = self._logfire()
        if lf is not None:
            lf.chunk(doc, rank=rank, label=label)
        cid = getattr(doc, "chunk_id", None) or getattr(doc, "id", "")
        doc_id = getattr(doc, "document_id", "")
        node = getattr(doc, "node_type", "") or getattr(doc, "chunk_type", "")
        section = getattr(doc, "section", "")
        subsection = getattr(doc, "subsection", "")
        breadcrumb = list(getattr(doc, "breadcrumb", []) or [])
        table_id = getattr(doc, "table_id", None)
        figure_id = getattr(doc, "figure_id", None)
        score = getattr(doc, "rrf_score", None)
        methods = getattr(doc, "methods", [])
        raw_text = getattr(doc, "text", "")

        self._write("")
        self._write(f"[chunk:{label}] rank={rank} | {cid}")
        self._write(f"    document_id : {doc_id}")
        self._write(f"    node_type   : {node}")
        self._write(f"    section     : {section!r}")
        self._write(f"    subsection  : {subsection!r}")
        self._write(f"    breadcrumb  : {' > '.join(str(b) for b in breadcrumb)}")
        self._write(f"    table_id    : {table_id}   figure_id: {figure_id}")
        if score is not None:
            self._write(f"    rrf_score   : {float(score):.5f}")
        if methods:
            self._write(f"    methods     : {methods}")
        body = str(raw_text)
        if not _full_texts_enabled():
            head = " ".join(body.split())[:_TEXT_LIMIT]
            suffix = f" ... [+{len(body) - len(head)} chars]" if len(body) > len(head) else ""
            self._write("    -------- TEXT (truncated; TRACE_FULL_TEXTS=1 for all) --------")
            self._write(f"    | {head}{suffix}" if head else "    | (empty text)")
            return
        self._write("    -------- FULL TEXT (untruncated) --------")
        for ln in body.splitlines():
            self._write(f"    | {ln}")
        if not body.strip():
            self._write("    | (empty text)")
        self._write("    -------- END TEXT --------")

    @staticmethod
    def _indent(text: str, prefix: str = "    ") -> str:
        return "\n".join(prefix + ln for ln in text.splitlines())


# Module-level singleton
_TRACE = Trace()


def get_trace() -> Trace:
    """Return the process-wide trace logger."""
    return _TRACE


def set_trace(t: Trace) -> None:
    global _TRACE
    _TRACE = t


# Compat alias: retrieval_v2 still imports this historical name.
TraceLogger = Trace