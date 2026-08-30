"""Logfire observability bridge — PydanticAI GenAI tracing + app trace events.

PydanticAI 2.x does NOT instrument agents just because ``logfire`` is
installed: ``Agent._instrument_default`` is ``False`` and instrumentation has
to be enabled explicitly. This module is the single place that

  1. configures Logfire once per process (token from ``LOGFIRE_TOKEN`` or the
     credentials file written by ``logfire auth`` under ``.logfire/``),
  2. enables PydanticAI's GenAI instrumentation for EVERY agent
     (``logfire.instrument_pydantic_ai()`` == ``Agent.instrument_all``, which
     is read at run time, so agents created before or after are all traced):
     each model request emits OpenTelemetry GenAI spans — system instructions,
     input/output messages, model + request parameters — plus token-usage and
     latency metrics,
  3. forwards the application-level trace events (stages, agent calls,
     retrieval ranks, verdicts, deep inspections, worker reports) into Logfire
     as spans/logs carrying run/task context attributes, and
  4. records a small set of aggregate metrics (LLM latency, prompt tokens,
     call counts, retrieval counts, evidence verdicts).

Gating — ``LOGFIRE_ENABLED`` (env or ``AppConfig.logfire_mode``):
    auto (default)  enabled iff credentials/token are present (``.logfire/``
                    logfire_credentials.json or ``LOGFIRE_TOKEN``),
    1/on/true       force enable,
    0/off/false     force disable,
With no token present, ``send_to_logfire="if-token-present"`` keeps the app
fully functional offline (console spans only).

Every helper below is a no-op when Logfire is disabled, so call sites never
need to branch.
"""

from __future__ import annotations

import contextvars
import json
import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import logfire

_LOG_DIR = ".logfire"
_CREDENTIALS_FILE = "logfire_credentials.json"

_CONFIGURED = False
_ENABLED = False
_metrics: Dict[str, Any] = {}

_run_ctx: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "medrag_run_ctx", default={})

_TEXT_LIMIT = int(os.environ.get("TRACE_TEXT_CHARS", "1500"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def credentials_present() -> bool:
    """True when a token can be sourced (env or the logfire credentials file)."""
    if os.environ.get("LOGFIRE_TOKEN", "").strip():
        return True
    return (_project_root() / _LOG_DIR / _CREDENTIALS_FILE).is_file()


def _mode(config: Any = None) -> str:
    if config is not None and getattr(config, "logfire_mode", ""):
        return str(config.logfire_mode).strip().lower()
    return os.environ.get("LOGFIRE_ENABLED", "auto").strip().lower()


def _console_option() -> Any:
    """Preserve the app's quiet-terminal default; opt in via LOGFIRE_CONSOLE."""
    value = os.environ.get("LOGFIRE_CONSOLE", "").strip().lower()
    if value in ("1", "true", "on", "yes", "auto"):
        return None  # logfire's own default (coloured console)
    if value in ("0", "false", "off", "no"):
        return False
    return False  # unset -> stay quiet (README: terminal is quiet by default)


def _data_dir() -> Optional[str]:
    """Credentials dir; pin to the project .logfire so cwd does not matter."""
    env_dir = os.environ.get("LOGFIRE_CREDENTIALS_DIR", "").strip()
    if env_dir:
        return env_dir
    return str(_project_root() / _LOG_DIR)


def enabled() -> bool:
    return _ENABLED


def ensure_configured(config: Any = None) -> bool:
    """Idempotent one-time setup; safe to call from any entry point."""
    if not _CONFIGURED:
        configure_logfire(config)
    return _ENABLED


def configure_logfire(config: Any = None) -> bool:
    """Configure Logfire + PydanticAI instrumentation once. Returns enabled.

    ``config`` may be an ``AppConfig`` (its ``logfire_*`` fields win over
    env); with ``None`` only environment variables are consulted.
    """
    global _CONFIGURED, _ENABLED
    if _CONFIGURED:
        return _ENABLED
    _CONFIGURED = True

    mode = _mode(config)
    if mode in ("0", "false", "off", "no", "disabled"):
        _ENABLED = False
        return False
    force_on = mode in ("1", "true", "on", "yes")
    if not force_on and not credentials_present():
        _ENABLED = False
        return False

    service = (
        getattr(config, "logfire_service_name", "") or
        os.environ.get("LOGFIRE_SERVICE_NAME") or "medrag"
    ).strip() or "medrag"
    environment = (
        getattr(config, "logfire_environment", "") or
        os.environ.get("LOGFIRE_ENVIRONMENT") or "development"
    ).strip() or "development"

    token = os.environ.get("LOGFIRE_TOKEN", "").strip() or None
    kwargs: Dict[str, Any] = {"send_to_logfire": "if-token-present"}
    if token is not None:
        kwargs["token"] = token
    logfire.configure(
        service_name=service,
        service_version=os.environ.get("LOGFIRE_SERVICE_VERSION") or None,
        environment=environment,
        data_dir=_data_dir(),
        console=_console_option(),
        inspect_arguments=False,  # no f-string magic; avoids introspection noise
        **kwargs,
    )

    # PydanticAI GenAI spans + token/latency metrics for every agent run.
    logfire.instrument_pydantic_ai(include_content=True)
    _init_metrics()
    _ENABLED = True
    return True


def flush(timeout_millis: int = 3000) -> Optional[bool]:
    """Flush pending telemetry (call before process exit).

    Returns True when the span flush succeeded, False on timeout, None when
    Logfire is disabled.
    """
    if not _ENABLED:
        return None
    try:
        return logfire.force_flush(timeout_millis=timeout_millis)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _init_metrics() -> None:
    _metrics["llm_calls"] = logfire.metric_counter(
        "medrag.llm.calls", unit="1",
        description="LLM calls through src.llm.run.ask_structured")
    _metrics["llm_latency"] = logfire.metric_histogram(
        "medrag.llm.latency.seconds", unit="s",
        description="Wall-clock latency of one ask_structured call")
    _metrics["llm_prompt_tokens"] = logfire.metric_histogram(
        "medrag.llm.prompt_tokens", unit="1",
        description="Estimated prompt tokens per ask_structured call")
    _metrics["llm_output_chars"] = logfire.metric_histogram(
        "medrag.llm.output_chars", unit="1",
        description="Response characters per ask_structured call")
    _metrics["retrieval_calls"] = logfire.metric_counter(
        "medrag.retrieval.calls", unit="1",
        description="Retrieval searches executed")
    _metrics["retrieved_chunks"] = logfire.metric_counter(
        "medrag.retrieval.chunks", unit="1",
        description="Chunk ranks reported by retrieval")
    _metrics["evidence_accepted"] = logfire.metric_counter(
        "medrag.evidence.accepted", unit="1",
        description="Critic-accepted evidence items")
    _metrics["evidence_rejected"] = logfire.metric_counter(
        "medrag.evidence.rejected", unit="1",
        description="Critic-rejected evidence items")
    _metrics["deep_inspections"] = logfire.metric_counter(
        "medrag.deep_inspections", unit="1",
        description="Deep paper inspections started")


def metric(name: str) -> Any:
    return _metrics.get(name)


def record_metric(name: str, value: float, **attrs: Any) -> None:
    """Record one data point on a registered metric (no-op if disabled)."""
    if not _ENABLED:
        return
    m = _metrics.get(name)
    if m is None:
        return
    try:
        attributes = {k: v for k, v in attrs.items() if v is not None} or None
        if hasattr(m, "add"):
            m.add(value, attributes=attributes)
        elif hasattr(m, "record"):
            m.record(value, attributes=attributes)
        else:
            m.set(value, attributes=attributes)
    except Exception:
        pass  # observability must never break the run


# ---------------------------------------------------------------------------
# Run / task context (attached to every span/log emitted while active)
# ---------------------------------------------------------------------------

def set_run_context(**kw: Any) -> None:
    ctx = dict(_run_ctx.get())
    ctx.update({k: v for k, v in kw.items() if v is not None})
    _run_ctx.set(ctx)


def clear_run_context() -> None:
    _run_ctx.set({})


@contextmanager
def task_context(task_id: str = "", task_title: str = "") -> Iterator[None]:
    """Scoped context for one worker task; restores the previous context."""
    token = _run_ctx.set({
        **_run_ctx.get(), "task_id": str(task_id), "task_title": str(task_title),
    })
    try:
        yield
    finally:
        _run_ctx.reset(token)


def _attrs(**kw: Any) -> Dict[str, Any]:
    out = {k: v for k, v in kw.items() if v is not None}
    for k, v in _run_ctx.get().items():
        if v is not None and k not in out:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------

@asynccontextmanager
async def run_span(query: str = "") -> Any:
    """Top-level span for one pipeline run; LLM GenAI spans nest under it."""
    if not _ENABLED:
        yield None
        return
    with logfire.span("medrag.run", _tags=("medrag", "run"),
                      query=(query or "")[:500], **_attrs()) as s:
        yield s


@contextmanager
def span(title: str, *, _tags: tuple = ("medrag",), **attrs: Any) -> Iterator[Any]:
    """An application-level span (deep inspection, stage, ...)."""
    if not _ENABLED:
        yield None
        return
    with logfire.span(title, _tags=_tags, **attrs) as s:
        yield s


@contextmanager
def llm_span(label: str, output_type: str = "", model: str = "") -> Iterator[Any]:
    """Span around one ask_structured call; GenAI model spans nest inside.

    The full prompt/response content is recorded by PydanticAI's own
    instrumentation; this span only adds OUR label + the model name for
    grouping/filtering (the child GenAI spans also carry gen_ai.request.model).
    """
    if not _ENABLED:
        yield None
        return
    base = str(label or "").split(":")[0]
    attrs: Dict[str, Any] = {"label": str(label), "agent": base,
                             "output_type": output_type}
    if model:
        attrs["model"] = model
    with logfire.span("llm.ask_structured", _tags=("medrag", "llm"), **attrs) as s:
        yield s


# ---------------------------------------------------------------------------
# Application trace events (mirrors src.trace.Trace)
# ---------------------------------------------------------------------------

def _cap(text: Any, limit: Optional[int] = None) -> str:
    s = text if isinstance(text, str) else str(text)
    if limit is None:
        if os.environ.get("TRACE_FULL_TEXTS", "").strip().lower() in ("1", "true", "yes", "on"):
            return s
        limit = _TEXT_LIMIT
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [+{len(s) - limit} chars truncated]"


def stage(title: str, **attrs: Any) -> None:
    if _ENABLED:
        logfire.info("◉ {title}", title=title, _tags=("medrag", "stage"), **attrs)


def bullet(msg: str, agent: str = "") -> None:
    if not _ENABLED:
        return
    agent = (agent or "").strip()
    if agent:
        # template form: msg may contain literal {} (dict reprs) that logfire's
        # str.format-style parser would otherwise complain about
        logfire.info("{agent}: {msg}", agent=agent.upper(),
                     msg=str(msg)[:2000], _tags=("medrag", agent.lower()))
    else:
        logfire.info("{msg}", msg=str(msg)[:2000], _tags=("medrag", "bullet"))


def agent(name: str, output_type: str = "", meta: Optional[dict] = None) -> None:
    if not _ENABLED:
        return
    logfire.info("agent.call", _tags=("medrag", "agent"),
                 agent=name, output_type=output_type,
                 meta=json.dumps(meta, default=str)[:1000] if meta else None)


def prompt(label: str, text: Any) -> None:
    if _ENABLED:
        logfire.info("llm.prompt", _tags=("medrag", "llm"),
                     label=str(label), prompt=_cap(text))


def response(label: str, text: Any) -> None:
    if _ENABLED:
        logfire.info("llm.response", _tags=("medrag", "llm"),
                     label=str(label), response=_cap(text))


def parsed(type_name: str, data: Any) -> None:
    if _ENABLED:
        logfire.info("llm.parsed", _tags=("medrag", "llm"), type=type_name,
                     data=_cap(json.dumps(data, default=str), 4000))


def log(event: str, **fields: Any) -> None:
    if not _ENABLED:
        return
    attrs = {k: _cap(v, 2000) for k, v in fields.items() if v is not None}
    logfire.info(f"event.{event}", _tags=("medrag", "event"), **attrs)


def final_answer(report: Any, run_id: str = "") -> None:
    """Log the FULL final synthesized answer report to Logfire.

    Accepts a Pydantic model (SynthesisReport) or a plain dict; the complete
    summary is rendered in the message and the structured report lands in
    attributes (sections, citations, gaps, limitations, confidence).
    """
    if not _ENABLED:
        return
    if hasattr(report, "model_dump"):
        data = report.model_dump(mode="json")
    elif isinstance(report, dict):
        data = report
    else:
        data = {}
    summary = str(data.get("summary") or "")
    sections = data.get("sections") or []
    citations = data.get("citations") or []
    if isinstance(citations, list):
        citations = [str(c) for c in citations][:50]

    attrs: Dict[str, Any] = {
        "run_id": run_id or None,
        "confidence": float(data.get("confidence") or 0.0),
        "sections": json.dumps(sections, default=str)[:20000] or None,
        "limitations": [str(x) for x in (data.get("limitations") or [])][:20] or None,
        "unresolved_gaps": [str(x) for x in (data.get("unresolved_gaps") or [])][:20] or None,
        "resolved_contradictions": [str(x) for x in
                                    (data.get("resolved_contradictions") or [])][:20] or None,
        "unresolved_contradictions": [str(x) for x in
                                      (data.get("unresolved_contradictions") or [])][:20] or None,
        "citations": citations or None,
    }
    attrs = {k: v for k, v in attrs.items() if v is not None and v != []}
    logfire.info("FINAL ANSWER: {summary}", summary=summary[:4000],
                 _tags=("medrag", "answer"), **attrs)


def tool(name: str, args: dict, result: Any = None) -> None:
    if not _ENABLED:
        return
    if result is None:
        logfire.info("tool.invoke", _tags=("medrag", "tool"), name=name,
                     args=json.dumps(args, default=str)[:2000])
    else:
        logfire.info("tool.result", _tags=("medrag", "tool"), name=name,
                     result=_cap(result, 4000))


def retrieved(method: str, query: str, ranks: list, texts: Any = None) -> None:
    if not _ENABLED:
        return
    top = []
    for i, (cid, score) in enumerate(list(ranks)[:30], 1):
        try:
            top.append(f"{i}. {cid} ({float(score):.4f})")
        except (TypeError, ValueError):
            top.append(f"{i}. {cid}")
    logfire.info("Retrieved {n} candidate(s) via {method}",
                 n=len(ranks), method=method, _tags=("medrag", "retrieval"),
                 query=str(query)[:200], top=" | ".join(top))
    record_metric("retrieval_calls", 1, method=method)
    record_metric("retrieved_chunks", len(ranks), method=method)


def chunk(doc: Any, rank: int = 0, label: str = "retrieved") -> None:
    """One retrieved chunk with the FULL untruncated text + metadata.

    Logfire is the observability sink: the complete chunk text is always sent
    (the TRACE_TEXT_CHARS cap only applies to the logs.txt mirror in
    src.trace, to keep that file small). Long papers are chunked by the
    corpus, so a single chunk text stays well within Logfire's size limits.
    """
    if not _ENABLED:
        return
    cid = getattr(doc, "chunk_id", None) or getattr(doc, "id", "")
    doc_id = getattr(doc, "document_id", "")
    node = getattr(doc, "node_type", "") or getattr(doc, "chunk_type", "")
    section = getattr(doc, "section", "")
    subsection = getattr(doc, "subsection", "")
    breadcrumb = " > ".join(str(b) for b in (getattr(doc, "breadcrumb", None) or []))
    table_id = getattr(doc, "table_id", None)
    figure_id = getattr(doc, "figure_id", None)
    score = getattr(doc, "rrf_score", None)
    methods = list(getattr(doc, "methods", None) or [])
    text = str(getattr(doc, "text", "") or "")

    attrs: Dict[str, Any] = {
        "rank": rank,
        "document_id": str(doc_id)[:200] or None,
        "node_type": str(node)[:100] or None,
        "subsection": str(subsection)[:200] or None,
        "breadcrumb": breadcrumb[:500] or None,
        "table_id": str(table_id) if table_id else None,
        "figure_id": str(figure_id) if figure_id else None,
        "methods": methods[:10] or None,
        "score": float(score) if score is not None else None,
        "text": text,  # FULL text, never truncated
    }
    attrs = {k: v for k, v in attrs.items() if v is not None}
    logfire.info("chunk.{label}: {chunk_id} ({section})",
                 label=label, chunk_id=str(cid)[:200],
                 section=str(section)[:300], _tags=("medrag", "retrieval"),
                 **attrs)


def record_llm(label: str, *, latency: float, prompt_tokens: int,
               output_chars: int = 0, success: bool = True,
               model: str = "") -> None:
    """Aggregate metrics for one ask_structured call (image-free, text counts)."""
    agent_name = str(label or "").split(":")[0]
    attrs = {"agent": agent_name, "label": str(label)}
    if model:
        attrs["model"] = model
    record_metric("llm_calls", 1, **attrs, success=str(success).lower())
    record_metric("llm_latency", latency, **attrs)
    record_metric("llm_prompt_tokens", prompt_tokens, **attrs)
    if output_chars:
        record_metric("llm_output_chars", output_chars, **attrs)


__all__ = [
    "agent", "bullet", "chunk", "clear_run_context", "configure_logfire",
    "credentials_present", "enabled", "ensure_configured", "final_answer",
    "flush", "llm_span", "log", "metric", "parsed", "prompt", "record_llm",
    "record_metric", "response", "retrieved", "run_span", "set_run_context",
    "span", "stage", "task_context", "tool",
]