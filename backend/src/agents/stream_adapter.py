"""NDJSON streaming adapter for the deep-research agent.

Bridge replacement: the legacy ``src.agents.bridge::stream_xdeep`` no
longer exists. This module wraps the real deep-agent run
(``Agent.run`` with an event handler — ``run_stream`` would stop the
graph at the first text output and silently drop the tool loop) and
re-emits its ``pydantic_ai`` stream events as UI NDJSON event dicts::

    {"type": "status", "run_id": ..., "state": ..., "message": ..., "ts": ...}
    {"type": "plan", "run_id": ..., "seq": N, "items": [...], "ts": ...}
    {"type": "task", "run_id": ..., "seq": N, "id": ..., "state": ..., "ts": ...}
    {"type": "thinking", "run_id": ..., "seq": N, "delta": ..., "done": false}
    {"type": "tool_call", "run_id": ..., "call_id": ..., "name": ..., "args": {...}, "ts": ...}
    {"type": "tool_result", "run_id": ..., "call_id": ..., "name": ..., "ok": ..., "duration_ms": ... | None, "ts": ...}
    {"type": "tool_progress", "run_id": ..., "seq": N, "call_id": ..., "name": ..., "stage": ..., "result": ..., "ts": ...}
    {"type": "verdict_table", "run_id": ..., "seq": N, "columns": [...], "rows": [...]}
    {"type": "answer", "run_id": ..., "seq": N, "delta": ..., "done": false}
    {"type": "answer_replace", "run_id": ..., "seq": N, "delta": ...}
    {"type": "error", "run_id": ..., "code": ..., "message": ..., "retryable": ...}
    {"type": "done", "run_id": ..., "usage": {...}, "citations": [...]}

Every event carries ``run_id``. ``seq`` increases per stream and is
carried by every event except ``status`` and ``done``. ``error`` is
terminal (no ``done`` follows it); a client disconnect surfaces as
``asyncio.CancelledError`` and likewise yields no ``done``.

The ``thinking`` stream carries only model reasoning (planner, leg and
judge ``ThinkingPart`` deltas, plus one ``[task] question`` header per
sub-agent leg so parallel legs stay distinguishable). Every
``thinking`` event carries an ``agent`` field — ``"orchestrator"`` for
the master agent, the task id (``T1``, …) for a delegated sub-agent —
so the UI can stream each agent's thoughts into its own section while
the main thought stream stays reserved for the orchestrator. Tool
internals, pool waits, verdict lines and fallback notes still print to
the terminal / logs but never enter ``thinking``, so tool calls cannot
interrupt the thought stream.

The plan phase runs first: a ``planning`` status, then one ``plan``
event carrying the task-list planner's items (``[{id, question,
deep_research, ...}]``) so the UI can render the task list before any
research streams. The planner import is lazy (no LLM env needed to
import this module); a planner failure yields a status note and the run
continues without a plan rather than failing the whole stream.

Per-chat run-state JSON (``src.runstate``) is the stateful layer.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from src.budget.budget import BudgetLimits, BudgetManager, BudgetState
from src.budget.config import budget_enabled, load_budget_config
from src.runstate.models import RunStatus, TaskState
from src.runstate.progress import progress_view, render_receipt
from src.runstate.store import get_store
from src.agents.clarify import normalize_questions
from src.lib import narrate

# Tool names that drive the verdict table (structured-output final answer).
_OUTPUT_TOOL_NAMES = frozenset({"final_result"})

# Tool-name prefix -> status state for the UI pipeline indicator.
_STATUS_BY_TOOL = (
    ("submit_plan", "planning"),
    ("spawn_subagent", "planning"),
    ("ask_user", "clarifying"),
    ("lookup_medical", "planning"),
    ("local_search", "retrieving"),
    ("web_search", "retrieving"),
    ("retrieve", "retrieving"),
    ("firecrawl", "retrieving"),
    ("synthesize", "synthesizing"),
    ("judge", "verifying"),
    ("verif", "verifying"),
    ("trust", "verifying"),
    ("check_gaps", "verifying"),
)

_MAX_RESULT_CHARS = 4000

# Evidence-tool results feed the UI tables — truncating them mid-JSON
# breaks parsing, so they get a cap that fits real payloads instead.
# Web search carries full scraped markdown (untruncated per page), so
# its cap is sized for ~10 full articles, not snippets.
_EVIDENCE_RESULT_TOOLS = frozenset({"local_search", "web_search"})
_EVIDENCE_RESULT_CAP = 1_048_576

# Text buffered per model turn before it may stream as answer. Short
# preambles ("I'll gather literature on both…") stay under it and get
# reclassified as thinking when the turn calls tools; a long synthesis
# trips it and keeps token-streaming live.
_ANSWER_LIVE_CHARS = 600


def _cap_for_tool(tool_name: str) -> int:
    if (tool_name or "").lower() in _EVIDENCE_RESULT_TOOLS:
        return _EVIDENCE_RESULT_CAP
    return _MAX_RESULT_CHARS


# Attachment cards only need a title/meta tooltip; keep each passage's text
# to a snippet so the progress frame stays small and parseable.
_PROGRESS_TEXT_CAP = 400


def _compact_retrieval_progress(output: Any) -> dict | None:
    """Local-corpus passages from a retrieval output, with clipped text.

    The parallel tool returns ``{query, local: [...], web: {...}}`` where the
    web leg carries whole scraped pages; shipping that hit the 1 MB cap and
    truncated mid-JSON. Returns None for non-retrieval shapes so the caller
    falls back to the generic summary.
    """
    try:
        data = json.loads(output) if isinstance(output, str) else output
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("local"), list):
        return None
    local: list[Any] = []
    for item in data["local"]:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        local.append({
            **item,
            "text": text[:_PROGRESS_TEXT_CAP] if isinstance(text, str) else "",
        })
    return {"query": data.get("query"), "local": local,
            "local_count": len(local)}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def status_for_tool(name: str) -> str | None:
    """Map a tool name to a pipeline status state (None = no transition)."""
    lower = (name or "").lower()
    for prefix, state in _STATUS_BY_TOOL:
        if lower.startswith(prefix):
            return state
    return None


def _coerce_args(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"_raw": args}
        except (ValueError, TypeError):
            return {"_raw": args} if args else {}
    return {"_raw": str(args)} if args is not None else {}


def _summarize_result(content: Any, max_chars: int = _MAX_RESULT_CHARS) -> tuple[Any, bool]:
    """Return (result_payload, truncated). Keeps NDJSON lines bounded."""
    if isinstance(content, dict):
        text = json.dumps(content)
        if len(text) > max_chars:
            return {"_truncated": text[:max_chars]}, True
        return content, False
    text = content if isinstance(content, str) else str(content)
    if len(text) <= max_chars:
        return text, False
    # Evidence payloads must stay parseable: a mid-JSON slice corrupts the
    # whole blob and the side sheet degrades to a raw dump. Shrink the
    # passages structurally (clip long texts, keep every row + the full
    # verdict block) instead of cutting the string.
    head, sep, tail = text.partition("[EVIDENCE JUDGMENT]")
    if sep:
        shrunk = _shrink_evidence_json(head, max_chars - len(sep) - len(tail))
        if shrunk is not None:
            return shrunk + sep + tail, True
    return text[:max_chars], True


# Per-text display budget inside a shrunk evidence payload. Rows and
# verdicts stay complete; expanded text carries an explicit marker.
_SHRUNK_TEXT_CAP = 4000
_SHRUNK_MARKER = " [clipped for display — full text in ledger]"


def _clip_evidence_texts(obj: Any, cap: int = _SHRUNK_TEXT_CAP) -> Any:
    if isinstance(obj, dict):
        return {
            key: (_clip_one_text(value, cap) if key in
                    ("text", "markdown", "snippet", "content")
                    else _clip_evidence_texts(value, cap))
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_clip_evidence_texts(value, cap) for value in obj]
    return obj


def _clip_one_text(value: Any, cap: int) -> Any:
    if isinstance(value, str) and len(value) > cap:
        return value[:cap] + _SHRUNK_MARKER
    return value


def _shrink_evidence_json(text: str, max_chars: int) -> str | None:
    """Shrink an evidence JSON blob to fit, or None when not possible."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, (dict, list)):
        return None
    out = json.dumps(_clip_evidence_texts(data), ensure_ascii=False)
    if len(out) > max_chars:
        return None
    return out


def _verdict_rows(parsed: Any) -> list | None:
    """Extract verdict rows from a parsed final_result payload, if present."""
    if not isinstance(parsed, dict):
        return None
    for key in ("rows", "verdicts", "claims"):
        rows = parsed.get(key)
        if isinstance(rows, list) and rows:
            return rows
    return None


def _tag_error(exc: BaseException, *, leg: str,
               tool: str | None = None) -> BaseException:
    """Stamp where an error happened so the UI can say so explicitly."""
    setattr(exc, "_medrag_leg", leg)
    if tool:
        setattr(exc, "_medrag_tool", tool)
    model = getattr(exc, "model_name", None) or os.environ.get("AGENT_MODEL")
    if model:
        setattr(exc, "_medrag_model", model)
    return exc


def _format_error(exc: BaseException, leg: str) -> str:
    """One explicit line: leg · model · tool · HTTP status · root cause."""
    name = type(exc).__name__
    detail = " ".join(str(exc).split())
    if len(detail) > 300:
        detail = detail[:300] + "…"
    model = getattr(exc, "model_name", None) or getattr(exc, "_medrag_model", None)
    tool = getattr(exc, "_medrag_tool", None)
    status = getattr(exc, "status_code", None)
    bits = [leg]
    if model:
        bits.append(f"model {model}")
    if tool:
        bits.append(f"tool {tool}")
    if status is not None:
        bits.append(f"HTTP {status}")
    bits.append(f"{name}: {detail}" if detail else name)
    return " · ".join(bits)


def _error_fields(exc: BaseException, leg: str) -> dict:
    """Structured error context for the `error` / `task` event."""
    return {
        "leg": leg,
        "tool": getattr(exc, "_medrag_tool", None),
        "model": getattr(exc, "model_name", None)
        or getattr(exc, "_medrag_model", None),
        "status_code": getattr(exc, "status_code", None),
    }


def _usage_tokens(result: Any) -> tuple[int, int]:
    """(input, output) tokens from an agent result — run stats only.

    Tolerant reader for pydantic-ai usage shapes; unreadable usage is
    (0, 0). Never feeds the budget ledger, which counts tool calls.
    """
    try:
        usage = getattr(result, "usage", None)
        if callable(usage):
            usage = usage()
        if usage is None:
            return (0, 0)
        request = getattr(usage, "input_tokens", None)
        if request is None:
            request = getattr(usage, "request_tokens", 0)
        response = getattr(usage, "output_tokens", None)
        if response is None:
            response = getattr(usage, "response_tokens", 0)
        return (int(request or 0), int(response or 0))
    except Exception:  # noqa: BLE001 — usage is best-effort
        return (0, 0)


def _resolve_task_allocation(requested: Any, defaults: BudgetLimits,
                             remaining: Any) -> tuple[BudgetLimits, dict]:
    """Resolve one sub-agent's explicit budget.

    Orchestrator-chosen values win; missing/invalid/non-positive fields
    fall back to task defaults; anything above the master budget's
    remaining is clamped to what is actually left (a promise can never
    exceed the pool it is drawn from). Returns (limits, meta) where
    meta has ``explicit`` (orchestrator-set fields) and ``clamped``.
    """
    asked = requested if isinstance(requested, dict) else {}
    try:
        number = int(asked.get("max_tool_calls"))
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        return BudgetLimits(max_tool_calls=defaults.max_tool_calls), {
            "explicit": [], "clamped": []}
    cap = getattr(remaining, "remaining_tool_calls", None) if remaining else None
    if cap is not None and number > cap:
        return BudgetLimits(max_tool_calls=int(cap)), {
            "explicit": ["max_tool_calls"], "clamped": ["max_tool_calls"]}
    return BudgetLimits(max_tool_calls=number), {
        "explicit": ["max_tool_calls"], "clamped": []}


def _task_coverage_complete(ledger: Any, run_id: str, task_id: str) -> bool:
    """True when every requirement of the task has verified evidence.

    A budget-exhausted leg that covered everything delivered its work —
    failure means the model found nothing relevant at all (or left holes),
    never a spent budget on top of complete coverage. Tasks with no
    requirements can never prove coverage, so they never count as complete.
    Never raises: unreadable ledgers read as incomplete.
    """
    try:
        turn = ledger.turns.get(run_id) if ledger is not None else None
        leg_task = turn.task(task_id) if turn is not None else None
        reqs = (leg_task.evidence_requirements
                if leg_task is not None else [])
        return bool(reqs) and all(
            getattr(req, "supporting_evidence", None) for req in reqs)
    except Exception:  # noqa: BLE001 — coverage check never breaks a run
        return False


def _task_sources(task: Any) -> list[dict]:
    """Verified passages grouped by document for the Sources card.

    Each entry carries the document id, PMC id (local corpus),
    URL (web), section, and every verified passage id from it — the
    frontend resolves in-text ``[pid]`` markers against these.
    """
    import re
    grouped: dict[str, dict] = {}
    order: list[str] = []
    reqs = getattr(task, "evidence_requirements", None) or []
    for req in reqs:
        for ev in getattr(req, "supporting_evidence", None) or []:
            url = (getattr(ev, "url", "") or "").strip()
            doc = (getattr(ev, "document_id", "") or "").strip()
            key = url or doc or (getattr(ev, "passage_id", "") or "")
            if not key:
                continue
            entry = grouped.get(key)
            if entry is None:
                pmc = re.search(r"PMC\d+", f"{doc} {url}")
                entry = {
                    "id": doc or url,
                    "pmcid": pmc.group(0) if pmc else None,
                    "url": url or None,
                    "title": (getattr(ev, "title", "") or "") or None,
                    "section": (getattr(ev, "section", "") or "") or None,
                    "passage_ids": [],
                }
                grouped[key] = entry
                order.append(key)
            pid = getattr(ev, "passage_id", "") or ""
            if pid and pid not in entry["passage_ids"]:
                entry["passage_ids"].append(pid)
            # Short citation ref (P1, ...) — what the answer actually cites.
            ref = getattr(ev, "ref", "") or ""
            if ref and ref not in entry["passage_ids"]:
                entry["passage_ids"].append(ref)
    return [grouped[key] for key in order]


def _requirements_block(task: Any) -> str:
    """Render a task's ledger requirements as prompt context.

    The leg never plans for itself: this block is the only requirements
    it gets, delivered code-side so it cannot be skipped or reinvented.
    """
    reqs = getattr(task, "evidence_requirements", None) or []
    lines = []
    for req in reqs:
        rid = getattr(req, "id", "?")
        desc = getattr(req, "description", "")
        lines.append(f"{rid}: {desc}".strip())
    if not lines:
        return ""
    return ("Evidence requirements for this task (pre-planned — use "
            "these verbatim, do not invent your own):\n"
            + "\n".join(f"- {line}" for line in lines))


async def _run_planner_leg(task_text: str) -> tuple[str, list]:
    """Run the planner as a spawned subagent leg (never a tool).

    Pinned sampling comes from ``planner.build_agent`` (temp 0, seed 43,
    top_p 0.1). Returns (plan JSON text, agent results) — the caller
    folds the results' usage into the leg's own explicit allocation.
    The orchestrator publishes the JSON via ``submit_plan`` before
    spawning research.
    """
    from src.agents import planner as planner_mod
    used: list = []
    agent = planner_mod.build_agent()
    plan = await planner_mod.generate_plan(task_text, agent=agent,
                                           usage_sink=used)
    return plan.model_dump_json(), used


async def stream_deep_agent(
    question: str,
    ctx: dict | None = None,
    *,
    run_fn: Callable[[str, Callable[..., Awaitable[None]]], Awaitable[Any]] | None = None,
    umls: Any | None = None,
    warmup: bool = True,
    timeout_s: float | None = None,
    orchestrator: Any | None = None,
    run_id: str | None = None,
    conversation_id: str | None = None,
) -> AsyncIterator[dict]:
    """Stream one deep-agent run as NDJSON event dicts.

    ``run_fn`` (tests / embedding callers) is ``await run_fn(question,
    emit)`` where ``emit(type, **fields)`` stamps ``run_id``/``seq``.
    When ``run_fn`` is None the master orchestrator drives execution: it
    publishes a plan (``plan`` event), spawns sub deep/shallow agents per
    task (sub-agent tool activity streams with namespaced call ids,
    ``task`` started/done events mark progress), and authors the final
    answer, which streams as ``answer`` deltas. ``orchestrator`` overrides
    the master agent (tests); when None the real orchestrator runs via
    ``Agent.run`` with an event handler that translates ``pydantic_ai``
    stream events into the same ``emit`` calls. ``run_id`` names this
    turn. The per-chat state JSON lives at ``chats/{chat}.json`` where
    ``chat`` is ``conversation_id`` (the API server passes the chat id)
    or the run id for one-off runs; every request appends a turn.
    """
    ctx = dict(ctx or {})
    run_id = run_id or _new_run_id()
    chat_id = (conversation_id or "").strip() or run_id
    seq = 0
    seen_thinking = False
    seen_answer = False
    current_state = "started"

    def _status(state: str, message: str) -> dict:
        return {"type": "status", "run_id": run_id, "state": state,
                "message": message, "ts": _now_iso()}

    _queue: asyncio.Queue = asyncio.Queue()
    _loop = asyncio.get_running_loop()
    _seq_lock = threading.Lock()

    def _emit_nowait(etype: str, **fields: Any) -> None:
        """Queue an event from any thread (warmup narration is off-loop)."""
        nonlocal seq
        event: dict[str, Any] = {"type": etype, "run_id": run_id}
        if etype not in ("status", "done"):
            with _seq_lock:
                seq += 1
                event["seq"] = seq
        event.update(fields)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if running is _loop:
                _queue.put_nowait(event)
            else:
                _loop.call_soon_threadsafe(_queue.put_nowait, event)
        except RuntimeError:  # noqa: BLE001 — loop closed; run is over
            pass

    async def emit(etype: str, **fields: Any) -> None:
        _emit_nowait(etype, **fields)

    def _account(event: dict) -> None:
        """Fold a queued event into the run-level flags both drains share."""
        nonlocal seen_thinking, seen_answer, current_state
        if event.get("type") == "thinking":
            seen_thinking = True
        if event.get("type") in ("answer", "answer_replace"):
            seen_answer = True
        if event.get("type") == "status":
            current_state = event.get("state", current_state)

    async def _drain() -> AsyncIterator[dict]:
        """Yield everything queued so far, in order (narration first)."""
        while True:
            try:
                event = _queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            _account(event)
            yield event

    # Deliberately no narrate sink: narration (tool internals, pool
    # waits, judge verdicts…) stays on the terminal / logs only. The UI
    # thought stream carries just what the model is thinking, so tool
    # calls can never interrupt it.

    # Raw tool_call_id -> scoped call id, so judge-middleware progress
    # (which only knows the raw id) can be attributed to the right step.
    _raw_to_scoped: dict[str, str] = {}

    def _on_retrieval_progress(tool_call_id: str, tool_name: str,
                               output: str) -> None:
        """Forward passages-found progress as a partial tool result.

        Everything the UI needs for the attachment cards is the local-corpus
        passages; the combined parallel payload also carries whole scraped web
        pages, which blew past the 1 MB evidence cap and got truncated
        mid-JSON (unparseable on the client). Send a compact local-only shape
        with clipped passage text instead.
        """
        scoped = _raw_to_scoped.get(tool_call_id or "")
        if scoped is None:
            return
        payload = _compact_retrieval_progress(output)
        if payload is None:
            payload, _truncated = _summarize_result(
                output, _cap_for_tool(tool_name))
        _emit_nowait("tool_progress", call_id=scoped, name=tool_name,
                     stage="retrieved", result=payload, ts=_now_iso())

    from src.middleware.llm_as_a_judge import install_retrieval_progress_sink
    # Task-scoped like the old narrate binding was: dies with the task,
    # next run overwrites it.
    install_retrieval_progress_sink(_on_retrieval_progress)

    def _on_subagent_thinking(tool_call_id: str, delta: str) -> None:
        """Forward a delegated model's thinking into the thought stream.

        Attributed to the owning leg via the raw->scoped call-id map:
        middleware (the judge) only knows the raw tool id, but the
        scoped id carries the ``T1:…`` namespace, so judge thoughts land
        in the right agent's section.
        """
        if not delta:
            return
        scoped = _raw_to_scoped.get(tool_call_id or "")
        agent = scoped.split(":", 1)[0] if scoped and ":" in scoped else "orchestrator"
        # `origin="judge"` lets the UI drop the judge's reasoning: it is
        # internal verification, not part of the agent's own narration.
        _emit_nowait("thinking", agent=agent, delta=delta, done=False,
                     origin="judge")

    from src.middleware.llm_as_a_judge import install_judge_thinking_sink
    install_judge_thinking_sink(_on_subagent_thinking)

    if not isinstance(question, str) or not question.strip():
        yield {"type": "error", "run_id": run_id, "code": "INVALID_REQUEST",
               "message": "question must be a non-empty string",
               "retryable": False}
        return

    yield _status("started", "run started")

    # Plan phase first: the orchestrator publishes its plan of action as
    # its first move, so the UI renders the plan card as soon as it
    # arrives mid-stream.
    yield _status("planning", "making a plan…")
    current_state = "planning"

    # Anything queued during startup comes out now, in order, before
    # any research streams.
    async for _early in _drain():
        yield _early

    _call_started: dict[str, float] = {}

    async def _emit_tool_call(call_id: str, name: str, args: Any) -> None:
        nonlocal current_state
        state = status_for_tool(name)
        if state and state != current_state:
            current_state = state
            _queue.put_nowait(_status(state, f"{name} running"))
        _call_started[call_id] = time.perf_counter()
        await emit("tool_call", call_id=call_id, name=name,
                   args=_coerce_args(args), ts=_now_iso())

    async def _real_run() -> dict:
        # Lazy imports: this module must import without LLM env vars set
        # (tests, wire-compat probes) — only the real run needs them.
        from src.agents import deep_agent as deep_agent_mod
        from src.agents import orchestrator as orchestrator_mod
        from src.agents.orchestrator import (
            MAX_SUBAGENTS, build_orchestrator, normalize_plan_items,
        )
        from src.tools.umls import build_umls_client
        from src.lib.streaming import tool_call_line, tool_result_line, verdict_line
        from pydantic_ai.messages import (
            FunctionToolCallEvent, FunctionToolResultEvent,
            OutputToolCallEvent, PartDeltaEvent, PartStartEvent,
            TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta,
            ToolCallPart, ToolCallPartDelta,
        )

        from src.tools.umls import DeepDeps

        _runner: Any = orchestrator
        _deps: Any = None
        _deep_runner: Any = None
        _shallow_runner: Any = None
        _spawns_used = 0
        # The synthesizer streams its answer deltas; set once the tool
        # runs so the end-of-stream fallback never re-emits the whole
        # text as one chunk (the flash) — or echoes it from _drive.
        synth_streamed = False

        def _ensure_orchestrator() -> tuple[Any, Any]:
            """Build the real orchestrator on first use (never when a fake
            covers the run — keeps tests LLM-env-free)."""
            nonlocal _runner, _deps
            if _runner is None:
                _runner = build_orchestrator(
                    spawn_impl=_spawn, synthesize_impl=_synthesize_impl)
            if _deps is None:
                _deps = DeepDeps(umls=umls or build_umls_client(),
                                 budget=_orch_budget, chat_id=chat_id)
            return _runner, _deps

        if warmup and orchestrator is None:
            deep_agent_mod.kick_retriever_warmup("stream")
        usage: dict[str, int] = {"prompt": 0, "completion": 0}

        # The stateful chat ledger: this turn's plan, tasks, verified
        # evidence, gaps, and structured findings accumulate in the
        # chat's JSON. Tools resolve it through deps.chat_id; the
        # orchestrator steers from compact progress views, not by
        # re-reading full outputs.
        _store = get_store()
        await _store.create_turn(chat_id, run_id, question)

        # The run-global budget: the ultimate resource ceiling everything
        # draws from (None = enforcement off, legacy path). The
        # orchestrator gets its OWN master budget as a child of the run:
        # its tool calls and every task allocation spend from it, so past
        # it no more spawns and no more tool calls are possible.
        _budget_cfg = load_budget_config() if budget_enabled() else None
        _run_budget: BudgetManager | None = None
        _orch_budget: BudgetManager | None = None
        if _budget_cfg is not None:
            _run_budget = BudgetManager(
                _budget_cfg.global_limits,
                task_id="run",
                low_threshold=_budget_cfg.low_threshold,
                critical_threshold=_budget_cfg.critical_threshold,
            )
            _orch_budget = _run_budget.spawn_child(
                "orchestrator", _budget_cfg.orch_limits)

        async def _record_usage(result: Any) -> None:
            # Run-stat token usage only (returned to the UI); the budget
            # ledger counts tool calls, never tokens.
            request, response = _usage_tokens(result)
            usage["prompt"] += request
            usage["completion"] += response

        def _make_handler(tag: str, *, stream_text: bool,
                          state: dict | None = None):
            """Translate one agent leg's pydantic-ai events into UI emits.

            ``tag`` namespaces tool call ids (one namespace per plan-task
            sub-agent) and attributes that leg's thinking deltas
            (``agent=tag``, or ``"orchestrator"`` when untagged), so each
            agent's reasoning streams into its own UI section. Thinking
            deltas always stream — every leg's model reasoning is visible
            from start to finish, uninterrupted by tool calls.

            Text is routed per model turn (one handler invocation is one
            turn): text buffered during a turn that calls tools is
            narration, not the answer, so it flushes as thinking (a
            preamble like "I'll gather literature on both…" must never
            print as the response). Text from a tool-free turn flushes
            as answer when ``stream_text`` is set; a long tool-free text
            additionally live-streams past ``_ANSWER_LIVE_CHARS`` so the
            final synthesis keeps token-streaming. With ``stream_text``
            unset, tool-free text stays swallowed, so sub-task draft
            findings stay out of both the thought stream and the final
            answer (the full reply is still captured for synthesis).
            """
            tool_names: dict[int, str] = {}
            call_names: dict[str, str] = {}
            last: dict[str, str] = {"value": "", "raw": "", "call_id": ""}

            def _scoped(raw: str) -> str:
                return f"{tag}:{raw}" if tag else raw

            async def _handler(ctx_run: Any, events: Any) -> None:
                nonlocal seen_thinking, seen_answer
                # One handler invocation is one model turn: buffer its
                # text until the turn's shape is known (tool calls or
                # clean), then flush as thinking or answer.
                turn_text: list[str] = []
                turn_len = 0
                turn_has_tools = False

                async def _flush_text() -> None:
                    """Emit the buffered turn text (and clear it).

                    With tools in the turn the text is narration →
                    thinking; without tools it is candidate answer text
                    (also used mid-turn, so a long tool-free synthesis
                    keeps token-streaming live).
                    """
                    nonlocal turn_text, turn_len, seen_thinking, seen_answer
                    if not turn_text:
                        return
                    chunk = "".join(turn_text)
                    turn_text = []
                    turn_len = 0
                    if turn_has_tools:
                        seen_thinking = True
                        await emit("thinking",
                                   agent=tag or "orchestrator",
                                   delta=chunk, done=False)
                    elif stream_text:
                        seen_answer = True
                        if state is not None:
                            # Accumulate chunks; joining per delta was
                            # O(n^2) copying for a long answer.
                            state.setdefault("answer_parts", []).append(chunk)
                        await emit("answer",
                                   delta=chunk, done=False)
                    # Otherwise swallowed: sub-task draft text from a
                    # tool-free turn is neither model thinking nor the
                    # final answer.

                async for event in events:
                    if isinstance(event, FunctionToolCallEvent):
                        part = event.part
                        raw = (getattr(part, "tool_call_id", None)
                               or f"tc_{id(part):x}")
                        call_id = _scoped(raw)
                        last["value"] = part.tool_name
                        last["raw"] = raw
                        last["call_id"] = call_id
                        call_names[call_id] = part.tool_name
                        _raw_to_scoped[raw] = call_id
                        if state is not None:
                            state["tool"] = part.tool_name
                        await _emit_tool_call(call_id, part.tool_name,
                                              getattr(part, "args", {}))
                        if part.tool_name == "submit_plan":
                            # The orchestrator's plan of action doubles as
                            # the UI plan card (single emission point: the
                            # tool itself only acks back to the model).
                            # NOTE: streamed args arrive as a JSON string.
                            plan_items = normalize_plan_items(
                                _coerce_args(getattr(part, "args", {})))
                            if plan_items:
                                await emit("plan", items=plan_items,
                                           ts=_now_iso())
                                # …and as the ledger's plan: the fill-
                                # the-JSON contract starts here.
                                await _store.record_plan(chat_id, plan_items)
                        if part.tool_name == "ask_user":
                            # Human-in-the-loop: surface the clarification
                            # questions as their own event (the tool itself
                            # parks the run awaiting the user's answer).
                            questions = normalize_questions(
                                _coerce_args(getattr(part, "args", {})))
                            if questions:
                                await emit(
                                    "question", call_id=call_id,
                                    questions=[q.to_event() for q in questions],
                                    ts=_now_iso())
                        # Same line the terminal prints (shared builder).
                        narrate.say(tool_call_line(
                            tag or "deep", part.tool_name,
                            getattr(part, "args", {})))
                    elif isinstance(event, FunctionToolResultEvent):
                        content = getattr(event, "part", event)
                        content = getattr(content, "content", content)
                        if hasattr(event, "result"):
                            content = getattr(event.result, "content", content)
                        # tool_call_id comes from part.tool_call_id; fall back
                        # to the most recent call so the frontend can still
                        # correlate result -> call for the side sheet.
                        result_raw = (
                            getattr(event, "tool_call_id", None)
                            or last.get("raw")
                            or f"tc_{seq + 1}"
                        )
                        _raw_to_scoped.pop(result_raw, None)
                        result_call_id = _scoped(result_raw)
                        tool_name = call_names.get(result_call_id,
                                                   last["value"])
                        payload, truncated = _summarize_result(
                            content, _cap_for_tool(tool_name))
                        ok = not (isinstance(content, dict)
                                  and content.get("error") is not None)
                        started = _call_started.pop(result_call_id, None)
                        duration_ms = (
                            round((time.perf_counter() - started) * 1000)
                            if started is not None else None
                        )
                        await emit("tool_result",
                                   call_id=result_call_id,
                                   name=tool_name,
                                   ok=ok, result=payload, truncated=truncated,
                                   duration_ms=duration_ms, ts=_now_iso())
                        narrate.say(tool_result_line(tag or "deep", content))
                    elif isinstance(event, OutputToolCallEvent):
                        raw = getattr(event.part, "args", "")
                        # Same line the terminal prints, for every verdict
                        # call (even when no table rows parse out of it).
                        narrate.say(verdict_line(tag or "deep", raw))
                        try:
                            parsed = (json.loads(raw)
                                      if isinstance(raw, str) else raw)
                        except ValueError:
                            parsed = None
                        rows = _verdict_rows(parsed)
                        if rows is not None:
                            cols = (list(parsed.keys())
                                    if isinstance(parsed, dict) else [])
                            await emit("verdict_table", columns=cols or
                                       ["claim", "verdict", "evidence"],
                                       rows=rows, format="data-not-html")
                    elif (isinstance(event, PartStartEvent)
                          and isinstance(event.part, TextPart)):
                        # Whole text can arrive in the start event (no
                        # deltas follow): buffer it exactly like a delta.
                        if event.part.content:
                            turn_text.append(event.part.content)
                            turn_len += len(event.part.content)
                            if (stream_text and not turn_has_tools
                                    and turn_len > _ANSWER_LIVE_CHARS):
                                await _flush_text()
                    elif (isinstance(event, PartStartEvent)
                          and isinstance(event.part, ToolCallPart)):
                        if event.part.tool_name:
                            turn_has_tools = True
                            tool_names[event.index] = event.part.tool_name
                    elif (isinstance(event, PartDeltaEvent)
                          and isinstance(event.delta, ToolCallPartDelta)):
                        if event.delta.tool_name_delta:
                            turn_has_tools = True
                            tool_names[event.index] = (
                                tool_names.get(event.index, "")
                                + event.delta.tool_name_delta)
                    elif (isinstance(event, PartStartEvent)
                          and isinstance(event.part, ThinkingPart)):
                        seen_thinking = True
                    elif (isinstance(event, PartDeltaEvent)
                          and isinstance(event.delta, ThinkingPartDelta)):
                        if event.delta.content_delta:
                            seen_thinking = True
                            await emit("thinking",
                                       agent=tag or "orchestrator",
                                       delta=event.delta.content_delta,
                                       done=False)
                    elif (isinstance(event, PartDeltaEvent)
                          and isinstance(event.delta, TextPartDelta)):
                        if not event.delta.content_delta:
                            continue
                        turn_text.append(event.delta.content_delta)
                        turn_len += len(event.delta.content_delta)
                        # Long tool-free text is the synthesis streaming:
                        # flush live so it keeps token-streaming instead
                        # of arriving as one chunk at turn end. Short
                        # preambles stay buffered and get reclassified
                        # below once the turn's shape is known.
                        if (stream_text and not turn_has_tools
                                and turn_len > _ANSWER_LIVE_CHARS):
                            await _flush_text()
                # Turn stream exhausted: the turn's shape is now known.
                await _flush_text()

            return _handler

        async def _spawn(*, task: str, depth: str = "deep",
                         context: str = "", task_id: str = "",
                         deps: Any = None,
                         budget: dict | None = None) -> str:
            """Run one delegated sub-agent leg (the spawn_subagent tool).

            ``budget`` is the orchestrator's explicit per-sub-agent
            allocation, resolved against task defaults and clamped to the
            master budget's remainder. ``depth="plan"`` runs the planner
            subagent (pinned sampling, plan JSON back for submit_plan);
            otherwise a research leg runs and a compact receipt comes
            back — that return value is the correspondence channel: the
            orchestrator reads it and decides what to do next. One sick
            sub-agent never kills the run; the failure is reported back
            as findings text.
            """
            nonlocal _spawns_used, _deep_runner, _shallow_runner
            if _spawns_used >= MAX_SUBAGENTS:
                return (f"Spawn budget exhausted ({MAX_SUBAGENTS} max). "
                        f"Do not spawn more sub-agents; finish with the "
                        f"evidence gathered so far.")
            _spawns_used += 1
            sid = (task_id or "").strip() or f"S{_spawns_used}"
            depth_name = str(depth or "").strip().lower()
            sane_depth = ("shallow" if depth_name == "shallow"
                          else "plan" if depth_name == "plan"
                          else "deep")
            is_planner = sane_depth == "plan"
            task_text = str(task or "").strip()
            await emit("task", id=sid, state="started",
                       question=task_text[:160],
                       model=os.environ.get("AGENT_MODEL"),
                       ts=_now_iso())
            # The planner leg IS planning, so the indicator stays put;
            # research legs move it. Every leg heads the thought stream
            # so parallel sub-agents stay distinguishable.
            if is_planner:
                await emit("status", state="planning",
                           message=f"Planner sub-agent {sid} planning…",
                           ts=_now_iso())
            else:
                await emit("status", state="retrieving",
                           message=f"Sub-agent {sid} researching…",
                           ts=_now_iso())
            await emit("thinking", agent=sid,
                       delta=f"[{sid}] {task_text}\n\n",
                       done=False)
            leg = f"sub-agent {sid}"
            state: dict = {}
            # Explicit per-sub-agent budget, assigned by the orchestrator
            # and carved out of ITS master budget (deps.budget is the
            # orchestrator's own allocation). The child can never spend
            # more than min(child, parent) — the parent ceiling always
            # wins, even for parallel legs.
            child_budget: BudgetManager | None = None
            parent_budget = getattr(deps, "budget", None) or _run_budget
            alloc_meta: dict = {"explicit": [], "clamped": []}
            if parent_budget is not None and _budget_cfg is not None:
                parent_remaining = None
                try:
                    parent_remaining = await parent_budget.snapshot()
                except Exception:  # noqa: BLE001 — clamp is best-effort
                    parent_remaining = None
                task_limits, alloc_meta = _resolve_task_allocation(
                    budget, _budget_cfg.task_limits, parent_remaining)
                child_budget = parent_budget.spawn_child(sid, task_limits)
            # Fresh task-scoped deps: the ledger attributes this leg's
            # evidence to sid, and the shared orchestrator deps are never
            # mutated (a reused object would leak one task's id into the
            # next leg's writes). The notes/kept-passages pools stay
            # shared run-wide, preserving the legacy gap-checker view of
            # the full accepted set.
            umls_client = getattr(deps, "umls", None) or umls
            if umls_client is None:
                umls_client = build_umls_client()
            sub_deps = DeepDeps(
                umls=umls_client, budget=child_budget,
                chat_id=chat_id, task_id=sid,
                notes=getattr(deps, "notes", None) or [],
                kept_passages=getattr(deps, "kept_passages", None) or [])
            await _store.record_task_started(
                chat_id, sid, task_text, sane_depth)
            reply = ""
            try:
                if is_planner:
                    # The planner subagent: pinned sampling, its own
                    # allocation, plan JSON back for submit_plan.
                    reply, used = await _run_planner_leg(task_text)
                    for result in used:
                        await _record_usage(result)
                else:
                    prompt = deep_agent_mod.render_task_prompt(task_text)
                    if (context or "").strip():
                        prompt += ("\n\nContext from orchestrator:\n"
                                   f"{context.strip()}")
                    # Pre-planned requirements, injected code-side: the leg
                    # works from the planner's contract, never its own.
                    ledger = _store.get_chat(chat_id)
                    turn = (ledger.turns.get(run_id)
                            if ledger is not None else None)
                    leg_task = (turn.task(sid)
                                if turn is not None else None)
                    req_block = (_requirements_block(leg_task)
                                 if leg_task is not None else "")
                    if req_block:
                        prompt += f"\n\n{req_block}"
                    handler = _make_handler(tag=sid, stream_text=False,
                                            state=state)
                    # NOTE: run() here, not run_stream(): run_stream treats
                    # the first text output as final and silently drops the
                    # rest of the tool loop (tools still execute
                    # server-side, but no tool events reach the handler and
                    # the graph never continues). run() drives the graph to
                    # completion while the handler keeps streaming every
                    # event live.
                    if sane_depth == "shallow":
                        if _shallow_runner is None:
                            _shallow_runner = orchestrator_mod.build_shallow_agent()
                        shallow_result = await _shallow_runner.run(
                            prompt, event_stream_handler=handler)
                        reply = shallow_result.output
                    else:
                        if _deep_runner is None:
                            _deep_runner = deep_agent_mod.build_deep_agent()
                        deep_result = await _deep_runner.run(
                            prompt, deps=sub_deps,
                            event_stream_handler=handler)
                        reply = deep_result.output
                        await _record_usage(deep_result)
            except Exception as exc:  # noqa: BLE001 — see docstring
                _tag_error(exc, leg=leg, tool=state.get("tool"))
                if child_budget is not None and parent_budget is not None:
                    await parent_budget.reclaim_child(child_budget)
                await _store.record_task_finished(chat_id, sid, TaskState.FAILED)
                await emit("task", id=sid, state="failed",
                           error=_format_error(exc, leg),
                           **_error_fields(exc, leg),
                           ts=_now_iso())
                if is_planner:
                    return (f"Planner sub-agent {sid} failed "
                            f"({str(exc)[:200]}): re-spawn the planner "
                            f"sub-agent before any research.")
                return (f"Sub-agent {sid} failed "
                        f"({str(exc)[:200]}): continue with the evidence "
                        f"gathered so far.")
            # A budget-exhausted task is NOT a failure when it delivered:
            # coverage complete (every requirement has verified evidence)
            # closes DONE — failure means the model found nothing relevant
            # at all, or left holes. The budget ledger still records the
            # exhaustion, so no information is lost.
            budget_exhausted = False
            budget_ledger = None
            if child_budget is not None:
                snapshot = await child_budget.snapshot()
                budget_exhausted = snapshot.state == BudgetState.EXHAUSTED
                budget_ledger = await child_budget.ledger_snapshot()
                if parent_budget is not None:
                    await parent_budget.reclaim_child(child_budget)
            # The leg's verified evidence is already in the ledger (the
            # judge wrote it per tool call); here the task just closes
            # with its state and its budget ledger.
            coverage_complete = (
                budget_exhausted
                and _task_coverage_complete(_store.get_chat(chat_id),
                                            run_id, sid))
            task_state = (TaskState.DONE if not budget_exhausted
                          or coverage_complete
                          else TaskState.BUDGET_EXHAUSTED)
            # The leg's own closing text (its completion note / quick answer).
            # UI-only: the orchestrator still receives the compact receipt, but
            # the agent sheet can now show what this sub-agent reported back.
            # Tagged origin="response" so the frontend separates it from the
            # streamed narration; skipped for the planner leg, whose JSON reply
            # is already published as the plan.
            if not is_planner and str(reply).strip():
                await emit("thinking", agent=sid, delta=str(reply).strip(),
                           done=True, origin="response", ts=_now_iso())
            await emit("task", id=sid, state="done",
                       budget_exhausted=budget_exhausted, ts=_now_iso())
            # Research legs return arrays-only: coverage, gaps, budget
            # remainder. Evidence texts stay in the ledger (pulled later
            # via run_progress(task_id) for synthesis) — never dumped
            # into orchestrator context here. The planner leg instead
            # returns its full plan JSON: the orchestrator must publish
            # it via submit_plan before spawning research.
            notes: list[str] = []
            if is_planner:
                notes.append(
                    "[PLANNER: publish this plan with submit_plan before "
                    "spawning any research sub-agent.]")
            if budget_exhausted and coverage_complete:
                notes.append(
                    "[budget exhausted after all requirements were covered "
                    "— evidence is complete, synthesize normally.]")
            elif budget_exhausted:
                notes.append(
                    "[TASK BUDGET EXHAUSTED: research stopped early; "
                    "synthesize from stored evidence and list unresolved gaps.]")
            await _store.record_task_finished(
                chat_id, sid, task_state,
                budget_allocated=(budget_ledger or {}).get("allocated"),
                budget_expenditure_history=(budget_ledger or {}).get("expenditure"),
                budget_remaining=(budget_ledger or {}).get("remaining"))
            ledger = _store.get_chat(chat_id)
            receipt = render_receipt(ledger, run_id, sid) if ledger is not None else ""
            # Feed the Sources card: verified passages grouped by
            # document (no texts — the card resolves [pid] markers).
            if ledger is not None and not is_planner:
                turn = ledger.turns.get(run_id)
                leg_task = turn.task(sid) if turn is not None else None
                if leg_task is not None:
                    leg_sources = _task_sources(leg_task)
                    if leg_sources:
                        await emit("sources", sources=leg_sources)
            # The explicit allocation, stated back: what this sub-agent
            # was given, who set it, and what was clamped to the master
            # budget's remainder. Compact — one line, no evidence text.
            alloc_line = ""
            if child_budget is not None:
                alloc = (budget_ledger or {}).get("allocated") or {}
                tools_n = alloc.get("max_tool_calls", "?")
                source = ("orchestrator-set" if alloc_meta.get("explicit")
                          else "default")
                clamped = alloc_meta.get("clamped") or []
                clamp_note = (f" (clamped to master-budget remainder: "
                              f"{', '.join(clamped)})" if clamped else "")
                alloc_line = (
                    f"[BUDGET] {sid} allocated {tools_n} tool calls "
                    f"({source}){clamp_note}.")
            head = reply if is_planner else receipt
            parts = [p for p in (head, alloc_line, *notes) if p]
            return "\n".join(parts)

        def _ledger_refs() -> list[dict]:
            """The turn's verified evidence refs (ref/title/url/document id),
            sorted by ref number — the citation-guarantee pool."""
            try:
                ledger = _store.get_chat(chat_id)
                turn = (ledger.turns.get(run_id)
                        if ledger is not None else None)
                if turn is None:
                    return []
                out: list[dict] = []
                for task in turn.plan:
                    for req in task.evidence_requirements:
                        for ev in req.supporting_evidence:
                            ref = getattr(ev, "ref", "") or ""
                            if not ref:
                                continue
                            out.append({
                                "ref": ref,
                                "title": getattr(ev, "title", "") or "",
                                "url": getattr(ev, "url", "") or "",
                                "document_id": getattr(ev, "document_id", "") or "",
                            })
                def _num(ref: str) -> int:
                    digits = ref[1:] if ref[:1].lower() == "p" else ""
                    return int(digits) if digits.isdigit() else 0

                out.sort(key=lambda r: _num(r["ref"]))
                return out
            except Exception:  # noqa: BLE001 — best-effort, never breaks
                return []

        async def _synthesize_impl(question: str, chat_id: str) -> str:
            """Run the synthesizer as a visible, live-streamed leg.

            The leg drives the shared handler (tag ``"synthesizer"``) so
            its thinking streams into its own UI section and its answer
            text streams as live ``answer`` deltas — the same treatment
            every research leg gets. The question and every proof passage
            ride in one message; the model never fetches.

            The citation guarantee still has the last word: the guard
            appends verified refs to any claim block the model left
            uncited and rebuilds `## References` from the ledger, so a
            missing citation can never ship. When the guard rewrites
            anything, the corrected full text goes out as one
            ``answer_replace`` event (the UI swaps the streamed draft
            for it) and the returned value — what the orchestrator
            echoes — is always that corrected text.
            """
            nonlocal synth_streamed
            synth_streamed = True
            from src.agents.synthesizer import (
                build_agent as build_synth,
                build_synthesis_message as build_synth_message,
            )
            from src.lib.skills import detect_writing_mode
            from src.lib.answer_cite import enforce_citations
            await emit("task", id="synthesizer", state="started",
                       question=f"Synthesize: {question[:160]}",
                       model=os.environ.get("AGENT_MODEL"),
                       ts=_now_iso())
            await emit("status", state="synthesizing",
                       message="Synthesizer composing the final answer…",
                       ts=_now_iso())
            # The skill system picks the writing style from the question
            # (skill always used; humanizer always applied inside).
            synth = build_synth(mode=detect_writing_mode(question))
            synth_state: dict = {}
            synth_handler = _make_handler("synthesizer", stream_text=True,
                                          state=synth_state)
            try:
                message = build_synth_message(question, chat_id)
                result = await synth.run(
                    message,
                    event_stream_handler=synth_handler)
            except Exception as exc:  # noqa: BLE001 — reported, not raised
                await emit("task", id="synthesizer", state="failed",
                           error=_format_error(exc, "synthesizer"),
                           ts=_now_iso())
                raise _tag_error(exc, leg="synthesizer",
                                 tool=synth_state.get("tool"))
            await _record_usage(result)
            final = enforce_citations(result.output, _ledger_refs())
            if final != "".join(synth_state.get("answer_parts") or []):
                await emit("answer_replace", delta=final)
            await emit("task", id="synthesizer", state="done", ts=_now_iso())
            return final

        async def _drive() -> str:
            """Run the orchestrator: it plans, delegates, then answers.

            Via run(), not run_stream(): run_stream stops the graph at
            the first text output, so a chatty preamble ("I'll gather
            literature…") ends the run before any delegation — tools
            execute silently with no events and the run stalls. run()
            completes the whole plan/delegate/synthesize loop while the
            handler streams thinking, tool, and answer events live.
            """
            runner, deps = _ensure_orchestrator()
            state: dict = {}
            # stream_text=False: the orchestrator itself writes no answer —
            # the synthesize tool runs the synthesizer sub-stream, which
            # emits the live ``answer`` deltas. Without this, the model's
            # verbatim echo of the tool result would re-stream (and the
            # end-of-run fallback below would flash) the whole answer twice.
            handler = _make_handler("", stream_text=False, state=state)
            try:
                drive_result = await runner.run(
                    question, deps=deps,
                    event_stream_handler=handler)
                output = drive_result.output
                await _record_usage(drive_result)
            except Exception as exc:
                await _store.record_run_finished(
                    chat_id, run_id, RunStatus.FAILED,
                    error={"code": "ORCHESTRATOR_FAILED",
                           "message": _format_error(exc, "orchestrator")})
                raise _tag_error(exc, leg="orchestrator",
                                 tool=state.get("tool"))
            return output

        try:
            if timeout_s:
                async with asyncio.timeout(timeout_s):
                    output = await _drive()
            else:
                output = await _drive()
        except (asyncio.TimeoutError, TimeoutError):
            await emit("error", code="UPSTREAM_TIMEOUT",
                       message="deep-agent run timed out", retryable=True)
            await _store.record_run_finished(
                chat_id, run_id, RunStatus.FAILED,
                error={"code": "UPSTREAM_TIMEOUT",
                       "message": "deep-agent run timed out"})
            return {"timeout": True, "usage": usage, "output": ""}
        budget_summary = None
        if _run_budget is not None:
            budget_summary = (await _run_budget.snapshot()).to_dict()
        await _store.record_run_finished(
            chat_id, run_id, RunStatus.COMPLETE, budget=budget_summary)
        progress = None
        ledger = _store.get_chat(chat_id)
        if ledger is not None:
            progress = progress_view(ledger, run_id)
        return {"timeout": False, "usage": usage, "output": output,
                "budget": budget_summary, "progress": progress,
                "answer_streamed": synth_streamed}

    # Drain-while-running: the handler / run_fn pushes into _queue while we
    # yield; _done signals the producer finished.
    _done = asyncio.Event()
    _failure: list[dict] = []
    _outcome: dict = {}

    async def _produce() -> None:
        try:
            if run_fn is not None:
                if timeout_s:
                    async with asyncio.timeout(timeout_s):
                        _outcome.update(
                            {"output": await run_fn(question, emit),
                             "usage": {"prompt": 0, "completion": 0}})
                else:
                    _outcome.update(
                        {"output": await run_fn(question, emit),
                         "usage": {"prompt": 0, "completion": 0}})
            else:
                _outcome.update(await _real_run())
        except (asyncio.TimeoutError, TimeoutError):
            _failure.append({"type": "error", "run_id": run_id,
                             "code": "UPSTREAM_TIMEOUT",
                             "message": "deep-agent run timed out",
                             "retryable": True})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced as an event
            leg = getattr(exc, "_medrag_leg", "research")
            _failure.append({"type": "error", "run_id": run_id,
                             "code": "TOOL_FAILED",
                             "message": _format_error(exc, leg),
                             **_error_fields(exc, leg),
                             "retryable": False})
        finally:
            _done.set()

    producer = asyncio.ensure_future(_produce())
    try:
        while not (_done.is_set() and _queue.empty()):
            try:
                event = await asyncio.wait_for(_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                if producer.done() and _queue.empty():
                    break
                continue
            _account(event)
            yield event
        await asyncio.shield(producer)
    except asyncio.CancelledError:
        producer.cancel()
        raise
    if _failure:
        yield _failure[0]
        return
    if _outcome.get("timeout"):
        return
    output = _outcome.get("output", "")
    usage = _outcome.get("usage", {"prompt": 0, "completion": 0})
    if output and not seen_answer and not _outcome.get("answer_streamed"):
        # Non-streaming runner (or no text deltas): deliver as one chunk.
        # Skipped when the synthesizer already streamed the answer live.
        yield {"type": "answer", "run_id": run_id, "seq": seq + 1,
               "delta": str(output), "done": False}
        seq += 1
    if seen_answer or output:
        if current_state != "answering":
            yield _status("answering", "composing answer")
        yield {"type": "answer", "run_id": run_id, "seq": seq + 1,
               "done": True}
        seq += 1
    if seen_thinking:
        yield {"type": "thinking", "run_id": run_id, "seq": seq + 1,
               "done": True}
        seq += 1
    done_event: dict = {"type": "done", "run_id": run_id,
                        "usage": usage, "citations": ctx.get("citations", [])}
    if _outcome.get("budget") is not None:
        done_event["budget"] = _outcome["budget"]
    if _outcome.get("progress") is not None:
        done_event["progress"] = _outcome["progress"]
    yield done_event