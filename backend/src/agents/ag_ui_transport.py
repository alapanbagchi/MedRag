"""Encode the existing deep-research pipeline as AG-UI.

This is a transport adapter, not a second pipeline: it consumes the exact
`BackendEvent` dicts that `stream_adapter.stream_deep_agent` yields (the same
generator the NDJSON endpoint uses) and re-encodes them as AG-UI events. The
orchestrator, sub-agent spawning, budgets, ledger, requirement injection, and
citation guard are untouched.

Standard AG-UI events are used for text and tool calls; MedRAG's richer events
(`plan`, `step`, `task`, `sources`, ...) ride as AG-UI `CUSTOM` events, which
`@assistant-ui/react-ag-ui` turns into `data` parts.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import (
    CustomEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder

__all__ = ["encode_ag_ui_events", "step_kind_for"]

# Mirrors the frontend's tool-name -> step-kind mapping (frontend/src/lib/run.ts).
_PREFIX_KINDS: tuple[tuple[str, str], ...] = (
    ("submit_plan", "planner"),
    ("plan", "planner"),
    ("lookup_medical", "umls"),
    ("lookup", "research"),
    ("local_search", "retrieve"),
    ("web_search", "web_search"),
    ("retrieve", "retrieve"),
    ("firecrawl_web", "web_search"),
    ("firecrawl_fetch", "web_fetch"),
    ("browser_navigate", "web_search"),
    ("browser_snapshot", "web_fetch"),
    ("browser_", "web_search"),
    ("check_gaps", "gap_check"),
    ("check_evidence", "gap_check"),
    ("run_progress", "evidence"),
    ("synthesize", "synthesize"),
    ("spawn_subagent", "delegate"),
)


def step_kind_for(name: str) -> str:
    lowered = (name or "").lower()
    for prefix, kind in _PREFIX_KINDS:
        if lowered.startswith(prefix):
            return kind
    return "research"


def _short(value: Any, limit: int = 160) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    # Tool results reach ~1 MB; collapsing all of it to keep 110 characters
    # was the single largest per-event CPU cost in the stream. Eight times
    # the limit is enough headroom that the collapsed head is identical,
    # with a fallback for whitespace-heavy payloads.
    head = text[: limit * 8] if len(text) > limit * 8 else text
    collapsed = " ".join(head.split())
    if len(head) < len(text) and len(collapsed) < limit:
        collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe copy for a CUSTOM payload."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001 — payloads are best-effort
        return str(value)


# ---------------------------------------------------------------------------
# Stream gating
# ---------------------------------------------------------------------------
# While we chase UI lag, everything except the run lifecycle is opt-in. Set
# ``AGUI_EVENTS`` to a comma-separated list of the events to forward, or to
# ``all`` to restore the original firehose. Default (unset) = none, so we can
# add one back at a time and measure.
#
# Recognised tokens:
#   text            answer token streaming (TEXT_MESSAGE_*)
#   answer_replace  citation-guard rewrite
#   thinking        agent/judge thought deltas
#   tool            standard TOOL_CALL_START/ARGS/END + TOOL_CALL_RESULT
#   step            the custom `step` rows the UI renders tools from
#   plan status task run_stats error_detail tool_progress
#   sources verdict_table pipeline question
#
# The run lifecycle (RUN_STARTED / RUN_FINISHED / RUN_ERROR) is always sent.
# Answer text (TEXT_MESSAGE_* and answer_replace) is also always sent: it is
# the product output, not an optional debug detail.
_ALL_EVENTS = frozenset({"*"})
# (raw env value, parsed set); re-reads os.environ once per value, not once
# per streamed event (this is called two or three times per event).
_ENABLED_CACHE: tuple[str, frozenset] | None = None


def _enabled(name: str) -> bool:
    global _ENABLED_CACHE
    raw = os.environ.get("AGUI_EVENTS", "").strip().lower()
    cached = _ENABLED_CACHE
    if cached is None or cached[0] != raw:
        if raw == "all":
            allowed = _ALL_EVENTS
        elif not raw:
            allowed = frozenset()
        else:
            allowed = frozenset(p.strip() for p in raw.split(",") if p.strip())
        cached = (raw, allowed)
        _ENABLED_CACHE = cached
    return "*" in cached[1] or name in cached[1]


async def encode_ag_ui_events(
    events: AsyncIterator[dict],
    *,
    thread_id: str,
    run_id: str,
) -> AsyncIterator[str]:
    """Yield SSE-encoded AG-UI frames for one pipeline run."""
    encoder = EventEncoder()
    yield encoder.encode(RunStartedEvent(thread_id=thread_id, run_id=run_id))

    answer_id: str | None = None
    calls: dict[str, dict[str, Any]] = {}
    finished = False

    async for event in events:
        etype = event.get("type")

        if etype in ("answer", "token"):
            delta = event.get("delta") or event.get("content") or ""
            if delta:
                if answer_id is None:
                    answer_id = str(uuid.uuid4())
                    yield encoder.encode(
                        TextMessageStartEvent(message_id=answer_id, role="assistant")
                    )
                yield encoder.encode(
                    TextMessageContentEvent(message_id=answer_id, delta=str(delta))
                )
            if event.get("done") and answer_id is not None:
                yield encoder.encode(TextMessageEndEvent(message_id=answer_id))
                answer_id = None

        elif etype == "answer_replace":
            # The citation guard rewrote the streamed draft. The frontend
            # replaces the last answer text with this payload.
            yield encoder.encode(
                CustomEvent(
                    name="answer_replace",
                    value={"text": event.get("delta") or event.get("text") or ""},
                )
            )

        elif etype == "thinking":
            if not _enabled("thinking"):
                continue
            # Agent-attributed thought delta; the frontend concatenates these
            # per agent into the streaming thought stream / sub-agent cards.
            # `origin` marks judge-middleware reasoning, which the UI drops.
            yield encoder.encode(
                CustomEvent(
                    name="thinking",
                    value={
                        "agent": event.get("agent") or "",
                        "delta": str(event.get("delta") or ""),
                        "done": bool(event.get("done")),
                        "origin": event.get("origin") or "",
                        "ts": event.get("ts"),
                    },
                )
            )

        elif etype == "tool_call":
            call_id = str(event.get("call_id") or uuid.uuid4())
            name = str(event.get("name") or "tool")
            args = event.get("args")
            # Bookkeeping is unconditional: the completed step needs the args.
            calls[call_id] = {"name": name, "args": args, "ts": event.get("ts")}
            if _enabled("tool"):
                yield encoder.encode(
                    ToolCallStartEvent(tool_call_id=call_id, tool_call_name=name)
                )
                if args is not None:
                    yield encoder.encode(
                        ToolCallArgsEvent(
                            tool_call_id=call_id,
                            delta=json.dumps(_jsonable(args), ensure_ascii=False),
                        )
                    )
                yield encoder.encode(ToolCallEndEvent(tool_call_id=call_id))
            if not _enabled("step"):
                continue
            # A running step row (the completed row follows on tool_result).
            yield encoder.encode(
                CustomEvent(
                    name="step",
                    value={
                        "kind": step_kind_for(name),
                        "label": name,
                        "detail": _short(args, 110),
                        "callId": call_id,
                        "done": False,
                        "rawArgs": _jsonable(args),
                        "startedTs": event.get("ts"),
                        "timeline": [{"t": event.get("ts"), "text": f"Called {name}"}],
                    },
                )
            )

        elif etype == "tool_result":
            call_id = str(event.get("call_id") or "")
            info = calls.pop(call_id, {})
            name = str(event.get("name") or info.get("name") or "tool")
            result = event.get("result")
            want_tool = _enabled("tool")
            want_step = _enabled("step")
            if not want_tool and not want_step:
                continue
            # Serialize the (possibly ~1 MB) result once and reuse it for the
            # tool frame and the step's rawResult, instead of round-tripping
            # the same payload twice.
            safe_result = result if isinstance(result, str) else _jsonable(result)
            content = result if isinstance(result, str) else json.dumps(
                safe_result, ensure_ascii=False, default=str
            )
            if want_tool:
                yield encoder.encode(
                    ToolCallResultEvent(
                        message_id=str(uuid.uuid4()), tool_call_id=call_id, content=content
                    )
                )
            if not want_step:
                continue
            ok = event.get("ok", True)
            yield encoder.encode(
                CustomEvent(
                    name="step",
                    value={
                        "kind": step_kind_for(name),
                        "label": name,
                        "detail": _short(content, 110),
                        "callId": call_id,
                        "done": True,
                        "rawArgs": _jsonable(info.get("args")),
                        "rawResult": safe_result,
                        "startedTs": info.get("ts"),
                        "finishedTs": event.get("ts"),
                        "error": None if ok else (_short(content, 80) or "tool failed"),
                    },
                )
            )

        elif etype == "plan":
            if not _enabled("plan"):
                continue
            yield encoder.encode(
                CustomEvent(
                    name="plan",
                    value={"fields": {"items": _jsonable(event.get("items") or [])}},
                )
            )

        elif etype == "status":
            if not _enabled("status"):
                continue
            yield encoder.encode(
                CustomEvent(
                    name="status",
                    value={
                        "stage": event.get("stage") or event.get("state") or "",
                        "message": event.get("message") or "",
                        "count": event.get("count") or 0,
                    },
                )
            )

        elif etype == "done":
            finished = True
            if _enabled("run_stats"):
                yield encoder.encode(
                    CustomEvent(
                        name="run_stats",
                        value={
                            "usage": _jsonable(event.get("usage") or {}),
                            "citations": _jsonable(event.get("citations") or []),
                        },
                    )
                )
            yield encoder.encode(
                RunFinishedEvent(
                    thread_id=thread_id,
                    run_id=run_id,
                    outcome=RunFinishedSuccessOutcome(),
                )
            )

        elif etype == "error":
            if _enabled("error_detail"):
                yield encoder.encode(
                    CustomEvent(
                        name="error_detail",
                        value={
                            "message": event.get("message") or "run failed",
                            "code": event.get("code"),
                            "leg": event.get("leg"),
                            "tool": event.get("tool"),
                            "model": event.get("model"),
                            "statusCode": event.get("status_code"),
                        },
                    )
                )
            yield encoder.encode(
                RunErrorEvent(message=str(event.get("message") or "run failed"))
            )
            finished = True

        elif etype == "task":
            if not _enabled("task"):
                continue
            state = event.get("state")
            yield encoder.encode(
                CustomEvent(
                    name="task",
                    value={
                        "id": event.get("id"),
                        "state": state,
                        "question": event.get("question"),
                        "model": event.get("model"),
                        "startedTs": event.get("ts") if state == "started" else None,
                        "finishedTs": event.get("ts") if state in ("done", "failed") else None,
                        "error": event.get("error"),
                    },
                )
            )

        elif etype == "tool_progress":
            if not _enabled("tool_progress"):
                continue
            # Retrieval progress fires before the judge runs (see
            # `_report_progress` in the judge middleware) — carrying the raw
            # passages lets the UI show the found articles live, then gray the
            # irrelevant ones when the judged tool_result lands.
            yield encoder.encode(
                CustomEvent(
                    name="tool_progress",
                    value={
                        "callId": event.get("call_id"),
                        "name": event.get("name"),
                        "detail": _short(event.get("result"), 110),
                        "result": _jsonable(event.get("result")),
                        "ts": event.get("ts"),
                    },
                )
            )

        elif etype in ("sources", "verdict_table", "pipeline", "question"):
            if not _enabled(etype):
                continue
            name = "question" if etype == "question" else etype
            payload = {k: _jsonable(v) for k, v in event.items() if k not in ("type", "run_id")}
            if etype == "question":
                payload = {
                    "callId": event.get("call_id"),
                    "questions": _jsonable(event.get("questions") or []),
                    # The pipeline uses the AG-UI thread id as the chat id; the
                    # broker answer route needs it back from the client.
                    "threadId": thread_id,
                }
            yield encoder.encode(CustomEvent(name=name, value=payload))

        # Unknown event types are ignored.

    if answer_id is not None:
        yield encoder.encode(TextMessageEndEvent(message_id=answer_id))
    if not finished:
        yield encoder.encode(
            RunFinishedEvent(
                thread_id=thread_id, run_id=run_id, outcome=RunFinishedSuccessOutcome()
            )
        )
