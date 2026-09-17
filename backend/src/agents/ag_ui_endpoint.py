"""AG-UI HTTP endpoint.

AG-UI is the transport over the existing deep-research pipeline. This endpoint
parses an AG-UI `RunAgentInput`, extracts the user question, runs
`stream_adapter.stream_deep_agent` — the exact same generator `POST
/v1/chat/stream` uses, with the same orchestrator, sub-agent spawn/synthesize
implementations, budgets, ledger, requirement injection, and citation guard —
and encodes its events as AG-UI SSE frames (see `ag_ui_transport`).

Human-in-the-loop stays on the existing broker: the pipeline parks the run and
the stream stays open until the client POSTs the answer to
`/v1/chats/{chat_id}/questions/{question_id}/answer`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable
from http import HTTPStatus

from pydantic import ValidationError
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from src.agents.ag_ui_transport import encode_ag_ui_events

__all__ = ["PipelineFactory", "set_pipeline_factory", "handle_ag_ui"]

# (question, thread_id, run_id) -> stream of BackendEvent dicts.
PipelineFactory = Callable[[str, str, str], AsyncIterator[dict]]

_pipeline_factory: PipelineFactory | None = None


def set_pipeline_factory(factory: PipelineFactory | None) -> None:
    """Override the pipeline (tests/embedding); None restores stream_deep_agent."""
    global _pipeline_factory
    _pipeline_factory = factory


def _default_pipeline(question: str, thread_id: str, run_id: str) -> AsyncIterator[dict]:
    # Imported lazily so this module imports without LLM env vars set.
    from src.agents.stream_adapter import stream_deep_agent

    return stream_deep_agent(question, run_id=run_id, conversation_id=thread_id)


def _question_of(run_input: object) -> str:
    """The last user message's text from an AG-UI run input."""
    messages = getattr(run_input, "messages", None) or []
    for message in reversed(list(messages)):
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        parts: list[str] = []
        for part in content or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return " ".join(parts).strip()
    return ""


async def handle_ag_ui(request: Request) -> Response:
    """Serve one AG-UI run over the existing pipeline as `text/event-stream`."""
    try:
        run_input = AGUIAdapter.build_run_input(await request.body())
    except ValidationError as exc:
        return Response(
            content=json.dumps(exc.errors(), default=str),
            media_type="application/json",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    question = _question_of(run_input)
    thread_id = (getattr(run_input, "thread_id", "") or "").strip() or uuid.uuid4().hex[:12]
    run_id = (getattr(run_input, "run_id", "") or "").strip() or uuid.uuid4().hex[:12]
    factory = _pipeline_factory or _default_pipeline

    async def pipeline() -> AsyncIterator[dict]:
        if not question:
            yield {"type": "error", "code": "INVALID_REQUEST",
                   "message": "question must be a non-empty string"}
            return
        async for event in factory(question, thread_id, run_id):
            yield event

    stream = encode_ag_ui_events(pipeline(), thread_id=thread_id, run_id=run_id)
    return StreamingResponse(stream, media_type="text/event-stream")
