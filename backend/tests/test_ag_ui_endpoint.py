"""AG-UI transport: /v1/ag-ui encodes the existing pipeline's events.

The endpoint must run `stream_deep_agent`'s exact event stream (`BackendEvent`
dicts) and re-encode it as AG-UI. These tests inject a fake pipeline via
`set_pipeline_factory` so no model is needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.agents import ag_ui_endpoint


def _body(thread_id: str = "t1", content: str = "hello") -> dict:
    return {
        "threadId": thread_id,
        "runId": "r1",
        "messages": [{"id": "u1", "role": "user", "content": content}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def _client() -> TestClient:
    app = FastAPI()

    @app.post("/v1/ag-ui")
    async def _ag_ui(request: Request):  # type: ignore[no-untyped-def]
        return await ag_ui_endpoint.handle_ag_ui(request)

    return TestClient(app)


async def _fake_pipeline(question: str, thread_id: str, run_id: str) -> AsyncIterator[dict]:
    yield {"type": "status", "state": "retrieving", "message": "working"}
    yield {
        "type": "tool_call", "call_id": "c1", "name": "retrieve_evidence",
        "args": {"query": "statins"}, "ts": "t0",
    }
    yield {
        "type": "tool_result", "call_id": "c1", "name": "retrieve_evidence",
        "ok": True, "result": {"count": 3}, "ts": "t1",
    }
    yield {"type": "plan", "items": [{"id": "T1", "question": "Test plan item"}]}
    yield {"type": "answer", "delta": "Final answer text", "done": False}
    yield {"type": "answer", "done": True}
    yield {"type": "done", "usage": {"prompt": 12, "completion": 7}}


def test_endpoint_encodes_pipeline_events(monkeypatch) -> None:
    monkeypatch.setenv("AGUI_EVENTS", "all")
    ag_ui_endpoint.set_pipeline_factory(_fake_pipeline)
    try:
        resp = _client().post("/v1/ag-ui", json=_body(), headers={"accept": "text/event-stream"})
    finally:
        ag_ui_endpoint.set_pipeline_factory(None)

    assert resp.status_code == 200, resp.text
    text = resp.text
    for fragment in (
        '"type":"RUN_STARTED"',
        '"type":"CUSTOM"',
        '"name":"status"',
        '"name":"step"',
        '"name":"plan"',
        '"name":"run_stats"',
        '"type":"TOOL_CALL_START"',
        '"type":"TOOL_CALL_RESULT"',
        '"type":"TEXT_MESSAGE_CONTENT"',
        '"type":"RUN_FINISHED"',
    ):
        assert fragment in text, f"missing {fragment}"
    assert '"callId":"c1"' in text
    assert '"kind":"retrieve"' in text
    assert "Final answer text" in text
    assert '"prompt":12' in text


def test_endpoint_default_streams_lifecycle_and_answer(monkeypatch) -> None:
    """With AGUI_EVENTS unset, only the lifecycle and the answer are sent."""
    monkeypatch.delenv("AGUI_EVENTS", raising=False)
    ag_ui_endpoint.set_pipeline_factory(_fake_pipeline)
    try:
        resp = _client().post("/v1/ag-ui", json=_body(), headers={"accept": "text/event-stream"})
    finally:
        ag_ui_endpoint.set_pipeline_factory(None)

    text = resp.text
    assert '"type":"RUN_STARTED"' in text
    assert '"type":"RUN_FINISHED"' in text
    assert "TEXT_MESSAGE_CONTENT" in text  # the answer is never optional
    assert "Final answer text" in text
    assert '"type":"CUSTOM"' not in text
    assert "TOOL_CALL" not in text


def test_endpoint_gates_events_individually(monkeypatch) -> None:
    """Only the named events are forwarded — the whole point of the gate."""
    monkeypatch.setenv("AGUI_EVENTS", "step,status")
    ag_ui_endpoint.set_pipeline_factory(_fake_pipeline)
    try:
        resp = _client().post("/v1/ag-ui", json=_body(), headers={"accept": "text/event-stream"})
    finally:
        ag_ui_endpoint.set_pipeline_factory(None)

    text = resp.text
    assert '"name":"status"' in text
    assert '"name":"step"' in text
    assert '"name":"plan"' not in text
    assert '"name":"run_stats"' not in text
    assert "TEXT_MESSAGE_CONTENT" in text  # the answer is never gated
    assert "TOOL_CALL_START" not in text


def test_endpoint_errors_on_empty_question() -> None:
    ag_ui_endpoint.set_pipeline_factory(_fake_pipeline)
    try:
        resp = _client().post("/v1/ag-ui", json=_body(content=""))
    finally:
        ag_ui_endpoint.set_pipeline_factory(None)

    assert resp.status_code == 200
    assert '"type":"RUN_ERROR"' in resp.text
