"""Trace logging regression tests (log hygiene)."""

from __future__ import annotations

from src.trace import Trace


def test_tool_call_and_result_are_not_double_logged(tmp_path):
    """Each tool call must produce exactly ONE [tool] line; passing a result
    to the completion call must not re-print the invocation line."""
    log = tmp_path / "trace.log"
    trace = Trace()
    trace.open_stream(log, query="q")
    trace.tool("search_umls", {"term": "heart failure"})
    result = {"term": "heart failure", "found": True, "cui": "C0018801"}
    trace.tool("search_umls", {"term": "heart failure"}, result)
    trace.close_stream()

    text = log.read_text()
    assert text.count("[tool] search_umls") == 1
    assert text.count("[tool-result] search_umls") == 1
