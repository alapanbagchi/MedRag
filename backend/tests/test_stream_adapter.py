"""Tests for the deep-agent NDJSON stream adapter (bridge replacement)."""

from __future__ import annotations

import json

import pytest
from pydantic_ai import RunContext

from src.tools.umls import DeepDeps

from src.agents.stream_adapter import (
    _coerce_args,
    _summarize_result,
    _verdict_rows,
    status_for_tool,
    stream_deep_agent,
)


async def _collect(question="what treats X?", **kwargs):
    return [e async for e in stream_deep_agent(question, **kwargs)]


async def _await(value):
    """Wrap a plain value as an awaitable (for async-fn fakes)."""
    return value


def _scripted_brain(*responses):
    """FunctionModel brain replaying canned streamed responses.

    Each response is a list of chunks: ("text", str) for answer text,
    ("thinking", str) for thinking, ("tool", name, args, tool_call_id)
    for one tool call. One response is consumed per model turn.
    """
    import json as _json

    from pydantic_ai.models.function import (
        DeltaThinkingPart,
        DeltaToolCall,
        FunctionModel,
    )

    calls = {"n": 0}

    async def _stream(messages, info):
        i = calls["n"]
        calls["n"] += 1
        for chunk in responses[min(i, len(responses) - 1)]:
            kind = chunk[0]
            if kind == "text":
                yield chunk[1]
            elif kind == "thinking":
                yield {0: DeltaThinkingPart(content=chunk[1])}
            elif kind == "tool":
                _, name, args, call_id = chunk
                yield {0: DeltaToolCall(name=name,
                                        json_args=_json.dumps(args),
                                        tool_call_id=call_id)}
            else:  # pragma: no cover - test script typo
                raise AssertionError(f"bad scripted chunk {chunk!r}")

    return FunctionModel(stream_function=_stream)


def _orchestrator_env(monkeypatch):
    """LLM env so the real orchestrator builder runs (model is faked)."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")


def _patch_orchestrator_model(monkeypatch, brain):
    """Route the orchestrator builder at the FunctionModel brain."""
    from src.agents import orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "make_model",
                        lambda *args, **kwargs: brain)


class _ScriptedResult:
    """Fake Agent.run result replaying scripted pydantic-ai events."""

    def __init__(self, events, output="out"):
        self._events = events
        self.output = output
        self._handler = None

    def bind(self, handler):
        self._handler = handler
        return self

    async def drive(self):
        await self._handler(None, self._events)
        return self

    def usage(self):
        from types import SimpleNamespace

        return SimpleNamespace(request_tokens=1, response_tokens=1)


class _ScriptedAgent:
    """Fake agent (orchestrator or sub-agent) replaying scripted events."""

    def __init__(self, events, output="out"):
        self._events = events
        self._output = output

    async def run(self, prompt, *args, **kwargs):
        return await _ScriptedResult(
            self._events, self._output
        ).bind(kwargs.get("event_stream_handler")).drive()


async def _no_events():
    if False:  # pragma: no cover - empty delegation
        yield


async def test_full_run_event_sequence():
    async def fake_run(question, emit):
        assert question == "what treats X?"
        await emit("thinking", delta="planning...", done=False)
        await emit("tool_call", call_id="tc_1", name="retrieve_evidence",
                   args={"q": "X"})
        await emit("tool_result", call_id="tc_1", name="retrieve_evidence",
                   ok=True, result={"n": 2}, truncated=False)
        await emit("answer", delta="It is Y.", done=False)
        return "It is Y."

    events = await _collect(run_fn=fake_run)
    types = [e["type"] for e in events]
    assert types[0] == "status" and events[0]["state"] == "started"
    assert types[-1] == "done"
    assert "thinking" in types and "tool_call" in types
    assert "tool_result" in types and "answer" in types
    # every event carries run_id; exactly one run_id overall
    run_ids = {e["run_id"] for e in events}
    assert len(run_ids) == 1
    # seq increases on every event except status/done
    seqs = [e["seq"] for e in events if e["type"] not in ("status", "done")]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert all("seq" not in e for e in events
               if e["type"] in ("status", "done"))
    # done carries usage + citations
    assert events[-1]["usage"] == {"prompt": 0, "completion": 0}
    assert events[-1]["citations"] == []
    # every event is NDJSON-serializable (one JSON object per line)
    for e in events:
        assert "\n" not in json.dumps(e)


async def test_non_streaming_output_delivered_as_answer():
    async def fake_run(question, emit):
        return "final only"

    events = await _collect(run_fn=fake_run)
    answers = [e for e in events if e["type"] == "answer"]
    assert answers and answers[0]["delta"] == "final only"
    assert answers[-1]["done"] is True
    assert events[-1]["type"] == "done"


async def test_run_fn_failure_yields_terminal_error_without_done():
    async def boom(question, emit):
        raise RuntimeError("llm down")

    events = await _collect(run_fn=boom)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "TOOL_FAILED"
    assert "RuntimeError" in events[-1]["message"]
    assert events[-1]["retryable"] is False
    assert not [e for e in events if e["type"] == "done"]


async def test_empty_question_rejected():
    events = await _collect(question="   ")
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "INVALID_REQUEST"


async def test_timeout_maps_to_upstream_timeout():
    import asyncio

    async def slow(question, emit):
        await asyncio.sleep(5)
        return "late"

    events = await _collect(run_fn=slow, timeout_s=0.05)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "UPSTREAM_TIMEOUT"
    assert events[-1]["retryable"] is True


def test_status_for_tool_mapping():
    assert status_for_tool("spawn_subagent") == "planning"
    assert status_for_tool("local_search") == "retrieving"
    assert status_for_tool("web_search") == "retrieving"
    assert status_for_tool("judge_claims") == "verifying"
    assert status_for_tool("final_result") is None
    assert status_for_tool("") is None


def test_coerce_args_and_result_truncation():
    assert _coerce_args('{"a": 1}') == {"a": 1}
    assert _coerce_args("nope{[") == {"_raw": "nope{["}
    assert _coerce_args({}) == {}
    payload, truncated = _summarize_result({"k": "v"})
    assert payload == {"k": "v"} and truncated is False
    payload, truncated = _summarize_result("x" * 5000)
    assert truncated is True and len(payload) == 4000
    assert _verdict_rows({"rows": [[1]]}) == [[1]]
    assert _verdict_rows({"nope": 1}) is None
    assert _verdict_rows([1]) is None


def test_evidence_tool_results_not_truncated():
    from src.agents.stream_adapter import _cap_for_tool, _summarize_result

    big = "x" * 50_000
    payload, truncated = _summarize_result(big, _cap_for_tool("local_search"))
    assert truncated is False and payload == big
    # Web search carries full scraped markdown — truncating it mid-JSON
    # would break the frontend table, so it rides the evidence cap too.
    payload, truncated = _summarize_result(big, _cap_for_tool("web_search"))
    assert truncated is False and payload == big
    payload, truncated = _summarize_result(big, _cap_for_tool("lookup_medical_term"))
    assert truncated is True and len(payload) == 4000


async def test_first_tool_result_correlates_to_its_call():
    """Regression: the first tool result in a run must not raise
    UnboundLocalError — the call id is assigned before use."""
    from types import SimpleNamespace
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        ToolCallPart,
        ToolReturnPart,
    )

    call_part = ToolCallPart(tool_name="lookup_medical_term",
                             args={"term": "x"}, tool_call_id="tc_1")
    ret_part = ToolReturnPart(tool_name="lookup_medical_term",
                              content='{"term": "x", "found": false}',
                              tool_call_id="tc_1")

    async def _events():
        yield FunctionToolCallEvent(part=call_part)
        yield FunctionToolResultEvent(part=ret_part)

    events = await _collect(orchestrator=_ScriptedAgent(_events(), "final"),
                            umls=SimpleNamespace(), warmup=False)
    assert not [e for e in events if e["type"] == "error"]
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == 1 and len(results) == 1
    assert results[0]["call_id"] == calls[0]["call_id"] == "tc_1"
    assert results[0]["name"] == "lookup_medical_term"
    assert events[-1]["type"] == "done"


async def test_thought_stream_is_model_reasoning_only():
    """Thinking carries model reasoning only: narrated backend lines never
    enter it, tool calls stream alongside without interrupting the thought
    flow, and the orchestrator's answer text streams as answer."""
    from types import SimpleNamespace
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        PartDeltaEvent,
        TextPartDelta,
        ThinkingPartDelta,
        ToolCallPart,
        ToolReturnPart,
    )
    from src.lib import narrate

    async def _events():
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(
            content_delta="real thought"))
        narrate.say("[deep] retrieve_evidence … (backend log line)")
        yield FunctionToolCallEvent(part=ToolCallPart(
            tool_name="retrieve_evidence", args={"q": "Q1"},
            tool_call_id="tc_1"))
        yield FunctionToolResultEvent(part=ToolReturnPart(
            tool_name="retrieve_evidence", content='{"n": 1}',
            tool_call_id="tc_1"))
        yield PartDeltaEvent(index=1, delta=TextPartDelta(
            content_delta="final answer"))

    events = await _collect(orchestrator=_ScriptedAgent(_events(), "final answer"),
                            umls=SimpleNamespace(), warmup=False)
    thinking = [e["delta"] for e in events
                if e["type"] == "thinking" and "delta" in e]
    assert thinking == ["real thought"]
    assert not [e for e in events if e["type"] == "tool_call"
                and e.get("name") != "retrieve_evidence"]
    answers = [e.get("delta", "") for e in events
               if e["type"] == "answer" and "delta" in e]
    assert "final answer" in "".join(answers)
    assert not [e for e in events
                if "backend log line" in json.dumps(e, default=str)]
    assert events[-1]["type"] == "done"


def test_no_memory_imports():
    import src.agents.stream_adapter as mod

    import re
    src = open(mod.__file__, encoding="utf-8").read()
    imports = [ln for ln in src.splitlines()
               if re.match(r"\s*(import|from)\s", ln)]
    assert not any("memory" in ln for ln in imports)


@pytest.mark.skip(reason="live LLM smoke; run manually with env set")
async def test_live_smoke_manual():
    events = await _collect(question="What is metformin?")
    assert events[-1]["type"] == "done"  # pragma: no cover


async def test_plan_event_streamed_first():
    """The orchestrator's submit_plan call doubles as the UI plan card."""
    from types import SimpleNamespace
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        PartDeltaEvent,
        TextPartDelta,
        ToolCallPart,
    )

    async def _events():
        yield FunctionToolCallEvent(part=ToolCallPart(
            tool_name="submit_plan",
            args={"items": [
                {"id": "P1", "question": "First task", "depth": "deep"},
                {"question": "Second task"},
            ]},
            tool_call_id="m1"))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(
            content_delta="Y"))

    events = await _collect(orchestrator=_ScriptedAgent(_events(), "Y"),
                            umls=SimpleNamespace(), warmup=False)
    types = [e["type"] for e in events]
    assert types[0] == "status" and events[0]["state"] == "started"
    plan = next(e for e in events if e["type"] == "plan")
    assert [i["question"] for i in plan["items"]] == ["First task", "Second task"]
    assert [i["id"] for i in plan["items"]] == ["P1", "T2"]
    assert [i["deep_research"] for i in plan["items"]] == [True, False]
    # plan streams before any answer output
    assert types.index("plan") < types.index("answer")
    assert events[-1]["type"] == "done"


async def test_orchestrator_spawns_deep_and_shallow(monkeypatch):
    """Full delegation loop with faked models: plan card, task marks on
    the planned ids, namespaced sub-agent tool calls, findings flowing
    back, final answer, done last."""
    from types import SimpleNamespace
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        ThinkingPartDelta,
        PartDeltaEvent,
        ToolCallPart,
        ToolReturnPart,
    )
    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod

    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("tool", "submit_plan",
          {"items": [
              {"id": "T1", "question": "Deep task", "depth": "deep"},
              {"id": "T2", "question": "Quick task", "depth": "shallow"},
          ]}, "m1")],
        [("tool", "spawn_subagent",
          {"task": "Deep task", "depth": "deep", "task_id": "T1"}, "m2")],
        [("tool", "spawn_subagent",
          {"task": "Quick task", "depth": "shallow", "task_id": "T2"}, "m3")],
        [("text", "FINAL")],
    ))

    async def _sub_events():
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(
            content_delta="sub thinking"))
        yield FunctionToolCallEvent(part=ToolCallPart(
            "retrieve_evidence", {"q": "Deep task"}, "tc_1"))
        yield FunctionToolResultEvent(part=ToolReturnPart(
            "retrieve_evidence", '[{"chunk_id": "c1"}]', "tc_1"))

    async def _shallow_events():
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(
            content_delta="quick thought"))

    monkeypatch.setattr(deep_agent_mod, "build_deep_agent",
                        lambda: _ScriptedAgent(_sub_events(), "findings-T1"))
    monkeypatch.setattr(orchestrator_mod, "build_shallow_agent",
                        lambda: _ScriptedAgent(_shallow_events(), "quick-T2"))

    events = await _collect(umls=SimpleNamespace(), warmup=False)
    types = [e["type"] for e in events]
    plan = next(e for e in events if e["type"] == "plan")
    assert [i["id"] for i in plan["items"]] == ["T1", "T2"]
    assert [i["deep_research"] for i in plan["items"]] == [True, False]
    tasks = [e for e in events if e["type"] == "task"]
    started = [e for e in tasks if e["state"] == "started"]
    assert started and all("model" in e for e in started)
    by_id: dict = {}
    for e in tasks:
        by_id.setdefault(e["id"], []).append(e["state"])
    # spawns track the planned ids, so the plan card marks progress
    assert by_id["T1"] == ["started", "done"]
    assert by_id["T2"] == ["started", "done"]
    # sub-agent tool calls are namespaced to their task
    calls = [e for e in events if e["type"] == "tool_call"]
    assert "T1:tc_1" in [c["call_id"] for c in calls]
    thinking = [e["delta"] for e in events
                if e["type"] == "thinking" and "delta" in e]
    # The task header is emitted with a trailing paragraph break so the UI
    # shows it as its own bubble rather than running into the first delta.
    assert any("[T1] Deep task" in t for t in thinking)
    assert "sub thinking" in thinking
    # order: plan first, tasks during research, final answer, done last
    assert types.index("plan") < types.index("task") < types.index("answer")
    assert events[-1]["type"] == "done"
    assert "".join(e.get("delta", "") for e in events
                   if e["type"] == "answer" and "delta" in e) == "FINAL"
    assert len({e["run_id"] for e in events}) == 1


async def test_preamble_text_with_tools_routes_to_thinking(monkeypatch):
    """Text in a tool-calling turn is narration, not the answer: a short
    preamble ("has two parts…") followed by submit_plan must land in
    thinking, never as a response."""
    from types import SimpleNamespace
    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod

    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("text", "has two parts, gathering literature on each. "),
         ("tool", "submit_plan",
          {"items": [{"id": "T1", "question": "Q", "depth": "shallow"}]},
          "m1")],
        [("tool", "spawn_subagent",
          {"task": "Q", "depth": "shallow", "task_id": "T1"}, "m2")],
        [("text", "FINAL")],
    ))
    monkeypatch.setattr(deep_agent_mod, "build_deep_agent",
                        lambda: _ScriptedAgent(_no_events(), "findings"))
    monkeypatch.setattr(orchestrator_mod, "build_shallow_agent",
                        lambda: _ScriptedAgent(_no_events(), "findings"))

    events = await _collect(umls=SimpleNamespace(), warmup=False)
    thinking = "".join(e.get("delta", "") for e in events
                       if e["type"] == "thinking" and "delta" in e)
    answers = "".join(e.get("delta", "") for e in events
                      if e["type"] == "answer" and "delta" in e)
    assert "has two parts" in thinking
    assert "has two parts" not in answers
    assert "FINAL" in answers
    assert next(e for e in events if e["type"] == "plan")
    assert events[-1]["type"] == "done"


async def test_long_tool_free_text_streams_live_as_answer(monkeypatch):
    """A long tool-free turn is the synthesis: it must still stream as
    answer deltas (not one silent chunk, not thinking)."""
    from types import SimpleNamespace

    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("text", "SYNTHESIS " * 100)],
    ))

    events = await _collect(umls=SimpleNamespace(), warmup=False)
    answers = "".join(e.get("delta", "") for e in events
                      if e["type"] == "answer" and "delta" in e)
    assert answers == "SYNTHESIS " * 100
    thinking = "".join(e.get("delta", "") for e in events
                       if e["type"] == "thinking" and "delta" in e)
    assert "SYNTHESIS" not in thinking
    assert events[-1]["type"] == "done"


async def test_sick_subagent_marks_failed_and_run_continues(monkeypatch):
    """A failing spawn reports back as findings text; the orchestrator
    finishes with what it has instead of killing the run."""
    from types import SimpleNamespace
    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod

    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("tool", "spawn_subagent",
          {"task": "Bad task", "depth": "deep", "task_id": "T1"}, "m1")],
        [("text", "PARTIAL")],
    ))

    class BrokenSubAgent:
        async def run(self, prompt, *args, **kwargs):
            raise RuntimeError("sub-agent down")

    monkeypatch.setattr(deep_agent_mod, "build_deep_agent",
                        lambda: BrokenSubAgent())
    monkeypatch.setattr(orchestrator_mod, "build_shallow_agent",
                        lambda: BrokenSubAgent())

    events = await _collect(umls=SimpleNamespace(), warmup=False)
    states = {(e["id"], e["state"]) for e in events if e["type"] == "task"}
    assert ("T1", "started") in states and ("T1", "failed") in states
    assert "".join(e.get("delta", "") for e in events
                   if e["type"] == "answer" and "delta" in e) == "PARTIAL"
    assert events[-1]["type"] == "done"


async def test_orchestrator_thinking_streams_before_plan():
    from types import SimpleNamespace
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        PartDeltaEvent,
        ThinkingPartDelta,
        ToolCallPart,
    )

    async def _events():
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(
            content_delta="considering…"))
        yield FunctionToolCallEvent(part=ToolCallPart(
            "submit_plan",
            {"items": [{"id": "P1", "question": "Q1", "depth": "deep"}]},
            tool_call_id="m1"))

    events = await _collect(orchestrator=_ScriptedAgent(_events(), "A"),
                            umls=SimpleNamespace(), warmup=False)
    thinking = [e["delta"] for e in events
                if e["type"] == "thinking" and "delta" in e]
    assert thinking == ["considering…"]
    plan_at = next(i for i, e in enumerate(events) if e["type"] == "plan")
    first_thought_at = next(i for i, e in enumerate(events)
                            if e.get("delta") == "considering…")
    assert first_thought_at < plan_at
    # the submit_plan call moves the indicator into planning
    statuses = [(e.get("state"), e.get("message")) for e in events
                if e["type"] == "status"]
    assert any(s[0] == "planning" for s in statuses)
    assert events[-1]["type"] == "done"


def test_format_error_names_leg_model_tool_and_status():
    from pydantic_ai.exceptions import ModelHTTPError
    from src.agents.stream_adapter import _format_error, _tag_error

    exc = ModelHTTPError(500, "qwen-x", "upstream boom")
    _tag_error(exc, leg="synthesis", tool="retrieve_evidence")
    message = _format_error(exc, "synthesis")
    assert "synthesis" in message
    assert "model qwen-x" in message
    assert "tool retrieve_evidence" in message
    assert "HTTP 500" in message
    assert "ModelHTTPError" in message


def test_format_error_truncates_long_bodies():
    from src.agents.stream_adapter import _format_error

    exc = RuntimeError("x" * 1000)
    message = _format_error(exc, "research")
    assert len(message) < 500
    assert message.startswith("research · RuntimeError:")


async def test_failed_subagent_error_names_model_and_status(monkeypatch):
    from types import SimpleNamespace
    from pydantic_ai.exceptions import ModelHTTPError
    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod

    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("tool", "spawn_subagent",
          {"task": "Q1", "depth": "deep", "task_id": "T1"}, "m1")],
        [("text", "A")],
    ))

    class OverloadedSubAgent:
        async def run(self, prompt, *args, **kwargs):
            raise ModelHTTPError(503, "qwen-x", "overloaded")

    monkeypatch.setattr(deep_agent_mod, "build_deep_agent",
                        lambda: OverloadedSubAgent())
    monkeypatch.setattr(orchestrator_mod, "build_shallow_agent",
                        lambda: OverloadedSubAgent())

    events = await _collect(umls=SimpleNamespace(), warmup=False)
    failed = next(e for e in events if e["type"] == "task" and e["state"] == "failed")
    assert "sub-agent T1" in failed["error"]
    assert "qwen-x" in failed["error"]
    assert "503" in failed["error"]
    assert failed["leg"] == "sub-agent T1"
    assert failed["model"] == "qwen-x"
    assert failed["status_code"] == 503
    assert events[-1]["type"] == "done"


async def test_terminal_error_carries_leg_and_model():
    from pydantic_ai.exceptions import ModelHTTPError

    async def boom(question, emit):
        raise ModelHTTPError(500, "planner-m", "bad gateway")

    events = await _collect(run_fn=boom)
    error = next(e for e in events if e["type"] == "error")
    assert "research" in error["message"]
    assert "planner-m" in error["message"]
    assert "HTTP 500" in error["message"]
    assert error["model"] == "planner-m"
    assert error["status_code"] == 500


async def test_orchestrator_failure_yields_terminal_error_without_done():
    """A dead orchestrator surfaces a terminal error (leg: orchestrator)."""
    from types import SimpleNamespace

    class BrokenOrchestrator:
        async def run(self, prompt, *args, **kwargs):
            raise RuntimeError("orchestrator down")

    events = await _collect(orchestrator=BrokenOrchestrator(),
                            umls=SimpleNamespace(), warmup=False)
    assert not [e for e in events if e["type"] == "plan"]
    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "TOOL_FAILED"
    assert "orchestrator" in error["message"]
    assert "RuntimeError" in error["message"]
    assert error["retryable"] is False
    assert not [e for e in events if e["type"] == "done"]


async def test_judge_thinking_sink_streams_into_thoughts():
    from src.middleware.llm_as_a_judge import _thinking_sink

    async def fake_run(question, emit):
        sink = _thinking_sink.get()
        assert sink is not None
        sink("tc_9", "judge hmm…")
        await emit("thinking", delta="leg thought", done=False)
        return "out"

    events = await _collect(run_fn=fake_run)
    thinking = [e["delta"] for e in events
                if e["type"] == "thinking" and "delta" in e]
    assert thinking == ["judge hmm…", "leg thought"]
    assert events[-1]["type"] == "done"


async def test_spawn_budget_caps_delegation(monkeypatch):
    """Past MAX_SUBAGENTS the spawn tool refuses gracefully and the run
    finishes with gathered evidence instead of spawning forever."""
    from types import SimpleNamespace
    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod
    from src.agents.orchestrator import MAX_SUBAGENTS

    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        *[[("tool", "spawn_subagent",
            {"task": f"Task {n}", "depth": "shallow"},
            f"m{n}")] for n in range(MAX_SUBAGENTS + 3)],
        [("text", "DONE")],
    ))
    monkeypatch.setattr(deep_agent_mod, "build_deep_agent",
                        lambda: _ScriptedAgent(_no_events(), "r"))
    monkeypatch.setattr(orchestrator_mod, "build_shallow_agent",
                        lambda: _ScriptedAgent(_no_events(), "r"))

    events = await _collect(umls=SimpleNamespace(), warmup=False)
    started = [e for e in events if e["type"] == "task"
               and e["state"] == "started"]
    assert len(started) == MAX_SUBAGENTS
    # the refused spawns still return tool results telling the model why
    exhausted = [e for e in events if e["type"] == "tool_result"
                 and "budget exhausted" in json.dumps(e.get("result", ""))]
    assert len(exhausted) == 3
    assert events[-1]["type"] == "done"


async def test_retrieval_progress_streams_passages_before_result(monkeypatch):
    """The judge middleware's passages-found ping surfaces as a partial
    tool_progress event (attributed to the right call), ahead of the
    final tool_result; unknown call ids are dropped silently."""
    from types import SimpleNamespace
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        ToolCallPart,
        ToolReturnPart,
    )
    from src.middleware.llm_as_a_judge import _progress_sink

    passages = ('[{"chunk_id": "c1", "document_id": "d1", '
                '"section": "Results", "score": 0.9, "text": "t"}]')
    call_part = ToolCallPart(tool_name="retrieve_evidence",
                             args={"query": "q"}, tool_call_id="tc_1")
    ret_part = ToolReturnPart(tool_name="retrieve_evidence",
                              content=passages, tool_call_id="tc_1")

    async def _events():
        yield FunctionToolCallEvent(part=call_part)
        sink = _progress_sink.get()
        assert sink is not None
        sink("unknown-id", "retrieve_evidence", passages)
        sink("tc_1", "retrieve_evidence", passages)
        yield FunctionToolResultEvent(part=ret_part)

    events = await _collect(orchestrator=_ScriptedAgent(_events(), "final"),
                            umls=SimpleNamespace(), warmup=False)
    types = [e["type"] for e in events]
    assert types.index("tool_call") < types.index("tool_progress") < types.index("tool_result")
    progress = [e for e in events if e["type"] == "tool_progress"]
    assert len(progress) == 1
    assert progress[0]["call_id"] == "tc_1"
    assert progress[0]["name"] == "retrieve_evidence"
    assert progress[0]["stage"] == "retrieved"
    assert progress[0]["result"] == passages
    assert progress[0]["run_id"] == events[0]["run_id"]
    assert events[-1]["type"] == "done"


def test_synthesize_tool_maps_to_synthesizing_status():
    assert status_for_tool("synthesize") == "synthesizing"


async def test_synthesizer_streams_live_as_visible_leg(monkeypatch):
    """The synthesizer runs as a visible leg: its thinking streams under
    agent=synthesizer, its answer streams as live deltas (not one
    flash), and the citation-guaranteed text lands via answer_replace."""
    from types import SimpleNamespace

    from pydantic_ai.messages import (
        PartDeltaEvent,
        TextPartDelta,
        ThinkingPartDelta,
    )

    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("tool", "submit_plan",
          {"items": [{"id": "T1", "question": "Q", "depth": "shallow"}]},
          "m1")],
        [("tool", "check_gaps", {}, "m2")],
        [("tool", "synthesize", {}, "m3")],
        [("text", "ECHO")],
    ))

    raw_answer = ("DASH lowers blood pressure. " * 60
                  + "\n\n## References\n\n[P99] Made-up entry")

    async def _synth_events():
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(
            content_delta="weighing the DASH evidence"))
        # token-style deltas, like the live model stream
        for i in range(0, len(raw_answer), 120):
            yield PartDeltaEvent(index=0, delta=TextPartDelta(
                content_delta=raw_answer[i:i + 120]))

    import src.agents.synthesizer as synth_mod
    monkeypatch.setattr(synth_mod, "build_agent",
                        lambda **kw: _ScriptedAgent(_synth_events(), raw_answer))

    events = await _collect(umls=SimpleNamespace(), warmup=False)
    by_type: dict = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)

    # thinking streams under the synthesizer\u2019s own section
    synth_thinking = [e.get("delta", "") for e in by_type.get("thinking", [])
                      if e.get("agent") == "synthesizer"]
    assert any("weighing the DASH evidence" in d for d in synth_thinking)

    # pipeline indicator + leg lifecycle are visible
    assert any(e.get("state") == "synthesizing"
               for e in by_type.get("status", []))
    tasks = {(e.get("id"), e.get("state")) for e in by_type.get("task", [])}
    assert ("synthesizer", "started") in tasks
    assert ("synthesizer", "done") in tasks

    # the answer streams live (long text trips several live flushes),
    # not as one end-of-run flash
    live = [e for e in by_type.get("answer", []) if "delta" in e]
    assert len(live) >= 2

    # the citation guard cut the model-written References (no ledger
    # refs): the corrected full text arrives via answer_replace
    replaced = [e for e in by_type.get("answer_replace", [])]
    assert len(replaced) == 1
    final = replaced[0]["delta"]
    assert "[P99]" not in final
    assert "References" not in final
    assert "DASH lowers blood pressure." in final
    assert events[-1]["type"] == "done"


def test_summarize_result_shrinks_evidence_json_structurally():
    # An oversized evidence payload must stay parseable: texts clip with a
    # marker, every row and the full verdict block survive (no mid-JSON cut
    # that would dump the side sheet to raw JSON).
    payload = json.dumps({
        "query": "q",
        "local": [{"chunk_id": "c1", "text": "x" * 6000}],
        "web": {"available": True,
                 "results": [{"url": "https://e.example/"}],
                 "pages": [], "trust": [], "dropped": 0},
    }) + "\n\n[EVIDENCE JUDGMENT]\n" + json.dumps(
        {"evidence_results": [{"passage_id": "c1", "intent_score": 0.9,
                               "coverage": ["E1"], "reason": "r"}]})
    assert len(payload) > 5500
    out, truncated = _summarize_result(payload, 5500)
    assert truncated is True
    assert len(out) <= 5500
    head, _, tail = out.partition("[EVIDENCE JUDGMENT]")
    assert tail and "evidence_results" in tail
    data = json.loads(head)
    assert data["local"][0]["chunk_id"] == "c1"
    assert "clipped for display" in data["local"][0]["text"]
    assert len(data["web"]["results"]) == 1


def test_summarize_result_slices_plain_text():
    out, truncated = _summarize_result("y" * 5000, 100)
    assert truncated is True
    assert out == "y" * 100
    out, truncated = _summarize_result("short", 100)
    assert truncated is False
    assert out == "short"


def _covered_ledger(req_evidence, task_id="T1", run_id="r1"):
    from types import SimpleNamespace

    from src.runstate.models import (
        EvidenceRecord,
        RequirementRecord,
        RunState,
        TaskRecord,
    )
    reqs = [
        RequirementRecord(
            id=rid, description=f"req {rid}",
            supporting_evidence=[
                EvidenceRecord(passage_id=f"{rid}-p{i}")
                for i in range(n)
            ],
        )
        for rid, n in req_evidence.items()
    ]
    task = TaskRecord(id=task_id, task="Q", depth="deep",
                      evidence_requirements=reqs)
    return SimpleNamespace(turns={run_id: RunState(
        run_id=run_id, question="Q", plan=[task])})


def test_task_coverage_complete_all_covered():
    from src.agents.stream_adapter import _task_coverage_complete

    ledger = _covered_ledger({"E1": 3, "E2": 1})
    assert _task_coverage_complete(ledger, "r1", "T1") is True


def test_task_coverage_complete_partial_is_incomplete():
    from src.agents.stream_adapter import _task_coverage_complete

    ledger = _covered_ledger({"E1": 3, "E2": 0})
    assert _task_coverage_complete(ledger, "r1", "T1") is False


def test_task_coverage_complete_empty_is_incomplete():
    from src.agents.stream_adapter import _task_coverage_complete

    ledger = _covered_ledger({"E1": 0})
    assert _task_coverage_complete(ledger, "r1", "T1") is False


def test_task_coverage_complete_no_requirements_is_incomplete():
    from src.agents.stream_adapter import _task_coverage_complete

    ledger = _covered_ledger({})
    assert _task_coverage_complete(ledger, "r1", "T1") is False


def test_task_coverage_complete_missing_everything_is_incomplete():
    from src.agents.stream_adapter import _task_coverage_complete

    ledger = _covered_ledger({"E1": 5})
    assert _task_coverage_complete(None, "r1", "T1") is False
    assert _task_coverage_complete(ledger, "nope", "T1") is False
    assert _task_coverage_complete(ledger, "r1", "TX") is False


async def _run_deep_leg_with_tiny_budget(monkeypatch, tmp_path, chat_id, run_id, seed):
    # One deep leg with a single tool-call budget that burns one stubbed
    # retrieval. When seed is True the stub also records judge-shaped
    # evidence for E1, so coverage completes despite the exhaustion.
    from types import SimpleNamespace

    from src.agents import deep_agent as deep_agent_mod
    from src.runstate.models import EvidenceRecord
    from src.runstate.store import get_store, record_evidence, set_store, RunStore

    from src.runstate.store import get_store as _get
    _orchestrator_env(monkeypatch)
    monkeypatch.setenv("MEDRAG_BUDGET_TASK_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("MEDRAG_BUDGET_ENABLED", "1")
    set_store(RunStore(tmp_path / "chats"))

    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("tool", "submit_plan",
          {"items": [{"id": "T1", "question": "Q", "depth": "deep",
                      "evidence_requirements": [{"id": "E1",
                                                 "description": "d"}]}]},
          "m1")],
        [("tool", "spawn_subagent",
          {"task": "Q", "depth": "deep", "task_id": "T1"}, "m2")],
        [("text", "DONE")],
    ))

    async def local_search(ctx: RunContext[DeepDeps], query: str = "",
                           evidence_requirements=None):
        if seed:
            await record_evidence(
                ctx, "local_search",
                [{"chunk_id": "seed-c1", "document_id": "SEED",
                  "text": "seed passage", "url": ""}],
                [SimpleNamespace(passage_id="seed-c1", coverage=["E1"],
                                 intent_score=0.9, reason="seeded")])
        return "[]"

    deep_brain = _scripted_brain(
        [("tool", "local_search",
          {"query": "q", "evidence_requirements": []}, "d1")],
        [("text", "findings")],
    )
    monkeypatch.setattr(deep_agent_mod, "make_model",
                        lambda *a, **k: deep_brain)
    monkeypatch.setattr(deep_agent_mod, "local_search",
                        local_search)
    events = await _collect(question="cov?", umls=SimpleNamespace(),
                            warmup=False, run_id=run_id,
                            conversation_id=chat_id)
    assert events[-1]["type"] == "done"
    return _get().get_chat(chat_id).turns[run_id].task("T1")


async def test_budget_exhausted_with_coverage_closes_done(monkeypatch, tmp_path):
    from src.runstate.models import TaskState
    from src.runstate.store import get_store, set_store

    old = get_store()
    try:
        task = await _run_deep_leg_with_tiny_budget(
            monkeypatch, tmp_path, "cov-chat-done", "cov-run-done", True)
    finally:
        set_store(old)
    # The budget really ran out (not a vacuous pass)...
    assert (task.budget_remaining or {}).get("tool_calls") == 0
    assert task.verified_count == 1
    # ...yet full coverage closes DONE, never a failure state.
    assert task.state == TaskState.DONE


async def test_budget_exhausted_without_coverage_stays_exhausted(monkeypatch, tmp_path):
    from src.runstate.models import TaskState
    from src.runstate.store import get_store, set_store

    old = get_store()
    try:
        task = await _run_deep_leg_with_tiny_budget(
            monkeypatch, tmp_path, "cov-chat-hungry", "cov-run-hungry", False)
    finally:
        set_store(old)
    assert (task.budget_remaining or {}).get("tool_calls") == 0
    assert task.verified_count == 0
    assert task.state == TaskState.BUDGET_EXHAUSTED


async def test_ask_user_parks_stream_and_resumes(monkeypatch):
    """The orchestrator calls ask_user -> a question event streams, the
    run parks; resolving via the broker resumes it to completion."""
    import asyncio
    import uuid as _uuid
    from types import SimpleNamespace

    from src.agents import orchestrator as orchestrator_mod
    from src.agents.clarify import has_pending, submit_answer

    chat_id = "hitl-" + _uuid.uuid4().hex[:10]
    _orchestrator_env(monkeypatch)
    _patch_orchestrator_model(monkeypatch, _scripted_brain(
        [("tool", "ask_user",
          {"questions": [
              {"id": "q1", "text": "How does diet affect hypertension?",
               "options": ["Physiological mechanisms",
                           "Named diets (DASH, sodium)",
                           "Management recommendations"]},
              {"id": "q2", "text": "Sources?",
               "options": ["Local corpus", "Web plus local"],
               "multi_select": False},
          ]}, "m1")],
        [("text", "FINAL")],
    ))

    collect = asyncio.create_task(_collect(
        umls=SimpleNamespace(), warmup=False, conversation_id=chat_id))
    # Wait until the tool has parked each question, then deliver answers.
    for _ in range(200):
        if has_pending(chat_id, "q1") and has_pending(chat_id, "q2"):
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("ask_user never parked")
    assert submit_answer(chat_id, "q1",
                         {"selections": ["Named diets (DASH, sodium)"],
                          "other": "", "find_all": False})
    assert submit_answer(chat_id, "q2",
                         {"selections": ["Web plus local"],
                          "other": "", "find_all": False})

    events = await asyncio.wait_for(collect, timeout=20)
    questions = [e for e in events if e["type"] == "question"]
    assert questions, "expected a question event"
    assert [q["id"] for q in questions[0]["questions"]] == ["q1", "q2"]
    assert questions[0]["questions"][0]["multi_select"] is True
    assert questions[0]["questions"][1]["multi_select"] is False
    # the parked run resumed and finished normally
    assert events[-1]["type"] == "done"
    answers = [e for e in events if e["type"] == "answer"]
    assert answers and answers[0]["delta"] == "FINAL"

