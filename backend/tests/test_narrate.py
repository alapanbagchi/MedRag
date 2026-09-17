"""Narration tests: terminal fan-out, and the run thought stream staying
model-reasoning-only (narrated backend lines print to the terminal but
never enter ``thinking``)."""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.stream_adapter import stream_deep_agent
from src.lib import narrate
from src.lib.streaming import tool_call_line, tool_result_line, verdict_line


def test_say_prints_without_sink(capsys):
    narrate.say("hello terminal")
    assert capsys.readouterr().out == "hello terminal\n"


def test_say_forwards_stripped_line(capsys):
    seen: list[str] = []
    with narrate.forwarding(seen.append):
        narrate.say("\x1b[36m⚙\x1b[0m [deep] calling tool x with {}")
    out = capsys.readouterr().out
    assert "[deep] calling tool x with {}" in out  # terminal keeps color
    assert seen == ["⚙ [deep] calling tool x with {}\n"]  # UI gets plain


def test_say_folded_terminal_folded_forward_single_line(capsys):
    import shutil
    seen: list[str] = []
    cols = shutil.get_terminal_size(fallback=(100, 24)).columns
    long_text = "word " * (cols)  # forces a terminal-width fold
    with narrate.forwarding(seen.append):
        narrate.say_folded("prefix: ", long_text)
    out = capsys.readouterr().out
    assert out.count("\n") > 1  # terminal folded to its width
    assert len(seen) == 1 and seen[0].count("\n") == 1  # UI: one line
    assert seen[0] == f"prefix: {' '.join(long_text.split())}\n"


def test_say_forward_false_stays_terminal_only(capsys):
    seen: list[str] = []
    with narrate.forwarding(seen.append):
        narrate.say("FULL DUMP", forward=False)
        narrate.say_folded("pre: ", "body", forward=False)
        narrate.say("thought")
    out = capsys.readouterr().out
    assert "FULL DUMP" in out and "pre: body" in out.replace("\n", " ")
    assert seen == ["thought\n"]


def test_say_chunk_forwards_raw_without_newline(capsys):
    seen: list[str] = []
    with narrate.forwarding(seen.append):
        narrate.say_chunk("hel")
        narrate.say_chunk("lo")
    assert capsys.readouterr().out == "hello"
    assert seen == ["hel", "lo"]


def test_install_uninstall_token():
    token = narrate.install(lambda line: None)
    assert narrate.current_sink() is not None
    narrate.uninstall(token)
    assert narrate.current_sink() is None


def test_shared_builders_match_console_format():
    assert tool_call_line("deep", "retrieve_evidence", {"q": "X"}) == (
        "⚙ [deep] calling tool retrieve_evidence with {'q': 'X'}")
    assert tool_result_line("P1", "[1, 2]") == "✓ [P1] tool result: [1, 2]"
    assert verdict_line("deep", {"a": 1}) == "⚖ [deep] verdict: {'a': 1}"


async def test_run_fn_narration_stays_terminal_only(capsys):
    async def fake_run(question, emit):
        await emit("thinking", delta="t1. ", done=False)
        narrate.say("n1")
        await emit("thinking", delta="t2.", done=False)
        narrate.say("n2")
        return "done"

    events = [e async for e in stream_deep_agent("q?", run_fn=fake_run)]
    thoughts = "".join(e.get("delta", "") for e in events
                       if e.get("type") == "thinking" and not e.get("done"))
    assert thoughts == "t1. t2."
    term = capsys.readouterr().out
    assert term == "n1\nn2\n"  # terminal still got the narrated lines


async def test_handler_tool_lines_are_terminal_only(capsys):
    """Tool lines print to the console but never enter the thought stream."""
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        ToolCallPart,
        ToolReturnPart,
    )

    call_part = ToolCallPart(tool_name="lookup_medical_term",
                             args={"term": "x"}, tool_call_id="tc_1")
    ret_part = ToolReturnPart(tool_name="lookup_medical_term",
                              content='{"found": false}',
                              tool_call_id="tc_1")

    async def _events():
        yield FunctionToolCallEvent(part=call_part)
        yield FunctionToolResultEvent(part=ret_part)

    class FakeResult:
        def __init__(self, handler):
            self._handler = handler
            self.output = "final"

        async def drive(self):
            await self._handler(None, _events())
            return self

        def usage(self):
            return SimpleNamespace(request_tokens=1, response_tokens=1)

    class FakeAgent:
        async def run(self, prompt, *args, **kwargs):
            return await FakeResult(
                kwargs.get("event_stream_handler")).drive()

    events = [e async for e in stream_deep_agent(
        "q?", orchestrator=FakeAgent(), umls=SimpleNamespace(), warmup=False)]
    thoughts = "".join(e.get("delta", "") for e in events
                       if e.get("type") == "thinking" and not e.get("done"))
    call_line = "⚙ [deep] calling tool lookup_medical_term with {'term': 'x'}\n"
    result_line = '✓ [deep] tool result: {"found": false}\n'
    assert call_line not in thoughts and result_line not in thoughts
    term = capsys.readouterr().out
    assert call_line in term and result_line in term
    assert term.index(call_line) < term.index(result_line)


async def test_console_stream_forwards_thinking_suppresses_text(capsys):
    from pydantic_ai.messages import (
        PartDeltaEvent,
        TextPartDelta,
        ThinkingPartDelta,
    )
    from src.lib.streaming import console_stream

    seen: list[str] = []
    handler = console_stream("judge:t", echo_tool_args=False, echo_text=False)

    async def _feed():
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="hmm "))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="SECRET-JSON"))

    with narrate.forwarding(seen.append):
        await handler(None, _feed())
    out = capsys.readouterr().out
    assert "hmm " in out and "SECRET-JSON" not in out  # terminal gated
    assert "".join(seen) == "hmm "  # UI gets thinking only, nothing else
