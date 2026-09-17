"""Console streaming for agent runs.

Streams one run's thoughts inside |thinking|...|thinking|, its final text
inside |answer|...|answer|, and tool calls/results as [label] lines.
Shareable by any agent: pass the built handler as run_stream's
event_stream_handler.
"""

from __future__ import annotations

from collections.abc import AsyncIterable

from pydantic_ai import RunContext
from src.lib import narrate
from src.lib.pretty import rule, style, summarize_tool_text
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    OutputToolCallEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

# Model name of the structured-output ("final answer") tool call. Verdict
# JSON from a judge agent streams as deltas on this part.
_OUTPUT_TOOL_NAMES = frozenset({"final_result"})


def tool_call_line(label: str, name: str, args: object) -> str:
    """One narration line for a tool call (terminal and UI share it)."""
    return f"⚙ [{label}] calling tool {name} with {summarize_tool_text(args)}"


def tool_result_line(label: str, content: object) -> str:
    """One narration line for a tool result (terminal and UI share it)."""
    return f"✓ [{label}] tool result: {summarize_tool_text(content)}"


def verdict_line(label: str, args: object) -> str:
    """One narration line for a structured verdict call (both share it)."""
    return f"⚖ [{label}] verdict: {summarize_tool_text(args)}"


def console_stream(label: str, *, echo_tool_args: bool = True, echo_text: bool = True):
    """Build an event handler that narrates one labeled run to stdout.

    ``echo_tool_args=False`` suppresses the raw tool-call argument JSON and
    ``echo_text=False`` suppresses raw text deltas (useful when the caller
    parses the token stream itself, e.g. the evidence judge printing parsed
    verdicts instead of token soup).
    """

    open_block: str | None = None

    def _close() -> None:
        nonlocal open_block
        if open_block is not None:
            print(style(f"|{open_block}|", "2"), flush=True)
            open_block = None

    def _open(kind: str) -> None:
        nonlocal open_block
        if open_block != kind:
            _close()
            marker = f"|{kind}|"
            if kind == "thinking":
                print(style(marker, "2"), flush=True)
            else:
                print(style(f"── {marker}", "1", "32"), flush=True)
            open_block = kind

    announced = False
    # Tool-call part index -> model tool name, so arg deltas can be labeled:
    # the structured-output ("verdict") JSON vs. ordinary function-call args.
    tool_names: dict[int, str] = {}

    async def _handle(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]) -> None:
        nonlocal announced
        if not announced:
            print(style(rule(f"live · {label}"), "2"), flush=True)
            announced = True
        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                _close()
                narrate.say_folded(
                    f"{style('⚙', '36')} [{label}] calling tool "
                    f"{style(event.part.tool_name, '1')} with ",
                    summarize_tool_text(event.part.args),
                )
            elif isinstance(event, FunctionToolResultEvent):
                narrate.say_folded(
                    f"{style('✓', '32')} [{label}] tool result: ",
                    summarize_tool_text(event.part.content),
                )
            elif isinstance(event, OutputToolCallEvent):
                _close()
                narrate.say_folded(
                    f"{style('⚖', '33')} [{label}] verdict: ",
                    summarize_tool_text(event.part.args),
                )
            elif isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
                if event.part.tool_name:
                    tool_names[event.index] = event.part.tool_name
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta):
                if event.delta.tool_name_delta:
                    tool_names[event.index] = (
                        tool_names.get(event.index, "") + event.delta.tool_name_delta
                    )
                # Live-stream tool-call argument JSON: this is the only place
                # the verifier's verdict tokens appear — a structured-output
                # judge emits no text parts, so without this its JSON arrives
                # silently and only the final summary line would ever print.
                if event.delta.args_delta and echo_tool_args:
                    chunk = event.delta.args_delta
                    chunk = chunk if isinstance(chunk, str) else str(chunk)
                    if chunk:
                        kind = (
                            "verdict"
                            if tool_names.get(event.index, "") in _OUTPUT_TOOL_NAMES
                            else "tool-args"
                        )
                        _open(kind)
                        narrate.say_chunk(chunk)
            elif isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                _open("thinking")
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                _open("thinking")
                if event.delta.content_delta:
                    narrate.say_chunk(event.delta.content_delta)
            elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                if not echo_text:
                    continue
                _open("answer")
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                if not echo_text:
                    continue
                _open("answer")
                narrate.say_chunk(event.delta.content_delta)
            elif isinstance(event, (PartEndEvent, FinalResultEvent)):
                _close()
        _close()

    return _handle
