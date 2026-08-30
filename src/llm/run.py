"""Single choke point every agent uses to talk to the LLM.

Responsibilities:
  - charge the shared rate-limit bucket BEFORE each request,
  - try a REAL structured run first (PydanticAI validates + retries against
    the schema), then fall back to text-mode + manual JSON decoding only when
    the provider cannot do structured output reliably (e.g. Gemma behind an
    OpenAI-compatible endpoint without function calling),
  - wrap attempts in 429-aware retries honoring Retry-After,
  - emit the same trace events the old per-agent code emitted,
  - (optional) notify an LLM observer with {label, input, thought, output} so
    the agentic v2 UI can render the prompt, the model's <think>/<thought>
    reasoning, and the parsed response per node.

Fallback order per attempt:
  1. ``agent.run(prompt, output_type=<Model>)``   -> validated object
  2. ``agent.run(prompt, output_type=str)``       -> strip <think> blocks ->
     ``decode_structured``                          (+ one compaction nudge
                                                      retry on parse failure)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from src.lib import COMPACTION_NUDGE, decode_structured, strip_think
from src.llm.ratelimit import estimate_tokens, get_bucket, is_rate_limit_error, run_with_retry

logger = logging.getLogger("src.llm.run")

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Optional LLM observer (used by the agentic v2 UI to capture prompt/thought/
# response per node). Defaults to None; set via set_llm_observer.
# ---------------------------------------------------------------------------

_llm_observer: Optional[Callable[..., None]] = None

_THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)


def set_llm_observer(observer: Optional[Callable[..., None]]) -> None:
    """Install a callable invoked as ``observer(label=, input=, thought=, output=)``."""
    global _llm_observer
    _llm_observer = observer


def reset_llm_observer() -> None:
    global _llm_observer
    _llm_observer = None


def _extract_thought(raw: str) -> str:
    if not raw:
        return ""
    chunks = _THINK_RE.findall(raw) + _THOUGHT_RE.findall(raw)
    return "\n\n".join(t.strip() for t in chunks if t and t.strip())


def _raw_response_text(result: Any) -> str:
    """Best-effort raw model text from a pydantic_ai AgentRunResult."""
    try:
        parts = []
        for msg in result.all_messages():
            if type(msg).__name__ == "ModelResponse":
                for part in getattr(msg, "parts", []):
                    if type(part).__name__ == "TextPart":
                        parts.append(getattr(part, "content", "") or "")
        return "".join(parts)
    except Exception:
        return ""


def _notify_observer(label: str, prompt: str, raw: str, output_obj: Any,
                       phase: str = "end") -> None:
    """Notify the observer that an LLM call FINISHED (phase="end")."""
    if _llm_observer is None:
        return
    raw_text = raw or ""
    thought = _extract_thought(raw_text)
    if isinstance(output_obj, BaseModel):
        try:
            out: Any = output_obj.model_dump(mode="json")
        except Exception:
            out = str(output_obj)
    else:
        out = output_obj
    try:
        _llm_observer(label=label, input=prompt, thought=thought, output=out,
                      phase=phase)
    except Exception:  # observer must never break the request path
        logger.debug("llm observer failed", exc_info=True)


def _notify_observer_start(label: str, prompt: str) -> None:
    """Notify the observer that an LLM call STARTED (timing + live spinner)."""
    if _llm_observer is None:
        return
    try:
        _llm_observer(label=label, input=prompt, thought=None, output=None,
                      phase="start")
    except Exception:
        logger.debug("llm observer start failed", exc_info=True)


async def _text_fallback(
    agent: Any,
    prompt: str,
    output_type: Type[T],
    settings: dict,
    label: str,
    fallback_parser: Optional[Any] = None,
) -> T:
    """Text-mode request parsed manually; one nudge retry on bad JSON."""
    result = await agent.run(prompt, output_type=str, model_settings=settings)
    raw_text = str(result.output)                      # includes any <think> block
    raw = strip_think(raw_text)
    parse = fallback_parser or (lambda text: decode_structured(text, output_type))
    try:
        parsed = parse(raw)
        _notify_observer(label, prompt, raw_text, parsed)
        return parsed
    except Exception as exc:
        logger.warning("[%s] structured/text parse failed (%s); nudging", label, exc)
        result2 = await agent.run(prompt + COMPACTION_NUDGE, output_type=str, model_settings=settings)
        raw_text2 = str(result2.output)
        raw2 = strip_think(raw_text2)
        try:
            parsed2 = parse(raw2)
            _notify_observer(label, prompt + COMPACTION_NUDGE, raw_text2, parsed2)
            return parsed2
        except Exception as exc2:
            raise ValueError(f"[{label}] unparseable output: {exc2}") from exc2


def _agent_model_name(agent: Any) -> str:
    """Best-effort model name of a pydantic-ai Agent ('' when unavailable)."""
    try:
        m = getattr(agent, "model", None)
        if m is None:
            return ""
        for attr in ("model_name", "model", "name"):
            v = getattr(m, attr, None)
            if v:
                return str(v)
    except Exception:
        pass
    return ""


async def ask_structured(
    agent: Any,
    prompt: str,
    output_type: Type[T],
    *,
    label: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 4,
    allow_text_fallback: bool = True,
    fallback_parser: Optional[Any] = None,
) -> T:
    """Run ``prompt`` through ``agent`` and return a validated ``output_type``.

    Logfire wrapper: ensures the observability bridge is configured, wraps the
    call in an ``llm.ask_structured`` span (the PydanticAI GenAI spans for the
    model requests nest inside), and records LLM latency / prompt-token /
    call-count metrics. When Logfire is disabled this is a thin passthrough.
    """
    from src import logfire_obs as lf

    lf.ensure_configured()
    t0 = time.monotonic()
    out_type = getattr(output_type, "__name__", str(output_type))
    model_name = _agent_model_name(agent)
    if lf.enabled():
        with lf.llm_span(label=label, output_type=out_type, model=model_name):
            try:
                out = await _ask_structured_core(
                    agent, prompt, output_type, label=label, max_tokens=max_tokens,
                    temperature=temperature, max_attempts=max_attempts,
                    allow_text_fallback=allow_text_fallback,
                    fallback_parser=fallback_parser,
                )
            except Exception:
                lf.record_llm(label, latency=time.monotonic() - t0,
                              prompt_tokens=estimate_tokens(prompt), success=False,
                              model=model_name)
                raise
        lf.record_llm(label, latency=time.monotonic() - t0,
                      prompt_tokens=estimate_tokens(prompt),
                      output_chars=_output_chars(out), success=True,
                      model=model_name)
        return out
    return await _ask_structured_core(
        agent, prompt, output_type, label=label, max_tokens=max_tokens,
        temperature=temperature, max_attempts=max_attempts,
        allow_text_fallback=allow_text_fallback,
        fallback_parser=fallback_parser,
    )


def _output_chars(output: Any) -> int:
    """Response size in characters for the output-chars metric."""
    try:
        if isinstance(output, str):
            return len(output)
        return len(json.dumps(output, default=str))
    except Exception:
        return 0


async def _ask_structured_core(
    agent: Any,
    prompt: str,
    output_type: Type[T],
    *,
    label: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_attempts: int = 4,
    allow_text_fallback: bool = True,
    fallback_parser: Optional[Any] = None,
) -> T:
    """The actual request path (rate limit, structured run, text fallback)."""
    from src.trace import get_trace

    trace = get_trace()
    bucket = get_bucket()
    settings = {"temperature": temperature, "max_tokens": max_tokens}

    trace.agent(label.split(":")[0], output_type=getattr(output_type, "__name__", str(output_type)),
                meta={"label": label, "model": _agent_model_name(agent)})
    trace.prompt(f"{label}_user_prompt", prompt)

    # Fire the START observer event before the first attempt so the UI can
    # show the spinner / timing span immediately (end fires on completion).
    _notify_observer_start(label, prompt)

    await bucket.acquire(estimate_tokens(prompt))

    async def _structured_once() -> T:
        result = await agent.run(prompt, output_type=output_type, model_settings=settings)
        _notify_observer(label, prompt, _raw_response_text(result), result.output)
        return result.output

    async def _guarded_structured() -> T:
        # Charge again for each retry attempt triggered by run_with_retry.
        await bucket.acquire(estimate_tokens(prompt))
        return await _structured_once()

    try:
        output = await run_with_retry(
            _guarded_structured, max_attempts=max_attempts, bucket=bucket
        )
        trace.parsed(getattr(output_type, "__name__", "output"), output.model_dump(mode="json"))
        return output
    except Exception as exc:
        if not allow_text_fallback:
            raise
        if is_rate_limit_error(exc):
            # A rate limit is not a parsing problem: the text fallback would hit
            # the same 429 and just burn another full retry round (doubling the
            # wall-clock wait before the caller can fail/move on). Surface it
            # now; the bucket is already penalized and callers handle retry.
            logger.warning("[%s] rate limited after retries; not falling back (%s)",
                           label, str(exc)[:120])
            raise
        logger.info("[%s] structured run failed (%s); using text fallback", label, exc)

    # Fallback path (also 429-aware). ``spent`` already accounted attempt 1.
    async def _fallback_once() -> T:
        await bucket.acquire(estimate_tokens(prompt))
        return await _text_fallback(agent, prompt, output_type, settings, label, fallback_parser)

    output = await run_with_retry(_fallback_once, max_attempts=max_attempts, bucket=bucket)
    trace.parsed(getattr(output_type, "__name__", "output"), output.model_dump(mode="json"))
    return output


__all__ = ["ask_structured", "set_llm_observer", "reset_llm_observer"]
