"""Two LLM lanes for the whole backend — nothing else.

Shared OpenAI-compatible endpoint plus one model id per lane:

- ``AGENT_MODEL`` — the big thinking agentic LLM (deep agent, task
  planner, synthesis).
- ``SMALL_MODEL`` — small tasks (evidence planner, verifier judge,
  trust checker).

Endpoint (shared by both lanes):

- ``LLM_BASE_URL`` / ``LLM_API_KEY``

The provider caches prompts per session, so the backend sends ONE fixed
session id on every call — master, sub-agents, verifier, every judge — and the
handful of shared system prompts is cached once and reused everywhere.
``LLM_SESSION_ID`` pins it; unset falls back to a stable per-process id so the
gateway always sees a session.

Every variable is required; a missing one raises a RuntimeError naming
it instead of a bare KeyError deep in a builder.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIResponsesModel,
)
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

BASE_URL_VAR = "LLM_BASE_URL"
API_KEY_VAR = "LLM_API_KEY"
AGENT_MODEL_VAR = "AGENT_MODEL"
SMALL_MODEL_VAR = "SMALL_MODEL"
SESSION_VAR = "LLM_SESSION_ID"
# Legacy alias, still honored so existing deployments keep working.
_LEGACY_SESSION_VAR = "OPENCODE_SESSION"

# OpenCode Go (https://opencode.ai/zen/go) rejects requests that do not
# carry a stable session id (HTTP 400 MissingSessionID) because it routes
# traffic and caches prompts per session. Only opencode endpoints get the
# header (plus any endpoint when a session id is configured explicitly).
OPENCODE_HOST = "opencode.ai"

# Stable for the lifetime of the process: the session every call uses when
# LLM_SESSION_ID is unset, so the gateway still routes/caches.
_PROCESS_SESSION = uuid.uuid4().hex


def _configured_session() -> str:
    """Explicit ``LLM_SESSION_ID`` (or legacy ``OPENCODE_SESSION``), else ""."""
    return (
        os.environ.get(SESSION_VAR, "").strip()
        or os.environ.get(_LEGACY_SESSION_VAR, "").strip()
    )


def session_id() -> str:
    """The one session id every LLM call carries.

    ``LLM_SESSION_ID`` pins it; otherwise a stable per-process id keeps the
    gateway happy and the prompt cache warm inside the process.
    """
    return _configured_session() or _PROCESS_SESSION


# Model-class selection per lane: OpenCode Go serves most models through the
# OpenAI Chat Completions API, but Muse Spark is served only through the
# OpenAI Responses API (/responses). OPENAI_API_STYLE=responses|chat forces
# either; "auto" (default) picks responses for muse-spark models and chat
# for everything else.
API_STYLE_VAR = "OPENAI_API_STYLE"


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str
    model: str


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"LLM lane not configured; set {name} — see backend/.env.example"
        )
    return value


def agent_endpoint() -> Endpoint:
    """The big thinking agentic LLM."""
    return Endpoint(
        base_url=_required(BASE_URL_VAR),
        api_key=_required(API_KEY_VAR),
        model=_required(AGENT_MODEL_VAR),
    )


def small_endpoint() -> Endpoint:
    """The small-task LLM (judges, planners of requirements)."""
    return Endpoint(
        base_url=_required(BASE_URL_VAR),
        api_key=_required(API_KEY_VAR),
        model=_required(SMALL_MODEL_VAR),
    )


def _opencode_session_header(endpoint: Endpoint) -> dict[str, str]:
    # x-opencode-session header required by OpenCode Go's gateway
    # (Console Go answers HTTP 400 MissingSessionID without one). One id for
    # every call in the process — master, sub-agents, verifier, judges — so
    # the shared system prompts are cached once and reused everywhere.
    # Non-opencode endpoints carry the header only when configured.
    if OPENCODE_HOST not in endpoint.base_url.lower():
        configured = _configured_session()
        return {"x-opencode-session": configured} if configured else {}
    return {"x-opencode-session": session_id()}


def _api_style(endpoint: Endpoint) -> str:
    # "responses" or "chat" for this lane. OPENAI_API_STYLE=responses|chat
    # forces one; "auto" (default) picks the OpenAI Responses API for
    # muse-spark models (OpenCode Go serves Muse Spark only through
    # /responses) and Chat Completions otherwise.
    forced = os.environ.get(API_STYLE_VAR, "auto").strip().lower()
    if forced in ("responses", "chat", "completions"):
        return "responses" if forced == "responses" else "chat"
    if "muse-spark" in endpoint.model.lower():
        return "responses"
    return "chat"


# Reasoning effort for every non-judge lane. The orchestrator, planner,
# research legs, shallow agent and synthesizer all build through make_model;
# left unset the endpoint picks its own (slow) default. The judge pins its
# own lever explicitly (verifier._verifier_settings), so this default never
# overrides it. AGENT_REASONING_EFFORT=default/provider/auto/off opts out.
REASONING_EFFORT_VAR = "AGENT_REASONING_EFFORT"
DEFAULT_REASONING_EFFORT = "low"

_UNSET = object()


def default_reasoning_effort() -> str:
    """Resolved agent-lane reasoning effort; "" means send no field."""
    raw = os.environ.get(REASONING_EFFORT_VAR, DEFAULT_REASONING_EFFORT)
    effort = (raw or "").strip().lower()
    return "" if effort in ("", "default", "provider", "auto", "off") else effort


def _lane_settings(settings) -> dict:
    """Merge lane defaults into a caller's ModelSettings (never mutates it).

    Temperature defaults to 0.0 on every lane. Reasoning effort defaults to
    AGENT_REASONING_EFFORT (low) unless the caller names the key explicitly;
    an explicit falsy value opts out and sends no reasoning field at all,
    which is how the verifier keeps its own lever.
    """
    lane = dict(settings or {})
    lane.setdefault("temperature", 0.0)
    explicit = lane.get("openai_reasoning_effort", _UNSET)
    if explicit is _UNSET:
        effort = default_reasoning_effort()
        if effort:
            lane["openai_reasoning_effort"] = effort
    elif not explicit:
        lane.pop("openai_reasoning_effort", None)
    return lane


def make_model(endpoint: Endpoint, **settings):
    # Build the lane model for an endpoint.
    # Temperature defaults to 0.0 (greedy/deterministic) and reasoning
    # effort to AGENT_REASONING_EFFORT (low) on every lane; an explicit
    # settings= override still wins. OpenCode Go endpoints get the required
    # x-opencode-session header (one process-wide session), and muse-spark
    # lanes speak the OpenAI Responses API (OpenAIChatModel would hit
    # /chat/completions and be rejected by the gateway).
    settings["settings"] = _lane_settings(settings.get("settings"))
    headers = _opencode_session_header(endpoint)
    if headers and endpoint.api_key:
        from openai import AsyncOpenAI

        provider = OpenAIProvider(openai_client=AsyncOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            default_headers=headers,
        ))
    else:
        provider = OpenAIProvider(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
        )
    if _api_style(endpoint) == "responses":
        return OpenAIResponsesModel(endpoint.model, provider=provider,
                                    **settings)
    return OpenAIChatModel(endpoint.model, provider=provider, **settings)


def pool_for(endpoint: Endpoint):
    """Shared rate-limit pool for an endpoint (lazy import: no cycle)."""
    from src.llm.pool import get_pool

    return get_pool(endpoint.base_url, endpoint.api_key)
