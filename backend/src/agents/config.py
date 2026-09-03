"""xdeep config: provider + model ROUTING for deepagents.

Routing contract (agreed): QWEN (Alibaba MaaS, OpenAI-compatible) for
thinking/long tasks + MISTRAL (paced) for short verification decisions.

  * THINK = the "good agent" - QWEN for thinking + long-running tasks
    (decompose, research agents, synthesis, conflict reasoning).
    Provider is auto-selected:
      - QWEN when QWEN_API_KEY / QWEN_BASE_URL are set, or the shared
        GENERAL_LLM_BASE_URL points at Alibaba MaaS (aliyuncs.com /
        dashscope / maas) - this is the DEFAULT for this deployment. Model
        defaults to qwen3.8-flash (override via QWEN_MODEL or
        GENERAL_LLM_MODEL).
      - GROQ when GROQ_API_KEY is present (fast alternative; model defaults
        to Gemma2 9B via GROQ_MODEL).
      - else the shared GENERAL_LLM_* endpoint (Gemini-compatible), with a
        Gemma fallback model so the shared AppConfig flash default is never
        used.
    Force a choice with XDEEP_THINK_PROVIDER = qwen | alibaba | aliyun |
    groq | google.
  * MISTRAL = the "hammerable" agent - short decisions with a hard pace of
    one request per 1.5s (verification, contradiction checks, resolution
    decisions). Pacing is enforced by PacedChatOpenAI.

Provider/credentials come straight from the shared src.config.AppConfig
(backend/.env).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from src.agents.reuse import AppConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MISTRAL_MIN_INTERVAL_S = 1.5     # one Mistral request per 1.5 seconds
THINK_ROLES = frozenset({"decompose", "research", "synthesis", "conflict"})
MISTRAL_ROLES = frozenset({"verify", "contradiction", "resolution"})
_KNOWN_ROLES = THINK_ROLES | MISTRAL_ROLES

SEARXNG_URL_DEFAULT = "http://127.0.0.1:8888"
SEARXNG_TIMEOUT_S_DEFAULT = 15.0
SEARXNG_TOP_K_DEFAULT = 8

GROQ_BASE_URL_DEFAULT = "https://api.groq.com/openai/v1"
GROQ_MODEL_DEFAULT = "gemma2-9b-it"   # Groq hosts Gemma 2 9B
GOOGLE_MODEL_FALLBACK = "gemma-4-31b-it"  # only when forced to google

# Qwen (Alibaba MaaS): the primary thinking provider of this deployment.
# No hard base-url default - it follows GENERAL_LLM_BASE_URL (which points at
# Alibaba MaaS in .env); if QWEN_BASE_URL is set it wins.
QWEN_MODEL_FALLBACK = "qwen3.8-flash"
_ALIBABA_MARKERS = ("aliyuncs.com", "dashscope", "maas.aliyuncs", ".maas.")
_ALIAS_TO_QWEN = frozenset({"qwen", "alibaba", "aliyun", "aliyuncs", "dashscope"})


def get_config() -> Any:
    """Shared AppConfig (provider, UMLS, pgvector, budgets)."""
    return AppConfig()


def think_provider() -> str:
    """Which provider serves the 'good agent' (thinking) side.

    Resolution order:
      1. XDEEP_THINK_PROVIDER forced: qwen | alibaba | aliyun | groq | google | opencode
         (the Alibaba aliases all normalize to "qwen").
      2. QWEN_API_KEY / QWEN_BASE_URL set -> qwen.
      3. GROQ_API_KEY set -> groq (fast, no Alibaba quota).
      4. GENERAL_LLM_BASE_URL points at Alibaba MaaS -> qwen (the deployment
         default; the shared .env sets it).
      5. LLM_PROVIDER=opencode -> opencode (Muse Spark).
      6. else -> google (shared GENERAL_LLM_* Gemini-compatible endpoint).
    """
    forced = os.environ.get("XDEEP_THINK_PROVIDER", "").strip().lower()
    if forced:
        if forced in _ALIAS_TO_QWEN:
            return "qwen"
        if forced in ("groq", "google", "opencode"):
            return forced
        raise ValueError(
            f"XDEEP_THINK_PROVIDER={forced!r} must be one of "
            "'qwen|alibaba|aliyun|groq|google|opencode'"
        )
    if os.environ.get("QWEN_API_KEY", "").strip() or os.environ.get("QWEN_BASE_URL", "").strip():
        return "qwen"
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    base = (os.environ.get("GENERAL_LLM_BASE_URL", "") or "").lower()
    if any(m in base for m in _ALIBABA_MARKERS):
        return "qwen"
    # Check if LLM_PROVIDER is opencode (Muse Spark)
    llm_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if llm_provider == "opencode":
        return "opencode"
    return "google"


def role_model_name(role: str) -> str:
    """Which side serves a role: 'think' (thinking/long) or 'mistral' (paced)."""
    role = (role or "").strip().lower()
    if role not in _KNOWN_ROLES:
        raise ValueError(f"unknown role: {role!r}; known: {sorted(_KNOWN_ROLES)}")
    return "think" if role in THINK_ROLES else "mistral"


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _groq_endpoint():
    """(base_url, api_key, model) for the Groq 'good agent' endpoint.

    Groq is OpenAI-compatible and FAST - no free-tier input quota issues.
    """
    return (
        os.environ.get("GROQ_BASE_URL", GROQ_BASE_URL_DEFAULT).rstrip("/"),
        os.environ.get("GROQ_API_KEY", "") or "dummy",
        os.environ.get("GROQ_MODEL", GROQ_MODEL_DEFAULT),
    )


def _qwen_endpoint(cfg: Any):
    """(base_url, api_key, model) for the Qwen (Alibaba MaaS) think endpoint.

    QWEN_* wins; GENERAL_LLM_* is the shared fallback the deployment .env
    actually uses. The model is resolved from env with a Qwen fallback - never
    the shared AppConfig gemini_model default (gemini-2.0-flash).
    """
    model = (
        os.environ.get("QWEN_MODEL")
        or os.environ.get("GENERAL_LLM_MODEL")
        or QWEN_MODEL_FALLBACK
    )
    base = (
        os.environ.get("QWEN_BASE_URL")
        or cfg.gemini_base_url  # GENERAL_LLM_BASE_URL in the shared config
        or ""
    )
    key = (
        os.environ.get("QWEN_API_KEY")
        or os.environ.get("GENERAL_LLM_API_KEY")
        or cfg.gemini_api_key
        or "dummy"
    )
    return (base.rstrip("/"), key, model)


def _opencode_endpoint(cfg: Any):
    """(base_url, api_key, model) for the OpenCode (Muse Spark) think endpoint.

    OPENCODE_* wins; GENERAL_LLM_* is the shared fallback. The model
    defaults to muse-spark-1.3-contributor-free.
    """
    model = (
        os.environ.get("OPENCODE_MODEL")
        or os.environ.get("GENERAL_LLM_MODEL")
        or "muse-spark-1.3-contributor-free"
    )
    base = (
        os.environ.get("OPENCODE_BASE_URL")
        or cfg.gemini_base_url  # GENERAL_LLM_BASE_URL in the shared config
        or ""
    )
    key = (
        os.environ.get("OPENCODE_API_KEY")
        or os.environ.get("GENERAL_LLM_API_KEY")
        or cfg.gemini_api_key
        or "dummy"
    )
    return (base.rstrip("/"), key, model)


def _google_endpoint(cfg: Any):
    """(base_url, api_key, model) for the Google 'good agent' endpoint.

    Uses the shared GENERAL_LLM_* settings (Gemini OpenAI-compatible) but
    resolves the MODEL from env with a GEMMA fallback - the shared
    AppConfig defaults gemini_model to "gemini-2.0-flash", which is never
    acceptable here.
    """
    model = (
        os.environ.get("GENERAL_LLM_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or GOOGLE_MODEL_FALLBACK
    )
    return (
        cfg.gemini_base_url or "",
        cfg.gemini_api_key or "dummy",
        model,
    )


def _think_endpoint(cfg: Any):
    """(base_url, api_key, model) for the active thinking-provider."""
    provider = think_provider()
    if provider == "groq":
        return _groq_endpoint()
    if provider == "qwen":
        return _qwen_endpoint(cfg)
    if provider == "opencode":
        return _opencode_endpoint(cfg)
    return _google_endpoint(cfg)


def _mistral_endpoint(cfg: Any):
    """(base_url, api_key, model) for the Mistral 'hammerable' agent."""
    return (
        cfg.mistral_base_url or "",
        cfg.mistral_api_key or "dummy",
        cfg.mistral_model or "mistral-medium-latest",
    )


# ---------------------------------------------------------------------------
# Searxng web-search config
# ---------------------------------------------------------------------------

def searxng_url() -> str:
    """Searxng instance base URL.

    Env: XDEEP_SEARXNG_URL or SEARXNG_URL (default http://127.0.0.1:8888).
    """
    return (os.environ.get("XDEEP_SEARXNG_URL") or os.environ.get("SEARXNG_URL")
            or SEARXNG_URL_DEFAULT).rstrip("/")


def searxng_timeout() -> float:
    """HTTP timeout (seconds) for Searxng requests.

    Env: XDEEP_SEARXNG_TIMEOUT or SEARXNG_TIMEOUT (default 15.0).
    """
    raw = os.environ.get("XDEEP_SEARXNG_TIMEOUT") or os.environ.get("SEARXNG_TIMEOUT")
    if raw and raw.strip():
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return SEARXNG_TIMEOUT_S_DEFAULT


def searxng_top_k() -> int:
    """Default number of results per search.

    Env: XDEEP_SEARXNG_TOP_K or SEARXNG_TOP_K (default 8).
    """
    raw = os.environ.get("XDEEP_SEARXNG_TOP_K") or os.environ.get("SEARXNG_TOP_K")
    if raw and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return SEARXNG_TOP_K_DEFAULT


# ---------------------------------------------------------------------------
# Paced ChatOpenAI (Mistral 1 req / 1.5s)
# ---------------------------------------------------------------------------

_pacer_lock = asyncio.Lock()
_last_request_at: float = 0.0


async def _wait_for_min_interval(min_interval: float) -> None:
    """Process-wide pacing: never issue two requests closer than interval."""
    global _last_request_at
    async with _pacer_lock:
        now = time.monotonic()
        wait = min_interval - (now - _last_request_at)
        if wait > 0:
            await asyncio.sleep(wait)
            now = time.monotonic()
        _last_request_at = now


class PacedChatOpenAI(ChatOpenAI):
    """A ChatOpenAI subclass that enforces a minimum inter-request interval.

    Being a real BaseChatModel subclass, deepagents' resolve_model accepts it
    unchanged. Every generate path (sync + async) first awaits the shared
    process-wide gate, so concurrent mistral calls are spaced >= 1.5s.
    """

    min_interval: float = MISTRAL_MIN_INTERVAL_S

    def __init__(self, *, min_interval: float = MISTRAL_MIN_INTERVAL_S, **kwargs):
        if not kwargs or not (kwargs.get("model") or kwargs.get("base_url")):
            cfg = get_config()
            base_url, api_key, name = _mistral_endpoint(cfg)
            kwargs = {
                "model": name,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": 0.0,
                **kwargs,
            }
        super().__init__(**kwargs)  # type: ignore[call-arg]
        self.min_interval = max(0.0, min_interval)

    # -- pacing hooks -------------------------------------------------------
    def _generate(self, *args, **kwargs):
        _synchronously_pace(self.min_interval)
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        await _wait_for_min_interval(self.min_interval)
        return await super()._agenerate(*args, **kwargs)

    def invoke(self, input, config: Optional[dict] = None, **kwargs):
        _synchronously_pace(self.min_interval)
        return super().invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config: Optional[dict] = None, **kwargs):
        await _wait_for_min_interval(self.min_interval)
        return await super().ainvoke(input, config=config, **kwargs)


def _synchronously_pace(min_interval: float) -> None:
    """Pace from a sync context (used when no running loop)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_wait_for_min_interval(min_interval))
        return
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # sync called from inside a loop (rare) - schedule and wait
        future = asyncio.run_coroutine_threadsafe(
            _wait_for_min_interval(min_interval), loop)
        future.result()


# ---------------------------------------------------------------------------
# Role-aware builders
# ---------------------------------------------------------------------------

def build_think_model():
    """The 'good agent' for thinking + long-running tasks.

    Routes to the active think provider (qwen by default in this deployment,
    or groq/google/opencode when forced / env-selected):
      - qwen    -> plain ChatOpenAI against QWEN_BASE_URL / GENERAL_LLM_BASE_URL
                   (the Alibaba MaaS endpoint), model from QWEN_MODEL /
                   GENERAL_LLM_MODEL (default qwen3.8-flash). No in-process
                   guard; the endpoint quota applies.
      - opencode -> OpenCodeLangChainWrapper wrapping the Responses API
                   (Muse Spark endpoint), model from OPENCODE_MODEL /
                   GENERAL_LLM_MODEL (default muse-spark-1.3-contributor-free).
      - groq    -> TPMLimitedChatOpenAI (8,000 tokens/min default,
                   XDEEP_GROQ_TPM).
      - google  -> plain ChatOpenAI on the shared GENERAL_LLM_* endpoint with a
                   Gemma fallback model.
    """
    from langchain_openai import ChatOpenAI

    from src.agents.ratelimit import TPMLimitedChatOpenAI

    cfg = get_config()
    base_url, api_key, name = _think_endpoint(cfg)
    provider = think_provider()
    from src.agents.logging import xdeep_log

    xdeep_log("think_model_selected", provider=provider,
              model=name, base_url=base_url)
    if provider == "groq":
        tpm = os.environ.get("XDEEP_GROQ_TPM", "").strip()
        tpm = int(tpm) if tpm.isdigit() else 8000
        xdeep_log("groq_tpm_guard", tpm=tpm)
        return TPMLimitedChatOpenAI(
            model=name,
            api_key=api_key or "dummy",
            base_url=base_url,
            temperature=0.0,
            tpm=tpm,
        )
    if provider == "opencode":
        # OpenCode uses the Responses API, not Chat Completions.
        # Use our custom adapter.
        from src.llm.opencode_client import build_opencode_model
        return build_opencode_model(
            base_url=base_url,
            api_key=api_key or "dummy",
            model=name,
            temperature=0.0,
        )
    return ChatOpenAI(
        model=name,
        api_key=api_key or "dummy",
        base_url=base_url,
        temperature=0.0,
    )


def build_mistral_model(min_interval: float = MISTRAL_MIN_INTERVAL_S):
    """The 'hammerable' agent: Mistral, paced at one request per 1.5s."""
    return PacedChatOpenAI(min_interval=min_interval)


def build_model_for_role(role: str):
    """Router: think (qwen/groq/google) for thinking/long tasks, mistral
    (paced) for verification/contradiction/resolution."""
    side = role_model_name(role)
    return build_think_model() if side == "think" else build_mistral_model()
