"""LLM provider client (global + per-agent role routing)."""

from __future__ import annotations

import contextvars
import os
from typing import Any, Optional, Tuple

from pydantic_ai.models.openai import OpenAIChatModel

from src.config import AppConfig

# Optional per-request event sink (callable emit(type_, **fields)) bound via
# set_llm_event_sink. Contextvars scope it to ONE request's task tree, so
# concurrent streams never cross-talk. With no sink bound, models behave
# exactly as before (zero overhead).
_LLM_SINK: contextvars.ContextVar = contextvars.ContextVar("llm_event_sink", default=None)


def set_llm_event_sink(sink: Any) -> None:
    """Bind an event sink for LLM-call observations in the CURRENT context."""
    _LLM_SINK.set(sink)


class _ObservedOpenAI(OpenAIChatModel):
    """OpenAIChatModel subclass that emits ``llm_call`` events (start ->
    ok | failed) around each request when an event sink is bound
    (set_llm_event_sink). Being a real pydantic_ai Model instance it passes
    infer_model() untouched. Agent-level retries surface naturally as
    failed -> start -> ok sequences in the UI log, so rate-limit retries are
    visible like Logfire shows them. With no sink bound it behaves exactly
    like its parent (zero overhead).
    (ponytail: request_stream is not wrapped - nothing here streams model
    output; add it when an agent uses run(stream=True).)"""

    def __init__(self, *args: Any, role: str = "default", **kwargs: Any):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_role", (role or "default").strip() or "default")
        object.__setattr__(self, "_attempt", 0)

    def _emit(self, status: str, exc: Optional[BaseException] = None) -> None:
        sink = _LLM_SINK.get()
        if sink is None:
            return
        fields: dict[str, Any] = {
            "role": object.__getattribute__(self, "_role"),
            "model": self.model_name,
            "attempt": object.__getattribute__(self, "_attempt"),
            "status": status,
        }
        if exc is not None:
            fields["error"] = str(exc)[:300]
            code = getattr(exc, "status_code", None)
            if code is None and getattr(exc, "response", None) is not None:
                code = getattr(exc.response, "status_code", None)
            fields["status_code"] = code
        sink("llm_call", **fields)

    async def request(self, *args: Any, **kwargs: Any):
        object.__setattr__(self, "_attempt", object.__getattribute__(self, "_attempt") + 1)
        self._emit("start")
        try:
            resp = await super().request(*args, **kwargs)
            self._emit("ok")
            return resp
        except Exception as exc:
            self._emit("failed", exc)
            raise


def _resolve(provider: str, base_url: str, api_key: str, model: str
             ) -> Tuple[str, str, str, Optional[dict]]:
    """(base_url, api_key, model, profile) for one provider."""
    if provider == "kaggle":
        return base_url, api_key, model, {"temperature": 0.0}
    if provider == "openai":
        return base_url, api_key, model, None
    if provider == "ollama":
        return base_url, "ollama", model, {"temperature": 0.0}
    if provider == "vllm":
        return base_url, "vllm", model, {"temperature": 0.0}
    if provider == "gemini":
        return base_url, api_key, model, None
    if provider == "mistral":
        # Mistral's OpenAI-compatible API (https://api.mistral.ai/v1)
        return base_url, api_key, model, None
    raise ValueError(f"unknown provider: {provider!r}")


def resolve_provider(cfg: AppConfig) -> Tuple[str, str, str, Optional[dict]]:
    """Return (base_url, api_key, model, profile) for configured provider."""
    provider = cfg.provider
    if provider == "kaggle":
        return _resolve("kaggle", cfg.kaggle_base_url, cfg.kaggle_api_key, cfg.kaggle_model)
    if provider == "openai":
        return _resolve("openai", cfg.openai_base_url, cfg.openai_api_key, cfg.openai_model)
    if provider == "ollama":
        return _resolve("ollama", cfg.ollama_base_url, "ollama", cfg.ollama_model)
    if provider == "vllm":
        return _resolve("vllm", cfg.vllm_base_url, "vllm", cfg.vllm_model)
    if provider == "gemini":
        return _resolve("gemini", cfg.gemini_base_url, cfg.gemini_api_key, cfg.gemini_model)
    if provider == "mistral":
        return _resolve("mistral", cfg.mistral_base_url, cfg.mistral_api_key, cfg.mistral_model)
    raise ValueError(f"unknown provider: {provider!r}")


# Roles understood by build_model_for (each maps to {ROLE}_PROVIDER/_BASE_URL/
# _API_KEY/_MODEL env overrides). The generic role "default" uses the global
# provider.
KNOWN_ROLES = ("orchestrator", "planner", "verifier", "synthesizer",
               "contradiction", "resolution", "deep_inspector")


def role_provider_name(config: Optional[AppConfig] = None, role: str = "default"
                       ) -> str:
    """Effective provider name for one role (env override or global)."""
    cfg = config or AppConfig()
    r = (role or "default").strip().lower()
    if r in ("", "default", "main"):
        return cfg.provider
    return (os.environ.get(f"{r.upper()}_PROVIDER", "").strip().lower()
            or cfg.provider)


def resolve_role_provider(cfg: AppConfig, role: str
                          ) -> Tuple[str, str, str, Optional[dict]]:
    """Per-agent (base_url, api_key, model, profile) with env overrides.

    ``{ROLE}_PROVIDER`` switches the provider for one agent; ``{ROLE}_BASE_URL``
    / ``{ROLE}_API_KEY`` / ``{ROLE}_MODEL`` override the individual fields.
    Anything not overridden falls back to the global provider, so the common
    setup is: global provider (e.g. Gemma) + one override (e.g.
    VERIFIER_PROVIDER=mistral) for verification.
    """
    r = (role or "default").strip().lower()
    pre = f"{r.upper()}_" if r and r != "default" else ""
    env = lambda key: os.environ.get(pre + key, "").strip()  # noqa: E731
    provider = env("PROVIDER").lower() or cfg.provider
    if provider == "kaggle":
        return _resolve("kaggle", env("BASE_URL") or cfg.kaggle_base_url,
                        env("API_KEY") or cfg.kaggle_api_key,
                        env("MODEL") or cfg.kaggle_model)
    if provider == "openai":
        return _resolve("openai", env("BASE_URL") or cfg.openai_base_url,
                        env("API_KEY") or cfg.openai_api_key,
                        env("MODEL") or cfg.openai_model)
    if provider == "ollama":
        return _resolve("ollama", env("BASE_URL") or cfg.ollama_base_url,
                        "ollama", env("MODEL") or cfg.ollama_model)
    if provider == "vllm":
        return _resolve("vllm", env("BASE_URL") or cfg.vllm_base_url,
                        "vllm", env("MODEL") or cfg.vllm_model)
    if provider == "gemini":
        return _resolve("gemini", env("BASE_URL") or cfg.gemini_base_url,
                        env("API_KEY") or cfg.gemini_api_key,
                        env("MODEL") or cfg.gemini_model)
    if provider == "mistral":
        return _resolve("mistral", env("BASE_URL") or cfg.mistral_base_url,
                        env("API_KEY") or cfg.mistral_api_key,
                        env("MODEL") or cfg.mistral_model)
    raise ValueError(f"unknown provider: {provider!r}")


def _observe(model_name: str, provider: Any, profile: Optional[Any],
             role: str = "default") -> Any:
    """Use the LLM-call observer when a sink is bound, else the plain model."""
    if _LLM_SINK.get() is None:
        return OpenAIChatModel(model_name, provider=provider, profile=profile)
    return _ObservedOpenAI(model_name, provider=provider, profile=profile, role=role)


def build_model_for(config: Optional[AppConfig] = None, role: str = "default"):
    """Build a PydanticAI model for one agent role ("default" = global)."""
    from pydantic_ai.providers.openai import OpenAIProvider

    cfg = config or AppConfig()
    if not (role or "").strip() or role.strip().lower() in ("default", "main"):
        return build_model(cfg)
    base_url, api_key, model, profile = resolve_role_provider(cfg, role)
    provider = os.environ.get(f"{role.upper()}_PROVIDER", "").strip().lower() or cfg.provider
    _require_api_key(provider, api_key, base_url, role=role)
    return _observe(model, OpenAIProvider(base_url=base_url, api_key=api_key),
                    profile, role)


def build_model(config: Optional[AppConfig] = None):
    """Build a PydanticAI model for the configured provider."""
    from pydantic_ai.providers.openai import OpenAIProvider

    cfg = config or AppConfig()
    base_url, api_key, model, profile = resolve_provider(cfg)
    _require_api_key(cfg.provider, api_key, base_url)
    return _observe(model, OpenAIProvider(base_url=base_url, api_key=api_key),
                    profile, "default")


# env var(s) that carry the key for providers that need one
_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GENERAL_LLM_API_KEY (or GEMINI_API_KEY)",
    "mistral": "MISTRAL_API_KEY",
}


def _require_api_key(provider: str, api_key: str, base_url: str,
                     role: str = "") -> None:
    """Fail fast with an actionable message instead of the raw OpenAI error.

    Local providers carry literal keys (ollama/vllm) or default to 'dummy'
    (kaggle), so only remote key-requiring providers are checked.
    """
    if provider in _KEY_ENV and not (api_key or "").strip():
        env = _KEY_ENV[provider]
        if role:
            env = f"{role.upper()}_API_KEY (from {env})"
        raise ValueError(
            f"LLM_PROVIDER={provider!r} requires an API key: set {env} "
            f"in your environment or .env. If your endpoint at {base_url} "
            f"does not need one (local proxy), set a dummy value, e.g. "
            f"{env.split(' ')[0]}=dummy."
        )
