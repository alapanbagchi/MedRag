"""LLM provider client (global + per-agent role routing)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

from src.config import AppConfig


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


def build_model_for(config: Optional[AppConfig] = None, role: str = "default"):
    """Build a PydanticAI model for one agent role ("default" = global)."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    cfg = config or AppConfig()
    if not (role or "").strip() or role.strip().lower() in ("default", "main"):
        return build_model(cfg)
    base_url, api_key, model, profile = resolve_role_provider(cfg, role)
    provider = os.environ.get(f"{role.upper()}_PROVIDER", "").strip().lower() or cfg.provider
    _require_api_key(provider, api_key, base_url, role=role)
    return OpenAIChatModel(
        model,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        profile=profile,
    )


def build_model(config: Optional[AppConfig] = None):
    """Build a PydanticAI model for the configured provider."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    cfg = config or AppConfig()
    base_url, api_key, model, profile = resolve_provider(cfg)
    _require_api_key(cfg.provider, api_key, base_url)
    return OpenAIChatModel(
        model,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        profile=profile,
    )


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
