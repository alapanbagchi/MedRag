"""Provider factory tests: swapping the LLM provider is a config change only."""

from __future__ import annotations

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from src.config import AppConfig
from src.llm.client import build_model, build_model_for, resolve_provider


@pytest.fixture
def gemini_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.delenv("AGENT_MAX_TOKENS", raising=False)
    yield


def test_gemini_provider_construction(gemini_env):
    cfg = AppConfig()
    assert cfg.provider == "gemini"
    # Gemini has no 512-token cap -> provider-aware default budget
    assert cfg.agent_max_tokens == 2048

    base_url, api_key, model_name, _profile = resolve_provider(cfg)
    # AppConfig rstrips trailing slashes on provider URLs
    assert base_url.rstrip("/") == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert api_key == "AIza-test-key"
    assert model_name == "gemini-2.0-flash"

    model = build_model(cfg)
    assert isinstance(model, OpenAIChatModel)


def test_gemini_free_tier_gets_token_bucket(gemini_env):
    cfg = AppConfig()
    assert cfg.tokens_per_minute > 0, "gemini free tier must enable the shared bucket"


@pytest.fixture
def mistral_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.setenv("MISTRAL_MODEL", "mistral-large-latest")
    monkeypatch.delenv("AGENT_MAX_TOKENS", raising=False)
    yield


def test_mistral_provider_construction(mistral_env):
    cfg = AppConfig()
    assert cfg.provider == "mistral"
    # Mistral runs 32k-context models -> no 512-token cap
    assert cfg.agent_max_tokens == 2048

    base_url, api_key, model_name, _profile = resolve_provider(cfg)
    assert base_url.rstrip("/") == "https://api.mistral.ai/v1"
    assert api_key == "mistral-test-key"
    assert model_name == "mistral-large-latest"

    model = build_model(cfg)
    assert isinstance(model, OpenAIChatModel)


def test_mistral_base_url_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://proxy.example/v1/")
    cfg = AppConfig()
    assert cfg.mistral_base_url == "https://proxy.example/v1"  # trailing slash stripped


def test_mistral_missing_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        build_model(AppConfig())


def test_gemini_missing_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GENERAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GENERAL_LLM_API_KEY"):
        build_model(AppConfig())


def test_dummy_key_allowed_for_local_endpoint(monkeypatch):
    # local proxy with a dummy key must still construct fine
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    monkeypatch.setenv("MISTRAL_BASE_URL", "http://127.0.0.1:8083/v1")
    cfg = AppConfig()
    assert isinstance(build_model(cfg), OpenAIChatModel)


# ---------------------------------------------------------------------------
# Per-agent role routing (build_model_for)
# ---------------------------------------------------------------------------

@pytest.fixture
def gemma_global_mistral_verifier(monkeypatch):
    """Global = Gemma (gemini provider), verifier = Mistral override."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemma")
    monkeypatch.setenv("GEMINI_MODEL", "gemma-2-27b-it")
    monkeypatch.setenv("VERIFIER_PROVIDER", "mistral")
    monkeypatch.setenv("VERIFIER_API_KEY", "dummy-mistral")
    monkeypatch.setenv("VERIFIER_MODEL", "mistral-medium-latest")
    for k in ("ORCHESTRATOR_PROVIDER", "SYNTHESIZER_PROVIDER",
              "PLANNER_PROVIDER", "DEEP_INSPECTOR_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    yield


def test_role_verifier_uses_mistral(gemma_global_mistral_verifier):
    cfg = AppConfig()
    verifier = build_model_for(cfg, "verifier")
    assert verifier.model_name == "mistral-medium-latest"
    assert "api.mistral.ai" in verifier.provider.base_url


def test_other_roles_fall_back_to_global(gemma_global_mistral_verifier):
    cfg = AppConfig()
    for role in ("orchestrator", "synthesizer", "planner", "contradiction",
                 "resolution", "deep_inspector"):
        m = build_model_for(cfg, role)
        assert m.model_name == "gemma-2-27b-it", role
        assert "generativelanguage" in m.provider.base_url, role


def test_default_role_is_global(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_MODEL", "gemma-2-27b-it")
    cfg = AppConfig()
    assert build_model_for(cfg, "default").model_name == "gemma-2-27b-it"


def test_role_missing_key_fails_clearly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("VERIFIER_PROVIDER", "mistral")
    monkeypatch.delenv("VERIFIER_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VERIFIER_API_KEY"):
        build_model_for(AppConfig(), "verifier")


def test_local_provider_disables_bucket(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("GLOBAL_TOKENS_PER_MIN", raising=False)
    monkeypatch.delenv("VERIFIER_TOKEN_BUDGET", raising=False)
    cfg = AppConfig()
    assert cfg.tokens_per_minute == 0


def test_kaggle_default_budget_stays_512(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kaggle")
    monkeypatch.delenv("AGENT_MAX_TOKENS", raising=False)
    cfg = AppConfig()
    assert cfg.agent_max_tokens == 512


def test_kaggle_base_url_whitespace_stripped(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kaggle")
    monkeypatch.setenv("KAGGLE_BASE_URL", "  https://tunnel.example/v1  ")
    cfg = AppConfig()
    assert cfg.kaggle_base_url == "https://tunnel.example/v1"


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError):
        build_model(AppConfig())
