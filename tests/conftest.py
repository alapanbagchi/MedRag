"""Shared fixtures/helpers for the local PydanticAI application tests."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from pydantic_ai.models.test import TestModel

# PydanticAI test model profile: force the same "native" structured-output
# mode the Kaggle endpoint uses (response_format json_schema, no tools).
NATIVE_PROFILE: Dict[str, Any] = {
    "default_structured_output_mode": "native",
    "supports_json_schema_output": True,
}


def native_test_model(output_text: str) -> TestModel:
    """A PydanticAI TestModel that emits one canned JSON text response."""
    return TestModel(custom_output_text=output_text, profile=NATIVE_PROFILE)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep env/config stable during tests.

    The repo's root .env must NOT influence tests (load_env_file needs to be
    neutralized), and every config-relevant variable is cleared so each test
    controls exactly what it sets.
    """
    import src.config as src_config

    monkeypatch.setattr(src_config, "load_env_file", lambda *a, **k: None)
    for key in (
        "LLM_PROVIDER",
        "KAGGLE_BASE_URL", "KAGGLE_API_KEY", "KAGGLE_MODEL",
        "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
        "OLLAMA_BASE_URL", "OLLAMA_MODEL",
        "VLLM_BASE_URL", "VLLM_MODEL",
        "GEMINI_BASE_URL", "GEMINI_API_KEY", "GEMINI_MODEL",
        "GENERAL_LLM_BASE_URL", "GENERAL_LLM_API_KEY", "GENERAL_LLM_MODEL",
        "UMLS_API_KEY", "INDEX_DIR", "MAX_DOCUMENTS", "MAX_WORKERS",
        "AGENT_MAX_TOKENS", "VERIFIER_MAX_GROUPS", "SYNTHESIZER_MAX_EVIDENCE",
        "GLOBAL_TOKENS_PER_MIN", "VERIFIER_TOKEN_BUDGET", "MAX_LLM_RETRIES",
        "ENABLE_DENSE", "REWRITE_ENABLED", "ENABLE_CROSS_ENCODER",
        "TRACE_FULL_TEXTS", "LOCAL_MODE", "VECTOR_DB_URL",
        "PLANNER_USE_UMLS_TOOL",
        # per-role model routing + mistral batch (the ambient .env may set these)
        "MISTRAL_API_KEY", "MISTRAL_MODEL", "MISTRAL_BASE_URL",
        "ORCHESTRATOR_PROVIDER", "ORCHESTRATOR_BASE_URL",
        "ORCHESTRATOR_API_KEY", "ORCHESTRATOR_MODEL",
        "PLANNER_PROVIDER", "PLANNER_BASE_URL", "PLANNER_API_KEY",
        "PLANNER_MODEL",
        "VERIFIER_PROVIDER", "VERIFIER_BASE_URL", "VERIFIER_API_KEY",
        "VERIFIER_MODEL",
        "SYNTHESIZER_PROVIDER", "SYNTHESIZER_BASE_URL",
        "SYNTHESIZER_API_KEY", "SYNTHESIZER_MODEL",
        "CONTRADICTION_PROVIDER", "RESOLUTION_PROVIDER",
        "DEEP_INSPECTOR_PROVIDER",
        # Logfire must stay OFF in tests: with credentials present (the repo's
        # .logfire/) an accidental export would pollute the real project.
        "LOGFIRE_ENABLED", "LOGFIRE_TOKEN", "LOGFIRE_SERVICE_NAME",
        "LOGFIRE_SERVICE_VERSION", "LOGFIRE_ENVIRONMENT", "LOGFIRE_CONSOLE",
        "LOGFIRE_CREDENTIALS_DIR", "LOGFIRE_SEND_TO_LOGFIRE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOGFIRE_ENABLED", "0")
    yield


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Process singletons must not leak state between tests."""
    from src.retrieval.retriever import reset_retrieval_service
    from src.retrieval.fullpaper import reset_unit_index
    from src.llm.ratelimit import reset_bucket
    from src.llm.run import reset_llm_observer

    reset_retrieval_service()
    reset_unit_index()
    reset_bucket()
    reset_llm_observer()
    yield
    reset_retrieval_service()
    reset_unit_index()
    reset_bucket()
    reset_llm_observer()