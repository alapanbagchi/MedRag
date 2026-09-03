"""Shared fixtures/helpers for the backend tests."""

from __future__ import annotations

import pytest


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
        "QWEN_BASE_URL", "QWEN_API_KEY", "QWEN_MODEL",
        "XDEEP_THINK_PROVIDER", "XDEEP_GROQ_TPM",
        "GENERAL_LLM_BASE_URL", "GENERAL_LLM_API_KEY", "GENERAL_LLM_MODEL",
        "UMLS_API_KEY", "INDEX_DIR", "MAX_DOCUMENTS",
        "ENABLE_DENSE",
        "LOCAL_MODE", "VECTOR_DB_URL",
        # think/mistral routing (the ambient .env may set these)
        "MISTRAL_API_KEY", "MISTRAL_MODEL", "MISTRAL_BASE_URL",
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
def _isolate_xdeep_log(tmp_path, monkeypatch):
    """Redirect agents structured logging to a temp file.

    xdeep_log()/progress() write wherever XDEEP_LOG_FILE points (default
    logs.txt in the CWD = backend/logs.txt) - so without this, EVERY pytest
    run pollutes the production log with test events. log_path() is dynamic,
    so pointing the env var here is enough; the tmp path is per-test and
    auto-cleaned.
    """
    monkeypatch.setenv("XDEEP_LOG_FILE", str(tmp_path / "xdeep-test.log"))
    yield


@pytest.fixture(autouse=True)
def _no_network_fetch(monkeypatch):
    """Offset page fetches during tests: fetch_page_text hits the real web
    (WHO/PMC/etc.) whenever a web hit is processed. Pin it to a deterministic
    offline stub (returns page text when the test opts in via
    monkeypatch.setattr on the module, else '')."""
    async def _offline(url, max_chars=10000, timeout_s=12.0):
        return ""   # tests that care stub fetch_page_text explicitly
    import src.agents.tools.fetch as fetch_mod

    if not hasattr(fetch_mod, "_REAL_FETCH"):
        # stash the real implementation so offline-pinning can be undone
        fetch_mod._REAL_FETCH = fetch_mod.fetch_page_text
    monkeypatch.setattr(fetch_mod, "fetch_page_text", _offline)
    yield


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Process singletons must not leak state between tests."""
    from src.retrieval.retriever import reset_retrieval_service
    from src.retrieval.fullpaper import reset_unit_index

    reset_retrieval_service()
    reset_unit_index()
    yield
    reset_retrieval_service()
    reset_unit_index()