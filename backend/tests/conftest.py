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
        # verify-lane routing (Gemini Gemma 4 31B vs mistral fallback)
        "VERIFY_PROVIDER", "GEMINI_VERIFY_MODEL",
        # LLM lanes: src/agents/__main__ runs load_dotenv() at
        # import, so the ambient .env leaks into the pytest process unless
        # cleared here (order-dependent failures, e.g. VERIFIER_LOG_RAW).
        "LLM_BASE_URL", "LLM_API_KEY", "AGENT_MODEL", "SMALL_MODEL",
        "VERIFIER_LOG_RAW", "VERIFIER_TIMEOUT_S",
        # Debug routing must stay off in tests (parallel assertions, no locks).
        "SEQUENTIAL",
        # Trust-checker timeout knob.
        "TRUST_TIMEOUT_S",
        # MyBib credibility floor: tests pin their own threshold.
        "MYBIB_MIN_SCORE",
        # Logfire must stay OFF in tests: with credentials present (the repo's
        # .logfire/) an accidental export would pollute the real project.
        "LOGFIRE_ENABLED", "LOGFIRE_TOKEN", "LOGFIRE_SERVICE_NAME",
        "LOGFIRE_SERVICE_VERSION", "LOGFIRE_ENVIRONMENT", "LOGFIRE_CONSOLE",
        "LOGFIRE_CREDENTIALS_DIR", "LOGFIRE_SEND_TO_LOGFIRE",
        # Langfuse must stay OFF in tests: no trace export, no network, and
        # the tracing helpers take their inert path (bare ainvoke).
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST", "LANGFUSE_DEBUG",
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
    """Pin page fetches offline. Skips itself when the agents layer
    (trashed during the layer-by-layer rebuild) provides no fetch module."""
    try:
        import src.agents.tools.fetch as fetch_mod
    except ModuleNotFoundError:
        yield
        return

    async def _offline(url, max_chars=10000, timeout_s=12.0):
        return ""   # tests that care stub fetch_page_text explicitly
    async def _offline_bundle(url, max_chars=10000, timeout_s=12.0):
        return {"text": "", "fetched_ok": False, "page_title": "",
                "page_header": "", "favicon_url": "", "fetched_chars": 0}

    if not hasattr(fetch_mod, "_REAL_FETCH"):
        # stash the real implementation so offline-pinning can be undone
        fetch_mod._REAL_FETCH = fetch_mod.fetch_page_text
    if not hasattr(fetch_mod, "_REAL_BUNDLE"):
        fetch_mod._REAL_BUNDLE = fetch_mod.fetch_page_bundle
    monkeypatch.setattr(fetch_mod, "fetch_page_text", _offline)
    monkeypatch.setattr(fetch_mod, "fetch_page_bundle", _offline_bundle)
    yield


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Process singletons must not leak state between tests."""
    from src.retrieval.retriever import reset_retrieval_service
    from src.retrieval.fullpaper import reset_unit_index
    try:
        from src.agents.verify_queue import reset_for_tests
    except ModuleNotFoundError:
        reset_for_tests = None

    reset_retrieval_service()
    reset_unit_index()
    if reset_for_tests is not None:
        reset_for_tests()
    yield
    reset_retrieval_service()
    reset_unit_index()
    if reset_for_tests is not None:
        reset_for_tests()