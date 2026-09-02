"""x_deepagents config routing + tools tests (offline).

Covers:
  * model routing: think (qwen/groq/google) vs mistral (paced short decisions)
  * qwen (Alibaba MaaS) vs groq vs google provider selection (auto + forced)
  * the mistral pacing behavior (1 req / 1.5s) is enforced structurally
  * searxng tool degrades cleanly when no instance is present
"""

from __future__ import annotations

import pytest


def test_role_model_names():
    from src.x_deepagents.config import (
        MISTRAL_ROLES,
        THINK_ROLES,
        role_model_name,
    )

    for role in ("decompose", "research", "synthesis", "conflict"):
        assert role in THINK_ROLES
        assert role_model_name(role) == "think"
    for role in ("verify", "contradiction", "resolution"):
        assert role in MISTRAL_ROLES
        assert role_model_name(role) == "mistral"


def test_unknown_role_rejected():
    from src.x_deepagents.config import role_model_name

    with pytest.raises(ValueError):
        role_model_name("nonsense")


def test_mistral_model_is_paced_base_chat_model():
    from langchain_core.language_models.chat_models import BaseChatModel

    from src.x_deepagents.config import MISTRAL_MIN_INTERVAL_S, build_mistral_model

    m = build_mistral_model()
    assert isinstance(m, BaseChatModel)      # deepagents accepts it unchanged
    assert m.min_interval == MISTRAL_MIN_INTERVAL_S


def test_think_model_builds(monkeypatch):
    from src.x_deepagents.config import build_think_model

    monkeypatch.delenv("XDEEP_THINK_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    m = build_think_model()
    assert m is not None
    assert getattr(m, "model_name", "") or getattr(m, "model", "")


def test_think_provider_auto_selects_groq_when_key_present(monkeypatch):
    from src.x_deepagents.config import think_provider

    monkeypatch.delenv("XDEEP_THINK_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")
    assert think_provider() == "groq"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert think_provider() == "google"


def test_think_provider_forced(monkeypatch):
    from src.x_deepagents.config import think_provider

    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "groq")
    assert think_provider() == "groq"
    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "google")
    assert think_provider() == "google"
    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        think_provider()


def test_think_provider_auto_selects_qwen_when_alibaba_base(monkeypatch):
    """GENERAL_LLM_BASE_URL pointing at Alibaba MaaS selects qwen."""
    from src.x_deepagents.config import think_provider

    monkeypatch.delenv("XDEEP_THINK_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GENERAL_LLM_BASE_URL",
                       "https://ws-x.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
    assert think_provider() == "qwen"


def test_think_provider_prefers_qwen_key_over_groq(monkeypatch):
    """QWEN_API_KEY outranks GROQ_API_KEY in auto-selection."""
    from src.x_deepagents.config import think_provider

    monkeypatch.delenv("XDEEP_THINK_PROVIDER", raising=False)
    monkeypatch.setenv("QWEN_API_KEY", "qk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")
    assert think_provider() == "qwen"


def test_think_provider_aliases_normalize_to_qwen(monkeypatch):
    from src.x_deepagents.config import think_provider

    for alias in ("qwen", "alibaba", "aliyun", "aliyuncs", "dashscope"):
        monkeypatch.setenv("XDEEP_THINK_PROVIDER", alias)
        assert think_provider() == "qwen", alias


def test_qwen_model_routes_to_alibaba_endpoint(monkeypatch):
    """The think model points at QWEN_* (falling back to GENERAL_LLM_*)."""
    from src.x_deepagents.config import build_think_model

    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_BASE_URL", "https://my-qwen.endpoint/v1")
    monkeypatch.setenv("QWEN_API_KEY", "qk-test")
    monkeypatch.setenv("QWEN_MODEL", "qwen-max")
    monkeypatch.delenv("GENERAL_LLM_BASE_URL", raising=False)
    m = build_think_model()
    assert (m.openai_api_base or "").rstrip("/") == "https://my-qwen.endpoint/v1"
    assert (getattr(m, "model_name", "") or getattr(m, "model", "")) == "qwen-max"
    assert getattr(m, "openai_api_key", "")


def test_qwen_defaults_model_from_general_llm(monkeypatch):
    """No QWEN_MODEL -> GENERAL_LLM_MODEL -> qwen3.8-flash fallback."""
    from src.x_deepagents.config import QWEN_MODEL_FALLBACK, build_think_model

    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "qwen")
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("GENERAL_LLM_MODEL", raising=False)
    m = build_think_model()
    model = getattr(m, "model_name", "") or getattr(m, "model", "")
    assert model == QWEN_MODEL_FALLBACK
    assert model != "gemini-2.0-flash"  # never the shared gemini flash default


def test_qwen_think_model_not_tpm_limited(monkeypatch):
    """Qwen uses a plain ChatOpenAI (no Groq TPM guard)."""
    from src.x_deepagents.config import build_think_model
    from src.x_deepagents.ratelimit import TPMLimitedChatOpenAI

    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "qwen")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    m = build_think_model()
    assert not isinstance(m, TPMLimitedChatOpenAI)


def test_groq_model_routes_to_groq_endpoint(monkeypatch):
    """With a GROQ_API_KEY set, the think model points at Groq."""
    from src.x_deepagents.config import (
        GROQ_BASE_URL_DEFAULT,
        GROQ_MODEL_DEFAULT,
        build_think_model,
    )

    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")
    monkeypatch.setenv("GROQ_MODEL", "my-groq-model")
    m = build_think_model()
    assert (m.openai_api_base or "").rstrip("/") == GROQ_BASE_URL_DEFAULT
    assert (getattr(m, "model_name", "") or getattr(m, "model", "")) == "my-groq-model"
    # key is set (masked in repr, so just assert it is truthy)
    assert getattr(m, "openai_api_key", "")


def test_agents_build_with_routed_models():
    """Every stage agent constructs with its routed model (no LLM call)."""
    from src.x_deepagents.agents.hello import make_research_agent
    from src.x_deepagents.agents.stages import (
        make_contradiction_agent,
        make_deep_inspector,
        make_orchestrator,
        make_replanner,
        make_resolution_agent,
        make_search_planner,
        make_synthesizer,
        make_verifier,
    )

    for maker in (make_orchestrator, make_search_planner, make_replanner,
                  make_research_agent, make_verifier, make_deep_inspector,
                  make_contradiction_agent, make_resolution_agent,
                  make_synthesizer):
        agent = maker()
        nodes = set(agent.get_graph().nodes)
        assert {"model", "tools"} <= nodes, maker.__name__


def test_searxng_tool_registered():
    from src.x_deepagents.tools import all_tools, selectable

    assert "searxng_search" in all_tools()
    assert "searxng_search" in selectable()


def test_searxng_degrades_cleanly(monkeypatch):
    """With no instance reachable, returns available:false (never throws)."""
    import asyncio
    import json

    monkeypatch.setenv("XDEEP_SEARXNG_URL", "http://127.0.0.1:1")   # truly unroutable
    from src.x_deepagents.tools.searxng import searxng_search

    out = asyncio.run(searxng_search.ainvoke({"query": "hypertension", "top_k": 3}))
    data = json.loads(out)
    assert data["available"] is False
    assert data["results"] == []


# --- Groq TPM guard --------------------------------------------------------

def test_tpm_bucket_acquire_respects_window():
    import asyncio
    import time

    from src.x_deepagents.ratelimit import TPMBucket, reset_tpm_bucket

    reset_tpm_bucket()
    b = TPMBucket(tokens_per_minute=8000)
    # reserve 5000 tokens - fits within the window
    waited = asyncio.run(b.acquire(5000))
    assert waited == 0.0
    assert b._tokens_used == 5000
    # 4000 more exceeds the 8000 window -> needs the window to roll over.
    # Simulate a stale window (started 90s ago, in MONOTONIC time like the
    # bucket's clock) so the roll happens instantly.
    b._window_start = time.monotonic() - 90.0
    waited2 = asyncio.run(b.acquire(4000))
    assert waited2 == 0.0  # window rolled, admitted immediately
    assert b._tokens_used == 4000


def test_tpm_bucket_oversized_request_admitted():
    import asyncio

    from src.x_deepagents.ratelimit import TPMBucket

    b = TPMBucket(tokens_per_minute=8000)
    # a single request larger than the whole window is admitted immediately
    waited = asyncio.run(b.acquire(99999))
    assert waited == 0.0


def test_tpm_bucket_disabled_when_zero():
    import asyncio

    from src.x_deepagents.ratelimit import TPMBucket

    b = TPMBucket(tokens_per_minute=0)
    assert b.enabled is False
    assert asyncio.run(b.acquire(5000)) == 0.0


def test_groq_think_model_is_tpm_limited(monkeypatch):
    from langchain_core.language_models.chat_models import BaseChatModel

    from src.x_deepagents.config import build_think_model
    from src.x_deepagents.ratelimit import TPMLimitedChatOpenAI, reset_tpm_bucket

    reset_tpm_bucket()
    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")
    m = build_think_model()
    assert isinstance(m, TPMLimitedChatOpenAI)
    assert isinstance(m, BaseChatModel)
    assert m._bucket.tokens_per_minute == 8000
    reset_tpm_bucket()


def test_groq_tpm_env_override(monkeypatch):
    from src.x_deepagents.config import build_think_model
    from src.x_deepagents.ratelimit import TPMLimitedChatOpenAI, reset_tpm_bucket

    reset_tpm_bucket()
    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")
    monkeypatch.setenv("XDEEP_GROQ_TPM", "6000")
    m = build_think_model()
    assert isinstance(m, TPMLimitedChatOpenAI)
    assert m._bucket.tokens_per_minute == 6000
    reset_tpm_bucket()


def test_google_think_model_not_tpm_limited(monkeypatch):
    from src.x_deepagents.config import build_think_model
    from src.x_deepagents.ratelimit import TPMLimitedChatOpenAI, reset_tpm_bucket

    reset_tpm_bucket()
    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "google")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    m = build_think_model()
    assert not isinstance(m, TPMLimitedChatOpenAI)
    reset_tpm_bucket()


# --- Gemma-everywhere guarantees (no flash) -------------------------------

def test_groq_default_model_is_gemma(monkeypatch):
    from src.x_deepagents.config import GROQ_MODEL_DEFAULT

    assert "gemma" in GROQ_MODEL_DEFAULT.lower()


def test_google_think_model_never_flash_even_with_shared_default(monkeypatch):
    """xdeep must never emit the shared AppConfig gemini-2.0-flash default."""
    from src.x_deepagents.config import build_think_model

    # simulate the bad case: shared AppConfig defaults to flash AND no env
    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "google")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GENERAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    m = build_think_model()
    model = getattr(m, "model_name", "") or getattr(m, "model", "")
    assert "flash" not in model.lower()
    assert "gemma" in model.lower()


def test_groq_think_model_is_gemma_and_tpm_limited(monkeypatch):
    from src.x_deepagents.config import build_think_model
    from src.x_deepagents.ratelimit import TPMLimitedChatOpenAI, reset_tpm_bucket

    reset_tpm_bucket()
    monkeypatch.setenv("XDEEP_THINK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")
    monkeypatch.delenv("GROQ_MODEL", raising=False)  # use the Gemma default
    m = build_think_model()
    assert isinstance(m, TPMLimitedChatOpenAI)
    model = getattr(m, "model_name", "") or getattr(m, "model", "")
    assert "gemma" in model.lower()
    assert m._bucket.tokens_per_minute == 8000
    reset_tpm_bucket()


# --- Searxng env config -----------------------------------------------------

def test_searxng_env_defaults(monkeypatch):
    from src.x_deepagents.config import (
        SEARXNG_TIMEOUT_S_DEFAULT,
        SEARXNG_TOP_K_DEFAULT,
        SEARXNG_URL_DEFAULT,
        searxng_timeout,
        searxng_top_k,
        searxng_url,
    )

    for key in ("XDEEP_SEARXNG_URL", "SEARXNG_URL",
                "XDEEP_SEARXNG_TIMEOUT", "SEARXNG_TIMEOUT",
                "XDEEP_SEARXNG_TOP_K", "SEARXNG_TOP_K"):
        monkeypatch.delenv(key, raising=False)
    assert searxng_url() == SEARXNG_URL_DEFAULT
    assert searxng_timeout() == SEARXNG_TIMEOUT_S_DEFAULT
    assert searxng_top_k() == SEARXNG_TOP_K_DEFAULT


def test_searxng_env_overrides(monkeypatch):
    from src.x_deepagents.config import (
        searxng_timeout,
        searxng_top_k,
        searxng_url,
    )

    monkeypatch.setenv("XDEEP_SEARXNG_URL", "http://127.0.0.1:9999")
    assert searxng_url() == "http://127.0.0.1:9999"
    monkeypatch.setenv("SEARXNG_URL", "http://other:7777")
    # XDEEP_ wins over the bare SEARXNG_ form
    assert searxng_url() == "http://127.0.0.1:9999"
    monkeypatch.setenv("SEARXNG_TIMEOUT", "30")
    assert searxng_timeout() == 30.0
    monkeypatch.setenv("XDEEP_SEARXNG_TOP_K", "12")
    assert searxng_top_k() == 12


def test_searxng_env_bad_values_fall_back(monkeypatch):
    from src.x_deepagents.config import (
        SEARXNG_TIMEOUT_S_DEFAULT,
        SEARXNG_TOP_K_DEFAULT,
        searxng_timeout,
        searxng_top_k,
    )

    monkeypatch.setenv("XDEEP_SEARXNG_TIMEOUT", "not-a-number")
    assert searxng_timeout() == SEARXNG_TIMEOUT_S_DEFAULT
    monkeypatch.setenv("XDEEP_SEARXNG_TOP_K", "abc")
    assert searxng_top_k() == SEARXNG_TOP_K_DEFAULT


def test_searxng_tool_uses_env_url(monkeypatch):
    import asyncio
    import json

    from src.x_deepagents.tools.searxng import searxng_search

    # unreachable URL from env -> clean fallback, proving the tool reads env
    monkeypatch.setenv("XDEEP_SEARXNG_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("XDEEP_SEARXNG_TIMEOUT", "1")
    out = asyncio.run(searxng_search.ainvoke({"query": "vitamin D"}))
    data = json.loads(out)
    assert data["available"] is False
    assert "127.0.0.1:1" in data["note"]
