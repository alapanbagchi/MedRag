"""LLM lane defaults (no LLM env required)."""

from __future__ import annotations

from pydantic_ai.settings import ModelSettings

from src.llm.models import Endpoint, make_model


def _endpoint() -> Endpoint:
    return Endpoint(base_url="http://localhost:1/v1", api_key="k",
                    model="m")


def test_make_model_defaults_temperature_zero():
    model = make_model(_endpoint())
    assert model.settings is not None
    assert model.settings["temperature"] == 0.0


def test_make_model_explicit_settings_win():
    model = make_model(_endpoint(),
                       settings=ModelSettings(temperature=0.7))
    assert model.settings is not None
    assert model.settings["temperature"] == 0.7


def test_make_model_defaults_reasoning_effort_low(monkeypatch):
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)
    model = make_model(_endpoint())
    assert model.settings["openai_reasoning_effort"] == "low"


def test_make_model_reasoning_effort_env_opt_out(monkeypatch):
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "default")
    model = make_model(_endpoint())
    assert "openai_reasoning_effort" not in model.settings


def test_make_model_explicit_reasoning_effort_wins(monkeypatch):
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "high")
    model = make_model(_endpoint(), settings=ModelSettings(
        temperature=0.0, openai_reasoning_effort="minimal"))
    assert model.settings["openai_reasoning_effort"] == "minimal"


def test_make_model_explicit_blank_effort_removes_key(monkeypatch):
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "low")
    model = make_model(_endpoint(), settings=ModelSettings(
        temperature=0.0, openai_reasoning_effort=""))
    assert "openai_reasoning_effort" not in model.settings


def _opencode_endpoint(model: str = "mimo-v2.5"):
    return Endpoint(base_url="https://opencode.ai/zen/go/v1/",
                    api_key="sk-opencode", model=model)


def _clear_session_env(monkeypatch):
    monkeypatch.delenv("LLM_SESSION_ID", raising=False)
    monkeypatch.delenv("OPENCODE_SESSION", raising=False)


def _headers(model) -> dict:
    provider = getattr(model, "provider", None)
    client = getattr(provider, "_client", None) or getattr(model, "client", None)
    return getattr(client, "default_headers", {}) or {}


# ── one fixed session id for every LLM call ──────────────────────────
def test_opencode_endpoint_carries_session_header(monkeypatch):
    """OpenCode Go's gateway rejects requests without x-opencode-session
    (HTTP 400 MissingSessionID): the header must be attached."""
    _clear_session_env(monkeypatch)
    monkeypatch.setattr("src.llm.models._PROCESS_SESSION", "proc-session-1")
    model = make_model(_opencode_endpoint())
    assert _headers(model).get("x-opencode-session") == "proc-session-1"


def test_session_id_env_pins_every_call(monkeypatch):
    """LLM_SESSION_ID is the one session every lane carries: the big agent
    model and the small (verifier/judge) model share the same id."""
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("LLM_SESSION_ID", "shared-cache")
    agent_model = make_model(_opencode_endpoint("muse-spark-1.3-contributor"))
    small_model = make_model(_opencode_endpoint("mimo-v2.5"))
    assert _headers(agent_model).get("x-opencode-session") == "shared-cache"
    assert _headers(small_model).get("x-opencode-session") == "shared-cache"


def test_session_id_falls_back_to_stable_process_id(monkeypatch):
    from src.llm.models import session_id

    _clear_session_env(monkeypatch)
    monkeypatch.setattr("src.llm.models._PROCESS_SESSION", "proc-session-1")
    assert session_id() == "proc-session-1"


def test_legacy_opencode_session_still_honored(monkeypatch):
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("OPENCODE_SESSION", "legacy-42")
    model = make_model(_opencode_endpoint())
    assert _headers(model).get("x-opencode-session") == "legacy-42"


def test_non_opencode_endpoint_gets_no_session_header(monkeypatch):
    _clear_session_env(monkeypatch)
    model = make_model(_endpoint())
    assert "x-opencode-session" not in _headers(model)


def test_non_opencode_endpoint_carries_configured_session(monkeypatch):
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("LLM_SESSION_ID", "shared-cache")
    model = make_model(_endpoint())
    assert _headers(model).get("x-opencode-session") == "shared-cache"


# ── API style: chat completions vs the OpenAI Responses API ─────────
def test_muse_spark_uses_responses_api(monkeypatch):
    """Muse Spark on OpenCode Go is served only through /responses; the
    chat-completions path would 400/404."""
    _clear_session_env(monkeypatch)
    model = make_model(_opencode_endpoint("muse-spark-1.3-contributor"))
    assert isinstance(model, _responses_model_type())
    assert "x-opencode-session" in model.client.default_headers


def test_non_muse_model_defaults_to_chat_completions(monkeypatch):
    _clear_session_env(monkeypatch)
    model = make_model(_opencode_endpoint())  # mimo-v2.5
    assert isinstance(model, _chat_model_type())


def test_api_style_env_forces_responses(monkeypatch):
    monkeypatch.setenv("OPENAI_API_STYLE", "responses")
    model = make_model(_endpoint())  # plain localhost, model "m"
    assert isinstance(model, _responses_model_type())


def test_api_style_env_forces_chat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_STYLE", "chat")
    endpoint = Endpoint(base_url="https://opencode.ai/zen/go/v1/",
                        api_key="sk", model="muse-spark-1.3-contributor")
    assert isinstance(make_model(endpoint), _chat_model_type())


def _responses_model_type():
    from pydantic_ai.models.openai import OpenAIResponsesModel
    return OpenAIResponsesModel


def _chat_model_type():
    from pydantic_ai.models.openai import OpenAIChatModel
    return OpenAIChatModel
