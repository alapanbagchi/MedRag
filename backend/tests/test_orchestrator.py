"""Master orchestrator tests: plan normalization, prompt, builder wiring."""

from __future__ import annotations

from src.agents.orchestrator import (
    MAX_SUBAGENTS,
    SHALLOW_SYSTEM,
    load_orchestrator_prompt,
    normalize_plan_items,
)


def test_orchestrator_prompt_loads_with_delegation_policy():
    prompt = load_orchestrator_prompt()
    assert "submit_plan" in prompt
    assert "spawn_subagent" in prompt
    assert "shallow" in prompt


def test_spawn_budget_is_bounded():
    assert isinstance(MAX_SUBAGENTS, int) and MAX_SUBAGENTS > 0


def test_shallow_prompt_declares_no_tools():
    assert "No tools" in SHALLOW_SYSTEM


def test_normalize_plan_items():
    assert normalize_plan_items(None) == []
    assert normalize_plan_items("nope") == []
    assert normalize_plan_items([{"question": "  "}]) == []
    items = normalize_plan_items([
        {"id": "T1", "question": "Deep q", "depth": "deep"},
        {"question": "Quick q"},
        "bare string task",
        {"question": "Legacy", "deep_research": True},
        {"id": "X", "question": "Shallow", "depth": "shallow"},
        42,
    ])
    assert [(i["id"], i["deep_research"]) for i in items] == [
        ("T1", True), ("T2", False), ("T3", False), ("T4", True), ("X", False),
    ]
    assert items[0]["question"] == "Deep q"
    assert items[2]["question"] == "bare string task"


def test_build_orchestrator_wires_all_tools(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    from pydantic_ai import Agent

    from src.agents import orchestrator as orchestrator_mod

    async def fake_spawn(**kwargs):
        return "findings"

    async def fake_plan(items):
        return "ack"

    from pydantic_ai.models.test import TestModel

    monkeypatch.setattr(orchestrator_mod, "make_model",
                        lambda *args, **kwargs: TestModel())
    # Agent construction validates every tool schema (Literal depth,
    # list-of-dict plan items) — this fails loudly on bad definitions.
    agent = orchestrator_mod.build_orchestrator(spawn_impl=fake_spawn,
                                                plan_impl=fake_plan)
    assert isinstance(agent, Agent)


def _orchestrator_tools(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    from pydantic_ai.models.test import TestModel

    from src.agents import orchestrator as orchestrator_mod

    async def fake_spawn(**kwargs):
        return "findings"

    monkeypatch.setattr(orchestrator_mod, "make_model",
                        lambda *args, **kwargs: TestModel())
    agent = orchestrator_mod.build_orchestrator(spawn_impl=fake_spawn)
    return (orchestrator_mod,
            {name for toolset in agent.toolsets for name in toolset.tools})


def test_orchestrator_plans_only_through_planner_subagent(monkeypatch):
    """The orchestrator never plans itself: no planning tool exists —
    planning happens by spawning the planner subagent (depth "plan")."""
    _, names = _orchestrator_tools(monkeypatch)
    assert "submit_plan" in names
    assert "spawn_subagent" in names
    assert "check_gaps" in names
    assert "synthesize" in names
    assert "generate_plan" not in names
    assert "plan_evidence_requirements" not in names
    assert "check_evidence_gaps" not in names


async def test_synthesize_uses_injected_streaming_impl(monkeypatch, tmp_path):
    """The adapter injects a streaming synthesize_impl — the tool must
    route through it (question + chat id) instead of running the
    synthesizer unstreamed."""
    from types import SimpleNamespace

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")

    from src.agents import orchestrator as orchestrator_mod
    from src.runstate.store import RunStore

    store = RunStore(tmp_path / "chats")
    await store.create_turn("c1", "t1", "Q?")
    monkeypatch.setattr(orchestrator_mod, "get_store", lambda: store)
    await store.mark_gap_checked("c1")

    async def fake_spawn(**kwargs):
        return "findings"

    seen = {}

    async def fake_synthesize(question, chat_id):
        seen["question"] = question
        seen["chat_id"] = chat_id
        return "streamed answer"

    agent = orchestrator_mod.build_orchestrator(
        spawn_impl=fake_spawn, synthesize_impl=fake_synthesize)
    tool = agent._function_toolset.tools["synthesize"]
    ctx = SimpleNamespace(deps=SimpleNamespace(chat_id="c1"))
    out = await tool.function(ctx)
    assert out == "streamed answer"
    assert seen["question"] == "Q?"
    assert seen["chat_id"] == "c1"


async def test_synthesize_denied_until_gap_checked(monkeypatch, tmp_path):
    """The synthesizer runs only behind the gap-checked mark."""
    from types import SimpleNamespace

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")

    from src.agents import orchestrator as orchestrator_mod
    from src.runstate.store import RunStore

    store = RunStore(tmp_path / "chats")
    await store.create_turn("c1", "t1", "Q?")
    # The tool binds get_store at import: patch its namespace.
    monkeypatch.setattr(orchestrator_mod, "get_store", lambda: store)

    async def fake_spawn(**kwargs):
        return "findings"

    agent = orchestrator_mod.build_orchestrator(spawn_impl=fake_spawn)
    tool = agent._function_toolset.tools["synthesize"]
    ctx = SimpleNamespace(deps=SimpleNamespace(chat_id="c1"))

    denied = await tool.function(ctx)
    assert "SYNTHESIZE DENIED" in denied
    assert "check_gaps" in denied

    await store.mark_gap_checked("c1")
    seen = {}

    class FakeSynth:
        async def run(self, prompt, *args, **kwargs):
            seen["prompt"] = prompt
            return SimpleNamespace(output="final answer")

    monkeypatch.setattr(orchestrator_mod, "build_synthesizer",
                        lambda: FakeSynth())
    out = await tool.function(ctx)
    assert out == "final answer"
    # The orchestrator hands the synthesizer one message containing the
    # question (and, from the ledger, all proof passages): the model runs
    # tool-free, nothing is fetched.
    assert "Q?" in seen["prompt"]
    assert "EVIDENCE" in seen["prompt"]


def test_synthesizer_build_pins_greedy(monkeypatch):
    import src.agents.synthesizer as synth_mod

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    agent = synth_mod.build_agent()
    assert agent.model.settings["temperature"] == 0.0
    prompt = synth_mod.load_prompt()
    assert "quote, no claim" in prompt


def test_synthesizer_takes_question_and_proofs_tool_free(monkeypatch):
    """One prompt, nothing fetched: the agent carries no tools and no
    deps — the question and every proof passage ride in the message."""
    import src.agents.synthesizer as synth_mod

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    agent = synth_mod.build_agent()
    names = {name for toolset in agent.toolsets for name in toolset.tools}
    assert names == set()
    prompt = synth_mod.load_prompt()
    assert "judge-verified proof passage" in prompt
    assert "quote, no claim" in prompt


def test_normalize_plan_items_keeps_planner_requirements():
    from src.agents.orchestrator import normalize_plan_items

    items = normalize_plan_items({
        "items": [{
            "id": "P1", "question": "Define hypertension.",
            "deep_research": True,
            "evidence_requirements": [
                {"id": "E1", "description": "definition"},
                {"id": "", "description": "dropped"},
                "junk",
            ],
        }],
    })
    assert items[0]["evidence_requirements"] == [
        {"id": "E1", "description": "definition"}]


def test_spawn_subagent_accepts_plan_depth(monkeypatch):
    """The planner is spawned, not tooled: depth "plan" is valid."""
    orchestrator_mod, _ = _orchestrator_tools(monkeypatch)

    async def fake_spawn(**kwargs):
        return "findings"

    agent = orchestrator_mod.build_orchestrator(spawn_impl=fake_spawn)
    tool = agent._function_toolset.tools["spawn_subagent"]
    schema = tool.tool_def.parameters_json_schema
    depth = schema["properties"]["depth"]
    assert set(depth.get("enum", [])) == {"deep", "shallow", "plan"}


async def test_run_planner_leg_returns_json_and_results(monkeypatch):
    """The spawned planner leg returns plan JSON plus its agent results
    so the caller can fold usage into the leg's allocation."""
    from types import SimpleNamespace

    import src.agents.planner as planner_mod
    from src.agents.planner import Plan, PlanItem
    from src.agents.stream_adapter import _run_planner_leg

    seen = {}
    result = SimpleNamespace(
        usage=lambda: SimpleNamespace(request_tokens=100,
                                      response_tokens=50))

    async def fake_generate(question, agent=None, usage_sink=None):
        seen["question"] = question
        seen["agent"] = agent
        if usage_sink is not None:
            usage_sink.append(result)
        return Plan(items=[PlanItem(id="P1", question=question,
                                    deep_research=True)])

    monkeypatch.setattr(planner_mod, "generate_plan", fake_generate)
    monkeypatch.setattr(planner_mod, "build_agent",
                        lambda: SimpleNamespace())
    out, used = await _run_planner_leg("What is hypertension?")
    assert seen["question"] == "What is hypertension?"
    assert seen["agent"] is not None  # pinned-sampling agent, not default
    assert '"deep_research":true' in out.replace(" ", "")
    assert used == [result]
