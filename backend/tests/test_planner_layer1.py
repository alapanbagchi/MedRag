"""Planner + deep-agent scaffolding tests (fake agents, no network)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from src.agents import __main__ as agents_main
from src.agents.deep_agent import (
    build_deep_agent,
    load_system_prompt,
    render_task_prompt,
    run_deep_task,
)
from src.lib.streaming import console_stream
from src.agents.planner import Plan, PlanItem, build_agent, generate_plan, load_prompt
from src.tools.umls import DeepDeps, lookup_medical_term
from src.umls.client import UMLSConcept


class FakePlanResult:
    def __init__(self, output: Plan):
        self.output = output


class FakePlanner:
    """Stand-in for a Pydantic AI planner; records prompts, returns a fixed Plan."""

    def __init__(self, plan: Plan):
        self._plan = plan
        self.seen: list = []

    async def run(self, prompt):
        self.seen.append(prompt)
        return FakePlanResult(self._plan)


class FakeStream:
    """Stand-in for StreamedRunResult; printing is driven by the event handler."""

    def __init__(self, output: str):
        self._output = output

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get_output(self):
        return self._output


class FakeDeepAgent:
    """Stand-in for a Pydantic AI deep agent; records prompts, returns hi."""

    def __init__(self, reply: str = "Hi! I got your task."):
        self._reply = reply
        self.seen: list = []
        self.kwargs: dict = {}

    def run_stream(self, prompt, *args, **kwargs):
        self.seen.append(prompt)
        self.kwargs = kwargs
        return FakeStream(self._reply)


def test_empty_plan_is_empty():
    assert Plan(items=[]).is_empty


def test_non_empty_plan_is_not_empty():
    plan = Plan(items=[PlanItem(id="P1", question="Q?")])
    assert not plan.is_empty


def test_item_defaults():
    item = PlanItem(question="Q?")
    assert item.id == "P1"
    assert item.deep_research is False


def test_item_deep_research_flag():
    item = PlanItem.model_validate({"id": "P1", "question": "Q?", "deep_research": True})
    assert item.deep_research is True


async def test_generate_plan_returns_agent_output():
    want = Plan.model_validate({"items": [{"id": "P1", "question": "Q?", "deep_research": True}]})
    plan = await generate_plan("What is diabetes?", agent=FakePlanner(want))
    assert plan == want


async def test_generate_plan_sends_query_to_agent():
    want = Plan.model_validate({"items": [{"id": "P1", "question": "Q?"}]})
    agent = FakePlanner(want)
    await generate_plan("What is diabetes?", agent=agent)
    assert len(agent.seen) == 1
    assert "What is diabetes?" in agent.seen[0]


async def test_generate_plan_streams_thinking_to_callback():
    from pydantic_ai.messages import PartDeltaEvent, ThinkingPartDelta

    want = Plan(items=[PlanItem(id="P1", question="Q?")])

    class ThinkingStream:
        def __init__(self, plan, deltas, handler):
            self._plan = plan
            self._deltas = deltas
            self._handler = handler

        async def __aenter__(self):
            async def _events():
                for d in self._deltas:
                    yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=d))

            await self._handler(None, _events())
            return self

        async def __aexit__(self, *args):
            return False

        async def get_output(self):
            return self._plan

    class ThinkingPlanner:
        def __init__(self, plan, deltas):
            self._plan = plan
            self._deltas = deltas

        def run_stream(self, prompt, *args, **kwargs):
            return ThinkingStream(self._plan, self._deltas,
                                  kwargs.get("event_stream_handler"))

    seen: list[str] = []

    async def on_thinking(delta: str) -> None:
        seen.append(delta)

    plan = await generate_plan("What is diabetes?",
                               agent=ThinkingPlanner(want, ["hmm", "ok"]),
                               on_thinking=on_thinking)
    assert plan == want
    assert seen == ["hmm", "ok"]


async def test_generate_plan_retries_rate_limit(monkeypatch):
    import src.agents.planner as planner_module

    monkeypatch.setattr(planner_module, "retry_delay", lambda *a, **k: 0.0)

    class Rate429(Exception):
        status_code = 429

    want = Plan.model_validate({"items": [{"id": "P1", "question": "Q?"}]})

    class FlakyPlanner(FakePlanner):
        def __init__(self, plan):
            super().__init__(plan)
            self.calls = 0

        async def run(self, prompt):
            self.calls += 1
            if self.calls < 3:
                raise Rate429("slow down")
            return await super().run(prompt)

    agent = FlakyPlanner(want)
    assert await generate_plan("What is diabetes?", agent=agent) == want
    assert agent.calls == 3


async def test_generate_plan_reraises_persistent_failure(monkeypatch):
    import src.agents.planner as planner_module

    monkeypatch.setattr(planner_module, "retry_delay", lambda *a, **k: 0.0)

    class DownPlanner(FakePlanner):
        def __init__(self, plan):
            super().__init__(plan)
            self.calls = 0

        async def run(self, prompt):
            self.calls += 1
            raise RuntimeError("endpoint down")

    agent = DownPlanner(Plan(items=[]))
    with pytest.raises(RuntimeError, match="endpoint down"):
        await generate_plan("What is diabetes?", agent=agent)
    assert agent.calls == 1


async def test_kick_retriever_warmup_never_raises(monkeypatch):
    import src.agents.deep_agent as deep_module
    import src.tools.retrieval as retrieval_mod

    def _boom():
        raise RuntimeError("no db here")

    monkeypatch.setattr(retrieval_mod, "get_retriever", _boom)
    deep_module.kick_retriever_warmup("P9")  # must not raise


async def test_generate_plan_empty_query_raises():
    with pytest.raises(ValueError):
        await generate_plan("  ", agent=FakePlanner(Plan()))


def test_build_agent_wires_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    agent = build_agent()
    assert isinstance(agent, Agent)
    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "test-model"


def test_build_agent_missing_env_raises(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        build_agent()


def test_prompt_declares_items_schema():
    prompt = load_prompt()
    assert '"items"' in prompt and '"id"' in prompt and '"question"' in prompt
    assert "deep_research" in prompt


async def test_run_deep_task_returns_greeting():
    agent = FakeDeepAgent("Hi! Working on it.")
    assert await run_deep_task("Find what diabetes is", agent=agent, warmup=False) == "Hi! Working on it."


async def test_run_deep_task_sends_question():
    agent = FakeDeepAgent()
    await run_deep_task("Find what diabetes is", agent=agent, warmup=False)
    assert "Find what diabetes is" in agent.seen[0]


async def test_run_deep_task_empty_question_raises():
    with pytest.raises(ValueError):
        await run_deep_task("  ", agent=FakeDeepAgent(), warmup=False)


async def test_run_deep_task_warms_retriever_by_default(monkeypatch):
    import src.agents.deep_agent as deep_module

    seen: list = []
    monkeypatch.setattr(deep_module, "kick_retriever_warmup",
                        lambda item_id="deep": seen.append(item_id))
    await run_deep_task("Find what diabetes is", agent=FakeDeepAgent(), item_id="P9")
    assert seen == ["P9"]


async def test_run_deep_task_skips_warmup_when_disabled(monkeypatch):
    import src.agents.deep_agent as deep_module

    seen: list = []
    monkeypatch.setattr(deep_module, "kick_retriever_warmup",
                        lambda item_id="deep": seen.append(item_id))
    await run_deep_task("Find what diabetes is", agent=FakeDeepAgent(), warmup=False)
    assert seen == []


class OverlapTask:
    """Stand-in for run_deep_task; records enter/exit around a yield point."""

    def __init__(self):
        self.log: list = []

    async def __call__(self, question, agent=None, item_id="deep", budget=None,
                       chat_id=None):
        self.log.append(("enter", question))
        await asyncio.sleep(0.01)
        self.log.append(("exit", question))
        return "Hi!"


def _stub_main(monkeypatch, plan):
    async def fake_plan(query):
        return plan

    monkeypatch.setattr(agents_main, "generate_plan", fake_plan)
    monkeypatch.setattr(agents_main, "build_deep_agent", lambda: FakeDeepAgent("Hi!"))
    monkeypatch.setattr(sys, "argv", ["src.agents", "some query"])


async def test_main_spawns_deep_and_skips_shallow(monkeypatch, capsys):
    _stub_main(monkeypatch, Plan(items=[
        PlanItem(id="P1", question="Q1?", deep_research=True),
        PlanItem(id="P2", question="Respond to hi", deep_research=False),
    ]))
    async def fake_task(question, agent=None, item_id="deep", budget=None,
                      chat_id=None):
        return "Hi!"

    monkeypatch.setattr(agents_main, "run_deep_task", fake_task)
    await agents_main.main()
    out = capsys.readouterr().out
    assert "spawning deep agent" in out
    assert "deep agent replied: Hi!" in out
    assert "shallow task, no deep agent" in out


async def test_main_runs_deep_items_concurrently(monkeypatch, capsys):
    _stub_main(monkeypatch, Plan(items=[
        PlanItem(id="P1", question="Q1?", deep_research=True),
        PlanItem(id="P2", question="Q2?", deep_research=True),
    ]))
    tracker = OverlapTask()
    monkeypatch.setattr(agents_main, "run_deep_task", tracker)
    await agents_main.main()
    kinds = [kind for kind, _ in tracker.log]
    # Both tasks enter before either exits; sequential awaits would interleave.
    assert kinds == ["enter", "enter", "exit", "exit"]


async def test_main_missing_query_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["src.agents"])
    with pytest.raises(SystemExit):
        await agents_main.main()


def test_build_deep_agent_wires_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    agent = build_deep_agent()
    assert isinstance(agent, Agent)
    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "test-qwen"


def test_build_deep_agent_missing_env_raises(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        build_deep_agent()


class FakeUmls:
    """Stand-in for UMLSClient; returns a fixed concept."""

    def __init__(self, concept: UMLSConcept):
        self._concept = concept
        self.seen: list = []

    async def search_concept(self, term):
        self.seen.append(term)
        return self._concept


def _tool_ctx(concept: UMLSConcept):
    return SimpleNamespace(deps=DeepDeps(umls=FakeUmls(concept)))


async def test_lookup_medical_term_returns_json():
    concept = UMLSConcept(
        term="diabetes",
        found=True,
        cui="C0011849",
        preferred_name="Diabetes Mellitus",
        synonyms=["DM", "diabetes mellitus NOS"],
    )
    data = json.loads(await lookup_medical_term(_tool_ctx(concept), "diabetes"))
    assert data["found"] is True
    assert data["preferred_name"] == "Diabetes Mellitus"
    assert data["cui"] == "C0011849"
    assert "DM" in data["synonyms"]


async def test_lookup_medical_term_no_match_is_json():
    out = await lookup_medical_term(_tool_ctx(UMLSConcept(term="xyzzy")), "xyzzy")
    data = json.loads(out)
    assert data["found"] is False
    assert data["term"] == "xyzzy"


async def test_lookup_medical_term_blank_is_json():
    data = json.loads(await lookup_medical_term(_tool_ctx(UMLSConcept(term="")), "  "))
    assert data["found"] is False


def test_subagent_prompts_live_outside_agents():
    root = Path(__file__).resolve().parent.parent / "src" / "prompts"
    assert (root / "subagent_system.txt").exists()
    assert (root / "subagent_task.txt").exists()


def test_render_task_prompt_includes_question():
    assert "Find what diabetes is" in render_task_prompt("  Find what diabetes is  ")


def test_system_prompt_mentions_umls_tool():
    assert "lookup_medical_term" in load_system_prompt()


def test_system_prompt_orders_evidence_gathering():
    prompt = load_system_prompt()
    assert "local_search" in prompt
    assert "verbatim quotes" in prompt


def test_system_prompt_asks_parallel_tool_calls():
    assert "same block" in load_system_prompt()


async def test_run_deep_task_passes_umls_deps():
    umls = FakeUmls(UMLSConcept(term="diabetes", found=True))
    agent = FakeDeepAgent()
    await run_deep_task("Find what diabetes is", agent=agent, umls=umls)
    assert agent.kwargs["deps"].umls is umls


async def test_run_deep_task_returns_output(capsys):
    agent = FakeDeepAgent(reply="Hi there!")
    assert await run_deep_task("Find what diabetes is", agent=agent) == "Hi there!"


async def test_console_stream_blocks_answer(capsys):
    agent = Agent(TestModel())
    async with agent.run_stream("hi", event_stream_handler=console_stream("P9")) as result:
        await result.get_output()
    out = capsys.readouterr().out
    assert out.count("|answer|") == 2


async def test_console_stream_logs_verdict(capsys):
    from pydantic_ai.messages import OutputToolCallEvent, ToolCallPart

    async def one(event):
        yield event

    handler = console_stream("P9")
    await handler(None, one(OutputToolCallEvent(
        part=ToolCallPart(tool_name="final", args={"usable": True}),
    )))
    assert "[P9] verdict:" in capsys.readouterr().out


async def test_console_stream_logs_tool_calls(capsys):
    async def probe_tool(term: str) -> str:
        """Probe tool."""
        return f"RESULT for {term}"

    agent = Agent(TestModel(), tools=[probe_tool])
    async with agent.run_stream("hi", event_stream_handler=console_stream("P9")) as result:
        await result.get_output()
    out = capsys.readouterr().out
    assert "[P9] calling tool probe_tool" in out
    assert "RESULT for a" in out


def test_build_deep_agent_registers_umls_tool(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    agent = build_deep_agent()
    names = [name for toolset in agent.toolsets for name in toolset.tools]
    assert "lookup_medical_term" in names


async def test_generate_plan_collects_usage_sink():
    """The orchestrator folds these results into its master budget —
    planning is billed, never free."""
    want = Plan(items=[PlanItem(id="P1", question="Q?")])
    sink: list = []
    plan = await generate_plan("What is diabetes?", agent=FakePlanner(want),
                               usage_sink=sink)
    assert plan == want
    assert len(sink) == 1


async def test_generate_plan_parses_planner_requirements():
    want = Plan.model_validate({"items": [{
        "id": "P1", "question": "Q?", "deep_research": True,
        "evidence_requirements": [{"id": "E1", "description": "definition"}],
    }]})
    plan = await generate_plan("What is diabetes?", agent=FakePlanner(want))
    assert plan.items[0].evidence_requirements[0].id == "E1"


def test_planner_prompt_demands_requirements():
    prompt = load_prompt()
    assert "evidence_requirements" in prompt
    assert '"E1"' in prompt


def test_planner_prompt_forbids_question_bloat():
    """Task questions stay short information needs: no memorized
    examples, no output-format boilerplate — that bloat is what made
    spawned plans diverge from manual ones."""
    prompt = load_prompt()
    assert "under 25 words" in prompt
    assert "verbatim quotes" in prompt
    assert "BAD question" in prompt and "GOOD question" in prompt


def test_build_agent_pins_sampling(monkeypatch):
    """Every delegation gets the pinned deck: temp 0, seed 43, top_p 0.1."""
    import src.agents.planner as planner_mod

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    agent = planner_mod.build_agent()
    settings = agent.model.settings
    assert settings["temperature"] == 0.0
    assert settings["seed"] == 43
    assert settings["top_p"] == 0.1


def test_no_langchain_dependency():
    root = Path(__file__).resolve().parent.parent / "src"
    for name in ("agents/planner.py", "agents/deep_agent.py", "agents/__main__.py", "tools/umls.py", "tools/retrieval.py", "tools/firecrawl.py", "tools/verifier.py", "middleware/llm_as_a_judge.py", "middleware/trust.py"):
        source = (root / name).read_text(encoding="utf-8").lower()
        assert "langchain" not in source, f"{name} still imports langchain"
