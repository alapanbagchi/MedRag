"""Run-state ledger tests: per-chat JSON with per-requirement evidence.

No LLM env required. Store tests use isolated tmp dirs; the global
default store is reset after each test that touches it.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.runstate.models import (
    ChatState,
    RunStatus,
    TaskState,
)
from src.runstate.progress import progress_view, render_receipt
from src.runstate.store import (
    RunStore,
    get_store,
    record_evidence,
    record_gaps,
    record_requirements,
    set_store,
)
from src.runstate.tools import run_progress


@pytest.fixture()
def store(tmp_path):
    return RunStore(tmp_path / "chats")


@pytest.fixture(autouse=True)
def _reset_global_store():
    set_store(None)
    yield
    set_store(None)


def _verdict(pid, coverage=("E1",), intent=0.9):
    return SimpleNamespace(passage_id=pid, coverage=list(coverage),
                           intent_score=intent, reason="supports E1")


def _ctx(chat_id="c1", task_id="T1"):
    return SimpleNamespace(deps=SimpleNamespace(chat_id=chat_id, task_id=task_id))


# --- schema -----------------------------------------------------------------------

def test_chat_schema_round_trip():
    chat = ChatState(chat_id="c1")
    loaded = ChatState.from_json(chat.to_json())
    assert loaded.chat_id == "c1"
    assert loaded.turn_order == []


def test_task_coverage_from_nested_evidence():
    from src.runstate.models import EvidenceRecord, TaskRecord

    task = TaskRecord(id="T1", evidence_requirements=[
        {"id": "E1", "description": "d1", "supporting_evidence": [
            {"passage_id": "p1", "requirement_ids": ["E1"]},
            {"passage_id": "p2", "requirement_ids": ["E1", "E2"]},
        ], "gaps": []},
        {"id": "E2", "description": "d2"},
    ])
    assert task.verified_count == 2
    assert task.coverage_by_requirement() == {"E1": 2, "E2": 0}
    assert task.open_gaps() == []


# --- store lifecycle -----------------------------------------------------------------

async def test_record_plan_persists_planner_requirements(store):
    """Planner-shaped items land in the ledger with their requirements —
    the pre-plan contract sub-agents execute against."""
    await store.create_turn("c1", "t1", "What is hypertension?")
    await store.record_plan("c1", [{
        "id": "T1", "question": "Define hypertension.",
        "deep_research": True,
        "evidence_requirements": [
            {"id": "E1", "description": "definition"},
            {"id": "E2", "description": "criteria"},
            {"no-id": True},
        ],
    }])
    task = store.get_chat("c1").turns["t1"].task("T1")
    assert [(r.id, r.description) for r in task.evidence_requirements] == [
        ("E1", "definition"), ("E2", "criteria")]
    # Re-planning the same task keeps already-collected evidence.
    from src.runstate.models import EvidenceRecord
    await store.record_evidence("c1", [EvidenceRecord(
        passage_id="p1", requirement_ids=["E1"], text="def…")],
        task_id="T1")
    await store.record_plan("c1", [{
        "id": "T1", "question": "Define hypertension (v2).",
        "deep_research": True,
        "evidence_requirements": [{"id": "E1", "description": "definition"}],
    }])
    task = store.get_chat("c1").turns["t1"].task("T1")
    assert task.task == "Define hypertension (v2)."
    assert [r.passage_id for r in task.evidence_requirements[0].supporting_evidence] == ["p1"]


async def test_check_gaps_lists_uncovered_requirements(store, monkeypatch):
    """Mechanical coverage rule: covered = at least one verified passage.
    The orchestrator calls this after all legs resolve."""
    import json

    import src.runstate.tools as tools_mod
    from src.runstate.models import EvidenceRecord
    from src.runstate.tools import check_gaps

    await store.create_turn("c1", "t1", "Q?")
    await store.record_plan("c1", [
        {"id": "T1", "question": "a?", "deep_research": True,
         "evidence_requirements": [{"id": "E1", "description": "d1"},
                                   {"id": "E2", "description": "d2"}]},
        {"id": "T2", "question": "b?", "deep_research": False,
         "evidence_requirements": [{"id": "E3", "description": "d3"}]},
    ])
    await store.record_evidence("c1", [EvidenceRecord(
        passage_id="p1", requirement_ids=["E1"], text="ev")], task_id="T1")
    monkeypatch.setattr(tools_mod, "get_store", lambda: store)
    ctx = SimpleNamespace(deps=SimpleNamespace(chat_id="c1"))
    uncovered = json.loads(await check_gaps(ctx))
    assert uncovered == [
        {"task_id": "T1", "task_state": "planned",
         "requirement_id": "E2", "description": "d2"},
        {"task_id": "T2", "task_state": "planned",
         "requirement_id": "E3", "description": "d3"},
    ]
    # check_gaps marks the turn: the synthesizer gate opens ...
    assert store.get_chat("c1").turns["t1"].gap_checked is True
    # ... and any later task finish or plan change closes it again.
    from src.runstate.models import TaskState
    await store.record_task_finished("c1", "T1", TaskState.DONE)
    assert store.get_chat("c1").turns["t1"].gap_checked is False
    await store.mark_gap_checked("c1")
    assert store.get_chat("c1").turns["t1"].gap_checked is True
    await store.record_plan("c1", [{"id": "T1", "question": "a?"}])
    assert store.get_chat("c1").turns["t1"].gap_checked is False
    assert await store.mark_gap_checked("nope") is False


async def test_check_gaps_empty_and_missing_chat(store, monkeypatch):
    import src.runstate.tools as tools_mod
    from src.runstate.tools import check_gaps

    monkeypatch.setattr(tools_mod, "get_store", lambda: store)
    ctx = SimpleNamespace(deps=SimpleNamespace(chat_id="nope"))
    assert "No run state found" in await check_gaps(ctx)
    ctx = SimpleNamespace(deps=SimpleNamespace(chat_id=None))
    assert "No run state is attached" in await check_gaps(ctx)


def test_task_sources_groups_by_document():
    from types import SimpleNamespace as NS

    from src.agents.stream_adapter import _task_sources

    task = NS(evidence_requirements=[
        NS(id="E1", description="d", supporting_evidence=[
            NS(passage_id="p1", document_id="PMC42", chunk_id="c1",
               url="", section="Results", text="t"),
            NS(passage_id="p2", document_id="PMC42", chunk_id="c2",
               url="", section="Results", text="t"),
        ], gaps=[]),
        NS(id="E2", description="d", supporting_evidence=[
            NS(passage_id="p3", document_id="web:example",
               chunk_id="", url="https://example.com/guideline",
               section="", text="t"),
        ], gaps=[]),
    ])
    sources = _task_sources(task)
    assert len(sources) == 2
    local, web = sources
    assert local["pmcid"] == "PMC42"
    assert local["url"] is None
    assert local["passage_ids"] == ["p1", "p2"]
    assert web["pmcid"] is None
    assert web["url"] == "https://example.com/guideline"
    assert web["passage_ids"] == ["p3"]
    assert _task_sources(NS(evidence_requirements=[])) == []
    assert _task_sources(None) == []


def test_requirements_block_renders_ledger_contract():
    from types import SimpleNamespace as NS

    from src.agents.stream_adapter import _requirements_block
    assert _requirements_block(None) == ""
    assert _requirements_block(NS(evidence_requirements=[])) == ""
    task = NS(evidence_requirements=[
        NS(id="E1", description="definition"),
        NS(id="E2", description="criteria"),
    ])
    block = _requirements_block(task)
    assert "- E1: definition" in block
    assert "- E2: criteria" in block
    assert "do not invent your own" in block


async def test_full_task_lifecycle(store):
    from src.runstate.models import EvidenceRecord

    await store.create_turn("c1", "t1", "How does diet impact hypertension?")
    await store.record_plan("c1", [{"id": "T1", "question": "factors?",
                                   "deep_research": True}])
    await store.record_task_started("c1", "T1", "factors?", "deep")
    await store.record_requirements(
        "c1", [{"id": "E1", "description": "diet factors"},
               {"id": "E2", "description": "magnitudes"}],
        task_id="T1")
    added = await store.record_evidence("c1", [
        EvidenceRecord(passage_id="p1", requirement_ids=["E1"],
                       tool="retrieve_evidence", text="sodium…"),
        EvidenceRecord(passage_id="p1", requirement_ids=["E1"], text="dup"),
        EvidenceRecord(passage_id="p2", requirement_ids=["E2"], text="3mmHg"),
    ], task_id="T1")
    assert added == 2  # deduped by passage id
    await store.record_gaps("c1", [{"evidence_id": "E2", "missing": "potassium?"}],
                            task_id="T1")
    await store.record_task_finished(
        "c1", "T1", TaskState.DONE,
        budget_allocated={"max_tool_calls": 20},
        budget_expenditure_history=[{"event": "budget_consumed"}],
        budget_remaining={"tool_calls": 14})

    chat = store.get_chat("c1")
    assert chat.turn_order == ["t1"]
    turn = chat.turns["t1"]
    assert turn.plan[0].id == "T1"
    assert turn.plan[0].task == "factors?"
    task = turn.task("T1")
    assert task.state == TaskState.DONE
    assert task.verified_count == 2
    e1 = next(r for r in task.evidence_requirements if r.id == "E1")
    assert [e.passage_id for e in e1.supporting_evidence] == ["p1"]
    e2 = next(r for r in task.evidence_requirements if r.id == "E2")
    assert [e.passage_id for e in e2.supporting_evidence] == ["p2"]
    assert e2.gaps[0].missing == "potassium?"
    assert task.open_gaps() == ["E2"]
    assert task.budget_allocated == {"max_tool_calls": 20}
    assert task.budget_expenditure_history == [{"event": "budget_consumed"}]
    assert task.budget_remaining == {"tool_calls": 14}
    assert turn.run_verified_count == 2


async def test_replanning_keeps_collected_evidence(store):
    from src.runstate.models import EvidenceRecord

    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "old"}],
                                    task_id="T1")
    await store.record_evidence("c1", [EvidenceRecord(
        passage_id="p1", requirement_ids=["E1"], text="kept")], task_id="T1")
    await store.record_requirements("c1", [{"id": "E1", "description": "new"},
                                           {"id": "E2", "description": "added"}],
                                    task_id="T1")
    task = store.get_chat("c1").turns["t1"].task("T1")
    assert [r.id for r in task.evidence_requirements] == ["E1", "E2"]
    assert task.evidence_requirements[0].description == "new"
    assert [e.passage_id for e in
            task.evidence_requirements[0].supporting_evidence] == ["p1"]


async def test_turns_accumulate_per_chat(store):
    await store.create_turn("c1", "t1", "first?")
    await store.create_turn("c1", "t2", "follow-up?")
    chat = store.get_chat("c1")
    assert chat.turn_order == ["t1", "t2"]
    assert chat.current.run_id == "t2"


async def test_unknown_chat_writes_are_noops(store):
    await store.record_plan("nope", [])
    await store.record_task_started("nope", "T1")
    assert store.get_chat("nope") is None


async def test_persists_to_disk_and_reloads(tmp_path):
    first = RunStore(tmp_path / "chats")
    await first.create_turn("c9", "t1", "Q?")
    await first.record_task_started("c9", "T1", "q")
    assert (tmp_path / "chats" / "c9.json").is_file()

    second = RunStore(tmp_path / "chats")
    loaded = second.get_chat("c9")
    assert loaded is not None
    assert loaded.turns["t1"].task("T1").state == TaskState.IN_PROGRESS


async def test_concurrent_evidence_writes_dont_lose_data(store):
    await store.create_turn("cc", "t1", "Q?")
    await store.record_task_started("cc", "T1", "q")
    await store.record_requirements("cc", [{"id": "E1", "description": "d"}],
                                    task_id="T1")
    from src.runstate.models import EvidenceRecord

    await asyncio.gather(*[
        store.record_evidence("cc", [EvidenceRecord(
            passage_id=f"p{i}", requirement_ids=["E1"])], task_id="T1")
        for i in range(20)
    ])
    task = store.get_chat("cc").turns["t1"].task("T1")
    assert task.verified_count == 20


# --- writer hooks -----------------------------------------------------------------------

async def test_record_evidence_routes_by_coverage(store, monkeypatch):
    import src.runstate.store as store_mod

    monkeypatch.setattr(store_mod, "get_store", lambda: store)
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "d1"},
                                           {"id": "E2", "description": "d2"}],
                                    task_id="T1")

    items = [
        {"chunk_id": "p1", "document_id": "PMC1", "section": "Results",
         "text": "sodium raises pressure", "url": ""},
        {"chunk_id": "p2", "text": "unjudged filler"},
    ]
    added = await record_evidence(_ctx(), "retrieve_evidence", items,
                                  [_verdict("p1", coverage=["E1", "E2"])])
    assert added == 1
    task = store.get_chat("c1").turns["t1"].task("T1")
    by_id = {req.id: req for req in task.evidence_requirements}
    assert [e.passage_id for e in by_id["E1"].supporting_evidence] == ["p1"]
    assert [e.passage_id for e in by_id["E2"].supporting_evidence] == ["p1"]
    assert by_id["E1"].supporting_evidence[0].document_id == "PMC1"


async def test_record_evidence_ignores_unmatched_coverage(store, monkeypatch):
    """Coverage naming no planned requirement must not be attributed to the
    first requirement: that would report coverage the judge never confirmed."""
    import src.runstate.store as store_mod

    monkeypatch.setattr(store_mod, "get_store", lambda: store)
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "d1"}],
                                    task_id="T1")
    added = await record_evidence(
        _ctx(), "retrieve_evidence",
        [{"chunk_id": "p1", "text": "unrelated"}],
        [_verdict("p1", coverage=["GENERAL"])])
    assert added == 0
    task = store.get_chat("c1").turns["t1"].task("T1")
    assert task.evidence_requirements[0].supporting_evidence == []


async def test_record_evidence_inert_without_chat_id(store, monkeypatch):
    import src.runstate.store as store_mod

    monkeypatch.setattr(store_mod, "get_store", lambda: store)
    added = await record_evidence(_ctx(chat_id=None), "retrieve_evidence",
                                  [{"chunk_id": "p1"}], [_verdict("p1")])
    assert added == 0


async def test_record_evidence_stores_web_listing_snippet(store, monkeypatch):
    """Web-search listing items carry no `text` key (snippet only). Without
    the fallback their recorded text is empty and websites never reach
    the synthesizer — this pins store text = text→snippet→content→markdown."""
    import src.runstate.store as store_mod

    monkeypatch.setattr(store_mod, "get_store", lambda: store)
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "d1"}],
                                    task_id="T1")
    url = "https://example.com/guideline"
    items = [{"url": url, "title": "Guideline",
              "snippet": "The DASH diet reduces systolic blood pressure by 11 mmHg."}]
    added = await record_evidence(_ctx(), "firecrawl_web_search", items,
                                  [_verdict(url)])
    assert added == 1
    rec = store.get_chat("c1").turns["t1"].task("T1") \
        .evidence_requirements[0].supporting_evidence[0]
    assert rec.url == url
    assert rec.title == "Guideline"
    assert "11 mmHg" in (rec.text or "")
    assert rec.ref


async def test_record_gaps_route_to_requirements(store, monkeypatch):
    import src.runstate.store as store_mod

    monkeypatch.setattr(store_mod, "get_store", lambda: store)
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "d1"},
                                           {"id": "E2", "description": "d2"}],
                                    task_id="T1")
    await record_gaps(_ctx(), [{"evidence_id": "E2", "missing": "dose?"}])
    task = store.get_chat("c1").turns["t1"].task("T1")
    by_id = {req.id: req for req in task.evidence_requirements}
    assert by_id["E1"].gaps == []
    assert by_id["E2"].gaps[0].missing == "dose?"


# --- progress reads -------------------------------------------------------------------------

async def test_progress_view_has_counts_no_text(store):
    await store.create_turn("c1", "t1", "Q?")
    await store.record_plan("c1", [{"id": "T1", "question": "q", "deep_research": True}])
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "d"},
                                           {"id": "E2", "description": "d2"}],
                                    task_id="T1")
    from src.runstate.models import EvidenceRecord

    await store.record_evidence("c1", [
        EvidenceRecord(passage_id="p1", requirement_ids=["E1"], text="full text here"),
        EvidenceRecord(passage_id="p2", requirement_ids=["E1"], text="more text"),
    ], task_id="T1")
    await store.record_gaps("c1", [{"evidence_id": "E2", "missing": "x"}],
                            task_id="T1")
    view = progress_view(store.get_chat("c1"), "t1")
    assert view["chat_id"] == "c1"
    assert view["turn_verified"] == 2
    (entry,) = view["plan"]
    assert entry["evidence_requirements"] == [{"id": "E1", "verified": 2,
                                               "gaps": []},
                                              {"id": "E2", "verified": 0,
                                               "gaps": ["E2"]}]
    assert view["unresolved_gaps"] == [{"task_id": "T1", "evidence_id": "E2",
                                        "missing": "x"}]
    assert view["previous_turns"] == []
    assert "full text here" not in json.dumps(view)


async def test_progress_view_lists_previous_turns(store):
    await store.create_turn("c1", "t1", "first?")
    await store.create_turn("c1", "t2", "second?")
    view = progress_view(store.get_chat("c1"), "t2")
    assert view["previous_turns"] == [{"turn_id": "t1", "question": "first?",
                                       "status": "running", "verified": 0}]


async def test_render_receipt_counts_not_text(store):
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "d"}],
                                    task_id="T1")
    from src.runstate.models import EvidenceRecord

    for i in range(4):
        await store.record_evidence("c1", [EvidenceRecord(
            passage_id=f"p{i}", requirement_ids=["E1"], text="secret ...)")],
            task_id="T1")
    await store.record_task_finished("c1", "T1", TaskState.DONE)
    receipt = render_receipt(store.get_chat("c1"), "t1", "T1")
    assert "4 verified" in receipt
    assert "E1:4" in receipt
    assert "secret" not in receipt
    await store.record_task_finished("c1", "T1", TaskState.FAILED)
    bad = render_receipt(store.get_chat("c1"), "t1", "T1")
    assert "task_state=failed" in bad


async def test_run_progress_tool_reads_ledger(store, monkeypatch):
    import src.runstate.tools as tools_mod

    monkeypatch.setattr(tools_mod, "get_store", lambda: store)
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    out = json.loads(await run_progress(_ctx()))
    assert out["chat_id"] == "c1"
    assert out["plan"][0]["id"] == "T1"


async def test_run_progress_task_fetch_returns_evidence(store, monkeypatch):
    """Detail pull is per-task: texts only when synthesizing."""
    import src.runstate.tools as tools_mod

    monkeypatch.setattr(tools_mod, "get_store", lambda: store)
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "d"}],
                                    task_id="T1")
    from src.runstate.models import EvidenceRecord

    await store.record_evidence("c1", [EvidenceRecord(
        passage_id="p1", requirement_ids=["E1"], text="secret text")],
        task_id="T1")
    default = json.loads(await run_progress(_ctx()))
    assert "secret text" not in json.dumps(default)
    detail = json.loads(await run_progress(_ctx(), task_id="T1"))
    assert detail["evidence_requirements"][0]["supporting_evidence"][0]["text"] == \
        "secret text"
    assert detail["budget_allocated"] is None
    missing = await run_progress(_ctx(), task_id="T9")
    assert "No task T9" in missing


async def test_run_progress_tool_free_and_inert():
    from src.budget.costs import ToolCostModel

    assert ToolCostModel().cost_for("run_progress").tool_calls == 0
    ctx = SimpleNamespace(deps=SimpleNamespace(chat_id=None))
    assert "No run state" in await run_progress(ctx)


class _FakeNoul:
    def __init__(self, noul):
        self.noul = noul


class _FakeUsage:
    input_tokens = 1
    output_tokens = 1


class _FakeResponse:
    def __init__(self, nouls):
        self.nouls = nouls
        self.choices = {}
        self.usage = _FakeUsage()


class _FakeJudge:
    """AsyncTypeSafeClient stand-in: reports coverage for known passages."""

    def __init__(self, report):
        self._report = report

    async def system_one(self, state, questions):
        pid = state["passage"]["id"]
        desired = {str(v.passage_id): v
                   for v in self._report.evidence_results or []}
        verdict = desired.get(pid)
        nouls = {}
        for key in questions:
            if key == "intent":
                nouls[key] = _FakeNoul(
                    verdict.intent_score if verdict else 0.0)
            elif key.startswith("coverage::"):
                rid = key.split("::", 1)[1]
                covered = bool(verdict and rid in (verdict.coverage or []))
                nouls[key] = _FakeNoul(1.0 if covered else 0.0)
        return _FakeResponse(nouls)


async def test_judge_writes_kept_evidence_under_requirement(store, monkeypatch):
    """Real judge middleware + fake judge LLM: kept passages land nested."""
    import src.runstate.store as store_mod
    from src.middleware.llm_as_a_judge import LLMAsJudge
    from src.tools.verifier import VerdictResponse

    monkeypatch.setattr(store_mod, "get_store", lambda: store)
    await store.create_turn("c1", "t1", "Q?")
    await store.record_task_started("c1", "T1", "q")
    await store.record_requirements("c1", [{"id": "E1", "description": "sodium"}],
                                    task_id="T1")

    report = VerdictResponse.model_validate({"evidence_results": [
        {"passage_id": "p1", "intent_score": 0.9,
         "coverage": ["E1"], "reason": "covers E1"}]})
    capability = LLMAsJudge(tools=["retrieve_evidence"],
                            judge_client=_FakeJudge(report))
    ctx = SimpleNamespace(deps=SimpleNamespace(
        chat_id="c1", task_id="T1", kept_passages=[], notes=[]))
    call = SimpleNamespace(tool_name="retrieve_evidence", tool_call_id="tc1")

    async def _handler(args):
        return json.dumps([{"chunk_id": "p1", "document_id": "PMC1",
                            "section": "Results", "text": "sodium passage"}])

    out = await capability.wrap_tool_execute(
        ctx, call=call, tool_def=None,
        args={"query": "sodium?",
              "evidence_requirements": [{"id": "E1",
                                          "description": "sodium effects"}]},
        handler=_handler)
    assert "[EVIDENCE JUDGMENT]" in out
    task = store.get_chat("c1").turns["t1"].task("T1")
    (record,) = task.evidence_requirements[0].supporting_evidence
    assert record.passage_id == "p1"
    assert record.document_id == "PMC1"


# --- full-run integration -----------------------------------------------------

class _ScriptedResult:
    def __init__(self, events, output="out"):
        self._events = events
        self.output = output

    def bind(self, handler):
        self._handler = handler
        return self

    async def drive(self):
        await self._handler(None, self._events)
        return self

    def usage(self):
        return SimpleNamespace(request_tokens=1, response_tokens=1)


class _ScriptedAgent:
    def __init__(self, events, output="out"):
        self._events = events
        self._output = output

    async def run(self, prompt, *args, **kwargs):
        return await _ScriptedResult(
            self._events, self._output
        ).bind(kwargs.get("event_stream_handler")).drive()


async def _no_events():
    if False:  # pragma: no cover
        yield


def _scripted_brain(*responses):
    import json as _json

    from pydantic_ai.models.function import (
        DeltaToolCall,
        FunctionModel,
    )

    calls = {"n": 0}

    async def _stream(messages, info):
        i = calls["n"]
        calls["n"] += 1
        for chunk in responses[min(i, len(responses) - 1)]:
            if chunk[0] == "text":
                yield chunk[1]
            else:
                _, name, args, call_id = chunk
                yield {0: DeltaToolCall(name=name, json_args=_json.dumps(args),
                                        tool_call_id=call_id)}

    return FunctionModel(stream_function=_stream)


async def test_stream_run_fills_chat_ledger_file(monkeypatch, tmp_path):
    """Scripted plan → spawn → done: per-chat JSON with budget ledger."""
    from types import SimpleNamespace

    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod
    from src.agents.stream_adapter import stream_deep_agent

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    monkeypatch.setenv("MEDRAG_RUNSTATE_DIR", str(tmp_path / "chats"))
    monkeypatch.setattr(
        orchestrator_mod, "make_model",
        lambda *args, **kwargs: _scripted_brain(
            [("tool", "submit_plan",
              {"items": [{"id": "T1", "question": "Deep task", "depth": "deep"}]},
              "m1")],
            [("tool", "spawn_subagent",
              {"task": "Deep task", "depth": "deep", "task_id": "T1"}, "m2")],
            [("text", "FINAL")]))
    monkeypatch.setattr(deep_agent_mod, "build_deep_agent",
                        lambda: _ScriptedAgent(_no_events(), "done researching"))
    monkeypatch.setattr(orchestrator_mod, "build_shallow_agent",
                        lambda: _ScriptedAgent(_no_events(), "quick"))

    events = [e async for e in stream_deep_agent(
        "diet question?", umls=SimpleNamespace(), warmup=False,
        conversation_id="chat9")]
    done = events[-1]
    assert done["type"] == "done"
    assert set(done["usage"]) == {"prompt", "completion"}

    # The orchestrator steered from this: arrays only, no evidence text.
    progress = done["progress"]
    assert progress["chat_id"] == "chat9"
    assert progress["plan"] == [{"id": "T1", "task": "Deep task",
                                 "state": "done", "depth": "deep",
                                 "verified": 0, "evidence_requirements": []}]

    # The spawn return the orchestrator actually read is lean.
    spawn_results = [e for e in events if e.get("type") == "tool_result"
                     and e.get("name") == "spawn_subagent"]
    assert len(spawn_results) == 1
    returned = json.dumps(spawn_results[0].get("result", ""))
    assert "[RUN STATE]" in returned
    assert "done researching" not in returned

    # …and the full JSON sits on the server under the chat id.
    ledger_file = tmp_path / "chats" / "chat9.json"
    assert ledger_file.is_file()
    chat = ChatState.from_json(ledger_file.read_text(encoding="utf-8"))
    assert [t.question for t in chat.turns.values()] == ["diet question?"]
    turn = chat.turns[done["run_id"]]
    assert turn.plan[0].id == "T1"
    assert turn.plan[0].task == "Deep task"
    assert turn.status.value == "complete"
    task = turn.task("T1")
    assert task.state == TaskState.DONE
    # Budget ledger populated from the task allocation (never null).
    assert task.budget_allocated["max_tool_calls"] == 20
    assert task.budget_remaining["tool_calls"] <= 20
    assert isinstance(task.budget_expenditure_history, list)


def test_builders_register_run_progress(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("SMALL_MODEL", "test-small")
    from pydantic_ai.models.test import TestModel

    from src.agents import deep_agent as deep_agent_mod
    from src.agents import orchestrator as orchestrator_mod

    monkeypatch.setattr(deep_agent_mod, "make_model",
                        lambda *args, **kwargs: TestModel())
    monkeypatch.setattr(orchestrator_mod, "make_model",
                        lambda *args, **kwargs: TestModel())

    for agent in (deep_agent_mod.build_deep_agent(),
                  orchestrator_mod.build_orchestrator(spawn_impl=None,
                                                      plan_impl=None)):
        assert "run_progress" in agent._function_toolset.tools.keys()


async def test_record_evidence_assigns_turn_wide_citation_refs(store):
    from src.runstate.models import EvidenceRecord

    await store.create_turn("c1", "t1", "Q?")
    await store.record_plan("c1", [
        {"id": "T1", "question": "a?",
         "evidence_requirements": [{"id": "E1", "description": "d1"}]},
        {"id": "T2", "question": "b?",
         "evidence_requirements": [{"id": "E2", "description": "d2"}]},
    ])
    await store.record_evidence("c1", [
        EvidenceRecord(passage_id="chunk-a", requirement_ids=["E1"],
                       url="", text="paper text"),
        EvidenceRecord(passage_id="https://example.com/g", requirement_ids=["E1"],
                       url="https://example.com/g", text="web text"),
    ], task_id="T1")
    await store.record_evidence("c1", [EvidenceRecord(
        passage_id="chunk-b", requirement_ids=["E2"], text="more")],
        task_id="T2")
    turn = store.get_chat("c1").turns["t1"]
    refs = [r.ref for t in turn.plan for q in t.evidence_requirements
            for r in q.supporting_evidence]
    assert refs == ["P1", "P2", "P3"]
    # task_detail (what the synthesizer reads) exposes the refs
    from src.runstate.progress import task_detail
    detail = task_detail(turn.task("T1"))
    got = [(e["ref"], e["passage_id"]) for e in
           detail["evidence_requirements"][0]["supporting_evidence"]]
    assert got == [("P1", "chunk-a"), ("P2", "https://example.com/g")]


async def test_record_evidence_backfills_legacy_refs(store):
    from src.runstate.models import EvidenceRecord

    await store.create_turn("c1", "t1", "Q?")
    await store.record_plan("c1", [{"id": "T1", "question": "a?",
         "evidence_requirements": [{"id": "E1", "description": "d1"}]}])
    await store.record_evidence("c1", [EvidenceRecord(
        passage_id="old", requirement_ids=["E1"], text="t")], task_id="T1")
    # simulate a pre-ref chat: strip the assigned ref, record again
    rec = store.get_chat("c1").turns["t1"].task("T1") \
        .evidence_requirements[0].supporting_evidence[0]
    rec.ref = ""
    await store.record_evidence("c1", [EvidenceRecord(
        passage_id="new", requirement_ids=["E1"], text="t")], task_id="T1")
    refs = [r.ref for r in store.get_chat("c1").turns["t1"].task("T1")
            .evidence_requirements[0].supporting_evidence]
    assert refs == ["P1", "P2"]


def test_task_sources_carries_refs_and_titles():
    from types import SimpleNamespace as NS

    from src.agents.stream_adapter import _task_sources

    task = NS(evidence_requirements=[
        NS(id="E1", description="d", supporting_evidence=[
            NS(passage_id="chunk-a", ref="P1", document_id="PMC42",
               chunk_id="c1", url="", title="", section="Results", text="t"),
            NS(passage_id="https://example.com/g", ref="P2", document_id="",
               chunk_id="", url="https://example.com/g", title="Guideline",
               section="", text="t"),
        ], gaps=[]),
    ])
    local, web = _task_sources(task)
    assert local["passage_ids"] == ["chunk-a", "P1"]
    assert web["passage_ids"] == ["https://example.com/g", "P2"]
    assert web["title"] == "Guideline"
