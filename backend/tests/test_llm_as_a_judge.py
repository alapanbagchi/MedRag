"""LLM-as-a-judge middleware tests (TestModel runs, fake judge, no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from src.lib.streaming import console_stream
from src.middleware.llm_as_a_judge import (
    EVIDENCE_TOOLS,
    LLMAsJudge,
    _requirements_from_args,
    install_judge_thinking_sink,
    install_retrieval_progress_sink,
)
from src.tools.verifier import INTENT_KEY, VerdictResponse


class _NoulAnswer:
    def __init__(self, noul):
        self.noul = noul


class _ChoiceAnswer:
    def __init__(self, choice):
        self.choice = choice


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _SystemOneResponse:
    def __init__(self, nouls=None, choices=None):
        self.nouls = nouls or {}
        self.choices = choices or {}
        self.usage = _Usage()


def _judge_prompt(state) -> str:
    """The old-style prompt string, rebuilt from the System One state.

    FakeTypeSafeClient records this so the existing parsers (_asked_pids,
    _passage_texts) keep inspecting what the judge saw.
    """
    parts = ["Question: " + state["question"], "",
             "Evidence Requirements:", ""]
    parts += [f"{r['id']}: {r['description']}"
              for r in state.get("evidence_requirements", [])]
    parts += ["", "Passages:", "",
              f"{state['passage']['id']}: {state['passage']['text']}"]
    return chr(10).join(parts)


def _asked_pids(prompt: str) -> list[str]:
    """Every passsage id the prompt asks about (batched prompts list several)."""
    import re as _re

    return _re.findall(r"^(c\d+): ", prompt, _re.M)


def _asked_pid(prompt: str) -> str | None:
    """Passage id the judge was asked to score (line after Passages:)."""
    seen_passages = False
    for line in prompt.splitlines():
        if line.startswith("Passages:"):
            seen_passages = True
            continue
        if not seen_passages or not line.strip():
            continue
        if line.startswith(("Still missing", "Score all", "Question:", "Evidence")):
            continue
        head = line.split(":", 1)[0].strip()
        if head and not head.startswith("E") and '-' not in head[:2]:
            return head
    return None


def _passage_texts(prompt: str) -> dict[str, str]:
    """pid -> rendered passage text from the judge prompt (single-line)."""
    out: dict[str, str] = {}
    seen_passages = False
    for line in prompt.splitlines():
        if line.startswith("Passages:"):
            seen_passages = True
            continue
        if not seen_passages or not line.strip():
            continue
        if line.startswith(("Still missing", "Score all", "Question:", "Evidence")):
            continue
        if ": " in line:
            pid, text = line.split(": ", 1)
            pid = pid.strip()
            if pid and pid not in out:
                out[pid] = text
    return out


class FakeTypeSafeClient:
    """Stand-in for AsyncTypeSafeClient.

    Answers each coverage Noul 1.0 when the report covers that requirement
    for the passage (else 0.0), each intent Noul with the report's intent
    score, and each verbatim Choice with the first span. Passages the report
    does not know are answered 0.0 (rejected), so one pass converges.
    """

    def __init__(self, report: VerdictResponse):
        self._report = report
        self.seen: list = []

    async def system_one(self, state, questions):
        self.seen.append(_judge_prompt(state))
        pid = state["passage"]["id"]
        desired = {str(v.passage_id): v
                   for v in self._report.evidence_results or []}
        verdict = desired.get(pid)
        nouls, choices = {}, {}
        for key in questions:
            if key == INTENT_KEY:
                nouls[key] = _NoulAnswer(
                    verdict.intent_score if verdict else 0.0)
            elif key.startswith("coverage::"):
                rid = key.split("::", 1)[1]
                covered = bool(verdict and rid in (verdict.coverage or []))
                nouls[key] = _NoulAnswer(1.0 if covered else 0.0)
            elif key.startswith("verbatim::"):
                choices[key] = _ChoiceAnswer("s0")
        return _SystemOneResponse(nouls=nouls, choices=choices)


def _report(rid="GENERAL", pids=("c1",)):
    return VerdictResponse.model_validate({"evidence_results": [
        {"passage_id": pid,
         "intent_score": 0.9,
         "coverage": [rid],
         "reason": "covers definition"}
        for pid in pids]})


async def evidence_tool(query: str) -> str:
    """Fake evidence tool."""
    return json.dumps([{"chunk_id": "c1", "text": "smoking causes harm to lungs"}])


async def plain_tool(term: str) -> str:
    """Fake non-evidence tool."""
    return f"RESULT for {term}"


async def _run(agent):
    result = await agent.run("hi")
    return result.output


async def test_judged_tool_output_carries_judgment():
    judge = FakeTypeSafeClient(_report())
    agent = Agent(
        TestModel(),
        tools=[evidence_tool],
        capabilities=[LLMAsJudge(tools=["evidence_tool"], judge_client=judge)],
    )
    out = await _run(agent)
    assert "smoking causes harm" in out
    assert "[EVIDENCE JUDGMENT]" in out
    # NOTE: TestModel JSON-escapes the tool result, so exact parsing is
    # covered in the _wrap_direct tests below; here, substrings suffice.
    assert "evidence_results" in out
    assert "GENERAL" in out
    assert "GENERAL 1.00" in out  # deterministic probability reason
    assert judge.seen != []


async def test_judge_scores_each_citation_in_parallel_calls():
    async def ten_tool(query: str) -> str:
        """Ten-item tool."""
        return json.dumps([{"chunk_id": f"c{i}", "text": f"claim {i} about lungs"} for i in range(10)])

    judge = FakeTypeSafeClient(_report(pids=[f"c{i}" for i in range(10)]))
    agent = Agent(
        TestModel(),
        tools=[ten_tool],
        capabilities=[LLMAsJudge(tools=["ten_tool"], judge_client=judge)],
    )
    await _run(agent)
    # One System One call per passage, fanned out in parallel; every call
    # carries the identical cacheable prefix and differs only in the passage.
    # Every citation is asked about exactly once.
    assert len(judge.seen) == 10
    asked = sorted(
        pid for message in judge.seen for pid in _asked_pids(message))
    assert asked == [f"c{i}" for i in range(10)]
    prefixes = {m.split("Passages:\n\n", 1)[0] for m in judge.seen}
    assert len(prefixes) == 1


async def _wrap_direct(cap, tool_name, args, output):
    call = SimpleNamespace(tool_name=tool_name)

    async def _handler(a):
        return output

    return await cap.wrap_tool_execute(
        None, call=call, tool_def=None, args=args, handler=_handler,
    )


async def test_requirements_reach_the_judge_prompt():
    judge = FakeTypeSafeClient(_report("E1"))
    cap = LLMAsJudge(tools=["evidence_tool"], judge_client=judge)
    out = await _wrap_direct(
        cap, "evidence_tool",
        {"query": "What is hypertension?",
         "evidence_requirements": [{"id": "E1", "description": "definition of hypertension"}]},
        json.dumps([{"chunk_id": "c1", "text": "hypertension is high blood pressure"}]),
    )
    assert "[EVIDENCE JUDGMENT]" in out
    assert "definition of hypertension" in judge.seen[0]
    # chunk_id-keyed hits must align with the judge's passage ids, not
    # fall back to P1 and silently zero the scores.
    block = out.split("[EVIDENCE JUDGMENT]\n", 1)[1]
    scored = json.loads(block)["evidence_results"][0]
    assert scored["passage_id"] == "c1"
    assert scored["coverage"] == ["E1"]


async def test_missing_requirements_judged_against_question():
    judge = FakeTypeSafeClient(_report())
    cap = LLMAsJudge(tools=["evidence_tool"], judge_client=judge)
    out = await _wrap_direct(
        cap, "evidence_tool",
        {"query": "What is hypertension?"},
        json.dumps([{"chunk_id": "c1", "text": "hypertension is high blood pressure"}]),
    )
    assert "[EVIDENCE JUDGMENT]" in out
    block = out.split("[EVIDENCE JUDGMENT]\n", 1)[1]
    assert json.loads(block)["evidence_results"][0]["coverage"] == ["GENERAL"]
    assert "What is hypertension?" in judge.seen[0]


async def test_judge_failure_never_breaks_the_tool():
    class BoomJudge:
        async def system_one(self, state, questions):
            raise RuntimeError("judge is down")

    cap = LLMAsJudge(tools=["evidence_tool"], judge_client=BoomJudge())
    out = await _wrap_direct(
        cap, "evidence_tool",
        {"query": "q",
         "evidence_requirements": [{"id": "E1", "description": "d"}]},
        json.dumps([{"chunk_id": "c1", "text": "some text"}]),
    )
    assert "some text" in out
    # Per-item judging fails open: the judge crashing yields an explicit
    # FAILED note with the (bounded) output, never a raised tool call.
    assert "[EVIDENCE JUDGMENT FAILED: judge returned no verdicts]" in out


async def test_tool_failure_degrades_to_message_for_evidence_tools():
    cap = LLMAsJudge(tools=["evidence_tool"], judge_client=FakeTypeSafeClient(_report()))
    call = SimpleNamespace(tool_name="evidence_tool")
    out = await cap.on_tool_execute_error(
        None, call=call, tool_def=None, args={}, error=RuntimeError("CUDA OOM"),
    )
    assert "[TOOL FAILED: evidence_tool: CUDA OOM]" in out


async def test_tool_failure_reraises_for_other_tools():
    import pytest as _pytest

    cap = LLMAsJudge(tools=["evidence_tool"], judge_client=FakeTypeSafeClient(_report()))
    call = SimpleNamespace(tool_name="plain_tool")
    with _pytest.raises(RuntimeError, match="boom"):
        await cap.on_tool_execute_error(
            None, call=call, tool_def=None, args={}, error=RuntimeError("boom"),
        )


def test_requirements_from_args_tolerates_ragged_shapes():
    assert _requirements_from_args({}) == []
    assert _requirements_from_args({"evidence_requirements": None}) == []
    assert _requirements_from_args({"evidence_requirements": "E1: def"}) == ["E1: def"]
    assert _requirements_from_args({"evidence_requirements": {"id": "E1"}}) == [{"id": "E1"}]
    reqs = [{"id": "E1"}, "junk", 42, None]
    assert _requirements_from_args({"evidence_requirements": reqs}) == [{"id": "E1"}, "junk"]
    assert _requirements_from_args(None) == []


async def test_non_evidence_tool_passes_through():
    judge = FakeTypeSafeClient(_report())
    agent = Agent(
        TestModel(),
        tools=[plain_tool],
        capabilities=[LLMAsJudge(tools=["evidence_tool"], judge_client=judge)],
    )
    out = await _run(agent)
    assert "[EVIDENCE JUDGMENT]" not in out
    assert judge.seen == []


async def test_unparseable_output_passes_through():
    async def messy_tool(query: str) -> str:
        """Messy tool."""
        return "just prose, no json"

    judge = FakeTypeSafeClient(_report())
    agent = Agent(
        TestModel(),
        tools=[messy_tool],
        capabilities=[LLMAsJudge(tools=["messy_tool"], judge_client=judge)],
    )
    out = await _run(agent)
    assert "[EVIDENCE JUDGMENT]" not in out
    assert judge.seen == []


async def test_console_stream_emits_verdict_json_deltas(capsys):
    from pydantic_ai.messages import (
        PartDeltaEvent,
        PartStartEvent,
        ToolCallPart,
        ToolCallPartDelta,
    )

    handler = console_stream("judge:test")
    events = [
        PartStartEvent(index=0, part=ToolCallPart(tool_name="final_result", args="")),
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"evidence_results": [')),
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"requirement_id": "E1"}]}')),
    ]

    async def _feed():
        for e in events:
            yield e

    await handler(None, _feed())
    out = capsys.readouterr().out
    assert '{"evidence_results": [' in out
    assert '"requirement_id": "E1"' in out


def test_default_evidence_tools():
    assert set(EVIDENCE_TOOLS) == {"local_search", "web_search"}


def test_requirements_from_args_expands_json_strings():
    blob = '[{"id": "E1", "description": "sodium"}, {"id": "E2"}]'
    assert _requirements_from_args({"evidence_requirements": blob}) == [
        {"id": "E1", "description": "sodium"}, {"id": "E2"}]
    assert _requirements_from_args({"evidence_requirements": '{"id": "E9"}'}) == [
        {"id": "E9"}]


async def test_progress_sink_fires_with_passages_before_judgment():
    from src.middleware import llm_as_a_judge as mod

    fired = []
    judge = FakeTypeSafeClient(_report())

    def sink(tool_call_id, tool_name, output):
        # Must fire before the judge runs (judge.seen still empty).
        fired.append((tool_call_id, tool_name, output, list(judge.seen)))

    token = install_retrieval_progress_sink(sink)
    try:
        cap = LLMAsJudge(tools=["evidence_tool"], judge_client=judge)
        call = SimpleNamespace(tool_name="evidence_tool", tool_call_id="tc_9")

        async def _handler(a):
            return json.dumps([{"chunk_id": "c1", "text": "smoking causes harm"}])

        out = await cap.wrap_tool_execute(
            None, call=call, tool_def=None,
            args={"query": "q", "evidence_requirements": []}, handler=_handler,
        )
    finally:
        mod._progress_sink.reset(token)
    assert fired != []
    call_id, name, output, seen_before = fired[0]
    assert (call_id, name) == ("tc_9", "evidence_tool")
    assert "smoking causes harm" in output
    assert seen_before == []  # passages reported before judging started
    assert "[EVIDENCE JUDGMENT]" in out  # judging still completed


async def test_progress_sink_tolerates_missing_call_id():
    from src.middleware import llm_as_a_judge as mod

    fired = []
    token = install_retrieval_progress_sink(
        lambda *a: fired.append(a))
    try:
        cap = LLMAsJudge(tools=["evidence_tool"], judge_client=FakeTypeSafeClient(_report()))
        out = await _wrap_direct(
            cap, "evidence_tool", {"query": "q"},
            json.dumps([{"chunk_id": "c1", "text": "t"}]),
        )
    finally:
        mod._progress_sink.reset(token)
    assert fired != [] and fired[0][0] == "" and fired[0][1] == "evidence_tool"
    assert "[EVIDENCE JUDGMENT]" in out


async def test_progress_sink_failure_never_breaks_the_tool():
    from src.middleware import llm_as_a_judge as mod

    def boom(*a):
        raise RuntimeError("sink down")

    token = install_retrieval_progress_sink(boom)
    try:
        cap = LLMAsJudge(tools=["evidence_tool"], judge_client=FakeTypeSafeClient(_report()))
        out = await _wrap_direct(
            cap, "evidence_tool", {"query": "q"},
            json.dumps([{"chunk_id": "c1", "text": "t"}]),
        )
    finally:
        mod._progress_sink.reset(token)
    assert "[EVIDENCE JUDGMENT]" in out


async def test_judged_passages_accumulate_on_deps():
    cap = LLMAsJudge(tools=["evidence_tool"], judge_client=FakeTypeSafeClient(_report()))
    deps = SimpleNamespace(kept_passages=[])
    ctx = SimpleNamespace(deps=deps)
    call = SimpleNamespace(tool_name="evidence_tool", tool_call_id="tc_1")

    async def _handler(a):
        return json.dumps([
            {"chunk_id": "c1", "text": "smoking causes harm"},
            {"chunk_id": "c2", "text": "unrelated weather"},
        ])

    out = await cap.wrap_tool_execute(
        ctx, call=call, tool_def=None, args={"query": "q"}, handler=_handler,
    )
    assert "[EVIDENCE JUDGMENT]" in out
    # only the kept passage accumulates, with its full dict
    assert [p.get("chunk_id") for p in deps.kept_passages] == ["c1"]
    assert deps.kept_passages[0]["text"] == "smoking causes harm"

    # a later call adds only what's new — no duplicates
    call2 = SimpleNamespace(tool_name="evidence_tool", tool_call_id="tc_2")

    async def _handler2(a):
        return json.dumps([
            {"chunk_id": "c1", "text": "smoking causes harm"},
            {"chunk_id": "c3", "text": "smoking harms vessels"},
        ])

    cap2 = LLMAsJudge(tools=["evidence_tool"], judge_client=FakeTypeSafeClient(_report(pids=["c3"])))
    await cap2.wrap_tool_execute(
        ctx, call=call2, tool_def=None, args={"query": "q"}, handler=_handler2,
    )
    assert [p.get("chunk_id") for p in deps.kept_passages] == ["c1", "c3"]


async def test_accumulate_skips_without_deps():
    # ctx=None (direct-wrap tests) must not blow up
    cap = LLMAsJudge(tools=["evidence_tool"], judge_client=FakeTypeSafeClient(_report()))
    out = await _wrap_direct(
        cap, "evidence_tool", {"query": "q"},
        json.dumps([{"chunk_id": "c1", "text": "t"}]),
    )
    assert "[EVIDENCE JUDGMENT]" in out


async def test_fetch_without_query_falls_back_to_ledger_question(tmp_path, monkeypatch):
    """firecrawl_fetch_urls carries urls, not the question — the judge must
    still score fetched pages (against the ledger task question) and the
    kept web passages must land in the ledger with their URL."""
    from src.runstate.store import RunStore

    store = RunStore(tmp_path)
    await store.create_turn("chat1", "turn1",
                            question="Does diet affect hypertension?")
    await store.record_task_started("chat1", "T1", "Does diet affect hypertension?")
    await store.record_requirements(
        "chat1", [{"id": "E1", "description": "diet evidence"}], task_id="T1")
    monkeypatch.setattr("src.runstate.store.get_store", lambda: store)

    url = "https://example.com/hypertension-diet-guide"
    judge = FakeTypeSafeClient(_report())
    # FakeTypeSafeClient._report defaults to GENERAL/c1 — rebuild for the url pid.
    judge._report = VerdictResponse.model_validate({"evidence_results": [
        {"passage_id": url, "intent_score": 0.9,
         "coverage": ["E1"], "reason": "covers diet guidance"}]})
    cap = LLMAsJudge(tools=["firecrawl_fetch_urls"], judge_client=judge)
    ctx = SimpleNamespace(deps=SimpleNamespace(
        chat_id="chat1", task_id="T1", kept_passages=[]))
    call = SimpleNamespace(tool_name="firecrawl_fetch_urls", tool_call_id="tc_web")

    async def _handler(a):
        return json.dumps({"urls": [url], "results": [
            {"url": url, "title": "Diet guide",
             "text": "sodium reduction lowers blood pressure",
             "fetched_ok": True}]})

    out = await cap.wrap_tool_execute(
        ctx, call=call, tool_def=None, args={"urls": [url]}, handler=_handler,
    )
    assert "[EVIDENCE JUDGMENT]" in out
    assert "Does diet affect hypertension?" in judge.seen[0]
    # the task's planned requirement is used when the tool passes none
    assert "diet evidence" in judge.seen[0]
    kept = store.get_chat("chat1").turns["turn1"].task("T1") \
        .evidence_requirements[0].supporting_evidence
    assert [e.url for e in kept] == [url]


async def test_search_pages_are_judged_like_retrieval(tmp_path, monkeypatch):
    """firecrawl_web_search output carries listings + scraped pages: the
    judge must score the page markdown (not the snippets) and the kept
    pages must land in the ledger with their URL and text."""
    from src.runstate.store import RunStore

    store = RunStore(tmp_path)
    await store.create_turn("chat1", "turn1",
                            question="Does diet affect hypertension?")
    await store.record_task_started("chat1", "T1", "Does diet affect hypertension?")
    await store.record_requirements(
        "chat1", [{"id": "E1", "description": "diet evidence"}], task_id="T1")
    monkeypatch.setattr("src.runstate.store.get_store", lambda: store)

    url = "https://www.nhlbi.nih.gov/dash"
    judge = FakeTypeSafeClient(_report())
    judge._report = VerdictResponse.model_validate({"evidence_results": [
        {"passage_id": url, "intent_score": 0.85,
         "coverage": ["E1"], "reason": "guideline markdown"}]})
    cap = LLMAsJudge(tools=["firecrawl_web_search"], judge_client=judge)
    ctx = SimpleNamespace(deps=SimpleNamespace(
        chat_id="chat1", task_id="T1", kept_passages=[]))
    call = SimpleNamespace(tool_name="firecrawl_web_search", tool_call_id="tc_search")

    async def _handler(a):
        return json.dumps({
            "query": "DASH diet", "available": True,
            "results": [{"title": "DASH", "url": url, "snippet": "short snippet"}],
            "trust": [{"url": url, "trustworthy": True, "reason": "allowlisted"}],
            "dropped": 0,
            "pages": [{"url": url, "title": "DASH",
                       "text": "sodium reduction lowers blood pressure markedly",
                       "fetched_ok": True}],
        })

    out = await cap.wrap_tool_execute(
        ctx, call=call, tool_def=None,
        args={"query": "Does diet affect hypertension?",
              "evidence_requirements": [{"id": "E1", "description": "diet evidence"}]},
        handler=_handler,
    )
    assert "[EVIDENCE JUDGMENT]" in out
    # the judge saw the scraped markdown, not the snippet
    assert "sodium reduction lowers blood pressure markedly" in judge.seen[0]
    kept = store.get_chat("chat1").turns["turn1"].task("T1") \
        .evidence_requirements[0].supporting_evidence
    assert [(e.url, e.text) for e in kept] == [
        (url, "sodium reduction lowers blood pressure markedly")]


async def test_fetch_without_any_question_passes_through_unjudged():
    """No query in args and no ledger to fall back on: fail open, raw out."""
    judge = FakeTypeSafeClient(_report())
    cap = LLMAsJudge(tools=["firecrawl_fetch_urls"], judge_client=judge)
    raw = json.dumps({"urls": ["https://example.com/g"], "results": [
        {"url": "https://example.com/g", "title": "G",
         "text": "some fetched text", "fetched_ok": True}]})

    async def _handler(a):
        return raw

    out = await _wrap_direct(
        cap, "firecrawl_fetch_urls", {"urls": ["https://example.com/g"]}, raw,
    )
    assert out == raw
    assert judge.seen == []


def test_sequential_flag_reads_env(monkeypatch):
    from src.middleware.llm_as_a_judge import _sequential
    monkeypatch.delenv("SEQUENTIAL", raising=False)
    assert _sequential() is False
    monkeypatch.setenv("SEQUENTIAL", "1")
    assert _sequential() is True


def test_evidence_from_output_prefers_scraped_pages():
    from src.middleware.llm_as_a_judge import _evidence_from_output
    out = json.dumps({
        "query": "q", "available": True,
        "results": [{"url": "https://example.com/a", "snippet": "short"}],
        "pages": [{"url": "https://example.com/a", "success": True,
                   "markdown": "# A\n\nFull scraped body.",
                   "text": "# A\n\nFull scraped body."}],
    })
    items = _evidence_from_output("firecrawl_web_search", out)
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/a"


def test_evidence_from_output_skips_failed_scrapes():
    """Explicit scrape failures are not evidence and must not shadow the
    search-snippet fallback (which is still judgeable)."""
    from src.middleware.llm_as_a_judge import _evidence_from_output
    out = json.dumps({
        "query": "q", "available": True,
        "results": [{"url": "https://example.com/a",
                     "snippet": "snippet text here"}],
        "pages": [{"url": "https://example.com/a", "success": False,
                   "error": "missing from batch response"}],
    })
    items = _evidence_from_output("firecrawl_web_search", out)
    assert items == [{"url": "https://example.com/a",
                      "snippet": "snippet text here"}]


def test_evidence_from_output_keeps_pages_without_success_flag():
    """Flat batch items carry no success key — they are usable pages."""
    from src.middleware.llm_as_a_judge import _evidence_from_output
    out = json.dumps({
        "query": "q", "available": True,
        "results": [],
        "pages": [{"url": "https://example.com/a",
                   "markdown": "# A\n\nBody.", "text": "# A\n\nBody."}],
    })
    items = _evidence_from_output("firecrawl_web_search", out)
    assert len(items) == 1

def _aview(raw_output, verdicts):
    from src.middleware.llm_as_a_judge import _agent_evidence_view
    report = VerdictResponse.model_validate(
        {"evidence_results": verdicts})
    return json.loads(_agent_evidence_view(raw_output, report))


def test_agent_evidence_view_replaces_bodies_with_verbatim():
    """The agent lane must see the judge-verified VERBATIM excerpt, not the
    megabyte passage body — no arbitrary truncation."""
    raw = json.dumps({
        "query": "q", "local_count": 1, "local": [
            {"id": "L1", "text": "long passage body " * 1000}],
        "web": {"pages": [
            {"url": "https://example.com/a", "text": "web body " * 1000}]},
    })
    view = _aview(raw, [
        {"passage_id": "L1", "intent_score": 0.9, "coverage": ["E1"],
         "reason": "r", "verbatim": "long passage body"},
        {"passage_id": "https://example.com/a", "intent_score": 0.8,
         "coverage": ["E1"], "reason": "r", "verbatim": "web body"},
    ])
    # Bodies dropped; the verbatim excerpt IS the evidence.
    assert view["local"][0] == {"id": "L1", "verbatim": "long passage body"}
    page = view["web"]["pages"][0]
    assert page == {"url": "https://example.com/a", "verbatim": "web body"}
    assert len(view["local"][0].get("text", "")) == 0


def test_agent_evidence_view_keeps_other_fields():
    """Titles/urls/scores-like identity fields survive the rebuild."""
    raw = json.dumps([{"chunk_id": "c1", "title": "Keep me", "similarity": 0.91,
                       "text": "body that should vanish"}])
    view = _aview(raw, [
        {"passage_id": "c1", "intent_score": 0.9, "coverage": ["E1"],
         "reason": "r", "verbatim": "body that should vanish"}])
    assert view[0] == {"chunk_id": "c1", "title": "Keep me",
                       "similarity": 0.91, "verbatim": "body that should vanish"}
    assert "text" not in view[0]


def test_agent_evidence_view_marks_failed_excerpts():
    """Covered but excerpt failed the check: visible placeholder, never a
    silent megabyte body."""
    raw = json.dumps([{"chunk_id": "c1", "text": "large body"}])
    view = _aview(raw, [
        {"passage_id": "c1", "intent_score": 0.9, "coverage": ["E1"],
         "reason": "r", "verbatim": ""}])
    assert view[0]["verbatim"] == " (excerpt failed verbatim check)"
    assert "text" not in view[0]


def test_agent_evidence_view_leaves_unjudged_items_untouched():
    raw = json.dumps([{"chunk_id": "c1", "text": "judged body"},
                      {"chunk_id": "c2", "text": "unjudged body"}])
    view = _aview(raw, [
        {"passage_id": "c1", "intent_score": 0.9, "coverage": ["E1"],
         "reason": "r", "verbatim": "judged body"}])
    assert view[0] == {"chunk_id": "c1", "verbatim": "judged body"}
    # No verdict at all: item keeps its identity AND its text (bounded by
    # the outer compaction only) — never fabricated evidence.
    assert view[1] == {"chunk_id": "c2", "text": "unjudged body"}


def test_agent_evidence_view_falls_back_on_unparseable_output():
    from src.middleware.llm_as_a_judge import _agent_evidence_view
    report = VerdictResponse(evidence_results=[])
    out = _agent_evidence_view("not json at all", report)
    parsed = json.loads(out)
    assert parsed.get("clipped") is True
    assert "not json at all" in parsed.get("head", "")

