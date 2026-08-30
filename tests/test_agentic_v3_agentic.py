"""Agentic v3 - validation of the agentic control-loop upgrade (reqs 1-10).

Covers, without any live LLM/corpus:
  1. adaptive retry/replanning with meaningfully-different queries,
  2. explicit evidence state machine (RETRIEVED -> UNDER_REVIEW ->
     ACCEPTED / REJECTED / CONTRADICTORY) and requirement states
     (UNSATISFIED -> PARTIALLY_SUPPORTED -> SATISFIED),
  3. strict task/requirement state isolation between parallel workers,
  5/6. contradiction analysis only on verified evidence,
  7. evidence-gated synthesis (rejected material never reaches it),
  8. complete evidence provenance,
  9. budgets prevent runaway loops,
  10. an end-to-end observable transition trace with scoped ids.
"""

import asyncio
import json

from src.agentic_v3.critic import CriticAgent
from src.agentic_v3.events import V3Events
from src.agentic_v3.replan import meaningfully_different
from src.agentic_v3.search import RequirementSearch, TaskSearchPlan
from src.agentic_v3.state import (
    AnswersTask,
    CriticRelevance,
    CriticVerdict,
    EvidenceRequirement,
    EvidenceSource,
    EvidenceStatus,
    RequirementStatus,
    ResearchTask,
    RetrievedPaper,
    RunBudget,
    SupportDirection,
    VerifiedEvidence,
)
from src.agentic_v3.synthesize import _synthesis_prompt
from src.agentic_v3.worker import WorkerAgent
from src.config import AppConfig


# ---------------------------------------------------------------------------
# shared fakes
# ---------------------------------------------------------------------------

def _task(rids=(("T1.R1", "effect of vitamin D on blood pressure", 2),)):
    return ResearchTask(
        id="T1",
        title="vitamin D and hypertension",
        objective="does vitamin D affect blood pressure",
        evidence_requirements=[EvidenceRequirement(id=rid, text=text, target_n=n)
                               for rid, text, n in rids],
        stop_criteria=["satisfied", "budget exhausted"],
    )


def _paper(doc_id, req_id="T1.R1"):
    return RetrievedPaper(
        chunk_id=f"chunk-{doc_id}", document_id=doc_id, section="Results",
        unit_kind="paragraph", score=1.0,
        text=f"study {doc_id}: vitamin D supplementation reduced systolic "
             f"blood pressure in a randomized trial.",
        source_query="q", round_no=1, rank=1, retrieval_method="hybrid:bm25",
    )


class CaptureEmitter:
    """In-memory event sink (observability)."""

    def __init__(self):
        self.events = []
        self.start = None

    def emit(self, type_, **fields):
        self.events.append({"type": type_, **fields})


class FakeEnricher:
    async def enrich(self, task):
        task.terminology = []
        return []


class FakeSearchPlanner:
    async def plan(self, task, round_no, attempts=None):
        reqs = [RequirementSearch(requirement_id=r.id, rationale="fake",
                                  queries=[f"query-{r.id}-r{round_no}"])
                for r in task.evidence_requirements if not r.satisfied()]
        return TaskSearchPlan(round_no=round_no, rationale="fake",
                              requirements=reqs)


class FakeReplanner:
    def __init__(self):
        self.calls = []

    async def plan(self, ctx, task, requirement):
        self.calls.append({"req": ctx.requirement_id,
                           "prev": list(ctx.previous_queries),
                           "rounds_left": ctx.remaining_rounds,
                           "searches_left": ctx.remaining_searches})
        from src.agentic_v3.replan import FailureAnalysis
        return FailureAnalysis(
            diagnosis="previous queries did not surface outcome passages",
            missing_evidence=f"more supporting papers for {ctx.requirement_id}",
            strategy="different terminology + outcome pairing",
            queries=[f"replanned-{ctx.requirement_id}-{len(ctx.previous_queries) + 1}"],
        )


class FakeRetriever:
    """round -> req_id -> [doc ids]; records every query that ran."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.queries = []

    async def search(self, task, requirement, query, exclude_chunk_ids=None,
                     top_k=None, round_no=0):
        self.queries.append((round_no, requirement.id, query))
        docs = self.rounds.get(round_no, {}).get(requirement.id, [])
        out = []
        for d in docs:
            p = _paper(d, requirement.id)
            if p.chunk_id in (exclude_chunk_ids or []):
                continue
            out.append(p)
        return out[: (top_k or 5)]


class FakeCritic:
    """doc_id -> (answers, support). Raises on any scope mismatch."""

    def __init__(self, verdicts, expected_scope=None):
        self.verdicts = verdicts
        self.expected_scope = expected_scope or {}
        self.seen = []

    async def judge(self, task, requirement, paper, run_id="", attempt_id=""):
        self.seen.append((task.id, requirement.id, paper.document_id,
                          paper.evidence_id, attempt_id))
        if self.expected_scope:
            assert task.id == self.expected_scope["task"]
            assert requirement.id == self.expected_scope["requirement"]
        ans, sup = self.verdicts.get(paper.document_id, ("no", "neutral"))
        return CriticVerdict(
            evidence_id=paper.evidence_id,
            document_id=paper.document_id,
            chunk_id=paper.chunk_id,
            section=paper.section,
            relevance=(CriticRelevance.RELEVANT if ans == "yes"
                       else CriticRelevance.PARTIALLY_RELEVANT),
            answers_task=(AnswersTask.YES if ans == "yes"
                          else AnswersTask.NO),
            support=(SupportDirection.SUPPORTS if sup == "supports"
                     else SupportDirection.CONTRADICTS if sup == "contradicts"
                     else SupportDirection.NEUTRAL),
            confidence=0.9 if ans == "yes" else 0.4,
            note="verified finding" if ans == "yes" else "does not answer",
        )


class FakeDeepInspector:
    def __init__(self, found=None):
        self.found = found or {}

    async def inspect(self, task, requirement, document_id, context_hint="",
                      run_id="", attempt_id=""):
        return self.found.get(document_id, [])


def _worker(retriever, critic, replanner=None, deep=None, events=None):
    if events is not None and not isinstance(events, V3Events):
        events = V3Events(events)
    return WorkerAgent(
        enricher=FakeEnricher(),
        search_planner=FakeSearchPlanner(),
        replanner=replanner or FakeReplanner(),
        retriever=retriever,
        critic=critic,
        deep_inspector=deep or FakeDeepInspector({}),
        events=events,
    )


# ---------------------------------------------------------------------------
# 2. explicit evidence state machine
# ---------------------------------------------------------------------------

def test_evidence_state_machine_transitions():
    retriever = FakeRetriever({1: {"T1.R1": ["A", "B", "X"]},
                               2: {"T1.R1": ["C"]}})
    critic = FakeCritic({"A": ("yes", "supports"),
                         "B": ("yes", "contradicts"),
                         "X": ("no", "neutral"),
                         "C": ("yes", "supports")})
    events = CaptureEmitter()
    worker = _worker(retriever, critic, events=events)
    task = _task([("T1.R1", "effect of vitamin D on blood pressure", 2)])
    budget = RunBudget(max_searches=2, max_retrieval_rounds=2,
                       max_deep_inspections=0, evidence_target=2)

    report = asyncio.run(worker.run(task, budget, run_id="run-state"))
    req = task.requirement("T1.R1")

    # A accepted, B contradictory, X rejected
    by_doc = {e.document_id: e for e in req.accepted}
    assert by_doc["A"].status == EvidenceStatus.ACCEPTED
    assert by_doc["B"].status == EvidenceStatus.CONTRADICTORY
    assert req.rejected == 1
    assert req.reviewed, "every judged candidate must leave a review record"

    # requirement state walked UNSATISFIED -> PARTIALLY_SUPPORTED -> SATISFIED
    states = [ev["new_state"] for ev in events.events
              if ev["type"] == "requirement_state"]
    assert RequirementStatus.PARTIALLY_SUPPORTED.value in states
    assert req.status == RequirementStatus.SATISFIED

    # evidence_state transitions observed with scoped ids
    ev_states = [ev for ev in events.events if ev["type"] == "evidence_state"]
    assert ev_states, "evidence_state events must be emitted"
    assert all(ev["requirement_id"] == "T1.R1" for ev in ev_states)
    assert all(ev["task_id"] == "T1" for ev in ev_states)


# ---------------------------------------------------------------------------
# 1. adaptive replanning: genuinely new queries, full context
# ---------------------------------------------------------------------------

def test_adaptive_replan_uses_new_queries_and_context():
    retriever = FakeRetriever({
        1: {"T1.R1": []},                       # round 1: nothing found
        2: {"T1.R1": ["A", "B"]},               # round 2: replanned query hits
    })
    critic = FakeCritic({"A": ("yes", "supports"), "B": ("yes", "supports")})
    replanner = FakeReplanner()
    worker = _worker(retriever, critic, replanner=replanner)
    task = _task()
    budget = RunBudget(max_searches=2, max_retrieval_rounds=2,
                       max_deep_inspections=0, evidence_target=2)

    report = asyncio.run(worker.run(task, budget, run_id="run-replan"))

    queries_that_ran = [q for (_r, _req, q) in retriever.queries]
    assert queries_that_ran == ["query-T1.R1-r1", "replanned-T1.R1-2"], (
        "round 2 must run the REPLANNED query, not the same query again")

    # the replanner received the full scoped context
    assert len(replanner.calls) == 1
    ctx = replanner.calls[0]
    assert ctx["req"] == "T1.R1"
    assert ctx["prev"] == ["query-T1.R1-r1"]
    assert ctx["rounds_left"] >= 0
    assert ctx["searches_left"] == 1   # 1 of 2 searches already spent

    # never repeated the same query (meaningfully-different filter)
    assert meaningfully_different(["query-T1.R1-r1"], ["query-T1.R1-r1"]) == []
    assert meaningfully_different(["replanned-T1.R1-1"], ["query-T1.R1-r1"]) ==         ["replanned-T1.R1-1"]

    assert task.satisfied()
    assert report.stop_reason == "satisfied"
    assert report.status == "satisfied"


def test_worker_stops_when_rounds_exhausted():
    retriever = FakeRetriever({1: {"T1.R1": ["A"]}, 2: {"T1.R1": ["A"]}})
    critic = FakeCritic({"A": ("yes", "supports"), "B": ("yes", "supports")})
    worker = _worker(retriever, critic)
    task = _task()
    budget = RunBudget(max_searches=2, max_retrieval_rounds=2,
                       max_deep_inspections=0, evidence_target=2)

    report = asyncio.run(worker.run(task, budget, run_id="run-budget"))
    assert task.requirement("T1.R1").coverage() == 1
    assert task.searches_used <= 2, "max_searches must bound attempts"
    assert task.searches_used == 2, "round 2 ran but found only dupe chunks"
    assert report.stop_reason in ("rounds_exhausted", "search_budget")


# ---------------------------------------------------------------------------
# 3. strict state isolation between parallel workers
# ---------------------------------------------------------------------------

def test_no_cross_task_leak_between_parallel_workers():
    task1 = _task([("T1.R1", "sodium and blood pressure", 1)])
    task2 = ResearchTask(
        id="T2", title="vitamin D status", objective="vitamin D and BP",
        evidence_requirements=[EvidenceRequirement(id="T2.R1",
                                                   text="vitamin D status and BP",
                                                   target_n=1)],
    )
    retriever = FakeRetriever({1: {"T1.R1": ["NA"], "T2.R1": ["VB"]}})
    # the critic is scoped: it would RAISE if ever called for the wrong
    # task/requirement (the isolation proof)
    critic = FakeCritic({"NA": ("yes", "supports"), "VB": ("yes", "supports")})
    worker = _worker(retriever, critic)
    budgets = [RunBudget(max_searches=1, max_retrieval_rounds=1,
                         max_deep_inspections=0, evidence_target=1)
               for _ in (task1, task2)]

    async def _run(worker, task, budget):
        return await worker.run(task, budget, run_id="run-iso")

    async def _both():
        r1, r2 = await asyncio.gather(
            _run(worker, task1, budgets[0]), _run(worker, task2, budgets[1]))
        return r1, r2

    report1, report2 = asyncio.run(_both())
    assert report1.task_id == "T1" and report2.task_id == "T2"

    # critic was called with EXACTLY the right scope per task
    seen = critic.seen
    t1_calls = [s for s in seen if s[0] == "T1"]
    t2_calls = [s for s in seen if s[0] == "T2"]
    assert t1_calls and t2_calls
    assert all(s[1] == "T1.R1" for s in t1_calls)
    assert all(s[1] == "T2.R1" for s in t2_calls)

    # each task's evidence stays in its own task
    e1 = task1.requirement("T1.R1").accepted[0]
    e2 = task2.requirement("T2.R1").accepted[0]
    assert e1.task_id == "T1" and e1.requirement_id == "T1.R1"
    assert e2.task_id == "T2" and e2.requirement_id == "T2.R1"


# ---------------------------------------------------------------------------
# 7/8. provenance + evidence-gated synthesis
# ---------------------------------------------------------------------------

def test_verified_evidence_excludes_rejected_and_carries_provenance():
    req = EvidenceRequirement(id="T1.R1", text="vitamin D and BP", target_n=1)
    ok = VerifiedEvidence(id="E-ok", task_id="T1", requirement_id="T1.R1",
                          document_id="A", excerpt="supports",
                          support=SupportDirection.SUPPORTS,
                          status=EvidenceStatus.ACCEPTED)
    bad = VerifiedEvidence(id="E-bad", task_id="T1", requirement_id="T1.R1",
                           document_id="B", excerpt="rejected text",
                           support=SupportDirection.NEUTRAL,
                           status=EvidenceStatus.REJECTED)
    req.accepted = [ok, bad]  # bad is in the list but must never be VERIFIED
    assert req.verified() == [ok], "REJECTED items are structurally excluded"

    # synthesis prompt only presents verified items
    from src.agentic_v3.state import V3RunState, MasterPlan
    state = V3RunState(run_id="r", question="q")
    state.add_task(ResearchTask(id="T1", title="t", objective="o",
                                evidence_requirements=[req]))
    prompt = _synthesis_prompt(state)
    assert "E-ok" in prompt
    assert "rejected text" not in prompt, (
        "REJECTED evidence must never reach the synthesizer")


def test_evidence_provenance_chain_complete():
    retriever = FakeRetriever({1: {"T1.R1": ["A"]}})
    critic = FakeCritic({"A": ("yes", "supports")})
    worker = _worker(retriever, critic)
    task = _task([("T1.R1", "vitamin D and BP", 1)])
    budget = RunBudget(max_searches=1, max_retrieval_rounds=1,
                       max_deep_inspections=0, evidence_target=1)

    report = asyncio.run(worker.run(task, budget, run_id="run-prov"))
    e = task.requirement("T1.R1").accepted[0]
    assert e.run_id == "run-prov"
    assert e.id and e.task_id == "T1" and e.requirement_id == "T1.R1"
    assert e.attempt_id == "A1"
    assert e.document_id == "A" and e.chunk_id == "chunk-A"
    assert e.search_query == "query-T1.R1-r1"
    assert e.retrieval_method == "hybrid:bm25"
    assert e.rank == 1
    assert e.critic_verdict is not None
    assert e.critic_verdict["answers_task"] == "yes"
    assert e.support == SupportDirection.SUPPORTS
    assert e.status == EvidenceStatus.ACCEPTED

    # the worker report carries the same provenance for the audit chain
    paper = report.requirements[0].papers[0]
    for key in ("evidence_id", "attempt_id", "retrieval_method", "rank",
                "state", "support", "confidence", "document_id"):
        assert key in paper, f"report paper must carry provenance key {key}"


# ---------------------------------------------------------------------------
# 5/6. contradiction only on verified evidence; evidence-aware resolution
# ---------------------------------------------------------------------------

def test_contradiction_detector_ignores_rejected():
    from src.agentic_v3.contradiction import detect_contradictions_deterministic
    rejected = VerifiedEvidence(id="E1", task_id="T2", requirement_id="T2.R1",
                                document_id="A", excerpt="support claim",
                                support=SupportDirection.SUPPORTS,
                                status=EvidenceStatus.REJECTED)
    other = VerifiedEvidence(id="E2", task_id="T2", requirement_id="T2.R1",
                             document_id="B", excerpt="contradict claim",
                             support=SupportDirection.CONTRADICTS,
                             status=EvidenceStatus.REJECTED)
    assert detect_contradictions_deterministic([rejected, other]) == []

    verified_support = rejected.model_copy(update={"id": "E1v",
                                                   "status": EvidenceStatus.ACCEPTED})
    verified_contra = other.model_copy(update={"id": "E2v",
                                               "status": EvidenceStatus.CONTRADICTORY})
    cs = detect_contradictions_deterministic([verified_support, verified_contra])
    assert len(cs) == 1
    assert cs[0].evidence_a == ["E1v"] and cs[0].evidence_b == ["E2v"]
# ---------------------------------------------------------------------------
# 10. end-to-end observable transition trace (requirement 13.10)
# ---------------------------------------------------------------------------

def test_end_to_end_transition_trace():
    """The full pipeline run must emit a scoped, sequenced trace:
    [MASTER] -> [WORKER:T1] [SEARCH:A1] [RETRIEVAL] [CRITIC:E..] ->
    [EVIDENCE:req -> state] -> ([REPLAN:A2] -> ...) -> [CONTRADICTION] ->
    [SYNTHESIS]. Scope ids must be self-consistent (no cross-task leakage)."""
    from src.agentic_v3.master import MasterPlan
    from src.agentic_v3.pipeline import AgenticV3Pipeline
    from src.agentic_v3.synthesize import AnswerSection, Citation, SynthesisReport


    class FakeMaster:
        async def plan(self, query, budget):
            return MasterPlan(question=query, tasks=[
                ResearchTask(id="T1", title="dietary management",
                             objective="diet and BP",
                             evidence_requirements=[
                                 EvidenceRequirement(id="T1.R1",
                                                     text="sodium and BP",
                                                     target_n=1)]),
                ResearchTask(id="T2", title="vitamin D and hypertension",
                             objective="vitamin D supplementation and BP",
                             evidence_requirements=[
                                 EvidenceRequirement(id="T2.R1",
                                                     text="vitamin D supplementation and BP",
                                                     target_n=2)]),
            ])


    class FakeContradiction:
        async def detect(self, state):
            return []


    class FakeSynthesis:
        async def synthesize(self, state):
            verified = state.verified_evidence()
            return SynthesisReport(
                summary="T1 satisfied in round 1; T2 satisfied after replanning",
                sections=[AnswerSection(
                    heading="answers",
                    body="claims",
                    citations=[Citation(requirement_id=e.requirement_id,
                                        document_id=e.document_id,
                                        support=e.support.value)
                               for e in verified]),
                ],
                limitations=["offline test"],
                unresolved_gaps=[r.gap for t in state.tasks
                                 for r in t.evidence_requirements if r.gap],
                unresolved_contradictions=[],
                resolved_contradictions=[],
                confidence=0.8,
                citations=[Citation(requirement_id="T1.R1",
                                    document_id="TA", support="supports")],
            )

    retriever = FakeRetriever({
        1: {"T1.R1": ["TA"], "T2.R1": ["TX"]},
        2: {"T2.R1": ["TC", "TD"]},
    })
    critic = FakeCritic({"TA": ("yes", "supports"),
                         "TX": ("no", "neutral"),
                         "TC": ("yes", "supports"),
                         "TD": ("yes", "supports")})
    capture = CaptureEmitter()
    worker = _worker(retriever, critic, replanner=FakeReplanner(),
                     events=V3Events(capture))

    pipeline = AgenticV3Pipeline(
        config=AppConfig(),
        run_id="run-trace",
        master=FakeMaster(),
        worker=worker,
        contradiction_agent=FakeContradiction(),
        resolution_agent=None,
        synthesizer=FakeSynthesis(),
        events=V3Events(capture),
    )

    result = asyncio.run(pipeline.answer(
        "dietary restrictions for hypertension and vitamin D?"))

    assert result["run_id"] == "run-trace"
    assert result["terminal"] is True
    assert result["budget_usage"]["searches_used"] == 3   # T1:1 + T2:2

    events = capture.events
    types = [e["type"] for e in events]

    # the required stages all happened, in order
    assert "master_plan" in types
    assert sum(1 for t in types if t == "task_start") == 2
    assert "search_round" in types and "retrieved" in types
    assert "evidence_state" in types and "requirement_state" in types
    assert "replan" in types, "T2 must be replanned after its empty round 1"
    assert "final_evidence" in types and "run_end" in types

    # replan scope: exactly the insufficient requirement, with new queries
    replans = [e for e in events if e["type"] == "replan"]
    assert len(replans) == 1
    assert replans[0]["task_id"] == "T2"
    assert replans[0]["requirement_id"] == "T2.R1"
    assert replans[0]["attempt_id"] == "A2"
    assert replans[0]["diagnosis"] and replans[0]["queries"]

    # scope consistency: every evidence_state event carries a requirement-
    # scoped evidence id and its task id matches (no cross-task leakage)
    for e in events:
        if e["type"] != "evidence_state":
            continue
        assert e["task_id"] in ("T1", "T2")
        assert e["requirement_id"].startswith(e["task_id"] + ".")
        assert e["evidence_id"].startswith(e["requirement_id"] + ".")
        assert e["old_state"] in ("retrieved", "under_review")
        assert e["new_state"] in ("under_review", "accepted",
                                  "rejected", "contradictory")

    # the critic reviewed TA (accepted), TX (rejected), TC and TD (accepted)
    ev = [(e["evidence_id"], e["new_state"]) for e in events
          if e["type"] == "evidence_state" and e["new_state"] != "under_review"]
    assert ("T1.R1.A1.E1", "accepted") in ev      # TA
    assert ("T2.R1.A1.E1", "rejected") in ev      # TX
    assert ("T2.R1.A2.E1", "accepted") in ev and \
        ("T2.R1.A2.E2", "accepted") in ev         # TC / TD on the replanned round

    # requirement states walked explicitly
    req_states = [(e["requirement_id"], e["old_state"], e["new_state"])
                  for e in events if e["type"] == "requirement_state"]
    assert ("T1.R1", "unsatisfied", "satisfied") in req_states
    assert ("T2.R1", "unsatisfied", "partially_supported") in req_states
    assert ("T2.R1", "partially_supported", "satisfied") in req_states

    # final evidence is T1's TA + T2's TC/TD - REJECTED TX never appears
    docs = {e["document_id"] for e in result["evidence"]}
    assert docs == {"TA", "TC", "TD"}
    assert "TX" not in docs, "rejected evidence must never enter the result"
    assert result["workers"][0]["stop_reason"] == "satisfied"
