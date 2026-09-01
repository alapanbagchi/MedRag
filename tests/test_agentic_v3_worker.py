"""Agentic v3 - the Worker sub-orchestrator loop (scripted fakes, no LLM).

Covers the spec-critical behaviors:
  * Stage 10: N independent supporting papers per evidence requirement;
  * Stage 13: the same paper counted once even when several searches
    return it (dedup / provenance);
  * Stage 17: stop when satisfied (A) or the retrieval budget is spent (B);
  * Stages 11/14: deep paper inspection rescues "promising" papers whose
    search excerpts did not answer the requirement.
"""

from src.agents.replan import FailureAnalysis
from src.agents.search import RequirementSearch, TaskSearchPlan
from src.agentic.state import (
    AnswersTask,
    CriticRelevance,
    CriticVerdict,
    EvidenceRequirement,
    EvidenceSource,
    ResearchTask,
    RetrievedPaper,
    RunBudget,
    SupportDirection,
    VerifiedEvidence,
)
from src.agents.worker import WorkerAgent


def _task(requirements=(("T1.R1", "vitamin D supplementation and blood pressure", 3),)):
    return ResearchTask(
        id="T1",
        title="vitamin D and hypertension",
        objective="does vitamin D supplementation affect blood pressure",
        evidence_requirements=[EvidenceRequirement(
            id=rid, text=text, target_n=target)
            for rid, text, target in requirements],
        stop_criteria=["evidence satisfied", "budget exhausted"],
    )


def _paper(doc_id, text=("vitamin D supplementation reduced systolic blood pressure "
                         "in a randomized trial.")):
    return RetrievedPaper(
        chunk_id=f"chunk-{doc_id}", document_id=doc_id, section="Results",
        unit_kind="paragraph", score=1.0, text=text, source_query="q", round_no=1,
    )


class FakeEnricher:
    async def enrich(self, task):
        task.terminology = []
        return []


class FakeSearchPlanner:
    async def plan(self, task, round_no, attempts=None):
        reqs = []
        for req in task.evidence_requirements:
            if not req.satisfied():
                reqs.append(RequirementSearch(
                    requirement_id=req.id, rationale="fake",
                    queries=[f"query-{req.id}-r{round_no}"]))
        return TaskSearchPlan(round_no=round_no, rationale="fake",
                              requirements=reqs)


class FakeRetriever:
    """Returns per-round papers: round_no -> {req_id: [doc_ids]}."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    async def search(self, task, requirement, query, exclude_chunk_ids=None,
                     top_k=None, round_no=0):
        self.calls.append((round_no, requirement.id, query))
        doc_ids = self.rounds.get(round_no, {}).get(requirement.id, [])
        papers = []
        for doc_id in doc_ids:
            p = _paper(doc_id)
            if p.chunk_id in (exclude_chunk_ids or []):
                continue
            papers.append(p)
        return papers[: (top_k or 5)]


class FakeReplanner:
    """Emits a genuinely new query per replanning round."""

    def __init__(self):
        self.calls = []

    async def plan(self, ctx, task, requirement):
        self.calls.append((ctx.requirement_id, list(ctx.previous_queries)))
        return FailureAnalysis(
            diagnosis="previous attempts did not answer",
            missing_evidence="more supporting papers",
            strategy="new terms",
            queries=[f"new-query-{requirement.id}-{len(ctx.previous_queries) + 1}"],
        )


class FakeCritic:
    """Accepts docs in accept_set; flags docs in promising_set."""

    def __init__(self, accept, promising=()):
        self.accept = set(accept)
        self.promising = set(promising)

    async def judge(self, task, requirement, paper, run_id="", attempt_id=""):
        if paper.document_id in self.accept:
            return CriticVerdict(document_id=paper.document_id,
                                 chunk_id=paper.chunk_id,
                                 section=paper.section,
                                 relevance=CriticRelevance.RELEVANT,
                                 answers_task=AnswersTask.YES,
                                 support=SupportDirection.SUPPORTS,
                                 confidence=0.9,
                                 note="answers the requirement")
        if paper.document_id in self.promising:
            return CriticVerdict(document_id=paper.document_id,
                                 chunk_id=paper.chunk_id,
                                 section=paper.section,
                                 relevance=CriticRelevance.PARTIALLY_RELEVANT,
                                 answers_task=AnswersTask.NO,
                                 support=SupportDirection.NEUTRAL,
                                 confidence=0.5,
                                 note="excerpt lacks the outcome; inspect the paper")
        return CriticVerdict(document_id=paper.document_id,
                             relevance=CriticRelevance.NOT_RELEVANT,
                             answers_task=AnswersTask.NO,
                             support=SupportDirection.NEUTRAL, confidence=0.2)


class FakeDeepInspector:
    """Finds evidence inside specific documents."""

    def __init__(self, found):
        self.found = found  # doc_id -> [VerifiedEvidence]
        self.calls = []

    async def inspect(self, task, requirement, document_id, context_hint="",
                      run_id="", attempt_id=""):
        self.calls.append(document_id)
        items = self.found.get(document_id, [])
        out = []
        for i, item in enumerate(items):
            out.append(item.model_copy(update={
                "task_id": task.id, "requirement_id": requirement.id,
                "run_id": run_id, "attempt_id": attempt_id,
                "id": f"E-{task.id}-{requirement.id}-X{i}",
            }))
        return out


def _worker(retriever, critic, deep=None, enricher=None, planner=None,
             replanner=None, events=None):
    return WorkerAgent(
        enricher=enricher or FakeEnricher(),
        planner=planner or FakeSearchPlanner(),
        replanner=replanner or FakeReplanner(),
        retriever=retriever,
        critic=critic,
        deep_inspector=deep or FakeDeepInspector({}),
        events=events,
    )


async def _run_worker(task, worker, budget, run_id="run-test"):
    return await worker.run(task, budget, run_id=run_id)


def test_worker_reaches_n_with_cross_round_dedup():
    """Round 2 re-returns paper B (already accepted) - it must NOT count again."""
    retriever = FakeRetriever({
        1: {"T1.R1": ["A", "B"]},
        2: {"T1.R1": ["B", "C"]},
    })
    critic = FakeCritic(accept=["A", "B", "C"])
    worker = _worker(retriever, critic)
    task = _task()
    budget = RunBudget(max_searches=3, max_papers_per_round=5,
                       max_deep_inspections=1, evidence_target=3)

    import asyncio
    report = asyncio.run(_run_worker(task, worker, budget))
    assert task.satisfied()
    assert task.status.value == "satisfied"
    assert task.requirement("T1.R1").coverage() == 3
    assert task.requirement("T1.R1").supporting_papers() == ["A", "B", "C"]
    assert task.searches_used == 2   # stopped as soon as N was reached
    assert report.status == "satisfied"


def test_worker_exhausts_budget_when_literature_insufficient():
    retriever = FakeRetriever({1: {"T1.R1": ["A"]}, 2: {"T1.R1": ["A"]},
                              3: {"T1.R1": ["A"]}})
    critic = FakeCritic(accept=["A"])
    worker = _worker(retriever, critic)
    task = _task()
    budget = RunBudget(max_searches=3, max_deep_inspections=0, evidence_target=3)

    import asyncio
    report = asyncio.run(_run_worker(task, worker, budget))
    req = task.requirement("T1.R1")
    assert req.coverage() == 1
    assert req.status.value == "exhausted"
    assert "target 3" in req.gap
    assert report.gaps, "exhausted requirements must carry a gap message"


def test_worker_deep_inspection_recovers_promising_paper():
    # P1 is flagged promising (excerpt lacks outcome) -> deep inspection
    # finds the Results-table evidence the search missed.
    retriever = FakeRetriever({1: {"T1.R1": ["P1"]}, 2: {"T1.R1": ["D"]}})
    critic = FakeCritic(accept=[], promising=["P1"])
    deep_found = {
        "P1": [VerifiedEvidence(
            id="", task_id="T1", requirement_id="T1.R1", document_id="P1",
            chunk_id="", section="Results",
            excerpt="Table: supplementation reduced SBP by 4 mmHg",
            claim="reduction in SBP", support=SupportDirection.SUPPORTS,
            confidence=0.9, source=EvidenceSource.DEEP_INSPECTION,
        )],
    }
    deep = FakeDeepInspector(deep_found)
    worker = _worker(retriever, critic, deep=deep)
    task = _task()
    budget = RunBudget(max_searches=2, max_deep_inspections=2, evidence_target=3)

    import asyncio
    report = asyncio.run(_run_worker(task, worker, budget))
    req = task.requirement("T1.R1")
    assert "P1" in deep.calls
    assert req.coverage() == 1
    evidence = req.accepted[0]
    assert evidence.source == EvidenceSource.DEEP_INSPECTION
    assert task.deep_inspections_used == 1
    assert report.deep_inspections_used == 1


def test_worker_reports_per_requirement_packages():
    retriever = FakeRetriever({1: {"T1.R1": ["A"]}})
    critic = FakeCritic(accept=["A"])
    worker = _worker(retriever, critic)
    task = _task([("T1.R1", "vitamin D and BP", 1)])
    budget = RunBudget(max_searches=1, max_deep_inspections=0, evidence_target=1)

    import asyncio
    report = asyncio.run(_run_worker(task, worker, budget))
    assert len(report.requirements) == 1
    r = report.requirements[0]
    assert r.coverage == 1
    assert r.status == "satisfied"
    assert r.papers[0]["document_id"] == "A"
    assert r.papers[0]["support"] == "supports"
