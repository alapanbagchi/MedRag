"""Agentic v3 - state model tests: evidence thresholds, dedup, coverage."""

from src.agentic_v3.state import (
    AnswersTask,
    CriticRelevance,
    CriticVerdict,
    EvidenceRequirement,
    EvidenceSource,
    RequirementStatus,
    ResearchTask,
    RunBudget,
    SupportDirection,
    TaskStatus,
    TermConcept,
    V3RunState,
    VerifiedEvidence,
    WorkerReport,
)


def _evidence(req_id, doc_id, support=SupportDirection.SUPPORTS, n=1):
    return VerifiedEvidence(
        id=f"E-{req_id}-{doc_id}-{n}",
        task_id="T1",
        requirement_id=req_id,
        document_id=doc_id,
        chunk_id=f"{doc_id}-c{n}",
        excerpt=f"finding from {doc_id} number {n}",
        claim="some claim",
        support=support,
        confidence=0.9,
        source=EvidenceSource.RETRIEVAL,
    )


def test_requirement_coverage_and_satisfaction():
    req = EvidenceRequirement(id="T1.R1", text="vitamin D and BP", target_n=3)
    assert req.status == RequirementStatus.UNSATISFIED
    assert not req.satisfied()
    req.add_evidence(_evidence("T1.R1", "A"))
    assert req.status == RequirementStatus.PARTIALLY_SUPPORTED
    req.add_evidence(_evidence("T1.R1", "B"))
    req.add_evidence(_evidence("T1.R1", "C"))
    assert req.satisfied()
    assert req.status == RequirementStatus.SATISFIED
    assert req.coverage() == 3
    assert req.supporting_papers() == ["A", "B", "C"]


def test_same_paper_never_counts_twice():
    """spec section 13: the same paper cannot count as multiple articles."""
    req = EvidenceRequirement(id="T1.R2", text="potassium and BP", target_n=3)
    # The SAME paper (document+chunk+excerpt) re-arriving must be deduped,
    # and distinct chunks of the same paper never inflate the N count.
    duplicate = _evidence("T1.R2", "A", n=1)
    added1 = req.add_evidence(duplicate)
    added2 = req.add_evidence(duplicate.model_copy(update={
        "id": "E-T1.R2-A-x", "claim": "same claim again"}))
    assert added1 and not added2
    assert req.coverage() == 1
    # a different chunk of paper A is kept but still counts ONE paper
    other_chunk = _evidence("T1.R2", "A", n=2).model_copy(update={
        "chunk_id": "A-c99", "excerpt": "a different passage of the same paper"})
    req.add_evidence(other_chunk)
    assert req.coverage() == 1
    assert len(req.accepted) == 2


def test_distinct_papers_required_not_items():
    req = EvidenceRequirement(id="T1.R3", text="diet strategies", target_n=2)
    req.add_evidence(_evidence("T1.R3", "A", n=1))
    req.add_evidence(_evidence("T1.R3", "A", n=2))   # same doc, new excerpt
    req.add_evidence(_evidence("T1.R3", "B", n=1))
    assert req.coverage() == 2
    assert req.satisfied()


def test_contradicting_evidence_is_kept_and_separated():
    req = EvidenceRequirement(id="T1.R4", text="supplementation effect", target_n=2)
    req.add_evidence(_evidence("T1.R4", "A", support=SupportDirection.SUPPORTS))
    req.add_evidence(_evidence("T1.R4", "B", support=SupportDirection.CONTRADICTS))
    # B counts as accepted evidence (reaches the contradiction agent),
    # but does NOT inflate the supporting-paper count.
    assert req.contradicting_papers() == ["B"]
    assert req.coverage() == 1
    assert not req.satisfied()


def test_mark_exhausted_sets_gap():
    req = EvidenceRequirement(id="T1.R5", text="N evidence", target_n=3)
    req.add_evidence(_evidence("T1.R5", "A"))
    req.mark_exhausted("only 1 supporting paper found")
    assert req.status == RequirementStatus.EXHAUSTED
    assert req.gap


def test_task_status_finalize():
    task = ResearchTask(
        id="T1",
        title="Dietary management of hypertension",
        objective="determine dietary strategies for BP",
        evidence_requirements=[
            EvidenceRequirement(id="T1.R1", text="sodium and BP", target_n=1),
            EvidenceRequirement(id="T1.R2", text="potassium and BP", target_n=1),
        ],
        stop_criteria=["evidence satisfied", "budget exhausted"],
        entities=["sodium", "potassium", "hypertension"],
    )
    assert not task.satisfied()
    task.requirement("T1.R1").add_evidence(_evidence("T1.R1", "A"))
    task.requirement("T1.R2").add_evidence(_evidence("T1.R2", "B"))
    assert task.satisfied()
    task.finalize()
    assert task.status == TaskStatus.SATISFIED


def test_run_budget_exhaustion():
    b = RunBudget(max_searches=3, max_deep_inspections=2)
    assert not b.exhausted()
    b.searches_used = 3
    assert b.exhausted()


def test_worker_report_build():
    task = ResearchTask(id="T1", title="task one", objective="obj",
                        evidence_requirements=[EvidenceRequirement(
                            id="T1.R1", text="req", target_n=1)])
    task.requirement("T1.R1").add_evidence(_evidence("T1.R1", "A"))
    task.finalize()
    report = WorkerReport(
        task_id=task.id, task_title=task.title, status=task.status.value,
        evidence=task.all_evidence(), searches_used=1,
    )
    assert report.to_dict()["task_id"] == "T1"
    assert len(report.evidence) == 1


def test_critic_verdict_acceptance():
    yes = CriticVerdict(answers_task=AnswersTask.YES,
                        relevance=CriticRelevance.RELEVANT,
                        support=SupportDirection.SUPPORTS)
    no = CriticVerdict(answers_task=AnswersTask.NO,
                       relevance=CriticRelevance.PARTIALLY_RELEVANT)
    contradicts = CriticVerdict(answers_task=AnswersTask.YES,
                                support=SupportDirection.CONTRADICTS)
    assert yes.accepted
    assert not no.accepted
    assert contradicts.accepted  # spec section 19: contradictory evidence passes the gate


def test_term_concept_all_terms():
    c = TermConcept(surface_form="vitamin D", preferred_name="Cholecalciferol",
                    synonyms=["25-hydroxyvitamin D", "25(OH)D", "vitamin D"])
    terms = c.all_terms()
    assert "vitamin D" in terms
    assert "Cholecalciferol" in terms
    assert terms.count("vitamin D") == 1  # deduped


def test_run_state_evidence_aggregation():
    st = V3RunState(question="q")
    task = ResearchTask(id="T1", title="t", objective="o",
                        evidence_requirements=[EvidenceRequirement(
                            id="T1.R1", text="r", target_n=1)])
    task.requirement("T1.R1").add_evidence(_evidence("T1.R1", "A"))
    st.add_task(task)
    assert len(st.all_evidence()) == 1
    assert st.evidence_by_id()["E-T1.R1-A-1"].document_id == "A"
