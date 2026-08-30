"""Unit tests for agentic v2 ResearchState (no LLM)."""
from src.agentic_v2.state import (
    ActionType,
    EvidenceQuality,
    ObjectiveStatus,
    ResearchObjective,
    ResearchState,
    RetrievedDocument,
    VerifiedEvidence,
)
from src.agentic_v2.orchestrator import ActionDecision


def test_objective_upsert_replaces_by_id():
    state = ResearchState(question="q")
    state.upsert_objective(ResearchObjective(id="H1", statement="a", status=ObjectiveStatus.OPEN))
    state.upsert_objective(ResearchObjective(id="H1", statement="b", status=ObjectiveStatus.SUPPORTED))
    assert len(state.objectives) == 1
    assert state.objective("H1").statement == "b"
    assert state.objective("H1").status == ObjectiveStatus.SUPPORTED


def test_add_documents_dedupes_by_chunk_id():
    state = ResearchState()
    added = state.add_documents([
        RetrievedDocument(chunk_id="c1", document_id="PMC1", text="x"),
        RetrievedDocument(chunk_id="c1", document_id="PMC1", text="x"),
        RetrievedDocument(chunk_id="c2", document_id="PMC2", text="y"),
    ])
    assert added == 2
    assert len(state.documents) == 2


def test_seen_chunk_ids_spans_documents_candidates_evidence():
    state = ResearchState()
    state.add_documents([RetrievedDocument(chunk_id="c1")])
    assert "c1" in state.seen_chunk_ids()


def test_add_evidence_dedupes_by_objective_chunk_excerpt():
    state = ResearchState()
    e = VerifiedEvidence(objective_id="H1", chunk_id="c1", excerpt="same")
    assert state.add_evidence([e, e]) == 1
    assert len(state.evidence) == 1


def test_record_action_and_finish():
    state = ResearchState()
    state.iteration = 1
    state.record_action(ActionDecision(action=ActionType.GLOBAL_RETRIEVE, query="q"))
    assert state.prior_actions[-1].status == "running"
    state.finish_last_action(status="done", outcome="retrieved 3")
    assert state.prior_actions[-1].status == "done"
    assert state.prior_actions[-1].outcome == "retrieved 3"


def test_global_retrieves_used_counts_only_done():
    state = ResearchState()
    state.iteration = 1
    state.record_action(ActionDecision(action=ActionType.GLOBAL_RETRIEVE))
    state.finish_last_action(status="done")
    state.record_action(ActionDecision(action=ActionType.GLOBAL_RETRIEVE))
    state.finish_last_action(status="repeated")
    assert state.global_retrieves_used() == 1


def test_non_terminal_objectives():
    state = ResearchState()
    state.upsert_objective(ResearchObjective(id="H1", status=ObjectiveStatus.SUPPORTED))
    state.upsert_objective(ResearchObjective(id="H2", status=ObjectiveStatus.PARTIALLY_SUPPORTED))
    state.upsert_objective(ResearchObjective(id="H3", status=ObjectiveStatus.OPEN))
    assert [o.id for o in state.non_terminal_objectives()] == ["H2", "H3"]


def test_summarize_includes_question_objectives_and_budget():
    state = ResearchState(question="does aspirin reduce stroke?", max_rounds=10)
    state.upsert_objective(ResearchObjective(id="H1", statement="aspirin vs stroke", status=ObjectiveStatus.OPEN))
    text = state.summarize()
    assert "does aspirin reduce stroke?" in text
    assert "H1" in text
    assert "iteration=0/10" in text


def test_summarize_shows_anchored_evidence_not_head_slice():
    """The orchestrator must SEE the anchored evidence (a definition buried
    deep in a table), not a 320-char head slice that makes verified evidence
    look like false positives."""
    table = (
        "[Table ehae724-T2]\n"
        "Table 2: Acute coronary syndrome/percutaneous coronary intervention—clinical "
        "outcomes and their definitions\n"
        "Row: Acute coronary syndrome/PCI: Level 1 variables\n"
        "column_1: Acute coronary syndrome/PCI: Level 1 variables\n"
        "Row: Acute kidney injury requiring renal replacement therapy\n"
        "column_1: Renal replacement therapy includes ultrafiltration (haemofiltration), "
        "haemodialysis or peritoneal dialysis.50\n"
        "Row: Cardiac arrest\n"
        "column_1: Cardiac arrest is defined as a verified sudden cessation of cardiac "
        "activity causing unresponsiveness, absence of normal breathing and no signs "
        "of circulation (excluding syncope or profound vagally mediated bradycardia) "
        "with ventricular fibrillation, rapid ventricular tachycardia or bradycardia "
        "resulting in loss of consciousness, pulseless electrical activity, or "
        "asystole as the major causes.\n"
    )
    assert table.find("Cardiac arrest is defined as") > 400

    state = ResearchState(question="What is a cardiac arrest?")
    state.evidence = [VerifiedEvidence(
        objective_id="H1", document_id="PMC11704390", chunk_id="c-def",
        section="Results", excerpt=table,
        quality=EvidenceQuality.DIRECT, support="supports", confidence=1.0,
    )]
    text = state.summarize()
    assert "Cardiac arrest is defined as" in text
    assert "verified sudden cessation of cardiac activity" in text


def test_to_dict_is_json_safe():
    state = ResearchState(question="q")
    state.upsert_objective(ResearchObjective(id="H1", statement="a", status=ObjectiveStatus.SUPPORTED))
    d = state.to_dict()
    assert d["objectives"][0]["status"] == "supported"
    assert isinstance(d["objectives"][0]["status"], str)
