"""Unit tests for the agentic v2 runtime policy (pure, no LLM/corpus)."""
from src.agentic_v2.orchestrator import ActionDecision
from src.agentic_v2.policy import (
    ResearchPhase,
    determine_phase,
    evaluate_progress,
    fallback_action,
    legal_actions,
    progress_snapshot,
    strategy_key,
    validate_or_repair,
)
from src.agentic_v2.state import (
    ActionType,
    CandidatePassage,
    ObjectiveStatus,
    ResearchObjective,
    ResearchState,
    RetrievedDocument,
)


def _obj(id="H1", status=ObjectiveStatus.OPEN):
    return ResearchObjective(id=id, statement="statement", status=status)


# -- phase ------------------------------------------------------------

def test_phase_progression():
    assert determine_phase(ResearchState()) == ResearchPhase.EXPLORE

    s = ResearchState()
    s.upsert_objective(_obj())
    assert determine_phase(s) == ResearchPhase.EXPLORE       # objectives, no docs

    s.add_documents([RetrievedDocument(chunk_id="c1", document_id="P1", text="x")])
    assert determine_phase(s) == ResearchPhase.EXCAVATE       # docs, no candidates/evidence

    s.add_candidates([CandidatePassage(chunk_id="c1", document_id="P1", text="x")])
    assert determine_phase(s) == ResearchPhase.ASSESS         # candidates ready to verify

    s.objectives[0].status = ObjectiveStatus.SUPPORTED
    assert determine_phase(s) == ResearchPhase.ANSWER


# -- legal actions -----------------------------------------------------

def test_legal_actions_no_objectives():
    legal = legal_actions(ResearchState())
    assert ActionType.DECOMPOSE in legal
    assert ActionType.STOP in legal
    assert ActionType.VERIFY not in legal
    assert ActionType.SYNTHESIZE not in legal


def test_legal_actions_verify_requires_objectives_and_material():
    s = ResearchState()
    assert ActionType.VERIFY not in legal_actions(s)

    s.upsert_objective(_obj())
    assert ActionType.VERIFY not in legal_actions(s)          # no material yet

    s.add_documents([RetrievedDocument(chunk_id="c1", document_id="P1", text="x")])
    assert ActionType.VERIFY in legal_actions(s)


def test_legal_actions_synthesize_requires_evidence_or_investigated():
    s = ResearchState()
    s.upsert_objective(_obj(status=ObjectiveStatus.OPEN))
    assert ActionType.SYNTHESIZE not in legal_actions(s)

    s.objectives[0].status = ObjectiveStatus.SUPPORTED
    assert ActionType.SYNTHESIZE in legal_actions(s)


def test_legal_actions_global_retrieve_budget_enforced():
    s = ResearchState()
    s.upsert_objective(_obj())
    s.max_global_retrieves = 1
    assert ActionType.GLOBAL_RETRIEVE in legal_actions(s)

    s.record_action(ActionDecision(action=ActionType.GLOBAL_RETRIEVE))
    s.finish_last_action(status="done")
    assert ActionType.GLOBAL_RETRIEVE not in legal_actions(s)


def test_local_actions_require_material():
    s = ResearchState()
    s.upsert_objective(_obj())
    legal = legal_actions(s)
    assert ActionType.READ_DOCUMENT not in legal
    assert ActionType.FIND_SECTIONS not in legal

    s.add_documents([RetrievedDocument(chunk_id="c1", document_id="P1", text="x")])
    legal = legal_actions(s)
    assert ActionType.READ_DOCUMENT in legal
    assert ActionType.FIND_SECTIONS in legal


# -- strategy key ------------------------------------------------------

def test_strategy_key_normalizes_query_order():
    s = ResearchState()
    s.upsert_objective(_obj())
    d1 = ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1",
                        query="medial arterial calcification")
    d2 = ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1",
                        query="calcification arterial medial")
    assert strategy_key(d1, s) == strategy_key(d2, s)


def test_strategy_key_verify_signature_changes_with_candidates():
    s = ResearchState()
    s.upsert_objective(_obj())
    d = ActionDecision(action=ActionType.VERIFY, objective_id="H1")
    k1 = strategy_key(d, s)
    s.add_candidates([CandidatePassage(chunk_id="c1", document_id="P1", text="x")])
    k2 = strategy_key(d, s)
    assert k1 != k2


# -- progress evaluation ----------------------------------------------

def test_evaluate_progress_retrieve_new_doc():
    s = ResearchState()
    s.upsert_objective(_obj())
    before = progress_snapshot(s)
    s.add_documents([RetrievedDocument(chunk_id="c1", document_id="P1", text="x")])
    out = evaluate_progress(ActionDecision(action=ActionType.GLOBAL_RETRIEVE),
                            before, progress_snapshot(s), None)
    assert out.progress is True


def test_evaluate_progress_retrieve_no_new():
    s = ResearchState()
    s.upsert_objective(_obj())
    s.add_documents([RetrievedDocument(chunk_id="c1", document_id="P1", text="x")])
    before = progress_snapshot(s)
    out = evaluate_progress(ActionDecision(action=ActionType.GLOBAL_RETRIEVE),
                            before, progress_snapshot(s), None)
    assert out.progress is False
    assert "no new documents" in out.summary


def test_evaluate_progress_verify_status_change():
    s = ResearchState()
    s.upsert_objective(_obj(status=ObjectiveStatus.OPEN))
    s.add_documents([RetrievedDocument(chunk_id="c1", document_id="P1", text="x")])
    before = progress_snapshot(s)
    s.objectives[0].status = ObjectiveStatus.SUPPORTED
    out = evaluate_progress(ActionDecision(action=ActionType.VERIFY),
                            before, progress_snapshot(s), None)
    assert out.progress is True


def test_evaluate_progress_synthesize_is_terminal():
    s = ResearchState()
    before = progress_snapshot(s)
    s.terminal = True
    s.stop_reason = "synthesized"
    out = evaluate_progress(ActionDecision(action=ActionType.SYNTHESIZE),
                            before, progress_snapshot(s), None)
    assert out.progress is True


# -- validate / repair ------------------------------------------------

def test_validate_or_repair_illegal_action():
    s = ResearchState()  # no objectives -> VERIFY illegal
    d = ActionDecision(action=ActionType.VERIFY)
    d2, repaired, reason = validate_or_repair(d, s)
    assert repaired is True
    assert d2.action == ActionType.DECOMPOSE
    assert "illegal" in reason


def test_validate_or_repair_exhausted_strategy_pivots():
    s = ResearchState()
    s.upsert_objective(_obj())
    d = ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1", query="q1")
    key = strategy_key(d, s)
    s.exhaust_strategy(key)
    d2, repaired, reason = validate_or_repair(d, s)
    assert repaired is True
    assert d2.action != ActionType.GLOBAL_RETRIEVE


def test_validate_or_repair_stop_always_allowed():
    s = ResearchState()
    d = ActionDecision(action=ActionType.STOP)
    d2, repaired, _ = validate_or_repair(d, s)
    assert repaired is False
    assert d2.action == ActionType.STOP


def test_fallback_action_is_always_legal():
    s = ResearchState()
    assert fallback_action(s) in legal_actions(s)
    s.upsert_objective(_obj())
    assert fallback_action(s) in legal_actions(s)
