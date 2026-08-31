"""Unit tests for the agentic v2 intent verifier (prompt + aggregation, no LLM)."""
import pytest

from src.agentic_v2.pipeline import AgenticV2Pipeline
from src.agentic_v2.state import (
    ActionType,
    CandidatePassage,
    ObjectiveStatus,
    ResearchObjective,
    ResearchState,
)
from src.agentic_v2.verify import (
    _MAX_PASSAGES,
    _aggregate,
    _passage_prompt,
    _passage_text,
    PassageAssessment,
)
from src.config import AppConfig


def _objective() -> ResearchObjective:
    return ResearchObjective(
        id="H1", statement="does X relate to Y?",
        intent="mechanism", evidence_required=["survival"],
    )


# -- full-text prompt ---------------------------------------------------------

def test_passage_text_is_full_and_untruncated():
    long = ("row one of the survival table\n" * 500)  # far beyond any old cap
    p = CandidatePassage(document_id="P1", text=long)
    out = _passage_text(p)
    assert out == long.strip()
    assert "truncated" not in out
    # layout preserved (newlines kept so tables keep their rows)
    assert "\n" in out


def test_passage_text_strips_only():
    p = CandidatePassage(document_id="P1", text="\n\n  short passage  \n")
    assert _passage_text(p) == "short passage"


def test_passage_prompt_single_document_full_text():
    obj = _objective()
    body = "para-A sentence.\npara-B sentence with EF < 35% data."
    p = CandidatePassage(document_id="P1", chunk_id="c1", section="Intro", text=body)
    prompt = _passage_prompt(obj, p)
    # exactly ONE passage block, with its identifiers
    assert "CANDIDATE PASSAGE (doc=P1, chunk=c1, section=Intro):" in prompt
    assert "CANDIDATE PASSAGE 2" not in prompt
    # objective context and FULL passage body present
    assert "OBJECTIVE H1" in prompt
    assert "mechanism" in prompt
    assert "survival" in prompt
    assert "para-B sentence with EF < 35% data." in prompt
    assert "truncated" not in prompt


# -- aggregation --------------------------------------------------------------

def test_aggregate_supported_takes_max_confidence():
    verdict = _aggregate(_objective(), [
        PassageAssessment(chunk_id="c1", relevance="partially_relevant", confidence=0.4),
        PassageAssessment(chunk_id="c2", relevance="relevant", confidence=0.9),
        PassageAssessment(chunk_id="c3", relevance="not_relevant", confidence=0.7),
    ])
    assert verdict.status is ObjectiveStatus.SUPPORTED
    assert verdict.confidence == 0.9
    assert len(verdict.assessments) == 3


def test_aggregate_partially_supported():
    verdict = _aggregate(_objective(), [
        PassageAssessment(chunk_id="c1", relevance="partially_relevant", confidence=0.5),
        PassageAssessment(chunk_id="c2", relevance="not_relevant"),
    ])
    assert verdict.status is ObjectiveStatus.PARTIALLY_SUPPORTED
    assert verdict.confidence == 0.5


def test_aggregate_unresolved_builds_gap_from_notes():
    verdict = _aggregate(_objective(), [
        PassageAssessment(chunk_id="c1", relevance="not_relevant", note="about CABG only"),
        PassageAssessment(chunk_id="c2", relevance="not_relevant", note="off-topic methods"),
    ])
    assert verdict.status is ObjectiveStatus.UNRESOLVED
    assert verdict.gap.startswith("none of the 2 retrieved passage(s)")
    assert "CABG" in verdict.gap and "methods" in verdict.gap


def test_aggregate_empty_passages():
    verdict = _aggregate(_objective(), [])
    assert verdict.status is ObjectiveStatus.UNRESOLVED
    assert verdict.gap == ""
    assert verdict.objective_id == "H1"


# -- pipeline budget for sequential verification ------------------------------

class _StubExecutor:
    def __init__(self, n):
        self.n = n

    def n_verify_candidates(self, state, objective_id):
        return self.n


class _StubOrchestrator:
    async def decide(self, state):  # pragma: no cover - unused here
        raise AssertionError("not called")


def _pipeline(n_passages: int) -> AgenticV2Pipeline:
    return AgenticV2Pipeline(
        config=AppConfig(),
        orchestrator=_StubOrchestrator(),
        executor=_StubExecutor(n_passages),
    )


def test_verify_action_timeout_scales_with_passages():
    from src.agentic_v2.orchestrator import ActionDecision

    pipe = _pipeline(4)
    d = ActionDecision(action=ActionType.VERIFY, objective_id="H1")
    base = float(getattr(pipe.config, "agentic_v2_action_timeout", 180.0))
    assert pipe._action_timeout(d, ResearchState(question="q")) == pytest.approx(base * 4)


def test_non_verify_actions_keep_base_timeout():
    from src.agentic_v2.orchestrator import ActionDecision

    pipe = _pipeline(5)
    d = ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1")
    base = float(getattr(pipe.config, "agentic_v2_action_timeout", 180.0))
    assert pipe._action_timeout(d, ResearchState(question="q")) == pytest.approx(base)


def test_verify_candidate_cap_still_bounds_calls():
    # sanity: the module-level cap matches actions.py's bounded candidate list
    assert _MAX_PASSAGES == 6


# -- full-document backstop (defense in depth) ------------------------------

def test_is_full_document_detects_markers():
    from src.agentic_v2.verify import _is_full_document

    # explicit section marker from old READ_DOCUMENT
    assert _is_full_document(CandidatePassage(document_id="P1", section="(full document)", text="x" * 5000))
    # a passage with no chunk id but massive text is caught by the heuristic
    assert _is_full_document(CandidatePassage(document_id="P1", section="Results", text="word " * 9000))
    # a normal unit-like passage is fine
    assert not _is_full_document(CandidatePassage(chunk_id="c1", document_id="P1", section="Results", text="short"))
    # id-less huge blob heuristic
    assert _is_full_document(CandidatePassage(document_id="P1", section="Results", text="word " * 5000))
    # id-less but short = not treated as full doc
    assert not _is_full_document(CandidatePassage(document_id="P1", section="Results", text="word " * 100))


def test_aggregate_ignores_nothing_when_all_units():
    # sanity: the helper never throws and works on plain units
    from src.agentic_v2.verify import _is_full_document

    ok = CandidatePassage(chunk_id="c1", document_id="P1", section="Results", text="para")
    assert _is_full_document(ok) is False

