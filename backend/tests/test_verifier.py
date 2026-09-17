"""Verifier tests (fake System One client, no network)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice, Noul

from src.agents.planner import PlanItem
from src.tools.verifier import (
    INTENT_KEY,
    NONE_OPTION,
    EvidenceRequirement,
    PassageVerdict,
    VerdictResponse,
    _coerce_passage,
    _span_candidates,
    build_questions,
    build_state,
    coverage_key,
    is_verbatim_excerpt,
    normalize_response,
    verdict_from_response,
    verbatim_key,
    verify_passages,
)


class _Noul:
    def __init__(self, noul):
        self.noul = noul


class _Choice:
    def __init__(self, choice):
        self.choice = choice


class _Usage:
    input_tokens = 12
    output_tokens = 3


class _Response:
    def __init__(self, nouls=None, choices=None):
        self.nouls = nouls or {}
        self.choices = choices or {}
        self.usage = _Usage()


class FakeClient:
    """Answers coverage Noul from a pid->covered-id map and intent per pid."""

    def __init__(self, coverage=None, intent=None, choices=None, boom=False):
        self._coverage = coverage or {}
        self._intent = intent or {}
        self._choices = choices or {}
        self._boom = boom
        self.seen = []

    async def system_one(self, state, questions):
        if self._boom:
            raise RuntimeError("typesafe down")
        self.seen.append((state, questions))
        pid = state["passage"]["id"]
        covered = set(self._coverage.get(pid, []))
        nouls = {}
        for key in questions:
            if key == INTENT_KEY:
                nouls[key] = _Noul(self._intent.get(pid, 0.5))
            elif key.startswith("coverage::"):
                rid = key.split("::", 1)[1]
                nouls[key] = _Noul(1.0 if rid in covered else 0.0)
        choices = {key: _Choice(label)
                   for key, label in self._choices.items()}
        return _Response(nouls=nouls, choices=choices)


def _reqs():
    return [EvidenceRequirement(id="E1", description="definition"),
            EvidenceRequirement(id="E2", description="criteria")]


def test_verdict_models_validate_spec_shape():
    response = VerdictResponse.model_validate({"evidence_results": [{
        "passage_id": "P1", "intent_score": 0.75,
        "coverage": ["E1"], "reason": "r", "verbatim": "High blood pressure."}]})
    verdict = response.evidence_results[0]
    assert isinstance(verdict, PassageVerdict)
    assert (verdict.intent_score, verdict.coverage) == (0.75, ["E1"])


def test_build_state_is_structured_and_references_the_passage():
    state = build_state("What is hypertension?", _reqs(),
                        {"id": "c1", "text": "high blood pressure"})
    assert state["question"] == "What is hypertension?"
    assert state["evidence_requirements"] == [
        {"id": "E1", "description": "definition"},
        {"id": "E2", "description": "criteria"}]
    assert state["passage"] == {"id": "c1", "text": "high blood pressure"}


def test_build_questions_asks_intent_and_one_noul_per_requirement():
    questions = build_questions("Q?", _reqs(), [], ask_verbatim=False)
    assert isinstance(questions[INTENT_KEY], Noul)
    assert isinstance(questions[coverage_key("E1")], Noul)
    assert isinstance(questions[coverage_key("E2")], Noul)
    assert verbatim_key("E1") not in questions
    # The requirement text reaches the instructions.
    assert "definition" in questions[coverage_key("E1")].instructions


def test_build_questions_adds_span_choices_only_when_asked():
    spans = ["one.", "two."]
    questions = build_questions("Q?", _reqs(), spans, ask_verbatim=True)
    assert isinstance(questions[verbatim_key("E1")], Choice)
    assert set(questions[verbatim_key("E1")].criteria) == {"s0", "s1",
                                                           NONE_OPTION}
    # No spans means no Choice questions at all.
    assert verbatim_key("E1") not in build_questions(
        "Q?", _reqs(), [], ask_verbatim=True)


def test_verdict_from_response_thresholds_coverage():
    passage = {"id": "P1", "text": "a relevant sentence."}
    response = _Response(nouls={
        INTENT_KEY: _Noul(0.8),
        coverage_key("E1"): _Noul(0.6),
        coverage_key("E2"): _Noul(0.4),
    })
    verdict = verdict_from_response(response, _reqs(), passage, [], 0.5, False)
    assert verdict.intent_score == 0.8
    assert verdict.coverage == ["E1"]
    assert "E1 0.60>=0.50" in verdict.reason
    assert "E2 0.40<0.50" in verdict.reason


def test_verdict_from_response_copies_the_selected_span_verbatim():
    text = "First sentence here. Second sentence carries the evidence."
    spans = ["First sentence here.", "Second sentence carries the evidence."]
    response = _Response(
        nouls={INTENT_KEY: _Noul(0.9), coverage_key("E1"): _Noul(0.95)},
        choices={verbatim_key("E1"): _Choice("s1")},
    )
    verdict = verdict_from_response(response, _reqs()[:1],
                                    {"id": "P1", "text": text}, spans, 0.5, True)
    assert verdict.verbatim == spans[1]
    assert is_verbatim_excerpt(verdict.verbatim, text)


def test_verdict_from_response_falls_back_to_first_span_when_none_selected():
    text = "Only sentence here."
    spans = ["Only sentence here."]
    response = _Response(
        nouls={INTENT_KEY: _Noul(0.9), coverage_key("E1"): _Noul(0.95)},
        choices={verbatim_key("E1"): _Choice(NONE_OPTION)},
    )
    verdict = verdict_from_response(response, _reqs()[:1],
                                    {"id": "P1", "text": text}, spans, 0.5, True)
    assert verdict.verbatim == text
    assert is_verbatim_excerpt(verdict.verbatim, text)


async def test_verify_passages_keeps_only_covered_passages():
    client = FakeClient(coverage={"c1": ["E1"], "c2": []},
                        intent={"c1": 0.9, "c2": 0.4})
    report = await verify_passages(
        "What is hypertension?",
        [EvidenceRequirement(id="E1", description="definition")],
        [{"id": "c1", "text": "hypertension is high blood pressure"},
         {"id": "c2", "text": "unrelated weather patterns"}],
        client=client,
    )
    assert report.judged is True
    assert [v.passage_id for v in report.evidence_results] == ["c1"]
    assert report.evidence_results[0].coverage == ["E1"]
    assert len(client.seen) == 2          # one System One call per passage
    assert report.prompt_tokens == 24     # 2 calls x 12 input tokens
    assert report.completion_tokens == 6


async def test_verify_passages_degrades_without_raising():
    report = await verify_passages(
        "What is hypertension?",
        [EvidenceRequirement(id="E1", description="definition")],
        [{"id": "c1", "text": "hypertension is high blood pressure"}],
        client=FakeClient(boom=True),
    )
    assert report.judged is False
    assert report.evidence_results == []


async def test_verify_passages_empty_inputs_skip_client():
    client = FakeClient()
    assert await verify_passages(
        "Q?", [], [{"id": "P1", "text": "t"}],
        client=client) == VerdictResponse(evidence_results=[])
    assert await verify_passages(
        "Q?", _reqs(), [], client=client) == VerdictResponse(evidence_results=[])
    assert client.seen == []


async def test_verify_passages_empty_question_raises():
    with pytest.raises(ValueError):
        await verify_passages("  ", _reqs(),
                              [{"id": "P1", "text": "t"}],
                              client=FakeClient())


def test_span_candidates_caps_count_and_keeps_the_tail():
    text = " ".join(f"Sentence number {i} here." for i in range(30))
    spans = _span_candidates(text)
    assert len(spans) == 12
    assert "Sentence number 29 here." in spans[-1]
    # Every span is an exact contiguous slice of the normalized passage.
    assert " ".join(spans).split() == text.split() or all(
        s in text for s in spans)


def test_coerce_passage_accepts_retrieval_web_and_citation_ids():
    assert _coerce_passage({"chunk_id": "c1", "text": "t"}, 0)["id"] == "c1"
    assert _coerce_passage({"citation_id": "w-0", "text": "t"}, 0)["id"] == "w-0"
    assert _coerce_passage({"url": "https://example.com", "snippet": "n"}, 0)[
        "id"] == "https://example.com"
    assert _coerce_passage({"text": "t"}, 2)["id"] == "P3"
    assert "Body text here." in _coerce_passage(
        {"url": "u", "markdown": "# G\n\nBody text here."}, 0)["text"]


def test_normalize_realigns_and_drops_hallucinated_ids():
    raw = VerdictResponse.model_validate({"evidence_results": [
        {"passage_id": "P2", "intent_score": 0.7, "coverage": ["E2"],
         "reason": "criteria"},
        {"passage_id": "P1", "intent_score": 0.9,
         "coverage": ["E1", "E99"], "reason": "defines it"},
        {"passage_id": "EXTRA", "intent_score": 1.0, "coverage": ["E1"]},
    ]})
    passages = [{"id": "P1", "text": "Substantive hypertension definition text here"},
                {"id": "P2", "text": "Substantive hypertension criteria text here"}]
    out = normalize_response(raw, _reqs(), passages)
    assert [v.passage_id for v in out.evidence_results] == ["P1", "P2"]
    p1, p2 = out.evidence_results
    assert (p1.intent_score, p1.coverage) == (0.9, ["E1"])
    assert (p2.intent_score, p2.coverage) == (0.7, ["E2"])


def test_normalize_zeroes_boilerplate():
    reqs = [EvidenceRequirement(id="E1", description="d")]
    passages = [{"id": "P1", "text": "Menu login subscribe"},
                {"id": "P2", "text": "Substantive hypertension text here about care"}]
    raw = VerdictResponse.model_validate({"evidence_results": [
        {"passage_id": "P1", "intent_score": 0.9, "coverage": ["E1"],
         "reason": "sloppy judge"}]})
    out = normalize_response(raw, reqs, passages)
    assert (out.evidence_results[0].intent_score,
            out.evidence_results[0].coverage) == (0.0, [])
    assert out.evidence_results[0].reason == "empty or boilerplate"


def test_is_verbatim_excerpt_tolerates_elisions_and_rejects_paraphrase():
    passage = "Ibuprofen inhibits COX-1 and COX-2 enzymes, reducing prostaglandin synthesis."
    assert is_verbatim_excerpt("inhibits COX-1 and COX-2 enzymes,", passage)
    assert is_verbatim_excerpt("Ibuprofen ... prostaglandin synthesis.", passage)
    assert not is_verbatim_excerpt("the of and", passage)
    assert not is_verbatim_excerpt("Ibuprofen blocks COX enzymes", passage)


# -- unrelated tests kept from the old module -----------------------------

def test_plan_item_carries_optional_evidence_requirements():
    assert PlanItem(question="Q?").evidence_requirements == []
    item = PlanItem.model_validate({
        "id": "P1", "question": "Q?", "deep_research": True,
        "evidence_requirements": [{"id": "E1", "description": "definition"}],
    })
    assert item.evidence_requirements[0].id == "E1"


def test_build_deep_agent_has_no_planner_tool(monkeypatch):
    from src.agents.deep_agent import build_deep_agent

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    agent = build_deep_agent()
    names = [name for toolset in agent.toolsets for name in toolset.tools]
    assert "plan_evidence_requirements" not in names
    assert "generate_plan" not in names
    assert "local_search" in names
    assert "check_evidence_gaps" not in names


def test_system_prompt_uses_provided_requirements():
    from src.agents.deep_agent import load_system_prompt

    prompt = load_system_prompt()
    assert "plan_evidence_requirements" not in prompt
    assert "check_evidence_gaps" not in prompt
    assert "local_search" in prompt
    assert "verbatim quotes" in prompt


def test_is_oom_matches_cuda_failures():
    from src.tools.retrieval import _is_oom

    oom = type("OutOfMemoryError", (Exception,), {})("boom")
    assert _is_oom(oom) is True
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) is True
    assert _is_oom(ValueError("bad input")) is False
