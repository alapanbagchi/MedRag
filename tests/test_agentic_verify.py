"""Unit tests for Step 4 verification tool (fake verifier)."""
import pytest

from src.agentic.planner import PlannedEntity, SubQueryPlan
from src.agentic.retriever_tool import RetrievalResult
from src.agentic.verify import VerifyTool, VerifiedUnit, VerificationOutcome


class FakeVerdict:
    def __init__(self, document_id, relevance, confidence=0.0, reason=""):
        self.document_id = document_id
        self.relevance = relevance
        self.confidence = confidence
        self.reason = reason


class FakeVerificationResult:
    def __init__(self, results):
        self.results = results


class FakeVerifier:
    def __init__(self, mapping):
        self.mapping = mapping

    async def verify_papers(self, papers, query, evidence_required):
        return FakeVerificationResult([
            FakeVerdict(p["document_id"], **self.mapping.get(p["document_id"],
                          {"relevance": "not_relevant", "confidence": 0.0, "reason": ""}))
            for p in papers
        ])


def _sub():
    return SubQueryPlan(
        id="H1", target="radial artery vasospasm", query="radial artery vasospasm prevention",
        evidence_required=["pharmacological agents"],
        entities=[PlannedEntity(text="vasospasm")],
    )


def _result(cid, doc, text, score=1.0):
    return RetrievalResult(rank=1, chunk_id=cid, document_id=doc, paragraph_text=text,
                           section="Discussion", unit_kind="paragraph", rrf_score=score)


@pytest.mark.asyncio
async def test_verify_buckets_keep_reject_unknown():
    verifier = FakeVerifier({
        "c1": {"relevance": "relevant", "confidence": 0.9, "reason": "discusses vasospasm"},
        "c2": {"relevance": "not_relevant", "confidence": 0.8, "reason": "unrelated topic"},
        "c3": {"relevance": "unknown", "confidence": 0.0, "reason": "quota exceeded"},
    })
    tool = VerifyTool(verifier=verifier)
    results = [_result("c1", "PMC1", "txt1"), _result("c2", "PMC2", "txt2"), _result("c3", "PMC3", "txt3")]
    outcome = await tool.verify(_sub(), results)
    assert isinstance(outcome, VerificationOutcome)
    assert [u.chunk_id for u in outcome.kept] == ["c1"]
    assert [u.chunk_id for u in outcome.rejected] == ["c2"]
    assert [u.chunk_id for u in outcome.unknown] == ["c3"]
    assert outcome.rejection_reasons == ["unrelated topic"]
    assert outcome.kept_count == 1


@pytest.mark.asyncio
async def test_partially_relevant_is_kept():
    verifier = FakeVerifier({"c1": {"relevance": "partially_relevant", "confidence": 0.6, "reason": ""}})
    tool = VerifyTool(verifier=verifier)
    outcome = await tool.verify(_sub(), [_result("c1", "PMC1", "txt")])
    assert outcome.kept_count == 1
    assert outcome.kept[0].verdict == "keep"


@pytest.mark.asyncio
async def test_missing_verdict_counts_as_unknown_not_reject():
    verifier = FakeVerifier({})  # will emit no matching id -> missing
    # FakeVerifier maps every paper via mapping.get(...) with default not_relevant, so
    # use a verifier that returns an empty result list instead.
    class EmptyVerifier:
        async def verify_papers(self, papers, query, evidence_required):
            return FakeVerificationResult([])
    tool = VerifyTool(verifier=EmptyVerifier())
    outcome = await tool.verify(_sub(), [_result("c1", "PMC1", "txt")])
    assert outcome.kept == []
    assert outcome.rejected == []
    assert [u.chunk_id for u in outcome.unknown] == ["c1"]
    assert outcome.unknown[0].reason == "verifier missing verdict"


@pytest.mark.asyncio
async def test_total_failure_is_unknown_not_reject():
    class BoomVerifier:
        async def verify_papers(self, papers, query, evidence_required):
            raise RuntimeError("infra down")
    tool = VerifyTool(verifier=BoomVerifier())
    outcome = await tool.verify(_sub(), [_result("c1", "PMC1", "txt")])
    assert outcome.kept == [] and outcome.rejected == []
    assert [u.chunk_id for u in outcome.unknown] == ["c1"]
    assert "infra down" in outcome.unknown[0].reason


@pytest.mark.asyncio
async def test_empty_results_short_circuit():
    tool = VerifyTool(verifier=FakeVerifier({}))
    outcome = await tool.verify(_sub(), [])
    assert outcome.kept == [] and outcome.rejected == [] and outcome.unknown == []
    assert outcome.subquery_id == "H1"
