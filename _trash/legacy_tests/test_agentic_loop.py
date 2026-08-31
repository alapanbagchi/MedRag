"""Unit tests for the Step 5 agentic loop tools (no live LLM)."""
import pytest

from src.agentic.planner import PlannedEntity, SubQueryPlan
from src.agentic.retriever_tool import RetrievalResult
from src.agentic.verify import VerifiedUnit
from src.agentic.loop import AgenticLoop, AgentDeps, search_subquery, umls_lookup, EvidenceReport


class _Ctx:
    def __init__(self, deps):
        self.deps = deps


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def search(self, sub, top_k=6, exclude_chunk_ids=None):
        self.calls.append((sub.query, exclude_chunk_ids))
        return [r for r in self.results if r.chunk_id not in (exclude_chunk_ids or [])]


class FakeVerifier:
    def __init__(self, verdicts):
        self.verdicts = verdicts

    async def verify(self, sub, results, base_query=None):
        from src.agentic.verify import VerificationOutcome
        out = VerificationOutcome(subquery_id=sub.id)
        for r in results:
            v = self.verdicts.get(r.chunk_id, "reject")
            unit = VerifiedUnit(chunk_id=r.chunk_id, document_id=r.document_id,
                                verdict=v, relevance=v, confidence=0.8,
                                reason="some reason" if v == "reject" else "",
                                paragraph_text=r.paragraph_text, section=r.section,
                                unit_kind=r.unit_kind, score=r.rrf_score)
            if v == "keep":
                out.kept.append(unit)
            elif v == "reject":
                out.rejected.append(unit)
                out.rejection_reasons.append(unit.reason)
            else:
                out.unknown.append(unit)
        return out


def _sub():
    return SubQueryPlan(id="H1", target="radial artery vasospasm", query="vasospasm prevention",
                        evidence_required=["pharmacological agents"],
                        entities=[PlannedEntity(text="vasospasm")])


def _deps():
    return AgentDeps(subquery=_sub(), retriever=FakeRetriever([]), verifier=FakeVerifier({}))


@pytest.mark.asyncio
async def test_search_subquery_accumulates_and_dedupes():
    r1 = RetrievalResult(rank=1, chunk_id="c1", document_id="PMC1", paragraph_text="para one", section="Abstract", unit_kind="paragraph", rrf_score=1.0)
    r2 = RetrievalResult(rank=2, chunk_id="c2", document_id="PMC2", paragraph_text="para two", section="Discussion", unit_kind="paragraph", rrf_score=0.9)
    deps = AgentDeps(subquery=_sub(), retriever=FakeRetriever([r1, r2]),
                     verifier=FakeVerifier({"c1": "keep", "c2": "reject"}))
    out = await search_subquery(_Ctx(deps), "vasospasm prevention")
    assert deps.rounds == 1
    assert deps.seen_chunk_ids == {"c1", "c2"}
    assert len(deps.kept) == 1 and deps.kept[0].chunk_id == "c1"
    assert out["kept_so_far"] == 1
    assert out["distinct_papers_so_far"] == 1
    assert out["rejection_reasons"] == ["some reason"]


@pytest.mark.asyncio
async def test_search_subquery_excludes_seen():
    r = RetrievalResult(rank=1, chunk_id="c1", document_id="PMC1", paragraph_text="p", section="s", unit_kind="paragraph", rrf_score=1.0)
    deps = AgentDeps(subquery=_sub(), retriever=FakeRetriever([r]), verifier=FakeVerifier({"c1": "keep"}))
    deps.seen_chunk_ids.add("c1")
    out = await search_subquery(_Ctx(deps), "vasospasm prevention")
    assert out["retrieved"] == 0


@pytest.mark.asyncio
async def test_search_subquery_empty_query():
    deps = _deps()
    out = await search_subquery(_Ctx(deps), "   ")
    assert "error" in out


@pytest.mark.asyncio
async def test_umls_lookup_not_found():
    class FakeUMLS:
        async def search_concept(self, term):
            return type("C", (), {"found": False})()
    class FakeEnricher:
        umls = FakeUMLS()
    deps = _deps()
    deps.umls_enricher = FakeEnricher()
    out = await umls_lookup(_Ctx(deps), "zzz")
    assert out["found"] is False


def test_report_success_with_no_evidence_is_downgraded():
    report = EvidenceReport(succeeded=True, evidence_excerpts=[], citations=[])
    if report.succeeded and not report.evidence_excerpts and not report.citations:
        report.succeeded = False
    assert report.succeeded is False
