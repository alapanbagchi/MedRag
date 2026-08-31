"""Unit tests for Step 3 retriever tool (mocked service + unit index)."""
import pytest

from src.agentic.planner import PlannedEntity, SubQueryPlan
from src.agentic.retriever_tool import HybridRetrieverTool, RetrievalResult


class FakeDoc:
    def __init__(self, chunk_id, document_id, text, rrf_score, section="", node_type="paragraph", methods=None):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.rrf_score = rrf_score
        self.section = section
        self.node_type = node_type
        self.methods = methods or ["bm25"]


class FakeUnitIndex:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, chunk_id):
        return self.mapping.get(chunk_id, "")

    def unit_kind(self, chunk_id):
        return "table" if chunk_id.startswith("tbl") else "paragraph"


class FakeCorpus:
    def __init__(self):
        pass


class FakeService:
    def __init__(self, docs):
        self.docs = docs
        self.exclude_seen = None

    def _components(self):
        return {"corpus": FakeCorpus()}

    async def search_subquery(self, sub, exclude_chunk_ids=None):
        self.exclude_seen = exclude_chunk_ids
        return [d for d in self.docs if d.chunk_id not in (exclude_chunk_ids or [])]


def _sub():
    return SubQueryPlan(
        id="H1",
        target="radial artery vasospasm",
        query="radial artery vasospasm prevention",
        evidence_required=["pharmacological agents"],
        entities=[PlannedEntity(text="vasospasm", role="condition")],
    )


def test_to_legacy_subquery_adapts_fields():
    tool = HybridRetrieverTool(service=FakeService([]))
    legacy = tool._to_legacy_subquery(_sub())
    assert legacy.id == "H1"
    assert legacy.query == "radial artery vasospasm prevention"
    assert legacy.evidence_required == ["pharmacological agents"]
    assert legacy.terminology == ["vasospasm"]  # entity texts


@pytest.mark.asyncio
async def test_search_restores_paragraphs_and_ranks():
    docs = [
        FakeDoc("c1", "PMC1", "chunk text one", 0.9, section="Results"),
        FakeDoc("c2", "PMC2", "chunk text two", 0.7),
    ]
    service = FakeService(docs)
    tool = HybridRetrieverTool(service=service)
    tool._unit_index = lambda corpus: FakeUnitIndex({
        "c1": "FULL paragraph one restored", "c2": "FULL paragraph two restored",
    })
    results = await tool.search(_sub(), top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], RetrievalResult)
    assert results[0].rank == 1
    assert results[0].chunk_id == "c1"
    assert results[0].paragraph_text == "FULL paragraph one restored"
    assert results[1].paragraph_text == "FULL paragraph two restored"


@pytest.mark.asyncio
async def test_search_honors_exclusion():
    docs = [FakeDoc("c1", "PMC1", "t1", 0.9), FakeDoc("c2", "PMC2", "t2", 0.7)]
    service = FakeService(docs)
    tool = HybridRetrieverTool(service=service)
    tool._unit_index = lambda corpus: FakeUnitIndex({"c1": "p1", "c2": "p2"})
    results = await tool.search(_sub(), top_k=5, exclude_chunk_ids=["c1"])
    assert [r.chunk_id for r in results] == ["c2"]
    assert service.exclude_seen == ["c1"]


@pytest.mark.asyncio
async def test_search_returns_empty_on_failure():
    class BoomService:
        def _components(self):
            raise RuntimeError("index down")
        async def search_subquery(self, sub, exclude_chunk_ids=None):
            raise RuntimeError("boom")
    tool = HybridRetrieverTool(service=BoomService())
    results = await tool.search(_sub(), top_k=5)
    assert results == []
