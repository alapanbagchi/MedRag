"""Hybrid evidence-retrieval tool tests (stubbed retriever, no DB/models)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.agents.deep_agent import build_deep_agent
from src.tools import retrieval as retrieval_mod
from src.tools.retrieval import (
    EvidenceHit,
    EvidenceRetriever,
    assemble_passages,
    local_search,
    rrf_fuse,
)


class StubRetriever:
    """Stand-in for EvidenceRetriever; returns fixed hits."""

    def __init__(self, hits: list):
        self._hits = hits
        self.seen: list = []

    def search(self, query, per_leg=60, top_k=10, progress=None):
        self.seen.append(query)
        return self._hits


def _hit(**over):
    base = {
        "chunk_id": "c1",
        "document_id": "PMC1",
        "chunk_type": "paragraph",
        "section": "Results",
        "score": 0.9,
        "text": "passage one",
    }
    base.update(over)
    return EvidenceHit(**base)


def test_rrf_fuse_prefers_multi_leg_hits():
    assert rrf_fuse([["a", "b", "c"], ["c"]]) == ["c", "a", "b"]


def test_rrf_fuse_dedupes():
    assert rrf_fuse([["a", "b"], ["b", "a"]]) == ["a", "b"]


def test_assemble_table_row_expands_to_whole_table():
    metas = {
        "row1": {"id": "row1", "document_id": "PMC1", "chunk_type": "table_row",
                 "section": "Results", "parent_id": "sum1", "table_id": "t1",
                 "document_position": 2, "text": "row one"},
    }
    table_groups = {"t1": [
        {"id": "sum1", "chunk_type": "table_summary", "document_position": 1,
         "text": "Table 1. Caption."},
        {"id": "row1", "chunk_type": "table_row", "document_position": 2, "text": "row one"},
        {"id": "row2", "chunk_type": "table_row", "document_position": 3, "text": "row two"},
    ]}
    (hit,) = assemble_passages([("row1", 0.8)], metas, table_groups, {}, {}, metas)
    assert hit.text == "Table 1. Caption.\nrow one\nrow two"
    assert hit.document_id == "PMC1"


def test_assemble_paragraph_joins_siblings():
    metas = {
        "p2": {"id": "p2", "document_id": "PMC1", "chunk_type": "paragraph",
               "section": "Methods", "parent_id": "u1", "table_id": "",
               "document_position": 2, "text": "second half"},
    }
    unit_groups = {"u1": [
        {"id": "p1", "chunk_type": "paragraph", "document_position": 1, "text": "first half"},
        {"id": "p2", "chunk_type": "paragraph", "document_position": 2, "text": "second half"},
    ]}
    (hit,) = assemble_passages([("p2", 0.7)], metas, {}, unit_groups, {}, metas)
    assert hit.text == "first half\nsecond half"


def test_assemble_figure_passes_through():
    metas = {
        "f1": {"id": "f1", "document_id": "PMC1", "chunk_type": "figure",
               "section": "Results", "parent_id": "u9", "table_id": "",
               "document_position": 5, "text": "Figure 1: caption"},
    }
    (hit,) = assemble_passages([("f1", 0.5)], metas, {}, {}, {}, metas)
    assert hit.text == "Figure 1: caption"


def test_assemble_summary_without_table_id_links_via_parent():
    # Regression: a table_summary hit with an empty table_id must not crash;
    # rows are found through the summary parent instead.
    metas = {
        "sum1": {"id": "sum1", "document_id": "PMC1", "chunk_type": "table_summary",
                 "section": "Results", "parent_id": "", "table_id": "",
                 "document_position": 1, "text": "Table 1. Caption."},
    }
    child_groups = {"sum1": [
        {"id": "row1", "chunk_type": "table_row", "document_position": 2, "text": "row one"},
        {"id": "row2", "chunk_type": "table_row", "document_position": 3, "text": "row two"},
    ]}
    (hit,) = assemble_passages([("sum1", 0.8)], metas, {}, {}, child_groups, metas)
    assert hit.text == "Table 1. Caption.\nrow one\nrow two"


def test_assemble_row_without_table_id_links_via_parent():
    metas = {
        "row1": {"id": "row1", "document_id": "PMC1", "chunk_type": "table_row",
                 "section": "Results", "parent_id": "sum1", "table_id": "",
                 "document_position": 2, "text": "row one"},
    }
    by_id = dict(metas, **{"sum1": {
        "id": "sum1", "chunk_type": "table_summary", "document_position": 1,
        "text": "Table 1. Caption.",
    }})
    child_groups = {"sum1": [
        {"id": "row1", "chunk_type": "table_row", "document_position": 2, "text": "row one"},
    ]}
    (hit,) = assemble_passages([("row1", 0.8)], metas, {}, {}, child_groups, by_id)
    assert hit.text == "Table 1. Caption.\nrow one"


class FakeCursor:
    def execute(self, *args, **kwargs):
        pass

    def fetchall(self):
        return []

    def close(self):
        pass


class FakeStore:
    def __init__(self, ids):
        self._ids = ids

    def search(self, vec, top_k=60):
        return [(cid, 0.9) for cid in self._ids[:top_k]]

    def get_chunks(self, ids):
        return {
            cid: {"id": cid, "document_id": "PMC1", "chunk_type": "paragraph",
                  "section": "Results", "parent_id": "", "table_id": "",
                  "document_position": 0, "text": f"text {cid}"}
            for cid in ids
        }

    def connect(self):
        return SimpleNamespace(cursor=lambda: FakeCursor())


class FakeEncoder:
    def encode_single(self, query):
        import numpy as np

        return np.zeros(768, dtype=np.float32)


class FakeBm25:
    def __init__(self, ids):
        self._ids = ids

    def search_single(self, query, top_k):
        return [SimpleNamespace(chunk_id=cid, score=1.0) for cid in self._ids[:top_k]]


class CountingReranker:
    def __init__(self):
        self.n_pairs = 0

    def score(self, pairs):
        self.n_pairs = len(pairs)
        return [1.0 - i * 0.001 for i in range(len(pairs))]


def test_search_cross_reranks_60_and_returns_10():
    dense_ids = [f"d{i}" for i in range(60)]
    sparse_ids = [f"s{i}" for i in range(60)]
    reranker = CountingReranker()
    retriever = EvidenceRetriever(
        store=FakeStore(dense_ids),
        encoder=FakeEncoder(),
        bm25=FakeBm25(sparse_ids),
        reranker=reranker,
    )
    hits = retriever.search("what is diabetes")
    assert reranker.n_pairs == 60
    assert len(hits) == 10


async def test_retrieve_evidence_returns_hits_json(monkeypatch):
    stub = StubRetriever([_hit(), _hit(chunk_id="c2", text="passage two")])
    monkeypatch.setattr(retrieval_mod, "_retriever", stub)
    data = json.loads(await local_search(SimpleNamespace(), "what is diabetes"))
    assert [h["chunk_id"] for h in data] == ["c1", "c2"]
    assert data[0]["text"] == "passage one"
    assert stub.seen == ["what is diabetes"]


class BoomReranker:
    def score(self, pairs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 346.00 MiB")


def test_search_dead_reranker_falls_back_to_fusion_order(capsys):
    dense_ids = [f"d{i}" for i in range(60)]
    sparse_ids = [f"s{i}" for i in range(60)]
    retriever = EvidenceRetriever(
        store=FakeStore(dense_ids),
        encoder=FakeEncoder(),
        bm25=FakeBm25(sparse_ids),
        reranker=BoomReranker(),
    )
    stages: list = []
    hits = retriever.search("what is diabetes", progress=stages.append)
    assert len(hits) == 10
    # Fusion order kept, scores marked unranked.
    fused = rrf_fuse([dense_ids, sparse_ids])
    assert [h.chunk_id for h in hits] == fused[:10]
    assert all(h.score == 0.0 for h in hits)
    assert "keeping fusion order" in "\n".join(stages)


def test_search_caches_exact_repeat_queries():
    """A verbatim repeat skips the encoder, both DB legs and the rerank."""
    dense_ids = [f"d{i}" for i in range(60)]
    sparse_ids = [f"s{i}" for i in range(60)]
    reranker = CountingReranker()
    retriever = EvidenceRetriever(
        store=FakeStore(dense_ids),
        encoder=FakeEncoder(),
        bm25=FakeBm25(sparse_ids),
        reranker=reranker,
    )
    first = retriever.search("what is diabetes")
    stages: list = []
    second = retriever.search("what is diabetes", progress=stages.append)
    assert [h.chunk_id for h in first] == [h.chunk_id for h in second]
    assert reranker.n_pairs == 60  # reranked once, not twice
    assert "retrieval cache hit" in "\n".join(stages)


def test_cross_encoder_batch_size_from_env(monkeypatch):
    from src.tools.retrieval import MedCPTCrossEncoder

    monkeypatch.setenv("CROSS_ENCODER_BATCH_SIZE", "8")
    assert MedCPTCrossEncoder().batch_size == 8
    monkeypatch.setenv("CROSS_ENCODER_BATCH_SIZE", "junk")
    assert MedCPTCrossEncoder().batch_size == 16
    assert MedCPTCrossEncoder(batch_size=64).batch_size == 64


def test_warmup_builds_missing_components_once():
    calls: list = []

    class CountingEncoder:
        def __init__(self, *args, **kwargs):
            calls.append("encoder")

        def encode_single(self, query):
            return [0.0]

    class CountingReranker:
        def __init__(self, *args, **kwargs):
            calls.append("reranker")

        def score(self, pairs):
            return [0.0] * len(pairs)

    import src.tools.retrieval as mod

    orig_encoder, orig_reranker = mod.MedCPTQueryEncoder, mod.MedCPTCrossEncoder
    mod.MedCPTQueryEncoder, mod.MedCPTCrossEncoder = CountingEncoder, CountingReranker
    try:
        retriever = EvidenceRetriever(
            store=FakeStore([]), encoder=None, bm25=FakeBm25([]), reranker=None,
        )
        assert retriever.warmup() is True
        assert retriever.warmup() is True  # second run is a no-op
        assert calls == ["encoder", "reranker"]
    finally:
        mod.MedCPTQueryEncoder, mod.MedCPTCrossEncoder = orig_encoder, orig_reranker


def test_warmup_concurrent_with_search_builds_once():
    import threading
    import time as _time

    import src.tools.retrieval as mod

    calls: list = []

    class SlowReranker:
        def __init__(self, *args, **kwargs):
            _time.sleep(0.2)
            calls.append("reranker")

        def score(self, pairs):
            return [0.5] * len(pairs)

    orig = mod.MedCPTCrossEncoder
    mod.MedCPTCrossEncoder = SlowReranker
    try:
        retriever = EvidenceRetriever(
            store=FakeStore(["d0"]), encoder=FakeEncoder(),
            bm25=FakeBm25(["d0"]), reranker=None,
        )
        thread = threading.Thread(target=retriever.warmup)
        thread.start()
        hits = retriever.search("what is diabetes")
        thread.join()
        assert calls == ["reranker"]
        assert len(hits) == 1
    finally:
        mod.MedCPTCrossEncoder = orig


def test_warmup_reports_partial_on_failure(capsys):
    import src.tools.retrieval as mod

    class BoomReranker:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no CUDA here")

    orig = mod.MedCPTCrossEncoder
    mod.MedCPTCrossEncoder = BoomReranker
    try:
        retriever = EvidenceRetriever(
            store=FakeStore([]), encoder=FakeEncoder(),
            bm25=FakeBm25([]), reranker=None,
        )
        stages: list = []
        assert retriever.warmup(stages.append) is False
        assert "not ready" in "\n".join(stages)
    finally:
        mod.MedCPTCrossEncoder = orig


def test_search_reports_pipeline_progress():
    dense_ids = [f"d{i}" for i in range(60)]
    sparse_ids = [f"s{i}" for i in range(60)]
    retriever = EvidenceRetriever(
        store=FakeStore(dense_ids),
        encoder=FakeEncoder(),
        bm25=FakeBm25(sparse_ids),
        reranker=CountingReranker(),
    )
    stages: list = []
    hits = retriever.search("what is diabetes", progress=stages.append)
    assert len(hits) == 10
    text = "\n".join(stages)
    assert "dense leg: fetched 60" in text
    assert "bm25 leg: fetched 60" in text
    assert "keeping top 60" in text
    assert "scored 60 pairs -> top 10" in text
    assert "expanded 10 passages" in text


def test_search_silent_without_progress(capsys):
    retriever = EvidenceRetriever(
        store=FakeStore(["d0"]),
        encoder=FakeEncoder(),
        bm25=FakeBm25(["d0"]),
        reranker=CountingReranker(),
    )
    assert len(retriever.search("what is diabetes")) == 1
    assert capsys.readouterr().out == ""


async def test_retrieve_evidence_blank_query_is_empty(monkeypatch):
    stub = StubRetriever([_hit()])
    monkeypatch.setattr(retrieval_mod, "_retriever", stub)
    assert await local_search(SimpleNamespace(), "   ") == "[]"
    assert stub.seen == []


async def test_retrieve_evidence_narrates_pipeline(monkeypatch, capsys):
    dense_ids = [f"d{i}" for i in range(60)]
    sparse_ids = [f"s{i}" for i in range(60)]
    retriever = EvidenceRetriever(
        store=FakeStore(dense_ids),
        encoder=FakeEncoder(),
        bm25=FakeBm25(sparse_ids),
        reranker=CountingReranker(),
    )
    monkeypatch.setattr(retrieval_mod, "_retriever", retriever)
    hits = json.loads(await local_search(SimpleNamespace(), "what is diabetes"))
    assert len(hits) == 10
    out = capsys.readouterr().out
    assert "[local_search] dense leg: fetched 60" in out
    assert "[local_search] bm25 leg: fetched 60" in out
    assert "scored 60 pairs -> top 10" in out


def test_build_deep_agent_registers_retrieval_tool(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    agent = build_deep_agent()
    names = [name for toolset in agent.toolsets for name in toolset.tools]
    assert "local_search" in names
