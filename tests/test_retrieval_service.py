"""Retrieval service tests: query-keyed caching, exclusion lists, provenance.

Uses an in-memory BM25 index + a fake corpus; components are injected directly
(no disk IO), which also exercises the lazy `_components()` path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.agents.planner import SubQuery
from src.retrieval.retriever import (
    RetrievedDocument,
    RetrievalService,
    get_retrieval_service,
    reset_retrieval_service,
)

CORPUS_ROWS = [
    # chunk_id, document_id, chunk_type, breadcrumb, text
    ("c1", "PMC-A", "table_row", ["Results", "Techniques"],
     "End-to-end anastomosis was associated with recurrent coarctation in 23.1% of patients (p = 0.04)"),
    ("c2", "PMC-A", "table_row", ["Results", "Techniques"],
     "Patch aortoplasty showed a recurrence rate of 8.3% (p = 0.31)"),
    ("c3", "PMC-B", "paragraph", ["Imaging"],
     "MRI demonstrated focal narrowing at the anastomotic site in the recurrent lesions"),
    ("c4", "PMC-C", "paragraph", ["Methods"],
     "Unrelated background about statistical methods in general."),
]

TEXT_BY_ID = {r[0]: r[4] for r in CORPUS_ROWS}


class FakeCorpus:
    """Minimal stand-in for CorpusIndex (resolve + _df + document_id)."""

    def __init__(self, rows) -> None:
        self._df = pd.DataFrame(
            [
                {
                    "id": r[0],
                    "document_id": r[1],
                    "chunk_type": r[2],
                    "breadcrumb_str": " > ".join(r[3]),
                    "text": r[4],
                    "section": r[3][0] if r[3] else "",
                    "subsection": "",
                    "table_id": "T2" if r[2] == "table_row" else None,
                    "figure_id": None,
                }
                for r in rows
            ]
        )
        self._id_to_doc = dict(zip(self._df["id"], self._df["document_id"]))
        self.loaded = True

    def document_id(self, chunk_id: str):
        return self._id_to_doc.get(chunk_id)

    def resolve(self, chunk_ids, include_text=False):
        out = {}
        subset = self._df.loc[self._df["id"].isin(list(chunk_ids))]
        for row in subset.itertuples(index=False):
            record = {
                "document_id": row.document_id,
                "chunk_type": row.chunk_type,
                "breadcrumb": row.breadcrumb_str,
            }
            if include_text:
                record["text"] = row.text
            out[row.id] = record
        return out


def _build_bm25(tmp_path: Path):
    from src.retrieval.sparse import BM25Index

    return BM25Index.build(lambda: iter(TEXT_BY_ID.items()), tmp_path / "bm25", k1=1.5, b=0.75)


def _service(tmp_path: Path, **overrides) -> RetrievalService:
    cfg = AppConfig()
    cfg.index_dir = tmp_path
    cfg.enable_dense = False
    cfg.max_documents = overrides.pop("max_documents", 8)
    bm25 = _build_bm25(tmp_path)
    svc = RetrievalService(cfg)
    # Inject loaded components directly: skips disk loads, exercises the same
    # code path used in production after _load_components().
    svc._bm25 = bm25
    svc._corpus = FakeCorpus(CORPUS_ROWS)
    return svc


def _subquery(query: str | None = None, sub_id: str = "H1") -> SubQuery:
    return SubQuery(
        id=sub_id,
        target="repair techniques associated with recurrent coarctation",
        query=query or "repair techniques associated with recurrent coarctation percentages p-values",
        focus="comparison",
        evidence_required=["percentages", "p-values"],
        terminology=["patch aortoplasty", "end-to-end anastomosis"],
    )


async def test_produces_retrievable_candidates_with_provenance(tmp_path):
    service = _service(tmp_path)
    docs = await service.search_subquery(_subquery())

    assert docs, "the subquery must retrieve candidates"
    assert docs[0].document_id == "PMC-A"
    for doc in docs:
        assert isinstance(doc, RetrievedDocument)
        assert doc.subquery_id == "H1"
        assert doc.chunk_id
        assert doc.document_id in {r[1] for r in CORPUS_ROWS}
        assert doc.text
        assert doc.rrf_score >= 0.0
        assert "bm25" in doc.methods
    assert docs[0].rrf_score > 0.0
    ranks = [d.rank for d in docs]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1
    assert any(d.table_id == "T2" for d in docs)


async def test_cache_is_keyed_by_query_text_not_subquery_id(tmp_path):
    service = _service(tmp_path)
    first = await service.search_subquery(_subquery(sub_id="H1"))
    # Same query under a DIFFERENT id must hit the same cache entry (the old
    # implementation keyed by sub.id and always saw "H1" from tools).
    second = await service.search_subquery(_subquery(sub_id="H9"))
    assert [d.chunk_id for d in first] == [d.chunk_id for d in second]

    other = await service.search_subquery(_subquery(query="imaging focal narrowing recurrent lesions", sub_id="H2"))
    assert any(d.document_id == "PMC-B" for d in other)


async def test_exclusion_list_removes_seen_chunks(tmp_path):
    service = _service(tmp_path)
    all_docs = await service.search_subquery(_subquery())
    seen = {d.chunk_id for d in all_docs}
    assert seen

    again = await service.search_subquery(_subquery(), exclude_chunk_ids=sorted(seen))
    assert not ({d.chunk_id for d in again} & seen), \
        "excluded chunks must never resurface in the exclusion round"


async def test_singleton_service_is_shared(tmp_path):
    reset_retrieval_service()
    s1 = get_retrieval_service()
    s2 = get_retrieval_service()
    assert s1 is s2
    reset_retrieval_service()


async def test_missing_index_fails_clearly(tmp_path):
    cfg = AppConfig()
    cfg.index_dir = tmp_path / "missing"
    cfg.enable_dense = False
    service = RetrievalService(cfg)
    try:
        await service.search_subquery(_subquery())
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "retrieval components failed to load" in str(exc)
