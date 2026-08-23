"""Unit tests for the retrieval layer (no full corpus required)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from medrag.retrieval.corpus import _cast_to_corpus_schema, corpus_schema
from medrag.retrieval.dense import DenseIndex
from medrag.retrieval.hybrid import fuse, reciprocal_rank_fusion
from medrag.retrieval.sparse import BM25Index, tokenize


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------

def test_reciprocal_rank_fusion_prefers_consensus():
    ranked = [
        [("a", 0.9), ("b", 0.8), ("c", 0.7)],
        [("b", 0.9), ("a", 0.8), ("c", 0.7)],
    ]
    scores = reciprocal_rank_fusion(ranked, k=60.0)
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["c"]
    # a and b each hold rank 1 and rank 2 -> equal RRF score
    assert abs(scores["a"] - scores["b"]) < 1e-12


def test_fuse_sorts_descending():
    ranked = [[("a", 0.9), ("b", 0.5)], [("b", 0.9), ("a", 0.5)]]
    merged = fuse(ranked, method="rrf", k=60.0)
    assert [cid for cid, _ in merged] == ["a", "b"]
    assert merged[0][1] >= merged[1][1]


def test_fuse_minmax_normalizes_scales():
    # Different raw scales should not bias minmax fusion.
    ranked = [[("a", 1000.0), ("b", 900.0)], [("a", 0.2), ("b", 0.1)]]
    scores = {cid: s for cid, s in fuse(ranked, method="minmax")}
    assert abs(scores["a"] - 2.0) < 1e-9
    assert abs(scores["b"] - 0.0) < 1e-9


def test_fuse_rejects_unknown_method():
    with pytest.raises(ValueError):
        fuse([[]], method="nope")


# ----------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------

def test_tokenize_lowercases_alnum():
    assert tokenize("Type 1 & Type 2 diabetes!", __import__("re").compile(r"[a-z0-9]+")) == [
        "type", "1", "type", "2", "diabetes",
    ]


def test_bm25_tiny_corpus(tmp_path):
    docs = [
        ("d0", "diabetes treatment insulin"),
        ("d1", "diabetes diabetes insulin"),
        ("d2", "unrelated cancer therapy"),
    ]

    idx = BM25Index.build(lambda: iter(docs), tmp_path / "bm25")

    # "diabetes" appears in d0 and d1; d1 repeats it, so d1 should rank first.
    hits = idx.search_single("diabetes", top_k=3)
    assert hits[0].chunk_id == "d1"
    assert hits[1].chunk_id == "d0"

    # A term absent from the corpus returns no scored hits (all zero).
    all_scores = [h.score for h in idx.search_single("zzzznope", top_k=3)]
    assert all(s == 0.0 for s in all_scores)


def test_bm25_roundtrip_save_load(tmp_path):
    docs = [("a", "alpha beta"), ("b", "beta gamma"), ("c", "alpha gamma delta")]
    built = BM25Index.build(lambda: iter(docs), tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.n_docs == 3
    assert loaded.vocab_size == built.vocab_size
    top = loaded.search_single("alpha", 3)
    # "alpha" occurs in docs a and c; c also has more terms, so a ranks first.
    assert top[0].chunk_id == "a"


# ----------------------------------------------------------------------
# Dense index
# ----------------------------------------------------------------------

def _write_emb_parquet(path, ids, vectors):
    table = pa.table(
        {
            "chunk_id": pa.array(ids, type=pa.string()),
            "embedding": pa.array([pa.array(v, type=pa.float32()) for v in vectors],
                                  type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, str(path))


def test_dense_build_and_self_retrieval(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10, 768)).astype(np.float32)
    # force unique directions
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(10)]
    _write_emb_parquet(tmp_path / "doc1.embeddings.parquet", ids, vectors)

    idx = DenseIndex.build([tmp_path / "doc1.embeddings.parquet"], tmp_path, index_type="flat")
    assert idx.n_total == 10

    loaded = DenseIndex.load(tmp_path)
    hits = loaded.search_single(vectors[3], top_k=3)
    assert hits[0].chunk_id == "c3"
    assert abs(hits[0].score - 1.0) < 1e-5


# ----------------------------------------------------------------------
# Corpus schema casting
# ----------------------------------------------------------------------

def test_cast_null_column_to_string():
    t1 = pa.table({
        "id": pa.array(["a"], type=pa.string()),
        "document_id": pa.array(["d"], type=pa.string()),
        "text": pa.array(["x"], type=pa.string()),
        "embedding_text": pa.array(["x"], type=pa.string()),
        "chunk_type": pa.array(["paragraph"], type=pa.string()),
        "section": pa.array([None], type=pa.null()),
        "subsection": pa.array([None], type=pa.null()),
        "breadcrumb": pa.array([["A"]], type=pa.list_(pa.string())),
        "parent_id": pa.array([None], type=pa.null()),
        "table_id": pa.array([None], type=pa.null()),
        "figure_id": pa.array([None], type=pa.null()),
        "document_position": pa.array([0], type=pa.int64()),
    })
    cast = _cast_to_corpus_schema(t1)
    assert cast.schema.equals(corpus_schema())
    assert cast["parent_id"].type == pa.string()
