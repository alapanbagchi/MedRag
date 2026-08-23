"""Unit tests for the reranker (mocked models; no large downloads)."""

from __future__ import annotations

import numpy as np

from medrag.retrieval.cli import build_parser
from medrag.models import Candidate, RerankedCandidate
from medrag.retrieval.reranker import CrossEncoderReranker, build_passage


# ----------------------------------------------------------------------
# Passage construction
# ----------------------------------------------------------------------

def test_build_passage_plain_and_with_breadcrumb():
    c = Candidate(chunk_id="c1", breadcrumb="Methods > Outcomes", text="  Some text  ")
    assert build_passage(c, include_breadcrumb=False) == "Some text"
    assert build_passage(c, include_breadcrumb=True) == "Section: Methods > Outcomes\nPassage: Some text"


def test_build_passage_missing_breadcrumb_uses_text_only():
    c = Candidate(chunk_id="c1", text="hello")
    assert build_passage(c, include_breadcrumb=True) == "hello"


# ----------------------------------------------------------------------
# Dummy reranker for logic tests (no torch/transformers)
# ----------------------------------------------------------------------

class DummyReranker(CrossEncoderReranker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_count = 0

    def _load(self):
        if self._model is not None:
            return
        self.load_count += 1
        self._model = object()
        self._tokenizer = object()
        self._resolved_device = "cpu"

    def _score_pairs(self, pairs):
        # Deterministic fake score: passage length.
        return np.array([float(len(p)) for _, p in pairs], dtype=np.float32)


def _candidate(cid, length, orig_rank=0, orig_score=0.0, breadcrumb=None):
    return Candidate(
        chunk_id=cid,
        document_id=f"doc-{cid[0]}",
        original_score=orig_score,
        original_rank=orig_rank,
        chunk_type="paragraph",
        breadcrumb=breadcrumb,
        text="x" * length,
    )


def test_lazy_loading_and_model_loaded_once():
    r = DummyReranker()
    assert r.loaded is False
    r.rerank("q", [_candidate("a", 5), _candidate("b", 3)], top_k=2)
    assert r.loaded is True
    assert r.load_count == 1
    r.rerank("q", [_candidate("a", 5)], top_k=1)
    assert r.load_count == 1


def test_rerank_orders_by_score_desc():
    r = DummyReranker()
    out = r.rerank("q", [_candidate("a", 1), _candidate("b", 10), _candidate("c", 5)], top_k=3)
    assert [o.chunk_id for o in out] == ["b", "c", "a"]
    assert [o.reranked_rank for o in out] == [1, 2, 3]
    assert all(o.reranker_score > 0 for o in out)


def test_rerank_preserves_original_metadata():
    c = _candidate("a", 5, orig_rank=7, orig_score=0.123, breadcrumb="S1 > S2")
    r = DummyReranker()
    out = r.rerank("q", [c], top_k=1)
    assert len(out) == 1
    o = out[0]
    assert o.chunk_id == "a"
    assert o.document_id == "doc-a"
    assert o.original_rank == 7
    assert o.original_score == 0.123
    assert o.breadcrumb == "S1 > S2"
    assert o.text == "x" * 5
    assert o.reranked_rank == 1
    assert isinstance(o, RerankedCandidate)


def test_rerank_empty_candidates():
    r = DummyReranker()
    assert r.rerank("q", [], top_k=10) == []


def test_rerank_top_k_greater_than_candidate_count():
    r = DummyReranker()
    out = r.rerank("q", [_candidate("a", 1), _candidate("b", 2)], top_k=100)
    assert len(out) == 2
    assert [o.reranked_rank for o in out] == [1, 2]


def test_rerank_top_k_zero_or_negative():
    r = DummyReranker()
    assert r.rerank("q", [_candidate("a", 1)], top_k=0) == []


def test_device_resolution_cpu_fallback(monkeypatch):
    r = CrossEncoderReranker()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert r._resolve_device() == "cpu"
    r2 = CrossEncoderReranker(device="cpu")
    assert r2._resolve_device() == "cpu"


def test_device_resolution_explicit():
    r = CrossEncoderReranker(device="cuda")
    assert r._resolve_device() == "cuda"


# ----------------------------------------------------------------------
# Batched scoring with a fake tokenizer/model (real torch, CPU)
# ----------------------------------------------------------------------

def test_score_pairs_batches_and_returns_scores(monkeypatch):
    import torch

    class FakeTokenizer:
        def __call__(self, queries, passages, **kwargs):
            # deterministic input_ids: length equals passage length
            batch = [
                torch.ones(max(1, len(p)), dtype=torch.long) for p in passages
            ]
            input_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
            attention_mask = (input_ids != 0).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeModel:
        def __call__(self, input_ids, attention_mask, **kwargs):
            scores = attention_mask.float().sum(dim=1)  # = passage length
            return type("Out", (), {"logits": scores.unsqueeze(-1)})()

    r = CrossEncoderReranker(device="cpu", batch_size=2)
    r._tokenizer = FakeTokenizer()
    r._model = FakeModel()
    r._resolved_device = "cpu"

    pairs = [("q", "a"), ("q", "bb"), ("q", "ccc"), ("q", "dddd"), ("q", "eeeee")]
    scores = r._score_pairs(pairs)
    assert scores.shape == (5,)
    assert list(scores) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_score_pairs_empty():
    r = CrossEncoderReranker(device="cpu")
    r._tokenizer = object()
    r._model = object()
    r._resolved_device = "cpu"
    assert r._score_pairs([]).shape == (0,)


# ----------------------------------------------------------------------
# CLI integration
# ----------------------------------------------------------------------

def test_cli_search_rerank_flags():
    args = build_parser().parse_args(
        ["search", "--query", "x", "--method", "hybrid", "--top-k", "10",
         "--candidate-k", "50", "--rerank", "--reranker-batch-size", "8"]
    )
    assert args.rerank is True
    assert args.candidate_k == 50
    assert args.top_k == 10
    assert args.reranker_batch_size == 8
    assert args.reranker_model == "ncbi/MedCPT-Cross-Encoder"


def test_cli_search_no_rerank_default():
    args = build_parser().parse_args(["search", "--query", "x"])
    assert args.rerank is False


def test_cli_search_no_rerank_flag():
    args = build_parser().parse_args(["search", "--query", "x", "--no-rerank"])
    assert args.rerank is False


def test_cli_backward_compatible_methods():
    for method in ("dense", "bm25", "sparse", "hybrid"):
        args = build_parser().parse_args(["search", "--query", "x", "--method", method])
        assert args.method == method


def test_cli_compare_parses():
    args = build_parser().parse_args(["compare", "--query", "a", "b", "--top-k", "5"])
    assert args.query == ["a", "b"]
    assert args.top_k == 5


def test_reranked_candidate_score_rank_shims():
    rc = RerankedCandidate(
        chunk_id="c", document_id="d", original_score=0.5, reranker_score=0.9,
        original_rank=3, reranked_rank=1,
    )
    assert rc.score == 0.9
    assert rc.rank == 1
