"""Hybrid retrieval and score fusion.

Raw dense (cosine) and sparse (BM25) scores live on incompatible scales, so
they are never summed directly. Fusion strategies:

* ``rrf`` (default): Reciprocal Rank Fusion over ranks only.
      ``score(d) = sum_r 1 / (k + rank_r(d))``,  ``k=60``
  Robust to score scale and reproducible.

* ``minmax``: per-list min-max normalization to [0, 1], then weighted sum.
  Only meaningful when the candidate list has some score variance.

The hybrid retriever keeps the two base indexes separate and merges their
ranked chunk-id lists through the selected fusion function. Reranking is a
second stage handled one level up by the retrieval engine (``engine.py``),
which scores the full candidate pool produced here with a cross-encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval.dense import DenseIndex
from medrag.models import ScoredChunk
from medrag.retrieval.sparse import BM25Index

FUSION_RRF = "rrf"
FUSION_MINMAX = "minmax"
FUSION_METHODS = (FUSION_RRF, FUSION_MINMAX)

# How many candidates to pull from each base retriever before fusion.
DEFAULT_FUSION_DEPTH = 1000


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    k: float = 60.0,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    n = len(ranked_lists)
    if weights is None:
        weights = [1.0] * n
    for w, ranked in zip(weights, ranked_lists):
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w * (1.0 / (k + rank))
    return scores


def minmax_fusion(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    n = len(ranked_lists)
    if weights is None:
        weights = [1.0] * n

    for w, ranked in zip(weights, ranked_lists):
        if not ranked:
            continue
        vals = [s for _, s in ranked]
        lo, hi = min(vals), max(vals)
        for chunk_id, s in ranked:
            if hi == lo:
                norm = 1.0  # degenerate list: treat every hit as maximal
            else:
                norm = (s - lo) / (hi - lo)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w * norm
    return scores


def fuse(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    method: str = FUSION_RRF,
    k: float = 60.0,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Merge ranked (chunk_id, score) lists into one sorted list."""
    if method == FUSION_RRF:
        scores = reciprocal_rank_fusion(ranked_lists, k=k, weights=weights)
    elif method == FUSION_MINMAX:
        scores = minmax_fusion(ranked_lists, weights=weights)
    else:
        raise ValueError(f"unknown fusion method {method!r}; use one of {FUSION_METHODS}")
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """Combines dense + sparse ranking with a configurable fusion strategy.

    Produces a candidate pool; reranking is handled one level up by the
    retrieval engine so that the *full* candidate pool (not just the final
    top-k) can be scored by a second-stage model.
    """

    def __init__(
        self,
        dense: DenseIndex,
        sparse: BM25Index,
        fusion: str = FUSION_RRF,
        rrf_k: float = 60.0,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.weights = weights

    def candidates(
        self,
        query: str,
        top_k: int,
        query_vector: Optional[Any] = None,
        fusion_depth: int = DEFAULT_FUSION_DEPTH,
    ) -> List[ScoredChunk]:
        """Return the fused candidate pool (top ``top_k`` chunk ids)."""
        from medrag.trace import get_trace
        trace = get_trace()

        if query_vector is None:
            raise ValueError("hybrid retrieval requires a query_vector for the dense branch")

        dense_hits = self.dense.search_single(query_vector, fusion_depth)
        sparse_hits = self.sparse.search_single(query, fusion_depth)

        dense_ranked = [(h.chunk_id, h.score) for h in dense_hits]
        sparse_ranked = [(h.chunk_id, h.score) for h in sparse_hits]

        trace.log(
            "hybrid_prefusion",
            params={
                "query": query,
                "fusion_depth": fusion_depth,
                "fusion_method": self.fusion,
                "rrf_k": self.rrf_k,
                "dense_hits": len(dense_ranked),
                "sparse_hits": len(sparse_ranked),
                "dense_top5": [(cid, round(s, 4)) for cid, s in dense_ranked[:5]],
                "sparse_top5": [(cid, round(s, 4)) for cid, s in sparse_ranked[:5]],
            },
        )

        merged = fuse(
            [dense_ranked, sparse_ranked],
            method=self.fusion,
            k=self.rrf_k,
            weights=self.weights,
        )

        out = [
            ScoredChunk(
                chunk_id=chunk_id,
                score=score,
                retrieval_method="hybrid",
                rank=rank,
            )
            for rank, (chunk_id, score) in enumerate(merged[:top_k], start=1)
        ]

        trace.log(
            "hybrid_postfusion",
            params={"top_k": top_k, "n_merged": len(merged)},
            result={
                "n_output": len(out),
                "fusion_top10": [(sc.chunk_id, round(sc.score, 6)) for sc in out[:10]],
            },
        )
        return out

    def retrieve(
        self,
        query: str,
        top_k: int,
        query_vector: Optional[Any] = None,
        fusion_depth: int = DEFAULT_FUSION_DEPTH,
    ) -> List[ScoredChunk]:
        """Backward-compatible alias for :meth:`candidates`."""
        return self.candidates(
            query,
            top_k,
            query_vector=query_vector,
            fusion_depth=fusion_depth,
        )
