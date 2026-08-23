"""Retrieval engine: ties the indexes and reranker behind one query interface."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from medrag.retrieval.corpus import CORPUS_FILENAME, CorpusIndex
from medrag.retrieval.dense import DenseIndex
from medrag.retrieval.hybrid import FUSION_RRF, HybridRetriever
from medrag.models import Candidate, RetrievalResult, ScoredChunk
from medrag.retrieval.query import MedCPTQueryEncoder, QueryEncoder
from medrag.retrieval.reranker import Reranker
from medrag.retrieval.sparse import BM25Index

METHODS = ("dense", "bm25", "hybrid")
ALIASES = {"sparse": "bm25", "lexical": "bm25", "dense": "dense", "bm25": "bm25", "hybrid": "hybrid"}
_BM25_DIRNAME = "bm25"
DEFAULT_CANDIDATE_K = 100


class RetrievalEngine:
    """Loads the built indexes + optional reranker and answers queries.

    The model is loaded once and reused across queries, so a long-running
    service pays model initialization only once.

    Usage::

        engine = RetrievalEngine(Path("index"), reranker=CrossEncoderReranker())
        result = engine.search("diabetes treatment", method="hybrid", rerank=True, top_k=10)
    """

    def __init__(
        self,
        index_dir: Path,
        query_encoder: Optional[QueryEncoder] = None,
        fusion: str = FUSION_RRF,
        rrf_k: float = 60.0,
        weights: Optional[List[float]] = None,
        reranker: Optional[Reranker] = None,
        load_dense: bool = True,
    ) -> None:
        import os
        self.index_dir = Path(index_dir)
        self.corpus = CorpusIndex(self.index_dir / CORPUS_FILENAME, load=True)

        # Try pgvector first, fall back to FAISS
        self.dense = None
        if load_dense:
            use_pgvector = os.environ.get("MEDRAG_BACKEND", "pgvector").lower() == "pgvector"
            if use_pgvector:
                try:
                    from medrag.retrieval.dense_pgvector import PgDenseIndex
                    self.dense = PgDenseIndex.from_env()
                except Exception:
                    pass  # Fall through to FAISS
            if self.dense is None:
                try:
                    self.dense = DenseIndex.load(self.index_dir)
                except Exception:
                    pass

        self.sparse = BM25Index.load(self.index_dir / _BM25_DIRNAME)
        self.hybrid = (
            HybridRetriever(
                self.dense,
                self.sparse,
                fusion=fusion,
                rrf_k=rrf_k,
                weights=weights,
            )
            if load_dense
            else None
        )
        self.query_encoder = query_encoder if query_encoder is not None else MedCPTQueryEncoder()
        self.reranker = reranker
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.weights = weights

    def _normalize_method(self, method: str) -> str:
        key = method.lower()
        if key not in ALIASES:
            raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
        normalized = ALIASES[key]
        if normalized in ("dense", "hybrid") and self.dense is None:
            raise ValueError(
                f"method {method!r} requires the dense index, but the engine was "
                "loaded with load_dense=False"
            )
        return normalized

    def _generate_candidates(
        self,
        query: str,
        method: str,
        candidate_k: int,
        query_vector: Optional[object],
        fusion_depth: int,
    ) -> List[ScoredChunk]:
        if method == "dense":
            return self.dense.search_single(query_vector, candidate_k)
        if method == "bm25":
            return self.sparse.search_single(query, candidate_k)
        return self.hybrid.candidates(
            query,
            candidate_k,
            query_vector=query_vector,
            fusion_depth=fusion_depth,
        )

    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 20,
        include_text: bool = False,
        fusion_depth: int = 1000,
        rerank: bool = False,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        query_vector: Optional[object] = None,
    ) -> RetrievalResult:
        method = self._normalize_method(method)

        query_encode_ms: Optional[float] = None
        if method in ("dense", "hybrid") and query_vector is None:
            start = time.perf_counter()
            query_vector = self.query_encoder.encode([query])[0]
            query_encode_ms = (time.perf_counter() - start) * 1000.0

        # --------------------------------------------------------------
        # Second-stage path: candidate pool -> rerank -> top-k
        # --------------------------------------------------------------
        if rerank:
            if self.reranker is None:
                raise ValueError("reranking requested but no reranker was configured")

            start = time.perf_counter()
            candidates_hits = self._generate_candidates(
                query, method, candidate_k, query_vector, fusion_depth
            )
            self._resolve(candidates_hits, include_text=True)
            candidates = [Candidate.from_scored(h) for h in candidates_hits]
            candidate_gen_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            reranked = self.reranker.rerank(query, candidates, top_k)
            rerank_latency_ms = (time.perf_counter() - start) * 1000.0

            return RetrievalResult(
                query=query,
                method=f"{method}+rerank",
                results=reranked,
                latency_ms=candidate_gen_ms + rerank_latency_ms,
                query_encode_ms=query_encode_ms,
                reranked=True,
                rerank_latency_ms=rerank_latency_ms,
                candidate_count=len(candidates),
                candidates=candidates,
            )

        # --------------------------------------------------------------
        # First-stage only path (unchanged behaviour)
        # --------------------------------------------------------------
        start = time.perf_counter()
        if method == "dense":
            hits = self.dense.search_single(query_vector, top_k)
        elif method == "bm25":
            hits = self.sparse.search_single(query, top_k)
        else:
            hits = self.hybrid.retrieve(
                query,
                top_k,
                query_vector=query_vector,
                fusion_depth=fusion_depth,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        self._resolve(hits, include_text=include_text)

        return RetrievalResult(
            query=query,
            method=method,
            results=hits,
            latency_ms=latency_ms,
            query_encode_ms=query_encode_ms,
        )

    def _resolve(self, hits: List[ScoredChunk], include_text: bool) -> None:
        if not hits:
            return
        meta = self.corpus.resolve([h.chunk_id for h in hits], include_text=include_text)
        for hit in hits:
            record = meta.get(hit.chunk_id)
            if record is None:
                continue
            hit.document_id = record.get("document_id")
            hit.chunk_type = record.get("chunk_type")
            hit.breadcrumb = record.get("breadcrumb")
            if include_text:
                hit.text = record.get("text")

    def document_ids(self, hits: List[object]) -> List[Optional[str]]:
        return [
            getattr(h, "document_id", None)
            if getattr(h, "document_id", None) is not None
            else self.corpus.document_id(h.chunk_id)
            for h in hits
        ]
