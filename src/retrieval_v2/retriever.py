"""Per-query hybrid retrieval with full provenance (spec sections 10-12).

EVERY query is retrieved independently (BM25 top-100 + pgvector top-100) and
fused with RRF *within that query only*. Branches of different requirements
never compete at chunk level before paper-level evidence is preserved - this
is the core architectural rule that fixes the known failure where a correctly
retrieved branch paper disappears during global fusion (section 16).

All raw hits are retained as provenance events (section 11):
    {query_id, requirement_ids, paper_id, chunk_id, method, rank, score}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from src.retrieval_v2.models import (
    QueryLocalResult,
    RetrievalEvent,
    SearchQuery,
)


class NullTrace:
    """No-op trace sink (matches the TraceLogger.log interface)."""

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None

    def step(self, *args: Any, **kwargs: Any) -> Any:
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc: Any) -> None:
                return None
        return _Ctx()


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    k: float = 60.0,
    labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Fuse ranked (chunk_id, score) lists with reciprocal rank fusion.

    Returns per-chunk dicts: {chunk_id, scores: {label: score}, rrf_score}.
    RRF happens over ONE query only - callers must never feed lists from
    different queries into a single fusion.
    """
    if labels is None:
        labels = [f"s{i}" for i in range(len(ranked_lists))]
    best: Dict[str, Dict[str, float]] = {}
    n_lists = len(ranked_lists)
    for label, ranked in zip(labels, ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            # items may be (chunk_id, score) or (chunk_id, score, rank)
            chunk_id = item[0]
            if chunk_id not in best:
                best[chunk_id] = {lab: 0.0 for lab in labels}
            best[chunk_id][label] = 1.0 / (k + rank)
    out = []
    for chunk_id, scores in best.items():
        out.append({
            "chunk_id": chunk_id,
            "scores": scores,
            "rrf_score": sum(scores.values()),
        })
    out.sort(key=lambda d: d["rrf_score"], reverse=True)
    for rank, d in enumerate(out, start=1):
        d["rank"] = rank
    return out


class GlobalRetriever:
    """Runs per-query hybrid retrieval against the global corpus indexes.

    ``dense`` may be a FAISS DenseIndex or a PgDenseIndex; both expose
    ``search_single(qvec, top_k) -> List[ScoredChunk]`` (or ``search``).
    """

    def __init__(
        self,
        bm25: Any,
        dense: Any,
        doc_index: Any,
        query_encoder: Any,
        config: Optional[V2Config] = None,
        trace: Any = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.doc_index = doc_index
        self.query_encoder = query_encoder
        self.config = config or DEFAULT_CONFIG
        self.trace = trace or NullTrace()
        self._qvec_cache: Dict[str, Any] = {}
        self._bm25_cache: Dict[str, List[Tuple[str, float, int]]] = {}
        self._dense_cache: Dict[str, List[Tuple[str, float, int]]] = {}

    # ------------------------------------------------------------------
    # Caching (section 37)
    # ------------------------------------------------------------------
    def _encode(self, text: str) -> Any:
        if not self.config.cache_query_embeddings or text not in self._qvec_cache:
            vec = self.query_encoder.encode([text])[0]
            if self.config.cache_query_embeddings:
                self._qvec_cache[text] = vec
            return vec
        return self._qvec_cache[text]

    def _bm25_search(self, text: str, depth: int) -> List[Tuple[str, float, int]]:
        key = text
        if self.config.cache_retrieval and key in self._bm25_cache:
            cached = self._bm25_cache[key]
            return [h for h in cached if h[2] <= depth]
        hits = self.bm25.search_single(text, depth)
        out = [(h.chunk_id, float(h.score), h.rank) for h in hits]
        if self.config.cache_retrieval:
            self._bm25_cache[key] = out
        return out

    def _dense_search(self, text: str, qvec: Any, depth: int) -> List[Tuple[str, float, int]]:
        key = text
        if self.config.cache_retrieval and key in self._dense_cache:
            cached = self._dense_cache[key]
            return [h for h in cached if h[2] <= depth]
        if hasattr(self.dense, "search_single"):
            hits = self.dense.search_single(qvec, depth)
        elif hasattr(self.dense, "search"):
            hits = self.dense.search(qvec, depth)[0]
        else:
            raise TypeError("dense index must provide search_single or search")
        out = [(h.chunk_id, float(h.score), h.rank) for h in hits]
        if self.config.cache_retrieval:
            self._dense_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, queries: Sequence[SearchQuery]) -> List[QueryLocalResult]:
        """Retrieve every query independently; returns one QueryLocalResult each."""
        cfg = self.config
        results: List[QueryLocalResult] = []
        for query in queries:
            qid = query.id
            text = query.text or ""
            self.trace.log(
                f"{qid}_retrieval_start",
                params={"query_id": qid, "requirement_ids": query.requirement_ids,
                        "text": text, "bm25_depth": cfg.bm25_depth, "dense_depth": cfg.dense_depth},
            )

            events: List[RetrievalEvent] = []

            # BM25 branch
            bm25_hits = self._bm25_search(text, cfg.bm25_depth)
            for chunk_id, score, rank in bm25_hits:
                events.append(
                    RetrievalEvent(
                        query_id=qid,
                        requirement_ids=list(query.requirement_ids),
                        paper_id=self.doc_index.paper_of(chunk_id),
                        chunk_id=chunk_id,
                        method="bm25",
                        rank=rank,
                        score=score,
                    )
                )

            # Dense branch
            dense_hits: List[Tuple[str, float, int]] = []
            if self.dense is not None and self.config.enable_dense:
                try:
                    qvec = self._encode(text)
                    dense_hits = self._dense_search(text, qvec, cfg.dense_depth)
                    for chunk_id, score, rank in dense_hits:
                        events.append(
                            RetrievalEvent(
                                query_id=qid,
                                requirement_ids=list(query.requirement_ids),
                                paper_id=self.doc_index.paper_of(chunk_id),
                                chunk_id=chunk_id,
                                method="dense",
                                rank=rank,
                                score=score,
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    self.trace.log(f"{qid}_dense_warning", detail=str(exc), error=str(exc))

            # ── Query-local RRF (section 12) ──────────────────────────
            fused = rrf_fuse(
                [bm25_hits, dense_hits],
                k=cfg.rrf_k,
                labels=["bm25", "dense"],
            )
            # attach provenance to fused entries
            for entry in fused:
                entry["paper_id"] = self.doc_index.paper_of(entry["chunk_id"])
                entry["methods"] = [lab for lab in ("bm25", "dense") if entry["scores"].get(lab, 0.0) > 0]
                entry["requirement_ids"] = list(query.requirement_ids)
                entry["query_id"] = qid

            res = QueryLocalResult(query=query, events=events, fused=fused)
            results.append(res)
            self.trace.log(
                f"{qid}_retrieval_done",
                params={"query_id": qid},
                result={
                    "n_events": len(events),
                    "bm25_events": sum(1 for e in events if e.method == "bm25"),
                    "dense_events": sum(1 for e in events if e.method == "dense"),
                    "n_fused": len(fused),
                    "n_papers": len(set(e.paper_id for e in events if e.paper_id)),
                    "top10": [{"chunk_id": f["chunk_id"], "paper_id": f["paper_id"],
                               "rrf_score": round(f["rrf_score"], 6), "rank": f["rank"],
                               "methods": f["methods"]} for f in fused[:10]],
                },
            )
        return results

    def hit_papers_by_requirement(
        self, query_results: Sequence[QueryLocalResult]
    ) -> Dict[str, List[str]]:
        """Map requirement id -> ordered list of paper ids (by fused presence).

        Used by the paper-selection shortlist stage.
        """
        out: Dict[str, List[str]] = {}
        for res in query_results:
            for req_id in res.query.requirement_ids:
                seen: List[str] = out.setdefault(req_id, [])
                for f in res.fused:
                    pid = f.get("paper_id")
                    if pid and pid not in seen:
                        seen.append(pid)
        return out

