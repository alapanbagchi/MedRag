"""postgres_search - the PRIMARY pg-chunk retrieval tool (medrag.chunks).

This is the agent-facing search over the PostgreSQL + pgvector corpus - the
primary evidence source of this deployment. A single call runs the pg-native
hybrid:

    sparse  -> PostgreSQL full-text search over medrag.chunks.tsv (PgFtsSparse,
               BM25-like idf weighting from GIN counts)
    dense   -> pgvector cosine over medrag.embeddings (MedCPT Query Encoder +
               PgDenseIndex, optional - degrades to sparse-only)
    union + dedupe -> ONE intent reranker -> paper diversification -> top K
                    (exactly the reranker the parquet hybrid uses)

Health contract (so the agent can tell "no relevant doc" from "empty DB"):
    * available:false  -> postgres unreachable; the agent must fall back to
                          retrieve (parquet hybrid).
    * empty_corpus:true -> the medrag schema exists but has 0 chunks; the
                          corpus is NOT the gap, the DB is. The agent should
                          say so / switch sources rather than looping.

Every result carries full unit text restored from medrag.chunks (never a bare
snippet). Results are CANDIDATES - they still must pass the verifier before
they become evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from langchain_core.tools import tool

from src.retrieval.pgvector_store import PgConfig, PgVectorStore

logger = logging.getLogger("x_deepagents.postgres_search")

# How deep each leg may over-fetch before the union + rerank + diversify.
_LEG_DEPTH = 60
_max_per_paper = 2
_score_cap = 60

_pg_store: Optional[PgVectorStore] = None
_query_encoder = None


def reset_pg_store() -> None:
    """Drop cached store/encoder singletons (tests)."""
    global _pg_store, _query_encoder
    if _pg_store is not None:
        try:
            _pg_store.close()
        except Exception:
            pass
    _pg_store = None
    _query_encoder = None


def _store() -> PgVectorStore:
    global _pg_store
    if _pg_store is None:
        _pg_store = PgVectorStore(PgConfig.from_env())
    return _pg_store


def _dense_encoder():
    """Lazily built MedCPT query encoder (module cache). None on failure."""
    global _query_encoder
    if _query_encoder is None:
        try:
            from src.retrieval.query import MedCPTQueryEncoder
            from src.config import config as appconfig

            _query_encoder = MedCPTQueryEncoder(appconfig().embedding_model)
        except Exception as exc:
            logger.warning("dense query encoder unavailable: %s", exc)
            _query_encoder = False
    return _query_encoder or None


def _legacy_subquery(sub_query_plan: Any) -> Any:
    """Adapt SubQueryPlan to the LegacySubQuery the reranker expects."""
    from src.retrieval.plans import SubQuery as LegacySubQuery

    return LegacySubQuery(
        id=sub_query_plan.id,
        target=sub_query_plan.target or sub_query_plan.query,
        focus=sub_query_plan.focus or "evidence",
        query=sub_query_plan.query or sub_query_plan.target,
        evidence_required=list(sub_query_plan.evidence_required or []),
        terminology=[e.text for e in sub_query_plan.entities],
    )


async def _connect() -> PgVectorStore | None:
    """Connect with a short timeout; returns None (and logs) when unreachable."""
    try:
        store = _store()
        await asyncio.wait_for(asyncio.to_thread(store.connect), timeout=4.0)
        return store
    except Exception as exc:
        logger.warning("postgres unreachable: %s", exc)
        return None


def _corpus_stats(store: PgVectorStore) -> dict:
    try:
        return {
            "chunks": store.count_chunks(),
            "embeddings": store.count_embeddings(),
            "documents": store.count_documents(),
        }
    except Exception as exc:
        logger.warning("corpus stats failed: %s", exc)
        return {"chunks": -1, "embeddings": -1, "documents": -1}


def _sparse_leg(store: PgVectorStore, query: str, depth: int) -> list[tuple[str, float]]:
    """(chunk_id, score) from postgres FTS over medrag.chunks.tsv."""
    from src.retrieval.retriever import PgFtsSparse

    try:
        hits = PgFtsSparse(store).search_single(query, depth)
        return [(h.chunk_id, float(h.score or 0.0)) for h in hits]
    except Exception as exc:
        logger.warning("pgfts leg failed (column tsv missing?): %s", exc)
        return []


def _dense_leg(store: PgVectorStore, query: str, depth: int) -> list[tuple[str, float]]:
    """(chunk_id, score) from pgvector cosine (needs the MedCPT encoder)."""
    encoder = _dense_encoder()
    if encoder is None:
        return []
    try:
        from src.retrieval.dense_pgvector import PgDenseIndex

        qvec = encoder.encode_single(query)
        hits = PgDenseIndex(store).search_single(qvec, depth)
        return [(h.chunk_id, float(h.score or 0.0)) for h in hits]
    except Exception as exc:
        logger.warning("pgvector dense leg failed: %s", exc)
        return []


def _materialize(
    store: PgVectorStore,
    sparse_ranks: list[tuple[str, float]],
    dense_ranks: list[tuple[str, float]],
    exclude: set[str],
) -> list[Any]:
    """Union both legs' chunk ids, fetch full metadata from medrag.chunks,
    and build duck-typed docs for the shared reranker."""
    from src.retrieval.retriever import RetrievedDocument

    ids: list[str] = []
    seen: set = set()
    provenance: dict[str, list[str]] = {}
    scores: dict[str, float] = {}
    for cid, score in sparse_ranks:
        if cid in exclude or cid in seen:
            continue
        seen.add(cid)
        ids.append(cid)
        provenance[cid] = ["pgfts"]
        scores[cid] = score
    for cid, score in dense_ranks:
        if cid in exclude:
            continue
        if cid in seen:
            if "pgvector" not in provenance[cid]:
                provenance[cid].append("pgvector")
            scores[cid] = max(scores.get(cid, 0.0), score)
            continue
        seen.add(cid)
        ids.append(cid)
        provenance[cid] = ["pgvector"]
        scores[cid] = score

    if not ids:
        return []

    rows = store.get_chunks(ids)  # {chunk_id: metadata incl. full text}
    docs: list[Any] = []
    for rank, cid in enumerate(ids, start=1):
        meta = rows.get(cid) or {}
        chunk_type = meta.get("chunk_type") or "paragraph"
        docs.append(RetrievedDocument(
            subquery_id="PG",
            document_id=meta.get("document_id") or "",
            chunk_id=cid,
            rank=rank,
            rrf_score=float(scores.get(cid, 0.0)),
            methods=provenance.get(cid, ["pgfts"]),
            variant_ids=[cid],
            node_type=chunk_type,
            section=meta.get("section") or "",
            subsection=meta.get("subsection") or "",
            breadcrumb=meta.get("breadcrumb", []),
            table_id=meta.get("table_id"),
            figure_id=meta.get("figure_id"),
            text=meta.get("text") or "",
            token_count=len((meta.get("text") or "").split()),
        ))
    return docs


def _rerank(sub: Any, docs: list[Any], top_k: int) -> list[Any]:
    """Shared intent reranker + paper diversification (same as the hybrid)."""
    from src.retrieval.reranker import union_rerank_diversify

    if not docs:
        return []
    return union_rerank_diversify(
        sub, docs, [],
        top_k=max(1, top_k),
        max_per_paper=_max_per_paper,
        score_cap=_score_cap,
    )


async def postgres_search_impl(
    requirement_id: str,
    query: str,
    entities: Optional[list[str]] = None,
    top_k: int = 5,
    exclude_chunk_ids: Optional[list[str]] = None,
) -> str:
    """Core implementation: pg-native hybrid search (see the @tool wrapper)."""
    from src.x_deepagents.logging import xdeep_log
    from src.retrieval.plans import SubQueryPlan

    exclude = set(chunk_id for chunk_id in (exclude_chunk_ids or []) if chunk_id)
    xdeep_log("postgres_search_started", requirement_id=requirement_id,
              query=query, top_k=top_k, exclude=len(exclude))

    store = await _connect()
    if store is None:
        return json.dumps({
            "requirement_id": requirement_id,
            "query": query,
            "available": False,
            "results": [],
            "note": "postgres unreachable - use retrieve (parquet hybrid) instead",
        }, ensure_ascii=False)

    stats = _corpus_stats(store)
    if stats.get("chunks", 0) <= 0:
        xdeep_log("postgres_search_empty_corpus", requirement_id=requirement_id,
                  query=query, stats=stats)
        return json.dumps({
            "requirement_id": requirement_id,
            "query": query,
            "available": True,
            "empty_corpus": True,
            "stats": stats,
            "results": [],
            "note": "the Postgres corpus (medrag.chunks) is EMPTY - the local "
                    "database has no chunks loaded; consider web search or "
                    "reporting the corpus gap",
        }, ensure_ascii=False)

    sub = SubQueryPlan(
        id=requirement_id,
        target=query,
        intent="evidence",
        query=query,
        focus="evidence",
        evidence_required=[query],
        entities=[__import__("src.retrieval.plans", fromlist=["PlannedEntity"]).PlannedEntity(text=e)
                  for e in (entities or [])],
    )
    legacy = _legacy_subquery(sub)

    depth = max(_LEG_DEPTH, top_k * 4)
    sparse_ranks = _sparse_leg(store, query, depth)
    dense_ranks = _dense_leg(store, query, depth)
    docs = _materialize(store, sparse_ranks, dense_ranks, exclude)
    ranked = _rerank(legacy, docs, top_k=top_k)

    out = []
    for i, doc in enumerate(ranked, 1):
        out.append({
            "rank": i,
            "chunk_id": getattr(doc, "chunk_id", ""),
            "document_id": getattr(doc, "document_id", ""),
            "section": getattr(doc, "section", ""),
            "unit_kind": getattr(doc, "node_type", "paragraph"),
            "score": round(float(getattr(doc, "rrf_score", 0.0) or 0.0), 4),
            "methods": [m for m in (getattr(doc, "methods", None) or [])],
            "text": getattr(doc, "text", ""),
        })
    xdeep_log("postgres_search_done", requirement_id=requirement_id,
              query=query, count=len(out), stats=stats,
              legs={"sparse": len(sparse_ranks), "dense": len(dense_ranks)})
    return json.dumps({
        "requirement_id": requirement_id,
        "query": query,
        "available": True,
        "empty_corpus": False,
        "stats": stats,
        "count": len(out),
        "results": out,
    }, ensure_ascii=False)


@tool
async def postgres_search(
    requirement_id: str,
    query: str,
    entities: Optional[list[str]] = None,
    top_k: int = 5,
    exclude_chunk_ids: Optional[list[str]] = None,
) -> str:
    """Primary literature search over the LOCAL Postgres corpus (medrag.chunks).

    Hybrid pg-native retrieval (full-text + pgvector), ranked by the shared
    intent reranker with paper diversification. Returns each hit's FULL
    containing unit (whole paragraph/table/figure) as JSON with rank,
    chunk_id, document_id, section, unit_kind, score, methods and text.

    Uses the local Postgres database as the PRIMARY evidence source - prefer
    this over retrieve whenever the Postgres corpus is available. Results are
    CANDIDATES: every returned chunk must still pass the verifier.

    If postgres is unreachable the result reports "available": false so you
    should fall back to the retrieve tool. If the corpus is empty it reports
    "empty_corpus": true - that is a corpus problem, not a query problem.
    """
    return await postgres_search_impl(
        requirement_id=requirement_id,
        query=query,
        entities=entities,
        top_k=top_k,
        exclude_chunk_ids=exclude_chunk_ids,
    )


__all__ = ["postgres_search", "postgres_search_impl", "reset_pg_store"]
