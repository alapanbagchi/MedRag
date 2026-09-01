"""Retrieval service."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.retrieval.plans import QueryPlan, SubQuery
from src.retrieval.reranker import union_rerank_diversify
from src.lib.trace import get_trace

logger = logging.getLogger("src.retriever")


def _norm_breadcrumb(value: Any) -> List[str]:
    """Normalize breadcrumb metadata (string, list, tuple, ndarray, or repr)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip() and str(x).strip() != "nan"]
    text = str(value).strip()
    if not text:
        return []
    # Joined form: "Abstract > Background"
    if ">" in text:
        return [p.strip() for p in text.split(">") if p.strip()]
    # Quoted repr form: "['Abstract' 'Background']"
    if text.startswith("[") and text.endswith("]"):
        import re as _re
        inner = text[1:-1]
        quoted = _re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
        entries = [a or b for a, b in quoted if (a or b).strip()]
        if entries:
            return entries
    return [text]


class RetrievedDocument:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PgFtsSparse:
    """Sparse branch served by Postgres full-text search — no local BM25.

    Used when neither ``index/bm25_v2`` nor ``index/bm25`` exists (the whole
    index/ dir may be gone). Mirrors scripts/pg_hybrid_query.fts_ranked:
    per-query-term rarity weights (idf from GIN counts) merged over per-term
    ``ts_rank_cd`` pools — a BM25-like rank with no index files or big RAM.
    Requires the ``medrag.chunks.tsv`` column maintained by pg_load_v2.py
    (and the pgvvector connection from the environment).

    Implements the same ``search_single`` interface as BM25Index so the
    retriever/reranker flow is unchanged.
    """

    def __init__(self, store: Any = None) -> None:
        self._store = store
        self.meta: Dict[str, Any] = {"backend": "pgfts", "n_docs": 0}

    def _store_obj(self) -> Any:
        if self._store is None:
            from src.retrieval.pgvector_store import PgConfig, PgVectorStore
            self._store = PgVectorStore(PgConfig.from_env())
            self._store.connect()
        return self._store

    def search_single(self, query: str, top_k: int) -> List[Any]:
        import math
        import re as _re

        from src.lib.models import ScoredChunk

        store = self._store_obj()
        conn = store.connect()
        cur = conn.cursor()
        tokens = list(dict.fromkeys(_re.findall(r"[a-z0-9]+", (query or "").lower())))[:12]
        if not tokens:
            cur.close()
            return []
        cur.execute(
            "SELECT count(*) FROM medrag.chunks "
            "WHERE tsv IS NOT NULL AND tsv <> ''::tsvector"
        )
        n_docs = max(1, cur.fetchone()[0])
        self.meta["n_docs"] = n_docs

        idf: Dict[str, float] = {}
        for t in tokens:
            cur.execute(
                "SELECT count(*) FROM medrag.chunks "
                "WHERE tsv @@ to_tsquery('simple', %s)", (t,)
            )
            df = cur.fetchone()[0]
            if df > n_docs * 0.55:      # ~stopword: no signal
                continue
            idf[t] = math.log(1.0 + n_docs / (df + 1.0))

        acc: Dict[str, float] = {}
        for t, w in idf.items():
            cur.execute(
                "SELECT id, ts_rank_cd(tsv, to_tsquery('simple', %s), 1) AS r "
                "FROM medrag.chunks WHERE tsv @@ to_tsquery('simple', %s) "
                "ORDER BY r DESC LIMIT %s",
                (t, t, top_k),
            )
            for cid, r in cur.fetchall():
                acc[cid] = acc.get(cid, 0.0) + w * float(r)
        cur.close()

        ranked = sorted(acc.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            ScoredChunk(chunk_id=cid, score=score, retrieval_method="fts", rank=i + 1)
            for i, (cid, score) in enumerate(ranked)
        ]


class RetrievalService:
    def __init__(self, config: Optional[AppConfig] = None):
        self.cfg = config or AppConfig()
        self._bm25 = None
        self._splade = None
        self._dense = None
        self._query_encoder = None
        self._corpus = None
        self._load_error: Optional[str] = None
        # Cache keyed by QUERY TEXT (not subquery id — ids like "H1" collide
        # across subqueries/tools), including the exclusion signature.
        self._cache: Dict[str, List[RetrievedDocument]] = {}

    @staticmethod
    def _cache_key(sub: SubQuery, exclude_chunk_ids: Optional[List[str]]) -> str:
        query = (sub.query or sub.target or "").strip().lower()
        if not exclude_chunk_ids:
            return f"q::{query}"
        return "q::" + query + "::ex::" + ",".join(sorted(exclude_chunk_ids))

    def _components(self) -> Dict[str, Any]:
        if self._load_error:
            raise RuntimeError(self._load_error)
        if self._bm25 is None or self._corpus is None:
            self._load_components()
        return {"bm25": self._bm25, "splade": self._splade, "dense": self._dense, "corpus": self._corpus}

    def _ensure_corpus_shim(self) -> Path:
        """Create a tiny empty corpus.parquet when index/ is missing.

        CorpusIndex / StructuralUnitIndex need a loadable (even empty)
        DataFrame; all real chunk metadata is resolved from pgvector
        (medrag.chunks) via _resolve_rich's fallback.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.cfg.index_dir / "corpus_pgvector_shim.parquet"
        if path.is_file():
            return path
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("text", pa.string()),
            pa.field("breadcrumb", pa.list_(pa.string())),
        ])
        pq.write_table(schema.empty_table(), path, compression="zstd")
        logger.info("created empty corpus shim %s (metadata from pgvector)", path)
        return path

    def _load_components(self) -> None:
        index_dir = self.cfg.index_dir
        try:
            from src.retrieval.corpus import CorpusIndex

            corpus_path = self.cfg.corpus_path
            if not corpus_path.is_file():
                corpus_path = self._ensure_corpus_shim()
                self.cfg.corpus_path = corpus_path
            corpus = self._corpus or CorpusIndex(corpus_path)

            # Sparse leg: prefer the v2 BM25 index, then v1 index/bm25, then
            # Postgres FTS (tsv) so the retriever works with no index files.
            bm25 = self._bm25
            if bm25 is None:
                bm25_v2 = index_dir / "bm25_v2"
                bm25_v1 = index_dir / "bm25"
                if (bm25_v2 / "bm25_meta.json").is_file():
                    from src.retrieval.sparse import BM25Index
                    bm25 = BM25Index.load(bm25_v2)
                    logger.info("sparse index: BM25 v2 (%s)", bm25_v2)
                elif (bm25_v1 / "bm25_meta.json").is_file():
                    from src.retrieval.sparse import BM25Index
                    bm25 = BM25Index.load(bm25_v1)
                    logger.info("sparse index: BM25 v1 (%s)", bm25_v1)
                else:
                    bm25 = PgFtsSparse()
                    logger.info("sparse index: Postgres FTS (tsv, no local BM25)")
            splade = self._splade
            if self.cfg.retrieval_primary == "splade":
                from src.retrieval.splade import SPLADEIndex
                splade = self._splade or SPLADEIndex.load(index_dir / "splade")
            dense = self._dense
            query_encoder = None
            if self.cfg.enable_dense and dense is None:
                try:
                    from src.retrieval.query import MedCPTQueryEncoder
                    query_encoder = MedCPTQueryEncoder(self.cfg.embedding_model)
                    # Local mode (or no pgvector): use FAISS directly.
                    if self.cfg.local_mode or not self.cfg.vector_db_url:
                        from src.retrieval.dense import DenseIndex
                        dense = DenseIndex.load(index_dir)
                        logger.info("dense index: FAISS (local) (dim=%d, n=%d)", dense.dimension, dense.n_total)
                    elif self.cfg.vector_db_url:
                        try:
                            from src.retrieval.dense_pgvector import PgDenseIndex
                            from src.retrieval.pgvector_store import PgConfig, PgVectorStore
                            probe = PgVectorStore(PgConfig.from_env())
                            probe.connect()
                            probe.close()
                            dense = PgDenseIndex.from_env()
                            logger.info("dense index: pgvector (dim=%d)", dense.dimension)
                        except Exception as exc:
                            logger.warning("pgvector unavailable (%s); using FAISS index", exc)
                            from src.retrieval.dense import DenseIndex
                            dense = DenseIndex.load(index_dir)
                            logger.info("dense index: FAISS (dim=%d, n=%d)", dense.dimension, dense.n_total)
                except Exception as exc:
                    logger.warning("dense retrieval unavailable: %s", exc)
                    dense = None
                    query_encoder = None

            self._corpus = corpus
            self._bm25 = bm25
            self._splade = splade
            self._dense = dense
            self._query_encoder = query_encoder
        except Exception as exc:
            self._load_error = f"retrieval components failed to load: {exc}"
            raise RuntimeError(self._load_error) from exc

    async def search_subquery(
        self,
        sub: SubQuery,
        plan: Optional[QueryPlan] = None,
        exclude_chunk_ids: Optional[List[str]] = None,
    ) -> List[RetrievedDocument]:
        cache_key = self._cache_key(sub, exclude_chunk_ids)
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        documents = await asyncio.to_thread(self._search_subquery_sync, sub, exclude_chunk_ids or [])
        self._cache[cache_key] = documents
        return list(documents)

    async def search_plan(self, plan: QueryPlan) -> Dict[str, List[RetrievedDocument]]:
        results = await asyncio.gather(
            *(self.search_subquery(sub, plan) for sub in plan.subqueries)
        )
        return {sub.id: docs for sub, docs in zip(plan.subqueries, results)}

    def _resolve_rich(self, corpus: Any, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Resolve chunk metadata+text, including section/table/figure columns.

        CorpusIndex.resolve only returns document_id/chunk_type/breadcrumb/text;
        ids it does not know (v2 chunks live in pgvector, not the stale
        index/corpus.parquet) are resolved from medrag.chunks when a pgvector
        store is configured; the result is extended with the extra columns
        from the loaded DataFrame.
        """
        resolved = corpus.resolve(chunk_ids, include_text=True)
        missing = [cid for cid in chunk_ids if cid not in resolved]
        if missing and self.cfg.vector_db_url and not self.cfg.local_mode:
            try:
                from src.retrieval.pgvector_store import PgConfig, PgVectorStore
                store = PgVectorStore(PgConfig.from_env())
                store.connect()
                rows = store.get_chunks(missing)
                for cid, rec in rows.items():
                    resolved[cid] = rec
                store.close()
            except Exception:
                logger.warning("pgvector metadata fallback unavailable",
                               exc_info=True)
        df = getattr(corpus, "_df", None)
        if df is None:
            return resolved
        try:
            subset = df.loc[df["id"].isin(list(chunk_ids))]
            for row in subset.itertuples(index=False):
                record = resolved.get(row.id)
                if record is None:
                    continue
                for col in ("section", "subsection", "table_id", "figure_id"):
                    val = getattr(row, col, None)
                    if val is not None and not (isinstance(val, float) and val != val):
                        record[col] = val
                # Normalize breadcrumb to a list of parts.
                # Breadcumb_str can be a joined string ("A > B"), a numpy-style
                # repr ("['A' 'B']"), or a real list/array.
                record["breadcrumb"] = _norm_breadcrumb(record.get("breadcrumb"))
        except Exception:
            pass
        return resolved

    @staticmethod
    def _resolve_texts(corpus: Any, chunk_ids: List[str]) -> Dict[str, str]:
        """Resolve {chunk_id: full text} for logging (untruncated)."""
        if not chunk_ids:
            return {}
        try:
            resolved = corpus.resolve(list(dict.fromkeys(chunk_ids)), include_text=True)
            return {cid: (meta.get("text") or "") for cid, meta in resolved.items()}
        except Exception:
            return {}

    def _search_subquery_sync(
        self, sub: SubQuery, exclude_chunk_ids: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """The requested flow (primary = BM25 or SPLADE, dense stays same):

            primary top 30  --+
                              +--> UNION + dedupe --> ONE intent reranker -->
            Dense top 30    -+     paper diversification--> top K

        ``exclude_chunk_ids`` drops already-seen chunks BEFORE the union so a
        re-search round can surface fresh candidates from deeper ranks.
        """
        components = self._components()
        bm25 = components["bm25"]
        splade = components["splade"]
        dense = components["dense"]
        corpus = components["corpus"]

        trace = get_trace()
        exclude = set(exclude_chunk_ids or [])
        # Over-fetch at the configured branch depths so filtering and the
        # union never starve; double only when excluding seen chunks.
        base_depth = max(30, self.cfg.bm25_depth, self.cfg.dense_depth)
        TOP = base_depth * (2 if exclude else 1)

        # Pick which primary sparse retriever to use (bm25 or splade).
        primary_name = self.cfg.retrieval_primary if hasattr(self.cfg, "retrieval_primary") else "bm25"
        primary_index = splade if primary_name == "splade" else bm25
        if primary_name == "splade" and primary_index is None:
            trace.bullet("SPLADE index not loaded; falling back to BM25")
            primary_name = "bm25"
            primary_index = bm25

        # ---- 1. PRIMARY (bm25 or splade) top 30 (raw) --------------------
        query_text = sub.query if sub.query else sub.target
        raw_primary = [(h.chunk_id, float(h.score)) for h in primary_index.search_single(query_text, TOP)]
        trace.retrieved(primary_name, query_text, raw_primary[:30],
                        texts=self._resolve_texts(corpus, [c for c, _ in raw_primary[:30]]))
        primary_docs = self._materialize(corpus, sub, raw_primary, method=primary_name)
        trace.bullet(f"{primary_name} top {TOP}: {len(primary_docs)} docs")

        # ---- 2. Dense top 30 (raw) ---------------------------------------
        dense_docs = []
        if dense is not None and self.cfg.enable_dense:
            try:
                qvec = self._query_encoder.encode_single(query_text)
                raw_dense = [(h.chunk_id, float(h.score)) for h in dense.search_single(qvec, TOP)]
                trace.retrieved("dense", query_text, raw_dense[:30],
                                texts=self._resolve_texts(corpus, [c for c, _ in raw_dense[:30]]))
                dense_docs = self._materialize(corpus, sub, raw_dense, method="dense")
                trace.bullet(f"dense top {TOP}: {len(dense_docs)} docs")
            except Exception as exc:
                trace.bullet(f"dense retrieval failed: {exc}")

        # ---- 3. UNION + dedupe, ONE intent reranker, paper diversity -----
        if exclude:
            primary_docs = [d for d in primary_docs if d.chunk_id not in exclude]
            dense_docs = [d for d in dense_docs if d.chunk_id not in exclude]
            trace.bullet(f"excluded {len(exclude)} previously-seen chunk ids")
        documents = union_rerank_diversify(
            sub,
            primary_docs,
            dense_docs,
            top_k=self.cfg.max_documents,
            max_per_paper=self.cfg.max_per_paper if hasattr(self.cfg, "max_per_paper") else 2,
            score_cap=60,
        )
        trace.bullet(f"union={len(primary_docs)+len(dense_docs)} raw ({primary_name}+dense) -> [dedupe+intent+diversify] -> {len(documents)} top docs")

        # ---- 4. detailed logs: FULL untruncated text ----------------------
        trace.stage(f"RETRIEVED DOCS ({len(documents)}) - FULL TEXT")
        for i, doc in enumerate(documents, 1):
            doc.rank = i
            trace.chunk(doc, rank=i, label="retrieved")
        return documents

    def _materialize(self, corpus: Any, sub: SubQuery, ranks: List[tuple], *, method: str) -> List[RetrievedDocument]:
        """Convert raw (chunk_id, score) ranks into RetrievedDocument with full metadata."""
        ids = [cid for cid, _ in ranks]
        if not ids:
            return []
        resolved = self._resolve_rich(corpus, ids)
        docs = []
        for rank, (cid, score) in enumerate(ranks, start=1):
            meta = resolved.get(cid) or {}
            text = meta.get("text") or ""
            doc = RetrievedDocument(
                subquery_id=sub.id,
                document_id=meta.get("document_id") or "",
                chunk_id=cid,
                rank=rank,
                rrf_score=float(score),
                methods=[method],
                variant_ids=[sub.query or sub.target],
                node_type=meta.get("chunk_type") or "paragraph",
                section=meta.get("section") or "",
                subsection=meta.get("subsection") or "",
                breadcrumb=meta.get("breadcrumb", []),
                table_id=meta.get("table_id"),
                figure_id=meta.get("figure_id"),
                text=text,
                token_count=len(text.split()),
            )
            docs.append(doc)
        return docs


# ---------------------------------------------------------------------------
# Process-wide singleton: loading BM25 + corpus parquet (+ optional dense
# index / query encoder) costs seconds; it must happen ONCE per process, not
# once per tool call.
# ---------------------------------------------------------------------------

_SHARED_SERVICE: Optional[RetrievalService] = None


def get_retrieval_service(config: Optional[AppConfig] = None) -> RetrievalService:
    """Return the shared RetrievalService (created lazily on first use)."""
    global _SHARED_SERVICE
    if _SHARED_SERVICE is None:
        _SHARED_SERVICE = RetrievalService(config)
    return _SHARED_SERVICE


def reset_retrieval_service() -> None:
    """Drop the singleton (tests)."""
    global _SHARED_SERVICE
    _SHARED_SERVICE = None