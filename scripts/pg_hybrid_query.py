#!/usr/bin/env python3
"""Hybrid retrieval + reranking with exhaustive text logging.

Uses the SAME components the agentic retriever (src.retrieval.retriever's
RetrievalService, wrapped by src/agentic/retriever_tool.py) uses:

    BM25Index (index/bm25_v2 when present)        -- sparse branch
    PgDenseIndex (pgvector medrag.embeddings)      -- dense branch
    MedCPTQueryEncoder                              -- query embedding
    SubQuery + reranker.rerank_union + diversify_papers
                                                    -- union, intent rerank,
                                                       paper diversification

Flow:
    BM25  top N ──┐
                   ├── UNION + dedupe (methods merged)
    Dense top N ──┘
                   ↓
       intent reranker (whole union scored once)
                   ↓
       paper diversification (max_per_paper per paper)
                   ↓
       top K

The timestamped log in --log-dir prints the FULL TEXT (untruncated) of
EVERY BM25 hit and EVERY dense hit, the union/rerank decisions, and the
final diversified ranking — everything mirrored to the console.

Usage:
    python scripts/pg_hybrid_query.py "what is hypertension"
    python scripts/pg_hybrid_query.py "q" --top-k 8 --cross-encoder --log-dir logs
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass


class RunLog:
    """Mirror every line to the console AND a timestamped log file."""

    def __init__(self, log_dir: Path, query: str, tag: str = "hybrid") -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip())[:60].strip("_") or "query"
        self.path = log_dir / f"{tag}_{slug}_{stamp}.log"
        self.fh = open(self.path, "w", encoding="utf-8")
        self.line("#" * 76)
        self.line(f"# {tag} retrieval run  |  {_dt.datetime.now().isoformat()}")
        self.line(f"# query: {query}")
        self.line("#" * 76)

    def line(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def divider(self, title: str = "") -> None:
        self.line("=" * 76)
        if title:
            self.line(title)
            self.line("-" * 76)

    def close(self) -> None:
        self.line(f"\n# complete: {self.path}")
        self.fh.close()


def fts_ranked(store, query: str, top_k: int, config: str = "simple", max_terms: int = 12) -> List[Tuple[str, float]]:
    """Sparse branch straight from Postgres, with BM25-like IDF weighting.

    Plain ``ts_rank_cd`` on an OR'd query ranks by positional coverage, so
    common terms (e.g. "hypertension") flood the top hits and diet-specific
    chunks sink — there is no inverse-document-frequency weighting in
    Postgres FTS. Here each query term gets a rarity weight
    (``log(1 + N/df)``, df from a cheap GIN count) and per-term ``ts_rank_cd``
    pools are merged idf-weighted — a sparse rank close to BM25, still with
    no local index. Terms with df > 55% of corpus are treated as stopwords.
    """
    import math as _math

    tokens = list(dict.fromkeys(re.findall(r"[a-z0-9]+", (query or "").lower())))
    if not tokens:
        return []
    terms = tokens[:max_terms]

    conn = store.connect()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM medrag.chunks WHERE tsv IS NOT NULL AND tsv <> ''::tsvector")
    n_docs = max(1, cur.fetchone()[0])

    idf: Dict[str, float] = {}
    for t in terms:
        cur.execute(
            f"SELECT count(*) FROM medrag.chunks WHERE tsv @@ to_tsquery('{config}', %s)", (t,)
        )
        df = cur.fetchone()[0]
        if df > n_docs * 0.55:      # ~stopword (how/does/what...) — no signal
            continue
        idf[t] = _math.log(1.0 + n_docs / (df + 1.0))

    acc: Dict[str, float] = {}
    for t, w in idf.items():
        # per-term coverage rank, idf-weighted: tf-ish signal * rarity
        cur.execute(
            f"SELECT id, ts_rank_cd(tsv, to_tsquery('{config}', %s), 1) AS r "
            f"FROM medrag.chunks WHERE tsv @@ to_tsquery('{config}', %s) "
            f"ORDER BY r DESC LIMIT %s",
            (t, t, top_k),
        )
        for cid, r in cur.fetchall():
            acc[cid] = acc.get(cid, 0.0) + w * float(r)
    cur.close()

    if not acc:
        return []
    return sorted(acc.items(), key=lambda kv: -kv[1])[:top_k]


def resolve_chunks(store, chunk_ids: List[str]) -> Dict[str, dict]:
    """Fetch chunk metadata + TEXT for a list of ids from medrag.chunks."""
    if not chunk_ids:
        return {}
    conn = store.connect()
    cur = conn.cursor()
    out: Dict[str, dict] = {}
    BATCH = 1000
    for i in range(0, len(chunk_ids), BATCH):
        batch = chunk_ids[i:i + BATCH]
        cur.execute(
            "SELECT id, document_id, chunk_type, section, text, table_id, figure_id "
            "FROM medrag.chunks WHERE id = ANY(%s)",
            (batch,),
        )
        for r in cur.fetchall():
            out[r[0]] = {"document_id": r[1], "chunk_type": r[2], "section": r[3],
                         "text": r[4], "table_id": r[5], "figure_id": r[6]}
    cur.close()
    return out


class Doc:
    """Duck-typed candidate the reranker/diversifier expect (RetrievedDocument-like)."""

    def __init__(self, chunk_id: str, document_id: str, method: str, rank: int,
                 score: float, node_type: str, section: str, text: str,
                 table_id: Optional[str], figure_id: Optional[str]) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.rank = rank
        self.rrf_score = float(score)
        self.methods: List[str] = [method]
        self.variant_ids: List[str] = []
        self.node_type = node_type or "paragraph"
        self.section = section or ""
        self.subsection = ""
        self.breadcrumb: List[str] = []
        self.table_id = table_id
        self.figure_id = figure_id
        self.text = text or ""
        self.token_count = len(self.text.split())


def materialize(metas: Dict[str, dict], ranked: List[Tuple[str, float]], method: str) -> List[Doc]:
    docs: List[Doc] = []
    for rank, (cid, score) in enumerate(ranked, 1):
        m = metas.get(cid) or {}
        docs.append(Doc(
            chunk_id=cid, document_id=m.get("document_id") or "", method=method,
            rank=rank, score=float(score), node_type=m.get("chunk_type") or "paragraph",
            section=m.get("section") or "", text=m.get("text") or "",
            table_id=m.get("table_id"), figure_id=m.get("figure_id"),
        ))
    return docs


def log_full_list(log: RunLog, title: str, docs: List[Doc]) -> None:
    """Log every hit with FULL untruncated text."""
    log.divider(f"{title} ({len(docs)})")
    for i, d in enumerate(docs, 1):
        log.line(f"  {i:>2}. [{d.rrf_score:.4f}] {d.chunk_id}"
                 f"  (doc {d.document_id or '?'}, {d.node_type}, sec={d.section or '-'})")
        log.line("     FULL TEXT:")
        log.line("     " + (d.text or "(empty)").replace("\n", " \n     "))
        log.line("     --------")


def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("query", help="the question to search")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bm25-depth", type=int, default=60)
    parser.add_argument("--dense-depth", type=int, default=60)
    parser.add_argument("--score-cap", type=int, default=60,
                        help="how many union docs the intent reranker scores")
    parser.add_argument("--max-per-paper", type=int, default=2,
                        help="paper cap after rerank (0 = no cap / dedup OFF)")
    parser.add_argument("--cross-encoder", action="store_true",
                        help="rerank the union with ncbi/MedCPT-Cross-Encoder too")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--index-dir", type=Path, default=Path(os.environ.get("INDEX_DIR", "index")))
    parser.add_argument("--bm25-dir", type=Path, default=None,
                        help="local BM25 index dir (only with --sparse bm25)")
    parser.add_argument("--sparse", choices=("pgfts", "bm25"), default="pgfts",
                        help="sparse branch: Postgres FTS (default, no local index) "
                             "or the local BM25 index")
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", "medrag"))
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "medrag"))
    args = parser.parse_args(argv)

    log = RunLog(args.log_dir, args.query)
    log.line(f"config: bm25_depth={args.bm25_depth} dense_depth={args.dense_depth} "
             f"score_cap={args.score_cap} top_k={args.top_k} "
             f"max_per_paper={args.max_per_paper} cross_encoder={args.cross_encoder}")

    from src.retrieval.pgvector_store import PgConfig, PgVectorStore
    store = PgVectorStore(PgConfig(host=args.host, port=args.port, user=args.user,
                                   password=args.password, database=args.database))
    try:
        store.connect()
    except Exception as e:
        log.line(f"ERROR: cannot connect to PostgreSQL: {e}")
        log.close()
        return 1

    try:
        st = store.stats()
        log.line(f"DB: {args.host}:{args.port}/{args.database}  "
                 f"(chunks={st['chunks']:,}, embeddings={st['embeddings']:,}, "
                 f"documents={st['documents']:,})")
        cur = store.connect().cursor()
        cur.execute("SELECT 1 FROM pg_indexes WHERE schemaname='medrag' AND tablename='embeddings' "
                    "AND indexname='idx_embeddings_vector'")
        log.line(f"vector index: {'HNSW (index scan)' if cur.fetchone() else 'none yet (exact scan)'}")
        cur.close()

        # ---------------------------------------------------------------
        # SPARSE branch  [same role as RetrievalService's BM25 leg]
        #   default : Postgres FTS (tsvector @@ websearch, ts_rank_cd) —
        #             zero local index, straight from the DB
        #   --sparse bm25 : local BM25Index (index/bm25_v2 when present)
        # ---------------------------------------------------------------
        sparse_label = ""
        if args.sparse == "pgfts":
            sparse_label = f"pgfts(tsv GIN)"
            try:
                bm25_ranked = fts_ranked(store, args.query, args.bm25_depth)
                log.line(f"\nsparse index: {sparse_label}  (no local BM25 needed)")
            except Exception as e:
                log.line(f"WARNING: pgfts sparse branch failed ({e}); attempt local BM25 fallback")
                args.sparse = "bm25"

        if args.sparse == "bm25":
            from src.retrieval.sparse import BM25Index
            bm25_dir = args.bm25_dir
            if bm25_dir is None:
                v2 = args.index_dir / "bm25_v2"
                bm25_dir = v2 if (v2 / "bm25_meta.json").is_file() else args.index_dir / "bm25"
            bm25 = BM25Index.load(bm25_dir)
            sparse_label = f"bm25({bm25_dir.name}, n_docs={bm25.meta.get('n_docs', '?')})"
            log.line(f"\nsparse index: {sparse_label}")
            bm25_ranked = [(h.chunk_id, float(h.score))
                           for h in bm25.search_single(args.query, args.bm25_depth)]

        # ---------------------------------------------------------------
        # DENSE branch: MedCPT query encoder -> pgvector  [same as service]
        # ---------------------------------------------------------------
        from src.retrieval.dense_pgvector import PgDenseIndex
        from src.retrieval.query import MedCPTQueryEncoder
        enc = MedCPTQueryEncoder()
        log.line(f"dense encoder: {enc.model_name}")
        qvec = enc.encode_single(args.query)
        dense_hits = PgDenseIndex(store).search_single(qvec, args.dense_depth)
        dense_ranked = [(h.chunk_id, float(h.score)) for h in dense_hits]

        # materialize metadata + FULL TEXT for every hit
        all_ids = list(dict.fromkeys([c for c, _ in bm25_ranked] + [c for c, _ in dense_ranked]))
        metas = resolve_chunks(store, all_ids)
        sparse_method = "fts" if args.sparse == "pgfts" else "bm25"
        bm25_docs = materialize(metas, bm25_ranked, sparse_method)
        dense_docs = materialize(metas, dense_ranked, "dense")

        # ---- print FULL TEXT of every raw hit (the log asks --------------#
        log_full_list(log, f"SPARSE ({sparse_label}) top {len(bm25_docs)} — RAW, FULL TEXT", bm25_docs)
        log_full_list(log, f"DENSE top {len(dense_docs)} — RAW, FULL TEXT", dense_docs)

        # ---------------------------------------------------------------
        # UNION + dedupe (methods merged) — same as retriever.py
        # ---------------------------------------------------------------
        seen: Dict[str, Doc] = {}
        for d in [*bm25_docs, *dense_docs]:
            if d.chunk_id in seen:
                for m in d.methods:
                    if m not in seen[d.chunk_id].methods:
                        seen[d.chunk_id].methods.append(m)
                continue
            seen[d.chunk_id] = d
        union = list(seen.values())
        both = [d for d in union if len(d.methods) > 1]
        log.divider(f"UNION {len(union)}  (deduped from {len(bm25_docs)}+{len(dense_docs)}; "
                    f"found by BOTH: {len(both)})")
        for d in union:
            log.line(f"  {d.chunk_id}  methods={d.methods}  "
                     f"sparse_score={d.rrf_score if d.methods[0] == sparse_method else '-'}")

        # ---------------------------------------------------------------
        # INTENT RERANK (whole union scored once) — same as rerank_union
        # ---------------------------------------------------------------
        from src.retrieval.plans import SubQuery
        from src.retrieval.reranker import diversify_papers, rerank_union
        sub = SubQuery(id="H1", target=args.query, query=args.query,
                       evidence_required=[], terminology=[])
        scored = rerank_union(sub, union, top_k=args.score_cap)
        for s in scored:
            s["doc"].intent_score = s["intent_score"]   # visible on final docs
        log.divider(f"INTENT RERANK top {len(scored)} (whole union scored once)")

        if args.cross_encoder:
            try:
                from src.retrieval.cross_encoder import MedCPTCrossEncoder
                ce = MedCPTCrossEncoder()
                ce_scores = ce.score(args.query, [s["doc"].text for s in scored])
                for i, s in enumerate(scored):
                    s["ce_score"] = round(float(ce_scores[i]), 5)
                    s["doc"].ce_score = s["ce_score"]
                scored.sort(key=lambda s: -s["ce_score"])
                log.line("  repranked by ncbi/MedCPT-Cross-Encoder")
            except Exception as e:
                log.line(f"  WARNING: cross-encoder unavailable ({e}); keeping intent order")

        for i, s in enumerate(scored, 1):
            d = s["doc"]
            ce = f"  ce={s.get('ce_score', '-')}" if "ce_score" in s else ""
            log.line(f"  {i:>2}. intent={s['intent_score']:.4f}{ce}  {d.chunk_id}  "
                     f"({d.node_type}, sec={d.section or '-'}) found_by={d.methods}")

        # ---------------------------------------------------------------
        # FINAL top K — paper diversification unless --max-per-paper 0
        # ---------------------------------------------------------------
        if args.max_per_paper and args.max_per_paper > 0:
            final_docs = diversify_papers(scored, top_k=args.top_k,
                                          max_per_paper=args.max_per_paper)
            cap_label = f"(max {args.max_per_paper}/paper)"
        else:
            final_docs = [s["doc"] for s in scored[:args.top_k]]
            cap_label = "(paper dedup OFF)"
        log.divider(f"FINAL top {len(final_docs)} {cap_label}")
        for rank, d in enumerate(final_docs, 1):
            d.rank = rank
            ce = f" ce={d.ce_score:.4f}" if getattr(d, "ce_score", None) is not None else ""
            log.line(f"\n>>> {rank}. {d.chunk_id}  (doc {d.document_id or '?'}, "
                     f"{d.node_type}, sec={d.section or '-'})  found_by={d.methods}"
                     f"  intent={getattr(d, 'intent_score', 0.0):.4f}{ce}")
            log.line("    FULL TEXT:")
            log.line("    " + (d.text or "(empty)").replace("\n", " \n    "))
            log.line("    --------")
    except Exception as e:
        import traceback
        log.line("\nERROR:")
        log.line(traceback.format_exc())
        store.close()
        log.close()
        return 1

    store.close()
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())