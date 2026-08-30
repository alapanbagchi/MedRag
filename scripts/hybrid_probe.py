"""Hybrid retrieval probe with full logs: top 100 -> 50 -> 10 after rerank.

Shows how many docs are retrieved by BM25, dense, and their RRF fusion,
then how reranking trims the list (top 100 -> top 50 -> top 10).

Usage:
    python scripts/hybrid_probe.py "risk factors for COPD exacerbation"
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.planner import SubQuery
from src.config import AppConfig
from src.retrieval.retriever import RetrievedDocument, RetrievalService
from src.retrieval.reranker import rerank_candidates
from src.retrieval.search import build_subquery_variants


def log_doc(doc, idx):
    print(f"    [{idx:>3}] {doc.chunk_id}")
    print(f"          doc={doc.document_id} | type={doc.node_type} | section={doc.section!r} | br={doc.breadcrumb[:2]}")
    print(f"          rrf={doc.rrf_score:.5f} | methods={doc.methods} | variants={len(doc.variant_ids)} | tokens={doc.token_count}")


async def main() -> int:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "risk factors for COPD exacerbation"
    cfg = AppConfig()

    print("=" * 76)
    print("HYBRID RETRIEVAL PROBE (BM25 + dense + RRF + rerank)")
    print(f"query        : {query}")
    print(f"index_dir    : {cfg.index_dir}")
    print(f"enable_dense : {cfg.enable_dense}")
    print(f"bm25_depth   : {cfg.bm25_depth}   dense_depth: {cfg.dense_depth}")
    print("=" * 76)

    service = RetrievalService(cfg)
    components = service._components()
    bm25 = components["bm25"]
    dense = components["dense"]
    corpus = components["corpus"]

    print(f"\n  bm25  : {bm25.n_docs} docs, vocab={bm25.vocab_size}")
    if dense is not None:
        print(f"  dense : {dense.n_total} vectors, dim={dense.dimension}")
    else:
        print("  dense : UNAVAILABLE (BM25-only)")

    from src.retrieval_v2.retriever import rrf_fuse

    sub = SubQuery(id="H1", target=query, query=query, focus="evidence",
                   evidence_required=[], terminology=[])
    variants = [sub.query] + build_subquery_variants(sub, max_variants=cfg.max_documents + 1)
    variants = list(dict.fromkeys(variants))
    print(f"\n  retrieval variants ({len(variants)}): {[v[:40] for v in variants]}")

    # -------- Step 1: raw ranked lists per method --------
    raw_pool = {}
    method_hits = Counter()
    for variant in variants:
        print(f"\n  --- variant: {variant[:60]} ---")

        b25 = [(h.chunk_id, float(h.score)) for h in bm25.search_single(variant, cfg.bm25_depth)]
        method_hits["bm25"] += len(b25)
        print(f"    bm25  candidates : {len(b25)}")

        ranked_lists = [b25]
        labels = ["bm25"]
        if dense is not None:
            try:
                qvec = service._query_encoder.encode_single(variant)
                den = [(h.chunk_id, float(h.score)) for h in dense.search_single(qvec, cfg.dense_depth)]
                method_hits["dense"] += len(den)
                print(f"    dense candidates : {len(den)}")
                ranked_lists.append(den)
                labels.append("dense")
            except Exception as exc:
                print(f"    dense FAILED: {exc}")

        # RRF fusion
        for entry in rrf_fuse(ranked_lists, k=cfg.rrf_k, labels=labels):
            cid = entry["chunk_id"]
            if cid not in raw_pool or entry["rrf_score"] > raw_pool[cid]["rrf_score"]:
                raw_pool[cid] = entry
            methods = [lab for lab in labels if entry["scores"].get(lab, 0.0) > 0]
            raw_pool[cid]["_methods"] = methods

    print(f"\n  ===== RRF FUSION: {len(raw_pool)} unique chunks ===== ")
    print(f"  per-method candidates: {dict(method_hits)}")

    # -------- Step 2: top 100 after fusion (before rerank) --------
    ordered = sorted(raw_pool.values(), key=lambda d: d["rrf_score"], reverse=True)
    top100_ids = [d["chunk_id"] for d in ordered[:100]]
    print(f"\n  ===== TOP {len(top100_ids)} after RRF (pre-rerank) ===== ")
    resolved = corpus.resolve(top100_ids, include_text=True)
    docs = []
    for rank, cid in enumerate(top100_ids, 1):
        meta = resolved.get(cid) or {}
        docs.append(RetrievedDocument(
            subquery_id=sub.id,
            document_id=meta.get("document_id") or "",
            chunk_id=cid,
            rank=rank,
            rrf_score=float(raw_pool[cid]["rrf_score"]),
            methods=list(raw_pool[cid].get("_methods", [])),
            variant_ids=[],
            node_type=meta.get("chunk_type") or "paragraph",
            section=meta.get("section") or "",
            subsection=meta.get("subsection") or "",
            breadcrumb=meta.get("breadcrumb", []),
            table_id=meta.get("table_id"),
            figure_id=meta.get("figure_id"),
            text=meta.get("text") or "",
            token_count=len((meta.get("text") or "").split()),
        ))

    for i, d in enumerate(docs[:30], 1):
        log_doc(d, i)
    if len(docs) > 30:
        print(f"    ... ({len(docs) - 30} more in top {len(docs)})")

    # -------- Step 3: rerank then trim to 50 / 10 --------
    reranked_100 = rerank_candidates(sub, docs)
    print(f"\n  ===== AFTER RERANK (sorted by score) ===== ")
    for d in reranked_100[:10]:
        log_doc(d, d.rank)

    top50 = reranked_100[:50]
    top10 = reranked_100[:10]

    print(f"\n  -------- SUMMARY --------")
    print(f"  RRF unique chunks       : {len(raw_pool)}")
    print(f"  Top 100 (pre-rerank)    : {len(docs)}")
    print(f"  Top 50  (after rerank)  : {len(top50)}")
    print(f"  Top 10  (after rerank)  : {len(top10)}")

    # -------- Step 4: distribution at each cut --------
    for label, subset in (("TOP 100", docs), ("TOP 50", top50), ("TOP 10", top10)):
        types = Counter(d.node_type for d in subset)
        methods = Counter(m for d in subset for m in d.methods)
        print(f"  {label}: node_types={dict(types)} | methods={dict(methods)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
