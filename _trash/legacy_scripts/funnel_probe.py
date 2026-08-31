"""Funnel analysis: candidate count -> RRF pool -> topk cuts, with metadata."""
from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.plans import SubQuery
from src.config import AppConfig
from src.retrieval.retriever import RetrievedDocument, RetrievalService
from src.retrieval.reranker import rerank_candidates
from src.retrieval.search import build_subquery_variants


def _br(breadcrumb) -> str:
    items = list(breadcrumb or [])
    return " > ".join(str(b) for b in items[:4])


async def main() -> int:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "risk factors for COPD exacerbation"
    cfg = AppConfig()
    cfg.local_mode = True
    cfg.vector_db_url = ""
    cfg.max_documents = 30

    print("=" * 74)
    print("RETRIEVAL FUNNEL ANALYSIS (local: BM25 + FAISS dense + RRF)")
    print(f"query        : {query}")
    print(f"bm25_depth   : {cfg.bm25_depth} | dense_depth: {cfg.dense_depth}")
    print("=" * 74)

    svc = RetrievalService(cfg)
    components = svc._components()
    bm25 = components["bm25"]
    dense = components["dense"]
    corpus = components["corpus"]
    print(f"  bm25 : {bm25.n_docs} docs")
    print(f"  dense: {dense.n_total} vectors (FAISS local)")
    print(f"  corpus: {corpus.n_chunks} chunks\n")

    sub = SubQuery(id="H1", target=query, query=query, focus="evidence",
                   evidence_required=[], terminology=[])
    variants = list(dict.fromkeys([sub.query] + build_subquery_variants(sub, max_variants=10)))

    print(f"variants ({len(variants)}):")
    for v in variants:
        print(f"  - {v[:70]}")

    # ---- stage counts ----
    cand_counts = Counter()
    fused = {}
    methods_seen = {}

    from src.retrieval_v2.retriever import rrf_fuse

    for variant in variants:
        b25 = [(h.chunk_id, float(h.score)) for h in bm25.search_single(variant, cfg.bm25_depth)]
        cand_counts["bm25"] += len(b25)
        lists, labels = [b25], ["bm25"]
        if dense is not None:
            qvec = svc._query_encoder.encode_single(variant)
            den = [(h.chunk_id, float(h.score)) for h in dense.search_single(qvec, cfg.dense_depth)]
            cand_counts["dense"] += len(den)
            lists.append(den); labels.append("dense")
        for e in rrf_fuse(lists, k=cfg.rrf_k, labels=labels):
            cid = e["chunk_id"]
            if cid not in fused or e["rrf_score"] > fused[cid]["rrf_score"]:
                fused[cid] = e
            ms = [lab for lab in labels if e["scores"].get(lab, 0.0) > 0]
            methods_seen[cid] = ms

    print(f"\n── CANDIDATE COUNTS (summed over variants) ──")
    for k, v in cand_counts.items():
        print(f"   {k}: {v} raw candidates")
    print(f"   RRF unique pool: {len(fused)}")

    ordered = sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)
    all_ids = [d["chunk_id"] for d in ordered]

    # topk cuts
    for cut in (100, 50, 10, 5):
        ids = all_ids[:cut]
        resolved = svc._resolve_rich(corpus, ids)
        docs = []
        for rank, cid in enumerate(ids, 1):
            m = resolved.get(cid) or {}
            docs.append(RetrievedDocument(
                chunk_id=cid,
                document_id=m.get("document_id", ""),
                rank=rank,
                rrf_score=fused[cid]["rrf_score"],
                methods=methods_seen.get(cid, []),
                node_type=m.get("chunk_type", "paragraph"),
                section=m.get("section", ""),
                subsection=m.get("subsection", ""),
                breadcrumb=list(m.get("breadcrumb", []) or []),
                table_id=m.get("table_id"),
                figure_id=m.get("figure_id"),
                text=(m.get("text") or "")[:120],
                token_count=len((m.get("text") or "").split()),
            ))
        types = Counter(d.node_type for d in docs)
        meth = Counter(mm for d in docs for mm in d.methods)
        print(f"\n── TOP {cut} ──")
        print(f"   node_types : {dict(types)}")
        print(f"   methods    : {dict(meth)}")
        withsec = sum(1 for d in docs if d.section)
        print(f"   with section: {withsec}/{len(docs)}")
        if cut == 5:
            for d in docs:
                br = _br(d.breadcrumb)
                print(f"   [{d.rank}] {d.chunk_id} | {d.node_type} | sec={d.section!r} | br={br} | {d.methods} | rrf={d.rrf_score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
