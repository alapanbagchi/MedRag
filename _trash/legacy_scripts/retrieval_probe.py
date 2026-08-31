"""Standalone hybrid retrieval + rerank inspection.

Shows exactly what documents the retrieval stack returns after RRF fusion
and reranking, including document type (node_type, section, breadcrumb),
scores, and the methods (bm25/dense) that matched each chunk.

Usage:
    python scripts/retrieval_probe.py "risk factors for COPD exacerbation"

Writes the full detailed retrieval trace (per-method ranks + full chunk text
+ RRF fusion) incrementally to logs.txt.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.plans import SubQuery
from src.config import AppConfig
from src.retrieval.retriever import RetrievalService


def format_doc(doc, idx):
    lines = [
        f"  [{idx}] {doc.chunk_id}",
        f"       document : {doc.document_id}",
        f"       node_type: {doc.node_type}   section={doc.section!r}   subsection={doc.subsection!r}",
        f"       breadcrumb: {' > '.join(str(b) for b in doc.breadcrumb[:4]) if doc.breadcrumb else ''}",
        f"       table_id : {doc.table_id}   figure_id: {doc.figure_id}",
        f"       rrf      : {doc.rrf_score:.4f}   rank={doc.rank}",
        f"       methods  : {doc.methods}   variants={len(doc.variant_ids)}",
        f"       tokens   : {doc.token_count}",
        f"       text     : {doc.text[:160]!r}...",
    ]
    return "\n".join(lines)


async def main() -> int:
    import os

    from src.lib.trace import get_trace

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "risk factors for COPD exacerbation"
    cfg = AppConfig()
    cfg.local_mode = True
    cfg.vector_db_url = ""
    os.environ["LOCAL_MODE"] = "true"
    os.environ["VECTOR_DB_URL"] = ""

    print("=" * 72)
    print("STANDALONE HYBRID RETRIEVAL + RERANK (local)")
    print(f"query        : {query}")
    print(f"index_dir    : {cfg.index_dir}")
    print(f"enable_dense : {cfg.enable_dense}")
    print(f"max_documents: {cfg.max_documents}")
    print(f"bm25_depth   : {cfg.bm25_depth}")
    print(f"logging to logs.txt incrementally")
    print("=" * 72)

    log_path = os.environ.get("LOG_FILE", "logs.txt")
    trace = get_trace()
    trace.open_stream(log_path, query=query)

    service = RetrievalService(cfg)

    sub = SubQuery(
        id="H1",
        target=query,
        query=query,  # natural language query used as primary variant
        focus="evidence",
        evidence_required=[],
        terminology=[],
    )

    try:
        docs = await service.search_subquery(sub)
    finally:
        trace.close_stream()

    print(f"\nRetrieved {len(docs)} docs (after RRF fusion + rerank)\n")

    for i, d in enumerate(docs, 1):
        print(format_doc(d, i))
        print()

    print("-" * 72)
    methods = Counter()
    types = Counter()
    sections = Counter()
    for d in docs:
        methods.update(d.methods)
        types[d.node_type] += 1
        sections[d.section or "(none)"] += 1
    print(f"methods per doc : {dict(methods)}")
    print(f"node_type dist  : {dict(types)}")
    print(f"section dist    : {dict(sections)}")
    print("-" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))