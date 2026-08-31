"""Standalone SPLADE retrieval test.

TEST: swap BM25 -> SPLADE in the primary slot of the existing hybrid+rerank flow.
Everything else (FAISS dense, union, intent rerank, paper diversification,
full-text logs) is IDENTICAL - only the primary sparse retriever changes.

Usage:
    python scripts/splade_probe.py "cardiac arrest definition"

Requires a SPLADE index:
    python scripts/build_splade_index.py --max-docs 5000   # quick test
    python scripts/build_splade_index.py                    # full corpus
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

from src.retrieval.plans import SubQuery
from src.config import AppConfig
from src.retrieval.retriever import RetrievalService


async def main() -> int:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "cardiac arrest definition"

    # Force local + SPLADE as the primary retriever
    os.environ["LOCAL_MODE"] = "true"
    os.environ["VECTOR_DB_URL"] = ""
    os.environ["RETRIEVAL_PRIMARY"] = "splade"

    cfg = AppConfig()
    cfg.local_mode = True
    cfg.vector_db_url = ""
    cfg.retrieval_primary = "splade"
    cfg.max_documents = 8

    from src.lib.trace import get_trace
    trace = get_trace()
    log_path = os.environ.get("LOG_FILE", "logs.txt")
    trace.open_stream(log_path, query=query)

    print("=" * 74)
    print("SPLADE PROBE (primary=SPLADE + dense FAISS -> union -> intent rerank -> diversify)")
    print(f"query: {query}")
    print(f"max_documents: {cfg.max_documents}")
    print("=" * 74)

    try:
        svc = RetrievalService(cfg)
        sub = SubQuery(
            id="H1",
            target=query,
            query=query,
            focus="evidence",
            evidence_required=[],
            terminology=[],
        )
        docs = await svc.search_subquery(sub)
    finally:
        trace.close_stream()

    print(f"\nRetrieved {len(docs)} docs (SPLADE primary)\n")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.chunk_id}")
        print(f"       document : {d.document_id}")
        print(f"       node_type: {d.node_type}   section={d.section!r}")
        print(f"       methods  : {d.methods}   rank={d.rank}")
        print(f"       text     : {d.text[:120]!r}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
