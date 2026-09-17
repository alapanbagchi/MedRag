"""Scratch probe: cross-encoder intra-op thread count on CPU.

Keeps all 60 fused candidates (the rerank set is fixed), and measures a full
60-pair scoring pass at several torch intra-op thread counts.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

import torch  # noqa: E402

from src.tools.retrieval import get_retriever  # noqa: E402

QUERY = ("What are the diagnostic thresholds and first-line treatment for "
         "hypertension?")


def main() -> int:
    retriever = get_retriever()
    print("warming up…", flush=True)
    retriever.warmup(lambda stage: None)
    hits = retriever.search(QUERY, 60, 60)
    pairs = [(QUERY, h.text) for h in hits[:60]]
    ce = retriever._reranker_or_build(None)
    print(f"{len(pairs)} pairs | cpus={os.cpu_count()} | "
          f"torch default threads={torch.get_num_threads()}", flush=True)
    for n in (16, 8, 4, 12, 16):
        torch.set_num_threads(n)
        ce.score([("warm", "warm")])  # settle the pool at this width
        started = time.perf_counter()
        ce.score(pairs)
        print(f"  threads={n:2d}: {time.perf_counter() - started:6.2f}s",
              flush=True)
    return 0


raise SystemExit(main())
