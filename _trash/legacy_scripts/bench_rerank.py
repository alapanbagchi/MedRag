"""Benchmark reranker batch sizes, latency, throughput, and GPU memory.

Warms up the model once, then times reranking a fixed candidate pool for
each requested batch size so the comparison is apples-to-apples.

Usage:
    python scripts/bench_rerank.py \
        --query "diabetes mellitus treatment" \
        --candidate-k 50 100 \
        --batch-sizes 8 16 32 \
        --index-dir index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medrag.retrieval.engine import RetrievalEngine
from medrag.models import Candidate
from medrag.retrieval.reranker import CrossEncoderReranker


def _gpu_memory_mb():
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (
            torch.cuda.memory_allocated() / 1e6,
            torch.cuda.memory_reserved() / 1e6,
        )
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="diabetes mellitus treatment")
    ap.add_argument("--candidate-k", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    engine = RetrievalEngine(Path(args.index_dir))

    import torch

    print("=" * 80)
    print("RERANKER BENCHMARK")
    print("=" * 80)
    print(f"query: {args.query}")
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Pre-generate candidate pools (reused across batch sizes).
    pools = {}
    for ck in args.candidate_k:
        result = engine.search(args.query, method="hybrid", top_k=ck, include_text=True)
        pools[ck] = [Candidate.from_scored(h) for h in result.results]

    for bs in args.batch_sizes:
        reranker = CrossEncoderReranker(batch_size=bs)
        engine.reranker = reranker
        print(f"--- batch_size={bs} ---")
        for ck in args.candidate_k:
            candidates = pools[ck]
            # warm up
            reranker.rerank(args.query, candidates, top_k=10)

            lats = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                reranker.rerank(args.query, candidates, top_k=10)
                lats.append((time.perf_counter() - t0) * 1000.0)

            mean_ms = float(np.mean(lats))
            alloc, reserved = _gpu_memory_mb()
            gpu_alloc = "n/a" if alloc is None else f"{alloc:.0f} MB"
            gpu_res = "n/a" if reserved is None else f"{reserved:.0f} MB"
            print(
                f"  candidates={ck:>4}  mean_latency={mean_ms:8.1f} ms  "
                f"throughput={1000.0 * ck / mean_ms:6.1f} cand/s  "
                f"gpu_alloc={gpu_alloc:>9}  gpu_reserved={gpu_res:>9}"
            )


if __name__ == "__main__":
    main()
