#!/usr/bin/env python
"""Time the hybrid retrieval tool end to end.

Runs the real EvidenceRetriever.search pipeline — BM25 + MedCPT dense legs,
RRF fusion, MedCPT cross-encoder rerank, boundary expansion — and prints the
per-stage timings the retriever reports plus a per-run total, so a retrieval
change can be measured before/after. The first run is cold (weights load);
later runs are warm and are the meaningful comparison.

Usage:
    .venv/bin/python scripts/retrieval_check.py "hypertension thresholds" --runs 3
    .venv/bin/python scripts/retrieval_check.py "..." --runs 1 --json
    .venv/bin/python scripts/retrieval_check.py "..." --expect chunkA,chunkB

    # or via make:
    make retrieval query="hypertension thresholds" R=5 SHOW=3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

from src.tools.retrieval import get_retriever  # noqa: E402


def _one_line(text: str, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def _print_hits(hits, show: int) -> None:
    for i, hit in enumerate(hits[:show], 1):
        print(f"    {i:2d}. {hit.chunk_id}  score={hit.score:.4f}"
              f"  {hit.chunk_type or '-'}  [{hit.section or '-'}]"
              f"  {_one_line(hit.text, 110)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="retrieval query")
    parser.add_argument("--runs", type=int, default=3,
                        help="timed runs (default 3; run 1 is cold)")
    parser.add_argument("--per-leg", type=int, default=60,
                        help="candidates fetched per leg (BM25/dense)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="passages returned after the cross-encoder")
    parser.add_argument("--show", type=int, default=5,
                        help="hits to print per run")
    parser.add_argument("--expect", default="",
                        help="comma-separated chunk ids that must appear in top-k")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip explicit warmup (timings then include model load)")
    parser.add_argument("--json", action="store_true",
                        help="also emit a machine-readable summary")
    args = parser.parse_args()

    retriever = get_retriever()

    if not args.no_warmup:
        print("== warmup (query encoder / pgvector / BM25 / cross-encoder) ==")
        started = time.perf_counter()
        ready = retriever.warmup(lambda stage: print(f"    {stage}"))
        print(f"    warmup {'ready' if ready else 'PARTIAL'}"
              f" in {time.perf_counter() - started:.1f}s")

    runs = []
    for i in range(1, max(1, args.runs) + 1):
        print(f"\n== run {i}/{max(1, args.runs)}  query={args.query!r}"
              f"  per_leg={args.per_leg}  top_k={args.top_k} ==")
        started = time.perf_counter()
        hits = retriever.search(
            args.query, args.per_leg, args.top_k,
            progress=lambda stage: print(
                f"    [{time.perf_counter() - started:6.2f}s] {stage}"),
        )
        elapsed = time.perf_counter() - started
        print(f"    total: {elapsed:.2f}s  ({len(hits)} hit(s))")
        _print_hits(hits, args.show)
        runs.append({
            "run": i,
            "elapsed_s": round(elapsed, 3),
            "hits": len(hits),
            "chunk_ids": [h.chunk_id for h in hits],
        })

    times = [run["elapsed_s"] for run in runs]
    print("\n== summary ==")
    print(f"    runs={len(times)}  min={min(times):.2f}s"
          f"  median={statistics.median(times):.2f}s  max={max(times):.2f}s")
    if len(times) > 1:
        print(f"    warm-only: median={statistics.median(times[1:]):.2f}s"
              f"  (run 1 cold = {times[0]:.2f}s)")

    status = 0
    if args.expect:
        expected = [cid.strip() for cid in args.expect.split(",") if cid.strip()]
        got = set(runs[-1]["chunk_ids"])
        missing = [cid for cid in expected if cid not in got]
        print(f"    recall@top{args.top_k}: {len(expected) - len(missing)}/{len(expected)}"
              + (f"  MISSING {missing}" if missing else "  all found"))
        if missing:
            status = 1

    if args.json:
        print(json.dumps({"query": args.query, "runs": runs}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    sys.exit(main())
