#!/usr/bin/env python
"""A/B the evidence judge's reasoning effort on one identical batch.

Builds the exact verifier message for a query's top-k retrieved passages and
sends it once per reasoning effort (default / medium / low / minimal), in one
process, so the only variable is the reasoning effort. Prints tokens, the
reasoning vs JSON split, latency, and how many passages the judge kept, so a
speed win can be checked against a quality change.

Two modes:
  * default: one hand-built batched message per effort (--efforts), so the
    only variable is reasoning effort.
  * --batch-sizes: run the real verify_passages pipeline per passages-per-
    call value, so retries, the shared pool, and cache behaviour count.

Usage:
    .venv/bin/python scripts/judge_check.py "hypertension diagnosis thresholds" --efforts default,medium,low
    .venv/bin/python scripts/judge_check.py "..." --batch-sizes 1,2,4,10 --effort low
    .venv/bin/python scripts/judge_check.py "..." --runs 2 --json

    # or via make:
    make judge query="hypertension diagnosis thresholds" EFFORTS=default,medium,low
    make judge query="..." ARGS="--batch-sizes 1,2,4,10"
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

from src.tools.retrieval import get_retriever  # noqa: E402
from src.tools.verifier import (  # noqa: E402
    EvidenceRequirement,
    _get_verifier_agent,
    _parse_response,
    _usage_detail,
    build_verifier_message,
    normalize_response,
    verify_passages,
)

# Generic, query-agnostic requirements: enough to exercise the keep gate on any
# medical question and stable across arms.
DEFAULT_REQS = [
    ("E1", "Diagnostic thresholds, definitions, or classification"),
    ("E2", "Treatment, management, or intervention recommendations"),
    ("E3", "Epidemiology, prevalence, risk factors, or prognosis"),
]


def _label(effort: str) -> str:
    return effort or "default"


def _parse_reqs(raw: list[str]) -> list[tuple[str, str]]:
    if not raw:
        return DEFAULT_REQS
    out: list[tuple[str, str]] = []
    for i, item in enumerate(raw, 1):
        rid, _, desc = item.partition(":")
        if desc.strip():
            out.append((rid.strip(), desc.strip()))
        else:
            out.append((f"E{i}", item.strip()))
    return out


async def _arm(effort: str, message: str, requirements, passages) -> dict:
    """One judge call at a pinned reasoning effort on the shared message."""
    agent = _get_verifier_agent(False, effort)
    started = time.perf_counter()
    error = ""
    output = ""
    detail: dict = {}
    try:
        result = await agent.run(message)
        output = getattr(result, "output", "") or ""
        detail = _usage_detail(result)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    view = normalize_response(_parse_response(output), requirements, passages,
                              check_verbatim=False)
    verdicts = view.evidence_results or []
    kept = [v for v in verdicts if v.coverage]
    scores = [v.intent_score for v in verdicts]
    return {
        "effort": _label(effort),
        "wall_s": round(wall, 2),
        "prompt": detail.get("prompt", 0),
        "cached": detail.get("cached", 0),
        "completion": detail.get("completion", 0),
        "reasoning": detail.get("reasoning", 0),
        "json": max(0, detail.get("completion", 0) - detail.get("reasoning", 0)),
        "kept": len(kept),
        "passages": len(verdicts),
        "avg_intent": round(statistics.fmean(scores), 3) if scores else 0.0,
        "coverage": sorted({c for v in verdicts for c in (v.coverage or [])}),
        "verdicts": {v.passage_id: [round(v.intent_score, 2),
                                     sorted(v.coverage or [])]
                     for v in verdicts},
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="retrieval query / question")
    parser.add_argument("--efforts", default="default,medium,low",
                        help="comma-separated reasoning efforts (default, medium, low, minimal, none)")
    parser.add_argument("--runs", type=int, default=2,
                        help="runs per effort; run 2 is the warm-cache number")
    parser.add_argument("--batch", type=int, default=10,
                        help="passages judged in the single call")
    parser.add_argument("--top-k", type=int, default=10,
                        help="passages retrieved before slicing to --batch")
    parser.add_argument("--per-leg", type=int, default=60,
                        help="candidates fetched per retrieval leg")
    parser.add_argument("--req", action="append", default=[],
                        help="evidence requirement as 'E1: description' (repeatable)")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="truncate each passage to N chars before judging (0 = full text)")
    parser.add_argument("--batch-sizes", default="",
                        help="sweep passages-per-call through the real verify_passages")
    parser.add_argument("--effort", default="low",
                        help="reasoning effort pinned during a --batch-sizes sweep")
    parser.add_argument("--json", action="store_true",
                        help="also emit a machine-readable summary")
    args = parser.parse_args()

    retriever = get_retriever()
    print("== warmup (query encoder / pgvector / BM25 / cross-encoder) ==")
    retriever.warmup(lambda stage: print(f"    {stage}"))

    hits = retriever.search(args.query, args.per_leg, args.top_k)
    passages = [{"id": h.chunk_id, "text": h.text} for h in hits[: args.batch]]
    if not passages:
        print("!! retrieval returned no passages; nothing to judge")
        return 1
    print(f"    {len(passages)} passage(s) for query {args.query!r}")

    if args.max_chars > 0:
        raw_chars = sum(len(p["text"]) for p in passages)
        passages = [{"id": p["id"], "text": p["text"][: args.max_chars]}
                    for p in passages]
        kept_chars = sum(len(p["text"]) for p in passages)
        print(f"    truncated passages to {args.max_chars:,} chars"
              f" ({raw_chars:,} -> {kept_chars:,} chars)")

    requirements = [EvidenceRequirement(id=i, description=d)
                    for i, d in _parse_reqs(args.req)]
    message = build_verifier_message(args.query, requirements, passages)
    print(f"    judge message: {len(message):,} chars,"
          f" {len(requirements)} requirement(s)")

    if args.batch_sizes:
        # Real pipeline: verify_passages fans out one call per batch through
        # the shared LLM pool and re-asks any passage the judge left
        # unscored, so this is the wall time a query actually pays.
        os.environ["VERIFIER_REASONING_EFFORT"] = args.effort
        sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]

        async def run_batches() -> list[dict]:
            rows: list[dict] = []
            for size in sizes:
                os.environ["VERIFIER_BATCH"] = str(size)
                buf = io.StringIO()
                started = time.perf_counter()
                with contextlib.redirect_stdout(buf):
                    res = await verify_passages(args.query, requirements,
                                                passages,
                                                ask_verbatim=False,
                                                label=f"b{size}")
                wall = time.perf_counter() - started
                line = next((ln.strip() for ln in buf.getvalue().splitlines()
                             if "judge usage" in ln), "")
                match = re.search(r"judge usage: (\d+) call", line)
                verdicts = res.evidence_results or []
                row = {
                    "batch": size,
                    "wall_s": round(wall, 2),
                    "calls": int(match.group(1)) if match else 0,
                    "prompt": res.prompt_tokens,
                    "cached": res.cached_tokens,
                    "completion": res.completion_tokens,
                    "reasoning": res.reasoning_tokens,
                    "json": max(0, res.completion_tokens - res.reasoning_tokens),
                    "tokens": res.tokens,
                    "kept": len(verdicts),
                    "passages": len(passages),
                    "verdicts": {v.passage_id: [round(v.intent_score, 2),
                                                sorted(v.coverage or [])]
                                 for v in verdicts},
                }
                rows.append(row)
                print(f"  [batch={size}] wall {row['wall_s']:.1f}s | "
                      f"{row['calls']} call(s) | "
                      f"prompt {row['prompt']:,} ({row['cached']:,} cached) | "
                      f"completion {row['completion']:,} "
                      f"(reasoning {row['reasoning']:,}, json {row['json']:,}) | "
                      f"tokens {row['tokens']:,} | "
                      f"kept {row['kept']}/{row['passages']}")
            return rows

        rows = asyncio.run(run_batches())
        print("\n== batch summary ==")
        print(f"    {'batch':>5} {'calls':>6} {'wall':>7} {'reasoning':>10} "
              f"{'tokens':>9} {'kept':>6}")
        for row in rows:
            print(f"    {row['batch']:>5} {row['calls']:>6} {row['wall_s']:>6.1f}s "
                  f"{row['reasoning']:>10,} {row['tokens']:>9,} "
                  f"{row['kept']:>3}/{row['passages']}")
        base = rows[0] if rows else None
        for row in rows[1:]:
            if base is None:
                break
            shared = set(base["verdicts"]) & set(row["verdicts"])
            agree = [p for p in shared
                     if base["verdicts"][p][1] == row["verdicts"][p][1]]
            print(f"    coverage-set agreement batch={row['batch']} vs "
                  f"batch={base['batch']}: {len(agree)}/{len(shared)}")
        if args.json:
            print(json.dumps({"query": args.query, "rows": rows},
                             ensure_ascii=False, indent=2))
        return 0

    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]

    async def run_all() -> list[dict]:
        rows: list[dict] = []
        for effort in efforts:
            for run in range(1, max(1, args.runs) + 1):
                row = await _arm(effort, message, requirements, passages)
                row["run"] = run
                rows.append(row)
                pct = (row["cached"] / row["prompt"] * 100) if row["prompt"] else 0
                tail = f" | ERROR {row['error']}" if row["error"] else ""
                print(f"  [{row['effort']} run{run}] wall {row['wall_s']:.1f}s | "
                      f"prompt {row['prompt']:,} ({pct:.0f}% cached) | "
                      f"completion {row['completion']:,} "
                      f"(reasoning {row['reasoning']:,}, json {row['json']:,}) | "
                      f"kept {row['kept']}/{row['passages']} | avg intent {row['avg_intent']}"
                      + tail)
        return rows

    rows = asyncio.run(run_all())

    # Compare warm runs (last run per effort) against the first arm.
    warm: list[dict] = []
    for effort in efforts:
        arm = [r for r in rows if r["effort"] == _label(effort)]
        if arm:
            warm.append(arm[-1])
    base = warm[0] if warm else None
    print("\n== summary (warm run per effort) ==")
    print(f"    {'effort':<9} {'wall':>7} {'reason':>8} {'json':>7} {'kept':>6}"
          f" {'avg_intent':>10}  delta_vs_base")
    for row in warm:
        delta = ""
        if base is not None and row is not base:
            dw = row["wall_s"] - base["wall_s"]
            dr = row["reasoning"] - base["reasoning"]
            dk = row["kept"] - base["kept"]
            delta = f"{dw:+.1f}s wall, {dr:+,} reasoning, {dk:+d} kept"
        print(f"    {row['effort']:<9} {row['wall_s']:>6.1f}s {row['reasoning']:>8,}"
              f" {row['json']:>7,} {row['kept']:>3}/{row['passages']:<2}"
              f" {row['avg_intent']:>10.3f}  {delta}")

    if base is not None:
        base_v = base.get("verdicts", {})
        for row in warm[1:]:
            arm_v = row.get("verdicts", {})
            shared = sorted(set(base_v) & set(arm_v))
            agree = [pid for pid in shared if base_v[pid][1] == arm_v[pid][1]]
            diffs = [pid for pid in shared if base_v[pid][1] != arm_v[pid][1]]
            print(f"    coverage-set agreement {row['effort']} vs "
                  f"{base['effort']}: {len(agree)}/{len(shared)} passages")
            for pid in diffs:
                print(f"      {pid}: {base['effort']} {base_v[pid][1]} "
                      f"vs {row['effort']} {arm_v[pid][1]}")

    if args.json:
        print(json.dumps({"query": args.query, "passages": len(passages),
                          "rows": rows}, ensure_ascii=False, indent=2))
    return 0 if not any(r["error"] for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
