#!/usr/bin/env python3
"""
Run full MedRAG retrieval for a question.

Optionally calls the /expand endpoint to get a retrieval plan with
search queries, evidence requirements, and intent. Falls back to
using the raw question if no expand URL is provided.

Usage:
    # With /expand (recommended):
    python -m medrag.retrieval.run \
        --question "What is the optimal LDL threshold for statin therapy?" \
        --expand-url https://ballet-lots-feel-achieved.trycloudflare.com

    # Without /expand (uses raw question):
    python -m medrag.retrieval.run \
        --question "What is the optimal LDL threshold for statin therapy?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


def get_plan_from_expand(
    question: str,
    expand_url: str,
    timeout: int = 120,
) -> dict:
    """Call the /expand endpoint to get a retrieval plan."""
    import requests

    url = expand_url.rstrip("/") + "/expand"
    print(f"Calling {url} ...")
    t0 = time.perf_counter()

    try:
        resp = requests.post(
            url,
            json={"query": question},
            timeout=timeout,
        )
        resp.raise_for_status()
        plan = resp.json()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach /expand at {url}. Is the tunnel running?"
        ) from exc
    except requests.Timeout as exc:
        raise TimeoutError(f"/expand timed out after {timeout}s") from exc
    except requests.HTTPError as exc:
        raise RuntimeError(f"/expand returned HTTP {resp.status_code}: {resp.text[:300]}") from exc

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"/expand returned in {elapsed:.0f} ms")
    print(f"  Expanded query: {plan.get('expanded_query', '')[:100]}...")
    print(f"  Search queries: {len(plan.get('search_queries', []))}")
    print(f"  Evidence requirements: {len(plan.get('evidence_requirements', []))}")

    return plan


def build_fallback_plan(question: str) -> dict:
    """
    Build a minimal retrieval plan from just the question.
    Used when no /expand endpoint is available.
    """
    print("No /expand URL provided — using raw question as single search query.")
    print("For better results, provide --expand-url for LLM query expansion.")

    return {
        "query": question,
        "expanded_query": question,
        "search_queries": [question],
        "concepts": [],
        "umls": [],
        "expanded_terms": [],
        "intent": {
            "task": "information_retrieval",
            "target": question,
            "conditions": [],
        },
        "evidence_requirements": [
            f"Direct evidence addressing: {question}",
        ],
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full MedRAG retrieval: question → [optional /expand] → orchestrator → results",
    )
    parser.add_argument(
        "--question", "-q", required=True,
        help="The medical question to retrieve evidence for",
    )
    parser.add_argument(
        "--expand-url", "-u", type=str, default=None,
        help="Base URL of the /expand endpoint (optional — if omitted, uses raw question)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON path (default: auto-generated in retrieval_runs/)",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=None,
        help="Path to index directory",
    )
    parser.add_argument(
        "--final-top-k", type=int, default=20,
        help="Final passages to select (default: 20)",
    )
    parser.add_argument(
        "--rerank-top-k", type=int, default=100,
        help="Candidates after reranking (default: 100)",
    )
    parser.add_argument(
        "--candidate-pool", type=int, default=300,
        help="Candidate pool size (default: 300)",
    )
    parser.add_argument(
        "--bm25-depth", type=int, default=50,
        help="BM25 depth per query (default: 50)",
    )
    parser.add_argument(
        "--dense-depth", type=int, default=50,
        help="Dense depth per query (default: 50)",
    )
    parser.add_argument(
        "--req-threshold", type=float, default=0.50,
        help="Evidence coverage threshold (default: 0.50)",
    )

    args = parser.parse_args(argv)

    # ── Step 1: Get plan ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: GETTING RETRIEVAL PLAN")
    print("=" * 70)

    if args.expand_url:
        plan = get_plan_from_expand(args.question, args.expand_url)
    else:
        plan = build_fallback_plan(args.question)

    # Ensure the original query is set
    plan["query"] = args.question

    # ── Step 2: Run orchestrator ─────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: RUNNING RETRIEVAL ORCHESTRATOR")
    print("=" * 70)

    from medrag.retrieval.orchestrator import run_orchestration

    result = run_orchestration(
        plan=plan,
        index_dir=args.index_dir,
        bm25_depth=args.bm25_depth,
        dense_depth=args.dense_depth,
        candidate_pool=args.candidate_pool,
        rerank_top_k=args.rerank_top_k,
        final_top_k=args.final_top_k,
        req_threshold=args.req_threshold,
        output_path=args.output,
    )

    # ── Step 3: Print concise summary ────────────────────────────
    print("\n" + "=" * 70)
    print("EVIDENCE SUMMARY")
    print("=" * 70)

    evidence_requirements = plan.get("evidence_requirements", [])
    for i, req in enumerate(evidence_requirements):
        status = "COVERED" if result["diagnostics"]["evidence_coverage"].get(f"requirement_{i+1}", False) else "MISSING"
        n = result["diagnostics"]["per_requirement_coverage"].get(f"requirement_{i+1}", 0)
        print(f"  [{status:>7}] R{i+1}: {req[:90]}... ({n} passages)")

    print(f"\n  All evidence covered: {result['diagnostics']['all_evidence_covered']}")
    print(f"  Final passages: {result['diagnostics']['final_count']}")
    print(f"  Unique documents: {result['diagnostics']['document_count']}")
    print(f"  Mean coverage: {result['diagnostics']['coverage_distribution']['mean_coverage_fraction']:.2f}")

    if args.output:
        print(f"\n  Full results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
