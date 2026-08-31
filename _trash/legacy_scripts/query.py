"""End-to-end query: takes a question, builds a plan via LLM, executes retrieval, saves output.

Usage:
    python scripts/query.py "What is the treatment for heart failure?"
    python scripts/query.py "diabetes management" --output my_results/run1
    KAGGLE_URL=https://... python scripts/query.py "heart failure treatment"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def build_plan_from_query(
    query: str,
    kaggle_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the /expand endpoint to build a retrieval plan from a query."""
    import requests

    url = (kaggle_url or os.environ.get("KAGGLE_URL", "")).rstrip("/")
    if not url:
        raise ValueError(
            "KAGGLE_URL not set. Set the environment variable or pass --kaggle-url.\n"
            "This is the URL of your MedRAG server (medrag/server.py) running on Kaggle or locally."
        )

    expand_url = f"{url}/expand"
    print(f"Calling {expand_url} ...")
    resp = requests.post(expand_url, json={"query": query}, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    # ── Intent ──────────────────────────────────────────────────
    intent = data.get("intent", {})

    # ── Evidence requirements ───────────────────────────────────
    # Prefer what /expand explicitly returns; else derive per-condition.
    evidence_reqs = data.get("evidence_requirements") or intent.get("required_evidence") or []
    if not evidence_reqs:
        target = intent.get("target", "cardiovascular vulnerability")
        conditions = intent.get("conditions") or []
        if isinstance(conditions, str):
            conditions = [conditions]
        for cond in conditions:
            if cond:
                evidence_reqs.append(f"Mechanisms by which {cond} increases {target}")
                evidence_reqs.append(f"Clinical outcomes associated with {cond} and {target}")
        # Dedupe preserving order
        seen = set()
        evidence_reqs = [r for r in evidence_reqs if not (r.lower() in seen or seen.add(r.lower()))]
        if not evidence_reqs:
            evidence_reqs = [f"{intent.get('target', 'medical')} information"]

    # ── Search queries ──────────────────────────────────────────
    # Prefer /expand's search queries; else derive from conditions.
    search_queries = data.get("search_queries") or []
    if not search_queries:
        search_queries = [data.get("expanded_query", query)]
        target = intent.get("target", "cardiovascular vulnerability")
        conditions = intent.get("conditions") or []
        if isinstance(conditions, str):
            conditions = [conditions]
        for cond in conditions:
            if cond:
                search_queries.append(f"{cond} {target} mechanisms and outcomes")
        search_queries.append(query)
        # Dedupe preserving order
        seen = set()
        search_queries = [q for q in search_queries if not (q.lower() in seen or seen.add(q.lower()))]

    plan = {
        "query": query,
        "expanded_query": data.get("expanded_query", query),
        "search_queries": search_queries,
        "concepts": data.get("concepts", []),
        "bioportal": data.get("bioportal", []),
        "evidence_requirements": evidence_reqs,
        "intent": intent,
    }

    print(f"Expanded query: {plan['expanded_query'][:120]}")
    print(f"Search queries: {len(search_queries)}")
    for i, sq in enumerate(search_queries, 1):
        print(f"  {i}. {sq[:110]}")
    print(f"Evidence requirements: {len(evidence_reqs)}")
    for i, req in enumerate(evidence_reqs, 1):
        print(f"  R{i}: {req}")

    return plan


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Medical question to search for")
    parser.add_argument("--kaggle-url", default=None, help="URL of the MedRAG /expand server")
    parser.add_argument("--output", type=Path, default=None, help="Output base path (no extension)")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--final-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument("--no-expansion", action="store_true", help="Disable context expansion")
    parser.add_argument("--plan-only", action="store_true", help="Just print the plan, don't execute")
    args = parser.parse_args(argv)

    # Step 1: Build plan from query via LLM
    print("=" * 70)
    print(f"QUERY: {args.query}")
    print("=" * 70)

    plan = build_plan_from_query(args.query, kaggle_url=args.kaggle_url)

    if args.plan_only:
        print("\nPlan:")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    # Step 2: Execute the orchestrator
    from medrag.retrieval.orchestrator import run_orchestration

    output_path = args.output
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe = args.query[:50].replace(" ", "_").replace("/", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "_-")[:40]
        output_path = Path(f"retrieval_runs/{timestamp}_{safe}")

    print(f"\nExecuting retrieval...")
    result = run_orchestration(
        plan=plan,
        index_dir=args.index_dir,
        final_top_k=args.final_top_k,
        rerank_top_k=args.rerank_top_k,
        output_path=output_path,
        enable_context_expansion=not args.no_expansion,
    )

    # Step 3: Print where output was saved
    md_path = output_path.with_suffix(".md")
    log_path = output_path.with_suffix(".log")
    print(f"\n{'=' * 70}")
    print(f"OUTPUT FILES:")
    print(f"  Expanded evidence: {md_path}")
    print(f"  Execution trace:   {log_path}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
