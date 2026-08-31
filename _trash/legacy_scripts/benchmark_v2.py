#!/usr/bin/env python3
"""MedRAG Retrieval V2 regression benchmark (spec section 41).

Runs the pipeline on the known multi-hop question and verifies that the
three critical branch papers survive paper selection:

    q4 (metabolic syndrome + INOCA outcomes)    -> PMC11743015 (INOCA paper)
    q5 (non-diabetic CKD outcomes)              -> PMC11705662 (CKD paper)
    q6 (anemia + elderly hip fracture outcomes) -> PMC11694428 (hip fracture)

The historical failure: these papers WERE retrieved by their branches but
disappeared during global RRF / giant-query reranking. V2 must preserve them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medrag.retrieval_v2.cli import BENCHMARK_QUESTION, BENCHMARK_BRANCH_PAPERS
from medrag.retrieval_v2.config import config_from_env
from medrag.retrieval_v2.pipeline import load_components, run_v2_pipeline


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="V2 regression benchmark")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--top-papers", type=int, default=None)
    parser.add_argument("--final-slots", type=int, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Report JSON path")
    args = parser.parse_args(argv)

    overrides: Dict = {}
    if args.no_rerank:
        overrides["enable_rerank"] = False
    if args.no_dense:
        overrides["enable_dense"] = False
    if args.top_papers is not None:
        overrides["top_papers"] = args.top_papers
    if args.final_slots is not None:
        overrides["final_slots"] = args.final_slots
    cfg = config_from_env(overrides)

    print("=" * 72)
    print("MEDRAG RETRIEVAL V2 - REGRESSION BENCHMARK")
    print("=" * 72)
    print("Question: " + BENCHMARK_QUESTION)
    print("Branch targets:")
    for k, v in BENCHMARK_BRANCH_PAPERS.items():
        print(f"  {k}: {v}")
    print()

    components = load_components(args.index_dir, cfg)
    try:
        result = run_v2_pipeline(
            BENCHMARK_QUESTION,
            config=cfg,
            index_dir=args.index_dir,
            components=components,
            known_branch_papers=BENCHMARK_BRANCH_PAPERS,
        )
    finally:
        components.close()

    metrics = result.get("metrics", {})
    coverage = result.get("coverage", {})

    print()
    print("-" * 72)
    print("REGRESSION CHECKS")
    print("-" * 72)
    found = metrics.get("known_papers_found", [])
    missing = metrics.get("known_papers_missing", [])
    print(f"  known papers found in selected neighborhood: {found}")
    print(f"  known papers missing: {missing}")
    checks_ok = True
    for qid, papers in [("H1", ["PMC11743015"]), ("H2", ["PMC11705662"]),
                        ("H3", ["PMC11694428"]), ("H4", ["PMC11743015"]),
                        ("H5", ["PMC11705662"]), ("H6", ["PMC11694428"])]:
        ok = all(p in found for p in papers)
        checks_ok = checks_ok and ok
        print(f"  {qid}: {papers} preserved -> " + ("OK" if ok else "FAIL"))
    print()
    fpr = metrics.get("final_paper_recall")
    bps = metrics.get("branch_papers_preserved_in_selection")
    rr = metrics.get("requirement_recall")
    print(f"  final_paper_recall: {fpr}")
    print(f"  branch_papers_preserved_in_selection: {bps}")
    print(f"  requirement_recall: {rr}")
    cov_covered = coverage.get("covered")
    cov_uncovered = coverage.get("uncovered")
    print(f"  coverage covered: {cov_covered}")
    print(f"  coverage uncovered: {cov_uncovered}")
    print()
    pqs = metrics.get("per_query_paper_metrics", {})
    print("  Per-query paper metrics:")
    for k, v in pqs.items():
        mrr = v.get("mrr")
        rc5 = v.get("recall@5")
        known = v.get("known_papers")
        fin = v.get("found_in_ranking")
        print(f"    {k}: mrr={mrr:.3f} recall@5={rc5:.2f} known={known} found={fin}")

    report = {
        "question": BENCHMARK_QUESTION,
        "branch_targets": BENCHMARK_BRANCH_PAPERS,
        "metrics": metrics,
        "coverage": coverage,
        "selected_papers": [p.get("paper_id") for p in result.get("papers", [])],
        "regression_passed": bool(checks_ok and bps),
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print("Report: " + str(out))

    print()
    print("REGRESSION " + ("PASSED" if report["regression_passed"] else "FAILED"))
    return 0 if report["regression_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

