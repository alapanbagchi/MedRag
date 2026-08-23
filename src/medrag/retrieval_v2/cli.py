"""Command-line entry for MedRAG Retrieval V2.

Usage:
    python -m medrag.retrieval_v2 "question..." [--output FILE]
    python -m medrag.retrieval_v2 --plan plan.json
    python -m medrag.retrieval_v2 --benchmark   # known multi-hop regression
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


BENCHMARK_QUESTION = (
    "How do metabolic syndrome, advanced CKD, and anemia affect cardiovascular "
    "vulnerability, and how does this differ across INOCA, non-diabetic CKD, "
    "and elderly hip-fracture patients?"
)

# Spec section 41 regression targets: requirement id -> expected paper
BENCHMARK_BRANCH_PAPERS: Dict[str, List[str]] = {
    "H1": ["PMC11743015"],
    "H4": ["PMC11743015"],
    "H2": ["PMC11705662"],
    "H5": ["PMC11705662"],
    "H3": ["PMC11694428"],
    "H6": ["PMC11694428"],
}

def _print_run(result: Dict[str, Any], verbose: bool = False) -> None:
    coverage = result.get("coverage", {})
    metrics = result.get("metrics", {})
    print()
    print("=" * 72)
    print("MEDRAG RETRIEVAL V2 - RUN SUMMARY")
    print("=" * 72)
    print("Question: " + str(result.get("query", "")))
    print()
    plan = result.get("plan", {})
    n_reqs = len(plan.get("requirements", []))
    n_qs = len(plan.get("queries", []))
    qtype = plan.get("question_type", "")
    print(f"Requirements: {n_reqs} | Queries: {n_qs} | Type: {qtype}")
    print()
    print("Selected papers:")
    for p in result.get("papers", []):
        flag = " [branch guarantee]" if p.get("guarantee") else ""
        pid = p["paper_id"]
        score = p["paper_score"]
        reqs = ",".join(p.get("supported_requirements", []))
        reason = p.get("reason", "")
        print(f"  {pid:<12} score={score:.3f}  reqs={reqs}  {reason}{flag}")
    print()
    print("Coverage:")
    cov_covered = coverage.get("covered")
    cov_uncovered = coverage.get("uncovered")
    print(f"  covered:   {cov_covered}")
    print(f"  uncovered: {cov_uncovered}")
    pie = result.get("pageindex")
    if pie:
        print()
        print("PageIndex (paper-local navigation):")
        print(f"  library: {pie.get('library')} | artifacts loaded: {pie.get('artifacts_loaded')}")
        per = pie.get("per_paper", {})
        print(f"  per-paper status: {per}")
    metrics2 = result.get("metrics", {})
    if metrics2.get("pageindex_evidence_count") is not None:
        print(f"  final evidence with PageIndex provenance: {metrics2.get('pageindex_evidence_count')} | with table context: {metrics2.get('table_context_evidence_count')}")
    frac = coverage.get("coverage_fraction", 0)
    allc = coverage.get("all_requirements_covered")
    print(f"  fraction:  {frac:.3f} | all covered: {allc}")
    print()
    if metrics:
        print("Metrics:")
        fr = metrics.get("final_paper_recall")
        bp = metrics.get("branch_papers_preserved_in_selection")
        print(f"  final_paper_recall: {fr}")
        print(f"  branch_papers_preserved_in_selection: {bp}")
        kpf = metrics.get("known_papers_found")
        kpm = metrics.get("known_papers_missing")
        rr = metrics.get("requirement_recall")
        if kpf:
            print(f"  known papers found: {kpf}")
            print(f"  known papers missing: {kpm}")
        print(f"  requirement_recall: {rr}")
        pqs = metrics.get("per_query_paper_metrics", {})
        for qid, m in pqs.items():
            mrr = m.get("mrr")
            rc5 = m.get("recall@5")
            known = m.get("known_papers")
            found = m.get("found_in_ranking")
            print(f"    {qid}: mrr={mrr:.3f} recall@5={rc5:.2f}")
            print(f"        known={known} found={found}")
    print()
    timings = result.get("timings", {})
    tstr = " | ".join(f"{k}={v}" for k, v in timings.items())
    print("Timings (ms): " + tstr)
    print("=" * 72)

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MedRAG Retrieval V2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("question", nargs="?", default=None, help="Medical question")
    parser.add_argument("--plan", type=Path, default=None, help="V2 plan JSON (skips planning)")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--pageindex-dir", type=Path, default=None, help="PageIndex artifact directory")
    parser.add_argument("--output", type=Path, default=None, help="Output base path")
    parser.add_argument("--no-rerank", action="store_true", help="Disable MedCPT cross-encoder")
    parser.add_argument("--no-dense", action="store_true", help="Disable dense retrieval")
    parser.add_argument("--top-papers", type=int, default=None)
    parser.add_argument("--final-slots", type=int, default=None)
    parser.add_argument("--llm-planning", choices=["off", "enrich", "decompose"], default=None,
                        help="LLM planning mode: off=deterministic, enrich=LLM fills concepts, decompose=LLM decomposes question")
    parser.add_argument("--llm-base-url", default=None, help="LLM endpoint URL (e.g. http://localhost:11434)")
    parser.add_argument("--llm-model", default=None, help="LLM model name (e.g. deepseek, llama3)")
    parser.add_argument("--benchmark", action="store_true", help="Run the known multi-hop regression question")
    args = parser.parse_args(argv)

    from medrag.retrieval_v2.config import config_from_env
    from medrag.retrieval_v2.pipeline import load_components, run_v2_pipeline

    overrides: Dict[str, Any] = {}
    if args.no_rerank:
        overrides["enable_rerank"] = False
    if args.no_dense:
        overrides["enable_dense"] = False
    if args.top_papers is not None:
        overrides["top_papers"] = args.top_papers
    if args.final_slots is not None:
        overrides["final_slots"] = args.final_slots
    if args.llm_planning is not None:
        overrides["llm_planning"] = args.llm_planning
    if args.llm_base_url is not None:
        overrides["llm_base_url"] = args.llm_base_url
    if args.llm_model is not None:
        overrides["llm_model"] = args.llm_model
    if args.pageindex_dir is not None:
        overrides["pageindex_dir"] = str(args.pageindex_dir)
    cfg = config_from_env(overrides)

    question = args.question
    if args.benchmark:
        question = BENCHMARK_QUESTION
    if question is None:
        parser.error("a question is required (or use --benchmark)")

    plan_dict = None
    if args.plan is not None:
        plan_dict = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    components = load_components(args.index_dir, cfg)
    result = run_v2_pipeline(
        question, plan=plan_dict, config=cfg, index_dir=args.index_dir,
        components=components, output_path=args.output,
        known_branch_papers=BENCHMARK_BRANCH_PAPERS if args.benchmark else None,
    )
    _print_run(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
