#!/usr/bin/env python3
"""Standalone PageIndex stage test (V2.4) - required test script.

    1. load one real Markdown paper
    2. build/load its PageIndex tree
    3. print the tree
    4. issue the surgical-repair / recurrent-CoA navigation query
    5. print the navigation trace
    6. print selected nodes
    7. stop

Run:
    python test_pageindex.py [--paper PMC11743609] [--rebuild] [--llm-base-url ...] [--chat-model ...]

The full RAG pipeline is NOT run (no global retrieval, no reranking).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from medrag.retrieval_v2.pageindex_stage import (
    StageConfig,
    build_index,
    config_from_env,
    navigation_objective,
    navigation_trace,
    navigate,
    print_tree,
    scan_corpus,
    validate_markdown,
)

REQUIREMENT = {
    "target": "surgical repair techniques",
    "condition": "recurrent coarctation",
    "relationships": ["surgical repair techniques associated with recurrent coarctation"],
    "requested_fields": ["percentage", "p-value"],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PageIndex stages test (validation / indexing / navigation)")
    parser.add_argument("--paper", default="PMC11743609")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--chat-model", default=None)
    args = parser.parse_args(argv)

    overrides = {"rebuild": args.rebuild}
    if args.llm_base_url:
        overrides["llm_base_url"] = args.llm_base_url
    if args.chat_model:
        overrides["chat_model"] = args.chat_model
    cfg = config_from_env(overrides)

    print("=" * 70)
    print("STEP 1 - EXISTING MARKDOWN CORPUS VALIDATION")
    print("=" * 70)
    print("config:", json.dumps(cfg.as_dict(), indent=2))
    docs, summary = scan_corpus(cfg)
    print("corpus:", json.dumps(summary, indent=2))

    md_path = Path(cfg.md_dir) / f"{args.paper}.md"
    if not md_path.exists():
        print(json.dumps({
            "status": "failed", "error_type": "markdown_missing",
            "message": f"no markdown at {md_path} (run jats_convert first)",
            "paper_id": args.paper,
        }, indent=2))
        return 1
    doc = validate_markdown(args.paper, md_path)
    print(f"validation {args.paper}: headings={doc.heading_count} depth={doc.heading_depth} "
          f"tables={doc.table_count} captions={doc.caption_count} footnotes={doc.footnote_count} malformed={doc.malformed_headings}")
    print("heading hierarchy:")
    for crumb in doc.hierarchy:
        indent = "    " + "  " * (len(crumb) - 1)
        print(f"{indent}- {' > '.join(crumb[len(crumb) - 2:]) if len(crumb) > 1 else crumb[-1]}")
    if summary.get("n_markdown_files", 0) == 0:
        print(json.dumps({"status": "failed", "error_type": "corpus_empty",
                          "message": "no markdown papers found", "paper_id": args.paper}))
        return 1

    print()
    print("=" * 70)
    print("STEP 2 - PAGEINDEX INDEXING (Markdown -> native tree, heuristic)")
    print("=" * 70)
    indexed = build_index(args.paper, config=cfg, rebuild=args.rebuild)
    print(json.dumps({
        "paper_id": args.paper,
        "cached": indexed["cached"],
        "indexing_ms": indexed["latency_ms"],
        "node_count": indexed["node_count"],
        "tree_size_bytes": indexed["meta"].get("tree_size_bytes"),
        "builder": indexed["meta"].get("builder"),
        "sdk_store": indexed.get("sdk", {}),
    }, indent=2))
    print("native PageIndex tree (titles + node ids):")
    print_tree(indexed["structure"])

    print()
    print("=" * 70)
    print("STEP 3 - PAGEINDEX QUERY-TIME NAVIGATION")
    print("=" * 70)
    objective = navigation_objective(requirement=REQUIREMENT)
    print("objective:")
    print("    " + objective)
    print()
    result = navigate(args.paper, objective=objective, requirement=REQUIREMENT, config=cfg)
    print(navigation_trace(result))
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
