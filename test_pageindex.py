#!/usr/bin/env python3
"""PageIndex + MedGemma controlled navigation experiment (V2.5).

    python test_pageindex.py --paper PMC11743609

Execution (Part 16):
    1. verify environment
    2. test /v1/models
    3. test /v1/chat/completions
    4. initialize PageIndex
    5. verify document (list_documents)
    6. verify tree (get_document_structure / get_tree)
    7. build H1 requirement
    8. run navigation
    9. print trace
   10. print selected nodes
   11. stop

/expand and /query are NOT invoked. No BM25 / pgvector / reranking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from medrag.retrieval_v2.pageindex_stage import (
    MEDGEMMA_ENDPOINT_ERROR,
    config_from_env,
    navigation_objective,
    navigation_trace,
    navigate,
    verify_document_registration,
    verify_medgemma_endpoint,
)

H1 = {
    "id": "H1",
    "target": "surgical repair techniques",
    "condition": "recurrent coarctation",
    "relationship": "associated with",
    "focus": "comparative_numerical",
    "requested_fields": ["percentage", "p-value"],
    "required_concepts": ["surgical repair techniques", "recurrent coarctation"],
    "preferred_evidence_types": ["table_row", "table_summary", "table_footnotes",
                                 "results", "paragraph"],
}


def _print_report(label: str, report: dict) -> None:
    print(f"[{label}]")
    print(json.dumps(report, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PageIndex + MedGemma navigation experiment")
    parser.add_argument("--paper", default="PMC11743609")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args(argv)

    overrides = {}
    if args.llm_base_url:
        overrides["llm_base_url"] = args.llm_base_url
    if args.chat_model:
        overrides["chat_model"] = args.chat_model
    if args.api_key is not None:
        overrides["llm_api_key"] = args.api_key
    cfg = config_from_env(overrides)

    print("=" * 70)
    print("PAGEINDEX + MEDGEMMA NAVIGATION EXPERIMENT")
    print("=" * 70)
    print("env:")
    print(json.dumps(cfg.as_dict(), indent=2))

    # 1. verify environment -------------------------------------------------
    print()
    print("[1] environment:")
    if not cfg.llm_base_url:
        print("    PAGEINDEX_LLM_BASE_URL: NOT SET")
    else:
        print(f"    PAGEINDEX_LLM_BASE_URL: {cfg.llm_base_url}")
    print(f"    PAGEINDEX_CHAT_MODEL:    {cfg.chat_model or '(not set)'}")
    print(f"    PAGEINDEX_LLM_API_KEY:   {'configured' if cfg.llm_api_key else '(none)'}")
    print(f"    PAGEINDEX_LLM_TEMPERATURE: {cfg.temperature}")
    print(f"    PAGEINDEX_MAX_TURNS:     {cfg.max_steps}")
    print(f"    PAGEINDEX_LLM_TIMEOUT:   {cfg.timeout}")

    # 2 + 3. MedGemma endpoint (Part 1) --------------------------------------
    print()
    print("[2+3] MedGemma endpoint (GET /v1/models, POST /v1/chat/completions):")
    endpoint_report = None
    try:
        endpoint_report = verify_medgemma_endpoint(cfg)
        _print_report("endpoint", endpoint_report)
    except Exception as exc:  # noqa: BLE001
        err = exc.to_dict() if hasattr(exc, "to_dict") else {
            "status": "failed", "error_type": MEDGEMMA_ENDPOINT_ERROR,
            "message": str(exc), "paper_id": args.paper}
        _print_report("endpoint", err)
        print()
        print("STOP (Part 1: endpoint verification failed - see error above).")
        print("Layer failed: MEDGEMMA_ENDPOINT_ERROR")
        return 2

    # 4. initialize PageIndex (Part 2) ----------------------------------------
    print()
    print("[4] PageIndex client initialization:")
    t0 = time.perf_counter()
    from pageindex.client import PageIndexClient
    backend = {"base_url": cfg.llm_base_url.rstrip("/"), "api_key": cfg.llm_api_key}
    client = PageIndexClient(storage_path=str(cfg.sdk_storage),
                             chat_model=cfg.chat_model or None,
                             chat_backend=backend)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"    client ok in {init_ms:.1f} ms; storage={cfg.sdk_storage}")

    # 5. document registration (Part 3) --------------------------------------
    print()
    print("[5] document registration:")
    try:
        ver = verify_document_registration(client, args.paper)
        _print_report("registration", ver)
    except Exception as exc:  # noqa: BLE001
        err = exc.to_dict() if hasattr(exc, "to_dict") else {"status": "failed",
                                                             "message": str(exc)}
        _print_report("registration", err)
        print("Layer failed:", err.get("error_type", "UNKNOWN"))
        return 3

    # 6. tree status (Part 3) -------------------------------------------------
    print()
    print("[6] native tree status:")
    try:
        tree = client.get_tree(args.paper, include_text=False)
        _print_report("tree", tree if isinstance(tree, dict) else {"raw": str(tree)[:500]})
    except Exception as exc:  # noqa: BLE001
        _print_report("tree", {"error": str(exc)})
        return 4

    # 7. H1 requirement (Part 4) ----------------------------------------------
    print()
    print("[7] requirement H1:")
    print(json.dumps(H1, indent=2))
    objective = navigation_objective(requirement=H1)
    print()
    print("objective:")
    print("    " + objective)

    # 8. navigation (Parts 6-9) ------------------------------------------------
    print()
    print("[8] navigation:")
    result = navigate(args.paper, objective=objective, requirement=H1,
                      config=cfg, endpoint_report=endpoint_report)

    # 9. trace (Part 8) -------------------------------------------------------
    print()
    print("[9] trace:")
    print(navigation_trace(result))

    # 10. selected nodes + latency (Part 9 / 13) --------------------------------
    print()
    print("[10] result:")
    print(json.dumps(result.to_dict(), indent=2))

    print()
    print("latency breakdown (ms):")
    print(f"    endpoint:  {result.endpoint_ms:.1f}")
    print(f"    init:      {result.init_ms:.1f}")
    print(f"    tree:      {result.tree_ms:.1f}")
    print(f"    agent:     {result.agent_ms:.1f}")
    print(f"    total:     {result.total_ms:.1f}")

    # 11. stop ----------------------------------------------------------------
    print()
    print("[11] done - no reranker / no BM25 / no pgvector used.")
    if result.status == "success":
        expected = {"Recurrent coarctation", "Table", "Surgical technique"}
        titles = " ".join(n.title for n in result.selected_nodes)
        found = [k for k in ("Recurrent coarctation", "Table", "Surgical technique")
                 if k.lower() in titles.lower()]
        print("semantic check (Part 11):", found if found else "NONE FOUND")
        print("experiment status:", "PASS" if found else "FAIL")
    else:
        print("experiment status: FAILED at", (result.error or {}).get("error_type", "?"))
    return 0 if result.status == "success" else 5


if __name__ == "__main__":
    raise SystemExit(main())
