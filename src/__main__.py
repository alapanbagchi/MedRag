"""Entry point: python -m src 'query' [--agentic|--agentic-v2] [--local|--remote]"""

from __future__ import annotations

import asyncio
import json
import sys


def _model_label(cfg) -> str:
    """Display label for the model of the ACTIVE provider (not the default)."""
    label = {
        "kaggle": cfg.kaggle_model, "openai": cfg.openai_model,
        "ollama": cfg.ollama_model, "vllm": cfg.vllm_model,
        "gemini": cfg.gemini_model, "mistral": cfg.mistral_model,
    }.get(cfg.provider, "")
    return label or cfg.provider


def _print_agentic_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"AGENTIC RAG — question_type={result['question_type']} | "
          f"{result['succeeded_subqueries']}/{result['num_subqueries']} subqueries succeeded")
    print("=" * 70)
    for s in result["subqueries"]:
        status = "OK" if s["succeeded"] else ("partial" if s.get("citations") else "none")
        print(f"\n[{status}] {s['id']}: {s['target']}")
        print(f"   intent: {s.get('intent') or '-'}")
        print(f"   summary: {s['summary']}")
        if s.get("searches"):
            print(f"   searches: {len(s['searches'])}")
        if s.get("citations"):
            print(f"   citations: {', '.join(s['citations'])}")
    if result.get("evidence_excerpts"):
        print("\n--- EVIDENCE EXCERPTS ---")
        for e in result["evidence_excerpts"][:10]:
            print("  -", " ".join(e.split())[:200])


async def _agentic_main(argv: list[str]) -> None:
    import os

    from src.config import AppConfig
    from src.trace import get_trace
    from src.agentic.pipeline import AgenticPipeline

    query = " ".join(a for a in argv if not a.startswith("-"))
    cfg = AppConfig()

    # Open the incremental trace stream ("logs.txt" by default, LOG_FILE to
    # override) so every funnel stage is written live while the loop runs.
    log_path = os.environ.get("LOG_FILE", "logs.txt")
    trace = get_trace()
    trace.open_stream(log_path, query=query or "(none)")

    pipeline = AgenticPipeline(config=cfg)
    print(f"[mode] agentic | local_mode={cfg.local_mode} | dense={cfg.enable_dense} | model={_model_label(cfg)}")
    print(f"       logging to {log_path} incrementally")
    try:
        result = await pipeline.answer(query)
        _print_agentic_result(result)
        # also dump raw JSON for machine consumption
        print("\n--- RAW JSON ---")
        print(json.dumps(result, indent=2, default=str))
    finally:
        trace.close_stream()


def _print_agentic_v2_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"AGENTIC V2 — orchestrator loop | iterations={result['iterations']} | "
          f"terminal={result['terminal']} | stop_reason={result['stop_reason']}")
    print("=" * 70)
    for o in result["objectives"]:
        print(f"\n[{o['status']}] {o['id']}: {o['statement']}")
        if o.get("gap"):
            print(f"   gap: {o['gap']}")
        if o.get("caveats"):
            print(f"   caveats: {len(o['caveats'])}")
    print("\n--- ACTIONS ---")
    for a in result["actions"]:
        print(f"  #{a['iteration']} {a['action']}"
              f"{('[' + a['objective_id'] + ']') if a['objective_id'] else ''}"
              f" [{a['status']}] {a['outcome'][:110]}")
    if result.get("answer") and result["answer"].get("summary"):
        print("\n--- ANSWER ---")
        print(result["answer"]["summary"])
        for s in result["answer"].get("sections", []):
            print(f"\n## {s['heading']}")
            print(s["body"][:800])
    if result.get("gaps"):
        print("\n--- UNRESOLVED GAPS ---")
        for g in result["gaps"]:
            print("  -", g)
    if result.get("contradictions"):
        print("\n--- CONTRADICTIONS ---")
        for c in result["contradictions"]:
            print("  -", c)


async def _agentic_v2_main(argv: list[str]) -> None:
    import os

    from src.config import AppConfig
    from src.trace import get_trace
    from src.agentic_v2.pipeline import AgenticV2Pipeline

    query = " ".join(a for a in argv if not a.startswith("-"))
    cfg = AppConfig()

    log_path = os.environ.get("LOG_FILE", "logs.txt")
    trace = get_trace()
    trace.open_stream(log_path, query=query or "(none)")

    pipeline = AgenticV2Pipeline(config=cfg)
    print(f"[mode] agentic-v2 | local_mode={cfg.local_mode} | dense={cfg.enable_dense} | model={_model_label(cfg)}")
    print(f"       logging to {log_path} incrementally | max_rounds={cfg.agentic_v2_max_rounds}")
    try:
        result = await pipeline.answer(query)
        _print_agentic_v2_result(result)
        print("\n--- RAW JSON ---")
        print(json.dumps(result, indent=2, default=str))
    finally:
        trace.close_stream()


def _print_agentic_v3_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"AGENTIC V3 - master -> parallel workers -> contradiction -> answer | "
          f"terminal={result['terminal']} | stop={result['stop_reason']}")
    print("=" * 70)
    for w in result.get("workers", []):
        print(f"\n[WORKER {w.get('task_id')}] {w.get('task_title')} -> {w.get('status')} "
              f"(searches={w.get('searches_used')}, deep={w.get('deep_inspections_used')})")
        for rq in w.get("requirements", []):
            print(f"   req {rq.get('requirement_id')} [{rq.get('status')}] "
                  f"coverage={rq.get('coverage')}/{rq.get('target_n')}"
                  f"{(' | gap: ' + rq['gap'][:160]) if rq.get('gap') else ''}")
            for p in rq.get("papers", []):
                print(f"      - {p.get('document_id')} [{p.get('support')}] "
                      f"conf={p.get('confidence'):.2f} src={p.get('source')}")
    if result.get("evidence"):
        print(f"\n--- VERIFIED EVIDENCE ({len(result['evidence'])}) ---")
        for e in result["evidence"][:12]:
            print(f"  - {e['id']} {e['document_id']} [{e['support']}] "
                  f"{' '.join((e.get('excerpt') or '').split())[:160]}")
    if result.get("contradictions"):
        print("\n--- CONTRADICTIONS ---")
        for c in result["contradictions"]:
            res = c.get("resolution") or {}
            print(f"  - {c['id']} [{c.get('kind')}] {c.get('claim', '')[:140]}")
            print(f"      resolution: {res.get('status', 'none')} - "
                  f"{res.get('explanation', '')[:180]}")
    if result.get("gaps"):
        print("\n--- EVIDENCE GAPS ---")
        for g in result["gaps"]:
            print("  -", g)
    if result.get("unresolved"):
        print("\n--- UNRESOLVED CONTRADICTIONS ---")
        for cid in result["unresolved"]:
            print("  -", cid)
    if result.get("answer") and result["answer"].get("summary"):
        print("\n--- ANSWER ---")
        print(result["answer"]["summary"])
        for s in result["answer"].get("sections", []):
            print(f"\n## {s['heading']}")
            print(s["body"][:900])
        if result["answer"].get("limitations"):
            print("\nLimitations:")
            for lim in result["answer"]["limitations"]:
                print("  -", lim[:200])


async def _agentic_v3_main(argv: list[str]) -> None:
    import os

    from src.config import AppConfig
    from src.trace import get_trace
    from src.agentic_v3.pipeline import AgenticV3Pipeline

    query = " ".join(a for a in argv if not a.startswith("-"))
    cfg = AppConfig()

    log_path = os.environ.get("LOG_FILE", "logs.txt")
    trace = get_trace()
    trace.open_stream(log_path, query=query or "(none)")

    pipeline = AgenticV3Pipeline(config=cfg)
    print(f"[mode] agentic-v3 | local_mode={cfg.local_mode} | dense={cfg.enable_dense} | model={_model_label(cfg)}")
    print(f"       logging to {log_path} incrementally | evidence_target={cfg.agentic_v3_evidence_target}")
    try:
        result = await pipeline.answer(query)
        _print_agentic_v3_result(result)
        print("\n--- RAW JSON ---")
        print(json.dumps(result, indent=2, default=str))
    finally:
        trace.close_stream()


async def main(argv: list[str] | None = None):
    from src import logfire_obs as lf

    argv = sys.argv[1:] if argv is None else argv
    lf.ensure_configured()
    try:
        if "--agentic-v3" in argv:
            await _agentic_v3_main([a for a in argv if a != "--agentic-v3"])
            return
        if "--agentic-v2" in argv:
            await _agentic_v2_main([a for a in argv if a != "--agentic-v2"])
            return
        if "--agentic" in argv:
            await _agentic_main([a for a in argv if a != "--agentic"])
            return
        from src.pipeline import main as legacy_main
        await legacy_main(argv)
    finally:
        # push buffered telemetry to Logfire before the process exits
        lf.flush()


if __name__ == "__main__":
    asyncio.run(main())
