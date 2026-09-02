"""Entry point: python -m src "query" (agentic-v3 pipeline)."""

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
    mem = result.get("memory") or {}
    if mem.get("enabled"):
        stats = mem.get("committed") or {}
        print("\n--- MEMORY ---")
        print(f"session={mem.get('session_id')} title={mem.get('session_title') or '(new)'!r}")
        if stats:
            print(f"committed: {stats.get('claims_committed', 0)} claims, "
                  f"{stats.get('evidence_refs', 0)} evidence refs, "
                  f"{stats.get('contradictions', 0)} contradictions, "
                  f"{stats.get('gaps', 0)} gaps, "
                  f"{stats.get('questions', 0)} questions")


async def _agentic_v3_main(argv: list[str]) -> None:
    import os

    from src.config import AppConfig
    from src.lib.trace import get_trace
    from src.agentic.pipeline import AgenticV3Pipeline

    enable_memory = any(a == "--memory" for a in argv) or \
        os.environ.get("MEMORY_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
    query = " ".join(a for a in argv if not a.startswith("-"))
    cfg = AppConfig()

    log_path = os.environ.get("LOG_FILE", "logs.txt")
    trace = get_trace()
    trace.open_stream(log_path, query=query or "(none)")

    memory = None
    if enable_memory:
        from src.memory import MemoryAPI
        from src.memory.config import MemoryConfig
        memory = MemoryAPI.build(MemoryConfig.from_appconfig(cfg))
        print(f"[memory] enabled | backend={memory.config.backend} "
              f"embedder={memory.config.embedder}")

    pipeline = AgenticV3Pipeline(config=cfg, memory=memory)
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
    from src.lib import logfire_obs as lf

    argv = sys.argv[1:] if argv is None else argv
    lf.ensure_configured()
    try:
        await _agentic_v3_main(argv)
    finally:
        # push buffered telemetry to Logfire before the process exits
        lf.flush()


if __name__ == "__main__":
    asyncio.run(main())