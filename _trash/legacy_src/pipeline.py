"""Main pipeline entry point."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from src.config import AppConfig


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Third-party INFO chatter (httpx request lines, faiss's AVX2-fallback
    # notice, ...) stays out of the terminal unless VERBOSE=1.
    if os.getenv("VERBOSE", "").strip().lower() in {"1", "true", "yes"}:
        return
    for noisy in ("httpx", "httpcore", "openai", "urllib3",
                  "faiss", "faiss.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list) -> tuple[str, AppConfig]:
    """Parse CLI args. Supports --local/--remote flags + a query."""
    setup_logging()

    local_mode = None  # None = use env default
    query_parts = []
    for arg in argv:
        if arg == "--local":
            local_mode = True
        elif arg == "--remote":
            local_mode = False
        else:
            query_parts.append(arg)

    query = " ".join(query_parts).strip() or "What is a cardiac arrest?"
    cfg = AppConfig()
    if local_mode is not None:
        cfg.local_mode = local_mode
        cfg.vector_db_url = "" if local_mode else cfg.vector_db_url
    # Propagate to env so tool-side AppConfig() (e.g. hybrid_search) sees it too.
    import os
    os.environ["LOCAL_MODE"] = "true" if cfg.local_mode else ""
    os.environ["VECTOR_DB_URL"] = cfg.vector_db_url
    print(f"[mode] local_mode={cfg.local_mode} | dense={cfg.enable_dense} | index={cfg.index_dir}")
    print(f"       logging to {os.environ.get('LOG_FILE', 'logs.txt')} incrementally")
    return query, cfg


async def answer(query: str, config: AppConfig | None = None) -> dict:
    from src.orchestration.orchestrator import Orchestrator

    cfg = config or AppConfig()
    orchestrator = Orchestrator(cfg)
    return await orchestrator.answer(query)


def print_result(result: dict):
    plan = result["plan"]
    answer_obj = result["answer"]
    timings = result["timings"]

    print("\n" + "=" * 70)
    print("PLAN")
    print("=" * 70)
    print(f"  question_type: {plan.question_type}")
    print(f"  entities: {[e.text for e in plan.entities]}")
    print(f"  targets: {plan.targets}")
    for sub in plan.subqueries:
        print(f"  {sub.id}: {sub.target} (focus={sub.focus})")

    warnings = result.get("warnings") or []
    if warnings:
        print("\n" + "=" * 70)
        print("WARNINGS")
        print("=" * 70)
        for w in warnings:
            print(f"  ! {w}")

    funnel = result.get("funnel") or {}
    if funnel:
        print("\n" + "=" * 70)
        print("FUNNEL (attrition per stage)")
        print("=" * 70)
        for key, value in funnel.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer_obj.summary)
    for section in answer_obj.sections:
        if section.heading:
            print(f"\n## {section.heading}")
        print(section.body)
        for c in section.citations:
            print(f"  [cit] {c.document_id} {c.chunk_id} {c.section}")

    if answer_obj.limitations:
        print("\nLIMITATIONS")
        for lim in answer_obj.limitations:
            print(f"  - {lim}")

    print(f"\nTIMINGS: {timings}")


async def main(argv: list | None = None):
    import os

    from src.trace import get_trace

    argv = argv if argv is not None else sys.argv[1:]
    query, cfg = parse_args(argv)

    log_path = os.environ.get("LOG_FILE", "logs.txt")
    trace = get_trace()
    trace.open_stream(log_path, query=query)
    try:
        result = await answer(query, cfg)
        print_result(result)
    finally:
        trace.close_stream()


if __name__ == "__main__":
    asyncio.run(main())