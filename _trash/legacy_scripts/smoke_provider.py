#!/usr/bin/env python3
"""Smoke test / full-run against the configured provider.

With ``LLM_PROVIDER=gemini`` and a Google AI Studio key this verifies the key,
then runs the full pipeline. Demos the provider-swap (Kaggle -> Gemini) with
zero code changes.

Usage:
    uv run python scripts/smoke_provider.py            # tiny completion only
    uv run python scripts/smoke_provider.py --full     # tiny completion + full query
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROBE_QUERY = "What is a cardiac arrest?"


async def smoke(cfg) -> bool:
    from app.llm.kaggle import build_raw_client

    print(f"[smoke] provider={cfg.provider}")
    base_url = cfg.gemini_base_url if cfg.provider == "gemini" else cfg.kaggle_base_url
    api_key = cfg.gemini_api_key if cfg.provider == "gemini" else cfg.kaggle_api_key
    model = cfg.gemini_model if cfg.provider == "gemini" else cfg.kaggle_model
    print(f"[smoke] base_url={base_url}")
    print(f"[smoke] model={model} | api_key='{api_key[:6]}...'")
    print(f"[smoke] agent_max_tokens={cfg.agent_max_tokens}")

    if cfg.provider == "gemini" and not cfg.gemini_api_key:
        print("[smoke] ERROR: set GEMINI_API_KEY in .env (get one at "
              "https://aistudio.google.com/apikey)")
        return False

    client = build_raw_client(cfg)
    try:
        await asyncio.wait_for(
            client.chat("You are a health check.", "Reply with exactly: ok",
                        max_tokens=8, temperature=0.0),
            timeout=30,
        )
        print("[smoke] OK — chain works: local client -> provider -> model")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return False
    finally:
        await client.aclose()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Provider smoke test")
    parser.add_argument("--full", action="store_true", help="also run a full query")
    args = parser.parse_args()

    from app.config import AppConfig
    from app.logging_setup import setup_logging

    setup_logging()
    cfg = AppConfig()

    if not await smoke(cfg):
        return 2

    if args.full:
        from app.pipeline import answer

        print(f"\n[run ] full query: {PROBE_QUERY}\n")
        result = await answer(PROBE_QUERY)
        print("\n--- answer ---")
        print(result.answer.summary or "(empty summary)")
        for section in result.answer.sections:
            print(f"  [{section.heading}] {section.body[:160]}")
        print("\n--- pipeline ---")
        print("verification:", result.verification.overall_status,
              f"({len(result.verification.verdicts)} verdicts)")
        print("evidence:", len(result.evidence), "items | groups:", len(result.groups))
        print("timings (ms):", result.timings_ms)
        print("warnings:", result.warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))