"""Run a FULL query end-to-end against the live Kaggle/OpenAI-compatible
endpoint, waiting until the tunnel is healthy, and report every agent
outcome from the trace (parses, retries, failures)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERY = "What is a cardiac arrest?"


async def wait_healthy(cfg, attempts: int = 30, delay: float = 10.0) -> bool:
    """Provider-agnostic health probe: a tiny chat completion (works for any
    OpenAI-compatible endpoint, incl. Gemini's compat layer and loca.lt)."""
    from app.llm.kaggle import build_raw_client

    client = build_raw_client(cfg)
    try:
        for i in range(1, attempts + 1):
            try:
                content, usage = await client.chat(
                    "You are a health check.", "Reply with exactly: ok",
                    max_tokens=8, temperature=0.0,
                )
                if content.strip():
                    print(f"[wait] endpoint healthy after {i * delay / 60:.1f} min: "
                          f"{cfg.kaggle_base_url} -> {content.strip()[:40]!r}")
                    return True
            except Exception as exc:  # noqa: BLE001
                print(f"[wait] attempt {i}: {type(exc).__name__}: {str(exc)[:120]}")
            await asyncio.sleep(delay)
        return False
    finally:
        await client.aclose()


async def main() -> int:
    from app.config import AppConfig
    from app.logging_setup import setup_logging
    from app.pipeline import answer

    setup_logging()
    cfg = AppConfig()
    print(f"[run ] provider={cfg.provider} base_url={cfg.kaggle_base_url}")
    print(f"[run ] api_key='{cfg.kaggle_api_key[:6]}...' model={cfg.kaggle_model}")

    if not await wait_healthy(cfg):
        print("[run ] ENDPOINT UNHEALTHY — giving up (tunnel down / restarting)")
        return 2

    print(f"\n[run ] query: {QUERY}\n")
    t0 = time.time()
    result = await answer(QUERY)
    print(f"\n[run ] total {time.time() - t0:.1f}s")

    print("\n--- summary ---")
    print("answer summary:", (result.answer.summary or "")[:200])
    print("sections:", len(result.answer.sections), "| citations:",
          sum(len(s.citations) for s in result.answer.sections))
    print("verification:", result.verification.overall_status,
          f"({len(result.verification.verdicts)} verdicts)")
    print("evidence:", len(result.evidence), "items | groups:", len(result.groups))
    print("timings:", result.timings_ms)
    print("warnings:", result.warnings)

    # Agent outcome counts straight from logs.txt (parse-fails / retries / errors).
    try:
        text = Path(os.environ.get("LOG_FILE", "logs.txt")).read_text()
        print("\n--- per-agent diagnostics (from logs.txt) ---")
        print("agent blocks      :", text.count("-- AGENT #"))
        print("failed parses     :", text.count("first structured parse failed"))
        print("compaction retries:", text.count("with a compact-JSON nudge"))
        print("retry success     :", text.count("model_response[2]"))
    except FileNotFoundError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))