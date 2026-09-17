"""Scratch probe: does the verifier verdict cache skip repeat judging?

Runs the REAL verify_passages pipeline three times over the same retrieved
passages in one process and prints calls/wall per repetition. Reps 2 and 3
should be near-instant with zero judge calls.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

from src.tools.retrieval import get_retriever  # noqa: E402
from src.tools.verifier import EvidenceRequirement, verify_passages  # noqa: E402

REQS = [
    ("E1", "Diagnostic thresholds, definitions, or classification"),
    ("E2", "Treatment, management, or intervention recommendations"),
    ("E3", "Epidemiology, prevalence, risk factors, or prognosis"),
]


async def main() -> int:
    query = (sys.argv[1] if len(sys.argv) > 1
             else "What are the diagnostic thresholds and first-line "
                  "treatment for hypertension?")
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    retriever = get_retriever()
    print("warmup (encoder / pgvector / BM25 / cross-encoder)...", flush=True)
    retriever.warmup(lambda stage: None)
    hits = retriever.search(query, 60, top_k)
    passages = [{"id": h.chunk_id, "text": h.text} for h in hits[:top_k]]
    print(f"{len(passages)} passage(s) retrieved", flush=True)
    if not passages:
        return 1
    reqs = [EvidenceRequirement(id=i, description=d) for i, d in REQS]
    for rep in (1, 2, 3):
        buf = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            res = await verify_passages(query, reqs, passages,
                                        ask_verbatim=False, label=f"rep{rep}")
        wall = time.perf_counter() - started
        line = next((ln.strip() for ln in buf.getvalue().splitlines()
                     if "judge usage" in ln), "")
        match = re.search(r"judge usage: (\d+) call", line)
        print(f"  rep{rep}: wall {wall:6.1f}s | calls "
              f"{match.group(1) if match else '?'} | kept "
              f"{len(res.evidence_results)}/{len(passages)} | completion "
              f"{res.completion_tokens:,} (reasoning {res.reasoning_tokens:,})",
              flush=True)
    return 0


raise SystemExit(asyncio.run(main()))
