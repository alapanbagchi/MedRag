#!/usr/bin/env python3
"""Full-log demo of the local PydanticAI pipeline.

Because the Kaggle quick-tunnel URL rotates/expires (and the one from your
last session is currently not resolving), this demo boots the SAME thin
inference server (`kaggle/inference_server.py`) on localhost with a scripted
runtime that returns schema-correct JSON per agent. Every other component is
real:

  - PydanticAI agents (planner, evidence, verifier, synthesizer)
  - the exact HTTP path a live tunnel would use (system+user -> response)
  - the local BM25 retrieval index (index/corpus.parquet + index/bm25)
  - the deterministic aggregator and the orchestrator

Run:  uv run python scripts/demo_pipeline.py
The run also tees the complete log to retrieval_runs/full_trace.log.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# repo root on sys.path so `app` and `kaggle` resolve when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["ENABLE_DENSE"] = "false"  # force BM25-only demo (no encoder load)

from kaggle.inference_server import LLMRuntime  # noqa: E402


# ----------------------------------------------------------------------
# Scripted runtime: mimics the MedGemma model by returning the right JSON
# for each agent's prompt. Detection is by the SYSTEM prompt the agent uses.
# ----------------------------------------------------------------------

DEMO_QUERY = (
    "In patients undergoing surgical repair of coarctation of the aorta, which "
    "repair techniques were associated with early recurrent coarctation, what "
    "percentages and p-values were reported for those associations, and how did "
    "the imaging findings in the same study characterize the recurrent lesions "
    "or anatomical changes?"
)

PLAN_RESPONSE = {
    "original_query": DEMO_QUERY,
    "question_type": "comparative_numerical",
    "clinical_entities": [
        {"surface_form": "coarctation of the aorta", "base_concept": "coarctation of the aorta", "modifiers": [], "role": "condition"},
        {"surface_form": "surgical repair", "base_concept": "surgical repair", "modifiers": [], "role": "procedure"},
        {"surface_form": "early recurrent coarctation", "base_concept": "recurrent coarctation", "modifiers": ["early"], "role": "condition"},
    ],
    "query_targets": ["repair techniques", "associations", "imaging findings"],
    "requested_fields": ["percentage", "p-value"],
    "relationships": ["associated with"],
    "populations": ["patients undergoing surgical repair of coarctation of the aorta"],
    "search_concepts": ["coarctation of the aorta", "surgical repair", "recurrent coarctation", "repair technique"],
    "subqueries": [
        {
            "id": "H1",
            "question": "repair techniques associated with recurrent coarctation percentages p-values imaging findings",
            "objective": "find repair techniques associated with early recurrent coarctation, reported percentages and p-values, and imaging characterization",
            "evidence_required": ["percentages and p-values per repair technique", "imaging findings characterizing recurrent lesions"],
            "target": "repair techniques",
            "condition": "recurrent coarctation",
            "relationship": "associated with",
            "focus": "comparative_numerical",
            "requested_fields": ["percentage", "p-value"],
            "required_concepts": ["coarctation of the aorta", "surgical repair"],
            "preferred_evidence_types": ["table_row", "table_summary", "table_footnotes", "results", "paragraph"],
            "populations": ["patients undergoing surgical repair of coarctation of the aorta"],
        }
    ],
}


def _first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        s = s.strip()
        if len(s) >= 20:
            return s
    return text[:400]


class ScriptedRuntime(LLMRuntime):
    model_id = "medgemma"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        sp = system_prompt.lower()

        if "query planner" in sp or "subqueries" in sp and "clinical_entities" in sp:
            return json.dumps(PLAN_RESPONSE)

        if "evidence-extraction agent" in sp:
            return self._evidence(user_prompt)

        if "verification agent" in sp:
            return self._verifier(user_prompt)

        if "final answer synthesizer" in sp:
            return self._synthesizer(user_prompt)

        return json.dumps(PLAN_RESPONSE)

    # ------------------------------------------------------------------
    def _evidence(self, user_prompt: str) -> str:
        m = re.search(r"<document>\n(.*?)\n</document>", user_prompt, re.DOTALL)
        doc_text = m.group(1) if m else user_prompt
        quote = _first_sentence(doc_text)
        claim = quote[:160]
        return json.dumps(
            {
                "document_id": "",
                "subquery_id": "",
                "items": [
                    {
                        "claim": claim,
                        "supporting_text": quote,
                        "supports_claim": True,
                        "confidence": 0.9,
                        "evidence_type": "paragraph",
                        "section": "",
                        "subsection": "",
                        "breadcrumb": [],
                        "table_id": None,
                        "figure_id": None,
                        "page_start": None,
                        "page_end": None,
                        "is_inference": False,
                        "contradiction_note": "",
                    }
                ],
            }
        )

    @staticmethod
    def _verifier(user_prompt: str) -> str:
        verdicts = []
        blocks = re.split(r"GROUP (G\d+)", user_prompt)
        # blocks: [pre, id1, body1, id2, body2, ...]
        for i in range(1, len(blocks) - 1, 2):
            group_id = blocks[i]
            body = blocks[i + 1]
            claim_m = re.search(r"^claim:\s*(.*)$", body, re.MULTILINE)
            claim = claim_m.group(1).strip() if claim_m else ""
            ids = re.findall(r"\[(E-[^\]]+)\]", body)
            verdicts.append(
                {
                    "claim": claim,
                    "group_id": group_id,
                    "status": "supported",
                    "reasons": ["quote directly supports the claim (scripted demo)"],
                    "problems": [],
                    "verified_evidence_ids": ids,
                    "confidence": 0.9,
                }
            )
        return json.dumps(
            {
                "verdicts": verdicts,
                "overall_status": "verified" if verdicts else "unverified",
            }
        )

    @staticmethod
    def _synthesizer(user_prompt: str) -> str:
        ids = re.findall(r"\[(E-[^\]]+)\]", user_prompt)
        quotes = re.findall(r"quote: (.*)", user_prompt)
        citations = []
        seen = set()
        for eid, quote in zip(ids, quotes):
            if eid in seen:
                continue
            seen.add(eid)
            m = re.search(re.escape(eid) + r"\] subquery=(\S+) document=(\S+) chunk=(\S+) section=(\S+)", user_prompt)
            citations.append(
                {
                    "subquery_id": m.group(1) if m else "H1",
                    "document_id": m.group(2) if m else "",
                    "chunk_id": m.group(3) if m else None,
                    "section": m.group(4) if m else "",
                    "evidence_id": eid,
                    "quote": quote[:120],
                }
            )
        summary = "End-to-end anastomosis was associated with recurrent coarctation (23.1%, p=0.04) — see sections." if citations else "No verified evidence."
        return json.dumps(
            {
                "summary": summary,
                "sections": [
                    {
                        "heading": "Repair techniques and reported statistics",
                        "body": "The verified evidence reports recurrence percentages and p-values for the repair techniques (scripted demo).",
                        "citations": citations,
                    }
                ],
                "limitations": ["Scripted demo: numbers are illustrative, not from the corpus."],
                "coverage_note": "H1 covered by verified evidence.",
            }
        )


# ----------------------------------------------------------------------
# Boot server + run pipeline with full logging
# ----------------------------------------------------------------------


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main() -> int:
    from app.config import AppConfig
    from app.logging_setup import setup_logging, stage_banner
    from app.pipeline import answer

    setup_logging()

    port = _free_port()
    runtime = ScriptedRuntime()
    import kaggle.inference_server as server

    server._runtime = runtime
    os.environ["KAGGLE_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["KAGGLE_API_KEY"] = "dummy"
    os.environ["KAGGLE_MODEL"] = "medgemma"

    import uvicorn

    uvicorn_config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(uvicorn_config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    try:
        import time

        for _ in range(100):
            if uvicorn_server.started:
                break
            time.sleep(0.05)
        if not uvicorn_server.started:
            raise RuntimeError("local inference server did not start")

        cfg = AppConfig()
        # Hermetic: force the LOCAL thin server (the .env provider URL is
        # only used when no explicit override is given).
        cfg.kaggle_base_url = f"http://127.0.0.1:{port}/v1"
        cfg.kaggle_api_key = "dummy"
        cfg.kaggle_model = "medgemma"
        print()
        print("[demo] local OpenAI-compatible endpoint : ", cfg.kaggle_base_url)
        print("[demo] retrieval                      : BM25 index (index/bm25 + corpus.parquet)")
        print()

        result = asyncio.run(answer(DEMO_QUERY))

        stage_banner("FINAL ANSWER")
        print(result.answer.summary)
        for section in result.answer.sections:
            print(f"\n## {section.heading}")
            print(section.body)
            for citation in section.citations:
                print(
                    f"  [cit] {citation.evidence_id} {citation.document_id} "
                    f"{citation.chunk_id or ''} ({citation.section})"
                )
        print()
        print(f"[demo] Prettified full run log: {os.environ.get('LOG_FILE', 'logs.txt')}")
        return 0
    finally:
        server._runtime = None
        uvicorn_server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())