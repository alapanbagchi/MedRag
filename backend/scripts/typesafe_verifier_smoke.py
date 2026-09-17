"""Smoke test for the TypeSafe evidence verifier.

Default (offline): drives the real TypeSafe SDK client through a mock HTTP
transport, so request building, response decoding, and the verifier mapping
are exercised end to end with no network or API key.

Live: pass --live to send one real System One request. Requires
TYPESAFE_API_KEY in the environment or in backend/.env.

Usage:
    .venv/bin/python scripts/typesafe_verifier_smoke.py
    .venv/bin/python scripts/typesafe_verifier_smoke.py --live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.verifier import EvidenceRequirement, verify_passages

QUESTION = ("What is the definition of hypertension and what are its "
            "diagnostic criteria?")
REQUIREMENTS = [
    EvidenceRequirement(id="E1", description="definition of hypertension"),
    EvidenceRequirement(id="E2", description="diagnostic criteria/thresholds"),
]
PASSAGES = [
    {"id": "P1", "text": "Hypertension is persistently elevated arterial "
                         "blood pressure."},
    {"id": "P2", "text": "It is diagnosed when resting blood pressure is at "
                         "or above 140/90 mmHg on repeated measurement."},
    {"id": "P3", "text": "The Eiffel Tower is located in Paris, France."},
]
_COVERED = {"P1": {"E1"}, "P2": {"E2"}, "P3": set()}


def _mock_transport():
    """A transport that answers each System One request deterministically."""
    import httpx2

    class MockTransport(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request):
            body = json.loads(request.content)
            pid = body["state"]["passage"]["id"]
            covered = _COVERED.get(pid, set())
            answers = {}
            for key, question in body["questions"].items():
                if question["type"] == "noul":
                    if key == "intent":
                        value = 0.0 if pid == "P3" else 0.9
                    else:
                        rid = key.split("::", 1)[1]
                        value = 0.95 if rid in covered else 0.05
                    answers[key] = {"type": "noul", "noul": value}
                elif question["type"] == "choice":
                    answers[key] = {"type": "choice", "choice": "s0",
                                    "confidence": 0.9,
                                    "probabilities": {"s0": 0.9, "none": 0.1}}
            payload = {"model": "jev-1.13.0",
                       "usage": {"input_tokens": 120, "output_tokens": 8},
                       "answers": answers}
            return httpx2.Response(
                200, headers={"content-type": "application/json"},
                content=json.dumps(payload).encode())

    return MockTransport()


async def _run(live: bool) -> int:
    from typesafe_sdk import AsyncTypeSafeClient

    if live:
        if not (os.environ.get("TYPESAFE_API_KEY") or
                (Path(__file__).resolve().parent.parent / ".env").is_file()):
            print("TYPESAFE_API_KEY is not set (env or backend/.env)")
            return 2
        client = AsyncTypeSafeClient(
            model=os.environ.get("TYPESAFE_MODEL", "jev-latest"))
    else:
        client = AsyncTypeSafeClient(api_key="offline-smoke",
                                     model="jev-latest",
                                     transport=_mock_transport())
    try:
        report = await verify_passages(QUESTION, REQUIREMENTS, PASSAGES,
                                       client=client, label="smoke")
    finally:
        await client.aclose()

    print("judged:", report.judged)
    print("input_tokens:", report.prompt_tokens)
    for verdict in report.evidence_results:
        print(f"{verdict.passage_id}: intent={verdict.intent_score:.2f} "
              f"coverage={verdict.coverage} verbatim={verdict.verbatim!r}")
    kept = {v.passage_id for v in report.evidence_results}
    if not live:
        assert report.judged is True, "judge did not run"
        assert kept == {"P1", "P2"}, f"unexpected kept set: {kept}"
        assert "P3" not in kept, "unrelated passage must be rejected"
    print("OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="send a real request (needs TYPESAFE_API_KEY)")
    args = parser.parse_args()
    return asyncio.run(_run(args.live))


if __name__ == "__main__":
    raise SystemExit(main())
