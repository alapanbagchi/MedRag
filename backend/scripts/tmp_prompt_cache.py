#!/usr/bin/env python
"""TMP probe: does OpenCode Go cache the shared verifier prefix per session?

The planned verifier change fires one LLM call per passage, in parallel, all
sharing an identical prefix (system prompt + question + requirements +
scoring instructions) with only the passage tail varying. That is only cheap
if the gateway's per-session prompt cache actually hits on that prefix.

This script sends verifier-shaped judge calls in batches:

  batch 1  session A, N parallel calls   (cold — populates the cache)
  batch 2  session A, N parallel calls   (should HIT if caching works)
  batch 3  session B, N parallel calls   (control — should MISS)

and prints the raw ``usage`` object plus any ``*cache*`` fields so we can see
exactly what OpenCode Go reports (OpenAI-style ``prompt_tokens_details.
cached_tokens``, DeepSeek-style ``prompt_cache_hit_tokens``, Anthropic-style
``cache_read_input_tokens``, …).

Run:
    .venv/bin/python backend/scripts/tmp_prompt_cache.py
    .venv/bin/python backend/scripts/tmp_prompt_cache.py --calls 4 --parallel
    .venv/bin/python backend/scripts/tmp_prompt_cache.py --session <your-opencode-session-code>

Delete once the caching behaviour is confirmed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

PASSAGES = [
    "The 2017 ACC/AHA guideline defines hypertension as systolic blood pressure "
    "of 130 mmHg or higher, or diastolic blood pressure of 80 mmHg or higher, "
    "measured on two separate occasions in a clinical setting.",
    "ESC/ESH 2018 retains the 140/90 mmHg threshold for office blood pressure "
    "and reserves 130/80 mmHg for patients at high cardiovascular risk.",
    "The WHO 2021 guideline recommends pharmacological treatment at systolic "
    "blood pressure >= 140 mmHg or diastolic >= 90 mmHg for most adults.",
    "Home and ambulatory blood pressure monitoring use lower thresholds: 135/85 "
    "mmHg for daytime readings and 130/80 mmHg for 24-hour averages.",
]

# Fixed, identical filler in every call so the cacheable prefix is comfortably
# past the minimum the provider will cache. Replace with the real shared
# context once this probe is wired into the verifier.
CONTEXT_PAD = (
    "Clinical background (fixed reference context): hypertension is the most "
    "common modifiable cardiovascular risk factor worldwide. Diagnostic "
    "thresholds vary between guidelines and measurement modalities. Office "
    "measurement remains the reference standard for diagnosis, while home and "
    "ambulatory monitoring improve reproducibility and detect white-coat and "
    "masked hypertension. Treatment thresholds depend on global cardiovascular "
    "risk, comorbidity, and patient preference. "
) * 20


def _api_style(model: str, forced: str) -> str:
    forced = (forced or "auto").strip().lower()
    if forced in ("responses", "chat", "completions"):
        return "responses" if forced == "responses" else "chat"
    return "responses" if "muse-spark" in model.lower() else "chat"


def _walk_cache_fields(obj: object, prefix: str = "") -> dict:
    """Every key in the tree whose name mentions 'cache'."""
    found: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if "cache" in key.lower():
                found[path] = value
            found.update(_walk_cache_fields(value, path))
    return found


def _shared_body(question: str, pad: int) -> str:
    return (
        f"Question: {question}\n\n"
        "Evidence Requirements:\n\n"
        "E1: The systolic and diastolic blood pressure values used to define "
        "hypertension in adults.\n"
        "E2: The clinical practice guidelines that establish the diagnostic "
        "threshold.\n\n"
        + (CONTEXT_PAD * pad)
        + "\nScoring instructions: return one intent_score per passage plus the "
        "coverage array of requirement ids it meets, and a VERBATIM excerpt.\n"
    )


def _messages(shared: str, passage: str) -> list[dict]:
    system = (BACKEND / "src" / "prompts" / "evidence_judge.txt").read_text(
        encoding="utf-8"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{shared}\nPassage:\nP1: {passage}"},
    ]


def _verifier_messages() -> list[list[dict]]:
    """The exact per-passage messages the new verifier sends."""
    import sys

    sys.path.insert(0, str(BACKEND))
    from src.tools.verifier import EvidenceRequirement, build_verifier_message

    requirements = [
        EvidenceRequirement(id="E1", description="systolic/diastolic thresholds"),
        EvidenceRequirement(id="E2", description="guiding clinical guidelines"),
    ]
    system = (BACKEND / "src" / "prompts" / "evidence_judge.txt").read_text(
        encoding="utf-8"
    )
    return [
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": build_verifier_message(
                    "What blood pressure threshold defines hypertension in adult clinical practice?",
                    requirements,
                    [{"id": f"P{i + 1}", "text": text}],
                ),
            },
        ]
        for i, text in enumerate(PASSAGES)
    ]


async def _one_call(
    *,
    client: httpx.AsyncClient,
    base: str,
    key: str,
    model: str,
    style: str,
    session: str,
    messages: list[dict],
    tag: str,
    sem: asyncio.Semaphore,
) -> dict:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "x-opencode-session": session,
    }
    if style == "responses":
        url = f"{base}/responses"
        body = {
            "model": model,
            "input": messages,
            "max_output_tokens": 700,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        url = f"{base}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": 700,
            "temperature": 0.0,
            "stream": False,
        }

    async with sem:
        started = time.perf_counter()
        try:
            resp = await client.post(url, headers=headers, json=body)
        except Exception as exc:  # noqa: BLE001 — probe prints, never raises
            print(f"  {tag}: request error: {exc}")
            return {"tag": tag, "status": 0, "secs": 0.0, "usage": {}, "cache": {}}
        secs = time.perf_counter() - started

    try:
        data = resp.json()
    except ValueError:
        data = {"_text": resp.text[:300]}
    usage = data.get("usage") or (data.get("response") or {}).get("usage") or {}
    cache = _walk_cache_fields(usage)
    print(
        f"  {tag}: HTTP {resp.status_code} {secs:6.2f}s "
        f"usage={json.dumps(usage, ensure_ascii=False)}"
    )
    if resp.status_code >= 400:
        print(f"    error body: {json.dumps(data, ensure_ascii=False)[:400]}")
    return {"tag": tag, "status": resp.status_code, "secs": secs, "usage": usage, "cache": cache}


async def _batch(
    label: str,
    *,
    client: httpx.AsyncClient,
    base: str,
    key: str,
    model: str,
    style: str,
    session: str,
    shared: str,
    calls: int,
    sem: asyncio.Semaphore,
) -> list[dict]:
    print(f"\n== {label}  (session {session[:12]}…) ==")
    tasks = [
        _one_call(
            client=client,
            base=base,
            key=key,
            model=model,
            style=style,
            session=session,
            messages=_messages(shared, PASSAGES[i % len(PASSAGES)]),
            tag=f"{label} #{i + 1}",
            sem=sem,
        )
        for i in range(calls)
    ]
    return await asyncio.gather(*tasks)


def _summarize(batches: dict[str, list[dict]]) -> None:
    print("\n================ summary ================")
    for label, results in batches.items():
        ok = [r for r in results if r["status"] == 200]
        if not ok:
            print(f"{label}: no successful calls")
            continue
        avg = sum(r["secs"] for r in ok) / len(ok)
        cache = {}
        for r in ok:
            for path, value in r["cache"].items():
                cache.setdefault(path, []).append(value)
        print(f"{label}: {len(ok)} ok, avg {avg:.2f}s, cache fields: {json.dumps(cache)}")
    print(
        "\nIf batch 2 (warm, same session) reports cache hits and batch 3 "
        "(other session) does not, per-session prompt caching works."
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("SMALL_MODEL", ""))
    parser.add_argument(
        "--style",
        default=os.environ.get("OPENAI_API_STYLE", "auto"),
        help="auto | chat | responses",
    )
    parser.add_argument("--calls", type=int, default=3, help="calls per batch")
    parser.add_argument("--pad", type=int, default=2, help="fixed-context multiplier")
    parser.add_argument(
        "--verifier",
        action="store_true",
        help="use the real per-passage verifier messages (one per passage)",
    )
    parser.add_argument(
        "--session",
        default=os.environ.get("LLM_SESSION_ID", "")
        or os.environ.get("OPENCODE_SESSION", ""),
        help="opencode session code; default is two fresh random ones",
    )
    args = parser.parse_args()

    if not args.base or not args.key or not args.model:
        print("LLM_BASE_URL / LLM_API_KEY / SMALL_MODEL missing (check .env)")
        return 2

    base = args.base.rstrip("/")
    style = _api_style(args.model, args.style)
    shared = _shared_body(
        "What blood pressure threshold defines hypertension in adult clinical practice?",
        args.pad,
    )
    session_a = args.session or uuid.uuid4().hex
    session_b = uuid.uuid4().hex

    print(f"endpoint : {base}")
    print(f"model    : {args.model}  (style={style})")
    print(f"prefix   : {len(shared)} chars fixed + system prompt "
          f"{len((BACKEND / 'src/prompts/evidence_judge.txt').read_text())} chars")
    print(f"session A: {session_a}")
    print(f"session B: {session_b}")

    sem = asyncio.Semaphore(max(1, args.calls))
    timeout = httpx.Timeout(connect=30.0, read=240.0, write=60.0, pool=240.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.verifier:
            # Real verifier shape: one message per passage, fired in parallel
            # under one session; a second identical batch should all-hit.
            from src.llm.models import session_id

            session = args.session or session_id()
            messages = _verifier_messages()
            print(f"\n[verifier mode] {len(messages)} per-passage messages,"
                  f" session {session[:16]}…")

            async def _run(tag: str):
                return await asyncio.gather(*[
                    _one_call(
                        client=client, base=base, key=args.key, model=args.model,
                        style=style, session=session, messages=msg,
                        tag=f"{tag} #{i + 1}", sem=sem,
                    )
                    for i, msg in enumerate(messages)
                ])

            batches = {
                "verifier cold": await _run("verifier cold"),
                "verifier warm": await _run("verifier warm"),
            }
            _summarize(batches)
            return 0
        batches = {
            "batch1 cold A": await _batch(
                "batch1 cold A", client=client, base=base, key=args.key,
                model=args.model, style=style, session=session_a, shared=shared,
                calls=args.calls, sem=sem,
            ),
            "batch2 warm A": await _batch(
                "batch2 warm A", client=client, base=base, key=args.key,
                model=args.model, style=style, session=session_a, shared=shared,
                calls=args.calls, sem=sem,
            ),
            "batch3 other B": await _batch(
                "batch3 other B", client=client, base=base, key=args.key,
                model=args.model, style=style, session=session_b, shared=shared,
                calls=args.calls, sem=sem,
            ),
        }
    _summarize(batches)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
