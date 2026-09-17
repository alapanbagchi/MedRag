"""Manual Firecrawl check: exercises the REAL agent web-search tool in isolation.

Usage (from backend/):
    make firecrawl-test FC_QUERY="latest hypertension guidelines" FC_LIMIT=5
    make firecrawl-test FC_QUERY="..." FC_LIMIT=3 FULL=1   # print page texts
    .venv/bin/python scripts/firecrawl_check.py "query" --limit 5 --full

Pipeline (exactly what the deep agents run): v2 search -> MyBib
credibility gate (score >= 4) -> v2 batch scrape with 48h cache reuse.
Exit 0 when end-to-end evidence flows (>= 1 usable page), 1 otherwise
(make-friendly): a dead key, an empty search, or zero usable pages all
fail loudly with the verbatim backend error.

Reads FIRECRAWL_URL / FIRECRAWL_API_KEY from backend/.env (same file
the API server uses).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_dotenv() -> None:
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent / ".env", Path.cwd() / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip(chr(39)).strip(chr(34))
        break


def _short(text, n=200):
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= n else flat[:n] + "..."


async def _main(query, limit, full):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.tools import firecrawl as fc
    from src.middleware import mybib as mybib_mod

    print("endpoint: " + fc._base_url())
    print("api key:  " + ("present" if fc._api_key() else "MISSING"))
    print("mybib floor: %d" % mybib_mod.min_score())
    print("query: " + query)
    print("-" * 60)

    try:
        raw = await fc.web_search(
            SimpleNamespace(deps=None), query, limit=limit)
    except Exception as exc:
        print("TOOL RAISED: %s: %s" % (type(exc).__name__, exc))
        return 1
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        print("UNPARSEABLE TOOL OUTPUT:")
        print(str(raw)[:500])
        return 1
    if not data.get("available"):
        print("available: False")
        print("error: %s" % (data.get("error") or "(no error detail)"))
        return 1

    listings = data.get("results") or []
    print("available: True  listings: %d  dropped by gate: %s" % (
        len(listings), data.get("dropped", "?")))
    for i, r in enumerate(listings, 1):
        print("  [%d] %s" % (i, str(r.get("title") or r.get("url"))[:90]))
        print("       %s" % r.get("url", "(no url)"))
    verdicts = {v.get("url"): v for v in (data.get("trust") or [])
                if isinstance(v, dict)}
    if verdicts:
        print("gate verdicts:")
        for url, v in verdicts.items():
            mark = "keep" if v.get("trustworthy") else "drop"
            print("  [%s] %s %s -- %s" % (
                mark, v.get("category", "?"), str(url)[:60],
                _short(v.get("reason"), 100)))
    pages = data.get("pages") or []
    usable = [p for p in pages
              if isinstance(p, dict) and (p.get("text") or p.get("markdown"))]
    print("scrape: %d/%d usable page(s)" % (len(usable), len(pages)))
    for p in pages:
        text = " ".join(str(p.get("text") or p.get("markdown") or "").split())
        if text:
            print("  [ok] %s  chars=%d" % (p.get("url", "")[:70], len(text)))
            if full:
                print("  " + "-" * 56)
                print("  " + text)
        else:
            print("  [FAIL] %s  error=%s" % (
                p.get("url", "")[:70], _short(p.get("error"), 160)))
    if not usable:
        print("NO USABLE PAGES -- web evidence is not flowing")
        return 1
    print("OK: web evidence is flowing end to end")
    return 0


if __name__ == "__main__":
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Manual Firecrawl check")
    parser.add_argument("query", help="web-search query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--full", action="store_true",
                        help="print full scraped page texts")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(
        " ".join(args.query.split()),
        max(1, min(args.limit, 20)),
        full=args.full,
    )))
