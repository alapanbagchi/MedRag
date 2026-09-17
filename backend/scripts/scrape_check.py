"""Standalone scrape check: exercises the REAL scrape path in isolation.

Usage (from backend/):
    make scrape URL=https://pmc.ncbi.nlm.nih.gov/articles/PMC11975635/
    make scrape URLS="https://example.com https://example.org"
    .venv/bin/python scripts/scrape_check.py <url> [<url> ...]

One URL goes through src.tools.firecrawl.scrape; several go through
src.tools.firecrawl.scrape_many (the batch job the agent uses), so a
multi-URL run tests the real agent path. Prints a compact report plus
the ENTIRE markdown per URL. Exit 0 when every URL yielded usable
markdown, 1 otherwise (make-friendly).

Reads FIRECRAWL_URL / FIRECRAWL_API_KEY from backend/.env (same file
the API server uses); FIRECRAWL_URL defaults to http://localhost:3002.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


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
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value
        break


def _report(url: str, title: str, text: str, ok: bool, error: str = "") -> None:
    print(f"url:  {url}")
    print(f"success: {ok}")
    print(f"title: {title or '(none)'}")
    print(f"chars: {len(text)}")
    if not ok:
        print(f"error: {error or '(unknown)'}")
    print("-" * 60)
    print(text or "(no text)")


async def _scrape_one(fc, target: str) -> bool:
    try:
        row = await fc.scrape(target)
    except Exception as exc:  # noqa: BLE001 — report, don't traceback
        _report(target, "", "", False, str(exc))
        return False
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    text = " ".join(str(data.get("markdown") or "").split())
    ok = bool(row.get("success")) and len(text) >= 40
    _report(target, str(metadata.get("title") or ""), text, ok,
            str(row.get("error", "")))
    return ok


async def _scrape_many(fc, targets: list[str]) -> bool:
    try:
        rows = await fc.scrape_many(targets)
    except Exception as exc:  # noqa: BLE001 — report, don't traceback
        for target in targets:
            _report(target, "", "", False, str(exc))
        return False
    ok_all = True
    for target, row in zip(targets, rows):
        text = " ".join(str(row.get("text") or row.get("markdown") or "").split())
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        nested = row.get("data") if isinstance(row.get("data"), dict) else {}
        nested_meta = nested.get("metadata") if isinstance(nested.get("metadata"), dict) else {}
        title = str(meta.get("title") or nested_meta.get("title") or "")
        ok = row.get("success") is not False and len(text) >= 40
        ok_all = ok_all and ok
        _report(target, title, text, ok, str(row.get("error", "")))
    return ok_all


async def _main(urls: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.tools import firecrawl as fc

    targets = [u if "://" in u else f"https://{u}" for u in urls]
    print(f"base: {os.environ.get('FIRECRAWL_URL', fc.DEFAULT_BASE_URL)}")
    print(f"urls: {len(targets)}")
    print("=" * 60)
    if len(targets) == 1:
        return 0 if await _scrape_one(fc, targets[0]) else 1
    return 0 if await _scrape_many(fc, targets) else 1


if __name__ == "__main__":
    _load_dotenv()
    urls = [a.strip() for a in sys.argv[1:] if a.strip()]
    if not urls:
        print(__doc__.strip().splitlines()[2].strip())
        raise SystemExit(2)
    raise SystemExit(asyncio.run(_main(urls)))
