#!/usr/bin/env python
"""Interactive CLI: download all PMC Open Access articles for a term.

Prompts for:
  - the search TITLE / term,
  - an optional publication YEAR RANGE (e.g. 2015-2024).

If the year range is left BLANK, all years indexed by NCBI are searched.
Articles are downloaded as JATS XML into the output directory (default
data/raw/<slug-of-term>), resumable by file existence, with a tqdm progress
bar during download.

Usage:
    python scripts/download_articles.py
    python scripts/download_articles.py --workers 20
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _slug(text: str) -> str:
    """Filesystem-safe slug of the search term (fallback: articles)."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return s or "articles"


def _ask(question: str, default: str = "") -> str:
    """Prompt interactively; return the trimmed input (or the default)."""
    try:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{question}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n(aborted)", file=sys.stderr)
        sys.exit(130)
    return raw or default


def _parse_year_range(raw: str):
    """Parse 2015-2024 / 2015 into (min_year, max_year); blank -> (0, 0).

    Blank or "-" returns (0, 0) meaning ALL permissible years.
    """
    raw = (raw or "").strip()
    if not raw or raw == "-":
        return 0, 0
    m = re.match(r"^(\d{4})(?:\s*-\s*(\d{4}))?$", raw)
    if not m:
        raise ValueError(f"{raw!r} is not a valid year range; use e.g. 2015-2024 or 2015")
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    if lo > hi:
        raise ValueError(f"start year {lo} cannot be after end year {hi}")
    return lo, hi


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download PMC OA articles for a term.")
    parser.add_argument("--output-dir", default="",
                        help="Output directory (default: data/raw/<term-slug>).")
    parser.add_argument("--workers", type=int, default=20,
                        help="Concurrent download workers.")
    parser.add_argument("--title", default="",
                        help="Search term. If omitted, you are prompted.")
    parser.add_argument("--years", default="",
                        help="Year range like 2015-2024. Blank = all years.")
    args = parser.parse_args(argv)

    title = _ask("Title / search term", args.title)
    if not title:
        print("A search term is required.", file=sys.stderr)
        return 2

    years_raw = _ask("Year range (e.g. 2015-2024, blank = all years)", args.years)
    try:
        min_year, max_year = _parse_year_range(years_raw)
    except ValueError as exc:
        print(f"Invalid year range: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or str(ROOT / "data" / "raw" / _slug(title))
    if min_year or max_year:
        hi_str = str(max_year) if max_year else "present"
        scope = f"{min_year}-{hi_str}"
    else:
        scope = "all years"
    print(f"\nSearching PMC for: {title!r}  [{scope}]", flush=True)
    print(f"Output: {output_dir}", flush=True)

    from src.collector import PMCCollector

    PMCCollector(
        topic=title,
        output_dir=output_dir,
        max_workers=args.workers,
        min_year=min_year,
        max_year=max_year,
    ).collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())