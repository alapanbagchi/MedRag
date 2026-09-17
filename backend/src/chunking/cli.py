"""CLI: chunk a directory of Markdown -> medpat (ParadeDB).

    python -m src.chunking --input data/md            # whole directory
    python -m src.chunking --input data/md --workers 8
    python -m src.chunking --input data/md/PMC1.md    # single file

Flow: discover .md files -> for each: read, chunk (documents.chunk_document),
store in medpat (store_pg.store_file, idempotent via md_sha256 skip).
--workers defaults to 1 (strictly sequential); more workers chunk files in
parallel, one database connection per worker thread.
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chunking.documents import chunk_document
from src.chunking.tokens import _DEFAULT_SPLIT_OVERLAP_SENTENCES


def _discover_files(input_path: Path) -> List[Path]:
    """A file -> [file] if .md; a directory -> all .md files under it."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".md" else []
    if input_path.is_dir():
        return sorted(input_path.rglob("*.md"))
    return []


def _chunk_kwargs(args, hard_max: int) -> Dict[str, Any]:
    """The chunk_document options the CLI maps 1:1."""
    return dict(
        max_tokens=args.max_tokens,
        hard_max_tokens=hard_max,
        split_overlap_sentences=args.split_overlap_sentences,
        split_overlap_tokens=args.split_overlap_tokens,
    )


def _ingest(args) -> int:
    """Chunk MD files into the medpat ParadeDB container.

    workers=1 (default) is strictly sequential. With more workers every
    worker thread keeps its OWN psycopg2 connection (connections are not
    thread-safe to share); each file's upsert is still one transaction.
    """
    from src.chunking import store_pg

    dsn = args.dsn or None
    files = _discover_files(Path(args.input))
    if not files:
        print(f"No Markdown files found under {args.input}", file=sys.stderr)
        return 1

    hard_max = args.hard_max_tokens or 2 * args.max_tokens
    if args.split_overlap_tokens <= 0:
        args.split_overlap_tokens = None
    kwargs = _chunk_kwargs(args, hard_max)
    workers = max(1, args.workers)
    print(f"chunking {len(files)} md file(s) -> medpat "
          f"(workers={workers}, max_tokens={args.max_tokens}, "
          f"hard_max_tokens={hard_max}, "
          f"split_overlap_sentences={args.split_overlap_sentences})")

    # thread-local connection per worker, all tracked so they can be closed
    _local = threading.local()
    conns: List[Any] = []
    conns_lock = threading.Lock()

    def _conn():
        c = getattr(_local, "conn", None)
        if c is None:
            c = store_pg.connect(dsn)
            _local.conn = c
            with conns_lock:
                conns.append(c)
        return c

    def run_one(path: Path):
        return store_pg.store_file(
            _conn(),
            path.read_text(encoding="utf-8"),
            path.stem,
            lambda text, doc_id=None: chunk_document(text, doc_id=doc_id, **kwargs),
            overwrite=args.overwrite,
        )

    from tqdm import tqdm

    converted = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, msg in tqdm(pool.map(run_one, files), total=len(files),
                                desc="Chunking", unit="doc"):
            if status == "ok":
                converted += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"FAILED {msg}", file=sys.stderr)
    for c in conns:
        c.close()

    print(f"chunked={converted} skipped={skipped} failed={failed}")
    stats_conn = store_pg.connect(dsn)
    print("corpus in medpat:", store_pg.corpus_stats(stats_conn))
    stats_conn.close()
    return 0 if failed == 0 else 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chunk Markdown articles (no LLM) and store chunks + units "
                    "in the medpat ParadeDB container. Encoding is a separate "
                    "downstream step."
    )
    parser.add_argument("--input", default="",
                        help="MD file or directory to chunk (stored in medpat).")
    parser.add_argument("--dsn", default="",
                        help="Postgres DSN (default: MEDPAT_DSN env).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default 1 = sequential; one "
                             "connection per worker).")
    parser.add_argument("--max-tokens", type=int, default=320,
                        help="Soft token budget per prose chunk (default 320).")
    parser.add_argument("--hard-max-tokens", type=int, default=0,
                        help="Absolute per-chunk token ceiling (default: 2x --max-tokens).")
    parser.add_argument("--split-overlap-sentences", type=int,
                        default=_DEFAULT_SPLIT_OVERLAP_SENTENCES,
                        help="Sentence carry when a paragraph is split "
                             "(default 2; token-budgeted to ~25%% of max_tokens).")
    parser.add_argument("--split-overlap-tokens", type=int, default=0,
                        help="Token budget for the intra-split sentence carry "
                             "(default 0 = auto: ~25%% of --max-tokens, capped at half).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-chunk files whose md_sha256 already exists.")
    args = parser.parse_args(argv)

    if not args.input.strip():
        parser.error("provide --input <md dir/file>")
    return _ingest(args)


if __name__ == "__main__":
    sys.exit(main())
