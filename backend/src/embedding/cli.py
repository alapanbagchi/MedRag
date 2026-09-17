"""CLI: encode retrieval-eligible medpat chunks via an OpenAI-compatible server.

    python -m src.embedding --base-url https://host/v1 --model ncbi/MedCPT-Article-Encoder
    python -m src.embedding --limit 100 --batch 64 --workers 4

Reads (id, embedding_text) for retrieval-eligible chunks from medpat, skips
rows whose embedding_text hash is unchanged, POSTs batches to
{base_url}/embeddings, L2-normalizes the returned 768-dim vectors, and upserts
full-precision vector(768) rows.

Why normalize here: the MedCPT Article-Encoder server returns raw FP32 [CLS]
vectors (NOT unit-length), while medpat.chunk_embeddings is indexed with HNSW
vector_ip_ops and the query encoder is L2-normalized. Inner product only equals
cosine similarity when both sides are unit length, so the ingest client
normalizes before storing. Re-runs hash-skip unchanged rows.

Config comes from backend/.env (loaded automatically) or the process env: db
per MEDPAT_DSN, server per EMBEDDING_BASE_URL, model per MEDPAT_EMBED_MODEL.
The chunker never touches a model; this is the encoding step that consumes the
chunker's output straight from Postgres.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from src.chunking.store_pg import DEFAULT_DSN, connect
from src.embedding.client import EmbedClient
from src.embedding.store import (
    CREATE_VECTOR_INDEX,
    DROP_VECTOR_INDEX,
    DROP_VECTOR_INDEX_ALT,
    count_pending_chunks,
    format_vector,
    hash_text,
    iter_pending_chunks,
    load_pending_chunks,
    upsert_embeddings,
)

# backend/.env (three levels up from src/embedding/cli.py) - the file
# make medpat-embed users edit. Loaded explicitly because the CLI, unlike
# src.config's AppConfig, does not pull dotenv in on its own, and make does
# not source .env either: without this, an EMBEDDING_BASE_URL set in .env is
# invisible to os.environ and the run aborts with "no embedding server
# configured".
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def _load_env_file(env_path: Optional[Path] = None) -> None:
    """Load backend/.env into the process environment.

    Real environment values win, with one exception: an empty/whitespace
    process value never shadows a non-empty value from the file (a stale
    export EMBEDDING_BASE_URL= must not hide .env).
    """
    path = Path(env_path) if env_path is not None else ENV_PATH
    if not path.is_file():
        return
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:      # python-dotenv is a project dep; tolerate bare envs
        return
    load_dotenv(path, override=False)
    for key, val in dotenv_values(path).items():
        if val and val.strip() and not os.environ.get(key, "").strip():
            os.environ[key] = val.strip()


_load_env_file()

DEFAULT_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "").strip()
DEFAULT_MODEL = os.environ.get(
    "MEDPAT_EMBED_MODEL", "ncbi/MedCPT-Article-Encoder"
).strip()


def _normalize_args(args) -> None:
    """An empty flag value falls back to .env / the process env.

    An empty --base-url (for example the shell expanding an unset variable)
    would otherwise *override* the value loaded from backend/.env, defeating
    the whole point. Empty means unset here, for every configurable field.
    """
    args.base_url = (args.base_url or "").strip() or os.environ.get(
        "EMBEDDING_BASE_URL", "").strip()
    args.model = ((args.model or "").strip()
                  or os.environ.get("MEDPAT_EMBED_MODEL", "").strip()
                  or "ncbi/MedCPT-Article-Encoder")
    args.api_key = (args.api_key or "").strip()
    args.dsn = (args.dsn or "").strip() or DEFAULT_DSN
    if getattr(args, "commit_every", None) in (None, ""):
        args.commit_every = 2048
    args.commit_every = max(64, int(args.commit_every))
    if getattr(args, "drop_index", None) is None:
        args.drop_index = False
    if getattr(args, "tune_db", None) is None:
        args.tune_db = True


def _ingest(args) -> int:
    if args.dry_run and not args.base_url:
        pass  # dry-run never needs the server
    elif not args.base_url:
        print("no embedding server configured: set EMBEDDING_BASE_URL in "
              f"{ENV_PATH} (or the environment), or pass --base-url "
              '(OpenAI-compatible, e.g. https://host/v1). An empty '
              '--base-url "" is treated as unset.', file=sys.stderr)
        return 2

    conn = connect(args.dsn)
    print("counting pending chunks ...", flush=True)
    total = count_pending_chunks(conn)
    if args.limit:
        total = min(total, args.limit)
    print(f"chunks to (re)encode: {total}")
    if total == 0:
        conn.close()
        return 0

    if args.dry_run:
        print("[dry-run] no requests, no writes")
        for cid, text, _ in load_pending_chunks(conn, min(args.limit or 5, 5)):
            print(f"  {cid}: {text[:80]!r}")
        print(f"  ... {total} total (first 5 shown)")
        conn.close()
        return 0

    client = EmbedClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=args.retries,
        normalize=not args.no_normalize,
    )
    health = client.check_health()
    if health:
        print(f"server: {health.get('model')} | dim "
              f"{health.get('embedding_dimension')} | "
              f"{health.get('gpus')} gpu(s)")

    workers = max(1, args.workers)
    window = workers * 4   # batches in flight; bounds client memory
    print(f"workers={workers}, batch={args.batch}, window={window}, "
          f"normalize={not args.no_normalize}, "
          f"commit_every={args.commit_every}, "
          f"drop_index={args.drop_index}")

    done = 0
    failed = 0
    cur = conn.cursor()
    if args.tune_db:
        # Session-local only: skip a WAL fsync per COMMIT and give the
        # writer more sort/hash memory. Lost on disconnect; safe for a
        # hash-skipped idempotent re-run (worst case: re-embed the last txn).
        try:
            cur.execute("SET synchronous_commit=off")
            cur.execute("SET work_mem='256MB'")
        except Exception as exc:  # noqa: BLE001 - tuning is advisory
            print(f"  [warn] session tuning skipped ({exc})", flush=True)
    dropped_index = False
    if args.drop_index:
        # Per-row HNSW maintenance dominates insert cost at 3M+ rows; one
        # bulk CREATE INDEX at the end is several times faster overall.
        cur.execute(DROP_VECTOR_INDEX)
        cur.execute(DROP_VECTOR_INDEX_ALT)
        conn.commit()
        dropped_index = True
        print("dropped embeddings_vector_idx for bulk load "
              "(rebuilds once at end)")
    t0 = time.time()
    batches = iter_pending_chunks(conn, args.batch, args.limit)

    stats = {"sent": 0, "recv": 0, "norm": 0, "fail": 0}
    stats_lock = threading.Lock()
    bar = tqdm(total=total, desc="Embedding", unit="chunk", unit_scale=True,
               dynamic_ncols=True, mininterval=0.3)

    def _render_stats() -> None:
        # One in-place line: the bar updates its postfix instead of printing.
        s = stats
        done_now = max(0, s["sent"] - s["recv"] - s["fail"])
        bar.set_postfix_str(
            f"recv={s['recv']} norm={s['norm']} inflight={done_now} "
            f"fail={s['fail']}"
        )

    def _stage(event: str, n: int = 0) -> None:
        with stats_lock:
            stats[event] = stats.get(event, 0) + n
            _render_stats()

    def _embed_safe(batch):
        # Runs in a worker thread: embed, then format literals + hashes here
        # so the single main thread only does DB I/O. Returns ready-to-store
        # rows, or the exception on per-batch failure.
        texts = [t for _, t, _ in batch]
        try:
            vecs = client.embed(texts, _stage)
            return [
                (cid, format_vector(vec), client.model, hash_text(text))
                for (cid, text, _), vec in zip(batch, vecs)
            ]
        except Exception as exc:  # noqa: BLE001 - per-batch failure
            with stats_lock:
                stats["fail"] += len(texts)
                _render_stats()
            return exc

    _render_stats()
    pending_rows: list = []  # formatted rows awaiting COMMIT (<= commit_every)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            inflight = {}
            source = iter(batches)
            source_done = False
            while inflight or not source_done:
                # Fill the window, but never buffer more than window batches of
                # text: memory stays flat for the whole 4.5M-row corpus.
                while not source_done and len(inflight) < window:
                    try:
                        batch = next(source)
                    except StopIteration:
                        source_done = True
                        break
                    inflight[pool.submit(_embed_safe, batch)] = batch
                if not inflight:
                    break
                # Buffer landed batches and flush one multi-row INSERT per
                # commit_every rows; the bar tracks durable writes, so it lags
                # recv/norm (embed-stage) by up to one flush.
                completed, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in completed:
                    batch = inflight.pop(fut)
                    res = fut.result()
                    if isinstance(res, Exception):
                        failed += 1
                        bar.write(f"FAILED {batch[0][0]}..{batch[-1][0]}: {res}")
                        continue
                    pending_rows.extend(res)
                    # One multi-row INSERT + COMMIT per commit_every rows:
                    # executemany did one round-trip per ROW plus an fsync per
                    # embed-batch; this is one round-trip per ~500 rows and one
                    # fsync per commit_every rows.
                    if len(pending_rows) >= args.commit_every:
                        upsert_embeddings(cur, pending_rows)
                        conn.commit()
                        done += len(pending_rows)
                        bar.update(len(pending_rows))
                        pending_rows.clear()
        if pending_rows:
            upsert_embeddings(cur, pending_rows)
            conn.commit()
            done += len(pending_rows)
            bar.update(len(pending_rows))
            pending_rows.clear()
        conn.commit()
        if dropped_index:
            print("rebuilding embeddings_vector_idx ...", flush=True)
            t_idx = time.time()
            try:
                cur.execute("SET maintenance_work_mem='1GB'")
            except Exception as exc:  # noqa: BLE001 - advisory
                print(f"  [warn] maintenance_work_mem skipped ({exc})", flush=True)
            cur.execute(CREATE_VECTOR_INDEX)
            conn.commit()
            print(f"index rebuilt in {time.time() - t_idx:.0f}s", flush=True)
        try:
            cur.execute("ANALYZE medpat.chunk_embeddings")
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - advisory
            print(f"  [warn] ANALYZE skipped ({exc})", flush=True)
    except KeyboardInterrupt:
        bar.close()
        print("\ninterrupted; rolling back pending writes", file=sys.stderr)
        conn.rollback()
        if dropped_index:
            print("note: embeddings_vector_idx is still dropped; rebuild with: "
                  f"{CREATE_VECTOR_INDEX}", file=sys.stderr)
        return 130
    finally:
        bar.close()
        cur.close()
        client.close()
        conn.close()

    print(f"done: {done} embedded, {failed} batch(es) failed "
          f"in {time.time() - t0:.0f}s")
    return 0 if failed == 0 else 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Encode retrieval-eligible medpat chunks via an "
                    "OpenAI-compatible embedding server into "
                    "medpat.chunk_embeddings (idempotent, hash-skipped)."
    )
    parser.add_argument("--dsn", default=DEFAULT_DSN,
                        help="Postgres DSN (default: MEDPAT_DSN).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible base URL of the embedding "
                             "server, e.g. https://host/v1 "
                             "(default: EMBEDDING_BASE_URL).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model id sent to the server and recorded per "
                             "row (default: MEDPAT_EMBED_MODEL).")
    parser.add_argument("--api-key", default=os.environ.get("EMBEDDING_API_KEY", ""),
                        help="Bearer token if the server requires one "
                             "(default: EMBEDDING_API_KEY).")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--batch", type=int, default=64,
                        help="Texts per /embeddings request (default 64).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent HTTP requests (default 4; the server "
                             "runs one inference per GPU at a time).")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-request timeout in seconds (default 180).")
    parser.add_argument("--retries", type=int, default=5,
                        help="Retries per batch on transient errors (default 5).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Store raw server vectors; leave L2-normalization "
                             "to the server (not recommended).")
    parser.add_argument("--commit-every", type=int, default=2048,
                        help="Rows per COMMIT (default 2048; fewer fsyncs when "
                             "larger, more to re-embed on crash).")
    parser.add_argument("--drop-index", action="store_true",
                        help="Drop the HNSW vector index for the run and rebuild "
                             "it once at the end. Biggest win for million-row "
                             "loads: per-row HNSW maintenance is far slower "
                             "than one bulk CREATE INDEX.")
    parser.add_argument("--no-tune-db", dest="tune_db", action="store_false",
                        help="Skip session tuning (synchronous_commit=off, "
                             "larger work_mem).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count + preview chunks; no requests, no writes.")
    args = parser.parse_args(argv)
    _normalize_args(args)
    return _ingest(args)
