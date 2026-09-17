#!/usr/bin/env python3
"""medpat retrieval index (re)build: pgvector HNSW + ParadeDB BM25.

    python scripts/medpat_indexes.py [--dsn DSN] [--force]
                                     [--vector-only | --bm25-only]
                                     [--maintenance-work-mem 2GB]
                                     [--parallel-workers 2]

Creates the two indexes the hybrid retriever needs:

  * embeddings_vector_idx - pgvector HNSW (vector_ip_ops) over
    medpat.chunk_embeddings.embedding. Missing after a bulk
    `medpat-embed --drop-index` run or an interrupted build; this rebuilds it.
  * chunks_bm25_idx - ParadeDB pg_search BM25 over medpat.chunks.embedding_text
    (the sparse leg). Only one ParadeDB index per table is allowed.

Idempotent: existing indexes are left alone unless --force is given. Index
builds are atomic, so a killed build simply leaves no index and a re-run
rebuilds it cleanly.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking.store_pg import DEFAULT_DSN, connect  # noqa: E402

VECTOR_INDEX = "embeddings_vector_idx"
BM25_INDEX = "chunks_bm25_idx"

HNSW_DDL = (
    "CREATE INDEX {name} ON medpat.chunk_embeddings "
    "USING hnsw (embedding vector_ip_ops) "
    "WITH (m = {m}, ef_construction = {ef})"
)
# IVFFlat has no in-memory graph: the build is a k-means pass plus one
# assignment pass, so it finishes in minutes on a corpus where HNSW would need
# roughly the whole index resident in RAM. Recall is tuned at query time with
# ivfflat.probes (see pgvector_store.search).
IVFFLAT_DDL = (
    "CREATE INDEX {name} ON medpat.chunk_embeddings "
    "USING ivfflat (embedding vector_ip_ops) "
    "WITH (lists = {lists})"
)
# retrieval_eligible is listed so the planner pushes "WHERE retrieval_eligible"
# into the ParadeDB index as a Tantivy term filter. Omit it and ParadeDB falls
# back to a heap_filter - one heap fetch per matching doc - which turns a broad
# query (common medical terms) from ~0.1s into ~8s.
BM25_DDL = (
    "CREATE INDEX {name} ON medpat.chunks "
    "USING paradedb (id, retrieval_eligible, (embedding_text::pdb.whitespace)) "
    "WITH (key_field = 'id')"
)


def _index_exists(conn, name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'medpat' "
        "AND indexname = %s",
        (name,),
    )
    ok = cur.fetchone() is not None
    cur.close()
    return ok


def _paradedb_indexes(conn) -> list[str]:
    """Names of any existing ParadeDB (pg_search) index on medpat.chunks."""
    cur = conn.cursor()
    cur.execute(
        "SELECT i.indexrelid::regclass::text "
        "FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indexrelid "
        "JOIN pg_am am ON am.oid = c.relam "
        "WHERE i.indrelid = 'medpat.chunks'::regclass AND am.amname = 'paradedb'"
    )
    names = [r[0].split(".")[-1] for r in cur.fetchall()]
    cur.close()
    return names


def _scalar(conn, sql: str):
    cur = conn.cursor()
    cur.execute(sql)
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def _pretty_bytes(total: int) -> str:
    value = float(total)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def _size_estimate_bytes(rows: int, m: int) -> int:
    """Rough on-disk HNSW size in bytes: vector payload + graph links."""
    payload = rows * 768 * 4                       # float32 vectors
    graph = rows * m * 2 * 4 * 1.5                 # ~2 layers of int4 links
    return int((payload + graph) * 1.05)


def _size_estimate(rows: int, m: int) -> str:
    return _pretty_bytes(_size_estimate_bytes(rows, m))


def _ivfflat_size_estimate(rows: int, lists: int) -> str:
    """Rough on-disk IVFFlat size: vector payload + list pointers + centroids."""
    payload = rows * 768 * 4                       # float32 vectors
    lists_bytes = rows * 4 + lists * 768 * 4       # list entry ids + centroids
    return _pretty_bytes(int((payload + lists_bytes) * 1.05))


def _parse_size(text: str) -> int:
    """Parse '6GB' / '512MB' / '4096' (unitless = bytes) into bytes."""
    import re

    m = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", text or "")
    if not m:
        return 0
    value = float(m.group(1))
    unit = m.group(2).lower()
    mult = {
        "": 1, "b": 1,
        "k": 1024, "kb": 1024, "kib": 1024,
        "m": 1024 ** 2, "mb": 1024 ** 2, "mib": 1024 ** 2,
        "g": 1024 ** 3, "gb": 1024 ** 3, "gib": 1024 ** 3,
        "t": 1024 ** 4, "tb": 1024 ** 4, "tib": 1024 ** 4,
    }.get(unit)
    return 0 if mult is None else int(value * mult)


def _preflight_shm(mwm: str, workers: int) -> None:
    """Warn when a parallel HNSW build cannot fit in the container /dev/shm.

    dynamic_shared_memory_type=posix puts pgvector's parallel-build graph
    buffer in a POSIX shared-memory segment sized ~maintenance_work_mem. If
    that exceeds the container's shm_size the build dies with a misleading
    'DiskFull: could not resize shared memory segment'. MEDPAT_SHM (set by the
    compose file) tells us the container size; skip the check when unset.
    """
    shm_txt = os.environ.get("MEDPAT_SHM", "")
    shm = _parse_size(shm_txt)
    need = _parse_size(mwm)
    if workers <= 0 or not shm or not need or need <= shm:
        return
    print(
        f"  [warn] maintenance_work_mem ({mwm}) > container /dev/shm "
        f"({shm_txt}); a parallel HNSW build needs a ~{mwm} shared-memory "
        "segment and will fail with 'could not resize shared memory "
        "segment ... No space left on device'.\n"
        f"         Fix: lower the build memory (MWM=2GB), run serially "
        f"(IDX_WORKERS=0), or grow /dev/shm when RAM allows "
        f"(MEDPAT_SHM={_pretty_bytes(need)} make medpat-up).",
        flush=True,
    )


def _build_with_progress(conn, dsn: str, label: str, sql: str) -> None:
    """Run one CREATE INDEX while a poller prints server-side progress.

    A second connection is required: pg_stat_progress_create_index only shows
    work happening in *other* backends.
    """
    import psycopg2

    stop = threading.Event()

    def poll() -> None:
        try:
            pc = psycopg2.connect(dsn)
            pc.autocommit = True
            cur = pc.cursor()
            t0 = time.time()
            last = ""
            while not stop.is_set():
                try:
                    # Scope to *our* backend: with a build already queued on
                    # the table, the first row of pg_stat_progress_create_index
                    # belongs to some other CREATE INDEX and its progress is
                    # meaningless here.
                    cur.execute(
                        "SELECT p.phase, p.blocks_total, p.blocks_done, "
                        "p.tuples_total, p.tuples_done "
                        "FROM pg_stat_progress_create_index p "
                        "JOIN pg_stat_activity a USING (pid) "
                        "WHERE a.query ILIKE %s",
                        (f"%CREATE INDEX {label}%",),
                    )
                    row = cur.fetchone()
                except Exception:
                    row = None
                if row and row[0]:
                    phase, bt, bd, tt, td = row
                    if bd and bt:
                        line = (f"{label}: phase={phase} blocks={bd}/{bt} "
                                f"({100.0 * bd / bt:.0f}%) "
                                f"elapsed={time.time() - t0:.0f}s")
                    elif td and tt:
                        line = (f"{label}: phase={phase} tuples={td}/{tt} "
                                f"({100.0 * td / tt:.0f}%) "
                                f"elapsed={time.time() - t0:.0f}s")
                    else:
                        line = (f"{label}: phase={phase} "
                                f"elapsed={time.time() - t0:.0f}s")
                    if line != last:
                        print(f"    {line}", flush=True)
                        last = line
                elif int(time.time() - t0) % 15 == 0:
                    print(f"    {label}: building... "
                          f"elapsed={time.time() - t0:.0f}s", flush=True)
                stop.wait(3)
            cur.close()
            pc.close()
        except Exception:
            pass  # progress telemetry is best-effort

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.time()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
    finally:
        stop.set()
        th.join(timeout=8)
    print(f"    {label}: done in {time.time() - t0:.0f}s", flush=True)


def _prepare_session(conn, mwm: str, workers: int) -> None:
    conn.autocommit = True
    cur = conn.cursor()
    for stmt in (
        "SET maintenance_work_mem = '%s'" % mwm.replace("'", "''"),
        "SET max_parallel_maintenance_workers = %d" % max(0, workers),
        "SET synchronous_commit = off",
    ):
        try:
            cur.execute(stmt)
        except Exception as exc:  # noqa: BLE001 - tuning is advisory
            print(f"  [warn] {stmt} skipped ({exc})", flush=True)
    cur.close()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=None,
                    help="Postgres DSN (default: MEDPAT_DSN).")
    ap.add_argument("--force", action="store_true",
                    help="DROP and rebuild indexes even if they already exist.")
    ap.add_argument("--type", choices=("hnsw", "ivfflat"), default="hnsw",
                    help="Vector index type (default hnsw). ivfflat builds in "
                         "minutes without the in-memory graph HNSW needs.")
    ap.add_argument("--lists", type=int, default=0,
                    help="IVFFlat lists (default 0 = auto ~sqrt(rows)).")
    ap.add_argument("--vector-only", action="store_true",
                    help="Only ensure the pgvector vector index.")
    ap.add_argument("--bm25-only", action="store_true",
                    help="Only ensure the ParadeDB BM25 index.")
    ap.add_argument("--m", type=int, default=16,
                    help="HNSW m (default 16).")
    ap.add_argument("--ef-construction", type=int, default=128,
                    help="HNSW ef_construction (default 128).")
    ap.add_argument("--maintenance-work-mem", default="2GB",
                    help="Session maintenance_work_mem for the build "
                         "(default 2GB; raise on a bigger box).")
    ap.add_argument("--parallel-workers", type=int, default=2,
                    help="max_parallel_maintenance_workers (default 2; 0 to disable).")
    ap.add_argument("--no-analyze", action="store_true",
                    help="Skip the post-build ANALYZE.")
    ap.add_argument("--check", action="store_true",
                    help="Report index state and exit; build nothing.")
    args = ap.parse_args(argv)

    do_vector = not args.bm25_only
    do_bm25 = not args.vector_only
    dsn = args.dsn or os.environ.get("MEDPAT_DSN") or DEFAULT_DSN

    conn = connect(dsn)

    try:
        _prepare_session(conn, args.maintenance_work_mem, args.parallel_workers)

        # Ensure the extensions that own the index AMs are present.
        for ext in ("vector", "pg_search"):
            try:
                conn.cursor().execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] extension {ext} unavailable ({exc})")

        emb_rows = int(_scalar(conn, "SELECT count(*) FROM medpat.chunk_embeddings") or 0)
        chunk_rows = int(_scalar(conn, "SELECT count(*) FROM medpat.chunks") or 0)
        print(f"medpat: {chunk_rows:,} chunks, {emb_rows:,} embeddings")

        if args.check:
            print("\n== Index state (check only) ==")
            for name in (VECTOR_INDEX, BM25_INDEX):
                print(f"  {name}: "
                      f"{'present' if _index_exists(conn, name) else 'MISSING'}")
            return 0

        if do_vector and emb_rows:
            print("\n== pgvector HNSW: %s ==" % VECTOR_INDEX)
            exists = _index_exists(conn, VECTOR_INDEX)
            if exists and not args.force:
                print(f"  {VECTOR_INDEX} already exists; skipping "
                      f"(FORCE=1 to rebuild)")
            elif emb_rows < 100:
                print("  too few rows to benefit from an HNSW index; skipping")
            else:
                if exists:
                    print(f"  dropping {VECTOR_INDEX} (force rebuild)")
                    conn.cursor().execute(f"DROP INDEX medpat.{VECTOR_INDEX}")
                if args.type == "ivfflat":
                    lists = args.lists or max(1, int(emb_rows ** 0.5))
                    print(f"  type=ivfflat  rows={emb_rows:,}  lists={lists}  "
                          f"mwm={args.maintenance_work_mem}  "
                          f"parallel={args.parallel_workers}")
                    print(f"  estimated index size: "
                          f"~{_ivfflat_size_estimate(emb_rows, lists)}; the "
                          f"k-means build does not need the index in RAM")
                    ddl = IVFFLAT_DDL.format(name=VECTOR_INDEX, lists=lists)
                else:
                    print(f"  type=hnsw  rows={emb_rows:,}  m={args.m}  "
                          f"ef_construction={args.ef_construction}  "
                          f"mwm={args.maintenance_work_mem}  "
                          f"parallel={args.parallel_workers}")
                    est = _size_estimate_bytes(emb_rows, args.m)
                    print(f"  estimated index size: ~{_pretty_bytes(est)}; "
                          f"needs roughly that much RAM/shared memory or it "
                          f"builds on disk (very slow). Use TYPE=ivfflat when "
                          f"the box cannot hold it")
                    budget = _parse_size(args.maintenance_work_mem)
                    if budget and est > budget:
                        print(f"  [warn] index (~{_pretty_bytes(est)}) is larger "
                              f"than maintenance_work_mem "
                              f"({args.maintenance_work_mem}) — pgvector will "
                              f"build on disk and can take many hours. Strongly "
                              f"consider: make medpat-index VECTOR_ONLY=1 "
                              f"IDX_TYPE=ivfflat", flush=True)
                    ddl = HNSW_DDL.format(name=VECTOR_INDEX, m=args.m,
                                          ef=args.ef_construction)
                _preflight_shm(args.maintenance_work_mem, args.parallel_workers)
                try:
                    _build_with_progress(conn, dsn, VECTOR_INDEX, ddl)
                except Exception as exc:  # noqa: BLE001
                    if type(exc).__name__ != "DiskFull":
                        raise
                    shm_txt = os.environ.get("MEDPAT_SHM", "the container's /dev/shm")
                    print(
                        "\n  [error] the HNSW build ran out of POSIX shared "
                        f"memory ({shm_txt}).\n"
                        f"          maintenance_work_mem={args.maintenance_work_mem} "
                        f"needs a shared-memory segment of roughly that size, "
                        "but dynamic_shared_memory_type=posix caps it at "
                        "/dev/shm.\n"
                        "          Fix one of:\n"
                        "            * keep /dev/shm, lower memory:  "
                        "make medpat-index VECTOR_ONLY=1 MWM=2GB\n"
                        "            * no shared segment (serial):    "
                        "make medpat-index VECTOR_ONLY=1 IDX_WORKERS=0\n"
                        "            * grow /dev/shm when RAM allows:  "
                        "MEDPAT_SHM=8gb make medpat-up  (then rerun)",
                        flush=True,
                    )
                    return 1
        elif do_vector:
            print(f"\n== pgvector HNSW: {VECTOR_INDEX} ==\n  no embeddings; skipping")

        if do_bm25 and chunk_rows:
            print("\n== ParadeDB BM25: %s ==" % BM25_INDEX)
            others = _paradedb_indexes(conn)
            if BM25_INDEX in others and not args.force:
                print(f"  {BM25_INDEX} already exists; skipping "
                      f"(FORCE=1 to rebuild)")
            else:
                if args.force and others:
                    for name in others:
                        print(f"  dropping {name} (force rebuild)")
                        conn.cursor().execute(f"DROP INDEX medpat.{name}")
                elif others:
                    print(f"  [warn] ParadeDB only allows one index per table; "
                          f"found {others}. Dropping before rebuild.")
                    for name in others:
                        conn.cursor().execute(f"DROP INDEX medpat.{name}")
                print(f"  rows={chunk_rows:,}  tokenizer=whitespace")
                _build_with_progress(
                    conn, dsn, BM25_INDEX, BM25_DDL.format(name=BM25_INDEX)
                )
        elif do_bm25:
            print(f"\n== ParadeDB BM25: {BM25_INDEX} ==\n  no chunks; skipping")

        if not args.no_analyze:
            print("\nANALYZE medpat.chunks / medpat.chunk_embeddings ...",
                  flush=True)
            conn.cursor().execute("ANALYZE medpat.chunks")
            conn.cursor().execute("ANALYZE medpat.chunk_embeddings")

        print("\n== Index state ==")
        # Cosmetic report: pg_indexes has no indexrelid, and a failure here must
        # never fail a build that already succeeded — so it is best-effort.
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT c.relname, pg_size_pretty(pg_relation_size(c.oid)) "
                "FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'medpat' AND c.relname IN (%s, %s) "
                "ORDER BY c.relname",
                (VECTOR_INDEX, BM25_INDEX),
            )
            for name, size in cur.fetchall():
                print(f"  {name}: {size}")
            cur.close()
        except Exception as exc:  # noqa: BLE001 — report only
            print(f"  [warn] could not read index sizes: {exc}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
