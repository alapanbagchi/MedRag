#!/usr/bin/env python3
"""Bulk-load MedRAG v2 embeddings + chunk metadata into PostgreSQL pgvector.

Reads the v2 artifacts produced by the MedCPT article-encoder pipeline:

    embeddings_v2/PMC*.parquet   -> id + dim_000..dim_767 (FP16, already
                                    L2-normalized at generation time)
    chunks_v2/PMC*.parquet       -> full chunk metadata (text, type, section,
                                    breadcrumb, document info, ...)

The pipeline that created these vectors ran: FP16 -> FP32 -> L2-normalize ->
FP16, so the stored vectors are ALREADY unit vectors. This loader therefore
only upcasts to float32 and L2-normalizes at load (idempotent for already-unit vectors)
(doing so would round-trip the values pointlessly). Retrieval compares via
inner product / cosine on these unit vectors, and query vectors are
normalized at search time by the MedCPT query encoder.

Schema (schema "medrag", created idempotently):

    medrag.chunks      full v2 metadata columns; PK (id)
    medrag.embeddings  chunk_id PK REFERENCES chunks(id) ON DELETE CASCADE,
                       embedding vector(768)  (or halfvec(768) with --vector-type)

Reload semantics
----------------
*  --reset        DROP SCHEMA medrag CASCADE, recreate, then load everything.
*  --reset-only   DROP SCHEMA medrag CASCADE only (wipe).
*  no --reset     INCREMENTAL and interrupt-safe. Per-file checkpoints are
                  kept in medrag.load_marks; on re-run, files whose ids are
                  already fully present in the DB are skipped ("already
                  exists, skipping"), and only missing files are loaded.
                  Each flushed batch commits rows + checkpoint in one
                  transaction, so killing the loader mid-run never re-does
                  completed work.
*  The vector index is only (re)built when data actually changed, when it
  is missing, or with --force-reindex. A build can be interrupted safely:
  CREATE INDEX is atomic, so the old index stays until the new one finishes.
*  Embedding rows whose chunk_id has no matching chunks row are skipped
   (FK safety) and reported in the summary / verify step.
*  To revalidate everything (e.g. after regenerating the parquets), use
  `--reset`, or clear checkpoints: DELETE FROM medrag.load_marks;

Loading is fast: parquet files are read in parallel worker processes, and
batches are bulk-inserted with PostgreSQL COPY (staging table + ON CONFLICT
merge). The vector index is dropped first and rebuilt (HNSW) afterwards.

Example:
    python scripts/pg_load_v2.py                                  # load both dirs
    python scripts/pg_load_v2.py --reset                          # wipe + reload
    python scripts/pg_load_v2.py --embeddings-dir /data/embs \
                                 --chunks-dir /data/chunks --workers 8
    python scripts/pg_load_v2.py --dry-run                        # just count
    python scripts/pg_load_v2.py --stats-only                     # DB counts
    python scripts/pg_load_v2.py --reset-only                     # wipe only

PG connection: PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE env vars (the
project .env is loaded if present), or --host/--port/--user/--password/
--database flags.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

EMBEDDING_DIM = 768
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for pgvector_store import if used

DEFAULT_EMBEDDINGS_DIR = "embeddings_v2"
DEFAULT_CHUNKS_DIR = "chunks_v2"

# ---------------------------------------------------------------------------
# Schema DDL (v2). "vector" or "halfvec" chosen at runtime for the embedding
# column so both pgvector storage options are supported.
# ---------------------------------------------------------------------------

CHUNKS_COLUMNS: List[Tuple[str, str]] = [
    ("id", "TEXT PRIMARY KEY"),
    ("document_id", "TEXT"),
    ("chunk_type", "TEXT"),
    ("section", "TEXT"),
    ("subsection", "TEXT"),
    ("breadcrumb", "JSONB"),
    ("parent_id", "TEXT"),
    ("table_id", "TEXT"),
    ("figure_id", "TEXT"),
    ("document_position", "BIGINT"),
    ("text", "TEXT"),
    ("embedding_text", "TEXT"),
    ("metadata", "JSONB"),
    # ---- v2-only columns (absent from the v1 store schema) ----
    ("object_id", "TEXT"),
    ("source_block_ids", "TEXT"),
    ("equation_id", "TEXT"),
    ("reference_id", "TEXT"),
    ("row_label", "TEXT"),
    ("group_path", "TEXT"),
    ("citation_refs", "TEXT"),
    ("footnote_refs", "TEXT"),
    ("embedding_token_count", "BIGINT"),
    ("retrieval_eligible", "BOOLEAN"),
    ("concept_ids", "TEXT"),
    ("chunk_version", "TEXT"),
    # Sparse side of hybrid retrieval: Postgres full-text search over
    # embedding_text, maintained by the loader (no local BM25 needed).
    ("tsv", "TSVECTOR"),
]

CHUNK_LOAD_COLUMNS = [name for name, _ in CHUNKS_COLUMNS if name != "tsv"]

# Postgres FTS configuration used for the sparse leg (no stemming/stopwords,
# mirroring the [a-z0-9]+ tokenizer of the old local BM25 index).
FTS_CONFIG = "simple"

# Extra columns that only exist on the v2 schema (added via ALTER if the
# table predates this loader, e.g. created by PgVectorStore.ensure_schema()).
V2_ONLY_COLUMNS = [name for name, _ in CHUNKS_COLUMNS[13:]]


def chunks_ddl() -> str:
    cols = ",\n        ".join(f"{name} {typ}" for name, typ in CHUNKS_COLUMNS)
    return (
        f"CREATE TABLE IF NOT EXISTS medrag.chunks (\n"
        f"        {cols}\n"
        f")"
    )


def embeddings_ddl(vec_type: str) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS medrag.embeddings (\n"
        f"        chunk_id TEXT PRIMARY KEY\n"
        f"            REFERENCES medrag.chunks(id) ON DELETE CASCADE,\n"
        f"        embedding {vec_type}({EMBEDDING_DIM}),\n"
        f"        created_at TIMESTAMPTZ DEFAULT now()\n"
        f")"
    )


# ---------------------------------------------------------------------------
# Connection / CLI
# ---------------------------------------------------------------------------

def load_env(path: Path = PROJECT_ROOT / ".env") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
    except Exception:
        pass


def build_dsn(args: argparse.Namespace) -> str:
    parts = [f"host={args.host}", f"port={args.port}", f"dbname={args.database}"]
    if args.user:
        parts.append(f"user={args.user}")
    if args.password:
        parts.append(f"password={args.password}")
    return " ".join(parts)


def add_pg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", "medrag"))
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "medrag"))


# ---------------------------------------------------------------------------
# Worker functions (run in child processes)
# ---------------------------------------------------------------------------

def _read_embedding_files(paths: List[Path]) -> Optional[Tuple[List[str], np.ndarray]]:
    """Read one or more v2 embedding parquet files -> (ids, float32 matrix).

    Vectors are L2-normalized at load so pgvector inner-product search
    (embedding <#> query, vector_ip_ops HNSW) equals cosine similarity.
    This is idempotent for already-normalized input (norm ~= 1) and fixes
    unnormalized generations (e.g. raw MedCPT [CLS] FP32 output, whose
    self-dot product can be ~76 -> norm ~8.7).
    """
    id_lists: List[str] = []
    mats: List[np.ndarray] = []
    dim_cols = [f"dim_{i:03d}" for i in range(EMBEDDING_DIM)]

    for path in paths:
        table = pq.read_table(str(path), columns=["id"] + dim_cols)
        n = table.num_rows
        if n == 0:
            continue
        id_lists.extend(table["id"].to_pylist())
        mat = np.stack(
            [table.column(c).to_numpy() for c in dim_cols], axis=1
        ).astype(np.float32, copy=False)
        mat = _l2_normalize(mat)
        mats.append(mat)

    if not mats:
        return None
    return id_lists, np.vstack(mats)


def _read_chunk_files(paths: List[Path]) -> List[Dict[str, Any]]:
    """Read one or more chunks_v2 parquet files -> list of row dicts."""
    rows: List[Dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(str(path))
        data = table.to_pydict()
        n = table.num_rows
        colnames = CHUNK_LOAD_COLUMNS
        for i in range(n):
            row: Dict[str, Any] = {}
            for c in colnames:
                if c in data:
                    row[c] = data[c][i]
            rows.append(row)
    return rows


def _read_chunk_ids(paths: List[Path]) -> Dict[str, List[str]]:
    """Return {basename: [ids]} reading only the id column (cheap resume check)."""
    return {
        p.name: pq.read_table(str(p), columns=["id"])["id"].to_pylist()
        for p in paths
    }


def _read_embedding_ids(paths: List[Path]) -> Dict[str, List[str]]:
    """Return {basename: [ids]} reading only the id column (cheap resume check)."""
    return {
        p.name: pq.read_table(str(p), columns=["id"])["id"].to_pylist()
        for p in paths
    }


def _partition(files: List[Path], workers: int, target_files: int = 500) -> List[List[Path]]:
    """Split the file list into bounded-size tasks.

    Each task holds at most ``target_files`` files, and there are at least
    ``workers`` tasks. Bounding per-task size keeps worker memory and the
    pickled result per task small at large scale (46k+ files), independent
    of the worker count.
    """
    n_tasks = max(workers, (len(files) + target_files - 1) // target_files)
    size = max(1, (len(files) + n_tasks - 1) // n_tasks)
    return [files[i:i + size] for i in range(0, len(files), size)]


def _imap_bounded(pool: ProcessPoolExecutor, fn, tasks: List[Any], window: int, yield_tasks: bool = False):
    """Yield ``fn(task)`` results with a bounded number of in-flight tasks.

    ``ProcessPoolExecutor.map`` submits *every* task up-front, so with 46k+
    files the executor can hold dozens of large pickled results in memory at
    once (the OOM killer hit on a full chunks load). Here only ``window``
    tasks are ever submitted; their results are consumed in order, keeping
    peak memory ~= window x (largest single task result).

    With ``yield_tasks=True`` yields ``(task, result)`` pairs so callers can
    attribute results back to their source files.
    """
    it = iter(tasks)
    pending: List[Tuple[Any, Any]] = []
    while len(pending) < window:
        try:
            t = next(it)
        except StopIteration:
            break
        pending.append((t, pool.submit(fn, t)))
    while pending:
        t, fut = pending.pop(0)
        res = fut.result()
        yield (t, res) if yield_tasks else res
        try:
            t2 = next(it)
        except StopIteration:
            continue
        pending.append((t2, pool.submit(fn, t2)))


# ---------------------------------------------------------------------------
# COPY helpers (text format for vectors, CSV for chunks)
# ---------------------------------------------------------------------------

def _escape_copy(value: str) -> str:
    """Escape a single text field for PostgreSQL COPY text format."""
    return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """In-place L2-normalize rows; idempotent for already-unit vectors."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _vector_to_text(vec: np.ndarray) -> str:
    """Format a float32 vector row for COPY text format: [v0,v1,...].

    Uses 9 significant digits so float32 values round-trip exactly
    (6 digits was fine for the old FP16 parquets, lossy for FP32).
    """
    return "[" + ",".join(f"{v:.9g}" for v in vec) + "]"


def build_embedding_copy_data(ids: List[str], mat: np.ndarray) -> bytes:
    """Serialize (ids, matrix) into COPY text bytes: 'id\\t[vec]\\n' lines."""
    chunks: List[bytes] = []
    for cid, vec in zip(ids, mat):
        line = (_escape_copy(str(cid)) + "\t" + _vector_to_text(vec) + "\n").encode("utf-8")
        chunks.append(line)
    return b"".join(chunks)


def _j(v: Any) -> str:
    """Serialize a value for a JSONB column (handles pre-serialized JSON)."""
    if v is None:
        return "null"
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return "null"
        try:
            return json.dumps(json.loads(s))
        except Exception:
            return json.dumps(s)
    return json.dumps(v)


# (name, kind) kinds: text | jsonb | int | bool
CHUNK_KINDS = {
    "id": "text", "document_id": "text", "chunk_type": "text",
    "section": "text", "subsection": "text", "breadcrumb": "jsonb",
    "parent_id": "text", "table_id": "text", "figure_id": "text",
    "document_position": "int", "text": "text", "embedding_text": "text",
    "metadata": "jsonb",
    "object_id": "text", "source_block_ids": "text", "equation_id": "text",
    "reference_id": "text", "row_label": "text", "group_path": "text",
    "citation_refs": "text", "footnote_refs": "text",
    "embedding_token_count": "int", "retrieval_eligible": "bool",
    "concept_ids": "text", "chunk_version": "text",
}


def build_chunk_csv_data(rows: List[Dict[str, Any]]) -> bytes:
    """Serialize chunk rows to CSV bytes for COPY (CSV format, NULL '')."""
    out = io.StringIO()
    writer = csv.writer(out)

    for row in rows:
        line = []
        for name in CHUNK_LOAD_COLUMNS:
            v = row.get(name)
            kind = CHUNK_KINDS[name]
            if v is None:
                line.append("")
            elif kind == "jsonb":
                line.append(_j(v))
            elif kind == "int":
                line.append(str(int(v)) if v is not None else "")
            elif kind == "bool":
                line.append("t" if v else "f")
            else:
                line.append(str(v))
        writer.writerow(line)
    return out.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class Loader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.conn = None
        self.vec_type = args.vector_type
        self.cur = None
        self.stats: Dict[str, Any] = {
            "chunk_files": 0, "chunks": 0, "embedding_files": 0,
            "embeddings": 0, "orphans_skipped": 0, "elapsed_s": 0.0,
            "changed_files": 0,
            "chunks_loaded_this_run": 0, "chunks_skipped_files": 0,
            "embeddings_loaded_this_run": 0, "embeddings_skipped_files": 0,
        }

    # -- connection --------------------------------------------------------

    def connect(self):
        import psycopg2
        self.conn = psycopg2.connect(build_dsn(self.args))
        self.conn.autocommit = False
        self.cur = self.conn.cursor()
        return self.conn

    def close(self):
        if self.cur is not None:
            self.cur.close()
        if self.conn is not None and not self.conn.closed:
            self.conn.close()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        self.cur.execute(sql, params)

    def _commit(self):
        self.conn.commit()

    # -- schema ------------------------------------------------------------

    def ensure_schema(self) -> None:
        cur = self.cur
        cur.execute("CREATE SCHEMA IF NOT EXISTS medrag")
        self._commit()

        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._commit()
        except Exception:
            self.conn.rollback()
            cur.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'")
            if cur.fetchone() is None:
                raise RuntimeError(
                    "pgvector extension unavailable: 'vector' type not found. "
                    "Run `make pg-up` (docker) or install pgvector first."
                )
            self._commit()

        cur.execute(chunks_ddl())
        cur.execute(embeddings_ddl(self.vec_type))

        # Per-file load checkpoints so re-runs skip already-loaded parquet
        # files instead of re-upserting everything from scratch.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS medrag.load_marks (
                kind       TEXT NOT NULL,          -- 'chunks' | 'embeddings'
                file       TEXT NOT NULL,          -- parquet basename
                marked_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (kind, file)
            )
        """)

        # Add v2-only columns if the chunks table predates this loader.
        for name in V2_ONLY_COLUMNS:
            typ = dict(CHUNKS_COLUMNS)[name]
            safe_typ = typ.split()[0]  # e.g. BIGINT / TEXT / BOOLEAN
            cur.execute(
                f"ALTER TABLE medrag.chunks ADD COLUMN IF NOT EXISTS {name} {safe_typ}"
            )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON medrag.chunks (document_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON medrag.chunks (chunk_type)"
        )

        # ------------------------------------------------------------------
        # Sparse (FTS) support: tsv over embedding_text + GIN index, so
        # hybrid retrieval can run 100% from Postgres (no local BM25).
        # ------------------------------------------------------------------
        cur.execute("ALTER TABLE medrag.chunks ADD COLUMN IF NOT EXISTS tsv TSVECTOR")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON medrag.chunks USING gin (tsv)"
        )
        cur.execute("SELECT count(*) FROM medrag.chunks WHERE tsv IS NULL")
        missing = cur.fetchone()[0]
        if missing:
            print(f"  backfilling full-text column (tsv) for {missing:,} existing chunks...")
            batch = 250_000
            done = 0
            while True:
                cur.execute(
                    f"UPDATE medrag.chunks SET tsv = to_tsvector('{FTS_CONFIG}', "
                    f"coalesce(embedding_text, '')) "
                    f"WHERE tsv IS NULL AND ctid IN "
                    f"(SELECT ctid FROM medrag.chunks WHERE tsv IS NULL LIMIT {batch})"
                )
                n = cur.rowcount
                if n == 0:
                    break
                done += n
                self._commit()
                print(f"    {done:,}/{missing:,} backfilled", flush=True)
                if done >= missing:
                    break
        self._commit()

    def drop_schema(self) -> None:
        self.cur.execute("DROP SCHEMA IF EXISTS medrag CASCADE")
        self._commit()

    # -- resume bookkeeping ------------------------------------------------

    def _loaded_files(self, kind: str) -> set:
        self.cur.execute("SELECT file FROM medrag.load_marks WHERE kind = %s", (kind,))
        return {r[0] for r in self.cur.fetchall()}

    def _mark_loaded(self, kind: str, files: List[str]) -> None:
        """Record files whose rows are fully persisted (same tx as caller)."""
        if not files:
            return
        self.cur.executemany(
            "INSERT INTO medrag.load_marks (kind, file) VALUES (%s, %s) "
            "ON CONFLICT (kind, file) DO NOTHING",
            [(kind, f) for f in files],
        )

    def _existing_count(self, table: str, id_col: str, ids: List[str]) -> int:
        """How many of ``ids`` already exist in medrag.<table>."""
        if not ids:
            return 0
        self.cur.execute(
            f"SELECT count(*) FROM medrag.{table} WHERE {id_col} = ANY(%s)", (ids,)
        )
        return self.cur.fetchone()[0]

    def _skip_files(self, kind: str, names: List[str]) -> None:
        """Mark fully-present files as loaded and bump the skip counter."""
        self._mark_loaded(kind, names)
        self._commit()
        if kind == "chunks":
            self.stats["chunks_skipped_files"] += len(names)
        else:
            self.stats["embeddings_skipped_files"] += len(names)

    # -- chunks ------------------------------------------------------------

    def load_chunks(self, dir_path: Path) -> None:
        print(f"\n== Loading chunk metadata from {dir_path} ==")
        files = sorted(dir_path.glob("*.parquet"))
        if not files:
            print("  no .parquet files found")
            return

        done = self._loaded_files("chunks")
        todo = [f for f in files if f.name not in done]
        print(f"  {len(files):,} parquet files; {len(done):,} already loaded (checkpoint); {len(todo):,} to check")
        if not todo:
            print("  nothing to do")
            return
        t0 = time.time()

        # staging table for COPY -> upsert merge (regular temp table, kept
        # across commits, truncated between batches)
        self._exec(
            "CREATE TEMP TABLE pgload_chunk_stage (LIKE medrag.chunks) ON COMMIT PRESERVE ROWS"
        )

        pool = ProcessPoolExecutor(max_workers=self.args.workers)
        window = max(self.args.workers * 2, 4)

        # Pass 1: ids-only read -> skip files already fully present in DB.
        need_full: List[Path] = []
        skipped = 0
        for task, ids_by_file in _imap_bounded(
            pool, _read_chunk_ids,
            _partition(todo, self.args.workers, target_files=60),
            window=window, yield_tasks=True,
        ):
            if not ids_by_file:
                continue
            flat = [i for ids in ids_by_file.values() for i in ids]
            have = self._existing_count("chunks", "id", flat)
            if have == len(flat):
                self._skip_files("chunks", list(ids_by_file.keys()))
                skipped += len(ids_by_file)
            else:
                need_full.append(task)
        if skipped:
            print(f"  skipped {skipped:,} files: chunks already present in DB")
        if need_full:
            print(f"  loading {len(need_full):,} files with missing chunk rows...")

        # Pass 2: full load only for files with missing rows; each flushed
        # batch is checkpointed so an interrupt never re-does completed work.
        total = 0
        batch_rows: List[Dict[str, Any]] = []
        batch_files: List[str] = []
        processed_files = 0
        for task, rows in _imap_bounded(
            pool, _read_chunk_files, need_full, window=window, yield_tasks=True,
        ):
            processed_files += len(task)
            batch_rows.extend(rows)
            batch_files.extend(p.name for p in task)
            if len(batch_rows) >= self.args.chunk_batch:
                self._flush_chunk_batch(batch_rows, batch_files)
                total += len(batch_rows)
                batch_rows, batch_files = [], []
                self._chunk_progress(total, t0)
        if batch_rows:
            self._flush_chunk_batch(batch_rows, batch_files)
            total += len(batch_rows)
        pool.shutdown()

        self._exec("DROP TABLE IF EXISTS pgload_chunk_stage")
        self._commit()
        elapsed = time.time() - t0
        self.stats["chunks_loaded_this_run"] = total
        self.stats["chunk_files"] = processed_files
        if total:
            print(f"  Loaded {total:,} chunk rows in {elapsed:.0f}s")

    def _flush_chunk_batch(self, rows: List[Dict[str, Any]], files: List[str]) -> None:
        data = build_chunk_csv_data(rows)
        self._copy_csv("pgload_chunk_stage", tuple(CHUNK_LOAD_COLUMNS), data)
        cols = ", ".join(CHUNK_LOAD_COLUMNS)
        update = ", ".join(f"{name} = EXCLUDED.{name}" for name in CHUNK_LOAD_COLUMNS if name != "id")
        self._exec(
            f"INSERT INTO medrag.chunks ({cols}, tsv) "
            f"SELECT {cols}, to_tsvector('{FTS_CONFIG}', coalesce(s.embedding_text, '')) "
            f"FROM pgload_chunk_stage s "
            f"ON CONFLICT (id) DO UPDATE SET {update}, tsv = EXCLUDED.tsv"
        )
        self._exec("TRUNCATE pgload_chunk_stage")
        self._mark_loaded("chunks", files)
        self._commit()
        self.stats["changed_files"] += len(files)

    def _chunk_progress(self, total: int, t0: float) -> None:
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"  {total:>10,} chunks  ({rate:,.0f} rows/s)", flush=True)

    # -- embeddings --------------------------------------------------------

    def load_embeddings(self, dir_path: Path) -> None:
        print(f"\n== Loading embeddings from {dir_path} ==")
        files = sorted(dir_path.glob("*.parquet"))
        if not files:
            print("  no .parquet files found")
            return

        done = self._loaded_files("embeddings")
        todo = [f for f in files if f.name not in done]
        print(f"  {len(files):,} files, dim={EMBEDDING_DIM}, type={self.vec_type}; "
              f"{len(done):,} already loaded (checkpoint); {len(todo):,} to check")
        print("  (L2-normalized at load so IP search = cosine; idempotent for normalized input)")
        if not todo:
            print("  nothing to do")
            return
        t0 = time.time()

        self._exec(
            f"CREATE TEMP TABLE pgload_emb_stage (chunk_id TEXT, embedding {self.vec_type}({EMBEDDING_DIM})) "
            "ON COMMIT PRESERVE ROWS"
        )

        pool = ProcessPoolExecutor(max_workers=self.args.workers)
        window = max(self.args.workers * 2, 4)

        # Pass 1: ids-only read -> skip files already fully present in DB.
        need_full: List[Path] = []
        skipped = 0
        for task, ids_by_file in _imap_bounded(
            pool, _read_embedding_ids,
            _partition(todo, self.args.workers, target_files=60),
            window=window, yield_tasks=True,
        ):
            if not ids_by_file:
                continue
            flat = [i for ids in ids_by_file.values() for i in ids]
            have = self._existing_count("embeddings", "chunk_id", flat)
            if have == len(flat):
                self._skip_files("embeddings", list(ids_by_file.keys()))
                skipped += len(ids_by_file)
            else:
                need_full.append(task)
        if skipped:
            print(f"  skipped {skipped:,} files: embeddings already present in DB")
        if need_full:
            print(f"  loading {len(need_full):,} files with missing embeddings...")

        # Pass 2: full read + COPY merge only for files with missing rows.
        total = 0
        buffer_ids: List[str] = []
        buffer_mats: List[np.ndarray] = []
        buffer_files: List[str] = []
        buffer_rows = 0
        files_done = 0
        for task, result in _imap_bounded(
            pool, _read_embedding_files, need_full, window=window, yield_tasks=True,
        ):
            files_done += 1
            if result is None:
                continue
            ids, mat = result
            buffer_ids.extend(ids)
            buffer_mats.append(mat)
            buffer_files.extend(p.name for p in task)
            buffer_rows += len(ids)
            if buffer_rows >= self.args.flush_rows:
                self._flush_embedding_batch(buffer_ids, buffer_mats, buffer_files)
                total += buffer_rows
                buffer_ids, buffer_mats, buffer_files, buffer_rows = [], [], [], 0
                self._embedding_progress(total, t0, files_done, len(need_full))
        if buffer_rows:
            self._flush_embedding_batch(buffer_ids, buffer_mats, buffer_files)
            total += buffer_rows
        pool.shutdown()

        self._exec("DROP TABLE IF EXISTS pgload_emb_stage")
        self._commit()
        elapsed = time.time() - t0
        self.stats["embeddings_loaded_this_run"] = total
        self.stats["embedding_files"] = files_done
        if total:
            print(f"  Loaded {total:,} embeddings in {elapsed:.0f}s")

    def _flush_embedding_batch(self, ids: List[str], mats: List[np.ndarray], files: List[str]) -> None:
        mat = np.vstack(mats)
        data = build_embedding_copy_data(ids, mat)
        self._copy_text("pgload_emb_stage (chunk_id, embedding)", data)

        # Merge, skipping ids that have no chunk metadata (FK safety).
        self._exec(
            f"INSERT INTO medrag.embeddings (chunk_id, embedding) "
            f"SELECT s.chunk_id, s.embedding FROM pgload_emb_stage s "
            f"WHERE EXISTS (SELECT 1 FROM medrag.chunks c WHERE c.id = s.chunk_id) "
            f"ON CONFLICT (chunk_id) DO UPDATE SET "
            f"    embedding = EXCLUDED.embedding, created_at = now()"
        )
        self.cur.execute(
            "SELECT count(*) FROM pgload_emb_stage s "
            "LEFT JOIN medrag.chunks c ON c.id = s.chunk_id WHERE c.id IS NULL"
        )
        self.stats["orphans_skipped"] += self.cur.fetchone()[0]
        self._exec("TRUNCATE pgload_emb_stage")
        self._mark_loaded("embeddings", files)
        self._commit()
        self.stats["changed_files"] += len(files)

    def _embedding_progress(self, total: int, t0: float, files_done: int, files_total: int) -> None:
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        eta = 0
        if rate > 0 and files_done and files_total:
            rows_per_file = total / files_done
            eta = max(0, int((files_total * rows_per_file - total) / rate))
        print(f"  {total:>10,} embeddings  ({rate:,.0f} vec/s, ~ETA {eta}s)", flush=True)

    # -- COPY plumbing -----------------------------------------------------

    def _copy_text(self, target: str, data: bytes) -> None:
        buf = io.BytesIO(data)
        self.cur.copy_expert(
            f"COPY {target} FROM STDIN",
            buf,
        )

    def _copy_csv(self, table: str, columns: Tuple[str, ...], data: bytes) -> None:
        cols = ", ".join(columns)
        buf = io.BytesIO(data)
        self.cur.copy_expert(
            f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '')",
            buf,
        )

    # -- index -------------------------------------------------------------

    def finalize_index(self) -> None:
        """Build/rebuild the vector index only when it is missing or stale.

        Re-runs with no data changes leave an existing index untouched
        (previously the index was dropped + rebuilt on every run, which is
        what made an interrupted load feel like starting over).
        """
        kind = self.args.index
        if kind == "none":
            return
        cur = self.cur
        changed = self.stats.get("changed_files", 0)
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = 'medrag' "
            "AND tablename = 'embeddings' AND indexname = 'idx_embeddings_vector'"
        )
        index_exists = cur.fetchone() is not None

        if index_exists and changed == 0 and not self.args.force_reindex:
            print("\n== Vector index ==")
            print("  idx_embeddings_vector already exists and no data changed — "
                  "skipping rebuild (use --force-reindex or FORCE_REINDEX=1 to rebuild)")
            return

        print(f"\n== {'Rebuilding' if index_exists else 'Building'} vector index ({kind}) ==")
        if index_exists:
            cur.execute("DROP INDEX IF EXISTS medrag.idx_embeddings_vector")
            self._commit()

        cur.execute("SELECT count(*) FROM medrag.embeddings")
        n = cur.fetchone()[0]
        if n == 0:
            print("  no embeddings to index")
            return
        if n < 30:
            print("  too few rows for the vector index; skipping")
            return

        ops = f"{self.vec_type}_ip_ops"  # inner product = cosine for unit vectors
        t0 = time.time()

        def try_create(sql: str) -> bool:
            try:
                self._run_index_with_progress(sql)
                return True
            except Exception as e:
                self.conn.rollback()
                print(f"  -> {e}")
                return False

        created = False
        if kind == "hnsw":
            created = try_create(
                f"CREATE INDEX idx_embeddings_vector ON medrag.embeddings "
                f"USING hnsw (embedding {ops}) WITH (m = 16, ef_construction = 64)"
            )
            if not created:
                print("  hnsw not available (pgvector < 0.5 or unsupported); "
                      "falling back to ivfflat")
                kind = "ivfflat"
        if kind == "ivfflat":
            lists = max(100, min(int(n / 1000), 100000))
            created = try_create(
                f"CREATE INDEX idx_embeddings_vector ON medrag.embeddings "
                f"USING ivfflat (embedding {ops}) WITH (lists = {lists})"
            )
        if not created:
            print("  no vector index could be built; searches will use exact scans")
            return
        print(f"  Index built in {time.time() - t0:.0f}s")

    def _run_index_with_progress(self, sql: str) -> None:
        """Run CREATE INDEX while a poller thread prints progress from
        pg_stat_progress_create_index (phase + heap blocks, if reported).

        Index builds are atomic: if this is interrupted, the index simply
        does not exist and the next run rebuilds it cleanly.
        """
        import threading

        stop = threading.Event()

        def poll() -> None:
            try:
                import psycopg2
                conn = psycopg2.connect(build_dsn(self.args))
                conn.autocommit = True
                cur = conn.cursor()
                t0 = time.time()
                while not stop.is_set():
                    try:
                        cur.execute(
                            "SELECT phase, blocks_total, blocks_done, tuples_total, tuples_done "
                            "FROM pg_stat_progress_create_index"
                        )
                        row = cur.fetchone()
                    except Exception:
                        row = None
                    elapsed = time.time() - t0
                    if row and row[0]:
                        phase, bt, bd, tt, td = row
                        pct = f" {100.0 * bd / bt:.0f}%" if bt and bd is not None else ""
                        print(f"    index build: phase={phase} blocks={bd}/{bt}{pct} "
                              f"elapsed={elapsed:.0f}s", flush=True)
                    else:
                        print(f"    index build in progress... elapsed={elapsed:.0f}s", flush=True)
                    stop.wait(5)
                conn.close()
            except Exception:
                pass  # progress polling is best-effort telemetry

        th = threading.Thread(target=poll, daemon=True)
        th.start()
        try:
            self.cur.execute(sql)
            self._commit()
        finally:
            stop.set()
            th.join(timeout=7)

    # -- verification ------------------------------------------------------

    def verify(self) -> None:
        print("\n== Verification ==")
        cur = self.cur
        cur.execute("SELECT count(*) FROM medrag.chunks")
        chunks = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM medrag.embeddings")
        embeddings = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT document_id) FROM medrag.chunks")
        docs = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*) FROM medrag.embeddings e "
            "LEFT JOIN medrag.chunks c ON c.id = e.chunk_id WHERE c.id IS NULL"
        )
        orphans = cur.fetchone()[0]
        self.stats.update(chunks=chunks, embeddings=embeddings, documents=docs)

        print(f"  chunks:     {chunks:,}")
        print(f"  embeddings: {embeddings:,}")
        print(f"  documents:  {docs:,}")
        print(f"  embeddings without chunk metadata (skipped): {orphans:,}")

        if embeddings == 0:
            print("  nothing to probe; run `make pg-load` with both dirs")
            return

        # 1) Sampling norm: stored vectors should be unit-length
        #    (fp16 -> fp32 rounding keeps ||v|| ~= 1). -(v <#> v) = v.v ~= 1.
        cur.execute(
            f"SELECT avg(-(embedding <#> embedding)), min(-(embedding <#> embedding)), "
            f"max(-(embedding <#> embedding)) FROM "
            f"(SELECT embedding FROM medrag.embeddings ORDER BY random() LIMIT 500) s"
        )
        avg_dot, min_dot, max_dot = cur.fetchone()
        print(f"  sample self-dot product (unit vector => ~1.0): avg={avg_dot:.4f} min={min_dot:.4f} max={max_dot:.4f} "
              f"{'[OK: vectors are normalized]' if abs(avg_dot - 1.0) < 0.02 else '[WARNING: vectors may not be normalized]'}")

        # 2) Self-hit: a stored vector as query must match itself first.
        cur.execute(
            f"SELECT chunk_id, embedding FROM medrag.embeddings ORDER BY random() LIMIT 3"
        )
        qtype = "vector" if self.vec_type == "vector" else "vector"  # halfvec mixes with ::vector
        for cid, emb in cur.fetchall():
            cur.execute(
                f"SELECT chunk_id, 1 - (embedding <=> %s::{qtype}) AS sim "
                f"FROM medrag.embeddings ORDER BY embedding <=> %s::{qtype} LIMIT 1",
                (emb, emb),
            )
            top, sim = cur.fetchone()
            mark = "OK" if top == cid and sim > 0.99 else "FAIL"
            print(f"  self-hit {cid}: top={top} sim={sim:.4f} [{mark}]")

        # 3) Index usability for the store's <#> query (vector_ip_ops HNSW).
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'medrag' AND tablename = 'embeddings'"
        )
        rows = cur.fetchall()
        if rows:
            for name, defin in rows:
                print(f"  index {name}: {defin}")
            plan = self._explain_vector_scan()
            if plan:
                print("  query plan with a stored vector:")
                for line in plan:
                    print(f"    {line}")

    def _explain_vector_scan(self) -> Optional[List[str]]:
        cur = self.cur
        cur.execute(
            "SELECT embedding FROM medrag.embeddings LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        emb = row[0]
        cur.execute(
            f"EXPLAIN SELECT chunk_id FROM medrag.embeddings "
            f"ORDER BY embedding <#> %s::{('halfvec' if self.vec_type=='halfvec' else 'vector')} LIMIT 10",
            (emb,),
        )
        lines = [r[0] for r in cur.fetchall()]
        return [ln[:200] + ("..." if len(ln) > 200 else "") for ln in lines]

    # -- misc --------------------------------------------------------------

    def print_stats(self) -> int:
        import psycopg2
        conn = psycopg2.connect(build_dsn(self.args))
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM medrag.chunks")
        chunks = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM medrag.embeddings")
        embeddings = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT document_id) FROM medrag.chunks")
        docs = cur.fetchone()[0]
        cur.execute(
            "SELECT e.chunk_id FROM medrag.embeddings e "
            "LEFT JOIN medrag.chunks c ON c.id = e.chunk_id WHERE c.id IS NULL LIMIT 1"
        )
        orphan = cur.fetchone() is not None
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname='medrag' AND tablename='embeddings'"
        )
        indexes = [r[0] for r in cur.fetchall()]
        print(f"chunks:     {chunks:,}")
        print(f"embeddings: {embeddings:,}")
        print(f"documents:  {docs:,}")
        print(f"has orphans: {orphan}")
        print(f"embeddings indexes: {', '.join(indexes) or '(none)'}")
        conn.close()
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(
        description="Load MedRAG v2 embeddings + chunks into pgvector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_pg_args(parser)
    parser.add_argument("--embeddings-dir", type=Path, default=Path(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--chunks-dir", type=Path, default=Path(DEFAULT_CHUNKS_DIR))
    parser.add_argument("--reset", action="store_true",
                        help="DROP SCHEMA medrag CASCADE first, then load")
    parser.add_argument("--reset-only", action="store_true",
                        help="only DROP SCHEMA medrag CASCADE (no load)")
    parser.add_argument("--vectors-only", action="store_true",
                        help="skip chunk metadata loading")
    parser.add_argument("--chunks-only", action="store_true",
                        help="skip embedding loading")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 4),
                        help="parallel parquet reader processes")
    parser.add_argument("--flush-rows", type=int, default=25_000,
                        help="embeddings accumulated before one COPY+merge")
    parser.add_argument("--chunk-batch", type=int, default=10_000,
                        help="chunk rows accumulated before one COPY+merge")
    parser.add_argument("--vector-type", choices=("vector", "halfvec"), default="vector",
                        help="pgvector storage type for the embedding column")
    parser.add_argument("--index", choices=("hnsw", "ivfflat", "none"), default="hnsw")
    parser.add_argument("--force-reindex", action="store_true",
                        help="drop + rebuild the vector index even if nothing changed")
    parser.add_argument("--dry-run", action="store_true",
                        help="count files/rows in the input dirs; touch nothing")
    parser.add_argument("--stats-only", action="store_true",
                        help="print current DB counts and exit")
    args = parser.parse_args(argv)

    if args.stats_only:
        return Loader(args).print_stats()

    if args.dry_run:
        return dry_run(args)

    loader = Loader(args)

    try:
        loader.connect()
    except Exception as e:
        print(f"ERROR: cannot connect to PostgreSQL at "
              f"{args.host}:{args.port}/{args.database}: {e}")
        print("Is the container running? Try `make pg-up`.")
        return 1

    t0 = time.time()
    try:
        if args.reset_only or args.reset:
            print("Resetting: DROP SCHEMA medrag CASCADE")
            loader.drop_schema()
            if args.reset_only:
                print("  done.")
                return 0

        print("Ensuring schema (medrag.chunks, medrag.embeddings)...")
        loader.ensure_schema()

        if not args.vectors_only and not args.chunks_only:
            loader.load_chunks(args.chunks_dir)
            loader.load_embeddings(args.embeddings_dir)
        elif args.vectors_only:
            loader.load_embeddings(args.embeddings_dir)
        else:  # chunks-only
            loader.load_chunks(args.chunks_dir)

        loader.finalize_index()
        loader.verify()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {e}")
        return 1
    finally:
        loader.close()

    loader.stats["elapsed_s"] = round(time.time() - t0, 1)
    s = loader.stats
    print("\n" + "=" * 60)
    print("  LOAD COMPLETE")
    print("=" * 60)
    print(f"  chunks (DB):       {s['chunks']:,}"
          + (f"   [this run: {s['chunks_loaded_this_run']:,} loaded, "
             f"{s['chunks_skipped_files']:,} files already present]" if not args.reset else ""))
    print(f"  embeddings (DB):   {s['embeddings']:,}"
          + (f"   [this run: {s['embeddings_loaded_this_run']:,} loaded, "
             f"{s['embeddings_skipped_files']:,} files already present]" if not args.reset else ""))
    print(f"  documents:         {s['documents']:,}")
    print(f"  orphan embeddings: {s['orphans_skipped']:,} (skipped)")
    print(f"  elapsed:           {s['elapsed_s']}s")
    print()
    print("  Retrieval is ready:")
    print(f"    PGHOST={args.host} PGPORT={args.port} PGDATABASE={args.database}")
    print("    python scripts/pg_smoke.py          # smoke test")
    return 0


def _count_rows(files: List[Path]) -> List[Tuple[str, int]]:
    """Return [(name, row_count), ...] for parquet files (worker-safe)."""
    return [(f.name, pq.ParquetFile(f).metadata.num_rows) for f in files]


def dry_run(args: argparse.Namespace) -> int:
    print("Dry run — counting input files (no DB access)...")
    from concurrent.futures import ProcessPoolExecutor

    for label, d in (("embeddings", args.embeddings_dir), ("chunks", args.chunks_dir)):
        files = sorted(d.glob("*.parquet"))
        if not files:
            print(f"{label}: no parquet files in {d}")
            continue
        rows = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            window = max(args.workers * 2, 4)
            for part in _imap_bounded(
                pool, _count_rows, _partition(files, args.workers, target_files=300),
                window=window,
            ):
                rows += sum(n for _, n in part)
        print(f"{label}: {len(files):,} files, {rows:,} rows  ({d})")
    print("Nothing was changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())