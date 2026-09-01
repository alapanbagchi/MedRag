"""Metadata/text index and corpus diagnostics.

The raw per-document Parquet files are the source of truth. This module:

1. discovers the chunk / embedding Parquet files and the mapping between them,
2. consolidates retrieval-eligible chunks into a single ``corpus.parquet``
   (the metadata/text index) used for id -> metadata resolution and BM25
   builds,
3. provides ``run_diagnostics`` for the required safety checks (missing /
   orphan embeddings, duplicate ids, malformed vectors, id-set mismatches).

No embeddings or large text blobs are duplicated in the vector/sparse
indexes themselves; they only hold chunk IDs and raw scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EMBEDDING_SUFFIX = ".embeddings.parquet"
EMBEDDING_DIM = 768
CORPUS_FILENAME = "corpus.parquet"

# Columns persisted in the consolidated metadata/text index.
CORPUS_COLUMNS: List[str] = [
    "id",
    "document_id",
    "text",
    "embedding_text",
    "chunk_type",
    "section",
    "subsection",
    "breadcrumb",
    "parent_id",
    "table_id",
    "figure_id",
    "document_position",
]

# Extra columns that must be read to decide retrieval eligibility.
_READ_COLUMNS: List[str] = CORPUS_COLUMNS + ["retrieval_eligible"]


def corpus_schema() -> pa.Schema:
    """Fixed schema for the consolidated metadata/text index.

    Per-document tables can differ (e.g. an all-null ``parent_id`` column
    is inferred as type ``null`` in one file and ``string`` in another), so
    every table is cast to this schema before being written.
    """
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embedding_text", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("section", pa.string()),
            pa.field("subsection", pa.string()),
            pa.field("breadcrumb", pa.list_(pa.string())),
            pa.field("parent_id", pa.string()),
            pa.field("table_id", pa.string()),
            pa.field("figure_id", pa.string()),
            pa.field("document_position", pa.int64()),
        ]
    )


def _cast_to_corpus_schema(table: pa.Table) -> pa.Table:
    schema = corpus_schema()
    cols = []
    for field in schema:
        col = table[field.name]
        if pa.types.is_null(col.type):
            if pa.types.is_list(field.type):
                col = pa.nulls(len(table), type=field.type)
            else:
                col = pc.cast(col, field.type)
        elif col.type != field.type:
            col = pc.cast(col, field.type)
        cols.append(col)
    return pa.Table.from_arrays(cols, schema=schema)


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def discover_chunks(chunks_dir: Path) -> List[Path]:
    """Return sorted chunk Parquet paths (excludes embedding/tmp files)."""
    chunks_dir = Path(chunks_dir)
    return sorted(
        p
        for p in chunks_dir.rglob("*.parquet")
        if EMBEDDING_SUFFIX not in p.name and ".embeddings.tmp" not in p.name
    )


def discover_embeddings(embeddings_dir: Path) -> List[Path]:
    """Return sorted embedding Parquet paths."""
    embeddings_dir = Path(embeddings_dir)
    return sorted(embeddings_dir.rglob("*" + EMBEDDING_SUFFIX))


def embedding_path_for(chunk_path: Path, chunks_dir: Path, embeddings_dir: Path) -> Path:
    """Map ``chunks/{stem}.parquet`` -> ``embeddings/{stem}.embeddings.parquet``."""
    chunk_path = Path(chunk_path)
    rel = chunk_path.relative_to(Path(chunks_dir))
    name = rel.name
    new_name = name[: -len(".parquet")] + EMBEDDING_SUFFIX if name.endswith(".parquet") else name + EMBEDDING_SUFFIX
    return Path(embeddings_dir) / rel.parent / new_name


def stems(chunk_files: Sequence[Path]) -> set:
    return {p.stem for p in chunk_files}


def embedding_stem(p: Path) -> str:
    name = p.name
    return name[: -len(EMBEDDING_SUFFIX)] if name.endswith(EMBEDDING_SUFFIX) else p.stem


# ----------------------------------------------------------------------
# Reading eligible chunks
# ----------------------------------------------------------------------

def read_eligible_table(path: Path) -> pa.Table:
    """Read one chunk Parquet and return only retrieval-eligible rows."""
    table = pq.read_table(str(path), columns=_READ_COLUMNS)
    elig = pc.fill_null(pc.cast(table["retrieval_eligible"], pa.bool_()), False)
    filtered = table.filter(pc.equal(elig, True))
    return filtered.select(CORPUS_COLUMNS)


def iter_eligible_tables(chunk_files: Sequence[Path]) -> Iterator[pa.Table]:
    for path in chunk_files:
        yield read_eligible_table(path)


# ----------------------------------------------------------------------
# Consolidated metadata/text index
# ----------------------------------------------------------------------

def build_corpus_parquet(
    chunk_files: Sequence[Path],
    out_path: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Consolidate retrieval-eligible chunks into a single Parquet file.

    Returns build statistics. Existing rows are appended only when the file
    is absent; the caller is responsible for resumability decisions.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(chunk_files)
    if limit is not None and limit > 0:
        files = files[:limit]

    total_chunks = 0
    total_docs = 0
    schema: Optional[pa.Schema] = None
    writer: Optional[pq.ParquetWriter] = None

    try:
        for path in files:
            table = _cast_to_corpus_schema(read_eligible_table(path))
            if table.num_rows == 0:
                continue
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(str(out_path), schema)
            writer.write_table(table)
            total_chunks += table.num_rows
            total_docs += 1
    finally:
        if writer is not None:
            writer.close()

    stats = {
        "output": str(out_path),
        "documents": total_docs,
        "chunks": total_chunks,
        "schema": list(schema.names) if schema is not None else [],
    }
    return stats


# ----------------------------------------------------------------------
# Resolution store
# ----------------------------------------------------------------------

class CorpusIndex:
    """Resolves chunk ids back to original metadata/text.

    Reads the consolidated ``corpus.parquet`` lazily and caches a pandas
    DataFrame indexed by ``id`` in memory. This is the right size for the
    local subset; for the full multi-million-chunk corpus a SQLite-backed
    resolver can be substituted behind the same interface.
    """

    def __init__(self, corpus_path: Path, load: bool = True) -> None:
        self.corpus_path = Path(corpus_path)
        self._df = None
        self._id_to_doc = None
        if load and self.corpus_path.is_file():
            self.load()

    # -- build ---------------------------------------------------------

    def load(self) -> "CorpusIndex":
        import pandas as pd

        df = pd.read_parquet(self.corpus_path)
        df["breadcrumb_str"] = df["breadcrumb"].map(
            lambda b: " > ".join(b) if isinstance(b, (list, tuple)) else (str(b) if b is not None else "")
        )
        self._df = df
        self._id_to_doc = dict(zip(df["id"], df["document_id"]))
        return self

    @property
    def loaded(self) -> bool:
        return self._df is not None

    @property
    def n_chunks(self) -> int:
        return 0 if self._df is None else len(self._df)

    def document_ids(self) -> "pd.Series":  # noqa: F821
        return self._df["document_id"]

    def document_id(self, chunk_id: str) -> Optional[str]:
        if self._id_to_doc is None:
            return None
        return self._id_to_doc.get(chunk_id)

    def resolve(self, chunk_ids: Sequence[str], include_text: bool = False) -> Dict[str, Dict[str, Any]]:
        """Return ``{chunk_id: metadata}`` for the requested ids."""
        if self._df is None or not chunk_ids:
            return {}
        cols = ["document_id", "chunk_type", "breadcrumb_str"]
        if include_text:
            cols.append("text")
        subset = self._df.loc[self._df["id"].isin(list(chunk_ids)), ["id"] + cols]
        out: Dict[str, Dict[str, Any]] = {}
        for row in subset.itertuples(index=False):
            record: Dict[str, Any] = {
                "document_id": row.document_id,
                "chunk_type": row.chunk_type,
                "breadcrumb": row.breadcrumb_str,
            }
            if include_text:
                record["text"] = row.text
            out[row.id] = record
        return out

    def iter_id_text(self, text_field: str = "text") -> Iterator[tuple[str, str]]:
        """Yield ``(chunk_id, text)`` in row order (used by BM25 builds)."""
        if self._df is None:
            return
        for row in self._df[["id", text_field]].itertuples(index=False):
            text = row[1]
            if text is None:
                text = ""
            yield row[0], text


# ----------------------------------------------------------------------
# Diagnostics / safety checks
# ----------------------------------------------------------------------

def run_diagnostics(
    chunk_files: Sequence[Path],
    embedding_files: Sequence[Path],
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the mandatory safety checks before an index build.

    Does not modify data. Returns aggregate counts and (small) samples of
    any problems found.
    """
    chunk_files = list(chunk_files)
    embedding_files = list(embedding_files)
    if limit is not None and limit > 0:
        chunk_files = chunk_files[:limit]

    chunk_stem_set = stems(chunk_files)
    emb_by_stem = {embedding_stem(p): p for p in embedding_files}
    emb_stem_set = set(emb_by_stem)

    missing_emb = sorted(chunk_stem_set - emb_stem_set)
    orphan_emb = sorted(emb_stem_set - chunk_stem_set)

    n_missing_emb = 0
    n_orphan_chunk_ids = 0
    n_missing_chunk_ids = 0
    n_dup_chunk_ids = 0
    n_dup_emb_ids = 0
    n_nonfinite = 0
    n_bad_dim = 0
    n_files = 0
    n_eligible = 0
    n_emb = 0

    problems: Dict[str, List[str]] = {
        "missing_embeddings": missing_emb,
        "orphan_embeddings": orphan_emb,
        "id_mismatch_files": [],
        "duplicate_chunk_ids": [],
        "duplicate_embedding_ids": [],
        "nonfinite_files": [],
        "bad_dimension_files": [],
    }

    for chunk_path in chunk_files:
        stem = chunk_path.stem
        emb_path = emb_by_stem.get(stem)
        try:
            elig_table = read_eligible_table(chunk_path)
        except Exception as exc:  # noqa: BLE001
            problems["id_mismatch_files"].append(f"{chunk_path.name}: read error {exc!r}")
            continue

        n_files += 1
        elig_ids = elig_table["id"].to_pylist()
        n_eligible += len(elig_ids)

        if len(set(elig_ids)) != len(elig_ids):
            n_dup_chunk_ids += 1
            problems["duplicate_chunk_ids"].append(chunk_path.name)

        if emb_path is None or not emb_path.is_file():
            n_missing_emb += 1
            continue

        try:
            emb_table = pq.read_table(str(emb_path), columns=["chunk_id", "embedding"])
        except Exception as exc:  # noqa: BLE001
            problems["id_mismatch_files"].append(f"{emb_path.name}: read error {exc!r}")
            continue

        emb_ids = emb_table["chunk_id"].to_pylist()
        n_emb += len(emb_ids)

        if len(set(emb_ids)) != len(emb_ids):
            n_dup_emb_ids += 1
            problems["duplicate_embedding_ids"].append(emb_path.name)

        elig_set = set(elig_ids)
        emb_set = set(emb_ids)
        n_orphan_chunk_ids += len(emb_set - elig_set)
        n_missing_chunk_ids += len(elig_set - emb_set)
        if elig_set != emb_set:
            problems["id_mismatch_files"].append(emb_path.name)

        # Dimension + finiteness over the whole file (vectorized).
        if emb_table.num_rows > 0:
            flat = pc.list_flatten(emb_table["embedding"])
            arr = flat.to_numpy()
            dim = len(arr) // emb_table.num_rows
            if dim != EMBEDDING_DIM:
                n_bad_dim += 1
                problems["bad_dimension_files"].append(emb_path.name)
            if not bool(np.isfinite(arr).all()):
                n_nonfinite += 1
                problems["nonfinite_files"].append(emb_path.name)

    return {
        "chunk_files": n_files,
        "chunk_files_missing_embedding": n_missing_emb,
        "orphan_embedding_files": len(orphan_emb),
        "eligible_chunks": n_eligible,
        "embedding_rows": n_emb,
        "orphan_embedding_ids": n_orphan_chunk_ids,
        "missing_embedding_ids": n_missing_chunk_ids,
        "duplicate_chunk_id_files": n_dup_chunk_ids,
        "duplicate_embedding_id_files": n_dup_emb_ids,
        "nonfinite_files": n_nonfinite,
        "bad_dimension_files": n_bad_dim,
        "problems": problems,
    }


def write_diagnostics_report(diag: Dict[str, Any], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
