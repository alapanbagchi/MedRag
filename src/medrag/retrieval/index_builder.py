"""Orchestrated, resumable index build.

Produces, under a single index directory:

    corpus.parquet       consolidated retrieval-eligible metadata/text
    dense.faiss          FAISS dense index
    dense_ids.npy        chunk ids aligned with FAISS internal ids
    dense_meta.json      dense index parameters/build stats
    bm25/                BM25 sparse index files
    diagnostics.json     mandatory pre-build safety diagnostics

Each stage is independently resumable: if its output already exists (and is
non-empty / consistent) it is skipped unless ``force`` is set. The full
build refuses to proceed when diagnostics find missing/orphan embeddings,
duplicate ids, non-finite vectors or wrong dimensions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pyarrow.parquet as pq

from medrag.retrieval.corpus import (
    CORPUS_FILENAME,
    build_corpus_parquet,
    discover_chunks,
    discover_embeddings,
    run_diagnostics,
)
from medrag.retrieval.dense import DenseIndex
from medrag.retrieval.sparse import BM25Index

CORPUS_META_FILENAME = "corpus_meta.json"
DIAGNOSTICS_FILENAME = "diagnostics.json"
BM25_DIRNAME = "bm25"


def _iter_corpus_text(corpus_path: Path, text_field: str, batch_size: int = 100_000):
    """Stream (chunk_id, text) from the consolidated corpus without loading it all."""
    pf = pq.ParquetFile(str(corpus_path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=["id", text_field]):
        ids = batch.column("id").to_pylist()
        texts = batch.column(text_field).to_pylist()
        for chunk_id, text in zip(ids, texts):
            yield chunk_id, "" if text is None else text


# ----------------------------------------------------------------------
# Stage builders
# ----------------------------------------------------------------------

def build_corpus(
    chunks_dir: Path,
    index_dir: Path,
    limit: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    out_path = index_dir / CORPUS_FILENAME
    meta_path = index_dir / CORPUS_META_FILENAME

    if out_path.is_file() and not force:
        existing = pq.read_metadata(str(out_path))
        return {"output": str(out_path), "skipped": True, "chunks": existing.num_rows}

    files = discover_chunks(chunks_dir)
    stats = build_corpus_parquet(files, out_path, limit=limit)
    meta = {**stats, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "limit": limit}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stats["skipped"] = False
    return stats


def build_dense(
    embeddings_dir: Path,
    index_dir: Path,
    index_type: str = "flat",
    limit: Optional[int] = None,
    force: bool = False,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    use_pgvector: bool = False,
    corpus_path: Optional[Path] = None,
) -> Dict[str, Any]:
    import os

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # pgvector backend
    if use_pgvector or os.environ.get("MEDRAG_BACKEND", "").lower() == "pgvector":
        try:
            from medrag.retrieval.dense_pgvector import PgDenseIndex
            from medrag.retrieval.pgvector_store import PgVectorStore

            store = PgVectorStore.from_env()
            store.connect()
            store.ensure_schema()

            n_existing = store.count_embeddings()
            if n_existing > 0 and not force:
                return {
                    "output": "pgvector",
                    "skipped": True,
                    "n_total": n_existing,
                    "index_type": "pgvector_ivfflat",
                }

            files = discover_embeddings(embeddings_dir)
            if limit is not None and limit > 0:
                files = files[:limit]

            # Use the first embedding file's parent as the embedding dir
            emb_dir = files[0].parent if files else embeddings_dir
            dense = PgDenseIndex.build(
                embedding_files=files,
                out_dir=index_dir,
                corpus_path=corpus_path,
                batch_size=65536,
            )
            return {
                "output": "pgvector",
                "skipped": False,
                "n_total": dense.n_total,
                "index_type": "pgvector_ivfflat",
            }
        except Exception as exc:
            print(f"  pgvector build failed ({exc}), falling back to FAISS")

    # FAISS backend (default)
    import faiss

    index_path = index_dir / "dense.faiss"
    if index_path.is_file() and not force:
        idx = faiss.read_index(str(index_path))
        return {"output": str(index_path), "skipped": True, "n_total": int(idx.ntotal)}

    files = discover_embeddings(embeddings_dir)
    if limit is not None and limit > 0:
        files = files[:limit]
    dense = DenseIndex.build(
        files,
        index_dir,
        index_type=index_type,
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    return {
        "output": str(index_path),
        "skipped": False,
        "n_total": dense.n_total,
        "index_type": dense.index_type,
    }


def build_sparse(
    index_dir: Path,
    text_field: str = "text",
    force: bool = False,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, Any]:
    index_dir = Path(index_dir)
    bm25_dir = index_dir / BM25_DIRNAME
    bm25_dir.mkdir(parents=True, exist_ok=True)
    meta_path = bm25_dir / "bm25_meta.json"

    if meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"output": str(bm25_dir), "skipped": True, "n_docs": meta.get("n_docs")}

    corpus_path = index_dir / CORPUS_FILENAME
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus index not found: {corpus_path}")

    def factory():
        return _iter_corpus_text(corpus_path, text_field)

    bm25 = BM25Index.build(factory, bm25_dir, k1=k1, b=b)
    return {
        "output": str(bm25_dir),
        "skipped": False,
        "n_docs": bm25.n_docs,
        "vocab_size": bm25.vocab_size,
    }


# ----------------------------------------------------------------------
# Full build
# ----------------------------------------------------------------------

def build_all(
    chunks_dir: Path,
    embeddings_dir: Path,
    index_dir: Path,
    dense_type: str = "flat",
    text_field: str = "text",
    limit: Optional[int] = None,
    force: bool = False,
    run_full_diagnostics: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
    use_pgvector: bool = False,
) -> Dict[str, Any]:
    """Build corpus + dense + sparse with resumability and safety checks."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    report: Dict[str, Any] = {
        "index_dir": str(index_dir),
        "dense_type": dense_type,
        "text_field": text_field,
        "use_pgvector": use_pgvector,
    }

    if run_full_diagnostics:
        chunk_files = discover_chunks(chunks_dir)
        emb_files = discover_embeddings(embeddings_dir)
        if limit is not None and limit > 0:
            chunk_files = chunk_files[:limit]
            emb_files = emb_files[:limit]
        diag = run_diagnostics(chunk_files, emb_files)
        (index_dir / DIAGNOSTICS_FILENAME).write_text(
            json.dumps(diag, indent=2), encoding="utf-8"
        )
        report["diagnostics"] = {k: v for k, v in diag.items() if k != "problems"}

        fatal = []
        if diag["chunk_files_missing_embedding"]:
            fatal.append(f"{diag['chunk_files_missing_embedding']} chunks missing embedding file")
        if diag["orphan_embedding_files"]:
            fatal.append(f"{diag['orphan_embedding_files']} orphan embedding files")
        if diag["orphan_embedding_ids"] or diag["missing_embedding_ids"]:
            fatal.append(
                f"id-set mismatch ({diag['orphan_embedding_ids']} orphan, "
                f"{diag['missing_embedding_ids']} missing)"
            )
        if diag["duplicate_chunk_id_files"] or diag["duplicate_embedding_id_files"]:
            fatal.append("duplicate ids detected")
        if diag["nonfinite_files"] or diag["bad_dimension_files"]:
            fatal.append("malformed vectors detected")
        if fatal:
            raise RuntimeError(
                "Pre-build diagnostics failed: " + "; ".join(fatal) +
                f". See {index_dir / DIAGNOSTICS_FILENAME}"
            )

    corpus_path = index_dir / CORPUS_FILENAME

    report["corpus"] = build_corpus(chunks_dir, index_dir, limit=limit, force=force)
    report["dense"] = build_dense(
        embeddings_dir,
        index_dir,
        index_type=dense_type,
        limit=limit,
        force=force,
        use_pgvector=use_pgvector,
        corpus_path=corpus_path if corpus_path.is_file() else None,
    )
    report["sparse"] = build_sparse(
        index_dir, text_field=text_field, force=force, k1=k1, b=b
    )
    report["elapsed_seconds"] = round(time.time() - started, 3)
    return report
