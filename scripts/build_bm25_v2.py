#!/usr/bin/env python3
"""Build the BM25 sparse index over the v2 chunks (chunks_v2) in parallel.

Writes BM25Index-compatible files (bm25_tf.npz, bm25_vocab.pkl,
bm25_idf.npy, bm25_doc_lengths.npy, bm25_chunk_ids.npy, bm25_terms.txt,
bm25_meta.json) into --out-dir, so src.retrieval.sparse.BM25Index.load()
can read them unchanged.

Why this exists: the checked-in index/bm25 was built over the v1 corpus
(its chunk ids do not exist in the pgvector medrag.chunks table), so the
hybrid query resolved its hits to "doc ?". This rebuilds BM25 over the
actual v2 chunks that are in pgvector.

Usage:
    python scripts/build_bm25_v2.py                 # chunks_v2 -> index/bm25_v2
    python scripts/build_bm25_v2.py --chunks-dir chunks --out-dir index/bm25
    python scripts/build_bm25_v2.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.retrieval.sparse import (
    DEFAULT_TOKEN_PATTERN,
    DOC_LEN_FILENAME,
    IDF_FILENAME,
    IDS_FILENAME,
    META_FILENAME,
    TERMS_FILENAME,
    TF_FILENAME,
    VOCAB_FILENAME,
)


def _partition(files: List[Path], workers: int, target_files: int) -> List[List[Path]]:
    n_tasks = max(workers, (len(files) + target_files - 1) // target_files)
    size = max(1, (len(files) + n_tasks - 1) // n_tasks)
    return [files[i:i + size] for i in range(0, len(files), size)]


def _bounded(pool, fn, tasks, window):
    """Yield (task, result) pairs with only `window` tasks in flight."""
    it = iter(tasks)
    pending = []
    while len(pending) < window:
        try:
            t = next(it)
        except StopIteration:
            break
        pending.append((t, pool.submit(fn, t)))
    while pending:
        t, fut = pending.pop(0)
        yield t, fut.result()
        try:
            t2 = next(it)
        except StopIteration:
            continue
        pending.append((t2, pool.submit(fn, t2)))


def pq_read(path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path), columns=["id", "text"])


# Worker globals, populated once per process via ProcessPoolExecutor's
# initializer (avoid pickling the vocab for every task).
_VOCAB: Dict[str, int] = {}
_TOKEN_RE = re.compile(DEFAULT_TOKEN_PATTERN)


def _init_worker(vocab: Dict[str, int]) -> None:
    global _VOCAB
    _VOCAB = vocab


def _pass1(paths: List[Path]) -> Tuple[List[str], List[int], Dict[str, int], int]:
    """(ids, doc_lengths, df, nnz) for the file batch (tokenizes once)."""
    pat = re.compile(DEFAULT_TOKEN_PATTERN)
    ids: List[str] = []
    lengths: List[int] = []
    df: Dict[str, int] = {}
    nnz = 0
    for p in paths:
        t = pq_read(str(p))
        text_col = t["text"]
        for i, cid in enumerate(t["id"].to_pylist()):
            toks = pat.findall((text_col[i].as_py() or "").lower())
            lengths.append(len(toks))
            ids.append(cid)
            uniq = set(toks)
            nnz += len(uniq)
            for term in uniq:
                df[term] = df.get(term, 0) + 1
    return ids, lengths, df, nnz


def _pass2(paths: List[Path]) -> Any:
    """Return flat postings: (term_cols, doc_rows_local, counts) int32/float32.

    Doc rows are local to this task (0..n_docs-1); the caller adds the
    task's global document offset.
    """
    pat = _TOKEN_RE
    vocab = _VOCAB
    cols: List[int] = []
    rows: List[int] = []
    data: List[float] = []
    d = 0
    for p in paths:
        t = pq_read(str(p))
        text_col = t["text"]
        for i in range(t.num_rows):
            toks = pat.findall((text_col[i].as_py() or "").lower())
            if toks:
                tids = [vocab[tk] for tk in toks if tk in vocab]
                if tids:
                    arr = np.asarray(tids, dtype=np.int32)
                    uniq, counts = np.unique(arr, return_counts=True)
                    cols.extend(uniq.tolist())
                    rows.extend([d] * len(uniq))
                    data.extend(counts.tolist())
            d += 1
    return (
        np.asarray(cols, dtype=np.int32),
        np.asarray(rows, dtype=np.int32),
        np.asarray(data, dtype=np.float32),
    )


def build(chunks_dir: Path, out_dir: Path, workers: int, k1: float, b: float) -> Dict[str, int]:
    import scipy.sparse as sp

    files = sorted(chunks_dir.glob("*.parquet"))
    if not files:
        print(f"no .parquet files in {chunks_dir}")
        return {}

    t0 = time.time()
    window = max(workers * 2, 4)
    partitions = _partition(files, workers, target_files=60)
    print(f"{len(files):,} parquet files in {len(partitions)} tasks ({workers} workers)")

    # ---- pass 1: df + doc lengths + ids --------------------------------
    ids: List[str] = []
    lengths: List[int] = []
    df: Dict[str, int] = {}
    nnz_total = 0
    task_doc_counts: List[int] = []   # per-task doc counts, aligned to results order
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, (ids_i, lens_i, df_i, nnz_i) in _bounded(pool, _pass1, partitions, window):
            del task
            ids.extend(ids_i)
            lengths.extend(lens_i)
            nnz_total += nnz_i
            task_doc_counts.append(len(ids_i))
            for term, c in df_i.items():
                df[term] = df.get(term, 0) + c
            done += 1
            if done % 20 == 0 or done == len(partitions):
                print(f"  pass1 {done}/{len(partitions)} tasks ({len(ids):,} docs)", flush=True)

    n_docs = len(ids)
    terms = sorted(df)
    vocab = {t: i for i, t in enumerate(terms)}
    v = len(terms)
    df_arr = np.fromiter((df[t] for t in terms), dtype=np.float64, count=v)
    idf = (np.log((n_docs - df_arr + 0.5) / (df_arr + 0.5)) + 1.0).astype(np.float32)
    avgdl = float(np.mean(lengths)) if n_docs else 0.0
    print(f"  pass1 done: {n_docs:,} docs, vocab {v:,}, nnz {nnz_total:,}, avgdl {avgdl:.1f}")
    print(f"  (idf + vocab computed in {time.time() - t0:.0f}s)")
    del df, df_arr

    # ---- pass 2: flat postings -> preallocated COO -> one tocsc() --------
    # Memory-bounded: only one task's postings are in flight (bounded window)
    # and the final matrix materializes from a single coo->csc conversion.
    print(f"  allocating {nnz_total:,} postings slots ({nnz_total * 12 / 1e9:.1f} GB)...")
    rows_arr = np.empty(nnz_total, dtype=np.int32)   # doc indices (global)
    cols_arr = np.empty(nnz_total, dtype=np.int32)   # term indices
    data_arr = np.empty(nnz_total, dtype=np.float32)  # term counts
    cursor = 0
    doc_offset = 0
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(vocab,)) as pool:
        for task, (cols, rows_local, cnts) in _bounded(pool, _pass2, partitions, window):
            del task
            k = len(cnts)
            if k:
                cols_arr[cursor:cursor + k] = cols
                rows_arr[cursor:cursor + k] = rows_local + doc_offset
                data_arr[cursor:cursor + k] = cnts
                cursor += k
            doc_offset += task_doc_counts[done]
            done += 1
            if done % 20 == 0 or done == len(partitions):
                print(f"  pass2 {done}/{len(partitions)} tasks "
                      f"({cursor:,}/{nnz_total:,} postings)", flush=True)
    print(f"  pass2 done: {cursor:,} postings -> coo -> csc...")
    coo = sp.coo_matrix(
        (data_arr[:cursor], (rows_arr[:cursor], cols_arr[:cursor])),
        shape=(n_docs, v),
    )
    del rows_arr, cols_arr, data_arr
    tf = coo.tocsc()
    del coo
    print(f"  tf_csc shape={tf.shape} nnz={tf.nnz:,}")

    # ---- persist (same files BM25Index.load expects) ---------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(out_dir / TF_FILENAME), tf)
    np.save(str(out_dir / IDF_FILENAME), idf, allow_pickle=False)
    np.save(str(out_dir / DOC_LEN_FILENAME), np.asarray(lengths, dtype=np.int32), allow_pickle=False)
    np.save(str(out_dir / IDS_FILENAME), np.asarray(ids, dtype=object), allow_pickle=True)
    with open(out_dir / VOCAB_FILENAME, "wb") as fh:
        pickle.dump(vocab, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / TERMS_FILENAME).write_text("\n".join(terms), encoding="utf-8")
    meta = {
        "k1": k1,
        "b": b,
        "token_pattern": DEFAULT_TOKEN_PATTERN,
        "min_df": 1,
        "n_docs": n_docs,
        "vocab_size": v,
        "avgdl": avgdl,
        "nnz": int(tf.nnz),
        "build_seconds": round(time.time() - t0, 3),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "idf_formula": "ln((N-df+0.5)/(df+0.5))+1",
        "source": str(chunks_dir),
    }
    (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir} in {time.time() - t0:.0f}s")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--chunks-dir", type=Path, default=Path("chunks_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("index/bm25_v2"))
    parser.add_argument("--workers", type=int, default=min(8, (__import__("os").cpu_count() or 8)))
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args(argv)
    meta = build(args.chunks_dir, args.out_dir, args.workers, args.k1, args.b)
    if not meta:
        return 1
    print(f"\nDone: {meta['n_docs']:,} docs, vocab {meta['vocab_size']:,} in {meta['build_seconds']}s")
    print(f"Now query with: python scripts/pg_hybrid_query.py '...' --bm25-dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())