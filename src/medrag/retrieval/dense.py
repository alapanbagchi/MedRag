"""Dense retrieval index built on FAISS.

Index choices for the first implementation:

* ``flat``  - ``faiss.IndexFlatIP`` over L2-normalized float32 vectors.
  Exact, reproducible, trivial to rebuild. Inner product of unit vectors
  equals cosine similarity.
* ``hnsw``  - ``faiss.IndexHNSWFlat`` over the same normalized vectors.
  Approximate, faster for the full corpus, with configurable M /
  efConstruction / efSearch.

The corpus embeddings are mean-of-[CLS] MedCPT vectors that are *not*
already normalized, so every vector is L2-normalized at build time and every
query vector is normalized at search time. This makes ``IndexFlatIP`` a
cosine-similarity index regardless of the original scale.

Memory (local 883,975 x 768 float32): ~2.7 GB for the vector table plus a
small HNSW graph when used. Vectors are added to FAISS in batches so peak
memory stays close to the final index size.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from medrag.models import ScoredChunk

EMBEDDING_DIM = 768
INDEX_FILENAME = "dense.faiss"
IDS_FILENAME = "dense_ids.npy"
META_FILENAME = "dense_meta.json"


def _read_embedding_file(path: Path) -> tuple[List[str], np.ndarray]:
    """Return (chunk_ids, float32 matrix of shape (n, 768))."""
    table = pq.read_table(str(path), columns=["chunk_id", "embedding"])
    n = table.num_rows
    if n == 0:
        return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    flat = pc.list_flatten(table["embedding"]).to_numpy()
    mat = flat.reshape(n, EMBEDDING_DIM).astype(np.float32, copy=False)
    ids = table["chunk_id"].to_pylist()
    return ids, mat


class DenseIndex:
    """FAISS-backed dense index with an aligned chunk-id array."""

    def __init__(
        self,
        index: Any,
        chunk_ids: np.ndarray,
        meta: Dict[str, Any],
    ) -> None:
        self.index = index
        self.chunk_ids = chunk_ids
        self.meta = meta

    # -- properties ----------------------------------------------------

    @property
    def n_total(self) -> int:
        return int(self.index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    @property
    def index_type(self) -> str:
        return self.meta.get("index_type", "flat")

    @property
    def normalize(self) -> bool:
        return bool(self.meta.get("normalize", True))

    # -- build ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        embedding_files: Sequence[Path],
        out_dir: Path,
        index_type: str = "flat",
        normalize: bool = True,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        batch_size: int = 65536,
    ) -> "DenseIndex":
        import faiss

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if index_type == "flat":
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(EMBEDDING_DIM, hnsw_m)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"unknown index_type: {index_type!r} (use 'flat' or 'hnsw')")

        chunk_ids: List[str] = []
        pending_ids: List[str] = []
        pending_vecs: List[np.ndarray] = []
        pending_rows = 0
        start = time.time()

        def flush() -> None:
            nonlocal pending_rows
            if pending_rows == 0:
                return
            block = np.vstack(pending_vecs).astype(np.float32, copy=False)
            if normalize:
                faiss.normalize_L2(block)
            index.add(block)
            chunk_ids.extend(pending_ids)
            pending_ids.clear()
            pending_vecs.clear()
            pending_rows = 0

        for path in embedding_files:
            ids, mat = _read_embedding_file(Path(path))
            if mat.shape[0] == 0:
                continue
            pending_ids.extend(ids)
            pending_vecs.append(mat)
            pending_rows += mat.shape[0]
            if pending_rows >= batch_size:
                flush()

        flush()
        build_seconds = time.time() - start

        # Verify alignment before writing anything.
        if index.ntotal != len(chunk_ids):
            raise RuntimeError(
                f"internal mismatch: index has {index.ntotal} vectors but "
                f"{len(chunk_ids)} chunk ids were collected"
            )

        ids_arr = np.asarray(chunk_ids, dtype=object)
        meta = {
            "index_type": index_type,
            "metric": "inner_product",
            "cosine": True,
            "normalize": normalize,
            "dimension": EMBEDDING_DIM,
            "n_total": int(index.ntotal),
            "build_seconds": round(build_seconds, 3),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hnsw": (
                {"M": hnsw_m, "efConstruction": ef_construction, "efSearch": ef_search}
                if index_type == "hnsw"
                else None
            ),
            "source_files": len(embedding_files),
        }

        faiss.write_index(index, str(out_dir / INDEX_FILENAME))
        np.save(str(out_dir / IDS_FILENAME), ids_arr, allow_pickle=True)
        (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return cls(index, ids_arr, meta)

    # -- load ----------------------------------------------------------

    @classmethod
    def load(cls, out_dir: Path) -> "DenseIndex":
        import faiss

        out_dir = Path(out_dir)
        meta_path = out_dir / META_FILENAME
        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(out_dir / INDEX_FILENAME))
        if isinstance(index, faiss.IndexHNSWFlat) and "hnsw" in meta and meta["hnsw"]:
            index.hnsw.efSearch = int(meta["hnsw"].get("efSearch", 64))
        ids = np.load(str(out_dir / IDS_FILENAME), allow_pickle=True)
        return cls(index, ids, meta)

    # -- search --------------------------------------------------------

    def search(self, query_vectors: np.ndarray, top_k: int) -> List[List[ScoredChunk]]:
        """Search with one or more query vectors.

        ``query_vectors`` must be float32 of shape (n_queries, 768). Returns
        one ranked list per query.
        """
        import faiss

        if top_k <= 0:
            return [[] for _ in range(len(query_vectors))]

        q = np.asarray(query_vectors, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"query dimension {q.shape[1]} != index dimension {self.dimension}")
        if self.normalize:
            faiss.normalize_L2(q)

        top_k = min(top_k, self.n_total)
        if top_k == 0:
            return [[] for _ in range(q.shape[0])]

        scores, indices = self.index.search(q, top_k)
        out: List[List[ScoredChunk]] = []
        for row_scores, row_indices in zip(scores, indices):
            ranked: List[ScoredChunk] = []
            for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
                if idx < 0:
                    continue
                ranked.append(
                    ScoredChunk(
                        chunk_id=str(self.chunk_ids[idx]),
                        score=float(score),
                        retrieval_method="dense",
                        rank=rank,
                    )
                )
            out.append(ranked)
        return out

    def search_single(self, query_vector: np.ndarray, top_k: int) -> List[ScoredChunk]:
        return self.search(query_vector, top_k)[0]
