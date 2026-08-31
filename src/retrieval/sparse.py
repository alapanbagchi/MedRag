"""Sparse (BM25) retrieval index backed by a scipy CSC matrix.

Memory-efficient by design: documents are never stored as nested Python
lists of string tokens. The index is a CSC term-document matrix
(columns = vocabulary terms, rows = documents) plus flat numpy arrays for
idf and document lengths. Query-time scoring walks only the columns of the
query terms, so it never materializes a dense (n_docs x vocab) matrix.

BM25 variant: Robertson/Walker scoring with ``idf = ln((N - df + 0.5) /
(df + 0.5)) + 1`` (BM25+), ``k1=1.5``, ``b=0.75``, binary query-term
presence. Tokenization is lowercase ``[a-z0-9]+`` (configurable).

Files (in ``out_dir/``):
    bm25_tf.npz         CSC term-frequency matrix (float32)
    bm25_idf.npy        per-term idf (float32)
    bm25_doc_lengths.npy  per-document token counts (int32)
    bm25_chunk_ids.npy  aligned document -> chunk_id mapping
    bm25_vocab.pkl      term -> column id (query-time lookup)
    bm25_terms.txt      column id -> term (transparency/debugging)
    bm25_meta.json      parameters + build stats
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.lib.models import ScoredChunk

DEFAULT_TOKEN_PATTERN = r"[a-z0-9]+"

META_FILENAME = "bm25_meta.json"
TF_FILENAME = "bm25_tf.npz"
IDF_FILENAME = "bm25_idf.npy"
DOC_LEN_FILENAME = "bm25_doc_lengths.npy"
IDS_FILENAME = "bm25_chunk_ids.npy"
VOCAB_FILENAME = "bm25_vocab.pkl"
TERMS_FILENAME = "bm25_terms.txt"


def tokenize(text: str, pattern: re.Pattern) -> List[str]:
    return pattern.findall(text.lower())


class BM25Index:
    """scipy-backed BM25 index."""

    def __init__(
        self,
        tf_csc: Any,
        vocab: Dict[str, int],
        idf: np.ndarray,
        doc_lengths: np.ndarray,
        chunk_ids: np.ndarray,
        meta: Dict[str, Any],
    ) -> None:
        self.tf_csc = tf_csc
        self.vocab = vocab
        self.idf = idf
        self.doc_lengths = doc_lengths
        self.chunk_ids = chunk_ids
        self.meta = meta

    # -- properties ----------------------------------------------------

    @property
    def n_docs(self) -> int:
        return int(self.tf_csc.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.tf_csc.shape[1])

    @property
    def k1(self) -> float:
        return float(self.meta.get("k1", 1.5))

    @property
    def b(self) -> float:
        return float(self.meta.get("b", 0.75))

    @property
    def avgdl(self) -> float:
        return float(self.meta.get("avgdl", 0.0))

    # -- build ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        text_iter_factory: Callable[[], Iterable[Tuple[str, str]]],
        out_dir: Path,
        k1: float = 1.5,
        b: float = 0.75,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
        min_df: int = 1,
    ) -> "BM25Index":
        import scipy.sparse as sp

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        token_re = re.compile(token_pattern)

        # ---- pass 1: document frequency + document lengths ------------
        df: Dict[str, int] = {}
        doc_lengths: List[int] = []
        chunk_ids: List[str] = []
        start = time.time()

        for chunk_id, text in text_iter_factory():
            tokens = tokenize(text, token_re)
            doc_lengths.append(len(tokens))
            chunk_ids.append(chunk_id)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        n_docs = len(chunk_ids)
        if min_df > 1:
            df = {t: c for t, c in df.items() if c >= min_df}

        terms = sorted(df)
        vocab = {t: i for i, t in enumerate(terms)}
        v = len(terms)

        df_arr = np.fromiter((df[t] for t in terms), dtype=np.float64, count=v)
        idf = np.log((n_docs - df_arr + 0.5) / (df_arr + 0.5)) + 1.0
        idf = idf.astype(np.float32)

        nnz = int(df_arr.sum())
        indptr = np.zeros(v + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(df_arr, dtype=np.int64)
        cursors = indptr[:-1].copy()
        indices = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float32)

        # ---- pass 2: fill CSC columns --------------------------------
        for d, (_, text) in enumerate(text_iter_factory()):
            tokens = tokenize(text, token_re)
            if not tokens:
                continue
            tids_list = [vocab[t] for t in tokens if t in vocab]
            if not tids_list:
                continue
            tids = np.asarray(tids_list, dtype=np.int32)
            uniq, counts = np.unique(tids, return_counts=True)
            for t, c in zip(uniq, counts):
                pos = cursors[t]
                indices[pos] = d
                data[pos] = float(c)
                cursors[t] += 1

        tf_csc = sp.csc_matrix((data, indices, indptr), shape=(n_docs, v))

        doc_lengths_arr = np.asarray(doc_lengths, dtype=np.int32)
        avgdl = float(doc_lengths_arr.mean()) if n_docs else 0.0
        chunk_ids_arr = np.asarray(chunk_ids, dtype=object)

        meta = {
            "k1": k1,
            "b": b,
            "token_pattern": token_pattern,
            "min_df": min_df,
            "n_docs": n_docs,
            "vocab_size": v,
            "avgdl": avgdl,
            "nnz": nnz,
            "build_seconds": round(time.time() - start, 3),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "idf_formula": "ln((N-df+0.5)/(df+0.5))+1",
        }

        # ---- persist --------------------------------------------------
        sp.save_npz(str(out_dir / TF_FILENAME), tf_csc)
        np.save(str(out_dir / IDF_FILENAME), idf, allow_pickle=False)
        np.save(str(out_dir / DOC_LEN_FILENAME), doc_lengths_arr, allow_pickle=False)
        np.save(str(out_dir / IDS_FILENAME), chunk_ids_arr, allow_pickle=True)
        with open(out_dir / VOCAB_FILENAME, "wb") as fh:
            pickle.dump(vocab, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (out_dir / TERMS_FILENAME).write_text("\n".join(terms), encoding="utf-8")
        (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return cls(tf_csc, vocab, idf, doc_lengths_arr, chunk_ids_arr, meta)

    # -- load ----------------------------------------------------------

    @classmethod
    def load(cls, out_dir: Path) -> "BM25Index":
        import scipy.sparse as sp

        out_dir = Path(out_dir)
        meta_path = out_dir / META_FILENAME
        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tf_csc = sp.load_npz(str(out_dir / TF_FILENAME))
        with open(out_dir / VOCAB_FILENAME, "rb") as fh:
            vocab = pickle.load(fh)
        idf = np.load(str(out_dir / IDF_FILENAME))
        doc_lengths = np.load(str(out_dir / DOC_LEN_FILENAME))
        chunk_ids = np.load(str(out_dir / IDS_FILENAME), allow_pickle=True)
        return cls(tf_csc, vocab, idf, doc_lengths, chunk_ids, meta)

    # -- scoring -------------------------------------------------------

    def _score(self, query_terms: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        indptr = self.tf_csc.indptr
        indices = self.tf_csc.indices
        data = self.tf_csc.data
        dl = self.doc_lengths
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        seen: set = set()
        for term in query_terms:
            if term in seen:
                continue
            seen.add(term)
            t = self.vocab.get(term)
            if t is None:
                continue
            start, end = indptr[t], indptr[t + 1]
            if end == start:
                continue
            doc_ids = indices[start:end]
            tf = data[start:end]
            denom = tf + k1 * (1.0 - b + b * dl[doc_ids] / avgdl)
            contrib = self.idf[t] * tf * (k1 + 1.0) / denom
            scores[doc_ids] += contrib
        return scores

    def search(self, queries: str | Sequence[str], top_k: int) -> List[List[ScoredChunk]]:
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        token_re = re.compile(self.meta.get("token_pattern", DEFAULT_TOKEN_PATTERN))
        out: List[List[ScoredChunk]] = []
        top_k = min(top_k, self.n_docs)

        for query in query_list:
            terms = tokenize(query, token_re)
            scores = self._score(terms)
            if top_k <= 0:
                out.append([])
                continue
            if self.n_docs <= top_k:
                order = np.argsort(-scores, kind="stable")
            else:
                order = np.argpartition(-scores, top_k - 1)[:top_k]
                order = order[np.argsort(-scores[order], kind="stable")]
            ranked: List[ScoredChunk] = []
            for rank, doc_idx in enumerate(order, start=1):
                ranked.append(
                    ScoredChunk(
                        chunk_id=str(self.chunk_ids[doc_idx]),
                        score=float(scores[doc_idx]),
                        retrieval_method="bm25",
                        rank=rank,
                    )
                )
            out.append(ranked)
        return out

    def search_single(self, query: str, top_k: int) -> List[ScoredChunk]:
        return self.search(query, top_k)[0]

    # ------------------------------------------------------------------
    # Restricted (in-paper) search — V2 local retrieval (spec section 19)
    # ------------------------------------------------------------------

    def _id_to_row_index(self) -> Dict[str, int]:
        """Lazily build chunk_id -> row-index mapping for restricted scoring."""
        if getattr(self, "_id_to_row_cache", None) is None:
            self._id_to_row_cache = {
                str(cid): i for i, cid in enumerate(self.chunk_ids.tolist())
            }
        return self._id_to_row_cache

    def search_restricted(
        self,
        query: str,
        chunk_ids: Sequence[str],
        top_k: int,
    ) -> List[ScoredChunk]:
        """BM25 scoring restricted to a subset of chunk ids.

        Used by V2 local search: score a query only within one selected paper
        instead of the whole 883k corpus. Scans only the term columns of the
        query and only for the requested row subset.
        """
        if not chunk_ids:
            return []
        id_to_row = self._id_to_row_index()
        rows = [id_to_row[c] for c in chunk_ids if c in id_to_row]
        if not rows:
            return []
        rows_arr = np.asarray(rows, dtype=np.int64)

        token_re = re.compile(self.meta.get("token_pattern", DEFAULT_TOKEN_PATTERN))
        terms = tokenize(query, token_re)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        indptr = self.tf_csc.indptr
        indices = self.tf_csc.indices
        data = self.tf_csc.data
        dl = self.doc_lengths
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        seen: set = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            t = self.vocab.get(term)
            if t is None:
                continue
            start, end = indptr[t], indptr[t + 1]
            if end == start:
                continue
            col_docs = indices[start:end]
            mask = np.isin(col_docs, rows_arr)
            if not bool(mask.any()):
                continue
            sel_docs = col_docs[mask]
            tf = data[start:end][mask]
            denom = tf + k1 * (1.0 - b + b * dl[sel_docs] / avgdl)
            contrib = self.idf[t] * tf * (k1 + 1.0) / denom
            scores[sel_docs] += contrib

        top_k = min(top_k, len(rows))
        if top_k <= 0:
            return []
        order = np.argsort(-scores[rows_arr], kind="stable")[:top_k]
        ranked: List[ScoredChunk] = []
        for rank, pos in enumerate(order, start=1):
            row = int(rows_arr[pos])
            ranked.append(
                ScoredChunk(
                    chunk_id=str(self.chunk_ids[row]),
                    score=float(scores[row]),
                    retrieval_method="bm25",
                    rank=rank,
                )
            )
        return ranked
