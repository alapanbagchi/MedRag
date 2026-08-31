"""SPLADE learned-sparse retrieval (NeuML/pubmedbert-base-splade).

Offline: every corpus chunk is encoded through the SPLADE masked-LM head into
a sparse term-weight vector (vocab-dim, ~1e2-1e3 non-zeros per doc) stored as
a scipy csr matrix.  Query time: encode the query the same way, then score =
query_sparse @ doc_matrix.T -> top-k.

Interface mirrors BM25Index (search_single(query, top_k) -> List[ScoredChunk])
so the retrieval service can swap bm25 <-> splade without changing callers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from src.lib.models import ScoredChunk

logger = logging.getLogger("src.splade")

DEFAULT_MODEL = "NeuML/pubmedbert-base-splade"
VERSION = 1
MAX_TOKENS = 256
DEFAULT_THRESHOLD = 0.1


def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_tokens: int = MAX_TOKENS,
    threshold: float = DEFAULT_THRESHOLD,
) -> sp.csr_matrix:
    """SPLADE-encode a batch of texts into a sparse (n, vocab) matrix.

    Standard SPLADE A3 encoding:
        w_t = log1p( sum_l ReLU(logits[l, t]) )   over all tokens
    entries with weight <= threshold are dropped.
    """
    import torch

    if not texts:
        return sp.csr_matrix((0, model.config.vocab_size), dtype=np.float32)

    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits

    relu = torch.relu(logits)
    summed = relu.sum(dim=1)
    vecs = torch.log1p(summed)
    vecs[vecs <= threshold] = 0.0

    return sp.csr_matrix(vecs.cpu().numpy(), dtype=np.float32)


class SPLADEIndex:
    """Sparse SPLADE index over the corpus (doc-term weight matrix)."""

    def __init__(
        self,
        doc_matrix: sp.csr_matrix,
        chunk_ids: np.ndarray,
        model_name: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.doc_matrix = doc_matrix.tocsr()
        self.chunk_ids = np.asarray(chunk_ids)
        self.model_name = model_name
        self.meta = dict(meta or {})
        self._model = None
        self._tokenizer = None

    @property
    def n_docs(self) -> int:
        return self.doc_matrix.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.doc_matrix.shape[1]

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._model, self._tokenizer

    @classmethod
    def build_from_corpus(
        cls,
        corpus: Any,
        out_dir: Path,
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        max_docs: Optional[int] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Any:
        """Encode the whole corpus -> save index to out_dir, return index."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("loading SPLADE model %s", model_name)
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
        logger.info("model loaded in %.1fs", time.time() - t0)

        matrices = []
        chunk_ids = []
        n = 0
        for cid, text in corpus.iter_id_text():
            if max_docs is not None and n >= max_docs:
                break
            chunk_ids.append(str(cid))
            matrices.append(encode_texts(model, tokenizer, [text], threshold=threshold))
            n += 1
            if n % (batch_size * 20) == 0:
                logger.info("encoded %d docs", n)

        doc_matrix = sp.vstack(matrices).tocsr()

        meta = {
            "model_name": model_name,
            "n_docs": int(doc_matrix.shape[0]),
            "vocab_size": int(doc_matrix.shape[1]),
            "nonzeros": int(doc_matrix.nnz),
            "threshold": threshold,
            "version": VERSION,
        }
        logger.info("built matrix n=%d nnz=%d", doc_matrix.shape[0], doc_matrix.nnz)

        idx = cls(doc_matrix, chunk_ids, model_name, meta=meta)
        idx.save(out_dir)
        return idx

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(out_dir / "splade_matrix.npz", self.doc_matrix)
        np.save(out_dir / "splade_chunk_ids.npy", np.asarray(self.chunk_ids))
        (out_dir / "splade_meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, out_dir: Path, model_name: Optional[str] = None) -> Any:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "splade_meta.json").read_text())
        mname = model_name or meta.get("model_name", DEFAULT_MODEL)
        doc_matrix = sp.load_npz(out_dir / "splade_matrix.npz")
        chunk_ids = np.load(out_dir / "splade_chunk_ids.npy", allow_pickle=True)
        return cls(doc_matrix, chunk_ids, mname, meta=meta)

    def encode_query(self, query: str) -> sp.csr_matrix:
        model, tokenizer = self._load_model()
        return encode_texts(model, tokenizer, [query])

    def search_single(self, query: str, top_k: int) -> List[ScoredChunk]:
        return self.search([query], top_k)[0]

    def search(self, queries: Sequence[str], top_k: int) -> List[List[ScoredChunk]]:
        model, tokenizer = self._load_model()
        qmat = encode_texts(model, tokenizer, list(queries))
        scores = (qmat @ self.doc_matrix.T).toarray()

        out: List[List[ScoredChunk]] = []
        for qi, row in enumerate(scores):
            k = min(top_k, self.n_docs)
            idxs = np.argpartition(row, -k)[-k:]
            idxs = idxs[np.argsort(-row[idxs])]
            hits = []
            for rank, didx in enumerate(idxs, start=1):
                hits.append(ScoredChunk(
                    chunk_id=str(self.chunk_ids[didx]),
                    score=float(row[didx]),
                    retrieval_method="splade",
                    rank=rank,
                ))
            out.append(hits)
        return out


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent.parent / ".cache" / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

    parser = argparse.ArgumentParser(description="Build SPLADE index from corpus (python -m src.retrieval.splade)")
    parser.add_argument("--out-dir", type=Path, default=Path("index/splade"), help="output dir")
    parser.add_argument("--max-docs", type=int, default=None, help="limit docs (for testing)")
    parser.add_argument("--batch-size", type=int, default=4, help="encode batch size")
    parser.add_argument("--model", default="NeuML/pubmedbert-base-splade", help="SPLADE model")
    args = parser.parse_args()

    from src.config import AppConfig
    from src.retrieval.corpus import CorpusIndex

    corpus = CorpusIndex(AppConfig().corpus_path)
    idx = SPLADEIndex.build_from_corpus(
        corpus, Path(args.out_dir), model_name=args.model, batch_size=args.batch_size,
        max_docs=args.max_docs,
    )
    print(f"Saved SPLADE index to {args.out_dir}: {idx.n_docs} docs, vocab {idx.vocab_size}, nnz {idx.doc_matrix.nnz}")
