"""Second-stage reranking with a cross-encoder.

Selected model: ``ncbi/MedCPT-Cross-Encoder``.

* Architecture: BERT-base cross-encoder (``BertForSequenceClassification``,
  110M params), PubMedBERT-initialised and trained on 18M semantic
  query-article pairs plus localized negatives from the MedCPT retriever.
* Input: ``[CLS] query [SEP] passage [SEP]`` (joint encoding, max_length 512).
* Output: a single relevance logit; higher = more relevant (only relative
  order matters for ranking). Optional sigmoid for a 0-1 score.
* License: public domain (Hugging Face ``license: other``).
* Size: ~440 MB fp32; ~450 MB GPU at inference for small batches.

The reranker is a second-stage only: it receives a small candidate pool
(typically 50-100 chunks) produced by dense/BM25/RRF and reorders it. It is
never run over the whole corpus.

Inference is batched, uses ``torch.inference_mode()``, keeps the model in
``eval`` mode, moves scores off the GPU each batch, and the model is loaded
once and reused for the lifetime of the instance.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np

from medrag.models import Candidate, RerankedCandidate
from medrag._torch import resolve_device

DEFAULT_RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


def build_passage(candidate: Candidate, include_breadcrumb: bool = True) -> str:
    """Build the passage representation scored by the reranker.

    Kept deliberately light: breadcrumb (section path) is prepended when
    requested and available, nothing else. The chunk text remains the primary
    signal.
    """
    text = (candidate.text or "").strip()
    if include_breadcrumb and candidate.breadcrumb:
        return f"Section: {candidate.breadcrumb}\nPassage: {text}"
    return text


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int = 10,
    ) -> List[RerankedCandidate]:
        """Score candidates jointly with the query and return the reordered top-k."""


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker backed by a ``AutoModelForSequenceClassification``."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        include_breadcrumb: bool = True,
        apply_sigmoid: bool = False,
        quantize: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device  # None/'auto' -> cuda if available else cpu
        self.batch_size = batch_size
        self.max_length = max_length
        self.include_breadcrumb = include_breadcrumb
        self.apply_sigmoid = apply_sigmoid
        self.quantize = quantize
        self._model = None
        self._tokenizer = None
        self._resolved_device: Optional[str] = None

    # -- loading -------------------------------------------------------

    def _resolve_device(self) -> str:
        return resolve_device(self.device)

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._resolved_device = self._resolve_device()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.quantize:
            # Dynamic int8 quantization keeps the model on CPU (CUDA does not
            # support quantize_dynamic for Linear). Force the device so inputs
            # are not moved to a GPU the model cannot follow.
            import torch

            self._resolved_device = "cpu"
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            self._model.to(self._resolved_device)
        self._model.eval()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def resolved_device(self) -> Optional[str]:
        return self._resolved_device

    # -- scoring -------------------------------------------------------

    def _score_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: Optional[int] = None,
        progress: bool = False,
    ) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=np.float32)

        self._load()
        import torch

        from tqdm import tqdm

        tokenizer = self._tokenizer
        model = self._model
        device = self._resolved_device

        queries = [q for q, _ in pairs]
        passages = [p for _, p in pairs]
        bs = self.batch_size if batch_size is None else batch_size

        n_batches = (len(pairs) + bs - 1) // bs
        bar = (
            tqdm(total=n_batches, desc="Reranking", unit="batch", dynamic_ncols=True)
            if progress
            else None
        )

        all_scores: List[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(pairs), bs):
                q_batch = queries[i : i + bs]
                p_batch = passages[i : i + bs]
                encoded = tokenizer(
                    q_batch,
                    p_batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits.squeeze(-1)
                if self.apply_sigmoid:
                    logits = torch.sigmoid(logits)
                # Move off the device immediately; do not accumulate GPU tensors.
                all_scores.append(logits.cpu().numpy().astype(np.float32))
                del encoded, logits
                if bar is not None:
                    bar.update(1)

        if bar is not None:
            bar.close()

        return np.concatenate(all_scores) if all_scores else np.empty(0, dtype=np.float32)

    # -- rerank --------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int = 10,
    ) -> List[RerankedCandidate]:
        from medrag.trace import get_trace
        trace = get_trace()

        if not candidates:
            return []

        candidates = list(candidates)
        self._load()

        passages = [build_passage(c, include_breadcrumb=self.include_breadcrumb) for c in candidates]

        trace.log(
            "rerank_input",
            params={
                "query": query,
                "n_candidates": len(candidates),
                "top_k": top_k,
                "passages_preview": [{"chunk_id": c.chunk_id, "text_preview": (c.text or "")[:150]} for c in candidates[:10]],
            },
        )

        start = time.perf_counter()
        scores = self._score_pairs([(query, p) for p in passages])
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0

        top_k = min(top_k, len(candidates))
        order = np.argsort(-scores, kind="stable")

        out: List[RerankedCandidate] = []
        for new_rank, idx in enumerate(order[:top_k], start=1):
            c = candidates[idx]
            out.append(
                RerankedCandidate(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    original_score=c.original_score,
                    reranker_score=float(scores[idx]),
                    original_rank=c.original_rank,
                    reranked_rank=new_rank,
                    chunk_type=c.chunk_type,
                    breadcrumb=c.breadcrumb,
                    text=c.text,
                    retrieval_method=c.retrieval_method,
                )
            )

        trace.log(
            "rerank_output",
            params={"n_input": len(candidates), "top_k": top_k},
            result={
                "n_output": len(out),
                "score_range": [float(scores.min()), float(scores.max())] if len(scores) > 0 else [],
                "mean_score": float(scores.mean()) if len(scores) > 0 else 0,
                "reranked_top": [{"chunk_id": r.chunk_id, "reranker_score": round(r.reranker_score, 6), "original_rank": r.original_rank} for r in out[:10]],
            },
            duration_ms=self.last_inference_ms,
        )
        return out

    def rerank_many(
        self,
        tasks: Sequence[Tuple[str, Sequence[Candidate]]],
        top_k: int = 10,
        batch_size: Optional[int] = None,
        progress: bool = False,
    ) -> List[List[RerankedCandidate]]:
        """Rerank several queries' candidate pools in one macro-batched pass.

        All (query, passage) pairs across the queries are flattened and
        scored in a single batched loop. This keeps the CPU busy with larger
        matrices than a per-query loop would and amortizes the Python /
        tokenizer overhead per query. Results are split back per query.
        """
        if not tasks:
            return []
        self._load()

        flat_pairs: List[Tuple[str, str]] = []
        offsets: List[Tuple[int, int]] = []
        per_query_candidates: List[List[Candidate]] = []
        for query, cands in tasks:
            cands = list(cands)
            per_query_candidates.append(cands)
            start = len(flat_pairs)
            flat_pairs.extend(
                (query, build_passage(c, include_breadcrumb=self.include_breadcrumb))
                for c in cands
            )
            offsets.append((start, len(flat_pairs)))

        start = time.perf_counter()
        scores = self._score_pairs(flat_pairs, batch_size=batch_size, progress=progress)
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0

        out: List[List[RerankedCandidate]] = []
        for cands, (start_i, end_i) in zip(per_query_candidates, offsets):
            qscores = scores[start_i:end_i]
            k = min(top_k, len(cands))
            order = np.argsort(-qscores, kind="stable")[:k]
            ranked: List[RerankedCandidate] = []
            for new_rank, idx in enumerate(order, start=1):
                c = cands[idx]
                ranked.append(
                    RerankedCandidate(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        original_score=c.original_score,
                        reranker_score=float(qscores[idx]),
                        original_rank=c.original_rank,
                        reranked_rank=new_rank,
                        chunk_type=c.chunk_type,
                        breadcrumb=c.breadcrumb,
                        text=c.text,
                        retrieval_method=c.retrieval_method,
                    )
                )
            out.append(ranked)
        return out
