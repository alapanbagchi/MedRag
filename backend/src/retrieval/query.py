"""Query encoding for the MedCPT retrieval architecture.

MedCPT is an *asymmetric* bi-encoder:

* documents were embedded with ``ncbi/MedCPT-Article-Encoder`` (max_length=512)
* queries must be embedded with ``ncbi/MedCPT-Query-Encoder`` (max_length=64)

Both encoders use the [CLS] last hidden state as the representation
(``last_hidden_state[:, 0, :]``), per the official model cards. The stored
corpus embeddings are *not* L2-normalized, so both document vectors (at
index build) and query vectors (here) are L2-normalized before the dot
product, which makes the FAISS inner-product score equal cosine similarity.

torch / transformers are imported lazily so the rest of the retrieval layer
works without them (e.g. for index builds and BM25-only evaluation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.lib._torch import resolve_device

DEFAULT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DEFAULT_MAX_LENGTH = 64


class QueryEncoder(ABC):
    @abstractmethod
    def encode(self, queries: List[str]) -> np.ndarray:
        """Return a float32 array of shape (len(queries), 768)."""


class MedCPTQueryEncoder(QueryEncoder):
    """CLS-pooled MedCPT query encoder (lazy torch/transformers load)."""

    def __init__(
        self,
        model_name: str = DEFAULT_QUERY_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self._model = None
        self._tokenizer = None
        self._resolved_device: Optional[str] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._resolved_device = resolve_device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._resolved_device)
        self._model.eval()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def encode(self, queries: List[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, 768), dtype=np.float32)
        self._load()

        import torch

        tokenizer = self._tokenizer
        model = self._model
        device = self._resolved_device

        all_embeds: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(queries), self.batch_size):
                batch = queries[i : i + self.batch_size]
                encoded = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                outputs = model(**encoded)
                # MedCPT representation: [CLS] last hidden state.
                embeds = outputs.last_hidden_state[:, 0, :]
                if self.normalize:
                    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
                all_embeds.append(embeds.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeds)

    def encode_single(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
