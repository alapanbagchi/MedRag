"""Query encoding for the MedCPT retrieval architecture.

MedCPT is an *asymmetric* bi-encoder: documents embed with Article-Encoder,
queries with Query-Encoder (max_length=64). Both use the [CLS] last hidden
state, and both sides are L2-normalized so the FAISS inner product equals
cosine similarity. torch/transformers load lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


def resolve_device(device: Optional[str] = None) -> str:
    """Inference device: explicit value, else CUDA-if-available or CPU."""
    if device is not None and device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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

        from src.lib._torch import resolve_device

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
