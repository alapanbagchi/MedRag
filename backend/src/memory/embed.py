"""Memory + Context layer — embedders for recall (never for truth).

Embeddings in the memory layer are retrieval aids ONLY. The relational +
lexical + provenance fields are the truth; the vector is one recall signal
(design principle: memory is not "conversation -> embedding -> top-k").

Three embedder options:

* ``HashEmbedder``   — deterministic feature-hash bag-of-words, L2-
                      normalized, zero dependencies, works offline and in
                      tests. Default.
* ``MedCPTEmbedder`` — optional wrapper over the repo's existing
                      ``src.retrieval.query.MedCPTQueryEncoder``
                      (ncbi/MedCPT-Query-Encoder). Same model family as the
                      PMC corpus embeddings, so memory and evidence live in
                      one vector space. Requires torch/transformers and the
                      model weights; dim 768.
* ``NullEmbedder``   — dim 0; used when embeddings are disabled entirely
                      (pure lexical/relational recall).
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from typing import Protocol

logger = logging.getLogger("src.memory.embed")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on",
    "with", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "by", "at", "from", "do", "does",
    "did", "not", "no", "yes", "than", "then", "there", "their", "they",
    "you", "your", "we", "our", "i", "me", "my", "can", "could", "will",
    "would", "should", "may", "might", "what", "which", "who", "when",
    "where", "how", "why", "if", "about", "between", "has", "have", "had",
})


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts -> list of L2-normalized float vectors."""


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


class HashEmbedder:
    """Deterministic, dependency-free feature-hash bag-of-words embedder.

    Token features are hashed with blake2b (stable across processes and
    Python versions) into a fixed-dim signed bag; L2-normalized so cosine
    similarity is meaningful. Good enough for near-duplicate claim recall;
    NOT a semantic embedder — use MedCPTEmbedder when real semantics are
    required and the model is available.
    """

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def _features(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for tok in _TOKEN_RE.findall((text or "").casefold()):
            if tok in _STOP or len(tok) < 2:
                continue
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dim
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        # dampen heavy tokens (log scaling keeps frequent terms from
        # dominating the bag)
        return {k: 1.0 + (v - 1.0) * 0.5 for k, v in counts.items()}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for bucket, weight in self._features(text).items():
                vec[bucket] = weight
            out.append(_l2_normalize(vec))
        return out


class MedCPTEmbedder:
    """Optional semantic embedder over the repo's MedCPT query encoder.

    Lazy-loads torch/transformers on first use; dim 768 (must match the
    memory schema dimension configured at DB init time).
    """

    def __init__(self):
        self.dim = 768
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from src.retrieval.query import MedCPTQueryEncoder  # type: ignore
                self._encoder = MedCPTQueryEncoder("ncbi/MedCPT-Query-Encoder")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "MedCPTEmbedder unavailable (torch/transformers/model): "
                    f"{exc} — set MEMORY_EMBEDDER=hash") from exc
        return self._encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        enc = self._get_encoder()
        mat = enc.encode(list(texts))           # np.ndarray (n, 768)
        return np.asarray(mat, dtype=np.float32).tolist()


class NullEmbedder:
    """Embeddings disabled; recall is lexical/relational only."""

    dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


def embedder_from_config(embedder: str = "hash", dim: int = 256) -> Embedder:
    """Build the embedder named by config (hash | medcpt | null)."""
    name = (embedder or "hash").strip().lower()
    if name in ("", "hash"):
        return HashEmbedder(dim=dim)
    if name == "medcpt":
        warnings.warn(
            "MEMORY_EMBEDDER=medcpt loads torch/transformers + MedCPT weights "
            "on first embed; offline tests should use 'hash'.",
            stacklevel=2,
        )
        return MedCPTEmbedder()
    if name == "null":
        return NullEmbedder()
    raise ValueError(f"unknown memory embedder: {embedder!r} "
                     f"(use hash | medcpt | null)")


__all__ = ["Embedder", "HashEmbedder", "MedCPTEmbedder", "NullEmbedder",
           "embedder_from_config"]