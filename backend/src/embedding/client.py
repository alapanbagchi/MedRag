"""OpenAI-compatible /embeddings client for the MedCPT server.

One shared httpx.Client (thread-safe) so a worker pool reuses connections.
EmbedClient.embed() returns a float32 (n, dim) matrix in input order; vectors
are L2-normalized unless normalize=False. The server returns raw FP32 [CLS]
vectors, and medpat.chunk_embeddings is indexed with vector_ip_ops, so stored
vectors must be unit length for inner product to equal cosine similarity.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import httpx
import numpy as np

DEFAULT_DIM = 768   # medpat.chunk_embeddings.embedding is vector(768)


class _Retryable(Exception):
    """Transient HTTP status (429 / 5xx) worth retrying."""


class EmbedClient:
    """Thin OpenAI-compatible /embeddings client for the MedCPT server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 180.0,
        retries: int = 5,
        normalize: bool = True,
        dim: int = DEFAULT_DIM,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.model = model
        self.normalize = normalize
        self.dim = dim
        self.retries = max(0, retries)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=timeout),
            transport=transport,
        )

    # -- endpoint ----------------------------------------------------------

    @property
    def url(self) -> str:
        return str(self._client.base_url).rstrip("/") + "/embeddings"

    def close(self) -> None:
        self._client.close()

    def check_health(self) -> Optional[Dict[str, Any]]:
        """GET {root}/health (root = base_url without a trailing /v1)."""
        root = str(self._client.base_url).rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        try:
            resp = self._client.get(f"{root}/health", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - health is advisory
            print(f"  [warn] health probe failed ({exc}); continuing")
            return None

    # -- encoding ----------------------------------------------------------

    def embed(self, texts: List[str],
              on_stage: Optional[Callable[[str, int], None]] = None) -> np.ndarray:
        """POST one batch; returns (len(texts), dim) float32, input order.

        on_stage(event, n) reports pipeline stages for live progress:
        "sent" once before the request, "recv" once the response body is
        parsed, "norm" once L2 normalization is done.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        payload = {"model": self.model, "input": [t or " " for t in texts]}
        url = self.url  # absolute; relative paths would resolve against /v1
        if on_stage:
            on_stage("sent", len(texts))
        for attempt in range(self.retries + 1):
            try:
                resp = self._client.post(url, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise _Retryable(resp.status_code, resp.text[:200])
                resp.raise_for_status()
                body = resp.json()
                if on_stage:
                    on_stage("recv", len(texts))
                vecs = self._parse(body, len(texts))
                if on_stage:
                    on_stage("norm", len(texts))
                return vecs
            except (_Retryable, httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"embed request failed after {self.retries + 1} attempts "
                        f"({exc}); url={self.url}"
                    ) from exc
                sleep = min(2 ** attempt, 30.0)   # 1, 2, 4, ... capped at 30s
                print(f"  [retry {attempt + 1}/{self.retries}] {exc} - "
                      f"sleeping {sleep:.0f}s", flush=True)
                time.sleep(sleep)
        raise AssertionError("unreachable")

    def _parse(self, body: Dict[str, Any], expected: int) -> np.ndarray:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            raise RuntimeError(
                f"bad embeddings response: expected {expected} items, got "
                f"{len(data) if isinstance(data, list) else type(data).__name__}"
            )
        rows = sorted(data, key=lambda d: int(d.get("index", 0)))
        vecs = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape != (expected, self.dim):
            got = vecs.shape if vecs.ndim == 2 else ("!",)
            raise RuntimeError(
                f"embedding dimension mismatch: server returned {got}, "
                f"need ({expected}, {self.dim}) to match vector({self.dim})"
            )
        if self.normalize:
            norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        if not np.isfinite(vecs).all():
            raise RuntimeError("embedding contains NaN/Inf after normalization")
        return vecs
