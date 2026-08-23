"""Small shared helpers for torch/transformers loading.

Keeps device-resolution policy in one place so the embedding runner, the
query encoder and the reranker cannot drift apart. Nothing here imports
torch at module import time; the heavy libraries stay lazy-loaded.
"""

from __future__ import annotations

from typing import Optional


def resolve_device(device: Optional[str] = None) -> str:
    """Return the device to run inference on.

    ``None`` or ``"auto"`` means CUDA when available, otherwise CPU. Any
    other value is returned verbatim (explicit override). The check runs
    at call time so tests can monkeypatch ``torch.cuda.is_available``.
    """
    if device is not None and device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"