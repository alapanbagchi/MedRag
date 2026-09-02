"""Memory + Context layer — configuration.

Reads the same environment variables as ``src/config.py``'s memory block
(with identical defaults) so the layer can be constructed standalone or from
an ``AppConfig``. Keeping this separate avoids an import cycle and lets tests
construct a MemoryAPI without touching AppConfig.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class MemoryConfig:
    """Tunables of the memory + context layer.

    * backend         — auto | postgres | memory (store selection)
    * embedder        — hash | medcpt | null (recall signal, never truth)
    * embed_dim       — vector width; must match the DB schema at init
    * context_tokens  — total budget for the assembled memory/context region
    * claim_min_sim   — near-duplicate threshold for claim dedup/merge
    * retrieve_min_sim— semantic floor for vector recall
    * staleness_days  — data-driven revalidation horizon (0 = disabled;
                        staleness only from explicit mark_stale and
                        contradicting-evidence triggers)
    """
    backend: str = "auto"
    embedder: str = "hash"
    embed_dim: int = 256
    schema: str = "medrag_memory"
    context_tokens: int = 1800
    claim_min_sim: float = 0.86
    retrieve_min_sim: float = 0.15
    staleness_days: int = 0
    consolidation_batch: int = 200

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            backend=os.environ.get("MEMORY_BACKEND", "auto").strip().lower(),
            embedder=os.environ.get("MEMORY_EMBEDDER", "hash").strip().lower(),
            embed_dim=_int_env("MEMORY_EMBED_DIM", 256),
            schema=os.environ.get("MEMORY_SCHEMA", "medrag_memory").strip(),
            context_tokens=_int_env("MEMORY_CONTEXT_TOKENS", 1800),
            claim_min_sim=_float_env("MEMORY_CLAIM_MIN_SIM", 0.86),
            retrieve_min_sim=_float_env("MEMORY_RETRIEVE_MIN_SIM", 0.15),
            staleness_days=_int_env("MEMORY_STALENESS_DAYS", 0),
            consolidation_batch=_int_env("MEMORY_CONSOLIDATION_BATCH", 200),
        )

    @classmethod
    def from_appconfig(cls, cfg: Any) -> "MemoryConfig":
        def _g(name: str, default: Any) -> Any:
            return getattr(cfg, name, default)

        return cls(
            backend=_g("memory_backend", "auto"),
            embedder=_g("memory_embedder", "hash"),
            embed_dim=int(_g("memory_embed_dim", 256)),
            schema=_g("memory_schema", "medrag_memory"),
            context_tokens=int(_g("memory_context_tokens", 1800)),
            claim_min_sim=float(_g("memory_claim_min_sim", 0.86)),
            retrieve_min_sim=float(_g("memory_retrieve_min_sim", 0.15)),
            staleness_days=int(_g("memory_staleness_days", 0)),
            consolidation_batch=int(_g("memory_consolidation_batch", 200)),
        )

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


__all__ = ["MemoryConfig"]