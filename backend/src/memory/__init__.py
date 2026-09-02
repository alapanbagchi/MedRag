"""MedPat — Memory + Context layer.

Persistent, temporal, provenance-aware research state + context construction
for the evidence pipeline. The invariant this package enforces:

    Memory is not medical evidence.

L0-L2 (conversation / working research state / persistent research memory)
are advisory context; L3 (the PMC corpus + verified evidence pipeline) is the
only source of medical fact. Only ``EvidenceReferenceRecord`` crosses from
L3 into memory, and the provenance gate (``validation.evidence_gate``)
guarantees a claim cannot be persisted as evidence-backed without a verified
evidence link.

Public surface:
    MemoryAPI        — the facade the rest of MedPat uses
    MemoryConfig     — tunables (backend, embedder, budgets, thresholds)
    MemoryRunHooks   — AgenticV3Pipeline integration adapter
    MemoryStore      — store contract (InMemory / Postgres backends)
"""

from __future__ import annotations

from src.memory.api import MemoryAPI, RunPrep, RunRecordStats
from src.memory.config import MemoryConfig
from src.memory.hooks import MemoryRunHooks
from src.memory.store import (
    InMemoryMemoryStore,
    MemoryStore,
    PostgresMemoryStore,
    build_store,
)

__all__ = [
    "InMemoryMemoryStore",
    "MemoryAPI",
    "MemoryConfig",
    "MemoryRunHooks",
    "MemoryStore",
    "PostgresMemoryStore",
    "RunPrep",
    "RunRecordStats",
    "build_store",
]