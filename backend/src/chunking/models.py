"""Units artifact model (parent context for granular chunks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class UnitRecord:
    """One retrievable unit (section, table, or split paragraph) with its
    child chunk ids and its place in the unit tree."""
    unit_id: str
    document_id: str
    kind: str                      # "section" | "table" | "paragraph"
    title: str                     # unit title (heading or table label)
    breadcrumb: List[str]
    text: str                      # full unit text (parent context)
    chunk_ids: List[str]           # granular child chunks that live under it
    parent_unit_id: Optional[str] = None   # enclosing unit (section tree / table)

