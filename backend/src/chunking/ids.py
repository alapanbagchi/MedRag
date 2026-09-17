"""Per-document identity and position state: unique ids + counters.

Chunking is a set of pure functions over a DocumentState plus a
chunks.ChunkFactory; this module owns every mutable per-document value (doc
id, metadata, id-uniqueness, chunk/unit counters) so the concern modules
stay stateless.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict


def digest_id(text: str) -> str:
    """Deterministic document id from content (MD + sha1 prefix)."""
    return "MD" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


class DocumentState:
    """Mutable per-document chunking state (one instance per document)."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.meta: Dict[str, Any] = {}
        self.used_ids: set = set()
        self.prose_counter = 0
        self.table_counter = 0
        self.equation_counter = 0
        self.reference_counter = 0
        self.position_counter = 0
        self.unit_counter = 0
        self.split_paragraphs = 0   # source paragraphs that had to be split
        self.split_pieces = 0       # pieces emitted by those splits

    def unique(self, candidate: str) -> str:
        """Return candidate, or candidate_N on collision."""
        if candidate not in self.used_ids:
            self.used_ids.add(candidate)
            return candidate
        suffix = 1
        while f"{candidate}_{suffix}" in self.used_ids:
            suffix += 1
        out = f"{candidate}_{suffix}"
        self.used_ids.add(out)
        return out

    def next_pos(self) -> int:
        """Monotonic document position for one chunk."""
        pos = self.position_counter
        self.position_counter += 1
        return pos

    @staticmethod
    def slug(text: str) -> str:
        """URL-ish local slug for object labels ("Table 3" -> "table_3")."""
        s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
        return s or "x"
