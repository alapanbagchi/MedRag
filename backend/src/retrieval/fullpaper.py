"""Structural-unit loader: given a retrieved chunk, return the ENTIRE
containing structural unit (whole paragraph / whole table / whole figure)
instead of a full paper or a bare chunk snippet.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("src.structural_unit")


class StructuralUnitIndex:
    """Map chunk_id -> the text of its containing structural unit.

    Rules:
      - chunk with table_id  -> ALL chunks of that table
        (rows + summary + footnotes sharing (document_id, table_id))
      - chunk with figure_id -> ALL chunks of that figure
        (sharing (document_id, figure_id))
      - otherwise (paragraph)  -> the entire paragraph chunk itself
    """

    def __init__(self, corpus: Any) -> None:
        self._df = getattr(corpus, "_df", None)
        if self._df is None:
            raise ValueError("corpus DataFrame not loaded (corpus.load() first)")
        # chunk_id -> dict(id, document_id, section, text, table_id, figure_id, position)
        self._units: Optional[Dict[str, str]] = None
        self._kinds: Dict[str, str] = {}
        # Building scans the whole corpus DataFrame; guard concurrent builds
        # (get() may be called from executor threads).
        self._build_lock = threading.Lock()

    def _build(self) -> None:
        if self._units is not None:
            return
        with self._build_lock:
            if self._units is not None:
                return
            self._build_units()

    def _build_units(self) -> None:
        df = self._df
        cols = ["id", "document_id", "text", "section", "table_id", "figure_id", "document_position"]
        cols = [c for c in cols if c in df.columns]
        sub = df[cols]

        # 1) table groups: (document_id, table_id) -> ordered chunk rows
        table_groups: Dict[tuple, List[Any]] = {}
        tbl = sub[sub["table_id"].notna()]
        for _, row in tbl.iterrows():
            key = (str(row["document_id"]), str(row["table_id"]))
            table_groups.setdefault(key, []).append(row)

        # 2) figure groups: (document_id, figure_id) -> ordered chunk rows
        fig_groups: Dict[tuple, List[Any]] = {}
        fig = sub[sub["figure_id"].notna()]
        for _, row in fig.iterrows():
            key = (str(row["document_id"]), str(row["figure_id"]))
            fig_groups.setdefault(key, []).append(row)

        def unit_text(rows: List[Any], label: str) -> str:
            rows_sorted = sorted(rows, key=lambda r: int(r.get("document_position") or 0))
            parts = [f"[{label}]"]
            for r in rows_sorted:
                t = str(r.get("text") or "").strip()
                if t:
                    parts.append(t)
            return "\n".join(parts)

        units: Dict[str, str] = {}
        kinds: Dict[str, str] = {}

        # a) tabular chunks -> whole table
        for key, rows in table_groups.items():
            _, table_id = key
            txt = unit_text(rows, f"Table {table_id}")
            for row in rows:
                units[str(row["id"])] = txt
                kinds[str(row["id"])] = "table"

        # b) figure chunks -> whole figure
        for key, rows in fig_groups.items():
            _, fig_id = key
            txt = unit_text(rows, f"Figure {fig_id}")
            for row in rows:
                units[str(row["id"])] = txt
                kinds[str(row["id"])] = "figure"

        # c) everything else (paragraphs etc.) -> the chunk itself
        other = sub[sub["table_id"].isna() & sub["figure_id"].isna()]
        for _, row in other.iterrows():
            t = str(row.get("text") or "").strip()
            if t:
                sec = str(row.get("section") or "")
                cid = str(row["id"])
                units[cid] = f"[{sec}]\n{t}" if sec else t
                kinds[cid] = "paragraph"

        self._units = units
        self._kinds = kinds
        logger.info("StructuralUnitIndex built: %d chunk->unit mappings", len(units))

    def get(self, chunk_id: str) -> str:
        """Return the ENTIRE containing unit text for a chunk id ("" if unknown)."""
        self._build()
        return self._units.get(str(chunk_id), "") if self._units else ""

    def unit_kind(self, chunk_id: str) -> str:
        """'table' | 'figure' | 'paragraph' for the containing unit."""
        self._build()
        return self._kinds.get(str(chunk_id), "paragraph")

    def unit_token_count(self, chunk_id: str) -> int:
        return len(self.get(chunk_id).split())


# ---------------------------------------------------------------------------
# Process-wide singleton: building the unit map scans the entire corpus
# DataFrame, so it must be built ONCE and reused across queries/rounds.
# ---------------------------------------------------------------------------

_UNIT_CACHE: Dict[str, Any] = {}


def get_unit_index(corpus: Any) -> "StructuralUnitIndex":
    """Return a cached StructuralUnitIndex bound to ``corpus``."""
    if _UNIT_CACHE.get("corpus") is corpus and isinstance(_UNIT_CACHE.get("index"), StructuralUnitIndex):
        return _UNIT_CACHE["index"]
    index = StructuralUnitIndex(corpus)
    _UNIT_CACHE["corpus"] = corpus
    _UNIT_CACHE["index"] = index
    return index


def reset_unit_index() -> None:
    """Drop the singleton (tests / corpus reload)."""
    _UNIT_CACHE.clear()


