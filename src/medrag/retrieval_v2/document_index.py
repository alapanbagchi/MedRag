"""Logical Document Structure Index (spec sections 4-5, 19, 37).

Do NOT interpret this as a PDF page layout index. It is a LOGICAL index of
the XML-derived structure: paper -> section -> subsection -> evidence node,
with tables (summary/rows/footnotes) and figures as first-class nodes.

The index is memory-efficient: it holds structural metadata for every chunk
and fetches text on demand (from pgvector when available). It supports rapid
in-paper navigation: children, parent, siblings, table assembly, section
lookup and figure/caption lookup.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from medrag.retrieval.corpus import CORPUS_FILENAME

STRUCTURE_COLUMNS = [
    "id", "document_id", "chunk_type", "section", "subsection", "breadcrumb",
    "parent_id", "table_id", "figure_id", "document_position",
]

TEXT_COLUMN = "text"

# Node types that are evidence-bearing.
TABLE_TYPES = ("table_summary", "table_row", "table_footnotes")


def _norm_section(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 chars), matching corpus conventions."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class LogicalDocumentIndex:
    """Logical document structure index over the consolidated corpus.parquet.

    The corpus parquet rows ARE the leaves of the logical document tree; this
    class adds the navigation, hierarchy and text-resolution layer around it.
    """

    def __init__(
        self,
        corpus_path: Path,
        store: Any = None,
        max_paper_cache: int = 64,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.store = store          # optional PgVectorStore for text
        self._df = None
        self._id_to_paper: Dict[str, str] = {}
        self._paper_node_cache: Dict[str, List[str]] = {}
        self._paper_sizes: Dict[str, int] = {}
        self._text_parquet_loaded = False
        self._text_parquet: Dict[str, str] = {}
        self._max_paper_cache = max_paper_cache
        self.load()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def load(self) -> "LogicalDocumentIndex":
        import pandas as pd

        # numeric section names need preserving; project only needed columns
        cols_to_read = [c for c in STRUCTURE_COLUMNS]
        try:
            df = pd.read_parquet(str(self.corpus_path), columns=cols_to_read)
        except Exception:
            df = pd.read_parquet(str(self.corpus_path))

        # pandas 3 loads parquet with pyarrow/extension-backed dtypes by default,
        # which makes every .iloc/.isin scan go through slow pyarrow.take. Convert
        # every string column to the plain numpy object dtype once so all lookups
        # are fast (spec section 37). List columns become plain python lists.
        for col in df.columns:
            if df[col].dtype == "string" or df[col].dtype.name in ("string", "string[pyarrow]", "large_string"):
                df[col] = df[col].astype(object)
        if "breadcrumb" in df.columns:
            df["breadcrumb"] = df["breadcrumb"].map(
                lambda b: list(b) if b is not None else []
            )

        for col in ("section", "subsection", "parent_id", "table_id", "figure_id"):
            if col in df.columns:
                df[col] = df[col].fillna("")
            else:
                df[col] = ""
        if "breadcrumb" not in df.columns:
            df["breadcrumb"] = [[] for _ in range(len(df))]
        if "document_position" not in df.columns:
            df["document_position"] = range(len(df))
        if "document_id" not in df.columns:
            df["document_id"] = ""
        if "chunk_type" not in df.columns:
            df["chunk_type"] = "paragraph"

        df["section"] = df["section"].map(_norm_section)
        df["subsection"] = df["subsection"].map(_norm_section)
        df["document_position"] = df["document_position"].astype(np.int64)
        self._df = df
        self._id_to_paper = dict(zip(df["id"], df["document_id"]))
        sizes = df["document_id"].value_counts()
        self._paper_sizes = {k: int(v) for k, v in sizes.items()}
        self._id_to_pos: Optional[Dict[str, int]] = None
        self._paper_nodes_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._paper_nodes_by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
        return self

    def _row_positions(self, chunk_ids: Sequence[str]) -> List[int]:
        """Lazily build chunk_id -> DataFrame row index for fast vectorized lookups."""
        if self._id_to_pos is None:
            self._id_to_pos = {cid: i for i, cid in enumerate(self._df["id"])}
        out = []
        for cid in chunk_ids:
            pos = self._id_to_pos.get(cid)
            if pos is not None:
                out.append(pos)
        return out

    # ------------------------------------------------------------------
    # Basic stats
    # ------------------------------------------------------------------
    @property
    def n_chunks(self) -> int:
        return 0 if self._df is None else len(self._df)

    @property
    def n_papers(self) -> int:
        return len(self._paper_sizes)

    def paper_ids(self) -> List[str]:
        return list(self._paper_sizes.keys())

    def paper_size(self, paper_id: str) -> int:
        return self._paper_sizes.get(paper_id, 0)

    def paper_of(self, chunk_id: str) -> str:
        return self._id_to_paper.get(chunk_id, "")

    def get_node(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Return the logical node record for a chunk id (no text)."""
        if self._df is None:
            return None
        pos = self._row_positions([chunk_id])
        if not pos:
            return None
        r = self._df.iloc[pos[0]]
        return self._record_from_row(chunk_id, r)

    def _record_from_row(self, chunk_id: str, r: Any) -> Dict[str, Any]:
        breadcrumb = r["breadcrumb"]
        if breadcrumb is None or (isinstance(breadcrumb, float) and np.isnan(breadcrumb)):
            breadcrumb_t: List[str] = []
        elif isinstance(breadcrumb, str):
            breadcrumb_t = [x.strip() for x in breadcrumb.split(">")] if breadcrumb else []
        else:
            breadcrumb_t = [str(x) for x in breadcrumb]
        return {
            "chunk_id": chunk_id,
            "node_id": chunk_id,                        # spec section 4 alias
            "paper_id": str(r["document_id"]),
            "node_type": str(r["chunk_type"] or "paragraph"),
            "section": _norm_section(r["section"]),
            "subsection": _norm_section(r["subsection"]),
            "section_path": list(breadcrumb_t),         # spec section 4 name
            "breadcrumb": list(breadcrumb_t),
            "position": int(r["document_position"]),
            "parent_id": str(r["parent_id"]) if r["parent_id"] else None,
            "table_id": str(r["table_id"]) if r["table_id"] else None,
            "figure_id": str(r["figure_id"]) if r["figure_id"] else None,
            "token_count": 0,  # computed from text at expansion time (spec section 37)
        }

    # ------------------------------------------------------------------
    # Paper-level views (cached)
    # ------------------------------------------------------------------
    def get_paper_chunk_ids(self, paper_id: str) -> List[str]:
        """All node ids of a paper, in document order (spec: get all nodes)."""
        cached = self._paper_node_cache.get(paper_id)
        if cached is not None:
            return cached
        sub = self._df[self._df["document_id"] == paper_id]
        ids = sub.sort_values("document_position")["id"].tolist()
        if len(self._paper_node_cache) >= self._max_paper_cache:
            self._paper_node_cache.clear()
        self._paper_node_cache[paper_id] = ids
        return ids

    def get_paper_nodes(self, paper_id: str) -> List[Dict[str, Any]]:
        """All logical nodes of a paper in document order (cached per paper)."""
        cached_records = self._paper_nodes_cache.get(paper_id)
        if cached_records is not None:
            return cached_records
        ids = self.get_paper_chunk_ids(paper_id)
        if not ids:
            return []
        pos = self._row_positions(ids)
        if not pos:
            return []
        sub = self._df.iloc[pos]
        by_id: Dict[str, Dict[str, Any]] = {}
        records: List[Dict[str, Any]] = []
        for rec in sub.to_dict("records"):
            cid = rec["id"]
            node = self._record_from_row(cid, rec)
            by_id[cid] = node
            records.append(node)
        if len(self._paper_nodes_cache) >= self._max_paper_cache:
            self._paper_nodes_cache.clear()
            self._paper_nodes_by_id.clear()
        self._paper_nodes_cache[paper_id] = records
        self._paper_nodes_by_id[paper_id] = by_id
        return records

    def _paper_by_id(self, paper_id: str) -> Dict[str, Dict[str, Any]]:
        cached = self._paper_nodes_by_id.get(paper_id)
        if cached is None:
            self.get_paper_nodes(paper_id)
        return self._paper_nodes_by_id.get(paper_id, {})

    # ------------------------------------------------------------------
    # Navigation (section 5)
    # ------------------------------------------------------------------
    def get_children(self, paper_id: str, node_id: str) -> List[Dict[str, Any]]:
        """Direct children of a node.

        Table rows/footnotes are children of their table summary (via
        parent_id or chunk-id prefix); paragraphs are governed by the
        section/subsection hierarchy.
        """
        parent = self.get_node(node_id)
        if parent is None:
            return []
        if parent["node_type"] == "table_summary":
            prefix = self._table_prefix_for(paper_id, parent.get("table_id") or node_id)
            out = []
            for n in self.get_paper_nodes(paper_id):
                if n.get("parent_id") == node_id:
                    out.append(n)
                elif n["node_type"] in ("table_row", "table_footnotes") and (
                    self._table_prefix_for(paper_id, n.get("table_id") or n["chunk_id"]) == prefix
                ):
                    out.append(n)
            return out
        return []

    def get_parent(self, paper_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        node = self.get_node(node_id)
        if node is None:
            return None
        if node["node_type"] in ("table_row", "table_footnotes") and node.get("table_id"):
            pid = node.get("parent_id")
            if pid and self.get_node(pid) is not None:
                return self.get_node(pid)
            summary_id = f"{paper_id}_{node['table_id']}_summary"
            found = self.get_node(summary_id)
            if found is not None:
                return found
        if node.get("parent_id") and self.get_node(node["parent_id"]) is not None:
            return self.get_node(node["parent_id"])
        return None

    def get_siblings(self, paper_id: str, node_id: str) -> List[Dict[str, Any]]:
        """Sibling nodes: same section/subsection (paragraphs) or same table."""
        node = self.get_node(node_id)
        if node is None:
            return []
        if node["node_type"] in TABLE_TYPES and node.get("table_id"):
            table = self.get_table(paper_id, node["table_id"])
            sibs = list(table.get("rows", [])) + list(table.get("footnotes", []))
            return [n for n in sibs if n["chunk_id"] != node_id]
        return [
            n for n in self.get_paper_nodes(paper_id)
            if n["chunk_id"] != node_id
            and n["node_type"] == node["node_type"]
            and n["section"] == node["section"]
            and n["subsection"] == node["subsection"]
        ]

    def _table_prefix_for(self, paper_id: str, table_id_or_chunk: str) -> str:
        if table_id_or_chunk.endswith(("_summary", "_footnotes")) or "_row_" in table_id_or_chunk:
            m = re.match(r"^(.+)_(?:summary|row_d+|footnotes)$", table_id_or_chunk)
            if m:
                return m.group(1)
        return f"{paper_id}_{table_id_or_chunk}" if table_id_or_chunk else ""

    def get_table(self, paper_id: str, table_id: str) -> Dict[str, Any]:
        """Assemble a table: summary + rows + footnotes (sections 21-23).

        Returns {table_id, summary, rows, footnotes, columns_hint} where each
        entry is a node record; text is resolved lazily via get_text.
        """
        nodes = self.get_paper_nodes(paper_id)
        summary: Optional[Dict[str, Any]] = None
        rows: List[Dict[str, Any]] = []
        footnotes: List[Dict[str, Any]] = []

        for n in nodes:
            if n.get("table_id") != table_id:
                continue
            if n["node_type"] == "table_summary":
                summary = n
            elif n["node_type"] == "table_row":
                rows.append(n)
            elif n["node_type"] == "table_footnotes":
                footnotes.append(n)

        if summary is None or not rows:
            prefix = self._table_prefix_for(paper_id, table_id)
            if prefix:
                for n in nodes:
                    cid = n["chunk_id"]
                    if not cid.startswith(prefix + "_"):
                        continue
                    rest = cid[len(prefix) + 1:]
                    if rest == "summary" and summary is None:
                        summary = n
                    elif rest == "footnotes":
                        footnotes.append(n)
                    elif rest.startswith("row_"):
                        if n["chunk_id"] not in [r["chunk_id"] for r in rows]:
                            rows.append(n)

        rows.sort(key=lambda n: n["position"])
        all_ids: List[str] = []
        if summary:
            all_ids.append(summary["chunk_id"])
        all_ids += [r["chunk_id"] for r in rows]
        all_ids += [f_["chunk_id"] for f_ in footnotes]
        return {
            "table_id": table_id,
            "summary": summary,
            "rows": rows,
            "footnotes": footnotes,
            "all_node_ids": all_ids,
        }

    def get_figure(self, paper_id: str, figure_id: str) -> Optional[Dict[str, Any]]:
        """Figure node (caption + text live in the figure chunk; section 22)."""
        nodes = self.get_paper_nodes(paper_id)
        for n in nodes:
            if n.get("figure_id") == figure_id and n["node_type"] == "figure":
                return n
        cid = f"{paper_id}_{figure_id}"
        for n in nodes:
            if n["chunk_id"] == cid:
                return n
        return None

    def get_section(self, paper_id: str, section: str, subsection: Optional[str] = None) -> List[Dict[str, Any]]:
        """All nodes in a section (optionally a specific subsection)."""
        out = []
        for n in self.get_paper_nodes(paper_id):
            if n["section"] != section:
                continue
            if subsection is not None and n["subsection"] != subsection:
                continue
            out.append(n)
        return out

    def get_neighbors(self, paper_id: str, node_id: str, before: int = 1, after: int = 1) -> List[Dict[str, Any]]:
        """Adjacent nodes within the same section/subsection (paragraph window)."""
        node = self.get_node(node_id)
        if node is None:
            return []
        same = [
            n for n in self.get_paper_nodes(paper_id)
            if n["section"] == node["section"] and n["subsection"] == node["subsection"]
        ]
        same.sort(key=lambda n: n["position"])
        idx = next((i for i, n in enumerate(same) if n["chunk_id"] == node_id), -1)
        if idx < 0:
            return []
        lo = max(0, idx - before)
        hi = min(len(same), idx + after + 1)
        return same[lo:hi]

    # ------------------------------------------------------------------
    # Text resolution (memory-efficient: on demand)
    # ------------------------------------------------------------------
    def get_text(self, chunk_id: str) -> str:
        texts = self.get_texts([chunk_id])
        return texts.get(chunk_id, "")

    def get_texts(self, chunk_ids: Sequence[str]) -> Dict[str, str]:
        """Resolve text for chunk ids from pgvector (preferred) or parquet."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        out: Dict[str, str] = {}
        if self.store is not None:
            try:
                meta = self.store.get_chunks(ids)
                for cid in ids:
                    rec = meta.get(cid)
                    if rec and rec.get("text"):
                        out[cid] = rec["text"]
            except Exception:
                out = {}
        missing = [c for c in ids if c not in out]
        if missing:
            self._load_text_parquet()
            for cid in missing:
                t = self._text_parquet.get(cid)
                if t:
                    out[cid] = t
        return out

    def _load_text_parquet(self) -> None:
        if self._text_parquet_loaded:
            return
        try:
            import pandas as pd
            df = pd.read_parquet(str(self.corpus_path), columns=["id", TEXT_COLUMN])
            self._text_parquet = dict(zip(df["id"], df[TEXT_COLUMN].fillna("")))
        except Exception:
            self._text_parquet = {}
        self._text_parquet_loaded = True

    # ------------------------------------------------------------------
    # Statistics / diagnostics
    # ------------------------------------------------------------------
    def paper_stats(self, paper_id: str) -> Dict[str, Any]:
        nodes = self.get_paper_nodes(paper_id)
        types: Dict[str, int] = defaultdict(int)
        sections: Dict[str, int] = defaultdict(int)
        tables: set = set()
        figures: set = set()
        for n in nodes:
            types[n["node_type"]] += 1
            sections[n["section"]] += 1
            if n.get("table_id"):
                tables.add(n["table_id"])
            if n.get("figure_id"):
                figures.add(n["figure_id"])
        return {
            "paper_id": paper_id,
            "n_nodes": len(nodes),
            "node_types": dict(types),
            "sections": dict(sections),
            "n_tables": len(tables),
            "n_figures": len(figures),
        }

    def chunk_metadata_for(self, chunk_ids: Sequence[str], include_text: bool = False) -> Dict[str, Dict[str, Any]]:
        """Metadata dicts for arbitrary chunk ids (used by global retrieval)."""
        if self._df is None or not chunk_ids:
            return {}
        pos = self._row_positions(list(chunk_ids))
        if not pos:
            return {}
        id_at = self._df["id"].iloc[pos]
        id_map = {i: cid for i, cid in enumerate(id_at)}
        out: Dict[str, Dict[str, Any]] = {}
        for i, rec in enumerate(self._df.iloc[pos].to_dict("records")):
            out[id_map[i]] = self._record_from_row(id_map[i], rec)
        if include_text:
            texts = self.get_texts(list(chunk_ids))
            for cid, t in texts.items():
                if cid in out:
                    out[cid]["text"] = t
        return out
