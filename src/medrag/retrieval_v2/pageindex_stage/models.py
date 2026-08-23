"""Data models for the PageIndex stages (V2.4).

Navigation results preserve native PageIndex metadata untouched (raw) and add a
normalized view (node_id / title / path / reason) so the rest of the RAG can
consume the structural region without re-parsing the SDK output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PaperDocument:
    """Validation record for one Markdown paper (Stage 1)."""

    paper_id: str
    path: str = ""
    file_size: int = 0
    heading_count: int = 0
    heading_depth: int = 0
    table_count: int = 0
    caption_count: int = 0
    footnote_count: int = 0
    malformed_headings: int = 0
    hierarchy: List[List[str]] = field(default_factory=list)   # heading breadcrumbs
    suspicious: List[str] = field(default_factory=list)        # e.g. level jumps > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "path": self.path,
            "file_size": self.file_size,
            "heading_count": self.heading_count,
            "heading_depth": self.heading_depth,
            "table_count": self.table_count,
            "caption_count": self.caption_count,
            "footnote_count": self.footnote_count,
            "malformed_headings": self.malformed_headings,
            "suspicious": self.suspicious[:10],
            "hierarchy": self.hierarchy[:60],
        }


@dataclass
class SelectedNode:
    """One structural node selected by PageIndex navigation."""

    node_id: str
    title: str
    path: List[str] = field(default_factory=list)      # breadcrumb from the paper root
    level: int = 0
    reason: str = ""
    text_head: str = ""                                # first ~160 chars of node text
    raw: Dict[str, Any] = field(default_factory=dict)  # native PageIndex node metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "path": list(self.path),
            "level": self.level,
            "reason": self.reason,
            "text_head": self.text_head,
            "raw": self.raw,
        }


@dataclass
class NavigationStep:
    """One step of the navigation trace (Step 3D)."""

    index: int
    action: str            # e.g. "select", "expand", "tool-call", "final"
    node_title: str = ""
    node_path: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "node_title": self.node_title,
            "node_path": list(self.node_path),
            "detail": self.detail,
        }


@dataclass
class NavigationResult:
    """Normalized output of PageIndex query-time navigation (Step 3C)."""

    paper_id: str
    status: str                                    # success | failed | degraded
    objective: str = ""
    selected_nodes: List[SelectedNode] = field(default_factory=list)
    trace: List[NavigationStep] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)   # native PageIndex response
    latency_ms: float = 0.0
    error: Optional[Dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "status": self.status,
            "objective": self.objective,
            "selected_nodes": [n.to_dict() for n in self.selected_nodes],
            "trace": [s.to_dict() for s in self.trace],
            "raw": self.raw,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error or None,
        }


@dataclass
class NavigationError(Exception):
    """Structured error (task error-handling section): never fabricate results."""

    paper_id: str = ""
    error_type: str = ""   # markdown_missing | pageindex_unavailable | tree_build_failed | navigation_failed | llm_unavailable
    message: str = ""

    def __str__(self) -> str:
        return f"{self.error_type}: {self.message} ({self.paper_id})"

    def to_dict(self) -> Dict[str, str]:
        return {
            "status": "failed",
            "error_type": self.error_type,
            "message": self.message,
            "paper_id": self.paper_id,
        }
