"""PageIndex paper-local navigation adapter (V2.1 parts 9-14, 24, 28, 36-37).

Role boundary (the architecture repair):

    GLOBAL retrieval  -> discovers PAPERS        (BM25 + pgvector, unchanged)
    PageIndex         -> navigates WITHIN papers (hierarchy + reasoning)

PageIndex is a paper-local reasoning/navigation layer over PRECOMPUTED tree
artifacts (index/pageindex/{paper_id}.json, built offline by
pageindex_build from the same XML-derived structure the LogicalDocumentIndex
uses). The adapter maps PageIndex node ids onto EXISTING corpus chunk ids so
there is exactly one evidence identity per chunk (part 24):

    PageIndex node -> existing chunk_id -> LogicalDocumentIndex -> XML evidence

The adapter is REPLACEABLE and OPTIONAL (part 37): if PageIndex is
unavailable or an artifact is missing, retrieval falls back to
BM25 + pgvector + LogicalDocumentIndex without failing the request, and the
trace records pageindex_status = success | unavailable | failed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from src.retrieval_v2.models import PageIndexHit
from src.retrieval_v2.retriever import NullTrace

_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "with", "without", "to",
    "for", "was", "were", "is", "are", "be", "been", "as", "by", "at", "from",
    "that", "this", "these", "those", "that", "between", "among", "across",
    "which", "what", "who", "how", "do", "does", "did", "we", "our", "it",
    "its", "not", "no", "find", "finding", "report", "reports", "reported",
    "including", "measure", "measures", "measured", "associations",
}

_SECTION_PRIOR = {
    "abstract": 0.15,
    "introduction": 0.05,
    "methods": 0.08,
    "results": 0.20,
    "discussion": 0.12,
    "conclusions": 0.08,
}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP]


def _term_overlap(query_terms: Sequence[str], text: str) -> float:
    if not query_terms or not text:
        return 0.0
    tl = text.lower()
    hits = 0
    for t in query_terms:
        if t and t in tl:
            hits += 1.0
    return min(1.0, hits / len(query_terms))


class PageIndexAdapter:
    """Loads precomputed PageIndex artifacts and navigates them per paper.

    Deterministic hierarchical navigation by default (works offline). When an
    LLM endpoint is configured (config.reasoning_llm), navigation can be
    delegated to an LLM reasoning step; the adapter interface stays the same.
    """

    def __init__(
        self,
        doc_index: Any = None,
        pageindex_dir: Optional[Path] = None,
        config: Optional[V2Config] = None,
        trace: Any = None,
    ) -> None:
        self.doc_index = doc_index
        self.config = config or DEFAULT_CONFIG
        self.pageindex_dir = Path(pageindex_dir) if pageindex_dir is not None else Path(self.config.pageindex_dir)
        self.trace = trace or NullTrace()
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._status: Dict[str, str] = {}           # paper_id -> success|unavailable|failed
        self._library_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # library probe
    # ------------------------------------------------------------------
    def library_version(self) -> str:
        """Version of the official PageIndex library used to build artifacts."""
        try:
            from pageindex import _version  # type: ignore
            return _version.sdk_version()
        except Exception:  # noqa: BLE001
            return "unknown"

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def build_or_load(self, paper_id: str, force: bool = False) -> str:
        """Load the precomputed artifact; build it when missing (offline op).

        Returns 'loaded' | 'built'. In the ONLINE pipeline this must not be
        called for papers without artifacts - the builder is a separate
        offline command (spec part 25); this method exists so tests and tooling
        can lazily materialize an artifact.
        """
        path = self.pageindex_dir / f"{paper_id}.json"
        if not force and path.exists():
            return self.load_paper(paper_id)
        if not path.exists() and not self.config.enable_pageindex:
            self._status[paper_id] = "unavailable"
            return "unavailable"
        from src.retrieval_v2.pageindex_build import build_pageindex_artifact
        if self.doc_index is None:
            self._status[paper_id] = "unavailable"
            return "unavailable"
        try:
            build_pageindex_artifact(self.doc_index, paper_id, self.pageindex_dir, self.config)
            self.load_paper(paper_id)
            return "built"
        except Exception as exc:  # noqa: BLE001
            self._status[paper_id] = "failed"
            self.trace.log("PAGEINDEX_LOAD", params={"paper_id": paper_id},
                           result={"status": "failed", "error": str(exc)})
            return "unavailable"

    def load_paper(self, paper_id: str) -> str:
        path = self.pageindex_dir / f"{paper_id}.json"
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self._artifacts[paper_id] = artifact
            self._status[paper_id] = "success"
            self.trace.log(
                "PAGEINDEX_LOAD",
                params={"paper_id": paper_id, "artifact": str(path)},
                result={
                    "status": "success",
                    "n_chunks": artifact.get("metadata", {}).get("n_chunks"),
                    "n_tree_nodes": artifact.get("metadata", {}).get("n_tree_nodes"),
                    "format": artifact.get("metadata", {}).get("format"),
                },
            )
            return "loaded"
        except Exception as exc:  # noqa: BLE001
            self._status[paper_id] = "failed" if path.exists() else "unavailable"
            self.trace.log("PAGEINDEX_LOAD", params={"paper_id": paper_id},
                           result={"status": self._status[paper_id], "error": str(exc)})
            return "unavailable"

    def available(self, paper_id: str) -> bool:
        if paper_id in self._artifacts:
            return True
        return self.load_paper(paper_id) == "loaded"

    def paper_status(self, paper_id: str) -> str:
        return self._status.get(paper_id, "unavailable")

    def status_summary(self) -> Dict[str, Any]:
        loaded = sum(1 for s in self._status.values() if s == "success")
        return {
            "library": self.library_version(),
            "artifacts_loaded": loaded,
            "per_paper": dict(self._status),
            "pageindex_dir": str(self.pageindex_dir),
        }

    # ------------------------------------------------------------------
    # tree navigation helpers
    # ------------------------------------------------------------------
    def _nodes(self, paper_id: str) -> List[Dict[str, Any]]:
        artifact = self._artifacts.get(paper_id) or {}
        return artifact.get("tree") or []

    def _iter_preorder(self, nodes: Sequence[Dict[str, Any]], ancestors: List[Dict[str, Any]] = None):
        ancestors = ancestors or []
        for node in nodes:
            yield node, ancestors
            children = node.get("nodes") or []
            yield from self._iter_preorder(children, ancestors + [node])

    def children(self, paper_id: str, node_id: str) -> List[Dict[str, Any]]:
        for node, _anc in self._iter_preorder(self._nodes(paper_id)):
            if str(node.get("node_id", "")) == node_id:
                return node.get("nodes") or []
        return []

    def parent(self, paper_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        nodes = self._nodes(paper_id)
        for parent, _anc in self._iter_preorder(nodes):
            for child in parent.get("nodes") or []:
                if str(child.get("node_id", "")) == node_id:
                    return parent
        return None

    def node_id_for_chunk(self, paper_id: str, chunk_id: str) -> str:
        artifact = self._artifacts.get(paper_id) or {}
        return artifact.get("chunk_to_node", {}).get(chunk_id, "")

    def resolve_node(self, paper_id: str, pageindex_node_id: str) -> Dict[str, Any]:
        """Map a PageIndex node onto existing corpus chunk ids (part 13)."""
        artifact = self._artifacts.get(paper_id) or {}
        node_map = artifact.get("node_map", {}) or {}
        rec = node_map.get(pageindex_node_id, {}) or {}
        chunk_ids = list(rec.get("chunk_ids", []))
        self.trace.log(
            "PAGEINDEX_NODE_RESOLVED",
            params={"paper_id": paper_id, "pageindex_node": pageindex_node_id,
                    "title": rec.get("title", "")},
            result={"chunk_ids": chunk_ids, "section": rec.get("section", "")},
        )
        return {
            "paper_id": paper_id,
            "pageindex_node_id": pageindex_node_id,
            "title": rec.get("title", ""),
            "section": rec.get("section", ""),
            "chunk_ids": chunk_ids,
            "page_refs": rec.get("page_refs"),
        }

    # ------------------------------------------------------------------
    # navigation (reasoning-driven, deterministic by default)
    # ------------------------------------------------------------------
    def navigate(self, paper_id: str, navigation_query: str, top_k: Optional[int] = None,
                 requirement_id: str = "") -> List[PageIndexHit]:
        """Find WHERE the evidence lives inside one paper.

        The navigation query is the requirement's paper-local navigation
        objective (part 12) - never a giant new global retrieval query. Node
        references (with reasons) are returned, not text blobs.
        """
        top_k = top_k or self.config.pageindex_nav_top_k
        if not self.available(paper_id):
            return []
        q_terms = _tokenize(navigation_query or "")
        if not q_terms:
            return []

        scored: List[Dict[str, Any]] = []
        nodes = self._nodes(paper_id)
        for node, ancestors in self._iter_preorder(nodes):
            nid = str(node.get("node_id", ""))
            title = node.get("title", "") or ""
            text = node.get("text", "") or ""
            section_path = " > ".join([a.get("title", "") for a in ancestors] + [title])
            section_low = section_path.lower()

            # title + full breadcrumb context: "Table T2" under
            # "Results > Recurrent coarctation (re-CoA)" carries the condition.
            position_text = " ".join([a.get("title", "") for a in ancestors] + [title])
            title_overlap = _term_overlap(q_terms, position_text)
            text_head = _term_overlap(q_terms, text[:400])
            score = 2.6 * title_overlap + 1.0 * text_head

            # section priors
            for sec, prior in _SECTION_PRIOR.items():
                if sec in section_low:
                    score += prior
            # table / figure structural hints
            if re.search(r"\btable\b", section_low) or "table " in section_low.lower():
                score += 0.45
            if re.search(r"\bfigure\b", section_low):
                score += 0.30
            # field hint: percentages / p-values -> tables
            nav_low = (navigation_query or "").lower()
            if re.search(r"percent|p-values?|confidence interval|proportion", nav_low) and "table" in section_low.lower():
                score += 0.50
            if re.search(r"percent|p-values?|confidence interval|proportion", nav_low) and "figure" in section_low.lower():
                score += 0.15
            # target/condition hints
            cond_m = re.search(r"associated with ([a-z ]+)", nav_low)
            if cond_m and cond_m.group(1).strip() and cond_m.group(1).strip() in section_low:
                score += 0.60

            if score < self.config.pageindex_min_relevance:
                continue
            scored.append({
                "node_id": nid,
                "title": title,
                "section": section_path,
                "level": len(ancestors) + 1,
                "relevance": score,
                "text": text,
            })

        scored.sort(key=lambda d: d["relevance"], reverse=True)
        # favor structural breadth: don't flood with adjacent paragraph nodes
        picked: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for d in scored:
            if d["node_id"] in seen_ids:
                continue
            seen_ids.add(d["node_id"])
            picked.append(d)
            if len(picked) >= top_k:
                break

        hits: List[PageIndexHit] = []
        selected_nodes: List[Dict[str, Any]] = []
        reasons: List[str] = []
        for d in picked:
            reason = self._reason_for(d, navigation_query)
            reasons.append(reason)
            trimmed_section = d["section"]
            if trimmed_section.startswith(paper_id + " > "):
                trimmed_section = trimmed_section[len(paper_id) + 3:]
            hits.append(PageIndexHit(
                paper_id=paper_id,
                pageindex_node_id=d["node_id"],
                title=d["title"],
                section=trimmed_section,
                chunk_ids=self._chunk_ids_for(paper_id, d["node_id"]),
                reason=reason,
                relevance=min(1.0, d["relevance"] / max(1e-9, (scored[0]["relevance"] if scored else 1.0))),
                level=d["level"],
            ))
            selected_nodes.append({"node_id": d["node_id"], "title": d["title"],
                                   "section": trimmed_section,
                                   "relevance": round(min(1.0, d["relevance"] / max(1e-9, (scored[0]["relevance"] if scored else 1.0))), 4)})

        self.trace.log(
            "PAGEINDEX_SEARCH",
            params={"paper_id": paper_id, "requirement_id": requirement_id,
                    "query": navigation_query, "top_k": top_k},
            result={"selected_nodes": selected_nodes, "reasons": reasons,
                    "status": self.paper_status(paper_id)},
        )
        for d in picked:
            self.trace.log(
                "PAGEINDEX_NODE_SELECTED",
                params={"paper_id": paper_id, "pageindex_node_id": d["node_id"],
                        "title": d["title"], "section": d["section"],
                        "requirement_id": requirement_id},
                result={"reason": self._reason_for(d, navigation_query)},
            )
        return hits

    def _chunk_ids_for(self, paper_id: str, node_id: str) -> List[str]:
        artifact = self._artifacts.get(paper_id) or {}
        return (artifact.get("node_map", {}) or {}).get(node_id, {}).get("chunk_ids", [])

    def _reason_for(self, d: Dict[str, Any], navigation_query: str) -> str:
        q_terms = _tokenize(navigation_query or "")
        title_matches = [t for t in q_terms if t in (d.get("title") or "").lower()]
        text_matches = [t for t in q_terms if t in ((d.get("text") or "")[:400]).lower() and t not in title_matches]
        parts: List[str] = []
        if title_matches:
            parts.append("title matches " + ", ".join(title_matches[:4]))
        if text_matches:
            parts.append("text matches " + ", ".join(text_matches[:4]))
        section = (d.get("section") or "").lower()
        if "table" in section:
            parts.append("tabular evidence region")
        elif "figure" in section:
            parts.append("figure region")
        elif "results" in section:
            parts.append("Results section")
        elif "discussion" in section:
            parts.append("Discussion section")
        return "; ".join(parts) if parts else "structural navigation candidate"
