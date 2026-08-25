"""V2 local retrieval inside selected papers (V2.1 parts 14, 29-31).

After the paper neighborhood is chosen (5-20 papers), retrieval switches to
paper-local mode. For EVERY selected paper the three sources are merged:

        selected paper
       ────────────────
    PageIndex  |  BM25  |  pgvector
        │       │        │
        └───────┼────────┘
             candidate union
             structural expansion (LogicalDocumentIndex)
             table-aware context construction
             EvidenceCandidate

PageIndex is a high-precision structural/navigation source (decides WHERE to
look), BM25 recovers exact terminology, pgvector recovers semantic variants,
and the LogicalDocumentIndex reconstructs exact evidence context. Part 30:
every retrieval VARIANT of the requirement is executed independently
(BM25(v) + dense(v) + per-variant RRF) and the candidates are unioned.

The module also fixes the anchor-metadata defect that previously degraded
table rows into generic paragraphs: anchor records are enriched with their
logical node metadata BEFORE structural expansion so expand_anchor recognizes
table rows, assembles the whole table and attaches headers/footnotes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from src.retrieval_v2.document_index import tokens
from src.retrieval_v2.expand import expand_anchors
from src.retrieval_v2.models import EvidenceCandidate, Requirement, SearchQuery
from src.retrieval_v2.retriever import NullTrace, rrf_fuse
from src.retrieval_v2.table_context import (
    build_table_context,
    detect_paragraph_fields,
)


class LocalRetriever:
    """Runs hybrid retrieval restricted to one paper at a time."""

    def __init__(
        self,
        bm25: Any,
        dense: Any,
        doc_index: Any,
        query_encoder: Any,
        config: Optional[V2Config] = None,
        trace: Any = None,
        pageindex: Any = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.doc_index = doc_index
        self.query_encoder = query_encoder
        self.config = config or DEFAULT_CONFIG
        self.trace = trace or NullTrace()
        self.pageindex = pageindex
        self._qvec_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # restricted dense search helpers
    # ------------------------------------------------------------------
    def _dense_restricted(
        self, qvec: Any, paper_id: str, top_k: int
    ) -> List[tuple[str, float, int]]:
        """pgvector-filtered search; FAISS fallback filters a deep global scan."""
        target = getattr(self.dense, "store", None)
        store = target or self.dense
        if store is not None and hasattr(store, "search_filtered"):
            hits = store.search_filtered(qvec, top_k=top_k, document_id=paper_id)
            return [(cid, float(sc), i + 1) for i, (cid, sc) in enumerate(hits)]
        if hasattr(self.dense, "search_single"):
            hits = self.dense.search_single(qvec, max(200, top_k * 20))
            paper_ids = set(self.doc_index.get_paper_chunk_ids(paper_id))
            out = []
            rank = 1
            for h in hits:
                if h.chunk_id in paper_ids:
                    out.append((h.chunk_id, float(h.score), rank))
                    rank += 1
                    if len(out) >= top_k:
                        break
            return out
        return []

    def _variant_local_search(
        self, paper_id: str, variant_text: str, variant_id: str, cfg: V2Config
    ) -> List[Dict[str, Any]]:
        """BM25(variant) + pgvector(variant), fused with query-local RRF.

        Every variant is retrieved and fused INDEPENDENTLY (part 30); the
        caller unions the per-variant results.
        """
        qvec = None
        if self.query_encoder is not None:
            if variant_text in self._qvec_cache:
                qvec = self._qvec_cache[variant_text]
            else:
                qvec = self.query_encoder.encode([variant_text])[0]
                self._qvec_cache[variant_text] = qvec

        paper_chunk_ids = self.doc_index.get_paper_chunk_ids(paper_id)

        bm25_hits: List[tuple[str, float, int]] = []
        if self.bm25 is not None:
            try:
                hits = self.bm25.search_restricted(variant_text, paper_chunk_ids, cfg.local_bm25_depth)
                bm25_hits = [(h.chunk_id, float(h.score), h.rank) for h in hits]
            except Exception as exc:  # noqa: BLE001
                self.trace.log("local_bm25_warning", error=str(exc), detail=str(exc))

        dense_hits: List[tuple[str, float, int]] = []
        if qvec is not None and self.dense is not None and cfg.enable_dense:
            try:
                dense_hits = self._dense_restricted(qvec, paper_id, cfg.local_dense_depth)
            except Exception as exc:  # noqa: BLE001
                self.trace.log("local_dense_warning", error=str(exc), detail=str(exc))

        fused = rrf_fuse([bm25_hits, dense_hits], k=cfg.rrf_k, labels=["bm25", "dense"])
        for f in fused:
            f["paper_id"] = paper_id
            f["variant_id"] = variant_id
            f["methods"] = [lab for lab in ("bm25", "dense") if f["scores"].get(lab, 0.0) > 0]
        return fused[: cfg.local_variant_fused_depth]

    def _enrich_anchor(self, paper_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Attach LogicalDocumentIndex node metadata to an anchor record.

        This is the anchor-metadata fix: expand_anchor needs node_type /
        table_id to assemble whole tables instead of degrading table rows
        into generic paragraphs.
        """
        rec = self.doc_index.get_node(entry["chunk_id"]) or {}
        anchor = dict(entry)
        anchor["paper_id"] = paper_id
        anchor["node_type"] = rec.get("node_type", "paragraph")
        anchor["section"] = rec.get("section", "")
        anchor["subsection"] = rec.get("subsection", "")
        anchor["breadcrumb"] = list(rec.get("breadcrumb", []))
        anchor["position"] = int(rec.get("position", 0))
        anchor["table_id"] = rec.get("table_id") or None
        anchor["figure_id"] = rec.get("figure_id") or None
        anchor["parent_id"] = rec.get("parent_id") or None
        return anchor

    # ------------------------------------------------------------------
    # per-paper x requirement local search
    # ------------------------------------------------------------------
    def search_paper_requirement(
        self,
        paper_id: str,
        requirement: Requirement,
        queries: Sequence[SearchQuery],
    ) -> List[EvidenceCandidate]:
        """Hybrid paper-local retrieval for one requirement in one paper.

        Pipeline:
            1. pageindex navigation (reasoning WHERE to look)
            2. per-variant BM25 + dense, per-variant RRF
            3. candidate union (pageindex ∪ bm25 ∪ dense)
            4. structural expansion (LogicalDocumentIndex, bounded)
            5. table-aware context construction (part 15-16, 31)
            6. EvidenceCandidate with context + provenance
        """
        cfg = self.config
        variants = [q for q in (queries or []) if q and (q.text or "").strip()]
        if not variants:
            variants = [
                SearchQuery(id="q_local", requirement_ids=[requirement.id],
                            text=requirement.search_query_text(), variant_id="V0")
            ]

        # ── 1. PageIndex navigation ─────────────────────────────────
        nav_chunk_info: Dict[str, Dict[str, Any]] = {}
        pageindex_status = "unavailable"
        nav_query = requirement.navigation_objective or variants[0].text
        if self.pageindex is not None and cfg.enable_pageindex:
            try:
                nav_hits = self.pageindex.navigate(
                    paper_id, nav_query,
                    top_k=cfg.pageindex_nav_top_k,
                    requirement_id=requirement.id,
                )
                pageindex_status = self.pageindex.paper_status(paper_id) or "unavailable"
                for h in nav_hits:
                    for cid in h.chunk_ids:
                        info = nav_chunk_info.setdefault(
                            cid, {"node_ids": [], "reasons": [], "relevance": 0.0, "section": ""})
                        info["node_ids"].append(h.pageindex_node_id)
                        info["reasons"].append(h.reason)
                        info["relevance"] = max(info["relevance"], h.relevance)
                        if not info["section"]:
                            info["section"] = h.section
            except Exception as exc:  # noqa: BLE001
                pageindex_status = "failed"
                self.trace.log("PAGEINDEX_SEARCH_ERROR",
                               params={"paper_id": paper_id, "requirement_id": requirement.id},
                               error=str(exc))
        self.trace.log(
            "local_pageindex",
            params={"paper_id": paper_id, "requirement_id": requirement.id,
                    "status": pageindex_status, "navigation_query": nav_query},
            result={"nav_chunks": sorted(nav_chunk_info.keys()),
                    "n_nav_chunks": len(nav_chunk_info)},
        )

        # ── 2. per-variant hybrid retrieval + union (part 30) ────────
        union: Dict[str, Dict[str, Any]] = {}
        for q in variants:
            for f in self._variant_local_search(paper_id, q.text, q.variant_id or q.id, cfg):
                cid = f["chunk_id"]
                existing = union.get(cid)
                if existing is None:
                    f["variants"] = [q.variant_id or q.id]
                    union[cid] = f
                    continue
                existing["variants"].append(q.variant_id or q.id)
                if f["rrf_score"] > existing["rrf_score"]:
                    existing["rrf_score"] = f["rrf_score"]
                    existing["rank"] = f["rank"]
                    existing["scores"] = dict(f["scores"])
                for lab, sc in f["scores"].items():
                    if sc > existing["scores"].get(lab, 0.0):
                        existing["scores"][lab] = sc
                existing["methods"] = sorted(set(existing["methods"]) | set(f["methods"]))
        fused = sorted(union.values(), key=lambda d: d["rrf_score"], reverse=True)
        for i, f in enumerate(fused[: cfg.local_fused_depth], start=1):
            f["rank"] = i

        req_id = requirement.id

        # ── 3. anchors: hybrid top + PageIndex-selected chunks ───────
        anchors: List[Dict[str, Any]] = []
        seen_anchor: set = set()
        for f in fused[: cfg.local_anchor_hits]:
            if f["chunk_id"] in seen_anchor:
                continue
            seen_anchor.add(f["chunk_id"])
            anchors.append(self._enrich_anchor(paper_id, f))

        nav_chunks_by_relevance = sorted(
            nav_chunk_info.items(), key=lambda kv: kv[1]["relevance"], reverse=True)
        added_nav = 0
        for cid, info in nav_chunks_by_relevance:
            if cid in seen_anchor or len(anchors) >= cfg.local_anchor_hits + cfg.local_nav_anchor_rank:
                continue
            if added_nav >= cfg.local_nav_anchor_rank:
                break
            seen_anchor.add(cid)
            entry: Dict[str, Any] = {"chunk_id": cid, "rrf_score": nav_chunk_info[cid]["relevance"],
                                     "rank": 0, "methods": ["pageindex"], "scores": {"pageindex": nav_chunk_info[cid]["relevance"]}}
            anchors.append(self._enrich_anchor(paper_id, entry))
            added_nav += 1

        # evidence-type-aware boosting (kept from V2): tables/figures get a
        # chance to enter the pool even below the anchor cutoff.
        focus = requirement.focus or "evidence"
        pref = set(requirement.preferred_evidence_types or [])
        want_tables = bool(pref & {"table_row", "table_summary", "table_footnotes"})
        want_figures = bool(pref & {"figure"}) and focus != "table_lookup"
        boosted: List[Dict[str, Any]] = []
        table_seen, figure_seen = 0, 0
        for f in fused:
            if len(boosted) >= 3:
                break
            if f["chunk_id"] in seen_anchor:
                continue
            rec = self.doc_index.get_node(f["chunk_id"])
            ntype = rec["node_type"] if rec else ""
            if want_tables and ntype in ("table_summary", "table_row") and table_seen < 2:
                boosted.append(f)
                table_seen += 1
            elif want_figures and ntype == "figure" and figure_seen < 1:
                boosted.append(f)
                figure_seen += 1
        if boosted:
            self.trace.log(
                "local_evidence_boost",
                params={"paper_id": paper_id, "requirement_id": req_id, "focus": focus},
                result={"boosted": [f["chunk_id"] for f in boosted]},
            )
        for f in boosted:
            if f["chunk_id"] in seen_anchor:
                continue
            seen_anchor.add(f["chunk_id"])
            anchors.append(self._enrich_anchor(paper_id, f))

        # ── 4. bounded structural expansion (LogicalDocumentIndex) ───
        expanded = expand_anchors(self.doc_index, paper_id, anchors, cfg)

        # ── 5. candidates with table-aware context ───────────────────
        best_local_rank: Dict[str, int] = {}
        best_local_scores: Dict[str, Dict[str, float]] = {}
        best_methods: Dict[str, List[str]] = {}
        best_variants: Dict[str, List[str]] = {}
        for f in fused[: cfg.local_fused_depth]:
            cid = f["chunk_id"]
            best_local_rank.setdefault(cid, f["rank"])
            if cid not in best_local_scores or f["rrf_score"] > best_local_scores[cid].get("rrf", 0.0):
                best_local_scores[cid] = {
                    "rrf": float(f["rrf_score"]),
                    **{lab: float(v) for lab, v in f["scores"].items() if v > 0},
                }
                best_methods[cid] = list(f["methods"])
                best_variants[cid] = list(f.get("variants", []))

        qids = [q.id for q in variants]
        candidates: List[EvidenceCandidate] = []
        seen_cands: set = set()
        for node in expanded:
            cid = node["chunk_id"]
            if cid in seen_cands:
                continue
            seen_cands.add(cid)
            cand = EvidenceCandidate(
                chunk_id=cid,
                paper_id=paper_id,
                requirement_ids=[req_id],
                source_queries=list(qids),
                node_type=node.get("node_type", "paragraph"),
                section=node.get("section", ""),
                subsection=node.get("subsection", ""),
                breadcrumb=list(node.get("breadcrumb", [])),
                position=int(node.get("position", 0)),
                table_id=node.get("table_id") or None,
                figure_id=node.get("figure_id") or None,
                parent_id=node.get("parent_id") or None,
                text=node.get("text") or "",
                token_count=int(node.get("token_count", tokens(node.get("text") or ""))),
                local_scores=best_local_scores.get(cid, {"rrf": 0.0}),
                local_rank=best_local_rank.get(cid, 999),
                retrieval_methods=best_methods.get(cid, []),
                expanded_from=[node.get("expanded_from")] if node.get("expanded_from") else [],
            )
            # PageIndex provenance
            info = nav_chunk_info.get(cid)
            if info:
                cand.pageindex_node_ids = list(info["node_ids"])
                cand.pageindex_reasons = list(info["reasons"])
                cand.retrieval_methods = list(dict.fromkeys(cand.retrieval_methods + ["pageindex"]))
                if info["section"]:
                    cand.subsection = cand.subsection or info["section"]
            # table-aware context (parts 15-16, 31)
            if cand.is_table():
                tc, ctx_text, det = build_table_context(self.doc_index, paper_id, node, cfg)
                cand.table_context = tc or None
                cand.context_text = ctx_text
                if tc and tc.get("table_id"):
                    tbl = self.doc_index.get_table(paper_id, tc["table_id"])
                    cand.context_node_ids = list(tbl.get("all_node_ids", []))
                cand.detected_fields = det
            else:
                cand.detected_fields = detect_paragraph_fields(cand.text)
                cand.context_text = cand.text
                cand.context_node_ids = [cid]
            candidates.append(cand)

        candidates.sort(key=lambda c: c.local_scores.get("rrf", 0.0), reverse=True)
        self.trace.log(
            f"local_search_{paper_id}_{req_id}",
            params={"paper_id": paper_id, "requirement_id": req_id,
                    "navigation_query": nav_query, "n_variants": len(variants),
                    "n_paper_chunks": len(self.doc_index.get_paper_chunk_ids(paper_id)),
                    "pageindex_status": pageindex_status},
            result={"bm25+pgvector_fused": len(fused), "candidates": len(candidates),
                    "pageindex_chunks": len(nav_chunk_info),
                    "top": [{"chunk_id": c.chunk_id, "type": c.node_type, "rank": c.local_rank,
                             "pageindex": bool(c.pageindex_node_ids),
                             "table_id": c.table_id} for c in candidates[:8]]},
        )
        return candidates

    # ------------------------------------------------------------------
    # batch across papers
    # ------------------------------------------------------------------
    def search_papers(
        self,
        papers_by_requirement: Dict[str, List[str]],
        requirements: Sequence[Requirement],
        queries_by_requirement: Dict[str, List[SearchQuery]],
    ) -> Dict[str, List[EvidenceCandidate]]:
        """Run local search for every (paper, requirement) pair."""
        out: Dict[str, List[EvidenceCandidate]] = {}
        for req in requirements:
            papers = papers_by_requirement.get(req.id, [])
            qs = queries_by_requirement.get(req.id, [])
            pool: List[EvidenceCandidate] = []
            seen_ids: set = set()
            for pid in papers:
                cands = self.search_paper_requirement(pid, req, qs)
                for c in cands:
                    if c.chunk_id in seen_ids:
                        continue
                    seen_ids.add(c.chunk_id)
                    pool.append(c)
            out[req.id] = pool
        return out

    def search_requirement_targeted(
        self,
        requirement: Requirement,
        paper_ids: Sequence[str],
        query_override: Optional[str] = None,
        queries: Optional[Sequence[SearchQuery]] = None,
    ) -> List[EvidenceCandidate]:
        """Targeted repair search for one requirement (spec part 21).

        Re-runs local search with the requirement's PageIndex navigation
        objective (and every retrieval variant) over the retained papers only.
        When 'query_override' is given (e.g. an enriched objective), it is used
        as the navigation objective. Never triggers a second global run and
        never fills with generic related evidence.
        """
        qs = list(queries or [])
        if not qs:
            from src.retrieval_v2.models import SearchQuery as _SQ
            qs = [_SQ(id="q_repair", requirement_ids=[requirement.id],
                      text=requirement.search_query_text(), variant_id="V0")]
        if query_override and requirement.id:
            requirement.navigation_objective = query_override
        pool: List[EvidenceCandidate] = []
        seen: set = set()
        for pid in paper_ids:
            cands = self.search_paper_requirement(pid, requirement, qs)
            for c in cands:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                pool.append(c)
        return pool

    def query_ids_for(self, req_id: str, queries: Optional[Sequence[SearchQuery]]) -> List[str]:
        if not queries:
            return []
        return [q.id for q in queries if req_id in q.requirement_ids]
