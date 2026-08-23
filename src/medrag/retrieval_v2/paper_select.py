"""Paper-level aggregation and selection (spec sections 13-17).

The paper is the primary discovery unit. This module:

1. aggregates per-query fused chunks into paper_query_score (section 13):
       score = 0.50*max + 0.20*mean_top3 + 0.10*support (+section diversity
               + method agreement), normalized per query
2. builds per-requirement shortlists (top N per branch, section 15) and
   takes their UNION so a paper essential to ONE branch can never be
   eliminated by cross-branch competition (section 16),
3. ranks papers across requirements with strongest-branch dominance
   (section 14) and returns the handoff objects (section 17).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from medrag.retrieval_v2.models import (
    PaperQueryScore,
    PaperSelection,
    QueryLocalResult,
    Requirement,
    SearchQuery,
)


def _minmax(values: Sequence[float]) -> List[float]:
    """Min-max normalize to [0, 1] per pool; a degenerate pool (no spread)
    scores every member 1.0 - i.e. all the evidence there is for this query."""
    if not values:
        return []
    lo, hi = float(min(values)), float(max(values))
    if hi == lo:
        return [1.0 for _ in values]
    return [(float(v) - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# Stage 1: paper_query_score per (query, paper)  (section 13)
# ---------------------------------------------------------------------------


def aggregate_papers(
    query_results: Sequence[QueryLocalResult],
    doc_index: Any,
    config: Optional[V2Config] = None,
    trace: Any = None,
) -> Dict[Tuple[str, str], PaperQueryScore]:
    """Build paper_query_score for every (query, paper) pair.

    Each query is aggregated independently (query-local) so a paper strong for
    q4 alone is preserved before any cross-query fusion happens.
    """
    cfg = config or DEFAULT_CONFIG
    weights = cfg.paper_query_weights
    scores: Dict[Tuple[str, str], PaperQueryScore] = {}

    for res in query_results:
        # group fused hits by paper
        by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for f in res.fused:
            pid = f.get("paper_id") or ""
            if pid:
                by_paper[pid].append(f)
        if not by_paper:
            continue

        per_paper: Dict[str, Dict[str, Any]] = {}
        for pid, entries in by_paper.items():
            entries.sort(key=lambda e: e.get("rrf_score", 0.0), reverse=True)
            rrf_scores = [float(e.get("rrf_score", 0.0)) for e in entries]
            max_score = rrf_scores[0] if rrf_scores else 0.0
            mean_topk = sum(rrf_scores[: cfg.paper_query_mean_top_n]) / cfg.paper_query_mean_top_n
            n_support = len(entries)
            methods = set()
            for e in entries:
                methods.update(e.get("methods", []))
            chunk_ids = [e["chunk_id"] for e in entries]
            meta = doc_index.chunk_metadata_for(chunk_ids)
            sections = set()
            types = set()
            for cid in chunk_ids:
                rec = meta.get(cid)
                if rec:
                    if rec.get("section"):
                        sections.add(rec["section"])
                    if rec.get("node_type"):
                        types.add(rec["node_type"])
            section_diversity = len(sections) / max(1, n_support)
            method_agreement = len(methods) / 2.0
            evidence_diversity = len(types)
            per_paper[pid] = {
                "max_score": max_score,
                "mean_topk": mean_topk,
                "n_support": n_support,
                "section_diversity": section_diversity,
                "method_agreement": method_agreement,
                "evidence_diversity": evidence_diversity,
                "chunk_ids": chunk_ids,
                "sections": sorted(sections),
                "types": sorted(types),
            }

        # normalize per query across its papers, then weighted sum
        keys = list(per_paper.keys())
        norm_max = _minmax([per_paper[k]["max_score"] for k in keys])
        norm_mean = _minmax([per_paper[k]["mean_topk"] for k in keys])
        norm_support = _minmax([float(per_paper[k]["n_support"]) for k in keys])
        norm_section = _minmax([per_paper[k]["section_diversity"] for k in keys])
        norm_method = _minmax([per_paper[k]["method_agreement"] for k in keys])
        norm_evidence = _minmax([float(per_paper[k]["evidence_diversity"]) for k in keys])

        for i, pid in enumerate(keys):
            d = per_paper[pid]
            score = (
                weights["max_score"] * norm_max[i]
                + weights["mean_topk"] * norm_mean[i]
                + weights["support"] * norm_support[i]
                + weights["section_diversity"] * norm_section[i]
                + weights["method_agreement"] * norm_method[i]
            )
            pqs = PaperQueryScore(
                paper_id=pid,
                query_id=res.query.id,
                requirement_ids=list(res.query.requirement_ids),
                max_score=norm_max[i],
                mean_topk=norm_mean[i],
                n_support=d["n_support"],
                section_diversity=norm_section[i],
                method_agreement=norm_method[i],
                evidence_diversity=norm_evidence[i],
                score=score,
                best_chunks=d["chunk_ids"][:5],
                sections=d["sections"],
                evidence_types=d["types"],
                supporting_chunks=d["chunk_ids"],
            )
            scores[(res.query.id, pid)] = pqs

        if trace is not None:
            trace.log(
                f"{res.query.id}_paper_aggregation",
                params={"query_id": res.query.id, "requirement_ids": res.query.requirement_ids},
                result={"n_papers": len(per_paper),
                        "top_papers": [
                            {"paper_id": k, "score": round(scores[(res.query.id, k)].score, 4)}
                            for k in keys[:10]
                        ]},
            )
    return scores


# ---------------------------------------------------------------------------
# Stage 2: paper selection across requirements (sections 14-17)
# ---------------------------------------------------------------------------


def select_papers(
    paper_scores: Dict[Tuple[str, str], PaperQueryScore],
    queries: Sequence[SearchQuery],
    requirements: Sequence[Requirement],
    doc_index: Any,
    config: Optional[V2Config] = None,
    trace: Any = None,
) -> Tuple[List[PaperSelection], Dict[str, List[PaperSelection]]]:
    """Rank papers across requirements and return the selected neighborhood.

    Returns (selected_papers, per_requirement_papers) where the per-requirement
    dict maps every requirement to its retained candidates (shortlist +
    guarantees) for V2 local search and the repair pass.
    """
    cfg = config or DEFAULT_CONFIG
    n = len(requirements)
    if n == 0:
        return [], {}

    # query -> requirement mapping
    q_to_reqs: Dict[str, List[str]] = {}
    for q in queries:
        q_to_reqs[q.id] = list(q.requirement_ids)

    # per-requirement raw scores (normalized per requirement)
    req_raw: Dict[str, Dict[str, float]] = {r.id: {} for r in requirements}
    for (qid, pid), pqs in paper_scores.items():
        for req_id in q_to_reqs.get(qid, []):
            cur = req_raw[req_id].get(pid, 0.0)
            req_raw[req_id][pid] = max(cur, pqs.score)

    # normalize per requirement (min-max over its papers)
    req_norm: Dict[str, Dict[str, float]] = {}
    for rid, m in req_raw.items():
        if not m:
            req_norm[rid] = {}
            continue
        v = list(m.values())
        lo, hi = min(v), max(v)
        rng = hi - lo if hi > lo else 1.0
        req_norm[rid] = {pid: (s - lo) / rng for pid, s in m.items()}

    all_papers: set = set()
    for rid, m in req_raw.items():
        all_papers.update(m.keys())

    # shortlists per requirement (section 15)
    shortlist: Dict[str, List[str]] = {}
    for rid in [r.id for r in requirements]:
        ranked = sorted(req_norm[rid].items(), key=lambda kv: kv[1], reverse=True)
        shortlist[rid] = [pid for pid, _ in ranked[: cfg.per_requirement_shortlist]]

    # paper x requirement matrix + composite score
    matrix: Dict[str, Dict[str, float]] = {pid: {} for pid in all_papers}
    paper_meta: Dict[str, Dict[str, Any]] = {}
    paper_selections: List[PaperSelection] = []
    for pid in all_papers:
        row = {rid: req_norm[rid].get(pid, 0.0) for rid in req_norm}
        matrix[pid] = row
        supported = [rid for rid, s in row.items() if s >= cfg.requirement_presence_threshold or pid in shortlist.get(rid, [])]
        strengths = [s for s in row.values() if s > 0]
        strongest = max(strengths) if strengths else 0.0
        coverage = len(supported) / n if n else 0.0
        mean_branch = sum(strengths) / len(strengths) if strengths else 0.0
        # evidence diversity / method agreement from the underlying PQS records
        pqs_for_paper = [p for (qid, ppid), p in paper_scores.items() if ppid == pid]
        types: set = set()
        sections: set = set()
        methods: set = set()
        best_chunks: List[str] = []
        for p in pqs_for_paper:
            types.update(p.evidence_types)
            sections.update(p.sections)
            best_chunks.extend(p.best_chunks)
            if p.method_agreement > 0.5:
                methods.add("both")
        type_diversity = min(1.0, len(types) / 5.0)
        section_diversity = min(1.0, len(sections) / 5.0)
        evidence_diversity = 0.5 * type_diversity + 0.5 * section_diversity
        method_agreement = 1.0 if "both" in methods else 0.0
        bridge = max(0, len(supported) - 1) * cfg.bridge_bonus_weight if len(requirements) > 1 else 0.0
        paper_score = (
            cfg.paper_rank_weights["strongest_branch"] * strongest
            + cfg.paper_rank_weights["requirement_coverage"] * coverage
            + cfg.paper_rank_weights["mean_branch"] * mean_branch
            + cfg.paper_rank_weights["evidence_diversity"] * evidence_diversity
            + cfg.paper_rank_weights["method_agreement"] * method_agreement
            + bridge
        )
        paper_meta[pid] = {
            "supported": supported,
            "strongest": strongest,
            "coverage": coverage,
            "mean_branch": mean_branch,
            "evidence_diversity": evidence_diversity,
            "method_agreement": method_agreement,
            "bridge": bridge,
            "score": paper_score,
            "best_chunks": best_chunks[:5],
            "types": sorted(types),
            "sections": sorted(sections),
        }
        qids = sorted({qid for qid, ppid in paper_scores if ppid == pid})
        paper_selections.append(
            PaperSelection(
                paper_id=pid,
                paper_score=paper_score,
                supported_requirements=supported,
                query_branches=qids,
                best_chunks=best_chunks[:5],
                sections=sorted(sections),
                evidence_types=sorted(types),
                per_requirement_scores={rid: round(s, 4) for rid, s in row.items()},
                reason="ranked",
                bridge=bridge > 0,
            )
        )

    # rank + branch guarantees
    paper_selections.sort(key=lambda p: p.paper_score, reverse=True)
    selected: List[PaperSelection] = paper_selections[: cfg.top_papers]
    added_ids = {p.paper_id for p in selected}

    # guarantee: every requirement keeps its branch top (section 15)
    for rid in shortlist:
        for pid in shortlist[rid][: cfg.branch_guarantee_top]:
            if pid in added_ids:
                continue
            sel = next((p for p in paper_selections if p.paper_id == pid), None)
            if sel is None:
                continue
            sel.guarantee = True
            sel.reason = f"branch guarantee for {rid}"
            selected.append(sel)
            added_ids.add(pid)
    selected.sort(key=lambda p: p.paper_score, reverse=True)

    per_req: Dict[str, List[PaperSelection]] = {}
    for rid in [r.id for r in requirements]:
        per_req[rid] = [
            p for p in paper_selections if pid_in(p.paper_id, shortlist[rid]) or pid_in(p.paper_id, req_norm[rid])
        ][: cfg.per_requirement_shortlist + 4]

    if trace is not None:
        trace.log(
            "paper_selection",
            params={"n_requirements": n, "top_papers": cfg.top_papers,
                    "per_requirement_shortlist": cfg.per_requirement_shortlist,
                    "branch_guarantee_top": cfg.branch_guarantee_top},
            result={
                "n_ranked_papers": len(paper_selections),
                "selected": [{"paper_id": p.paper_id, "score": round(p.paper_score, 4),
                              "supported": p.supported_requirements, "reason": p.reason} for p in selected],
                "matrix": matrix,
            },
        )
    return selected, per_req


def pid_in(pid: str, ids: list) -> bool:
    return pid in ids

