"""Greedy requirement-coverage selection, repair pass, coverage report
(V2.1 parts 20-21, 30-33).

Selection is set-aware: each step maximizes MARGINAL value
(base relevance + new requirement coverage + evidence-type diversity
 + paper diversity - redundancy - mismatch).

COVERAGE asks "does this evidence satisfy the whole requirement?" (part 20):
for numerical technique-comparison questions a table row + table header +
footnote can satisfy target + outcome + requested statistics TOGETHER - the
coverage check uses the contextual requirement_match already computed on the
contextual evidence (table context for rows), never a literal per-chunk
phrase hunt. The greedy set therefore counts a table row WITH its assembled
context as covering the requirement.

REPAIR (part 21) never searches for generic related evidence: an uncovered
requirement triggers a FOCUSED re-run over the retained relevant papers -
PageIndex navigation on the requirement's navigation objective, local BM25
and local pgvector per variant, table/figure/section navigation. If still
uncovered after that, it stays HONESTLY uncovered - no silent substitution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from medrag.retrieval_v2.models import CoverageReport, EvidenceCandidate, Requirement


def _normalize_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1.0
    return [(v - lo) / rng for v in values]


def greedy_select(
    candidates: Sequence[EvidenceCandidate],
    requirements: Sequence[Requirement],
    config: Optional[V2Config] = None,
    trace: Any = None,
) -> List[EvidenceCandidate]:
    """Greedy marginal-value selection of the final evidence set (section 30).
    """
    cfg = config or DEFAULT_CONFIG
    pool = list(candidates)
    if not pool:
        return []
    req_ids = [r.id for r in requirements]
    n_reqs = max(1, len(req_ids))

    base_vals = [c.final_score for c in pool]
    norm = _normalize_scores(base_vals)
    for c, v in zip(pool, norm):
        c.final_score = v

    selected: List[EvidenceCandidate] = []
    covered: set = set()
    selected_papers: set = set()
    selected_types: set = set()
    doc_counts: Dict[str, int] = defaultdict(int)
    remaining = list(range(len(pool)))

    steps = min(cfg.final_slots, len(pool))
    for _ in range(steps):
        best_idx = -1
        best_marginal = -1e9
        best_reason: Dict[str, Any] = {}
        for idx in remaining:
            c = pool[idx]
            new_reqs = [r for r in c.covered_requirements if r not in covered]
            new_types = 1.0 if c.node_type not in selected_types else 0.0
            new_papers = 1.0 if c.paper_id not in selected_papers else 0.0
            doc_penalty = _doc_penalty(cfg, doc_counts.get(c.paper_id, 0))
            marginal = (
                c.final_score
                + cfg.new_requirement_bonus * (len(new_reqs) / n_reqs)
                + cfg.new_evidence_type_bonus * new_types
                + cfg.new_paper_bonus * new_papers
                - cfg.redundancy_alpha * doc_penalty
            )
            if marginal > best_marginal:
                best_marginal = marginal
                best_idx = idx
                best_reason = {
                    "base": round(c.final_score, 4),
                    "new_requirements": new_reqs,
                    "new_evid_type": bool(new_types),
                    "new_paper": bool(new_papers),
                    "doc_penalty": round(doc_penalty, 4),
                    "marginal": round(marginal, 4),
                }
        if best_idx < 0:
            break
        chosen = pool[best_idx]
        remaining.remove(best_idx)
        covered.update(chosen.covered_requirements)
        selected_papers.add(chosen.paper_id)
        selected_types.add(chosen.node_type)
        doc_counts[chosen.paper_id] += 1
        chosen.selection_rank = len(selected) + 1
        chosen.selection_reason = best_reason
        selected.append(chosen)

    if trace is not None:
        trace.log(
            "greedy_selection",
            params={"slots": cfg.final_slots, "n_candidates": len(pool)},
            result={
                "n_selected": len(selected),
                "n_documents": len(selected_papers),
                "covered_requirements": sorted(covered),
                "coverage_fraction": round(len(covered) / n_reqs, 4),
            },
        )
    return selected


def _doc_penalty(cfg: V2Config, count: int) -> float:
    if count >= cfg.doc_soft_cap:
        return 1.0
    return count / max(1, cfg.doc_soft_cap)


def repair_uncovered(
    selected: List[EvidenceCandidate],
    candidates_by_requirement: Dict[str, List[EvidenceCandidate]],
    requirements: Sequence[Requirement],
    local_retriever: Any = None,
    papers_by_requirement: Optional[Dict[str, List[str]]] = None,
    config: Optional[V2Config] = None,
    trace: Any = None,
) -> Tuple[List[EvidenceCandidate], List[str]]:
    """Targeted repair pass for uncovered requirements (V2.1 part 21).

    For every uncovered requirement the pass re-runs the FULL focused local
    machinery over the RETAINED per-requirement papers:
        - PageIndex navigation on the requirement's navigation objective
        - local BM25 + local pgvector for every retrieval variant
        - table/figure/section navigation via the logical document index
    A candidate that still cannot cover the requirement after that is left
    uncovered - never replaced with unrelated generic evidence.
    """
    if not requirements:
        return list(selected), []
    cfg = config or DEFAULT_CONFIG
    req_ids = [r.id for r in requirements]
    covered: set = set()
    for c in selected:
        covered.update(c.covered_requirements)
    uncovered = [rid for rid in req_ids if rid not in covered]
    if not uncovered:
        return list(selected), []

    selected_ids = {c.chunk_id for c in selected}
    selected = list(selected)

    for rid in uncovered:
        req = next((r for r in requirements if r.id == rid), None)
        if req is None:
            continue
        pool = [c for c in candidates_by_requirement.get(rid, []) if c.chunk_id not in selected_ids]
        if local_retriever is not None:
            papers = papers_by_requirement.get(rid, []) if papers_by_requirement else []
            if papers:
                # FOCUSED re-run on retained papers only (part 21): PageIndex
                # navigation objective + all variants; never generic text.
                new_cands = local_retriever.search_requirement_targeted(
                    req, papers, query_override=req.navigation_objective)
                pool = [c for c in new_cands if c.chunk_id not in selected_ids]
                new_cands_in_pool = pool
        # after the focused pass: does the requirement contextually cover?
        covering = [
            c for c in pool
            if rid in c.covered_requirements and c.final_score > -1.0
        ]
        if not covering:
            if trace is not None:
                trace.log(
                    f"repair_{rid}",
                    params={"requirement": rid, "status": "uncovered-no-evidence"},
                    result={"candidates_pool": len(pool), "covering": 0,
                            "focused_navigation": req.navigation_objective},
                )
            continue  # honest: stays uncovered

        best = max(covering, key=lambda c: c.final_score)

        replaceable = []
        for i, sc in enumerate(selected):
            others_cover = set()
            for j, oc in enumerate(selected):
                if i != j:
                    others_cover.update(oc.covered_requirements)
            unique = set(sc.covered_requirements) - others_cover
            if not unique:
                replaceable.append((i, sc))
        if not replaceable:
            replaceable = [(len(selected) - 1, selected[-1])]
        weakest_idx, _weakest = min(replaceable, key=lambda x: x[1].final_score)
        best.selection_rank = selected[weakest_idx].selection_rank
        best.selection_reason = {**best.selection_reason, "repair_replacement": True,
                                 "replaced_chunk": selected[weakest_idx].chunk_id}
        selected[weakest_idx] = best
        covered.add(rid)
        if trace is not None:
            trace.log(
                f"repair_{rid}",
                params={"requirement": rid, "status": "replaced",
                        "navigation_objective": req.navigation_objective},
                result={"replaced_chunk": best.selection_reason.get("replaced_chunk"),
                        "best_chunk": best.chunk_id, "node_type": best.node_type,
                        "table_id": best.table_id, "final_score": round(best.final_score, 4)},
            )

    still_uncovered = [rid for rid in req_ids if rid not in covered]
    selected.sort(key=lambda c: c.selection_rank)
    return selected, still_uncovered


def build_coverage_report(
    selected: Sequence[EvidenceCandidate],
    requirements: Sequence[Requirement],
    uncovered: Sequence[str],
    candidates_by_requirement: Optional[Dict[str, List[EvidenceCandidate]]] = None,
) -> CoverageReport:
    """Final coverage accounting (V2.1 part 20): honest, never fabricated."""
    req_ids = [r.id for r in requirements]
    covered: set = set()
    for c in selected:
        covered.update(c.covered_requirements)
    report = CoverageReport()
    report.covered = [rid for rid in req_ids if rid in covered]
    report.uncovered = list(dict.fromkeys(list(report.uncovered) + list(uncovered)))
    report.uncovered = [rid for rid in report.uncovered if rid in req_ids and rid not in report.covered]
    report.covered = [rid for rid in req_ids if rid in report.covered]
    n = len(req_ids)
    report.all_requirements_covered = (not report.uncovered) and n > 0
    report.coverage_fraction = (len(report.covered) / n) if n else 0.0

    per_req: Dict[str, Dict[str, Any]] = {}
    for req in requirements:
        idxs = [c for c in selected if req.id in c.covered_requirements]
        per_req[req.id] = {
            "covered": req.id in report.covered,
            "n_evidence": len(idxs),
            "best_chunks": [c.chunk_id for c in idxs[:3]],
            "node_types": sorted({c.node_type for c in idxs}),
            "table_ids": sorted({c.table_id for c in idxs if c.table_id}),
            "detected_fields": {c.chunk_id: dict(c.detected_fields) for c in idxs[:3]},
            "requirement": req.topic + (" " + req.population if req.population else ""),
        }
    report.per_requirement = per_req
    return report
