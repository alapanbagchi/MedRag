"""Evaluation metrics for retrieval V2 (spec sections 40-41).

Metrics:
    - branch paper recall@k / MRR per query (paper-level regression: q4
      must preserve PMC11743015, q5 -> PMC11705662, q6 -> PMC11694428)
    - final paper recall (known papers present in the selected neighborhood)
    - requirement recall and coverage (each hop must have evidence)
    - final evidence recall (known chunks in the final evidence set)

A pipeline cannot claim success simply because it retrieved 20 related
chunks: every branch counts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.retrieval_v2.models import (
    CoverageReport,
    EvidenceCandidate,
    PaperSelection,
    V2Plan,
)


def _recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant)
    return len(rel & set(ranked[:k])) / len(rel)


def _mrr(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, pid in enumerate(ranked, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0

def evaluate_run(
    plan: V2Plan,
    per_query_paper_ranking: Dict[str, List[str]],
    selected_papers: Sequence[PaperSelection],
    final_evidence: Sequence[EvidenceCandidate],
    coverage: CoverageReport,
    known_branch_papers: Optional[Dict[str, List[str]]] = None,
    known_chunks: Optional[Dict[str, List[str]]] = None,
    pageindex_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the V2 evaluation report for one run."""
    known_branch_papers = known_branch_papers or {}

    per_query: Dict[str, Dict[str, Any]] = {}
    for q in plan.queries:
        ranked = per_query_paper_ranking.get(q.id, [])
        known: List[str] = []
        for rid in q.requirement_ids:
            known.extend(known_branch_papers.get(rid, []))
        known.extend(known_branch_papers.get(q.id, []))
        known = list(dict.fromkeys(known))
        per_query[q.id] = {
            "known_papers": known,
            "recall@5": _recall_at_k(ranked, known, 5),
            "recall@10": _recall_at_k(ranked, known, 10),
            "recall@20": _recall_at_k(ranked, known, 20),
            "mrr": _mrr(ranked, known),
            "found_in_ranking": [p for p in known if p in ranked],
            "paper_ranking_top10": ranked[:10],
        }

    # final selection paper recall
    all_known_papers: List[str] = []
    for v in known_branch_papers.values():
        all_known_papers.extend(v)
    all_known_papers = list(dict.fromkeys(all_known_papers))
    selected_ids = [p.paper_id for p in selected_papers]
    final_paper_recall = (
        len([p for p in all_known_papers if p in selected_ids]) / len(all_known_papers)
        if all_known_papers else 0.0
    )
    known_selected = [p for p in all_known_papers if p in selected_ids]
    known_missing = [p for p in all_known_papers if p not in selected_ids]

    req_ids = [r.id for r in plan.requirements]
    n = len(req_ids)
    requirement_recall = len(coverage.covered) / n if n else 0.0
    branch_preserved = all(p in selected_ids for p in known_selected) if known_selected else True

    final_chunk_ids = [c.chunk_id for c in final_evidence]
    chunk_recall: Dict[str, float] = {}
    if known_chunks:
        for rid, chunks in known_chunks.items():
            chunk_recall[rid] = (
                len(set(chunks) & set(final_chunk_ids)) / len(chunks) if chunks else 0.0
            )

    # pageindex usage metrics (V2.1 part 28): how many final evidence
    # candidates carry PageIndex node provenance and were resolved to chunks
    pageindex_evidence = [c for c in final_evidence if c.pageindex_node_ids]
    table_context_evidence = [c for c in final_evidence if c.table_context]

    return {
        "per_query_paper_metrics": per_query,
        "final_paper_recall": round(final_paper_recall, 4),
        "known_papers_found": known_selected,
        "known_papers_missing": known_missing,
        "branch_papers_preserved_in_selection": bool(branch_preserved),
        "requirement_recall": round(requirement_recall, 4),
        "coverage_fraction": round(coverage.coverage_fraction, 4),
        "hop_coverage": {
            rid: {"covered": rid in coverage.covered} for rid in req_ids
        },
        "final_evidence_chunk_recall": chunk_recall,
        "n_final_evidence": len(final_evidence),
        "pageindex_evidence_count": len(pageindex_evidence),
        "table_context_evidence_count": len(table_context_evidence),
        "pageindex_status": (pageindex_summary or {}).get("per_paper", {}),
        "note": "Paper recall and MRR are computed against branch-level known papers (spec 40-41); requirement recall is the fraction of hops with evidence in the final set.",
    }
