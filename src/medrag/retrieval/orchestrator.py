#!/usr/bin/env python3
"""
Multi-Query Retrieval Orchestrator for MedRAG
=============================================

Implements evidence-aware, diversity-aware, coverage-aware retrieval selection.

The selector is iterative and stateful: each greedy step evaluates marginal
value relative to the already-selected set, not just individual scores.

Output: 2 files only
--------------------
    <output>.md   — all expanded evidence text keyed by PMC ID
    <output>.log  — full execution trace with timestamps and durations

Usage:
    python -m medrag.retrieval.orchestrator --plan plan.json
    python -m medrag.retrieval.orchestrator --plan plan.json --output retrieval_runs/my_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from medrag.trace import TraceLogger

# ---------------------------------------------------------------------------
# Index paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_INDEX_DIR = _ROOT / "index"
_BM25_DIR = _INDEX_DIR / "bm25"
_CORPUS_PATH = _INDEX_DIR / "corpus.parquet"

# ---------------------------------------------------------------------------
# Configurable thresholds and weights
# ---------------------------------------------------------------------------

# Evidence coverage threshold on per-requirement NORMALIZED scores
REQUIREMENT_THRESHOLD = 0.50

# Diversity: how many selected passages from same doc before penalty kicks in
DOC_DIVERSITY_SOFT_CAP = 2

# Selection weights
DEFAULT_WEIGHTS = {
    "medcpt": 0.55,
    "evidence": 0.15,
    "query_coverage": 0.10,
    "intent": 0.10,
    "diversity": 0.10,
}

# Marginal selection bonuses/penalties
NEW_REQUIREMENT_BONUS = 0.25
QUERY_COVERAGE_BONUS = 0.10
DYNAMIC_DIVERSITY_WEIGHT = 0.10
REDUNDANCY_PENALTY = 0.20

# Redundancy sub-weights
SEMANTIC_REDUNDANCY_WEIGHT = 0.65
DOCUMENT_REDUNDANCY_WEIGHT = 0.35

# Document redundancy tiers
DOC_PENALTY_TIERS = {
    0: 0.00,
    1: 0.10,
    2: 0.20,
    3: 0.35,  # 3+ uses this value
}

# Evidence threshold calibration:
# 0.50 on per-requirement normalized scores = median of the candidate pool.
# This means "above average for this requirement within this query's pool."
# Raise to 0.60 or 0.70 for stricter coverage.


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class RetrievalEvent:
    """A single retrieval hit from one query x one branch."""
    query_id: str
    branch: str          # "bm25" or "dense"
    rank: int
    score: float


@dataclass
class Candidate:
    """A deduplicated candidate with multi-query provenance."""
    chunk_id: str
    hits: List[RetrievalEvent] = field(default_factory=list)

    # Populated at resolve time
    document_id: Optional[str] = None
    chunk_type: Optional[str] = None
    breadcrumb: Optional[str] = None
    text: Optional[str] = None

    # Computed after fusion
    rrf_score: float = 0.0

    # Computed after reranking
    reranker_score: float = 0.0
    original_rrf_rank: int = 0

    # Computed after evidence matching (per-requirement normalized)
    requirement_scores: Dict[int, float] = field(default_factory=dict)
    covered_requirements: List[int] = field(default_factory=list)
    coverage_count: int = 0
    coverage_fraction: float = 0.0

    # Computed after intent matching
    intent_score: float = 0.0

    # Computed after query coverage
    query_coverage: int = 0
    matched_queries: List[str] = field(default_factory=list)

    # Cached embedding (reconstructed from FAISS, used for diversity)
    embedding: Optional[np.ndarray] = None

    # Selection
    selection_rank: int = 0
    selection_reason: Dict[str, float] = field(default_factory=dict)


# ===========================================================================
# Step 1: Prepare retrieval queries
# ===========================================================================

def prepare_queries(plan: Dict[str, Any]) -> List[Tuple[str, str]]:
    search_queries = plan.get("search_queries", [])
    if not search_queries:
        eq = plan.get("expanded_query", "")
        if eq:
            return [("q1", eq)]
        return [("q1", plan.get("query", ""))]
    return [(f"q{i}", q) for i, q in enumerate(search_queries, start=1)]


# ===========================================================================
# Step 2: UMLS / MeSH handling
# ===========================================================================

def enrich_query_with_umls(query_text: str, plan: Dict[str, Any]) -> str:
    umls_entries = plan.get("umls", [])
    if not umls_entries:
        return query_text
    extras = []
    q_lower = query_text.lower()
    for entry in umls_entries:
        if not entry.get("found"):
            continue
        pname = entry.get("preferred_name", "")
        if pname and pname.lower() not in q_lower:
            extras.append(pname)
        for syn in entry.get("synonyms", []):
            if syn and syn.lower() not in q_lower:
                extras.append(syn)
    if extras:
        return query_text + " " + " ".join(extras[:5])
    return query_text


# ===========================================================================
# Step 3: Multi-query BM25 + Dense retrieval
# ===========================================================================

def run_multi_query_retrieval(
    queries: List[Tuple[str, str]],
    plan: Dict[str, Any],
    bm25_index: Any,
    dense_index: Any,
    query_encoder: Any,
    bm25_depth: int = 100,
    dense_depth: int = 100,
) -> Dict[str, Candidate]:
    candidates: Dict[str, Candidate] = {}
    query_texts = [q for _, q in queries]
    query_ids = [qid for qid, _ in queries]

    query_vectors = None
    if dense_index is not None and query_encoder is not None:
        query_vectors = query_encoder.encode(query_texts)

    enriched_queries = [
        (qid, enrich_query_with_umls(qt, plan))
        for qid, qt in queries
    ]

    total_events = 0
    for idx, (qid, enriched_q) in enumerate(enriched_queries):
        try:
            bm25_hits = bm25_index.search_single(enriched_q, bm25_depth)
            for hit in bm25_hits:
                cid = hit.chunk_id
                if cid not in candidates:
                    candidates[cid] = Candidate(chunk_id=cid)
                candidates[cid].hits.append(RetrievalEvent(
                    query_id=qid, branch="bm25", rank=hit.rank, score=hit.score,
                ))
                total_events += 1
        except Exception as exc:
            print(f"  WARNING: BM25 failed for {qid}: {exc}", file=sys.stderr)

        if query_vectors is not None and dense_index is not None:
            try:
                qvec = query_vectors[idx]
                dense_hits = dense_index.search_single(qvec, dense_depth)
                for hit in dense_hits:
                    cid = hit.chunk_id
                    if cid not in candidates:
                        candidates[cid] = Candidate(chunk_id=cid)
                    candidates[cid].hits.append(RetrievalEvent(
                        query_id=qid, branch="dense", rank=hit.rank, score=hit.score,
                    ))
                    total_events += 1
            except Exception as exc:
                print(f"  WARNING: Dense failed for {qid}: {exc}", file=sys.stderr)

    print(f"  Total retrieval events: {total_events}")
    print(f"  Unique candidates: {len(candidates)}")
    return candidates


# ===========================================================================
# Step 5: Multi-query RRF fusion
# ===========================================================================

def multi_query_rrf_fusion(
    candidates: Dict[str, Candidate],
    rrf_k: float = 60.0,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    query_weight: float = 1.0,
) -> List[Candidate]:
    for cid, cand in candidates.items():
        best_by_pair: Dict[Tuple[str, str], float] = {}
        for event in cand.hits:
            key = (event.query_id, event.branch)
            w = bm25_weight if event.branch == "bm25" else dense_weight
            w *= query_weight
            s = w / (rrf_k + event.rank)
            if key not in best_by_pair or s > best_by_pair[key]:
                best_by_pair[key] = s
        cand.rrf_score = sum(best_by_pair.values())
        cand.matched_queries = sorted(set(h.query_id for h in cand.hits))
        cand.query_coverage = len(cand.matched_queries)

    ranked = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
    for i, c in enumerate(ranked, start=1):
        c.original_rrf_rank = i
    return ranked


# ===========================================================================
# Step 7: Resolve metadata
# ===========================================================================

def resolve_metadata(
    candidates: List[Candidate],
    corpus_index: Any,
    max_pool: int = 500,
) -> List[Candidate]:
    top = candidates[:max_pool]
    chunk_ids = [c.chunk_id for c in top]
    resolved = corpus_index.resolve(chunk_ids, include_text=True)
    for cand in top:
        meta = resolved.get(cand.chunk_id, {})
        cand.document_id = meta.get("document_id")
        cand.chunk_type = meta.get("chunk_type")
        cand.breadcrumb = meta.get("breadcrumb")
        cand.text = meta.get("text")
    return top


# ===========================================================================
# Step 8: MedCPT Cross-Encoder reranking
# ===========================================================================

def rerank_candidates(
    original_query: str,
    candidates: List[Candidate],
    cross_encoder: Any,
    top_k: int = 100,
) -> List[Candidate]:
    if not candidates:
        return []

    from medrag.models import Candidate as MedragCandidate

    mcandidates = []
    for c in candidates:
        mc = MedragCandidate(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            original_score=c.rrf_score,
            original_rank=c.original_rrf_rank,
            chunk_type=c.chunk_type,
            breadcrumb=c.breadcrumb,
            text=c.text,
            retrieval_method="hybrid",
        )
        mcandidates.append(mc)

    reranked = cross_encoder.rerank(original_query, mcandidates, top_k=top_k)

    cand_lookup = {c.chunk_id: c for c in candidates}
    result = []
    for rc in reranked:
        cand = cand_lookup.get(rc.chunk_id)
        if cand is None:
            continue
        cand.reranker_score = rc.reranker_score
        result.append(cand)
    return result


# ===========================================================================
# Step 9: Evidence requirement matching (FIXED)
# ===========================================================================

# ===========================================================================
# Table expansion: recursively retrieve all rows of any retrieved table
# ===========================================================================

def _extract_table_prefix(chunk_id: str) -> Optional[str]:
    """
    Extract the table prefix from a chunk ID.

    Examples:
        PMC10327125_T1_summary    -> PMC10327125_T1
        PMC10327125_T1_row_0      -> PMC10327125_T1
        PMC11772717_clc70089-tbl-0001_summary -> PMC11772717_clc70089-tbl-0001
        PMC11571068_hoi240076t1_summary -> PMC11571068_hoi240076t1
        PMC10327125_0             -> None (not a table)
    """
    import re
    # Match: {prefix}_{summary|row_N|footnotes} where prefix can be any table ID
    m = re.match(r'^(.+)_(?:summary|row_\d+|footnotes)$', chunk_id)
    if m:
        return m.group(1)
    return None


def _get_table_prefixes_from_text(text: str) -> List[str]:
    """
    Extract table prefixes from text content (e.g., 'Table 1', 'Table 2').
    This handles cases where a passage references tables.
    """
    import re
    prefixes = []
    # Match "Table 1", "Table 2", etc.
    for m in re.finditer(r'[Tt]able\s+(\d+)', text):
        prefixes.append(f"T{m.group(1)}")
    return prefixes


def expand_table_chunks(
    reranked: List[Candidate],
    corpus_index: Any,
    candidates_dict: Dict[str, Any],
    dense_index: Any = None,
    max_table_rows: int = 50,
) -> List[Candidate]:
    """
    When a table chunk is retrieved, recursively find all rows/footnotes
    of that table and add them to the candidate pool.
    
    This ensures complete table context is available for reranking.
    """
    import re
    
    # Build chunk_id -> Candidate lookup from the full candidate pool
    all_chunk_lookup: Dict[str, Candidate] = {}
    for cid, cand in candidates_dict.items():
        all_chunk_lookup[cid] = cand
    
    # Also add reranked candidates
    for c in reranked:
        all_chunk_lookup[c.chunk_id] = c
    
    # Find all table prefixes in the reranked results
    table_prefixes: set = set()
    for c in reranked:
        prefix = _extract_table_prefix(c.chunk_id)
        if prefix:
            table_prefixes.add(prefix)
    
    if not table_prefixes:
        return reranked
    
    print(f"  Found {len(table_prefixes)} tables in reranked results: {list(table_prefixes)[:5]}...")
    
    # Find all chunks belonging to these tables
    expanded_ids: set = set()
    for prefix in table_prefixes:
        # Search for all chunks with this prefix
        pattern = re.compile(f'^{re.escape(prefix)}_(?:summary|row_\\d+|footnotes)$')
        for cid in all_chunk_lookup:
            if pattern.match(cid):
                expanded_ids.add(cid)
                if len(expanded_ids) >= max_table_rows:
                    break
    
    # Also search the corpus for table chunks we haven't seen yet
    if len(expanded_ids) < 100:  # Only if not too many already
        # Get all chunk IDs from the corpus
        corpus_ids = corpus_index._df['id'].tolist() if corpus_index._df is not None else []
        for prefix in table_prefixes:
            pattern = re.compile(f'^{re.escape(prefix)}_(?:summary|row_\\d+|footnotes)$')
            for cid in corpus_ids:
                if pattern.match(cid) and cid not in expanded_ids:
                    expanded_ids.add(cid)
                    if len(expanded_ids) >= max_table_rows:
                        break
    
    # Remove IDs already in reranked
    existing_ids = {c.chunk_id for c in reranked}
    new_ids = expanded_ids - existing_ids
    
    if not new_ids:
        print(f"  No new table chunks to add")
        return reranked
    
    print(f"  Adding {len(new_ids)} table chunks for complete table context")
    
    # Resolve metadata for new chunks
    new_ids_list = list(new_ids)[:max_table_rows]
    resolved = corpus_index.resolve(new_ids_list, include_text=True)
    
    # Create candidates for new chunks
    new_candidates = []
    for cid in new_ids_list:
        meta = resolved.get(cid, {})
        cand = Candidate(
            chunk_id=cid,
            document_id=meta.get("document_id"),
            chunk_type=meta.get("chunk_type"),
            breadcrumb=meta.get("breadcrumb"),
            text=meta.get("text"),
        )
        # Set a base score from existing candidates of same document
        doc_candidates = [c for c in reranked if c.document_id == cand.document_id]
        if doc_candidates:
            cand.reranker_score = max(c.reranker_score for c in doc_candidates) * 0.9
        new_candidates.append(cand)
    
    # Add to reranked list
    expanded = list(reranked) + new_candidates
    
    # Re-sort by reranker_score
    expanded.sort(key=lambda c: c.reranker_score, reverse=True)
    
    print(f"  Expanded candidate pool: {len(reranked)} → {len(expanded)}")
    
    return expanded


def match_evidence_requirements(
    candidates: List[Candidate],
    evidence_requirements: List[str],
    query_encoder: Any,
    dense_index: Any,
    covered_threshold: float = REQUIREMENT_THRESHOLD,
) -> None:
    """
    For each evidence requirement, compute per-candidate cosine similarity,
    then normalize per-requirement (min-max across the pool) so that
    coverage is based on RELATIVE strength, not absolute similarity.

    Modifies candidates in-place:
        requirement_scores: raw cosine similarities
        covered_requirements: requirement indices above normalized threshold
        coverage_count, coverage_fraction
    """
    if not evidence_requirements or not candidates:
        return
    if query_encoder is None or dense_index is None:
        print("  WARNING: No query encoder or dense index — skipping evidence matching")
        return

    # Encode all requirements
    req_vectors = query_encoder.encode(evidence_requirements)

    n_reqs = len(evidence_requirements)
    n_cands = len(candidates)

    # Detect pgvector vs FAISS backend
    is_pgvector = hasattr(dense_index, 'store') and hasattr(dense_index.store, 'get_embedding')

    # Compute raw cosine similarities and cache embeddings
    raw_scores: List[Dict[int, float]] = []
    for cand in candidates:
        try:
            if is_pgvector:
                emb = dense_index.reconstruct(cand.chunk_id)
            else:
                # FAISS path: build position lookup on first use
                if not hasattr(match_evidence_requirements, '_id_to_pos'):
                    match_evidence_requirements._id_to_pos = {
                        cid: i for i, cid in enumerate(dense_index.chunk_ids)
                    }
                id_to_pos = match_evidence_requirements._id_to_pos
                pos = id_to_pos.get(cand.chunk_id)
                if pos is None:
                    emb = None
                else:
                    emb = dense_index.index.reconstruct(pos).astype(np.float32)
            if emb is None:
                raw_scores.append({i: 0.0 for i in range(n_reqs)})
                cand.embedding = None
                continue
            cand.embedding = emb
        except Exception:
            cand.embedding = None
            raw_scores.append({i: 0.0 for i in range(n_reqs)})
            continue

        scores = {}
        for req_idx in range(n_reqs):
            scores[req_idx] = float(np.dot(req_vectors[req_idx], cand.embedding))
        raw_scores.append(scores)

    # Per-requirement min-max normalization across the candidate pool
    for req_idx in range(n_reqs):
        col = np.array([rs[req_idx] for rs in raw_scores])
        lo, hi = float(col.min()), float(col.max())
        rng = hi - lo if hi > lo else 1.0
        for cand_idx in range(n_cands):
            raw = raw_scores[cand_idx][req_idx]
            norm = (raw - lo) / rng
            candidates[cand_idx].requirement_scores[req_idx] = round(norm, 6)

    # Determine covered requirements using NORMALIZED threshold
    for cand_idx in range(n_cands):
        cand = candidates[cand_idx]
        covered = [
            i for i in range(n_reqs)
            if cand.requirement_scores.get(i, 0.0) >= covered_threshold
        ]
        cand.covered_requirements = covered
        cand.coverage_count = len(covered)
        cand.coverage_fraction = len(covered) / n_reqs if n_reqs else 0.0

    # Diagnostic: best candidates per requirement
    print()
    print("  Per-requirement best candidates (top 5):")
    for req_idx in range(n_reqs):
        top5 = sorted(
            ((cand.chunk_id, cand.requirement_scores.get(req_idx, 0.0)) for cand in candidates),
            key=lambda x: x[1], reverse=True,
        )[:5]
        scores_str = ", ".join(f"{cid}({s:.3f})" for cid, s in top5)
        n_above = sum(1 for c in candidates if req_idx in c.covered_requirements)
        print(f"    R{req_idx+1}: {n_above:>3} covered | top5: {scores_str}")


# ===========================================================================
# Step 10: Intent matching
# ===========================================================================

def match_intent(
    candidates: List[Candidate],
    intent: Dict[str, Any],
    query_encoder: Any,
    dense_index: Any,
) -> None:
    if not intent or not candidates:
        return
    if query_encoder is None or dense_index is None:
        return

    parts = []
    for key in ("task", "target", "condition", "conditions"):
        val = intent.get(key, "")
        if isinstance(val, list):
            parts.extend(str(v).strip() for v in val if v)
        elif val:
            parts.append(str(val).strip())
    intent_text = "; ".join(parts) if parts else "medical information"

    intent_vec = query_encoder.encode([intent_text])[0]

    is_pgvector = hasattr(dense_index, 'store') and hasattr(dense_index.store, 'get_embedding')

    for cand in candidates:
        if cand.embedding is not None:
            cand.intent_score = float(np.dot(intent_vec, cand.embedding))
        else:
            try:
                if is_pgvector:
                    emb = dense_index.reconstruct(cand.chunk_id)
                else:
                    if not hasattr(match_intent, '_id_to_pos'):
                        match_intent._id_to_pos = {
                            cid: i for i, cid in enumerate(dense_index.chunk_ids)
                        }
                    pos = match_intent._id_to_pos.get(cand.chunk_id)
                    emb = dense_index.index.reconstruct(pos).astype(np.float32) if pos is not None else None
                cand.intent_score = float(np.dot(intent_vec, emb)) if emb is not None else 0.0
            except Exception:
                cand.intent_score = 0.0


# ===========================================================================
# Steps 11-14: Greedy marginal-coverage selection (REWRITTEN)
# ===========================================================================

def _doc_penalty(doc_count: int) -> float:
    """Document redundancy penalty based on selected passages from same doc."""
    if doc_count >= 3:
        return DOC_PENALTY_TIERS[3]
    return DOC_PENALTY_TIERS.get(doc_count, 0.0)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors (just dot product)."""
    return float(np.dot(a, b))


def greedy_marginal_selection(
    candidates: List[Candidate],
    evidence_requirements: List[str],
    intent: Dict[str, Any],
    top_k: int = 20,
    weights: Optional[Dict[str, float]] = None,
    new_req_bonus: float = NEW_REQUIREMENT_BONUS,
    query_cov_bonus: float = QUERY_COVERAGE_BONUS,
    diversity_weight: float = DYNAMIC_DIVERSITY_WEIGHT,
    redundancy_weight: float = REDUNDANCY_PENALTY,
) -> List[Candidate]:
    """
    Greedy marginal-coverage selection.

    Each step evaluates every remaining candidate's MARGINAL VALUE
    relative to the already-selected set:

        marginal = base_score
                   + new_req_bonus * new_requirement_gain
                   + query_cov_bonus * query_coverage_norm
                   + diversity_weight * dynamic_diversity
                   - redundancy_weight * redundancy_penalty

    where dynamic_diversity and redundancy_penalty are computed against
    the current selected set at each step.
    """
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)

    if not candidates:
        return []

    n_reqs = len(evidence_requirements) if evidence_requirements else 1

    # ── Normalize static scores ────────────────────────────────────
    medcpt_scores = np.array([c.reranker_score for c in candidates])
    m_lo, m_hi = float(medcpt_scores.min()), float(medcpt_scores.max())
    m_rng = m_hi - m_lo if m_hi > m_lo else 1.0

    intent_scores = np.array([c.intent_score for c in candidates])
    i_lo, i_hi = float(intent_scores.min()), float(intent_scores.max())
    i_rng = i_hi - i_lo if i_hi > i_lo else 1.0

    query_cov_vals = np.array([float(c.query_coverage) for c in candidates])
    qc_hi = float(query_cov_vals.max()) if len(query_cov_vals) else 1.0
    qc_rng = qc_hi if qc_hi > 0 else 1.0

    # Static base score for each candidate (computed ONCE)
    base_scores = np.zeros(len(candidates))
    for idx, c in enumerate(candidates):
        norm_medcpt = (float(medcpt_scores[idx]) - m_lo) / m_rng
        norm_intent = (float(intent_scores[idx]) - i_lo) / i_rng
        norm_qcov = float(query_cov_vals[idx]) / qc_rng

        base_scores[idx] = (
            weights["medcpt"] * norm_medcpt
            + weights["evidence"] * c.coverage_fraction
            + weights["query_coverage"] * norm_qcov
            + weights["intent"] * norm_intent
            + weights["diversity"] * 1.0  # initial diversity = 1.0 (no selections yet)
        )

        # Stash normalized values for output
        c._norm_medcpt = norm_medcpt
        c._norm_intent = norm_intent
        c._norm_querycov = norm_qcov

    # ── Greedy iterative selection ─────────────────────────────────
    selected: List[Candidate] = []
    selected_indices: List[int] = []
    selected_embeddings: List[np.ndarray] = []
    selected_doc_counts: Dict[str, int] = defaultdict(int)
    covered_reqs: set = set()

    remaining = set(range(len(candidates)))

    for step in range(min(top_k, len(candidates))):
        best_idx = -1
        best_marginal = -1e9
        best_diversity = 0.0
        best_redundancy = 0.0
        best_new_reqs: set = set()

        for cidx in remaining:
            c = candidates[cidx]

            # Base score (static)
            bs = base_scores[cidx]

            # New requirement gain
            new_reqs = set(c.covered_requirements) - covered_reqs
            new_req_gain = len(new_reqs) / n_reqs if n_reqs else 0.0

            # Query coverage bonus
            qcov_b = query_cov_bonus * c._norm_querycov

            # ── Dynamic diversity ──────────────────────────────────
            if not selected:
                # First selection: maximum diversity
                diversity = 1.0
                redundancy = 0.0
            else:
                # Semantic redundancy: max cosine sim to selected
                sem_red = 0.0
                if c.embedding is not None and selected_embeddings:
                    sims = [
                        _cosine_sim(c.embedding, sel_emb)
                        for sel_emb in selected_embeddings
                        if sel_emb is not None
                    ]
                    if sims:
                        sem_red = max(sims)

                # Document redundancy
                doc_count = selected_doc_counts.get(c.document_id or "", 0)
                doc_red = _doc_penalty(doc_count)

                redundancy = (
                    SEMANTIC_REDUNDANCY_WEIGHT * sem_red
                    + DOCUMENT_REDUNDANCY_WEIGHT * doc_red
                )

                diversity = 1.0 - redundancy

            # ── Marginal value ─────────────────────────────────────
            marginal = (
                bs
                + new_req_bonus * new_req_gain
                + qcov_b
                + diversity_weight * diversity
                - redundancy_weight * redundancy
            )

            if marginal > best_marginal:
                best_marginal = marginal
                best_idx = cidx
                best_diversity = diversity
                best_redundancy = redundancy
                best_new_reqs = new_reqs

        if best_idx < 0:
            break

        chosen = candidates[best_idx]
        remaining.remove(best_idx)

        # Update state
        covered_reqs.update(chosen.covered_requirements)
        doc_id = chosen.document_id or ""
        selected_doc_counts[doc_id] += 1

        if chosen.embedding is not None:
            selected_embeddings.append(chosen.embedding)

        chosen.selection_rank = len(selected) + 1
        chosen.selection_reason = {
            "base_score": round(float(base_scores[best_idx]), 4),
            "relevance": round(chosen._norm_medcpt, 4),
            "coverage": round(len(best_new_reqs) / n_reqs if n_reqs else 0, 4),
            "new_requirements": sorted(best_new_reqs),
            "intent": round(chosen._norm_intent, 4),
            "query_coverage": round(chosen._norm_querycov, 4),
            "diversity": round(best_diversity, 4),
            "redundancy": round(best_redundancy, 4),
            "marginal_value": round(best_marginal, 4),
        }

        selected.append(chosen)
        selected_indices.append(best_idx)

    return selected


# ===========================================================================
# Step 13: Repair pass for uncovered requirements
# ===========================================================================

def repair_uncovered_requirements(
    selected: List[Candidate],
    candidates: List[Candidate],
    evidence_requirements: List[str],
    covered_threshold: float = REQUIREMENT_THRESHOLD,
) -> List[Candidate]:
    """
    After greedy selection, check coverage. If any requirement is uncovered,
    try to replace the weakest selected candidate that doesn't uniquely cover
    another requirement with the best unselected candidate that covers the
    missing requirement.
    """
    if not evidence_requirements:
        return selected

    n_reqs = len(evidence_requirements)

    # Build selected chunk_id set
    selected_ids = {c.chunk_id for c in selected}

    # Find uncovered requirements
    covered = set()
    for c in selected:
        covered.update(c.covered_requirements)
    uncovered = [i for i in range(n_reqs) if i not in covered]

    if not uncovered:
        return selected

    print(f"  Repair pass: {len(uncovered)} uncovered requirements: {uncovered}")

    # Find candidates that cover each uncovered requirement
    unselected = [c for c in candidates if c.chunk_id not in selected_ids]

    for req_idx in uncovered:
        # Best unselected candidate covering this requirement
        covering = [
            c for c in unselected
            if req_idx in c.covered_requirements
        ]
        if not covering:
            print(f"    R{req_idx+1}: No candidate covers this requirement")
            continue

        best_replacement = max(covering, key=lambda c: c.reranker_score)

        # Find weakest selected candidate that doesn't uniquely cover
        # another requirement
        replaceable = []
        for i, sc in enumerate(selected):
            # Requirements uniquely covered by this candidate
            other_covered = set()
            for j, oc in enumerate(selected):
                if i != j:
                    other_covered.update(oc.covered_requirements)
            unique_to_this = set(sc.covered_requirements) - other_covered
            if not unique_to_this:
                replaceable.append((i, sc))

        if not replaceable:
            # If all candidates uniquely cover something, replace the weakest
            replaceable = [(len(selected) - 1, selected[-1])]

        # Replace the weakest replaceable candidate
        weakest_idx, weakest = min(replaceable, key=lambda x: x[1].reranker_score)

        print(f"    R{req_idx+1}: Replacing {weakest.chunk_id} "
              f"(rerank={weakest.reranker_score:.3f}) with "
              f"{best_replacement.chunk_id} (rerank={best_replacement.reranker_score:.3f})")

        best_replacement.selection_rank = weakest.selection_rank
        best_replacement.selection_reason = {
            **best_replacement.selection_reason,
            "repair_replacement": True,
            "replaced_chunk": weakest.chunk_id,
        }
        selected[weakest_idx] = best_replacement

        # Update covered set
        covered.update(best_replacement.covered_requirements)

    # Re-sort by selection_rank
    selected.sort(key=lambda c: c.selection_rank)
    return selected


# ===========================================================================
# Post-selection: expand selected tables to include all rows
# ===========================================================================

def _expand_selected_tables(
    selected: List[Candidate],
    corpus_index: Any,
    candidates_dict: Dict[str, Any],
    evidence_requirements: List[str],
) -> List[Candidate]:
    """
    After greedy selection, if any table chunk was selected, ensure all
    rows and footnotes of that table are included in the final set.

    This replaces the final top_k with complete tables rather than
    individual rows.
    """
    import re

    # Find table prefixes in selected results
    selected_table_prefixes: set = set()
    for c in selected:
        prefix = _extract_table_prefix(c.chunk_id)
        if prefix:
            selected_table_prefixes.add(prefix)

    if not selected_table_prefixes:
        print("  No tables in selected results — skipping table completion")
        return selected

    print(f"  Found {len(selected_table_prefixes)} tables in selected results")

    # Find ALL chunks for these tables from the corpus
    corpus_ids = corpus_index._df['id'].tolist() if corpus_index._df is not None else []
    all_table_chunks: Dict[str, List[str]] = defaultdict(list)

    for prefix in selected_table_prefixes:
        pat = re.compile(r'^' + re.escape(prefix) + r'_(?:summary|row_\d+|footnotes)$')
        for cid in corpus_ids:
            if pat.match(cid):
                all_table_chunks[prefix].append(cid)

    # Count total chunks to add
    selected_ids = {c.chunk_id for c in selected}
    new_chunks_needed = []
    for prefix, chunk_ids in all_table_chunks.items():
        for cid in chunk_ids:
            if cid not in selected_ids:
                new_chunks_needed.append(cid)

    if not new_chunks_needed:
        print("  All table rows already included")
        return selected

    print(f"  Adding {len(new_chunks_needed)} table chunks for complete tables")

    # Resolve metadata for new chunks
    resolved = corpus_index.resolve(new_chunks_needed, include_text=True)

    # Create candidates for new chunks
    new_candidates = []
    for cid in new_chunks_needed:
        meta = resolved.get(cid, {})
        cand = Candidate(
            chunk_id=cid,
            document_id=meta.get("document_id"),
            chunk_type=meta.get("chunk_type"),
            breadcrumb=meta.get("breadcrumb"),
            text=meta.get("text"),
        )
        # Score based on the original table's reranker score
        prefix = _extract_table_prefix(cid)
        original_table_chunks = [c for c in selected if _extract_table_prefix(c.chunk_id) == prefix]
        if original_table_chunks:
            cand.reranker_score = max(c.reranker_score for c in original_table_chunks) * 0.85
        new_candidates.append(cand)

    # Combine selected + new table chunks
    expanded = list(selected) + new_candidates

    # Re-sort by reranker_score
    expanded.sort(key=lambda c: c.reranker_score, reverse=True)

    print(f"  Expanded final set: {len(selected)} → {len(expanded)}")

    return expanded


# ===========================================================================
# Step 15: Coverage completeness check
# ===========================================================================

def check_coverage(
    selected: List[Candidate],
    evidence_requirements: List[str],
) -> Dict[int, bool]:
    n_reqs = len(evidence_requirements)
    covered = set()
    for c in selected:
        covered.update(c.covered_requirements)
    return {i: (i in covered) for i in range(n_reqs)}


# ===========================================================================
# Step 16: Build final result (FIXED)
# ===========================================================================

def build_result(
    selected: List[Candidate],
    plan: Dict[str, Any],
    queries: List[Tuple[str, str]],
    evidence_requirements: List[str],
    total_candidates: int,
    reranked_count: int,
    coverage_map: Dict[int, bool],
) -> Dict[str, Any]:
    # Per-query coverage in final set
    query_coverage = {}
    for qid, _ in queries:
        count = sum(1 for c in selected if qid in c.matched_queries)
        query_coverage[qid] = count

    # Unique documents
    doc_ids = set(c.document_id for c in selected if c.document_id)

    # Per-document counts
    selected_docs: Dict[str, int] = defaultdict(int)
    for c in selected:
        if c.document_id:
            selected_docs[c.document_id] += 1

    # Build per-candidate output
    results = []
    for c in selected:
        entry = {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "chunk_type": c.chunk_type,
            "breadcrumb": c.breadcrumb,
            "reranker_score": round(c.reranker_score, 6),
            "original_rrf_score": round(c.rrf_score, 6),
            "requirement_scores": {
                str(k): round(v, 4) for k, v in c.requirement_scores.items()
            },
            "covered_requirements": c.covered_requirements,
            "coverage_count": c.coverage_count,
            "coverage_fraction": round(c.coverage_fraction, 4),
            "matched_queries": c.matched_queries,
            "query_coverage": c.query_coverage,
            "selection_rank": c.selection_rank,
            "selection_reason": c.selection_reason,
            "text_preview": (c.text or "")[:500],
        }
        results.append(entry)

    # Evidence coverage
    evidence_coverage = {}
    for i in range(len(evidence_requirements)):
        evidence_coverage[f"requirement_{i+1}"] = coverage_map.get(i, False)

    # Coverage distribution stats
    coverage_counts = [c.coverage_count for c in selected]
    zero_cov = sum(1 for x in coverage_counts if x == 0)
    one_cov = sum(1 for x in coverage_counts if x == 1)
    multi_cov = sum(1 for x in coverage_counts if x >= 2)

    # Per-requirement coverage counts
    req_coverage_counts = {}
    for i in range(len(evidence_requirements)):
        n = sum(1 for c in selected if i in c.covered_requirements)
        req_coverage_counts[f"requirement_{i+1}"] = n

    diagnostics = {
        "candidate_count": total_candidates,
        "reranked_count": reranked_count,
        "final_count": len(selected),
        "document_count": len(doc_ids),
        "query_coverage": query_coverage,
        "evidence_coverage": evidence_coverage,
        "all_evidence_covered": all(coverage_map.values()),
        "selected_documents": dict(selected_docs),
        "coverage_distribution": {
            "mean_coverage_fraction": round(
                sum(c.coverage_fraction for c in selected) / len(selected) if selected else 0, 4
            ),
            "zero_requirements": zero_cov,
            "one_requirement": one_cov,
            "multiple_requirements": multi_cov,
        },
        "per_requirement_coverage": req_coverage_counts,
    }

    return {
        "query": plan.get("query", ""),
        "results": results,
        "diagnostics": diagnostics,
    }


# ===========================================================================
# Output: expanded evidence markdown
# ===========================================================================

def _write_expanded_evidence_md(
    path: Path,
    plan: Dict[str, Any],
    expanded_evidence: List[Dict[str, Any]],
    selected: List[Candidate],
    evidence_requirements: List[str],
    timings: Dict[str, float],
) -> None:
    """Write a single markdown file containing all expanded evidence text, keyed by PMC ID."""
    lines: List[str] = []
    query = plan.get("query", "")
    lines.append("# Expanded Evidence")
    lines.append("")
    lines.append(f"**Query:** {query}")
    lines.append("")
    lines.append(f"**Evidence Requirements ({len(evidence_requirements)}):**")
    for i, req in enumerate(evidence_requirements, 1):
        lines.append(f"  {i}. {req}")
    lines.append("")
    lines.append(f"**Total expanded blocks:** {len(expanded_evidence)}")
    lines.append(f"**Total selected passages:** {len(selected)}")
    lines.append(f"**Pipeline time:** {timings.get('total_ms', 0):.0f} ms")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 1: Expanded evidence (full text) ──────────────────
    lines.append("## Expanded Evidence (Full Text)")
    lines.append("")

    for i, ev in enumerate(expanded_evidence, 1):
        doc_id = ev.get("document_id", "unknown")
        ctx_type = ev.get("context_type", "")
        section = " > ".join(ev.get("section_path", []))
        token_count = ev.get("token_count", 0)
        truncated = ev.get("truncated", False)
        source_ids = ev.get("source_chunk_ids", [])
        covered = ev.get("covered_requirements", [])

        lines.append(f"### {i}. {doc_id}")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Document | `{doc_id}` |")
        lines.append(f"| Context type | {ctx_type} |")
        lines.append(f"| Section | {section} |")
        lines.append(f"| Source chunks | {', '.join(f'`{s}`' for s in source_ids)} |")
        lines.append(f"| Tokens | {token_count:,}{' (truncated)' if truncated else ''} |")
        lines.append(f"| Covered requirements | {', '.join(f'R{r+1}' for r in covered) if covered else 'none'} |")
        lines.append("")

        # Full text block
        text = ev.get("text", "")
        lines.append("**Full text:**")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Section 2: Selected passages (compact) ────────────────────
    lines.append("## Selected Passages (Final Ranking)")
    lines.append("")
    lines.append("| # | Chunk ID | Document | Type | Rerank | RRF | Coverage | Requirements |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for c in selected:
        cr = c.covered_requirements
        cr_str = ", ".join(f"R{r+1}" for r in cr) if cr else "—"
        lines.append(
            f"| {c.selection_rank} "
            f"| `{c.chunk_id}` "
            f"| `{c.document_id}` "
            f"| {c.chunk_type or '—'} "
            f"| {c.reranker_score:.4f} "
            f"| {c.rrf_score:.4f} "
            f"| {c.coverage_count}/{len(evidence_requirements)} "
            f"| {cr_str} |"
        )
    lines.append("")

    # ── Section 3: Full text of each selected passage ─────────────
    lines.append("## Selected Passages (Full Text)")
    lines.append("")

    for c in selected:
        cr = c.covered_requirements
        cr_str = ", ".join(f"R{r+1}" for r in cr) if cr else "none"
        lines.append(f"### #{c.selection_rank} — `{c.chunk_id}` (`{c.document_id}`)")
        lines.append("")
        lines.append(f"- **Chunk type:** {c.chunk_type or '—'}")
        lines.append(f"- **Breadcrumb:** {' > '.join(c.breadcrumb) if c.breadcrumb else '—'}")
        lines.append(f"- **Reranker score:** {c.reranker_score:.4f}")
        lines.append(f"- **RRF score:** {c.rrf_score:.4f}")
        lines.append(f"- **Coverage:** {c.coverage_count}/{len(evidence_requirements)} requirements")
        lines.append(f"- **Covered requirements:** {cr_str}")
        lines.append(f"- **Matched queries:** {', '.join(c.matched_queries) if c.matched_queries else '—'}")
        lines.append("")
        lines.append("**Text:**")
        lines.append("")
        lines.append(c.text or "*(no text)*")
        lines.append("")
        lines.append("---")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_orchestration(
    plan: Dict[str, Any],
    index_dir: Optional[Path] = None,
    bm25_depth: int = 100,
    dense_depth: int = 100,
    candidate_pool: int = 500,
    rerank_top_k: int = 100,
    final_top_k: int = 20,
    rrf_k: float = 60.0,
    req_threshold: float = REQUIREMENT_THRESHOLD,
    output_path: Optional[Path] = None,
    enable_context_expansion: bool = True,
    max_total_tokens: int = 12000,
    max_expanded_tokens_per_hit: int = 2000,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    trace = TraceLogger()
    from medrag.trace import set_trace
    set_trace(trace)

    index_dir = Path(index_dir or _INDEX_DIR)
    bm25_dir = index_dir / "bm25"
    corpus_path = index_dir / "corpus.parquet"

    original_query = plan.get("query", "")
    evidence_requirements = plan.get("evidence_requirements", [])
    intent = plan.get("intent", {})

    trace.log(
        "orchestration_start",
        params={
            "query": original_query,
            "n_evidence_requirements": len(evidence_requirements),
            "evidence_requirements": evidence_requirements,
            "intent": intent,
            "plan_keys": list(plan.keys()),
        },
    )

    print("=" * 70)
    print("MULTI-QUERY RETRIEVAL ORCHESTRATOR v2")
    print("=" * 70)
    print(f"Query: {original_query[:120]}...")

    # ── Step 1: Prepare queries ──────────────────────────────────
    queries = prepare_queries(plan)
    trace.log(
        "prepare_queries",
        params={"plan_query": original_query, "plan_search_queries": plan.get("search_queries", [])},
        result={"n_queries": len(queries), "queries": {qid: qt for qid, qt in queries}},
    )
    print(f"\nStep 1: Prepared {len(queries)} search queries:")
    for qid, qt in queries:
        print(f"  {qid}: {qt[:100]}{'...' if len(qt) > 100 else ''}")

    # ── Load indexes ─────────────────────────────────────────────
    print("\nLoading indexes...")
    t0 = time.perf_counter()

    from medrag.retrieval.sparse import BM25Index
    from medrag.retrieval.corpus import CorpusIndex, CORPUS_FILENAME

    trace.log("load_bm25", params={"path": str(bm25_dir)})
    bm25_index = BM25Index.load(bm25_dir)
    trace.log("load_bm25_done", result={"n_docs": bm25_index.n_docs, "vocab_size": bm25_index.vocab_size})

    trace.log("load_corpus", params={"path": str(corpus_path)})
    corpus_index = CorpusIndex(corpus_path, load=True)
    trace.log("load_corpus_done", result={"n_rows": corpus_index._df.shape[0]})

    dense_index = None
    query_encoder = None
    cross_encoder = None

    # Try pgvector first, fall back to FAISS
    use_pgvector = os.environ.get("MEDRAG_BACKEND", "pgvector").lower() == "pgvector"

    if use_pgvector:
        try:
            from medrag.retrieval.dense_pgvector import PgDenseIndex
            from medrag.retrieval.query import MedCPTQueryEncoder
            from medrag.retrieval.reranker import CrossEncoderReranker

            trace.log("load_dense", params={"backend": "pgvector", "host": os.environ.get("PGHOST", "localhost")})
            print("  Loading dense index from pgvector...")
            dense_index = PgDenseIndex.from_env()
            trace.log("load_dense_done", result={"n_total": dense_index.n_total, "dimension": dense_index.dimension, "backend": "pgvector"})
            print(f"  Dense: {dense_index.n_total:,} vectors, dim={dense_index.dimension}")

            trace.log("load_query_encoder", detail="MedCPT-Query-Encoder")
            print("  Loading MedCPT Query Encoder...")
            query_encoder = MedCPTQueryEncoder()
            trace.log("load_query_encoder_done", result="ok")

            trace.log("load_cross_encoder", detail="MedCPT-Cross-Encoder, batch=32, max_len=512")
            print("  Loading MedCPT Cross-Encoder...")
            cross_encoder = CrossEncoderReranker(batch_size=32, max_length=512)
            trace.log("load_cross_encoder_done", result="ok")
        except Exception as exc:
            print(f"  pgvector not available ({exc}), trying FAISS...")
            dense_index = None

    if dense_index is None:
        dense_faiss_path = index_dir / "dense.faiss"
        if dense_faiss_path.exists():
            from medrag.retrieval.dense import DenseIndex
            from medrag.retrieval.query import MedCPTQueryEncoder
            from medrag.retrieval.reranker import CrossEncoderReranker

            trace.log("load_dense", detail=f"FAISS ~2.7 GB from {index_dir}")
            print("  Loading dense index (FAISS ~2.7 GB)...")
            dense_index = DenseIndex.load(index_dir)
            trace.log("load_dense_done", result=f"{dense_index.n_total:,} vectors, dim={dense_index.dimension}")
            print(f"  Dense: {dense_index.n_total:,} vectors, dim={dense_index.dimension}")

            if query_encoder is None:
                trace.log("load_query_encoder", detail="MedCPT-Query-Encoder")
                print("  Loading MedCPT Query Encoder...")
                query_encoder = MedCPTQueryEncoder()
                trace.log("load_query_encoder_done", result="ok")

            if cross_encoder is None:
                trace.log("load_cross_encoder", detail="MedCPT-Cross-Encoder, batch=32, max_len=512")
                print("  Loading MedCPT Cross-Encoder...")
                cross_encoder = CrossEncoderReranker(batch_size=32, max_length=512)
                trace.log("load_cross_encoder_done", result="ok")
        else:
            print("  Dense index not found — using BM25 only")
            trace.log("load_dense", detail="no dense index found — BM25 only", result="skipped")

    timings["load_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log("load_complete", duration_ms=timings["load_ms"], result=f"all indexes loaded")
    print(f"  Loaded in {timings['load_ms']:.0f} ms")

    # ── Steps 3-5: Multi-query retrieval + RRF ──────────────────
    print(f"\nSteps 3-5: Multi-query retrieval ({bm25_depth} BM25 + {dense_depth} Dense per query)")
    t0 = time.perf_counter()

    trace.log(
        "retrieval_start",
        params={
            "n_queries": len(queries),
            "bm25_depth": bm25_depth,
            "dense_depth": dense_depth,
            "queries": {qid: qt for qid, qt in queries},
        },
    )

    candidates_dict = run_multi_query_retrieval(
        queries, plan, bm25_index, dense_index, query_encoder,
        bm25_depth=bm25_depth, dense_depth=dense_depth,
    )

    # Count events by branch
    bm25_events = sum(1 for c in candidates_dict.values() for h in c.hits if h.branch == "bm25")
    dense_events = sum(1 for c in candidates_dict.values() for h in c.hits if h.branch == "dense")

    trace.log(
        "retrieval_done",
        result={
            "n_unique_candidates": len(candidates_dict),
            "n_bm25_events": bm25_events,
            "n_dense_events": dense_events,
            "top_rrf_candidates": [
                {"chunk_id": c.chunk_id, "rrf_score": round(c.rrf_score, 6), "n_hits": len(c.hits), "matched_queries": c.matched_queries}
                for c in sorted(candidates_dict.values(), key=lambda x: x.rrf_score, reverse=True)[:5]
            ],
        },
    )

    ranked = multi_query_rrf_fusion(candidates_dict, rrf_k=rrf_k)

    timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log(
        "rrf_fusion",
        params={
            "rrf_k": rrf_k,
            "n_input_candidates": len(candidates_dict),
            "fusion_method": "rrf",
        },
        result={
            "n_ranked": len(ranked),
            "top10": [{"chunk_id": c.chunk_id, "rrf_score": round(c.rrf_score, 6), "matched_queries": c.matched_queries} for c in ranked[:10]],
        },
        duration_ms=timings["retrieval_ms"],
    )
    print(f"  RRF fusion complete: {len(ranked)} unique candidates")
    print(f"  Top-5 RRF scores: {[round(c.rrf_score, 4) for c in ranked[:5]]}")

    # ── Step 6-7: Create pool + resolve metadata ─────────────────
    print(f"\nSteps 6-7: Resolving metadata for top-{candidate_pool} candidates")
    t0 = time.perf_counter()

    trace.log(
        "resolve_metadata",
        params={
            "candidate_pool": candidate_pool,
            "n_ranked": len(ranked),
            "top_chunk_ids_to_resolve": [c.chunk_id for c in ranked[:20]],
        },
    )
    pool = resolve_metadata(ranked, corpus_index, max_pool=candidate_pool)

    timings["resolve_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log(
        "resolve_metadata_done",
        result={
            "n_resolved": len(pool),
            "n_with_text": sum(1 for c in pool if c.text),
            "unique_documents": len(set(c.document_id for c in pool if c.document_id)),
            "chunk_types": dict(__import__("collections").Counter(c.chunk_type for c in pool if c.chunk_type)),
        },
        duration_ms=timings["resolve_ms"],
    )
    print(f"  Resolved {len(pool)} candidates")

    # ── Step 8: Cross-encoder reranking ──────────────────────────
    print(f"\nStep 8: MedCPT Cross-Encoder reranking (original query → top-{rerank_top_k})")
    t0 = time.perf_counter()

    if cross_encoder is not None:
        trace.log(
            "rerank_start",
            params={
                "query": original_query,
                "n_candidates": len(pool),
                "top_k": rerank_top_k,
                "model": cross_encoder.model_name,
                "candidate_preview": [{"chunk_id": c.chunk_id, "rrf_score": round(c.rrf_score, 6)} for c in pool[:10]],
            },
        )
        reranked = rerank_candidates(original_query, pool, cross_encoder, top_k=rerank_top_k)
        trace.log(
            "rerank_done",
            result={
                "n_reranked": len(reranked),
                "top10": [{"chunk_id": r.chunk_id, "reranker_score": round(r.reranker_score, 6), "original_rrf_score": round(r.rrf_score, 6)} for r in reranked[:10]],
            },
        )
        print(f"  Reranked {len(pool)} → {len(reranked)} candidates")
    else:
        print("  WARNING: No cross-encoder available, using RRF ranking")
        trace.log("rerank", params={"status": "skipped", "reason": "no cross-encoder"})
        reranked = pool[:rerank_top_k]

    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log("rerank_complete", duration_ms=timings["rerank_ms"])
    print(f"  Reranking completed in {timings['rerank_ms']:.0f} ms")

    # ── Step 8b: Expand table chunks ────────────────────────────
    print(f"\nStep 8b: Expanding table chunks for complete context")
    t0 = time.perf_counter()

    trace.log("table_expand", detail=f"expanding table chunks from {len(reranked)} reranked")
    reranked = expand_table_chunks(
        reranked, corpus_index, candidates_dict, dense_index,
    )

    timings["table_expand_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log("table_expand_done", result=f"{len(reranked)} after expansion", duration_ms=timings["table_expand_ms"])

    # ── Step 9: Evidence requirement matching (FIXED) ────────────
    print(f"\nStep 9: Evidence requirement matching ({len(evidence_requirements)} requirements, threshold={req_threshold})")
    t0 = time.perf_counter()

    trace.log(
        "evidence_match_start",
        params={
            "n_requirements": len(evidence_requirements),
            "requirements": evidence_requirements,
            "threshold": req_threshold,
            "n_candidates": len(reranked),
            "n_encoded": sum(1 for c in reranked if c.embedding is not None),
        },
    )
    match_evidence_requirements(
        reranked, evidence_requirements, query_encoder, dense_index,
        covered_threshold=req_threshold,
    )

    timings["evidence_ms"] = round((time.perf_counter() - t0) * 1000)
    # Summarize coverage
    coverage_summary = {}
    for i, req in enumerate(evidence_requirements):
        n_covered = sum(1 for c in reranked if i in c.covered_requirements)
        coverage_summary[f"R{i+1}"] = {"n_covered": n_covered, "requirement": req}
    trace.log(
        "evidence_match_done",
        result={
            "coverage_summary": coverage_summary,
            "mean_coverage_fraction": round(sum(c.coverage_fraction for c in reranked) / max(1, len(reranked)), 4),
        },
        duration_ms=timings["evidence_ms"],
    )

    # ── Step 10: Intent matching ─────────────────────────────────
    print(f"\nStep 10: Intent matching")
    t0 = time.perf_counter()

    trace.log(
        "intent_match_start",
        params={
            "intent": intent,
            "intent_representation": "; ".join(filter(None, [intent.get("intent", ""), intent.get("target", ""), intent.get("condition", "")])),
            "n_candidates": len(reranked),
        },
    )
    match_intent(reranked, intent, query_encoder, dense_index)

    timings["intent_ms"] = round((time.perf_counter() - t0) * 1000)
    intent_scores = [c.intent_score for c in reranked]
    trace.log(
        "intent_match_done",
        result={
            "min_intent_score": round(min(intent_scores), 6),
            "max_intent_score": round(max(intent_scores), 6),
            "mean_intent_score": round(sum(intent_scores) / max(1, len(intent_scores)), 6),
        },
        duration_ms=timings["intent_ms"],
    )
    print(f"  Intent score range: [{min(intent_scores):.4f}, {max(intent_scores):.4f}]")

    # ── Steps 13-14: Greedy marginal-coverage selection ──────────
    print(f"\nSteps 13-14: Greedy marginal-coverage selection → top-{final_top_k}")
    t0 = time.perf_counter()

    trace.log(
        "greedy_select_start",
        params={
            "top_k": final_top_k,
            "n_candidates": len(reranked),
            "n_requirements": len(evidence_requirements),
            "weights": DEFAULT_WEIGHTS,
        },
    )
    selected = greedy_marginal_selection(
        reranked,
        evidence_requirements,
        intent,
        top_k=final_top_k,
    )

    timings["select_ms"] = round((time.perf_counter() - t0) * 1000)
    n_docs = len(set(c.document_id for c in selected if c.document_id))
    trace.log(
        "greedy_select_done",
        result={
            "n_selected": len(selected),
            "n_documents": n_docs,
            "selected": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "reranker_score": round(c.reranker_score, 6),
                    "coverage_count": c.coverage_count,
                    "selection_rank": c.selection_rank,
                    "selection_reason": c.selection_reason,
                }
                for c in selected
            ],
        },
        duration_ms=timings["select_ms"],
    )
    print(f"  Selected {len(selected)} passages from {n_docs} documents")

    # ── Step 13: Repair pass ─────────────────────────────────────
    print(f"\nStep 13: Repair pass for uncovered requirements")
    t0 = time.perf_counter()

    trace.log(
        "repair_pass_start",
        params={
            "n_selected": len(selected),
            "n_requirements": len(evidence_requirements),
            "uncovered_before": [i for i in range(len(evidence_requirements)) if not any(i in c.covered_requirements for c in selected)],
        },
    )
    selected = repair_uncovered_requirements(
        selected, reranked, evidence_requirements,
    )

    timings["repair_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log(
        "repair_pass_done",
        result={
            "n_selected_after": len(selected),
            "replacements": [c.selection_reason for c in selected if c.selection_reason.get("repair_replacement")],
        },
        duration_ms=timings["repair_ms"],
    )

    # ── Step 14: Hierarchical context expansion ──────────────────
    expansion_result = None

    # Build chunk dicts with metadata for expansion
    chunk_dicts = []
    for c in selected:
        chunk_dict = {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id or "",
            "chunk_type": c.chunk_type or "paragraph",
            "text": c.text or "",
            "breadcrumb": c.breadcrumb or [],
            "document_position": 0,  # Will be resolved from corpus
            "reranker_score": c.reranker_score,
            "covered_requirements": c.covered_requirements,
            "matched_queries": c.matched_queries,
            "selection_rank": c.selection_rank,
        }
        chunk_dicts.append(chunk_dict)

    if enable_context_expansion:
        print(f"\nStep 14: Hierarchical context expansion")
        t0 = time.perf_counter()
        trace.log(
            "context_expansion_start",
            params={
                "max_total_tokens": max_total_tokens,
                "max_expanded_tokens_per_hit": max_expanded_tokens_per_hit,
                "n_selected_chunks": len(chunk_dicts),
            },
        )

        from medrag.retrieval.context_expansion import (
            expand_selected_context, ExpansionConfig, print_expansion_debug
        )

        expansion_config = ExpansionConfig(
            max_total_tokens=max_total_tokens,
            max_expanded_tokens_per_hit=max_expanded_tokens_per_hit,
            paragraph_window_before=1,
            paragraph_window_after=1,
            max_section_tokens=3000,
        )
        expansion_config_defined = True
    else:
        expansion_config = None
        expansion_config_defined = False

    # Load corpus DataFrame for context lookup
    import pandas as pd
    corpus_df = pd.read_parquet(
        str(index_dir / "corpus.parquet"),
        columns=["id", "document_id", "chunk_type", "text", "breadcrumb", "document_position"],
    )

    # Add document_position to chunk_dicts
    for cd in chunk_dicts:
        match = corpus_df[corpus_df["id"] == cd["chunk_id"]]
        if len(match) > 0:
            cd["document_position"] = match.iloc[0]["document_position"]

    # Run expansion (once, outside the loop)
    if expansion_config_defined:
        expansion_result = expand_selected_context(chunk_dicts, corpus_df, expansion_config)

        timings["expansion_ms"] = round((time.perf_counter() - t0) * 1000)
        n_exp = len(expansion_result.get("expanded_evidence", []))
        trace.log(
            "context_expansion_done",
            result={
                "n_expanded_blocks": n_exp,
                "diagnostics": expansion_result.get("diagnostics", {}),
                "blocks": [
                    {
                        "document_id": e["document_id"],
                        "context_type": e["context_type"],
                        "source_chunks": e["source_chunk_ids"],
                        "token_count": e["token_count"],
                        "truncated": e["truncated"],
                    }
                    for e in expansion_result.get("expanded_evidence", [])
                ],
            },
            duration_ms=timings["expansion_ms"],
        )

        # Print expansion debug
        from medrag.retrieval.context_expansion import ExpandedEvidence
        expanded_evidence = [
            ExpandedEvidence(
                document_id=e["document_id"],
                context_type=e["context_type"],
                source_chunk_ids=e["source_chunk_ids"],
                section_path=e["section_path"],
                text=e["text"],
                retrieval_evidence=e["retrieval_evidence"],
                token_count=e["token_count"],
                truncated=e["truncated"],
                original_token_count=e.get("original_token_count", 0),
                covered_requirements=e["covered_requirements"],
                matched_queries=e["matched_queries"],
                priority_score=e["priority_score"],
            )
            for e in expansion_result["expanded_evidence"]
        ]

        print_expansion_debug(expanded_evidence)
    else:
        print("\nStep 14: Context expansion DISABLED")
        timings["expansion_ms"] = 0

    # ── Step 13b: Ensure complete tables in final set ────────────
    print(f"\nStep 13b: Ensuring complete tables are included")
    t0 = time.perf_counter()

    trace.log(
        "table_complete_start",
        params={"n_selected": len(selected)},
    )
    selected = _expand_selected_tables(
        selected, corpus_index, candidates_dict, evidence_requirements,
    )

    timings["table_complete_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log(
        "table_complete_done",
        result={"n_after": len(selected)},
        duration_ms=timings["table_complete_ms"],
    )

    # ── Step 15: Coverage check ──────────────────────────────────
    trace.log("coverage_check_start", params={"n_requirements": len(evidence_requirements)})
    coverage_map = check_coverage(selected, evidence_requirements)

    print(f"\nStep 15: Evidence coverage check:")
    for i, req in enumerate(evidence_requirements):
        status = "COVERED ✓" if coverage_map.get(i, False) else "MISSING ✗"
        n_cand = sum(1 for c in selected if i in c.covered_requirements)
        print(f"  R{i+1}: {status} ({n_cand} candidates) — {req[:80]}...")

    all_covered = all(coverage_map.values())
    trace.log(
        "coverage_check_done",
        result={
            "all_covered": all_covered,
            "n_covered": sum(coverage_map.values()),
            "n_total": len(evidence_requirements),
            "per_requirement": {
                f"R{i+1}": {"covered": coverage_map.get(i, False), "n_candidates": sum(1 for c in selected if i in c.covered_requirements), "requirement": req}
                for i, req in enumerate(evidence_requirements)
            },
        },
    )
    print(f"\n  Overall: {'ALL EVIDENCE COVERED ✓' if all_covered else 'SOME REQUIREMENTS UNCOVERED ✗'}")

    # ── Step 16: Build result ────────────────────────────────────
    total_ms = round((time.perf_counter() - total_start) * 1000)
    timings["total_ms"] = total_ms
    trace.log("build_result", detail=f"selected={len(selected)}, reranked={len(reranked)}, candidates={len(candidates_dict)}")

    result = build_result(
        selected=selected,
        plan=plan,
        queries=queries,
        evidence_requirements=evidence_requirements,
        total_candidates=len(candidates_dict),
        reranked_count=len(reranked),
        coverage_map=coverage_map,
    )

    result["diagnostics"]["timings"] = timings

    # Add expanded evidence to result
    if expansion_result is not None:
        result["expanded_evidence"] = expansion_result["expanded_evidence"]
        result["diagnostics"]["expansion"] = expansion_result["diagnostics"]

    # ── Print summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RETRIEVAL SUMMARY")
    print("=" * 70)
    print(f"  Search queries:          {len(queries)}")
    print(f"  Total candidates:        {result['diagnostics']['candidate_count']}")
    print(f"  After reranking:         {result['diagnostics']['reranked_count']}")
    print(f"  Final passages:          {result['diagnostics']['final_count']}")
    print(f"  Unique documents:        {result['diagnostics']['document_count']}")
    print(f"  All evidence covered:    {result['diagnostics']['all_evidence_covered']}")
    print(f"  Mean coverage fraction:  {result['diagnostics']['coverage_distribution']['mean_coverage_fraction']:.4f}")
    print(f"  Zero-requirement:        {result['diagnostics']['coverage_distribution']['zero_requirements']}")
    print(f"  One-requirement:         {result['diagnostics']['coverage_distribution']['one_requirement']}")
    print(f"  Multi-requirement:       {result['diagnostics']['coverage_distribution']['multiple_requirements']}")
    print(f"  Total time:              {timings['total_ms']:.0f} ms")
    print()
    print("  Per-query coverage in final set:")
    for qid, count in result["diagnostics"]["query_coverage"].items():
        print(f"    {qid}: {count} passages")
    print()
    print("  Per-requirement coverage:")
    for req_key, count in result["diagnostics"]["per_requirement_coverage"].items():
        print(f"    {req_key}: {count} passages")
    print()
    print("  Document distribution:")
    for doc, count in sorted(result["diagnostics"]["selected_documents"].items(),
                              key=lambda x: x[1], reverse=True):
        print(f"    {doc}: {count} passages")

    print("\n  Final ranking:")
    for r in result["results"]:
        cr = r["covered_requirements"]
        cr_str = ",".join("R" + str(x+1) for x in cr) if cr else "none"
        print(f"    #{r['selection_rank']:>2}  rerank={r['reranker_score']:.4f}  "
              f"rrf={r['original_rrf_score']:.4f}  "
              f"cov={r['coverage_count']}/{len(evidence_requirements)}  "
              f"reqs=[{cr_str}]  "
              f"{r['chunk_id']}  ({r['document_id']})")

    # ═══════════════════════════════════════════════════════════════
    # OUTPUT: 2 files only — expanded evidence markdown + trace log
    # ═══════════════════════════════════════════════════════════════

    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_query = original_query[:60].replace(" ", "_").replace("/", "_")
        safe_query = "".join(c for c in safe_query if c.isalnum() or c in "_-")[:50]
        output_path = Path(f"retrieval_runs/{timestamp}_{safe_query}")

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # File 1: expanded_evidence.md
    expanded_evidence_list = expansion_result["expanded_evidence"] if expansion_result else []
    md_path = output_path.with_suffix(".md")
    _write_expanded_evidence_md(
        path=md_path,
        plan=plan,
        expanded_evidence=expanded_evidence_list,
        selected=selected,
        evidence_requirements=evidence_requirements,
        timings=timings,
    )
    trace.log("write_expanded_evidence_md", params={"path": str(md_path)}, result={"n_blocks": len(expanded_evidence_list)})

    # File 2: trace.log
    trace.log(
        "orchestration_complete",
        result={
            "total_ms": timings["total_ms"],
            "n_queries": len(queries),
            "n_candidates_initial": len(candidates_dict),
            "n_reranked": len(reranked),
            "n_selected": len(selected),
            "n_expanded": len(expanded_evidence_list),
            "all_evidence_covered": all_covered,
            "timings": timings,
        },
    )
    trace_path = output_path.with_suffix(".log")
    trace.save(trace_path)
    print(f"\n  Expanded evidence: {md_path}")
    print(f"  Execution trace:   {trace_path}")

    return result


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-query retrieval orchestrator v2",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=_INDEX_DIR)
    parser.add_argument("--bm25-depth", type=int, default=100)
    parser.add_argument("--dense-depth", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument("--final-top-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--req-threshold", type=float, default=REQUIREMENT_THRESHOLD,
                        help=f"Normalized evidence coverage threshold (default: {REQUIREMENT_THRESHOLD})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output base path (no extension). Produces <path>.md and <path>.log")

    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))

    result = run_orchestration(
        plan=plan,
        index_dir=args.index_dir,
        bm25_depth=args.bm25_depth,
        dense_depth=args.dense_depth,
        candidate_pool=args.candidate_pool,
        rerank_top_k=args.rerank_top_k,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
        req_threshold=args.req_threshold,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
