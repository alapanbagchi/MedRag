"""MedRAG v2 Pipeline: Intent-Aware Retrieval + Reranking + Evidence Fusion.

Architecture:
    Original query  → Kaggle /expand (LLM concept extraction + BioPortal + intent)
    Expanded query  → LOCAL BM25 + Dense → RRF → TOP 100
    Original query  → LOCAL MedCPT Cross-Encoder → TOP 50
    Intent (from /expand) → LOCAL Intent Similarity + Answerability → Score Fusion → TOP 20

Usage:
    python -m medrag.pipeline_v2

The Kaggle /expand endpoint performs the expensive LLM and BioPortal work.
All retrieval, reranking, and scoring happen locally.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from medrag.retrieval.corpus import CORPUS_FILENAME, CorpusIndex  # noqa: E402
from medrag.retrieval.dense import DenseIndex  # noqa: E402
from medrag.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from medrag.retrieval.query import MedCPTQueryEncoder  # noqa: E402
from medrag.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from medrag.retrieval.sparse import BM25Index  # noqa: E402
from medrag.models import Candidate  # noqa: E402


# ============================================================
# Constants
# ============================================================

INDEX_DIR = Path("index")
BM25_DIR = INDEX_DIR / "bm25"

WEIGHT_MEDCPT = 0.70
WEIGHT_INTENT = 0.15
WEIGHT_ANSWERABILITY = 0.15

INTENT_TYPES = {
    "threshold", "treatment", "diagnosis", "prognosis", "risk", "dosage",
    "adverse_effects", "comparison", "mechanism", "epidemiology",
    "screening", "guideline", "definition",
}


# ============================================================
# Kaggle /expand client (with cache)
# ============================================================

class _ExpandCache:
    """Simple per-process cache for /expand responses."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(query)

    def set(self, query: str, data: Dict[str, Any]) -> None:
        self._cache[query] = data

    def clear(self) -> None:
        self._cache.clear()


_expand_cache = _ExpandCache()


def get_query_intelligence(
    query: str,
    kaggle_url: Optional[str] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call the Kaggle /expand endpoint to get enriched query intelligence.

    Returns:
        {
            "expanded_query": str,
            "intent": dict,
            "concepts": list,
            "expanded_terms": list,
            "bioportal": list,
        }
    """
    # Check cache
    cached = _expand_cache.get(query)
    if cached is not None:
        return cached

    url = (kaggle_url or os.environ.get("KAGGLE_URL", "")).rstrip("/")
    if not url:
        raise ValueError(
            "KAGGLE_URL not set. Set the environment variable or pass kaggle_url."
        )

    expand_url = f"{url}/expand"

    try:
        resp = requests.post(
            expand_url,
            json={"query": query},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach Kaggle /expand at {expand_url}. "
            f"Is the Cloudflare tunnel running? ({exc})"
        ) from exc
    except requests.Timeout as exc:
        raise TimeoutError(
            f"/expand request timed out after {timeout}s. "
            f"The LLM enrichment may be slow."
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"/expand returned HTTP {resp.status_code}: {resp.text[:200]}"
        ) from exc

    intelligence = {
        "expanded_query": data.get("expanded_query", query),
        "intent": data.get("intent", {}),
        "concepts": data.get("concepts", []),
        "expanded_terms": data.get("expanded_terms", []),
        "bioportal": data.get("bioportal", []),
    }

    _expand_cache.set(query, intelligence)
    return intelligence


# ============================================================
# Pipeline Context (LOCAL indexes + models only)
# ============================================================

class PipelineContext:
    """Holds all LOCAL indexes and models. No LLM/BioPortal here."""

    def __init__(
        self,
        sparse_index: BM25Index,
        dense_index: Optional[DenseIndex],
        query_encoder: MedCPTQueryEncoder,
        cross_encoder: CrossEncoderReranker,
        corpus_index: CorpusIndex,
    ) -> None:
        self.sparse = sparse_index
        self.dense = dense_index
        self.query_encoder = query_encoder
        self.cross_encoder = cross_encoder
        self.corpus = corpus_index

        # Pre-build chunk_id → FAISS-position lookup for embedding reconstruction
        self._id_to_pos: Dict[str, int] = {}
        if dense_index is not None:
            self._id_to_pos = {
                cid: i for i, cid in enumerate(dense_index.chunk_ids)
            }

    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Reconstruct the document embedding for a single chunk from FAISS."""
        if self.dense is None:
            return None
        pos = self._id_to_pos.get(chunk_id)
        if pos is None:
            return None
        return self.dense.index.reconstruct(pos).astype(np.float32)


# ============================================================
# Intent helpers (used locally for evidence requirements)
# ============================================================

# Keyword → intent mapping for rule-based fallback
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "threshold":        ["threshold", "cutoff", "cut-off", "score", "level", "above", "below"],
    "treatment":        ["treatment", "therapy", "medication", "drug", "manage"],
    "diagnosis":        ["diagnosis", "diagnose", "criteria", "classify"],
    "prognosis":        ["prognosis", "outcome", "survival", "mortality"],
    "risk":             ["risk factor", "risk of", "associated with"],
    "dosage":           ["dose", "dosage", "how much", "mg", "units"],
    "adverse_effects":  ["side effect", "adverse", "complication"],
    "comparison":       ["compare", "versus", "vs", "better", "superior"],
    "mechanism":        ["mechanism", "how does", "pathophysiology"],
    "epidemiology":     ["prevalence", "incidence", "epidemiology"],
    "screening":        ["screening", "screen", "test for"],
    "guideline":        ["guideline", "recommendation", "consensus"],
    "definition":       ["what is", "what are", "define"],
}


def build_intent_representation(intent: Dict[str, Any]) -> str:
    """Build a compact semicolon-delimited string of the intent fields."""
    parts = []
    for key in ("intent", "target", "condition", "action"):
        val = intent.get(key, "")
        if val:
            parts.append(str(val).strip())
    return "; ".join(parts) if parts else "medical information"


def build_evidence_requirements(intent: Dict[str, Any]) -> List[str]:
    """Return the required_evidence list from the intent dict."""
    evidence = intent.get("required_evidence", [])
    if evidence:
        return evidence
    parts = []
    if intent.get("target"):
        parts.append(f"{intent['target']} information")
    if intent.get("condition"):
        parts.append(f"{intent['condition']} evidence")
    return parts if parts else ["medical information"]


# ============================================================
# Intent Similarity (local — uses MedCPT query encoder)
# ============================================================

def compute_intent_similarities(
    intent_rep: str,
    chunk_ids: List[str],
    ctx: PipelineContext,
) -> Dict[str, float]:
    """Cosine similarity between intent and each candidate's embedding."""
    intent_vec = ctx.query_encoder.encode([intent_rep])[0]
    scores: Dict[str, float] = {}
    for cid in chunk_ids:
        emb = ctx.get_embedding(cid)
        scores[cid] = float(np.dot(intent_vec, emb)) if emb is not None else 0.0
    return scores


# ============================================================
# Answerability (local — uses MedCPT query encoder)
# ============================================================

def compute_answerability(
    evidence_requirements: List[str],
    chunk_ids: List[str],
    ctx: PipelineContext,
) -> Dict[str, float]:
    """Score each candidate against evidence requirements.

    answerability = 0.7 * max(scores) + 0.3 * mean(top_3_scores)
    """
    if not evidence_requirements:
        return {cid: 0.0 for cid in chunk_ids}

    evidence_vecs = ctx.query_encoder.encode(evidence_requirements)

    scores: Dict[str, float] = {}
    for cid in chunk_ids:
        emb = ctx.get_embedding(cid)
        if emb is None:
            scores[cid] = 0.0
            continue
        per_req = np.dot(evidence_vecs, emb)
        sorted_scores = np.sort(per_req)[::-1]
        max_score = float(sorted_scores[0])
        top_3 = sorted_scores[: min(3, len(sorted_scores))]
        scores[cid] = 0.7 * max_score + 0.3 * float(np.mean(top_3))
    return scores


# ============================================================
# Score normalization and fusion
# ============================================================

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()))
    lo, hi = float(vals.min()), float(vals.max())
    rng = hi - lo
    if rng == 0:
        return {k: 1.0 for k in scores}
    return {k: float((v - lo) / rng) for k, v in scores.items()}


def fuse_scores(
    medcpt_norm: Dict[str, float],
    intent_norm: Dict[str, float],
    answerability_norm: Dict[str, float],
    w_medcpt: float = WEIGHT_MEDCPT,
    w_intent: float = WEIGHT_INTENT,
    w_answer: float = WEIGHT_ANSWERABILITY,
) -> Dict[str, float]:
    """Weighted fusion of the three normalized score signals."""
    fused: Dict[str, float] = {}
    for cid in medcpt_norm:
        fused[cid] = (
            w_medcpt * medcpt_norm.get(cid, 0.0)
            + w_intent * intent_norm.get(cid, 0.0)
            + w_answer * answerability_norm.get(cid, 0.0)
        )
    return fused


# ============================================================
# Main pipeline
# ============================================================

def rerank_pipeline(
    original_query: str,
    ctx: PipelineContext,
    kaggle_url: Optional[str] = None,
    top_hybrid: int = 100,
    top_medcpt: int = 50,
    top_final: int = 20,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """Run the full staged pipeline.

    1. Call Kaggle /expand for enriched query intelligence
    2. Use expanded query for LOCAL BM25 + Dense → RRF → TOP 100
    3. Use original query for LOCAL MedCPT → TOP 50
    4. Use intent from /expand for LOCAL intent + answerability → TOP 20
    """
    total_start = time.perf_counter()

    # ── Stage 1: Query intelligence (REMOTE via Kaggle /expand) ────
    t0 = time.perf_counter()
    intelligence = get_query_intelligence(original_query, kaggle_url)
    expand_ms = (time.perf_counter() - t0) * 1000

    expanded_query = intelligence["expanded_query"]
    intent = intelligence["intent"]
    concepts = intelligence["concepts"]
    bioportal_terms = intelligence["bioportal"]

    intent_rep = build_intent_representation(intent)
    evidence_reqs = build_evidence_requirements(intent)

    print("\n" + "=" * 70)
    print("STAGE 1 — QUERY INTELLIGENCE (Kaggle /expand)")
    print("=" * 70)
    print(f"Original query : {original_query}")
    print(f"Expanded query : {expanded_query}")
    print(f"Concepts       : {len(concepts)}")
    for c in concepts[:5]:
        name = c.get("name", c) if isinstance(c, dict) else str(c)
        print(f"  • {name}")
    print(f"Intent         : {intent.get('intent', '')} | target={intent.get('target', '')} | condition={intent.get('condition', '')}")
    print(f"Required evidence:")
    for e in evidence_reqs:
        print(f"  • {e}")
    print(f"/expand latency: {expand_ms:.0f} ms")

    # ── Stage 2: Encode EXPANDED query for dense search ─────────────
    t0 = time.perf_counter()
    query_vec = ctx.query_encoder.encode([expanded_query])[0]
    query_encode_ms = (time.perf_counter() - t0) * 1000

    # ── Stage 3: LOCAL BM25 + Dense → RRF → TOP 100 ────────────────
    t0 = time.perf_counter()

    bm25_results = ctx.sparse.search(expanded_query, top_hybrid)[0]
    bm25_ranked = [(h.chunk_id, h.score) for h in bm25_results]

    dense_ranked: List[Tuple[str, float]] = []
    if ctx.dense is not None:
        dense_results = ctx.dense.search_single(query_vec, top_hybrid)
        dense_ranked = [(h.chunk_id, h.score) for h in dense_results]

    lists_to_fuse = [bm25_ranked]
    if dense_ranked:
        lists_to_fuse.append(dense_ranked)

    rrf_scores = reciprocal_rank_fusion(lists_to_fuse)
    sorted_rrf = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_100_ids = [cid for cid, _ in sorted_rrf[:top_hybrid]]
    retrieval_ms = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 70)
    print("STAGE 2 — LOCAL RETRIEVAL (EXPANDED query)")
    print("=" * 70)
    print(f"BM25 hits      : {len(bm25_ranked)}")
    print(f"Dense hits     : {len(dense_ranked)}")
    print(f"RRF candidates : {len(sorted_rrf)}")
    print(f"Retrieved TOP {len(top_100_ids)} in {retrieval_ms:.0f} ms")

    # ── Stage 4: Resolve metadata for TOP 100 ──────────────────────
    t0 = time.perf_counter()
    resolved = ctx.corpus.resolve(top_100_ids, include_text=True)

    candidates: List[Candidate] = []
    for rank_idx, cid in enumerate(top_100_ids):
        meta = resolved.get(cid, {})
        candidates.append(Candidate(
            chunk_id=cid,
            document_id=meta.get("document_id"),
            original_score=rrf_scores[cid],
            original_rank=rank_idx + 1,
            chunk_type=meta.get("chunk_type"),
            breadcrumb=meta.get("breadcrumb"),
            text=meta.get("text"),
            retrieval_method="hybrid",
        ))
    resolve_ms = (time.perf_counter() - t0) * 1000

    # ── Stage 5: LOCAL MedCPT reranking with ORIGINAL query ─────────
    t0 = time.perf_counter()
    reranked = ctx.cross_encoder.rerank(
        original_query, candidates, top_k=top_medcpt,
    )
    medcpt_ms = (time.perf_counter() - t0) * 1000

    medcpt_scores: Dict[str, float] = {}
    medcpt_ids: List[str] = []
    for r in reranked:
        medcpt_scores[r.chunk_id] = r.reranker_score
        medcpt_ids.append(r.chunk_id)

    print("\n" + "=" * 70)
    print("STAGE 3 — LOCAL MedCPT (ORIGINAL query)")
    print("=" * 70)
    print(f"Reranked {len(candidates)} → TOP {len(medcpt_ids)}")
    print(f"MedCPT latency: {medcpt_ms:.0f} ms")
    print(f"Score range: [{min(medcpt_scores.values()):.3f}, {max(medcpt_scores.values()):.3f}]")

    # ── Stage 6: LOCAL intent similarity + answerability ────────────
    t0 = time.perf_counter()

    has_embeddings = ctx.dense is not None
    if has_embeddings:
        intent_scores_raw = compute_intent_similarities(intent_rep, medcpt_ids, ctx)
        answerability_raw = compute_answerability(evidence_reqs, medcpt_ids, ctx)
    else:
        intent_scores_raw = {cid: 0.0 for cid in medcpt_ids}
        answerability_raw = {cid: 0.0 for cid in medcpt_ids}
        print("  (dense index not loaded — intent/answerability scores zeroed)")

    score_ms = (time.perf_counter() - t0) * 1000

    medcpt_norm = normalize_scores(medcpt_scores)
    intent_norm = normalize_scores(intent_scores_raw)
    answer_norm = normalize_scores(answerability_raw)

    # ── Stage 7: Fuse and select TOP 20 ────────────────────────────
    fused = fuse_scores(medcpt_norm, intent_norm, answer_norm)
    sorted_fused = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    final_ids = [cid for cid, _ in sorted_fused[:top_final]]

    total_ms = (time.perf_counter() - total_start) * 1000

    print("\n" + "=" * 70)
    print("STAGE 4 — LOCAL INTENT + ANSWERABILITY + FUSION")
    print("=" * 70)
    print(f"Computed in {score_ms:.0f} ms")

    # ── Build final results ─────────────────────────────────────────
    final_results: List[Dict[str, Any]] = []
    for rank, cid in enumerate(final_ids, start=1):
        r = next((x for x in reranked if x.chunk_id == cid), None)
        meta = resolved.get(cid, {})
        final_results.append({
            "rank": rank,
            "chunk_id": cid,
            "document_id": meta.get("document_id", r.document_id if r else None),
            "chunk_type": meta.get("chunk_type", r.chunk_type if r else None),
            "breadcrumb": meta.get("breadcrumb", r.breadcrumb if r else None),
            "text": meta.get("text", r.text if r else ""),
            "medcpt_score": medcpt_scores.get(cid, 0.0),
            "medcpt_norm": medcpt_norm.get(cid, 0.0),
            "intent_score": intent_scores_raw.get(cid, 0.0),
            "intent_norm": intent_norm.get(cid, 0.0),
            "answerability_score": answerability_raw.get(cid, 0.0),
            "answerability_norm": answer_norm.get(cid, 0.0),
            "final_score": fused.get(cid, 0.0),
        })

    # ── Print the ranking funnel ────────────────────────────────────
    print("\n" + "=" * 70)
    print("RANKING FUNNEL")
    print("=" * 70)
    print(f"\n  {'Hybrid TOP 100':.<50} {len(top_100_ids)} candidates")
    print(f"  {'MedCPT TOP 50':.<50} {len(medcpt_ids)} candidates")
    print(f"  {'Intent + Answerability + Fusion':.<50} {len(final_ids)} candidates")
    print(f"  {'FINAL TOP':.<50} {top_final}")
    print(f"\n  Total pipeline: {total_ms:.0f} ms  (Kaggle /expand: {expand_ms:.0f} ms)")

    print("\n" + "=" * 70)
    print("FINAL TOP 20 — DETAILED")
    print("=" * 70)
    print(f"\n  {'#':<4} {'MedCPT':>8} {'Intent':>8} {'Answer':>8} {'Final':>8}  {'Chunk ID':<30} {'Section'}")
    print("  " + "-" * 105)
    for r in final_results:
        bc = r["breadcrumb"]
        bc_str = str(bc) if bc else ""
        if len(bc_str) > 40:
            bc_str = bc_str[:37] + "..."
        print(
            f"  {r['rank']:<4} "
            f"{r['medcpt_norm']:>8.3f} "
            f"{r['intent_norm']:>8.3f} "
            f"{r['answerability_norm']:>8.3f} "
            f"{r['final_score']:>8.3f}  "
            f"{r['chunk_id']:<30} "
            f"{bc_str}"
        )

    print("\n" + "=" * 70)
    print("FINAL TOP 20 — TEXT PREVIEWS")
    print("=" * 70)
    for r in final_results:
        print(f"\n{'─' * 70}")
        print(f"  RANK {r['rank']}: {r['chunk_id']}  ({r['document_id']})")
        print(f"  MedCPT: {r['medcpt_score']:.3f}  Intent: {r['intent_score']:.3f}  "
              f"Answerability: {r['answerability_score']:.3f}  Final: {r['final_score']:.3f}")
        if r["breadcrumb"]:
            print(f"  Section: {r['breadcrumb']}")
        text = (r["text"] or "")[:500]
        print(f"  Text: {text}{'…' if len(r.get('text', '')) > 500 else ''}")

    if return_debug:
        return {
            "query": original_query,
            "expanded_query": expanded_query,
            "concepts": concepts,
            "bioportal": bioportal_terms,
            "intent": intent,
            "required_evidence": evidence_reqs,
            "intent_representation": intent_rep,
            "hybrid_top_100": [
                {"chunk_id": cid, "rrf_score": rrf_scores[cid]} for cid in top_100_ids
            ],
            "medcpt_top_50": [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "medcpt_score": r.reranker_score,
                    "original_rank": r.original_rank,
                    "text": (r.text or "")[:300],
                    "breadcrumb": r.breadcrumb,
                }
                for r in reranked
            ],
            "final_results": final_results,
            "timings": {
                "expand_ms": expand_ms,
                "retrieval_ms": retrieval_ms,
                "medcpt_ms": medcpt_ms,
                "score_ms": score_ms,
                "total_ms": total_ms,
            },
        }

    return {"query": original_query, "results": final_results}


# ============================================================
# Local pipeline context loader
# ============================================================

def load_pipeline_context() -> PipelineContext:
    """Load LOCAL indexes and models (no LLM/BioPortal — those are on Kaggle)."""
    print("Loading BM25 index...")
    sparse = BM25Index.load(BM25_DIR)
    print(f"  {sparse.n_docs:,} docs, vocab {sparse.vocab_size:,}")

    dense = None
    if (INDEX_DIR / "dense.faiss").exists():
        print("Loading dense index (2.7 GB)...")
        dense = DenseIndex.load(INDEX_DIR)
        print(f"  {dense.n_total:,} vectors, dim={dense.dimension}")
    else:
        print("Dense index not found — BM25-only for retrieval.")

    print("Loading query encoder (MedCPT-Query-Encoder)...")
    qe = MedCPTQueryEncoder()

    print("Loading cross-encoder (MedCPT-Cross-Encoder)...")
    ce = CrossEncoderReranker(batch_size=32, max_length=512)

    print("Loading corpus index...")
    corpus = CorpusIndex(INDEX_DIR / CORPUS_FILENAME, load=True)

    # Check Kaggle URL
    kaggle_url = os.environ.get("KAGGLE_URL", "")
    if kaggle_url:
        print(f"Kaggle /expand: {kaggle_url}")
    else:
        print("KAGGLE_URL not set — /expand calls will fail.")

    ctx = PipelineContext(sparse, dense, qe, ce, corpus)
    print("Pipeline ready.\n")
    return ctx


# ============================================================
# Interactive main
# ============================================================

def main():
    """Interactive pipeline: ask a question, get ranked evidence."""
    print("=" * 70)
    print("MedRAG v2 — Intent-Aware Retrieval Pipeline")
    print("=" * 70)

    ctx = load_pipeline_context()
    kaggle_url = os.environ.get("KAGGLE_URL", "")

    while True:
        print("\n" + "-" * 70)
        query = input("\nEnter your medical question: ").strip()
        if not query:
            break

        try:
            result = rerank_pipeline(query, ctx, kaggle_url=kaggle_url)
        except Exception as exc:
            print(f"\nERROR: {exc}")
            import traceback
            traceback.print_exc()
            continue

        again = input("\nRun another query? [y/N] ").strip().lower()
        if again != "y":
            break

    print("Done.")


if __name__ == "__main__":
    main()
