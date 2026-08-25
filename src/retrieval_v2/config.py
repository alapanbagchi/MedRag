"""Central configuration for the MedRAG retrieval V2 pipeline.

Every weight, threshold and depth lives here — never in business logic — so the
whole pipeline can be tuned from a single object. ``DEFAULT_CONFIG`` mirrors the
V2 specification suggested starting values; ``config_from_env`` overrides items
from environment variables.

The config also carries the PageIndex paper-local navigation settings
(spec parts 9-14, 24-27), the retrieval-variant budget (part 6), the
table-aware evidence-context settings (parts 15-16, 31) and the contextual
intent-scoring components (parts 18-19, 33-34).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict


def _weights(**kw: float) -> Dict[str, float]:
    return dict(kw)


@dataclass
class V2Config:
    # ----------------------------------------------------------------
    # Global per-query retrieval (spec section 10)
    # ----------------------------------------------------------------
    bm25_depth: int = 100          # BM25 top-k per query (global stage)
    dense_depth: int = 100         # pgvector top-k per query (global stage)
    rrf_k: float = 60.0            # RRF constant

    # ----------------------------------------------------------------
    # Retrieval variants (spec part 6): a small number of compact
    # per-requirement query variants, each retrieved independently.
    # ----------------------------------------------------------------
    max_retrieval_variants: int = 5    # V0..V4, never 10+
    max_plan_queries: int = 28         # hard cap on total plan queries

    # ----------------------------------------------------------------
    # Paper aggregation - paper_query_score (section 13)
    # ----------------------------------------------------------------
    paper_query_weights: Dict[str, float] = field(
        default_factory=lambda: _weights(
            max_score=0.50,
            mean_topk=0.20,
            support=0.10,
            section_diversity=0.10,
            method_agreement=0.10,
        )
    )
    paper_query_mean_top_n: int = 3  # mean of the paper top-N chunk scores

    # ----------------------------------------------------------------
    # Paper selection - per-branch shortlist + cross-query ranking (14-15)
    # ----------------------------------------------------------------
    per_requirement_shortlist: int = 5   # top N papers kept per requirement
    branch_guarantee_top: int = 2        # guaranteed kept per requirement
    top_papers: int = 12                 # final research-neighborhood size
    paper_rank_weights: Dict[str, float] = field(
        default_factory=lambda: _weights(
            strongest_branch=0.40,
            requirement_coverage=0.25,
            mean_branch=0.20,
            evidence_diversity=0.10,
            method_agreement=0.05,
        )
    )
    requirement_presence_threshold: float = 0.20  # normalized per-req score counts as supports
    bridge_bonus_weight: float = 0.05             # bonus per extra distinct requirement

    # ----------------------------------------------------------------
    # V2 local search inside selected papers (18-20)
    # ----------------------------------------------------------------
    local_bm25_depth: int = 40
    local_dense_depth: int = 60
    local_fused_depth: int = 40   # keep this many local fused hits per (paper, req)
    local_variant_fused_depth: int = 20   # fused depth per single variant
    local_anchor_hits: int = 8    # structural expansion anchors per (paper, req)
    local_nav_anchor_rank: int = 6  # how many PageIndex-derived chunks become anchors

    # ----------------------------------------------------------------
    # PageIndex paper-local navigation (parts 9-14, 24-27, 36-37)
    # ----------------------------------------------------------------
    pageindex_dir: str = "index/pageindex"   # precomputed artifacts per paper
    enable_pageindex: bool = True            # on/off switch (adapter is optional)
    pageindex_nav_top_k: int = 8             # nodes returned per (paper, requirement)
    pageindex_min_relevance: float = 0.05   # drop hits below this
    pageindex_max_breadth: int = 6           # max children explored per level

    # ----------------------------------------------------------------
    # Bounded parent/child expansion (24)
    # ----------------------------------------------------------------
    max_total_tokens: int = 12000
    max_tokens_per_hit: int = 2000
    max_expansions_per_hit: int = 12
    neighbor_window: int = 1      # paragraph neighbors before/after

    # ----------------------------------------------------------------
    # MedCPT cross-encoder (25, 36 performance)
    # ----------------------------------------------------------------
    rerank_pair_cap: int = 150    # hard cap on (candidate x requirement) pairs
    rerank_batch_size: int = 32
    rerank_max_length: int = 512

    # ----------------------------------------------------------------
    # Table-aware evidence context (parts 15-16, 19, 31)
    # ----------------------------------------------------------------
    table_context_max_chars: int = 3500   # context_text cap
    table_context_row_window: int = 8     # rows shown around the target row
    table_context_footnotes: int = 2      # max footnotes included
    table_context_include_headers: bool = True
    table_context_include_breadcrumb: bool = True

    # ----------------------------------------------------------------
    # Intent-aware final scoring (26-28, 33)
    # ----------------------------------------------------------------
    final_weights: Dict[str, float] = field(
        default_factory=lambda: _weights(
            medcpt=0.45,
            requirement_match=0.18,
            evidence_type=0.12,
            requested_field=0.12,
            paper_relevance=0.05,
            diversity=0.05,
        )
    )

    # Contextual intent components (spec part 18). requirement_match is the
    # weighted sum of these components; each component is computed against the
    # contextual evidence text (table context for rows, passage for paragraphs).
    intent_weights: Dict[str, float] = field(
        default_factory=lambda: _weights(
            concept_match=0.22,           # condition concept (surface/base/synonyms)
            clinical_condition_match=0.18,  # base concept + modifiers
            target_match=0.12,            # target: table headers/variable for rows (part 18)
            outcome_match=0.10,
            requested_field_match=0.18,
            evidence_type_match=0.12,
            population_match=0.08,
        )
    )
    # Multiplier applied to the requested-field signal for tabular evidence so a
    # structurally-correct table row can dominate generic topic prose (part 33).
    table_requested_field_multiplier: float = 1.25

    # Penalties (27, 34). Scaled by penalty_scale, subtracted from the weighted
    # positive score. The flags keep penalties CONTEXTUAL: a table row whose
    # header establishes the variable as the target must NOT be penalized for
    # not literally containing the target phrase.
    penalty_scale: float = 1.0
    penalty_target_mismatch: float = 3.0      # A - paragraphs only (tables use header context)
    penalty_population_mismatch: float = 3.0  # B
    penalty_outcome_mismatch: float = 2.0     # C
    penalty_requested_field: float = 1.5      # D
    penalty_wrong_concept: float = 4.0        # F - INOCA vs MINOCA etc.
    penalty_generic_topic: float = 2.0        # G
    redundancy_alpha: float = 0.35            # H - continuous redundancy scale

    # ----------------------------------------------------------------
    # Greedy coverage selection (30)
    # ----------------------------------------------------------------
    final_slots: int = 18
    new_requirement_bonus: float = 0.18
    new_evidence_type_bonus: float = 0.04
    new_paper_bonus: float = 0.04
    doc_soft_cap: int = 4          # per-paper soft cap in final set

    # ----------------------------------------------------------------
    # Coverage / honesty (31-33, 40, parts 20-21)
    # ----------------------------------------------------------------
    coverage_min_evidence_per_requirement: int = 1
    medcpt_coverage_threshold: float = 0.30       # normalized medcpt to count covered
    requirement_match_threshold: float = 0.35     # normalized slot match to count covered

    # ----------------------------------------------------------------
    # Evidence-type boosts by question focus (20)  focus -> {type: boost}
    # "results" is a section hint: paragraph nodes inside the Results section.
    # ----------------------------------------------------------------
    evidence_type_boosts: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "definition": {"paragraph": 0.15, "table_summary": 0.25, "table_row": 0.10},
            "numerical": {
                "table_row": 0.30, "table_summary": 0.20, "table_footnotes": 0.15,
                "figure": 0.15, "paragraph": 0.05, "results": 0.10,
            },
            "comparative_numerical": {
                "table_row": 0.35, "table_summary": 0.25, "table_footnotes": 0.20,
                "results": 0.20, "paragraph": 0.08, "figure": 0.10,
            },
            "mechanism": {"paragraph": 0.10, "figure": 0.20, "table_row": 0.05, "results": 0.10},
            "population": {"table_row": 0.20, "table_summary": 0.15, "paragraph": 0.10},
            "outcome": {
                "table_row": 0.20, "table_summary": 0.10, "figure": 0.15, "paragraph": 0.10,
            },
            "treatment": {"table_row": 0.20, "table_summary": 0.15, "paragraph": 0.10},
            "comparison": {"table_row": 0.25, "table_summary": 0.15, "paragraph": 0.10},
            "diagnosis": {"table_row": 0.20, "table_summary": 0.25, "paragraph": 0.10},
            "table_lookup": {"table_row": 0.40, "table_summary": 0.30, "table_footnotes": 0.20},
            "figure_interpretation": {"figure": 0.40, "paragraph": 0.10},
        }
    )

    # ----------------------------------------------------------------
    # Backends / performance (36-38)
    # ----------------------------------------------------------------
    use_pgvector: bool = True       # else FAISS dense fallback
    enable_dense: bool = True
    enable_rerank: bool = True
    enable_ontology: bool = False   # UMLS/MeSH/BioPortal normalization locally (fail-soft)
    cache_query_embeddings: bool = True
    cache_retrieval: bool = True

    # ----------------------------------------------------------------
    # Planner knobs
    # ----------------------------------------------------------------
    # LLM planning: "off" (deterministic), "enrich" (LLM fills concepts/entities),
    # "decompose" (LLM structured query understanding, deterministic as fallback)
    llm_planning: str = "off"
    llm_model: str = ""                 # overrides LLM_MODEL env var
    llm_base_url: str = ""              # overrides LLM_BASE_URL env var
    default_population: str = "adults"
    outcome_fallback: str = "clinical outcomes"

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for fname in __dataclass_fields__:  # type: ignore[name-defined]
            val = getattr(self, fname)
            if isinstance(val, dict):
                out[fname] = dict(val)
            else:
                out[fname] = val
        return out

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in self.as_dict().items()]
        return "V2Config: " + ", ".join(parts)


DEFAULT_CONFIG = V2Config()


def config_from_env(overrides: Dict[str, Any] | None = None) -> V2Config:
    """Build a config from environment variables + explicit overrides."""
    cfg = V2Config()

    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    cfg.bm25_depth = _int("MEDRAG_V2_BM25_DEPTH", cfg.bm25_depth)
    cfg.dense_depth = _int("MEDRAG_V2_DENSE_DEPTH", cfg.dense_depth)
    cfg.top_papers = _int("MEDRAG_V2_TOP_PAPERS", cfg.top_papers)
    cfg.final_slots = _int("MEDRAG_V2_FINAL_SLOTS", cfg.final_slots)
    cfg.max_total_tokens = _int("MEDRAG_V2_MAX_TOTAL_TOKENS", cfg.max_total_tokens)
    cfg.rerank_pair_cap = _int("MEDRAG_V2_RERANK_PAIR_CAP", cfg.rerank_pair_cap)
    cfg.local_bm25_depth = _int("MEDRAG_V2_LOCAL_BM25_DEPTH", cfg.local_bm25_depth)
    cfg.local_dense_depth = _int("MEDRAG_V2_LOCAL_DENSE_DEPTH", cfg.local_dense_depth)
    cfg.max_retrieval_variants = _int("MEDRAG_V2_MAX_VARIANTS", cfg.max_retrieval_variants)
    cfg.pageindex_dir = os.environ.get("MEDRAG_V2_PAGEINDEX_DIR", cfg.pageindex_dir)
    cfg.enable_pageindex = _bool("MEDRAG_V2_ENABLE_PAGEINDEX", cfg.enable_pageindex)
    cfg.enable_rerank = _bool("MEDRAG_V2_ENABLE_RERANK", cfg.enable_rerank)
    cfg.enable_dense = _bool("MEDRAG_V2_ENABLE_DENSE", cfg.enable_dense)
    cfg.enable_ontology = _bool("MEDRAG_V2_ENABLE_ONTOLOGY", cfg.enable_ontology)
    cfg.cache_query_embeddings = _bool("MEDRAG_V2_CACHE_EMBEDDINGS", cfg.cache_query_embeddings)
    cfg.llm_planning = os.environ.get("MEDRAG_V2_LLM_PLANNING", cfg.llm_planning)
    cfg.llm_base_url = os.environ.get("LLM_BASE_URL", cfg.llm_base_url)
    cfg.llm_model = os.environ.get("LLM_MODEL", cfg.llm_model)

    if overrides:
        for key, val in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
    return cfg


from dataclasses import fields as _fields  # noqa: E402
__dataclass_fields__ = {f.name: f for f in _fields(V2Config)}

