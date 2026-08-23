"""MedRAG Retrieval V2 pipeline - end-to-end orchestration.

Exact reference architecture (spec section 43):

    question -> planner -> requirement/hop graph
      -> per-query hybrid retrieval (BM25 + pgvector) with provenance
      -> query-local RRF (never global chunk fusion)
      -> paper-level aggregation
      -> per-branch shortlists (union) + cross-query paper ranking
      -> top papers -> logical document index -> paper-local search
      -> bounded parent/child structural expansion
      -> MedCPT cross-encoder (requirement <-> evidence)
      -> intent-aware scoring with bonuses/penalties
      -> greedy requirement-coverage selection
      -> targeted repair pass
      -> honest coverage report + trace
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG, config_from_env
from medrag.retrieval_v2.models import (
    CoverageReport,
    EvidenceCandidate,
    PaperSelection,
    QueryLocalResult,
    Requirement,
    SearchQuery,
    V2Plan,
)


_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INDEX_DIR = _ROOT / "index"


@dataclass
class V2Components:
    """Loaded indexes and models shared across pipeline stages (section 37)."""

    bm25: Any = None
    dense: Any = None
    store: Any = None
    doc_index: Any = None
    query_encoder: Any = None
    cross_encoder: Any = None
    llm: Any = None
    pageindex: Any = None          # PageIndexAdapter (paper-local navigation)
    trace: Any = field(default=None)
    index_dir: Path = field(default_factory=lambda: INDEX_DIR)
    load_ms: Dict[str, float] = field(default_factory=dict)

    def close(self) -> None:
        if self.store is not None and hasattr(self.store, "close"):
            try:
                self.store.close()
            except Exception:  # noqa: BLE001
                pass


def load_components(
    index_dir: Optional[Path] = None,
    config: Optional[V2Config] = None,
    trace: Any = None,
) -> V2Components:
    """Load BM25, dense (pgvector preferred), doc index and encoders once."""
    cfg = config or DEFAULT_CONFIG
    index_dir = Path(index_dir) if index_dir else INDEX_DIR
    comp = V2Components(index_dir=index_dir, trace=trace)
    from medrag.trace import TraceLogger as _TraceLogger
    from medrag.trace import set_trace
    from medrag.retrieval_v2.retriever import NullTrace
    if trace is not None:
        try:
            set_trace(trace)
        except Exception:  # noqa: BLE001
            pass
    elif comp.trace is None:
        # default to a real TraceLogger so every pipeline step is recorded
        # (spec section 39) - only tests inject NullTrace explicitly
        comp.trace = _TraceLogger()

    # BM25
    t0 = time.perf_counter()
    from medrag.retrieval.sparse import BM25Index
    comp.bm25 = BM25Index.load(index_dir / "bm25")
    comp.load_ms["bm25_ms"] = round((time.perf_counter() - t0) * 1000)

    # dense + pgvector store
    t0 = time.perf_counter()
    store = None
    dense = None
    if cfg.enable_dense:
        if cfg.use_pgvector:
            try:
                from medrag.retrieval.pgvector_store import PgVectorStore
                store = PgVectorStore()
                store.connect()
                store.ensure_schema()
                from medrag.retrieval.dense_pgvector import PgDenseIndex
                dense = PgDenseIndex(store)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: pgvector unavailable ({exc}); falling back to FAISS")
                store = None
                dense = None
        if dense is None:
            try:
                from medrag.retrieval.dense import DenseIndex
                dense = DenseIndex.load(index_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: dense index unavailable: {exc}")
                dense = None
    comp.dense = dense
    comp.store = store
    comp.load_ms["dense_ms"] = round((time.perf_counter() - t0) * 1000)

    # logical document index
    t0 = time.perf_counter()
    from medrag.retrieval_v2.document_index import LogicalDocumentIndex
    from medrag.retrieval.corpus import CORPUS_FILENAME
    comp.doc_index = LogicalDocumentIndex(index_dir / CORPUS_FILENAME, store=store)
    comp.load_ms["doc_index_ms"] = round((time.perf_counter() - t0) * 1000)
    print(f"  Document index: {comp.doc_index.n_chunks:,} chunks, {comp.doc_index.n_papers:,} papers")

    # PageIndex paper-local navigation adapter (V2.1 parts 24, 36-37)
    if cfg.enable_pageindex:
        t0 = time.perf_counter()
        from medrag.retrieval_v2.pageindex_adapter import PageIndexAdapter
        from medrag.retrieval_v2.config import V2Config as _V2Config
        pi_cfg = cfg
        pageindex_dir = Path(cfg.pageindex_dir)
        if not pageindex_dir.is_absolute():
            pageindex_dir = index_dir.parent / pageindex_dir if (index_dir.parent / pageindex_dir).exists() else index_dir / pageindex_dir
        comp.pageindex = PageIndexAdapter(
            doc_index=comp.doc_index, pageindex_dir=pageindex_dir,
            config=pi_cfg, trace=comp.trace or trace)
        comp.load_ms["pageindex_ms"] = round((time.perf_counter() - t0) * 1000)
        pi_dir = Path(cfg.pageindex_dir)
        n_artifacts = len(list(pi_dir.glob("*.json"))) if pi_dir.exists() else 0
        print(f"  PageIndex adapter: {comp.pageindex.library_version()} | artifacts={n_artifacts} | dir={pageindex_dir}")

    # MedCPT encoders
    t0 = time.perf_counter()
    from medrag.retrieval.query import MedCPTQueryEncoder
    comp.query_encoder = MedCPTQueryEncoder()
    if cfg.enable_rerank:
        from medrag.retrieval.reranker import CrossEncoderReranker
        comp.cross_encoder = CrossEncoderReranker(batch_size=cfg.rerank_batch_size, max_length=cfg.rerank_max_length)
    else:
        comp.cross_encoder = None
    comp.load_ms["encoders_ms"] = round((time.perf_counter() - t0) * 1000)

    # LLM / /expand planning support
    if cfg.llm_planning in ("enrich", "decompose") and cfg.llm_base_url:
        print(f"  Planning mode: {cfg.llm_planning} | Expand URL: {cfg.llm_base_url}")
    elif cfg.llm_planning in ("enrich", "decompose"):
        t0 = time.perf_counter()
        try:
            from medrag.llm_client import LLMClient
            llm_kwargs: Dict[str, Any] = {}
            if cfg.llm_model:
                llm_kwargs["model"] = cfg.llm_model
            comp.llm = LLMClient(**llm_kwargs)
            if not comp.llm.is_available():
                print(f"  WARNING: LLM client configured but no endpoint reachable at {comp.llm.base_url}")
                comp.llm = None
            else:
                print(f"  LLM: {comp.llm.base_url} / {comp.llm.model} (planning mode: {cfg.llm_planning})")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: LLM client failed to load ({exc})")
            comp.llm = None
        comp.load_ms["llm_ms"] = round((time.perf_counter() - t0) * 1000)

    return comp



def run_v2_pipeline(
    question: str,
    plan: Optional[Any] = None,
    config: Optional[V2Config] = None,
    index_dir: Optional[Path] = None,
    components: Optional[V2Components] = None,
    output_path: Optional[Path] = None,
    known_branch_papers: Optional[Dict[str, List[str]]] = None,
    keep_components: bool = False,
) -> Dict[str, Any]:
    """Run the full V2 retrieval pipeline for one question.

    ``known_branch_papers`` maps query ids (or requirement ids) to expected
    paper ids for regression metrics (spec section 41).
    """
    from medrag.trace import TraceLogger
    comp_trace = TraceLogger()
    cfg = config or DEFAULT_CONFIG

    if components is None:
        components = load_components(index_dir, cfg, trace=comp_trace)
    trace = components.trace or comp_trace
    if not hasattr(trace, "log"):
        trace = comp_trace

    result: Dict[str, Any] = {"query": question, "config": cfg.as_dict()}
    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    # ── 1. PLANNER (section 6-9) ────────────────────────────────────
    t0 = time.perf_counter()
    if isinstance(plan, V2Plan):
        v2_plan = plan
    elif isinstance(plan, dict):
        from medrag.retrieval_v2.planner import plan_from_dict
        v2_plan = plan_from_dict(plan)
    else:
        v2_plan = None
        # Priority: 1) /expand endpoint (Kaggle server), 2) direct LLM, 3) deterministic
        if cfg.llm_planning in ("decompose", "enrich") and cfg.llm_base_url:
            # Path 1: call /expand on the Kaggle server (concept extraction + BioPortal + intent)
            try:
                from medrag.retrieval_v2.llm_planner import call_expand, plan_from_expand
                expand_data = call_expand(cfg.llm_base_url, question)
                v2_plan = plan_from_expand(question, expand_data, cfg)
                v2_plan.planner_method = "expand+plan"
                trace.log("EXPAND_PLAN", params={"url": cfg.llm_base_url},
                          result={"method": "expand+plan",
                                  "n_requirements": len(v2_plan.requirements),
                                  "n_queries": len(v2_plan.queries),
                                  "conditions": v2_plan.conditions,
                                  "populations": v2_plan.populations,
                                  "warnings": v2_plan.warnings})
            except Exception as exc:  # noqa: BLE001
                trace.log("EXPAND_PLAN", error=str(exc))
                print(f"  WARNING: /expand call failed ({exc}), falling back to deterministic")

        if v2_plan is None and cfg.llm_planning == "decompose" and components.llm is not None:
            # Path 2: MedGemma STRUCTURED QUERY UNDERSTANDING via /chat/completions
            # (never free decomposition into requirements - part 4)
            try:
                from medrag.retrieval_v2.llm_planner import structured_plan_with_llm
                v2_plan = structured_plan_with_llm(question, components.llm, cfg)
                trace.log("LLM_STRUCTURED", result={"method": "llm_structured",
                                                    "n_requirements": len(v2_plan.requirements),
                                                    "n_queries": len(v2_plan.queries),
                                                    "warnings": v2_plan.warnings})
            except Exception as exc:  # noqa: BLE001
                trace.log("LLM_STRUCTURED", error=str(exc))
                print(f"  WARNING: LLM structured understanding failed ({exc}), falling back to deterministic")

        if v2_plan is None:
            # Path 3: deterministic planner
            from medrag.retrieval_v2.planner import plan_question
            v2_plan = plan_question(question, cfg)
    timings["plan_ms"] = round((time.perf_counter() - t0) * 1000)
    trace.log(
        "PLANNER",
        params={"query": question, "question_type": v2_plan.question_type},
        result={
            "conditions": v2_plan.conditions,
            "populations": v2_plan.populations,
            "target": v2_plan.target,
            "requirements": [r.to_dict() for r in v2_plan.requirements],
            "queries": [q.to_dict() for q in v2_plan.queries],
            "terminology_guard": v2_plan.terminology_guard,
        },
        duration_ms=timings["plan_ms"],
    )

    requirements = v2_plan.requirements
    queries = v2_plan.queries
    result["plan"] = v2_plan.to_dict()

    # ── 2. PER-QUERY HYBRID RETRIEVAL (sections 10-12) ─────────────
    from medrag.retrieval_v2.retriever import GlobalRetriever
    t0 = time.perf_counter()
    global_retriever = GlobalRetriever(
        components.bm25, components.dense, components.doc_index,
        components.query_encoder, cfg, trace,
    )
    query_results = global_retriever.retrieve(queries)
    timings["global_retrieval_ms"] = round((time.perf_counter() - t0) * 1000)
    result["query_results"] = [qr.summary() for qr in query_results]

    # ── 3. PAPER AGGREGATION (section 13) ───────────────────────────
    from medrag.retrieval_v2.paper_select import aggregate_papers, select_papers
    t0 = time.perf_counter()
    paper_scores = aggregate_papers(query_results, components.doc_index, cfg, trace)
    timings["paper_aggregation_ms"] = round((time.perf_counter() - t0) * 1000)

    # per-query paper ranking for metrics (paper recall / MRR)
    per_query_paper_ranking: Dict[str, List[str]] = {q.id: [] for q in queries}
    for q in queries:
        ranked = sorted(
            ((pid, pqs.score) for (qid, pid), pqs in paper_scores.items() if qid == q.id),
            key=lambda kv: kv[1], reverse=True,
        )
        per_query_paper_ranking[q.id] = [pid for pid, _ in ranked]
    result["paper_ranking_by_query"] = {
        qid: ps[:50] for qid, ps in per_query_paper_ranking.items()
    }

    # ── 4. PAPER SELECTION (sections 14-17) ─────────────────────────
    t0 = time.perf_counter()
    selected_papers, per_req_papers = select_papers(
        paper_scores, queries, requirements, components.doc_index, cfg, trace,
    )
    timings["paper_selection_ms"] = round((time.perf_counter() - t0) * 1000)
    result["papers"] = [p.to_dict() for p in selected_papers]
    trace.log(
        "PAPER_SELECTION",
        params={"n_requirements": len(requirements), "top_papers": cfg.top_papers},
        result={"selected": [p.to_dict() for p in selected_papers]},
        duration_ms=timings["paper_selection_ms"],
    )

    # papers_by_requirement for local search: shortlist + selected support
    papers_by_requirement: Dict[str, List[str]] = {}
    for req in requirements:
        pool: List[str] = []
        for p in per_req_papers.get(req.id, []):
            if p.paper_id not in pool:
                pool.append(p.paper_id)
        for p in selected_papers:
            if req.id in p.supported_requirements and p.paper_id not in pool:
                pool.append(p.paper_id)
        papers_by_requirement[req.id] = pool[: cfg.per_requirement_shortlist + 4]

    queries_by_requirement: Dict[str, List[SearchQuery]] = {}
    for q in queries:
        for rid in q.requirement_ids:
            queries_by_requirement.setdefault(rid, []).append(q)

    # ── 5. V2 LOCAL RETRIEVAL + EXPANSION (sections 18-24, parts 14/29-31) ─
    from medrag.retrieval_v2.local_search import LocalRetriever
    t0 = time.perf_counter()
    local_retriever = LocalRetriever(
        components.bm25, components.dense, components.doc_index,
        components.query_encoder, cfg, trace,
        pageindex=components.pageindex,
    )
    candidates_by_req = local_retriever.search_papers(
        papers_by_requirement, requirements, queries_by_requirement,
    )
    timings["local_search_ms"] = round((time.perf_counter() - t0) * 1000)
    result["local_search"] = {
        rid: [c.to_dict(include_text=False) for c in pool[:20]]
        for rid, pool in candidates_by_req.items()
    }

    # ── 6. MEDCPT RERANK (section 25) ───────────────────────────────
    from medrag.retrieval_v2.rerank import RequirementReranker
    t0 = time.perf_counter()
    reranker = RequirementReranker(components.cross_encoder, cfg, trace)
    reranker.rerank(candidates_by_req, requirements)
    timings["medcpt_ms"] = round((time.perf_counter() - t0) * 1000)

    # ── 7. INTENT-AWARE SCORING (sections 26-28) ────────────────────
    from medrag.retrieval_v2.intent import score_intent
    paper_relevance_map: Dict[str, float] = {}
    for p in selected_papers:
        vals = [v for v in p.per_requirement_scores.values() if v > 0]
        paper_relevance_map[p.paper_id] = max(vals) if vals else 0.0
    t0 = time.perf_counter()
    score_intent(candidates_by_req, requirements, cfg, paper_relevance_map, trace)
    timings["intent_ms"] = round((time.perf_counter() - t0) * 1000)

    # merge all candidates
    all_candidates: List[EvidenceCandidate] = []
    seen_candidates: Dict[str, EvidenceCandidate] = {}
    for rid, pool in candidates_by_req.items():
        for c in pool:
            if c.chunk_id in seen_candidates:
                # merge requirement coverage + max final score
                other = seen_candidates[c.chunk_id]
                other.requirement_ids = list(dict.fromkeys(other.requirement_ids + c.requirement_ids))
                other.medcpt_scores.update(c.medcpt_scores)
                other.medcpt_norm.update(c.medcpt_norm)
                other.covered_requirements = list(dict.fromkeys(other.covered_requirements + c.covered_requirements))
                other.final_score = max(other.final_score, c.final_score)
                continue
            seen_candidates[c.chunk_id] = c
            all_candidates.append(c)

    # ── 8. GREEDY COVERAGE SELECTION (section 30) ───────────────────
    from medrag.retrieval_v2.select import greedy_select
    t0 = time.perf_counter()
    selected = greedy_select(all_candidates, requirements, cfg, trace)
    timings["selection_ms"] = round((time.perf_counter() - t0) * 1000)

    # ── 9. REPAIR PASS (section 31) ─────────────────────────────────
    from medrag.retrieval_v2.select import repair_uncovered
    selected, still_uncovered = repair_uncovered(
        selected, candidates_by_req, requirements,
        local_retriever=local_retriever,
        papers_by_requirement=papers_by_requirement,
        config=cfg, trace=trace,
    )
    timings["repair_ms"] = round((time.perf_counter() - t0) * 1000)

    # ── 10. COVERAGE REPORT (sections 32-33) ────────────────────────
    from medrag.retrieval_v2.select import build_coverage_report
    coverage = build_coverage_report(selected, requirements, still_uncovered, candidates_by_req)
    result["coverage"] = coverage.to_dict()

    # ── 11. FINAL EVIDENCE SET ──────────────────────────────────────
    selected.sort(key=lambda c: c.selection_rank)
    result["final_evidence"] = [c.to_dict(include_text=False) for c in selected]
    result["n_final_evidence"] = len(selected)
    trace.log(
        "FINAL_EVIDENCE",
        params={"n_selected": len(selected)},
        result={
            "coverage": coverage.to_dict(),
            "evidence": [
                {"chunk_id": c.chunk_id, "paper_id": c.paper_id,
                 "node_type": c.node_type, "final_score": round(c.final_score, 4),
                 "covered_requirements": c.covered_requirements,
                 "medcpt": {k: round(v, 4) for k, v in c.medcpt_scores.items()},
                 "penalties": c.penalties,
                 "selection_reason": c.selection_reason}
                for c in selected
            ],
        },
    )

    # ── 12. EVALUATION METRICS (section 40-41) ──────────────────────
    from medrag.retrieval_v2.evaluator import evaluate_run
    pageindex_summary = None
    if components.pageindex is not None:
        pageindex_summary = components.pageindex.status_summary()
    metrics = evaluate_run(
        v2_plan,
        per_query_paper_ranking=per_query_paper_ranking,
        selected_papers=selected_papers,
        final_evidence=selected,
        coverage=coverage,
        known_branch_papers=known_branch_papers,
        pageindex_summary=pageindex_summary,
    )
    result["metrics"] = metrics
    result["pageindex"] = pageindex_summary
    timings["total_ms"] = round((time.perf_counter() - total_start) * 1000)
    result["timings"] = timings

    # ── OUTPUT ──────────────────────────────────────────────────────
    if output_path is not None:
        write_outputs(output_path, question, v2_plan, selected_papers, selected, coverage,
                      metrics, timings, trace)
    if not keep_components:
        components.close()
    return result



def write_outputs(
    output_path: Path,
    question: str,
    plan: V2Plan,
    selected_papers: Sequence[PaperSelection],
    selected: Sequence[EvidenceCandidate],
    coverage: CoverageReport,
    metrics: Dict[str, Any],
    timings: Dict[str, float],
    trace: Any,
) -> None:
    """Write trace log, JSON result and a readable evidence markdown."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    json_path = out.with_suffix(".json") if out.suffix else out / "v2_result.json"
    import json as _json
    payload: Dict[str, Any] = {
        "query": question,
        "plan": plan.to_dict(),
        "papers": [p.to_dict() for p in selected_papers],
        "coverage": coverage.to_dict(),
        "metrics": metrics,
        "timings": timings,
        "final_evidence": [c.to_dict(include_text=True) for c in selected],
    }
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if trace is not None and hasattr(trace, "save"):
        trace_path = output_path.with_suffix(".log") if output_path.suffix else out / "v2_trace.log"
        trace.save(Path(trace_path))

    md_path = output_path.with_suffix(".md") if output_path.suffix else out / "v2_evidence.md"
    lines: List[str] = []
    lines.append("# MedRAG Retrieval V2 Output")
    lines.append("")
    lines.append("**Query:** " + question)
    lines.append("")
    cov = coverage
    cov_line = "**Coverage:** " + str(len(cov.covered)) + "/" + str(len(cov.covered) + len(cov.uncovered)) + " requirements covered - " + str(round(cov.coverage_fraction, 2))
    lines.append(cov_line)
    lines.append("")
    lines.append("## Requirements")
    for r in plan.requirements:
        status = "COVERED" if r.id in cov.covered else "UNCOVERED"
        pop_part = (" / " + r.population) if r.population else ""
        lines.append("- **" + r.id + "** [" + status + "] " + r.topic + pop_part + " / " + r.focus)
    lines.append("")
    lines.append("## Selected Papers")
    lines.append("| Rank | Paper | Score | Requirements | Reason |")
    lines.append("|---|---|---|---|---|")
    for p in selected_papers:
        reqs = ", ".join(p.supported_requirements)
        row = "| " + str(selected_papers.index(p) + 1) + " | " + str(p.paper_id) + " | " + str(round(p.paper_score, 4)) + " | " + reqs + " | " + p.reason + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Final Evidence Set")
    lines.append("| # | Chunk | Paper | Type | Reqs | MedCPT | Penalty | Final |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in selected:
        med = max(c.medcpt_scores.values()) if c.medcpt_scores else 0.0
        reqs = ", ".join(c.covered_requirements) if c.covered_requirements else "-"
        row = "| " + str(c.selection_rank) + " | " + str(c.chunk_id) + " | " + str(c.paper_id) + " | " + str(c.node_type)
        row += " | " + reqs + " | " + str(round(med, 3)) + " | " + str(round(c.penalty_total, 2)) + " | " + str(round(c.final_score, 3)) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Evidence Text")
    for c in selected:
        lines.append("### #" + str(c.selection_rank) + " - " + str(c.chunk_id) + " (" + str(c.paper_id) + ", " + str(c.node_type) + ")")
        lines.append("")
        if c.breadcrumb:
            lines.append("*Section:* " + " > ".join(c.breadcrumb))
        lines.append("")
        if c.pageindex_node_ids:
            lines.append("*PageIndex nodes:* " + ", ".join(c.pageindex_node_ids))
        if c.detected_fields:
            present = [k for k, v in c.detected_fields.items() if v]
            if present:
                lines.append("*Detected fields:* " + ", ".join(present))
        lines.append("")
        if c.context_text and c.context_text != c.text:
            lines.append("**Context** (scored evidence object):")
            lines.append("")
            lines.append((c.context_text or "")[:2500])
            lines.append("")
            lines.append("**Chunk text** (provenance):")
            lines.append("")
            lines.append((c.text or "")[:2500])
        else:
            lines.append((c.text or "")[:2500])
        lines.append("")
        lines.append("---")
        lines.append("")
    lines.append("## Regression Metrics (spec section 41)")
    lines.append("```json")
    lines.append(_json.dumps(metrics, indent=2, ensure_ascii=False))
    lines.append("```")
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")
    print("  JSON:      " + str(json_path))
    print("  Markdown:  " + str(md_path))
    if trace is not None and hasattr(trace, "save"):
        print("  Trace:     " + str(output_path.with_suffix(".log") if output_path.suffix else out / "v2_trace.log"))

