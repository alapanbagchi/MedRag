"""Evaluation framework.

Metrics are computed only where ground truth exists. Queries without any
labelled relevant chunk/document ids are reported as ``null`` rather than
given a fabricated score.

Supported metrics:
    recall@k          fraction of relevant chunk ids found in the top-k
    document_recall@k fraction of relevant document ids found in the top-k
    mrr               reciprocal rank of the first relevant chunk
    mean_latency_ms   retrieval latency (query encoding excluded)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

DEFAULT_K_VALUES = (5, 10, 20, 50)
DEFAULT_METHODS = ("dense", "bm25", "hybrid")


@dataclass
class EvalQuery:
    query: str
    relevant_chunk_ids: List[str]
    relevant_document_ids: List[str] = field(default_factory=list)


def load_eval_queries(path: Path) -> List[EvalQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = data.get("queries", data if isinstance(data, list) else [])
    out: List[EvalQuery] = []
    for q in queries:
        out.append(
            EvalQuery(
                query=q["query"],
                relevant_chunk_ids=list(q.get("relevant_chunk_ids", []) or []),
                relevant_document_ids=list(q.get("relevant_document_ids", []) or []),
            )
        )
    return out


def recall_at_k(ranked_ids: Sequence[str], relevant: Sequence[str], k: int) -> Optional[float]:
    if not relevant:
        return None
    if k <= 0:
        return 0.0
    rel = set(relevant)
    top = set(ranked_ids[:k])
    return len(rel & top) / len(rel)


def document_recall_at_k(
    ranked_docs: Sequence[Optional[str]], relevant_docs: Sequence[str], k: int
) -> Optional[float]:
    if not relevant_docs:
        return None
    if k <= 0:
        return 0.0
    rel = set(relevant_docs)
    top = {d for d in ranked_docs[:k] if d is not None}
    return len(rel & top) / len(rel)


def mrr(ranked_ids: Sequence[str], relevant: Sequence[str]) -> Optional[float]:
    if not relevant:
        return None
    rel = set(relevant)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in rel:
            return 1.0 / rank
    return 0.0


def evaluate(
    engine,
    queries: Sequence[EvalQuery],
    methods: Sequence[str] = DEFAULT_METHODS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    candidate_k: int = 100,
    progress: bool = False,
) -> Dict[str, Any]:
    """Evaluate retrieval methods against labelled queries.

    When ``progress`` is true a single tqdm bar tracks the full method x
    query loop (each tick is one query for one method).
    """
    k_values = list(k_values)
    max_k = max(k_values) if k_values else 50

    summary: Dict[str, Any] = {}
    per_query: List[Dict[str, Any]] = []

    # Dense/hybrid methods need a query vector; encode each query once and
    # reuse it across every method instead of paying MedCPT encoding once
    # per (method, query) combination.
    needs_query_vector = {
        m[: -len("+rerank")] if m.endswith("+rerank") else m for m in methods
    } & {"dense", "hybrid"}
    vectors_by_query = (
        {q.query: engine.query_encoder.encode([q.query])[0] for q in queries}
        if needs_query_vector
        else {}
    )

    total_ticks = len(methods) * len(queries)
    progress_bar = (
        tqdm(total=total_ticks, desc="Evaluating", unit="q", dynamic_ncols=True)
        if progress
        else None
    )

    for method in methods:
        # A method named "hybrid+rerank" (or "dense+rerank"/"bm25+rerank")
        # runs the corresponding first-stage method with reranking enabled.
        rerank = method.endswith("+rerank")
        base_method = method[: -len("+rerank")] if rerank else method

        accum: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
        accum.update({f"document_recall@{k}": [] for k in k_values})
        accum["mrr"] = []
        accum["mean_latency_ms"] = []
        accum["mean_rerank_latency_ms"] = []
        accum["candidate_count"] = []

        for q in queries:
            result = engine.search(
                q.query,
                method=base_method,
                top_k=max_k,
                include_text=False,
                rerank=rerank,
                candidate_k=candidate_k,
                query_vector=vectors_by_query.get(q.query),
            )
            if progress_bar is not None:
                progress_bar.update(1)
            ranked_ids = [r.chunk_id for r in result.results]
            ranked_docs = engine.document_ids(result.results)

            row_metrics: Dict[str, Any] = {
                "query": q.query,
                "method": method,
                "latency_ms": result.latency_ms,
                "rerank_latency_ms": result.rerank_latency_ms,
                "candidate_count": result.candidate_count,
                "num_relevant_chunk_ids": len(q.relevant_chunk_ids),
                "num_relevant_document_ids": len(q.relevant_document_ids),
                "top_chunk_ids": ranked_ids[: max(k_values)] if k_values else [],
            }

            for k in k_values:
                rc = recall_at_k(ranked_ids, q.relevant_chunk_ids, k)
                dc = document_recall_at_k(ranked_docs, q.relevant_document_ids, k)
                row_metrics[f"recall@{k}"] = rc
                row_metrics[f"document_recall@{k}"] = dc
                if rc is not None:
                    accum[f"recall@{k}"].append(rc)
                if dc is not None:
                    accum[f"document_recall@{k}"].append(dc)

            m = mrr(ranked_ids, q.relevant_chunk_ids)
            row_metrics["mrr"] = m
            if m is not None:
                accum["mrr"].append(m)
            if result.latency_ms is not None:
                accum["mean_latency_ms"].append(result.latency_ms)
            if result.rerank_latency_ms is not None:
                accum["mean_rerank_latency_ms"].append(result.rerank_latency_ms)
            if result.candidate_count is not None:
                accum["candidate_count"].append(result.candidate_count)

            per_query.append(row_metrics)

        method_summary: Dict[str, Any] = {
            "num_queries": len(queries),
            "num_with_ground_truth": len(accum["mrr"]),
        }
        for key, values in accum.items():
            if not values:
                method_summary[key] = None
            else:
                method_summary[key] = round(sum(values) / len(values), 6)
        summary[method] = method_summary

    if progress_bar is not None:
        progress_bar.close()

    return {
        "summary": summary,
        "per_query": per_query,
        "k_values": k_values,
        "methods": list(methods),
        "note": (
            "Metrics are computed only over queries with ground truth. "
            "None / null means insufficient ground-truth data."
        ),
    }


def write_report(report: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
