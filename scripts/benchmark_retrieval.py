"""Proper retrieval benchmark over a deterministic 50-document sample.

Pipeline
--------
1. Select 50 documents deterministically (seeded) from the consolidated
   corpus index.
2. For each document, pick representative retrieval-eligible chunks
   (abstract paragraph, longest paragraph, list, table summary) and craft
   three query types per chunk with exact ground truth:

       verbatim  : first 60 words of the chunk (self-retrieval upper bound)
       keywords  : top TF-weighted non-stopword terms of the chunk
       contextual: "<Article title> - <first 40 words>" (harder, semantic)

3. Evaluate dense, bm25, hybrid, and hybrid+rerank against the FULL index
   (all 883,975 chunks stay in the pool; ground truth is the source chunk).
4. Write eval/benchmark_queries.json, eval/benchmark_report.json and a
   human-readable markdown report.

Usage
-----
    python scripts/benchmark_retrieval.py [--seed 42] [--n-docs 50]
        [--queries-per-doc 3] [--methods dense,bm25,hybrid,hybrid+rerank]
        [--k 5,10,20,50] [--candidate-k 100] [--max-queries 400]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medrag.retrieval.engine import RetrievalEngine  # noqa: E402
from medrag.retrieval.evaluation import (  # noqa: E402
    EvalQuery,
    evaluate,
    load_eval_queries,
)

STOPWORDS = frozenset(
    """a an the and or but if then than so for of in on at by to from with without
    into over under between during before after above below against among around
    this that these those it its is are was were be been being have has had having
    do does did doing will would shall should can could may might must ought need
    not no nor i you he she we they them his her their our your my its each both
    few more most other some such only own same very just also than too any all
    as per vs via e.g i.e et al fig table figure ref references abstract
    background methods results conclusion discussion introduction study patients
    group groups patient data results significant significant difference however
    moreover additionally furthermore according clinical trial clinical trials
    """.split()
)

WORD_RE = re.compile(r"[a-z0-9]+")
CHUNK_DIR = Path("chunks")
INDEX_DIR = Path("index")
CORPUS_PATH = Path("index/corpus.parquet")


# ----------------------------------------------------------------------
# Query construction
# ----------------------------------------------------------------------

def _words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def top_keywords(text: str, n: int = 6) -> List[str]:
    counts: Counter = Counter()
    for tok in WORD_RE.findall(text.lower()):
        if len(tok) > 2 and tok not in STOPWORDS:
            counts[tok] += 1
    return [tok for tok, _ in counts.most_common(n)]


def build_queries_from_doc(
    doc_id: str,
    title: str,
    chunks_df: pd.DataFrame,
    queries_per_doc: int = 3,
) -> List[Dict[str, Any]]:
    """Create (query, kind, chunk) triples for one document.

    Chunks are ranked by usefulness: abstract paragraph first, then longest
    paragraph, then first list, then first table summary. The first
    ``queries_per_doc`` distinct chunks produce three queries each.
    """
    elig = chunks_df[
        chunks_df["retrieval_eligible"].fillna(False).astype(bool)
        & chunks_df["chunk_type"].isin(
            ("paragraph", "list", "table_summary")
        )
    ].sort_values("document_position")

    def first_of(kind: str) -> Optional[pd.Series]:
        sub = elig[elig["chunk_type"] == kind]
        return None if sub.empty else sub.iloc[0]

    abstract = elig[(elig["section"] == "Abstract") & (elig["chunk_type"] == "paragraph")]
    paragraphs = elig[elig["chunk_type"] == "paragraph"].sort_values(
        "document_position", ascending=False
    )

    candidates: List[pd.Series] = []
    seen_ids = set()
    pick_order = [
        abstract.iloc[0] if not abstract.empty else None,
        first_of("paragraph"),
        paragraphs.iloc[0] if not paragraphs.empty else None,
        first_of("list"),
        first_of("table_summary"),
        first_of("paragraph"),
    ]
    for row in pick_order:
        if row is None or row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        candidates.append(row)
        if len(candidates) >= queries_per_doc:
            break

    queries: List[Dict[str, Any]] = []
    for row in candidates:
        text = str(row["text"] or "")
        if len(text.split()) < 8:
            continue
        chunk_id = str(row["id"])
        kind_data = (
            ("verbatim", _words(text, 60)),
            ("keywords", " ".join(top_keywords(text, 6)) or _words(text, 40)),
            ("contextual", f"{title} - {_words(text, 40)}"),
        )
        for kind, query in kind_data:
            queries.append({
                "query": query,
                "kind": kind,
                "chunk_id": chunk_id,
                "chunk_type": str(row["chunk_type"]),
                "section": str(row["section"] or ""),
                "relevant_chunk_ids": [chunk_id],
                "relevant_document_ids": [doc_id],
            })
    return queries


# ----------------------------------------------------------------------
# Selection of the 50-document sample
# ----------------------------------------------------------------------

def select_docs(seed: int, n_docs: int) -> List[str]:
    """Deterministically pick ``n_docs`` documents with a healthy chunk count."""
    meta = pq.read_table(
        str(CORPUS_PATH), columns=["id", "document_id", "chunk_type"]
    ).to_pandas()
    per_doc = meta.groupby("document_id").size()
    eligible = per_doc[per_doc >= 10]
    pool = eligible.sort_values(ascending=False).index.tolist()
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(n_docs, len(pool)))
    print(f"Selected {len(chosen)} documents from {len(pool)} candidates "
          f"(min 10 eligible chunks each)")
    return chosen


def analyze_sample(doc_ids: Sequence[str]) -> Dict[str, Any]:
    """Summarize chunk-type composition of the sampled documents."""
    counters: Counter = Counter()
    per_doc: List[Dict[str, Any]] = []
    for doc in doc_ids:
        files = sorted(CHUNK_DIR.glob(f"{doc}.*.parquet"))
        total = 0
        types: Counter = Counter()
        for f in files:
            t = pq.read_table(str(f))
            df = t.to_pandas()
            elig = df[df["retrieval_eligible"].fillna(False).astype(bool)]
            total += len(elig)
            types.update(elig["chunk_type"].value_counts().to_dict())
        counters.update(types)
        per_doc.append({"document_id": doc, "eligible_chunks": total, "types": dict(types)})
    return {
        "documents": len(doc_ids),
        "chunk_types": dict(sorted(counters.items())),
        "mean_chunks_per_doc": round(sum(p["eligible_chunks"] for p in per_doc) / len(per_doc), 1),
        "per_document": per_doc,
    }


# ----------------------------------------------------------------------
# Report writing
# ----------------------------------------------------------------------

def format_metric(value: Any) -> str:
    return "  -  " if value is None else f"{value:.4f}"


def write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("# MedRAG Retrieval Benchmark")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Corpus: {report['corpus']['eligible_chunks']:,} eligible chunks "
                 f"across {report['corpus']['documents']:,} documents (full index)")
    lines.append(f"- Sample: {report['sample']['documents']} documents, "
                 f"{report['sample']['queries']} queries")
    lines.append(f"- Index: dense={report['index']['dense_type']}, "
                 f"BM25(k1={report['index']['k1']}, b={report['index']['b']}), "
                 f"fusion={report['index']['fusion']}")
    lines.append("")

    lines.append("## Sample composition")
    lines.append("")
    lines.append("| Chunk type | Count |")
    lines.append("|---|---|")
    for ctype, count in report["sample"]["chunk_types"].items():
        lines.append(f"| {ctype} | {count} |")
    lines.append("")
    lines.append(f"Mean eligible chunks per sampled document: "
                 f"{report['sample']['mean_chunks_per_doc']}")
    lines.append("")

    lines.append("## Query mix")
    lines.append("")
    lines.append("| Kind | Count | Description |")
    lines.append("|---|---|---|")
    for kind, count in report["query_mix"].items():
        desc = {
            "verbatim": "First 60 words of the source chunk (self-retrieval upper bound)",
            "keywords": "Top TF-weighted non-stopword terms of the source chunk",
            "contextual": "Article title + first 40 words of the chunk (semantic)",
        }.get(kind, "")
        lines.append(f"| {kind} | {count} | {desc} |")
    lines.append("")

    lines.append("## Headline metrics (mean over queries with ground truth)")
    lines.append("")
    lines.append("| Method | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR | Doc Recall@10 | Mean latency (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for method, m in report["summary"].items():
        lines.append(
            f"| {method} | {format_metric(m.get('recall@5'))} | "
            f"{format_metric(m.get('recall@10'))} | {format_metric(m.get('recall@20'))} | "
            f"{format_metric(m.get('recall@50'))} | {format_metric(m.get('mrr'))} | "
            f"{format_metric(m.get('document_recall@10'))} | "
            f"{m.get('mean_latency_ms'):.1f} |"
        )
    lines.append("")

    lines.append("## Breakdown by query kind (recall@10 / MRR)")
    lines.append("")
    lines.append("| Kind | Method | Recall@10 | MRR | n |")
    lines.append("|---|---|---|---|---|")
    for kind in report["by_kind"]:
        for method in report["by_kind"][kind]:
            m = report["by_kind"][kind][method]
            lines.append(
                f"| {kind} | {method} | {format_metric(m['recall@10'])} | "
                f"{format_metric(m['mrr'])} | {m['n']} |"
            )
    lines.append("")

    lines.append("## Breakdown by chunk type (recall@10)")
    lines.append("")
    lines.append("| Chunk type | Method | Recall@10 | n |")
    lines.append("|---|---|---|---|")
    for ctype in report["by_chunk_type"]:
        for method in report["by_chunk_type"][ctype]:
            m = report["by_chunk_type"][ctype][method]
            lines.append(
                f"| {ctype} | {method} | {format_metric(m['recall@10'])} | {m['n']} |"
            )
    lines.append("")

    lines.append("## Worst queries (first rerank method, MRR = 0.0)")
    lines.append("")
    rows = report.get("worst_queries", [])
    if not rows:
        lines.append("None - every query was retrieved.")
    else:
        lines.append("| Document | Query | Recall@10 |")
        lines.append("|---|---|---|")
        for r in rows[:15]:
            lines.append(f"| {r['document_id']} | {r['query'][:80]} | {r['recall@10']} |")
    lines.append("")

    if report.get("latency_probe"):
        lines.append("## Production-style single-query latency (warm model)")
        lines.append("")
        lines.append("| Method | Mean (ms) | Median (ms) | Probes |")
        lines.append("|---|---|---|---|")
        for method, p in report["latency_probe"].items():
            lines.append(f"| {method} | {p['mean_ms']} | {p['median_ms']} | {p['probes']} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------

def _mean(values: Sequence[float]) -> Optional[float]:
    return round(float(np.mean(values)), 6) if values else None


def aggregate_rows(
    per_query: Sequence[Dict[str, Any]],
    key: str,
    methods: Sequence[str],
    k: int = 10,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group per-query rows by ``key`` and method; return metrics per group."""
    grouped: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counters: Dict[Any, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in per_query:
        group = row.get(key, "unknown")
        m = row["method"]
        if m not in methods:
            continue
        rc = row[f"recall@{k}"]
        mr = row["mrr"]
        if rc is not None:
            grouped[group][m].append(rc)
        if mr is not None:
            grouped[group]["_mrr_" + m].append(mr)
        counters[group][m] = counters[group].get(m, 0) + 1

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for group, by_method in grouped.items():
        out[str(group)] = {
            m: {
                f"recall@{k}": _mean(by_method.get(m, [])),
                "mrr": _mean(by_method.get("_mrr_" + m, [])),
                "n": counters[group].get(m, 0),
            }
            for m in methods
        }
    return out


def method_summary_from_rows(
    rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """Aggregate per-query rows for one method (mirrors evaluation.evaluate)."""
    k_values = list(k_values)
    accum: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    accum.update({f"document_recall@{k}": [] for k in k_values})
    accum["mrr"] = []
    accum["mean_latency_ms"] = []
    accum["mean_rerank_latency_ms"] = []
    accum["candidate_count"] = []

    for row in rows:
        for k in k_values:
            rc = row[f"recall@{k}"]
            dc = row[f"document_recall@{k}"]
            if rc is not None:
                accum[f"recall@{k}"].append(rc)
            if dc is not None:
                accum[f"document_recall@{k}"].append(dc)
        m = row["mrr"]
        if m is not None:
            accum["mrr"].append(m)
        if row["latency_ms"] is not None:
            accum["mean_latency_ms"].append(row["latency_ms"])
        if row["rerank_latency_ms"] is not None:
            accum["mean_rerank_latency_ms"].append(row["rerank_latency_ms"])
        if row["candidate_count"] is not None:
            accum["candidate_count"].append(row["candidate_count"])

    summary: Dict[str, Any] = {
        "num_queries": len(rows),
        "num_with_ground_truth": len(accum["mrr"]),
    }
    for key, values in accum.items():
        summary[key] = None if not values else round(sum(values) / len(values), 6)
    return summary


def macro_rerank_eval(
    engine,
    reranker,
    eval_queries: Sequence[EvalQuery],
    method: str,
    k_values: Sequence[int],
    candidate_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate one ``+rerank`` method with a single macro-batched rerank pass.

    Candidate generation still runs per query, but the (query, passage)
    pairs of ALL queries are scored in one big batched loop via
    ``reranker.rerank_many``. This keeps the CPU busy with much larger
    matrices than a per-query loop and amortizes Python/tokenizer overhead.
    """
    from medrag.models import Candidate
    from medrag.retrieval.evaluation import (
        document_recall_at_k,
        mrr,
        recall_at_k,
    )

    k_values = list(k_values)
    max_k = max(k_values) if k_values else 50
    base_method = method[: -len("+rerank")]

    # Query vectors are shared across rerank methods that need them.
    if base_method in ("dense", "hybrid"):
        vectors = {
            q.query: engine.query_encoder.encode([q.query])[0] for q in eval_queries
        }
    else:
        vectors = {}

    tasks: List[Tuple[str, List[Candidate]]] = []
    cand_gen_ms: List[float] = []
    candidates_per_query: List[List[Candidate]] = []

    with tqdm(
        eval_queries,
        total=len(eval_queries),
        desc=f"Retrieving candidates ({method})",
        unit="q",
        dynamic_ncols=True,
    ) as bar:
        for q in bar:
            result = engine.search(
                q.query,
                method=base_method,
                top_k=candidate_k,
                include_text=True,
                rerank=False,
                query_vector=vectors.get(q.query),
            )
            cands = [Candidate.from_scored(h) for h in result.results]
            cand_gen_ms.append(result.latency_ms or 0.0)
            tasks.append((q.query, cands))
            candidates_per_query.append(cands)

    t0 = time.perf_counter()
    reranked = reranker.rerank_many(
        tasks, top_k=max_k, batch_size=reranker.batch_size, progress=True
    )
    rerank_ms = (time.perf_counter() - t0) * 1000.0
    per_query_rerank_ms = rerank_ms / max(1, len(eval_queries))

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(eval_queries):
        hits = reranked[i]
        ranked_ids = [h.chunk_id for h in hits]
        ranked_docs = [h.document_id for h in hits]
        row: Dict[str, Any] = {
            "query": q.query,
            "method": method,
            "latency_ms": cand_gen_ms[i] + per_query_rerank_ms,
            "rerank_latency_ms": per_query_rerank_ms,
            "candidate_count": len(candidates_per_query[i]),
            "num_relevant_chunk_ids": len(q.relevant_chunk_ids),
            "num_relevant_document_ids": len(q.relevant_document_ids),
            "top_chunk_ids": ranked_ids[: max_k] if k_values else [],
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, q.relevant_chunk_ids, k)
            row[f"document_recall@{k}"] = document_recall_at_k(
                ranked_docs, q.relevant_document_ids, k
            )
        row["mrr"] = mrr(ranked_ids, q.relevant_chunk_ids)
        rows.append(row)

    return rows, method_summary_from_rows(rows, k_values)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--queries-per-doc", type=int, default=3)
    ap.add_argument("--methods", default="dense,bm25,hybrid,bm25+rerank,hybrid+rerank")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--reranker-batch-size", type=int, default=32)
    ap.add_argument("--reranker-max-length", type=int, default=512)
    ap.add_argument("--reranker-device", choices=("auto", "cuda", "cpu"), default="auto",
                    help="Device for the cross-encoder (auto = CUDA when available)")
    ap.add_argument("--reranker-quantize", action="store_true",
                    help="int8-dynamic-quantize the cross-encoder (CPU only, ~2-4x faster)")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="PyTorch intra-op threads (0 = torch default = 8 on this machine; more is NOT faster for the quantized cross-encoder)")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--queries-file", default=None,
                    help="Load pre-generated queries (with kind/ground truth) instead of "
                         "generating ones; see scripts/generate_qa_benchmark.py")
    ap.add_argument("--latency-probes", type=int, default=3,
                    help="Number of single-query warm-latency probes per rerank method (0 = skip)")
    ap.add_argument("--cool", action="store_true",
                    help="thermally conservative preset for laptops: cap torch to 4 threads, "
                         "batch 32, maxlen 192 (halves power draw and heat at a small recall cost)")
    ap.add_argument("--no-dense", action="store_true",
                    help="do not load the 2.7 GB dense FAISS index (BM25-only runs; "
                         "saves RAM and heat. Dense/hybrid methods will be rejected.)")
    args = ap.parse_args(argv)

    if args.cool:
        # Must be set before torch is imported (torch reads them at import).
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        if args.torch_threads <= 0:
            args.torch_threads = 4
        if args.reranker_batch_size > 32:
            args.reranker_batch_size = 32
        if args.reranker_max_length > 192:
            args.reranker_max_length = 192
        print("Cool mode: threads=4, batch<=32, maxlen<=192")

    if args.torch_threads and args.torch_threads > 0:
        import torch

        torch.set_num_threads(args.torch_threads)
        print(f"torch intra-op threads: {torch.get_num_threads()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    k_values = sorted(args.k)
    rng = random.Random(args.seed)

    t0 = time.perf_counter()

    # 1/2. Queries -----------------------------------------------------------
    # Either load a pre-generated QA benchmark (from generate_qa_benchmark.py)
    # or build the default self-retrieval queries from a document sample.
    if args.queries_file:
        payload = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
        queries = list(payload.get("queries", []))
        if args.max_queries > 0 and len(queries) > args.max_queries:
            queries = rng.sample(queries, args.max_queries)
        eval_queries = [
            EvalQuery(
                query=q["query"],
                relevant_chunk_ids=q["relevant_chunk_ids"],
                relevant_document_ids=q["relevant_document_ids"],
            )
            for q in queries
        ]
        query_by_text = {q["query"]: q for q in queries}
        query_mix = Counter(q.get("kind", "unknown") for q in queries)
        doc_ids = sorted({q.get("document_id", "") for q in queries if q.get("document_id")})
        sample_analysis: Dict[str, Any] = {
            "documents": len(doc_ids),
            "queries": len(eval_queries),
            "chunk_types": dict(sorted(Counter(q.get("chunk_type", "unknown") for q in queries).items())),
            "mean_chunks_per_doc": round(
                len(eval_queries) / max(1, len(doc_ids)), 2
            ),
        }
        print(f"Loaded {len(eval_queries)} queries from {args.queries_file}")
    else:
        doc_ids = select_docs(args.seed, args.n_docs)
        queries: List[Dict[str, Any]] = []
        titles: Dict[str, str] = {}
        for doc in doc_ids:
            files = sorted(CHUNK_DIR.glob(f"{doc}.*.parquet"))
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
            if df.empty:
                continue
            meta = df["metadata"].iloc[0] if "metadata" in df.columns else {}
            title = str(meta.get("title") or doc) if isinstance(meta, dict) else doc
            titles[doc] = title
            queries.extend(
                build_queries_from_doc(doc, title, df, args.queries_per_doc)
            )

        if args.max_queries > 0 and len(queries) > args.max_queries:
            queries = rng.sample(queries, args.max_queries)

        eval_queries = [
            EvalQuery(
                query=q["query"],
                relevant_chunk_ids=q["relevant_chunk_ids"],
                relevant_document_ids=q["relevant_document_ids"],
            )
            for q in queries
        ]
        query_by_text = {q["query"]: q for q in queries}
        query_mix = Counter(q["kind"] for q in queries)

        # 3. Sample analysis --------------------------------------------------
        sample_analysis = analyze_sample(doc_ids)

        queries_payload = {
            "note": (
                "Self-retrieval benchmark queries with exact ground truth "
                "(source chunk + document) derived from a deterministic 50-document sample. "
                "Queries: verbatim prefix, top-keyword, and title-contextual forms."
            ),
            "queries": [
                {k: v for k, v in q.items() if k in ("query", "relevant_chunk_ids", "relevant_document_ids")}
                for q in queries
            ],
        }
        queries_path = out_dir / "benchmark_queries.json"
        queries_path.write_text(json.dumps(queries_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(eval_queries)} queries -> {queries_path.resolve()}")

    # 4. Retrieval evaluation ------------------------------------------------
    if "titles" in locals():
        del titles
    gc.collect()
    print("Loading retrieval engine (corpus + dense 2.7 GB + BM25)...")
    first_stage_methods = [m for m in methods if not m.endswith("+rerank")]
    rerank_methods = [m for m in methods if m.endswith("+rerank")]
    needs_rerank = bool(rerank_methods)
    from medrag.retrieval.reranker import CrossEncoderReranker  # noqa: E402

    reranker = (
        CrossEncoderReranker(
            device=None if args.reranker_device == "auto" else args.reranker_device,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            quantize=args.reranker_quantize,
        )
        if needs_rerank
        else None
    )
    engine = RetrievalEngine(INDEX_DIR, reranker=reranker, load_dense=not args.no_dense)
    print("Engine loaded.")

    summary: Dict[str, Any] = {}
    per_query: List[Dict[str, Any]] = []
    if first_stage_methods:
        rep = evaluate(
            engine,
            eval_queries,
            methods=first_stage_methods,
            k_values=k_values,
            candidate_k=args.candidate_k,
            progress=not rerank_methods,
        )
        summary.update(rep["summary"])
        per_query.extend(rep["per_query"])

    # +rerank methods: candidate generation per query, then ONE macro-batched
    # cross-encoder pass across all queries (keeps all cores busy).
    for m in rerank_methods:
        print(f"Reranking method {m} ({len(eval_queries)} queries) in one macro-batch...")
        rows, msum = macro_rerank_eval(
            engine, reranker, eval_queries, m, k_values, args.candidate_k
        )
        summary[m] = msum
        per_query.extend(rows)

    # Production-style sequential latency: warm model, single query at a time.
    # (The macro-batched rerank above reports amortized throughput, which is
    # better than worst-case per-query latency for a live service.)
    latency_probe: Dict[str, Any] = {}
    if rerank_methods and args.latency_probes > 0:
        probe_queries = [q.query for q in eval_queries[: args.latency_probes]]
        for m in rerank_methods:
            base = m[: -len("+rerank")]
            engine.search(
                probe_queries[0], method=base, top_k=10,
                rerank=True, candidate_k=args.candidate_k,
            )  # warm-up
            times: List[float] = []
            for q in probe_queries:
                result = engine.search(
                    q, method=base, top_k=10,
                    rerank=True, candidate_k=args.candidate_k,
                )
                times.append(result.latency_ms or 0.0)
            latency_probe[m] = {
                "mean_ms": round(float(np.mean(times)), 1),
                "median_ms": round(float(np.median(times)), 1),
                "probes": len(times),
            }
            print(f"Single-query latency {m}: mean={latency_probe[m]['mean_ms']}ms "
                  f"median={latency_probe[m]['median_ms']}ms ({len(times)} probes)")

    # 5. Aggregations --------------------------------------------------------
    for row in per_query:
        row["kind"] = query_by_text.get(row["query"], {}).get("kind", "unknown")
        row["chunk_type"] = query_by_text.get(row["query"], {}).get("chunk_type", "unknown")

    by_kind = aggregate_rows(per_query, "kind", methods, k=min(10, max(k_values)))
    by_chunk_type = aggregate_rows(per_query, "chunk_type", methods, k=min(10, max(k_values)))

    worst: List[Dict[str, Any]] = []
    rerank_methods = [m for m in methods if m.endswith("+rerank")]
    if rerank_methods:
        rr = rerank_methods[0]
        for row in per_query:
            if row["method"] == rr and row["mrr"] == 0.0:
                worst.append({
                    "document_id": row["query"][:40],
                    "query": row["query"],
                    "recall@10": row.get("recall@10"),
                })

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": {
            "documents": len(pd.read_parquet(
                str(CORPUS_PATH), columns=["document_id"]
            )["document_id"].unique()),
            "eligible_chunks": int(pq.read_metadata(str(CORPUS_PATH)).num_rows),
        },
        "sample": {
            "documents": len(doc_ids),
            "queries": len(queries),
            "chunk_types": sample_analysis["chunk_types"],
            "mean_chunks_per_doc": sample_analysis["mean_chunks_per_doc"],
        },
        "query_mix": dict(query_mix),
        "index": {
            "dense_type": engine.dense.index_type if engine.dense is not None else None,
            "n_total": engine.dense.n_total if engine.dense is not None else 0,
            "bm25_docs": engine.sparse.n_docs,
            "k1": engine.sparse.k1,
            "b": engine.sparse.b,
            "fusion": engine.fusion,
        },
        "summary": summary,
        "by_kind": by_kind,
        "by_chunk_type": by_chunk_type,
        "latency_probe": latency_probe,
        "worst_queries": worst,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    report_path = out_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = out_dir / "benchmark_report.md"
    write_markdown_report(report, md_path)

    # 6. Console summary -----------------------------------------------------
    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY")
    print("=" * 78)
    dense_total = engine.dense.n_total if engine.dense is not None else 0
    dense_label = f"{dense_total:,} chunks (dense)" if engine.dense is not None else "dense index not loaded"
    print(f"Corpus: {dense_label} | BM25: {engine.sparse.n_docs:,} chunks | "
          f"Sample: {len(doc_ids)} docs, {len(queries)} queries")
    print()
    print(f"{'Method':<20} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'R@50':>7} {'MRR':>7} {'DocR@10':>8} {'ms':>7}")
    for method, m in report["summary"].items():
        print(
            f"{method:<20} {format_metric(m.get('recall@5')):>7} "
            f"{format_metric(m.get('recall@10')):>7} {format_metric(m.get('recall@20')):>7} "
            f"{format_metric(m.get('recall@50')):>7} {format_metric(m.get('mrr')):>7} "
            f"{format_metric(m.get('document_recall@10')):>8} {m.get('mean_latency_ms', 0):>6.1f}"
        )
    print()
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s")
    print(f"Reports: {report_path.resolve()}")
    print(f"         {md_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())