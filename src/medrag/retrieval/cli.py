"""Command-line interface for the MedRAG retrieval layer.

Subcommands:
    build-index   build corpus/dense/sparse indexes (resumable)
    search        run a single query (optionally rerank)
    compare       qualitative before/after reranking display for queries
    evaluate      run the evaluation set and print metrics
    diagnose      run pre-build safety diagnostics only

Example::

    python -m medrag.retrieval search --query "diabetes mellitus treatment" \\
        --method hybrid --top-k 10 --candidate-k 50 --rerank --show-text

    python -m medrag.retrieval compare --query "diabetes mellitus treatment" \\
        "Care methods for people who have suffered cardiac arrest" --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from medrag.retrieval.corpus import discover_chunks, discover_embeddings, run_diagnostics, write_diagnostics_report
from medrag.retrieval.engine import RetrievalEngine
from medrag.retrieval.evaluation import DEFAULT_K_VALUES, DEFAULT_METHODS, evaluate, load_eval_queries, write_report
from medrag.retrieval.index_builder import build_all
from medrag.models import Candidate, RerankedCandidate, ScoredChunk
from medrag.retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker

RERANK_SUFFIX = "+rerank"


# ----------------------------------------------------------------------
# Reranker construction
# ----------------------------------------------------------------------

def _add_reranker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-breadcrumb", action="store_true",
                        help="rerank on plain passage text (omit section breadcrumb)")


def _build_reranker(args: argparse.Namespace) -> CrossEncoderReranker:
    device = None if args.reranker_device == "auto" else args.reranker_device
    return CrossEncoderReranker(
        model_name=args.reranker_model,
        device=device,
        batch_size=args.reranker_batch_size,
        include_breadcrumb=not args.no_breadcrumb,
    )


def _make_engine(args: argparse.Namespace, reranker=None) -> RetrievalEngine:
    """Construct a retrieval engine from parsed CLI args.

    Shared by the search/compare/evaluate commands so engine options (fusion,
    RRF k, per-branch weights) stay wired in one place.
    """
    weights = None
    if getattr(args, "weight_dense", None) is not None or getattr(
        args, "weight_sparse", None
    ) is not None:
        dense_w = getattr(args, "weight_dense", None)
        sparse_w = getattr(args, "weight_sparse", None)
        weights = [
            dense_w if dense_w is not None else 1.0,
            sparse_w if sparse_w is not None else 1.0,
        ]
    return RetrievalEngine(
        Path(args.index_dir),
        fusion=args.fusion,
        rrf_k=args.rrf_k,
        weights=weights,
        reranker=reranker,
    )


# ----------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------

def _preview(text: Optional[str], n: int = 5000) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def _unique_docs(hits: List) -> int:
    return len({h.document_id for h in hits if getattr(h, "document_id", None)})


def _print_hit(hit, show_text: bool, reranked: bool) -> None:
    if reranked and isinstance(hit, RerankedCandidate):
        line = (
            f"{hit.reranked_rank:>3}. [rerank={hit.reranker_score:.5f}] "
            f"(orig_rank={hit.original_rank}, orig={hit.original_score:.5f}) "
            f"{hit.chunk_id} doc={hit.document_id} type={hit.chunk_type}"
        )
    else:
        line = (
            f"{hit.rank:>3}. [{hit.score:.5f}] {hit.chunk_id} "
            f"doc={hit.document_id} type={hit.chunk_type}"
        )
    print(line)
    if getattr(hit, "breadcrumb", None):
        print(f"      crumb={hit.breadcrumb}")
    if show_text and getattr(hit, "text", None):
        print(f"      {_preview(hit.text)}")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def _cmd_build_index(args: argparse.Namespace) -> int:
    report = build_all(
        chunks_dir=Path(args.chunks_dir),
        embeddings_dir=Path(args.embeddings_dir),
        index_dir=Path(args.index_dir),
        dense_type=args.dense_type,
        text_field=args.text_field,
        limit=args.limit or None,
        force=args.force,
        run_full_diagnostics=not args.no_diagnostics,
        k1=args.k1,
        b=args.b,
        use_pgvector=getattr(args, "pgvector", False),
    )
    print(json.dumps(report, indent=2))
    return 0


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fusion", choices=("rrf", "minmax"), default="rrf")
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--weight-dense", type=float, default=None,
                        help="RRF/minmax weight for the dense branch (default 1.0)")
    parser.add_argument("--weight-sparse", type=float, default=None,
                        help="RRF/minmax weight for the sparse branch (default 1.0)")


def _cmd_search(args: argparse.Namespace) -> int:
    reranker = _build_reranker(args) if args.rerank else None
    engine = _make_engine(args, reranker)
    result = engine.search(
        args.query,
        method=args.method,
        top_k=args.top_k,
        include_text=args.show_text,
        rerank=args.rerank,
        candidate_k=args.candidate_k,
    )

    print(f"query: {result.query}")
    print(f"method: {result.method}")
    if result.query_encode_ms is not None:
        print(f"query encoding: {result.query_encode_ms:.2f} ms")
    if result.reranked:
        print(f"candidates: {result.candidate_count}")
        print(f"rerank latency: {result.rerank_latency_ms:.2f} ms")
    print(f"latency: {result.latency_ms:.2f} ms")
    print("-" * 90)

    if result.reranked and result.candidates:
        before = result.candidates[: args.top_k]
        print("BEFORE RERANKING")
        print("----------------")
        for c in before:
            _print_hit(_candidate_to_scored(c), args.show_text, reranked=False)
        print()
        print("AFTER RERANKING")
        print("---------------")
        for hit in result.results:
            _print_hit(hit, args.show_text, reranked=True)
    else:
        for hit in result.results:
            _print_hit(hit, args.show_text, reranked=False)
    return 0


def _candidate_to_scored(c: Candidate) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=c.chunk_id,
        score=c.original_score,
        retrieval_method=c.retrieval_method,
        rank=c.original_rank,
        document_id=c.document_id,
        chunk_type=c.chunk_type,
        breadcrumb=c.breadcrumb,
        text=c.text,
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    queries: List[str] = []
    if args.query:
        queries = list(args.query)
    elif args.eval_file:
        queries = [q.query for q in load_eval_queries(Path(args.eval_file))]
    else:
        print("ERROR: provide --query or --eval-file", file=sys.stderr)
        return 2

    reranker = _build_reranker(args)
    engine = _make_engine(args, reranker)

    for query in queries:
        before = engine.search(
            query, method=args.method, top_k=args.top_k, include_text=True
        )
        after = engine.search(
            query, method=args.method, top_k=args.top_k, include_text=True,
            rerank=True, candidate_k=args.candidate_k,
        )

        print("=" * 90)
        print(f"QUERY: {query}")
        print("=" * 90)

        print(f"\nBEFORE RERANKING  (unique docs in top {args.top_k}: {_unique_docs(before.results)})")
        print("-" * 90)
        for hit in before.results:
            _print_hit(hit, show_text=True, reranked=False)

        print(f"\nAFTER RERANKING   (unique docs in top {args.top_k}: {_unique_docs(after.results)})")
        print("-" * 90)
        for hit in after.results:
            _print_hit(hit, show_text=True, reranked=True)

        print(f"\nlatency: before={before.latency_ms:.2f} ms  "
              f"rerank={after.rerank_latency_ms:.2f} ms  "
              f"total_after={after.latency_ms:.2f} ms")
        print()
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    queries = load_eval_queries(Path(args.eval_file))
    print(f"Loaded {len(queries)} queries from {args.eval_file}")

    methods = args.methods.split(",") if args.methods else list(DEFAULT_METHODS)
    needs_reranker = any(m.endswith(RERANK_SUFFIX) for m in methods)
    reranker = _build_reranker(args) if needs_reranker else None

    engine = _make_engine(args, reranker)

    report = evaluate(engine, queries, methods=methods, k_values=args.k, candidate_k=args.candidate_k)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for method, m in report["summary"].items():
        print(f"\n{method}")
        for key in ("recall@5", "recall@10", "recall@20", "recall@50", "mrr",
                    "mean_latency_ms", "mean_rerank_latency_ms", "candidate_count"):
            if key in m:
                print(f"    {key:<24} {m[key]}")

    if args.report_json:
        write_report(report, Path(args.report_json))
        print(f"\nFull report written to {args.report_json}")
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    chunks = discover_chunks(Path(args.chunks_dir))
    embs = discover_embeddings(Path(args.embeddings_dir))
    if args.limit:
        chunks = chunks[: args.limit]
        embs = embs[: args.limit]
    diag = run_diagnostics(chunks, embs)
    print(json.dumps({k: v for k, v in diag.items() if k != "problems"}, indent=2))
    if args.index_dir:
        write_diagnostics_report(diag, Path(args.index_dir) / "diagnostics.json")
        print(f"\nDiagnostics written to {args.index_dir}/diagnostics.json")
    return 0


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="build corpus/dense/sparse indexes")
    p_build.add_argument("--chunks-dir", default="chunks")
    p_build.add_argument("--embeddings-dir", default="embeddings")
    p_build.add_argument("--index-dir", default="index")
    p_build.add_argument("--dense-type", choices=("flat", "hnsw"), default="flat")
    p_build.add_argument("--text-field", choices=("text", "embedding_text"), default="text")
    p_build.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter (default 1.5)")
    p_build.add_argument("--b", type=float, default=0.75, help="BM25 b parameter (default 0.75)")
    p_build.add_argument("--limit", type=int, default=0, help="max documents (0 = all)")
    p_build.add_argument("--force", action="store_true", help="rebuild existing artifacts")
    p_build.add_argument("--no-diagnostics", action="store_true", help="skip pre-build diagnostics")
    p_build.add_argument("--pgvector", action="store_true",
                        help="use PostgreSQL + pgvector for dense index (requires running PostgreSQL)")
    p_build.set_defaults(func=_cmd_build_index)

    p_search = sub.add_parser("search", help="run a single query")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--method", choices=("dense", "bm25", "sparse", "hybrid"), default="hybrid")
    p_search.add_argument("--top-k", type=int, default=20)
    p_search.add_argument("--candidate-k", type=int, default=100)
    p_search.add_argument("--index-dir", default="index")
    p_search.add_argument("--show-text", action="store_true")
    _add_query_args(p_search)
    rerank_group = p_search.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", action="store_true", dest="rerank",
                              help="apply the second-stage reranker")
    rerank_group.add_argument("--no-rerank", action="store_false", dest="rerank",
                              help="disable reranking (default)")
    p_search.set_defaults(rerank=False)
    _add_reranker_args(p_search)
    p_search.set_defaults(func=_cmd_search)

    p_compare = sub.add_parser("compare", help="before/after reranking display for queries")
    p_compare.add_argument("--query", nargs="+", help="one or more query strings")
    p_compare.add_argument("--eval-file", default=None, help="or load queries from an eval file")
    p_compare.add_argument("--method", choices=("dense", "bm25", "sparse", "hybrid"), default="hybrid")
    p_compare.add_argument("--top-k", type=int, default=10)
    p_compare.add_argument("--candidate-k", type=int, default=50)
    p_compare.add_argument("--index-dir", default="index")
    _add_query_args(p_compare)
    _add_reranker_args(p_compare)
    p_compare.set_defaults(func=_cmd_compare)

    p_eval = sub.add_parser("evaluate", help="evaluate against a labelled query set")
    p_eval.add_argument("--eval-file", default="eval/queries.json")
    p_eval.add_argument("--methods", default=None,
                        help="comma separated, e.g. hybrid,hybrid+rerank")
    p_eval.add_argument("--index-dir", default="index")
    p_eval.add_argument("--report-json", default="eval/report.json")
    _add_query_args(p_eval)
    p_eval.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    p_eval.add_argument("--candidate-k", type=int, default=100)
    _add_reranker_args(p_eval)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_diag = sub.add_parser("diagnose", help="run pre-build safety diagnostics")
    p_diag.add_argument("--chunks-dir", default="chunks")
    p_diag.add_argument("--embeddings-dir", default="embeddings")
    p_diag.add_argument("--index-dir", default="index")
    p_diag.add_argument("--limit", type=int, default=0)
    p_diag.set_defaults(func=_cmd_diagnose)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
