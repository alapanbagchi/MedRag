"""Chunking pipeline runner.

One entry point, one pipeline:

    parse -> AST validate -> chunk -> chunk validate -> save chunks

The pipeline is deterministic and resumable: documents whose chunk Parquet
already exists are skipped.

Usage:
    python -m medrag.pipeline --dir data/raw/cardiology --chunks chunks
    python -m medrag.pipeline --dir data/raw/cardiology --limit 10 --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from medrag.chunker import ASTChunker
from medrag.models import Chunk, Document, ValidationReport
from medrag.parser import PMCASTParser
from medrag.validators import ASTValidator, ChunkValidator


@dataclass
class PipelineResult:
    source: Path
    document: Optional[Document] = None
    chunks: List[Chunk] = field(default_factory=list)
    ast_report: Optional[ValidationReport] = None
    chunk_report: Optional[ValidationReport] = None
    chunking_report: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        if self.ast_report is None or self.chunk_report is None:
            return False
        return self.ast_report.passed and self.chunk_report.passed


class ChunkingPipeline:
    """Deterministic parse -> validate -> chunk -> validate pipeline."""

    def __init__(
        self,
        parser: Optional[PMCASTParser] = None,
        chunker: Optional[ASTChunker] = None,
        ast_validator: Optional[ASTValidator] = None,
        chunk_validator: Optional[ChunkValidator] = None,
        max_prose_chars: Optional[int] = None,
        hard_max_prose_chars: Optional[int] = None,
    ) -> None:
        self.parser = parser or PMCASTParser()
        if chunker is not None:
            self.chunker = chunker
        else:
            kwargs: Dict[str, int] = {}
            if max_prose_chars is not None:
                kwargs["max_prose_chars"] = max_prose_chars
            if hard_max_prose_chars is not None:
                kwargs["hard_max_prose_chars"] = hard_max_prose_chars
            self.chunker = ASTChunker(**kwargs)
        self.ast_validator = ast_validator or ASTValidator()
        self.chunk_validator = chunk_validator or ChunkValidator()

    def process_file(self, path: Path) -> PipelineResult:
        result = PipelineResult(source=path)
        try:
            document = self.parser.parse(path)
            result.document = document

            result.ast_report = self.ast_validator.validate(document)

            chunks, chunking_report = self.chunker.chunk_with_report(document)
            result.chunks = chunks
            result.chunking_report = chunking_report

            result.chunk_report = self.chunk_validator.validate(
                chunks,
                document_id=document.pmcid,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            result.error = f"{type(exc).__name__}: {exc}"

        return result

    def process_directory(self, directory: Path, limit: int = 0) -> List[PipelineResult]:
        files = sorted(directory.rglob("*.xml"))
        if limit > 0:
            files = files[:limit]
        return [self.process_file(path) for path in files]


# ======================================================================
# Statistics
# ======================================================================

def prose_length_stats(chunks: List[Chunk]) -> Dict[str, float]:
    lengths = sorted(len(c.text) for c in chunks if c.chunk_type == "paragraph")
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0,
                "median": 0, "p95": 0, "p99": 0}
    return {
        "count": len(lengths),
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(statistics.mean(lengths), 2),
        "median": round(statistics.median(lengths), 2),
        "p95": lengths[int((len(lengths) - 1) * 0.95)],
        "p99": lengths[int((len(lengths) - 1) * 0.99)],
    }


def chunk_type_distribution(chunks: List[Chunk]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for chunk in chunks:
        distribution[chunk.chunk_type] = distribution.get(chunk.chunk_type, 0) + 1
    return dict(sorted(distribution.items()))


def corpus_summary(results: List[PipelineResult]) -> Dict[str, Any]:
    total = len(results)
    chunks = [c for r in results for c in r.chunks]
    prose = [c for c in chunks if c.chunk_type == "paragraph"]
    return {
        "documents": total,
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "total_chunks": len(chunks),
        "chunks_by_type": chunk_type_distribution(chunks),
        "prose_stats": prose_length_stats(prose),
    }


def print_corpus_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("CORPUS SUMMARY")
    print("=" * 70)
    print(f"Documents:    {summary['documents']:,}")
    print(f"Passed:       {summary['passed']:,}")
    print(f"Failed:       {summary['failed']:,}")
    print(f"Total chunks: {summary['total_chunks']:,}")

    print("\nChunk distribution:")
    for ctype, count in summary["chunks_by_type"].items():
        print(f"  {ctype:<22} {count:,}")

    print("\nProse stats:")
    for key, value in summary["prose_stats"].items():
        print(f"  {key:<10} {value}")


# ======================================================================
# Chunk persistence
# ======================================================================

def chunk_output_path(result: PipelineResult, output_dir: Path) -> Path:
    return output_dir / f"{result.source.stem}.parquet"


def save_result_chunks(result: PipelineResult, output_dir: Path) -> Optional[Path]:
    """Write one document's chunks; return the path or ``None``."""
    if not result.chunks:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_output_path(result, output_dir)
    records = [asdict(chunk) for chunk in result.chunks]
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


def completed_chunk_stems(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()
    return {p.stem for p in output_dir.glob("*.parquet")}


# ======================================================================
# Multiprocessing workers
# ======================================================================

_PIPELINE: Optional[ChunkingPipeline] = None
_MAX_PROSE_CHARS: Optional[int] = None
_HARD_MAX_PROSE_CHARS: Optional[int] = None


def _init_worker() -> None:
    global _PIPELINE
    _PIPELINE = ChunkingPipeline(
        max_prose_chars=_MAX_PROSE_CHARS,
        hard_max_prose_chars=_HARD_MAX_PROSE_CHARS,
    )


def _process_one(path: Path) -> PipelineResult:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = ChunkingPipeline(
            max_prose_chars=_MAX_PROSE_CHARS,
            hard_max_prose_chars=_HARD_MAX_PROSE_CHARS,
        )
    return _PIPELINE.process_file(path)


def _collect_error_codes(result: PipelineResult) -> Counter:
    error_codes: Counter = Counter()
    if result.passed:
        return error_codes

    if result.error:
        error_codes[result.error.split(":", 1)[0]] += 1
        return error_codes

    if result.ast_report:
        for diagnostic in result.ast_report.diagnostics:
            if diagnostic.severity == "ERROR":
                error_codes["AST:" + diagnostic.code] += 1

    if result.chunk_report:
        for diagnostic in result.chunk_report.diagnostics:
            if diagnostic.severity == "ERROR":
                error_codes["CHUNK:" + diagnostic.code] += 1

    return error_codes


def _worker_failure_result(path: Path, exc: BaseException) -> PipelineResult:
    return PipelineResult(
        source=path,
        error=f"WORKER_ERROR: {type(exc).__name__}: {exc}",
        ast_report=None,
        chunk_report=None,
        chunks=[],
    )


# ======================================================================
# CLI
# ======================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk an XML corpus into retrieval chunks.")
    parser.add_argument("--dir", type=Path, default=Path("data/raw/cardiology"),
                        help="Directory containing PMC XML files.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum number of files to process. 0 = all.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes. 0 = automatic.")
    parser.add_argument("--max-prose-chars", type=int, default=None,
                        help="Target maximum prose chunk length in characters.")
    parser.add_argument("--hard-max-prose-chars", type=int, default=None,
                        help="Absolute maximum prose chunk length (single paragraphs above this are split).")
    parser.add_argument("--save-failed", action="store_true",
                        help="Write chunk files even for documents whose validation failed (default: skip).")
    parser.add_argument("--chunks", type=Path, default=Path("chunks"),
                        help="Output directory for per-document chunk files.")
    parser.add_argument("--json", type=Path, default=Path("chunking_report.json"),
                        help="Output JSON report path.")
    args = parser.parse_args(argv)

    if not args.dir.is_dir():
        print(f"Directory does not exist: {args.dir}", file=sys.stderr)
        return 1
    if args.limit < 0:
        print("--limit must be >= 0", file=sys.stderr)
        return 1
    if args.workers < 0:
        print("--workers must be >= 0", file=sys.stderr)
        return 1
    if args.max_prose_chars is not None and args.max_prose_chars <= 0:
        print("--max-prose-chars must be > 0", file=sys.stderr)
        return 1
    if (args.hard_max_prose_chars is not None
            and args.hard_max_prose_chars <= 0):
        print("--hard-max-prose-chars must be > 0", file=sys.stderr)
        return 1
    if (args.max_prose_chars is not None and args.hard_max_prose_chars is not None
            and args.hard_max_prose_chars < args.max_prose_chars):
        print("--hard-max-prose-chars must be >= --max-prose-chars", file=sys.stderr)
        return 1

    files = sorted(args.dir.rglob("*.xml"))
    if not files:
        print("No XML files found.")
        return 1

    completed_stems = completed_chunk_stems(args.chunks) if args.chunks else set()
    if completed_stems:
        files = [f for f in files if f.stem not in completed_stems]
        print(f"Skipping {len(completed_stems)} already-chunked document(s).")

    if args.limit > 0:
        files = files[:args.limit]

    if not files:
        print("No new XML files to process.")
        return 0

    total = len(files)
    workers = args.workers or max(1, (os.cpu_count() or 1) - 1)
    workers = min(workers, total)

    # Configure worker processes with the requested prose limits.
    global _MAX_PROSE_CHARS, _HARD_MAX_PROSE_CHARS
    _MAX_PROSE_CHARS = args.max_prose_chars
    _HARD_MAX_PROSE_CHARS = args.hard_max_prose_chars

    print(f"Processing {total:,} documents using {workers} worker(s)...")

    results: List[PipelineResult] = []
    error_codes: Counter = Counter()
    saved_count = 0
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        future_iterator = executor.map(_process_one, files, chunksize=1)
        with tqdm(future_iterator, total=total, desc="Chunking", unit="doc",
                  dynamic_ncols=True) as progress:
            for path, mapped_result in zip(files, progress):
                try:
                    result = mapped_result
                except Exception as exc:  # noqa: BLE001 - keep the run alive
                    result = _worker_failure_result(path, exc)

                results.append(result)
                # Only persist chunk files for documents that passed validation,
                # unless --save-failed is set: writing failed output would mark
                # the document as done and silently poison the next resumable run.
                if args.chunks and (args.save_failed or result.passed):
                    if save_result_chunks(result, args.chunks):
                        saved_count += 1
                error_codes.update(_collect_error_codes(result))

    elapsed = time.perf_counter() - start
    summary = corpus_summary(results)
    print_corpus_summary(summary)
    print(f"Elapsed:      {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Throughput:   {total / elapsed:.2f} docs/s")

    if error_codes:
        print("\nError codes:")
        for code, count in error_codes.most_common():
            print(f"  {code:<45} {count:,}")

    failed_documents = []
    for result in results:
        if result.passed:
            continue
        failed_documents.append({
            "file": result.source.name,
            "path": str(result.source),
            "error": result.error,
            "ast_errors": [
                d.code for d in (result.ast_report.diagnostics if result.ast_report else [])
                if d.severity == "ERROR"
            ],
            "chunk_errors": [
                d.code for d in (result.chunk_report.diagnostics if result.chunk_report else [])
                if d.severity == "ERROR"
            ],
        })

    payload = {
        "summary": summary,
        "performance": {
            "elapsed_seconds": elapsed,
            "documents_per_second": total / elapsed if elapsed > 0 else 0.0,
            "workers": workers,
        },
        "error_codes": dict(error_codes),
        "failed_documents": failed_documents,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {args.json.resolve()}")
    if args.chunks:
        print(f"Chunk files written: {saved_count} (in {args.chunks.resolve()})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
