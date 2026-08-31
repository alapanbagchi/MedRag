"""MedCPT article embedding runner.

Generates per-chunk document embeddings for the retrieval-eligible chunks in
each chunk Parquet file.

This is the local, batched version of the MedCPT article-encoder pipeline:
documents are encoded with ``ncbi/MedCPT-Article-Encoder`` using the [CLS]
last hidden state (``last_hidden_state[:, 0, :]``). The chunker's
``embedding_text`` is the primary input; ``text`` is used as a fallback.

Outputs are written one file per source document:

    chunks/PMC123.parquet -> embeddings/PMC123.embeddings.parquet

Runs are resumable via ``embedding_manifest.json`` and atomic temp-file
promotion. Inference is single-process/GPU-bound, so this runner intentionally
does not parallelize across files.

Usage:
    python -m medrag.embedding --input-dir chunks --output-dir embeddings
    python -m medrag.embedding --input-dir chunks --output-dir embeddings --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src._torch import resolve_device

DEFAULT_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512
EMBEDDING_DIM = 768
OUTPUT_SUFFIX = ".embeddings.parquet"
TMP_SUFFIX = ".tmp.parquet"


@dataclass
class EmbeddingConfig:
    input_dir: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    embedding_dim: int = EMBEDDING_DIM
    limit: int = 0
    device: Optional[str] = None

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "embedding_manifest.json"


class EmbeddingRunner:
    """Batched MedCPT article encoder with resumable per-file output."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.tokenizer = None
        self.model = None
        self.manifest: Dict[str, Any] = {}

    # ==================================================================
    # Model
    # ==================================================================

    def load_model(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        print("\u2713 Tokenizer loaded")

        print("\nLoading MedCPT...")
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("\u2713 MedCPT loaded")

    # ==================================================================
    # Manifest
    # ==================================================================

    def load_manifest(self) -> None:
        path = self.config.manifest_path
        if not path.exists():
            self.manifest = {}
            return
        try:
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - manifest is best-effort resume state
            print("WARNING: Manifest could not be loaded.")
            self.manifest = {}

    def save_manifest(self) -> None:
        path = self.config.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)

    # ==================================================================
    # Paths
    # ==================================================================

    def output_path_for(self, input_path: Path) -> Path:
        relative = input_path.relative_to(self.config.input_dir)
        return (
            self.config.output_dir
            / relative.parent
            / (relative.stem + OUTPUT_SUFFIX)
        )

    # ==================================================================
    # Input preparation
    # ==================================================================

    @staticmethod
    def prepare_article(row: Any) -> List[str]:
        breadcrumb = row.get("breadcrumb", "")

        if isinstance(breadcrumb, (list, tuple, np.ndarray)):
            breadcrumb = " > ".join(str(x) for x in breadcrumb if x is not None)
        elif breadcrumb is None:
            breadcrumb = ""
        else:
            breadcrumb = str(breadcrumb)

        breadcrumb = breadcrumb.strip()

        embedding_text = row.get("embedding_text", None)
        if embedding_text is None:
            embedding_text = row.get("text", "")
        if embedding_text is None:
            embedding_text = ""

        embedding_text = str(embedding_text).strip()

        if not breadcrumb:
            breadcrumb = str(row.get("chunk_type", "PMC chunk"))

        return [breadcrumb, embedding_text]

    # ==================================================================
    # Embedding
    # ==================================================================

    def embed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        import torch

        eligible = df[
            df["retrieval_eligible"].fillna(False).astype(bool)
        ].copy()

        if len(eligible) == 0:
            return pd.DataFrame(columns=["chunk_id", "embedding"])

        articles = [self.prepare_article(row) for _, row in eligible.iterrows()]

        all_embeddings: List[np.ndarray] = []
        truncated_total = 0
        for start in range(0, len(articles), self.config.batch_size):
            batch = articles[start : start + self.config.batch_size]

            encoded = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.config.max_length,
            )
            # actual token lengths (attention mask sums, pre-padding):
            # lengths == max_length were really truncated by the encoder.
            lengths = encoded["attention_mask"].sum(dim=1)
            truncated_total += int((lengths >= self.config.max_length).sum().item())
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = self.model(**encoded)
                vectors = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(vectors.cpu().numpy())

        vectors = np.concatenate(all_embeddings, axis=0)

        if vectors.shape != (len(eligible), self.config.embedding_dim):
            raise RuntimeError(f"Unexpected shape: {vectors.shape}")

        if not np.isfinite(vectors).all():
            raise RuntimeError("Embeddings contain NaN/Inf.")

        return pd.DataFrame({
            "chunk_id": eligible["id"].values,
            "embedding": list(vectors),
        }), truncated_total

    # ==================================================================
    # Validation
    # ==================================================================

    def validate_output(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
        eligible = input_df["retrieval_eligible"].fillna(False).astype(bool)
        expected_ids = set(input_df.loc[eligible, "id"])
        actual_ids = set(output_df["chunk_id"])

        if len(output_df) != len(expected_ids):
            raise RuntimeError(
                f"Count mismatch: expected {len(expected_ids)}, got {len(output_df)}"
            )

        if expected_ids != actual_ids:
            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids
            raise RuntimeError(
                f"ID mismatch: missing={len(missing)}, extra={len(extra)}"
            )

        if len(output_df):
            dimensions = {len(x) for x in output_df["embedding"]}
            if dimensions != {self.config.embedding_dim}:
                raise RuntimeError(f"Invalid dimensions: {dimensions}")

            if not all(np.isfinite(x).all() for x in output_df["embedding"]):
                raise RuntimeError("Embedding contains NaN/Inf.")

    # ==================================================================
    # Per-file processing
    # ==================================================================

    def process_file(self, input_path: Path) -> Dict[str, Any]:
        output_path = self.output_path_for(input_path)
        key = str(input_path.relative_to(self.config.input_dir))

        # Resume by file existence (same policy as the notebook).
        if output_path.exists():
            return {"status": "skipped", "file": key}

        start = time.perf_counter()

        try:
            df = pd.read_parquet(input_path)

            eligible_count = int(
                df["retrieval_eligible"].fillna(False).astype(bool).sum()
            )

            result, truncated = self.embed_dataframe(df)
            self.validate_output(df, result)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_name(output_path.stem + TMP_SUFFIX)
            result.to_parquet(temp_path, index=False)
            temp_path.replace(output_path)

            elapsed = time.perf_counter() - start

            self.manifest[key] = {
                "status": "completed",
                "input": str(input_path),
                "output": str(output_path),
                "input_chunks": len(df),
                "eligible_chunks": eligible_count,
                "embedded_chunks": len(result),
                "truncated_at_max_length": truncated,
                "dimension": self.config.embedding_dim,
                "elapsed_seconds": elapsed,
            }
            self.save_manifest()

            return {
                "status": "completed",
                "file": key,
                "input_chunks": len(df),
                "embedded_chunks": len(result),
                "truncated": truncated,
                "elapsed": elapsed,
            }

        except Exception as exc:  # noqa: BLE001 - continue past per-file failures
            self.manifest[key] = {
                "status": "failed",
                "input": str(input_path),
                "output": str(output_path),
                "error": str(exc),
            }
            self.save_manifest()

            return {"status": "failed", "file": key, "error": str(exc)}

    # ==================================================================
    # Run
    # ==================================================================

    def run(self) -> Dict[str, Any]:
        import torch

        print("=" * 70)
        print("MEDCPT EMBEDDING RUNNER")
        print("=" * 70)
        print("Input :", self.config.input_dir)
        print("Output:", self.config.output_dir)
        print("Device:", self.device)

        if self.device == "cuda":
            print("GPU   :", torch.cuda.get_device_name(0))
            print(
                "VRAM  :",
                round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                "GB",
            )

        print("Batch :", self.config.batch_size)
        print("Length:", self.config.max_length)
        print("=" * 70)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_manifest()
        self.load_model()

        parquet_files = sorted(self.config.input_dir.rglob("*.parquet"))
        if self.config.limit > 0:
            parquet_files = parquet_files[: self.config.limit]

        if not parquet_files:
            raise RuntimeError(f"No Parquet files found in {self.config.input_dir}")

        print(f"\nFound {len(parquet_files):,} Parquet files")

        completed = 0
        skipped = 0
        failed = 0
        total_input_chunks = 0
        total_embedded_chunks = 0
        total_truncated = 0
        failed_files: List[Dict[str, str]] = []

        run_start = time.perf_counter()

        with tqdm(
            parquet_files,
            total=len(parquet_files),
            desc="Embedding",
            unit="file",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as progress:
            for input_path in progress:
                result = self.process_file(input_path)
                status = result["status"]

                if status == "skipped":
                    skipped += 1
                elif status == "completed":
                    completed += 1
                    total_input_chunks += result["input_chunks"]
                    total_embedded_chunks += result["embedded_chunks"]
                    total_truncated += result.get("truncated", 0)
                else:
                    failed += 1
                    failed_files.append({
                        "file": result["file"],
                        "error": result["error"],
                    })

                progress.set_postfix(
                    done=completed,
                    skipped=skipped,
                    failed=failed,
                    chunks=f"{total_embedded_chunks:,}",
                )

        elapsed = time.perf_counter() - run_start

        print()
        print("=" * 70)
        print("EMBEDDING COMPLETE")
        print("=" * 70)
        print(f"Files:             {len(parquet_files):,}")
        print(f"Completed:         {completed:,}")
        print(f"Skipped:           {skipped:,}")
        print(f"Failed:            {failed:,}")
        print(f"Truncated @{self.config.max_length}: {total_truncated:,} "
              f"({'' if not total_embedded_chunks else round(100*total_truncated/max(1,total_embedded_chunks),1)}%)")
        print(f"Input chunks:      {total_input_chunks:,}")
        print(f"Embeddings:        {total_embedded_chunks:,}")
        print(f"Elapsed:           {elapsed / 60:.2f} minutes")
        if elapsed > 0:
            print(f"Throughput:        {total_embedded_chunks / elapsed:.2f} embeddings/sec")
        print(f"Output:            {self.config.output_dir}")
        print(f"Manifest:          {self.config.manifest_path}")

        if failed_files:
            print()
            print(f"FAILED: {len(failed_files):,}")
            for item in failed_files[:50]:
                print(f"{item['file']}: {item['error']}")

        return {
            "files": len(parquet_files),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "input_chunks": total_input_chunks,
            "embedded_chunks": total_embedded_chunks,
            "elapsed_seconds": elapsed,
            "failed_files": failed_files,
        }


# ======================================================================
# CLI
# ======================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate MedCPT article embeddings for chunk Parquets.")
    parser.add_argument("--input-dir", default="chunks",
                        help="Directory containing chunk Parquet files (default: chunks).")
    parser.add_argument("--output-dir", default="embeddings",
                        help="Directory for embedding Parquet files (default: embeddings).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="MedCPT article encoder model name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Tokenizer max length.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum number of files to process. 0 = all.")
    parser.add_argument("--device", default=None, choices=("auto", "cuda", "cpu"),
                        help="Device override (default: auto).")
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 1
    if args.max_length < 1:
        print("--max-length must be >= 1", file=sys.stderr)
        return 1
    if args.limit < 0:
        print("--limit must be >= 0", file=sys.stderr)
        return 1

    device = None if args.device in (None, "auto") else args.device
    config = EmbeddingConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
        device=device,
    )

    try:
        EmbeddingRunner(config).run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
