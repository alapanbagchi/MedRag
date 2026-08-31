"""Build the SPLADE index from the local corpus.

Usage:
    python scripts/build_splade_index.py                    # full corpus
    python scripts/build_splade_index.py --max-docs 2000    # quick test
    python scripts/build_splade_index.py --out-dir index/splade
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# HF cache is read-only at ~/.cache; redirect to a writable project dir.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SPLADE index from corpus")
    parser.add_argument("--out-dir", default="index/splade", help="output dir")
    parser.add_argument("--max-docs", type=int, default=None, help="limit docs (for testing)")
    parser.add_argument("--batch-size", type=int, default=4, help="encode batch size")
    parser.add_argument("--model", default="NeuML/pubmedbert-base-splade", help="SPLADE model")
    parser.add_argument("--index-dir", default="index", help="corpus.parquet location")
    args = parser.parse_args()

    from src.config import AppConfig
    from src.retrieval.corpus import CorpusIndex
    from src.retrieval.splade import SPLADEIndex

    cfg = AppConfig()
    corpus = CorpusIndex(cfg.corpus_path)
    print(f"corpus: {corpus.n_chunks} chunks")

    idx = SPLADEIndex.build_from_corpus(
        corpus,
        Path(args.out_dir),
        model_name=args.model,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
    )
    print(f"\nSaved SPLADE index to {args.out_dir}")
    print(f"  docs        : {idx.n_docs}")
    print(f"  vocab size  : {idx.vocab_size}")
    print(f"  non-zeros   : {idx.doc_matrix.nnz}")
    print(f"  sparsity    : {100.0 * idx.doc_matrix.nnz / (idx.n_docs * idx.vocab_size):.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
