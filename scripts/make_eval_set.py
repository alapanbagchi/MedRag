"""Generate a small, manually-grounded evaluation set.

The ground truth is NOT fabricated: each query is the text of a real
retrieval-eligible chunk (verbatim, or the first 60 words). The relevant
answer is therefore known exactly -- the source chunk itself (and its
document). These "self-retrieval" queries verify that each retrieval path
can find a chunk from its own text, and provide exact recall@k / MRR
numbers. They do not measure semantic generalization; add human-labelled
queries to ``eval/queries.json`` for that.

Usage:
    python scripts/make_eval_set.py --corpus index/corpus.parquet \\
        --out eval/queries.json --n 25 --seed 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def first_n_words(text: str, n: int = 60) -> str:
    words = text.split()
    return " ".join(words[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="index/corpus.parquet")
    ap.add_argument("--out", default="eval/queries.json")
    ap.add_argument("--n", type=int, default=25, help="number of source chunks to sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunk-types", default="paragraph,list")
    args = ap.parse_args()

    import pandas as pd

    types = [t.strip() for t in args.chunk_types.split(",") if t.strip()]
    df = pd.read_parquet(
        args.corpus,
        columns=["id", "document_id", "text", "chunk_type"],
    )
    df = df[df["chunk_type"].isin(types)]
    df = df[df["text"].str.strip().str.len() > 40]

    # Sample distinct documents, then pick their longest eligible prose chunk.
    docs = df["document_id"].drop_duplicates().sample(n=args.n, random_state=args.seed).tolist()
    sampled = []
    for doc in docs:
        sub = df[df["document_id"] == doc]
        row = sub.loc[sub["text"].str.len().idxmax()]
        sampled.append((row["id"], row["document_id"], row["text"], row["chunk_type"]))

    queries = []
    for chunk_id, doc_id, text, ctype in sampled:
        queries.append({
            "id": f"{chunk_id}__verbatim",
            "query": text,
            "relevant_chunk_ids": [chunk_id],
            "relevant_document_ids": [doc_id],
            "source_chunk_type": ctype,
            "variant": "verbatim",
        })
        queries.append({
            "id": f"{chunk_id}__truncated60",
            "query": first_n_words(text, 60),
            "relevant_chunk_ids": [chunk_id],
            "relevant_document_ids": [doc_id],
            "source_chunk_type": ctype,
            "variant": "truncated60",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Self-retrieval evaluation set. Each query is a real chunk's text "
            "(verbatim or first 60 words); the relevant answer is that chunk "
            "and its document. Exact ground truth, not human-labelled."
        ),
        "queries": queries,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(queries)} queries to {out}")


if __name__ == "__main__":
    main()
