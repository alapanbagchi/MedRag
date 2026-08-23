#!/usr/bin/env python3
"""
Synthetic RAG Evaluation Dataset Generator
==========================================
Uses Ragas + local Ollama LLM to generate a synthetic Q&A benchmark
from pre-chunked PMC documents stored as Parquet files.

Usage:
    # Test with 1 document:
    python generate_rag_dataset.py --test

    # Test with 1 specific file:
    python generate_rag_dataset.py --parquet chunks/PMC10327125.4.parquet

    # Process all 14k+ parquet files in chunks/ (20 docs per batch):
    python generate_rag_dataset.py

    # Custom settings:
    python generate_rag_dataset.py \
        --parquet chunks/ \
        --model qwen3:8b \
        --embedding-model BAAI/bge-base-en-v1.5 \
        --batch-size 10 \
        --questions-per-batch 50 \
        --output-dir ./question_dataset

Requires:
    pip install ragas langchain-core langchain-ollama langchain-huggingface \
                sentence-transformers pyarrow pandas requests
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_gen")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic RAG evaluation data using Ragas + Ollama"
    )
    p.add_argument(
        "--parquet", default="chunks/",
        help="Path to a single .parquet file OR a directory of .parquet files "
             "(default: chunks/)"
    )
    p.add_argument("--model", default="qwen3:8b", help="Ollama model (default: qwen3:8b)")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5",
                    help="HuggingFace embedding model")
    p.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for embeddings (default: cpu)")
    p.add_argument("--batch-size", type=int, default=20,
                    help="Documents per batch (default: 20)")
    p.add_argument("--questions-per-batch", type=int, default=100,
                    help="Max questions to generate per batch (default: 100)")
    p.add_argument("--output-dir", default="question_dataset",
                    help="Output directory (default: question_dataset)")
    p.add_argument("--test", action="store_true",
                    help="Test mode: process only 1 document with 5 questions")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Ollama setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_ollama(model: str, base_url: str):
    """Ensure Ollama is installed, running, and has the model pulled."""
    import shutil

    # Install if missing
    if shutil.which("ollama") is None:
        log.info("Installing Ollama...")
        resp = subprocess.run(
            ["curl", "-fsSL", "https://ollama.com/install.sh"],
            capture_output=True, text=True,
        )
        if resp.returncode != 0:
            raise RuntimeError(f"Failed to download Ollama: {resp.stderr[:300]}")
        install = subprocess.run(["sh", "-c", resp.stdout], capture_output=True, text=True)
        if install.returncode != 0:
            raise RuntimeError(f"Ollama install failed: {install.stderr[:300]}")
        log.info("Ollama installed.")

    # Start server if not running
    try:
        import requests
        requests.get(f"{base_url}/api/tags", timeout=3)
    except Exception:
        log.info("Starting Ollama server...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        for _ in range(30):
            time.sleep(1)
            try:
                import requests
                requests.get(f"{base_url}/api/tags", timeout=3)
                log.info("Ollama server ready.")
                break
            except Exception:
                pass
        else:
            raise RuntimeError("Ollama server did not start within 30s.")

    # Pull model
    import requests
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    available = [m["name"] for m in resp.json().get("models", [])]
    if not any(model in m for m in available):
        log.info(f"Pulling model '{model}' (may take a while)...")
        subprocess.run(["ollama", "pull", model], check=True)
        log.info(f"Model '{model}' ready.")
    else:
        log.info(f"Model '{model}' already available.")


# ──────────────────────────────────────────────────────────────────────────────
# Parquet loading
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "id", "document_id", "text", "embedding_text",
    "chunk_type", "section", "subsection", "breadcrumb",
    "document_position", "table_id", "figure_id", "row_label",
    "metadata",
]

OPTIONAL_COLS = [
    "source_block_ids", "parent_id", "object_id",
    "citation_refs", "retrieval_eligible",
    "embedding_token_count", "chunk_version", "reference_id",
]


def load_parquet(path: str) -> pd.DataFrame:
    """Load a single parquet file into a DataFrame with safe defaults."""
    table = pq.read_table(path)
    available = [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in table.column_names]
    table = pq.read_table(path, columns=available)
    df = table.to_pandas()

    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))] if col in ("breadcrumb",) else None

    # Normalise metadata
    def safe_meta(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return {}

    df["metadata"] = df["metadata"].apply(safe_meta)
    for c in ("breadcrumb", "source_block_ids", "citation_refs"):
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x if isinstance(x, list) else [])

    return df


def discover_parquets(path: str) -> List[str]:
    """Return list of .parquet file paths from a file or directory."""
    p = Path(path)
    if p.is_file() and p.suffix == ".parquet":
        return [str(p)]
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        return [str(f) for f in files]
    raise FileNotFoundError(f"Not a parquet file or directory: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# LangChain Document conversion
# ──────────────────────────────────────────────────────────────────────────────

def chunks_to_lc_docs(df: pd.DataFrame):
    """Convert DataFrame rows to LangChain Documents with full metadata."""
    from langchain_core.documents import Document

    docs = []
    for _, row in df.iterrows():
        meta = {
            "chunk_id": str(row.get("id", "")),
            "document_id": str(row.get("document_id", "")),
            "chunk_type": str(row.get("chunk_type", "")),
            "section": str(row.get("section", "")),
            "subsection": str(row.get("subsection") or ""),
            "breadcrumb": row.get("breadcrumb", []) or [],
            "document_position": int(row.get("document_position") or 0),
            "table_id": str(row.get("table_id") or ""),
            "figure_id": str(row.get("figure_id") or ""),
            "row_label": str(row.get("row_label") or ""),
            "source_block_ids": row.get("source_block_ids", []) or [],
            "retrieval_eligible": bool(row.get("retrieval_eligible", True)),
        }
        doc_meta = row.get("metadata", {})
        if isinstance(doc_meta, dict):
            for k in ("pmcid", "title", "journal", "doi", "publication_date"):
                meta[k] = doc_meta.get(k, "")

        text = row.get("embedding_text", "") or row.get("text", "") or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata=meta))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Provenance validation
# ──────────────────────────────────────────────────────────────────────────────

def _norm_ws(text: str) -> str:
    """Normalise whitespace for fuzzy matching."""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def build_provenance_index(df: pd.DataFrame):
    """Build exact + normalised MD5 indices of all chunk texts."""
    exact = {}
    normalised = {}
    for _, row in df.iterrows():
        chunk_id = str(row["id"])
        doc_id = str(row["document_id"])
        info = {
            "chunk_id": chunk_id, "document_id": doc_id,
            "chunk_type": str(row.get("chunk_type", "")),
            "section": str(row.get("section", "")),
        }
        for txt in (row.get("text", ""), row.get("embedding_text", "")):
            if not txt or not str(txt).strip():
                continue
            txt = str(txt)
            h = hashlib.md5(txt.encode()).hexdigest()
            exact.setdefault(h, info)
            nh = hashlib.md5(_norm_ws(txt).encode()).hexdigest()
            # Store text in normalised index for substring matching
            norm_info = {**info, "text": txt}
            normalised.setdefault(nh, norm_info)
    return exact, normalised


def validate_provenance(contexts, exact_idx, norm_idx):
    """Map each Ragas reference_context to its source chunk."""
    results = []
    for ctx in (contexts or []):
        if not ctx or not ctx.strip():
            continue
        ctx = ctx.strip()
        h = hashlib.md5(ctx.encode()).hexdigest()

        # 1. Exact
        if h in exact_idx:
            m = exact_idx[h]
            results.append({**m, "match_type": "exact"})
            continue

        # 2. Normalised whitespace
        nh = hashlib.md5(_norm_ws(ctx).encode()).hexdigest()
        if nh in norm_idx:
            m = norm_idx[nh]
            results.append({**m, "match_type": "normalised_whitespace"})
            continue

        # 3. Substring containment (context is part of a chunk)
        found = False
        for nkey, minfo in norm_idx.items():
            chunk_norm = _norm_ws(minfo.get("text", ""))
            if len(ctx) > 50 and _norm_ws(ctx) in chunk_norm:
                results.append({**minfo, "match_type": "substring_contained"})
                found = True
                break
        if found:
            continue

        # 4. Chunk is part of context
        for nkey, minfo in norm_idx.items():
            chunk_norm = _norm_ws(minfo.get("text", ""))
            if len(chunk_norm) > 50 and chunk_norm in _norm_ws(ctx):
                results.append({**minfo, "match_type": "chunk_in_context"})
                found = True
                break
        if found:
            continue

        results.append({"chunk_id": "", "document_id": "", "match_type": "unresolved"})

    exact_count = sum(1 for r in results if r["match_type"] == "exact")
    norm_count = sum(1 for r in results if r["match_type"] != "exact" and r["match_type"] != "unresolved")
    unresolved = sum(1 for r in results if r["match_type"] == "unresolved")
    return results, exact_count, norm_count, unresolved


# ──────────────────────────────────────────────────────────────────────────────
# Core: generate for one batch
# ──────────────────────────────────────────────────────────────────────────────

def generate_batch(
    lc_docs: list,
    doc_ids: list,
    ragas_llm,
    ragas_emb,
    exact_idx: dict,
    norm_idx: dict,
    batch_num: int,
    total_batches: int,
    max_questions: int,
) -> Tuple[pd.DataFrame, dict]:
    """Run Ragas generate_with_chunks on one batch of documents."""

    from ragas.testset import TestsetGenerator

    t0 = time.time()
    valid = [d for d in lc_docs if d.page_content and len(d.page_content.strip()) > 20]

    log.info(
        f"BATCH {batch_num}/{total_batches}  "
        f"docs={len(doc_ids)}  chunks={len(valid)}  target_q={max_questions}"
    )

    if not valid:
        log.warning("  No valid chunks, skipping.")
        return pd.DataFrame(), _empty_stats(batch_num, doc_ids, "no valid chunks", t0)

    testset_size = min(max_questions, max(5, len(valid)))

    gen = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_emb)
    try:
        testset = gen.generate_with_chunks(
            chunks=valid,
            testset_size=testset_size,
            raise_exceptions=False,
        )
    except Exception as e:
        log.error(f"  generate_with_chunks failed: {e}")
        try:
            log.info("  Trying generate_with_langchain_docs fallback...")
            testset = gen.generate_with_langchain_docs(
                documents=valid,
                testset_size=testset_size,
                raise_exceptions=False,
            )
        except Exception as e2:
            log.error(f"  Fallback also failed: {e2}")
            return pd.DataFrame(), _empty_stats(batch_num, doc_ids, str(e2), t0)

    # Build output rows
    rows = []
    total_exact = total_norm = total_unresolved = 0

    for i, sample in enumerate(testset.samples):
        ev = sample.eval_sample
        synth = sample.synthesizer_name or "unknown"

        question = getattr(ev, "user_input", "") or ""
        reference = getattr(ev, "reference", "") or ""
        ref_contexts = getattr(ev, "reference_contexts", []) or []
        persona = getattr(ev, "persona_name", "") or ""

        # Hop type from synthesizer name
        if "single_hop" in synth.lower():
            qtype = "single_hop"
        elif "multi_hop" in synth.lower():
            qtype = "multi_hop"
        else:
            qtype = "other"

        prov, e, n, u = validate_provenance(ref_contexts, exact_idx, norm_idx)
        total_exact += e
        total_norm += n
        total_unresolved += u

        v_chunk_ids = list({v["chunk_id"] for v in prov if v["chunk_id"]})
        v_doc_ids = list({v["document_id"] for v in prov if v["document_id"]})
        v_types = list({v.get("chunk_type", "") for v in prov if v.get("chunk_type")})
        v_sections = list({v.get("section", "") for v in prov if v.get("section")})
        evidence_validated = total_unresolved == 0 and len(prov) > 0

        rows.append({
            "question_id": f"b{batch_num:03d}_q{i:04d}",
            "question": question,
            "answer_reference": reference,
            "question_type": qtype,
            "query_type": getattr(ev, "query_style", "") or "",
            "synthesizer": synth,
            "evolution_type": "",
            "difficulty": "",
            "persona_name": persona,
            "document_id": "; ".join(doc_ids[:5]),
            "source_document_ids": "; ".join(v_doc_ids),
            "source_chunk_ids": "; ".join(v_chunk_ids),
            "reference_contexts": " || ".join(ref_contexts),
            "evidence_quote": _extract_quote(reference),
            "evidence_match": "; ".join(v["match_type"] for v in prov),
            "evidence_validated": evidence_validated,
            "chunk_type": "; ".join(v_types),
            "section": "; ".join(v_sections),
            "metadata": json.dumps({
                "batch": batch_num, "persona": persona,
                "num_contexts": len(ref_contexts),
                "provenance": {"exact": e, "normalised": n, "unresolved": u},
            }),
            "batch_number": batch_num,
        })

    df = pd.DataFrame(rows)
    dt = time.time() - t0
    log.info(
        f"  Generated {len(rows)} questions  "
        f"(exact={total_exact} norm={total_norm} unresolved={total_unresolved})  "
        f"{dt:.1f}s"
    )

    stats = {
        "batch": batch_num, "docs": len(doc_ids), "chunks": len(valid),
        "questions": len(rows), "exact": total_exact, "normalised": total_norm,
        "unresolved": total_unresolved, "seconds": round(dt, 1), "error": None,
    }
    return df, stats


def _empty_stats(batch, doc_ids, error, t0):
    return {
        "batch": batch, "docs": len(doc_ids), "chunks": 0,
        "questions": 0, "exact": 0, "normalised": 0, "unresolved": 0,
        "seconds": round(time.time() - t0, 1), "error": error,
    }


def _extract_quote(text: str) -> str:
    """Pull a factual-looking sentence from the reference answer."""
    if not text:
        return ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if len(sent) > 30 and any(
            w in sent.lower() for w in [
                "found", "result", "show", "increase", "decrease",
                "significant", "effect", "value", "level", "compared",
                "observed", "total", "fold", "percent",
            ]
        ):
            return sent
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(sent.strip()) > 20:
            return sent.strip()
    return text[:300]


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        log.info(f"Resuming: {len(data.get('done', []))} docs already processed")
        return data
    return {"done": [], "batches": [], "total_q": 0}


def save_checkpoint(path: str, ckpt: dict):
    with open(path, "w") as f:
        json.dump(ckpt, f)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Discover parquet files ──
    parquet_files = discover_parquets(args.parquet)
    log.info(f"Found {len(parquet_files)} parquet file(s) in {args.parquet}")

    if args.test:
        parquet_files = parquet_files[:1]
        args.batch_size = 1
        args.questions_per_batch = 5
        log.info("TEST MODE: 1 document, 5 questions")

    # ── Load all chunks into one DataFrame ──
    dfs = []
    for fp in parquet_files:
        try:
            dfs.append(load_parquet(fp))
        except Exception as e:
            log.warning(f"Failed to load {fp}: {e}")
    if not dfs:
        log.error("No data loaded. Exiting.")
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded {len(df)} chunks from {df['document_id'].nunique()} document(s)")

    # ── Group by document ──
    doc_groups = {}
    for doc_id, grp in df.groupby("document_id"):
        doc_groups[doc_id] = chunks_to_lc_docs(grp)
    log.info(f"Document groups: {len(doc_groups)}")

    # ── Setup Ollama ──
    setup_ollama(args.model, args.ollama_url)

    # ── Create Ragas LLM ──
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOllama(
        model=args.model, base_url=args.ollama_url,
        temperature=args.temperature, num_ctx=8192, timeout=120,
    )
    ragas_llm = LangchainLLMWrapper(chat)
    log.info(f"Ragas LLM ready: {args.model}")

    # ── Create embeddings ──
    from ragas.embeddings import LangchainEmbeddingsWrapper
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    hf_emb = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.embedding_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    ragas_emb = LangchainEmbeddingsWrapper(hf_emb)
    log.info(f"Embeddings ready: {args.embedding_model} on {args.embedding_device}")

    # ── Provenance index ──
    log.info("Building provenance index...")
    exact_idx, norm_idx = build_provenance_index(df)
    log.info(f"Index: {len(exact_idx)} exact, {len(norm_idx)} normalised")

    # ── Output paths ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live_csv = out_dir / "question_dataset_live.csv"
    final_csv = out_dir / "question_dataset.csv"
    ckpt_file = out_dir / "checkpoint.json"

    # ── Resume from checkpoint ──
    ckpt = load_checkpoint(str(ckpt_file))
    done_ids = set(ckpt["done"])
    all_stats = ckpt["batches"]
    all_dfs = []

    # Load existing live CSV
    if live_csv.exists():
        try:
            existing = pd.read_csv(str(live_csv))
            all_dfs.append(existing)
            log.info(f"Loaded {len(existing)} rows from live CSV")
        except Exception:
            pass

    remaining = [d for d in doc_groups if d not in done_ids]
    batches = [remaining[i:i + args.batch_size] for i in range(0, len(remaining), args.batch_size)]

    log.info(f"\n{'='*60}")
    log.info(f"TO PROCESS: {len(remaining)} docs in {len(batches)} batches")
    log.info(f"{'='*60}\n")

    # ── Process batches ──
    batch_num = len(all_stats)
    for batch_ids in batches:
        batch_num += 1
        all_chunks = []
        for did in batch_ids:
            all_chunks.extend(doc_groups[did])

        try:
            result_df, stats = generate_batch(
                lc_docs=all_chunks, doc_ids=batch_ids,
                ragas_llm=ragas_llm, ragas_emb=ragas_emb,
                exact_idx=exact_idx, norm_idx=norm_idx,
                batch_num=batch_num, total_batches=batch_num + len(batches) - (batch_num - len(all_stats)),
                max_questions=args.questions_per_batch,
            )
            all_stats.append(stats)

            if len(result_df) > 0:
                all_dfs.append(result_df)
                combined = pd.concat(all_dfs, ignore_index=True)
                combined.to_csv(str(live_csv), index=False)

        except Exception as e:
            log.error(f"Batch {batch_num} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({
                "batch": batch_num, "docs": len(batch_ids), "chunks": 0,
                "questions": 0, "exact": 0, "normalised": 0, "unresolved": 0,
                "seconds": 0, "error": str(e),
            })

        # Checkpoint
        ckpt["done"] = list(done_ids | set(batch_ids))
        ckpt["batches"] = all_stats
        ckpt["total_q"] = sum(s.get("questions", 0) for s in all_stats)
        save_checkpoint(str(ckpt_file), ckpt)
        done_ids.update(batch_ids)

    # ── Final CSV ──
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    final_df.to_csv(str(final_csv), index=False)
    log.info(f"Saved: {final_csv}  ({len(final_df)} rows)")

    # ── Quality report ──
    print_report(final_df, all_stats, args)


def print_report(df: pd.DataFrame, stats: list, args):
    print(f"\n{'='*60}")
    print("QUALITY REPORT")
    print(f"{'='*60}")

    if df.empty:
        print("  No questions generated.")
        return

    n_docs = df["document_id"].str.split("; ").str[0].nunique()
    print(f"  Documents:       {n_docs}")
    print(f"  Questions:       {len(df)}")
    print(f"  Batches run:     {len(stats)}")
    print(f"  Failed batches:  {sum(1 for s in stats if s.get('error'))}")

    print(f"\n  Question types:")
    for qt, c in df["question_type"].value_counts().items():
        print(f"    {qt}: {c}")

    print(f"\n  Synthesizers:")
    for s, c in df["synthesizer"].value_counts().items():
        print(f"    {s}: {c}")

    # Provenance
    total_e = sum(s.get("exact", 0) for s in stats)
    total_n = sum(s.get("normalised", 0) for s in stats)
    total_u = sum(s.get("unresolved", 0) for s in stats)
    total_c = total_e + total_n + total_u
    print(f"\n  Provenance:")
    print(f"    Exact matches:    {total_e}")
    print(f"    Normalised:       {total_n}")
    print(f"    Unresolved:       {total_u}")
    if total_c:
        print(f"    Validation rate:  {(total_e + total_n) / total_c * 100:.1f}%")

    # Timing
    secs = [s.get("seconds", 0) for s in stats if s.get("seconds")]
    if secs:
        print(f"\n  Timing:")
        print(f"    Total:    {sum(secs):.0f}s")
        print(f"    Avg/batch: {np.mean(secs):.1f}s")

    # Sample
    print(f"\n  Sample questions:")
    for _, row in df.sample(min(10, len(df)), random_state=42).iterrows():
        print(f"\n    Q: {str(row['question'])[:120]}")
        print(f"    A: {str(row['answer_reference'])[:120]}")
        print(f"    Type: {row['question_type']}  Chunks: {row['source_chunk_ids'][:60]}")

    print(f"\n{'='*60}")
    print(f"OUTPUT: question_dataset/question_dataset.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
