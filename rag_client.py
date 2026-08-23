#!/usr/bin/env python3
"""
RAG Dataset Generation Client
==============================
Sends local parquet files to a remote Ragas server (e.g. Kaggle via localtunnel)
and collects the generated questions.

Usage:
    # Test with 1 file
    python rag_client.py --server https://your-localtunnel-url.loca.lt --test

    # Process entire chunks/ directory (20 docs per batch)
    python rag_client.py --server https://your-localtunnel-url.loca.lt

    # Custom batch size and output
    python rag_client.py --server https://URL --batch-size 10 --output-dir ./output

    # Single file
    python rag_client.py --server https://URL --parquet chunks/PMC10327125.4.parquet
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_client")


def parse_args():
    p = argparse.ArgumentParser(description="Send parquet files to Ragas server for generation")
    p.add_argument("--server", required=True, help="Server URL (e.g. https://xxx.loca.lt)")
    p.add_argument("--parquet", default="chunks/",
                    help="Parquet file or directory (default: chunks/)")
    p.add_argument("--batch-size", type=int, default=20,
                    help="How many parquet files to send per request (default: 20)")
    p.add_argument("--max-questions", type=int, default=100,
                    help="Max questions per batch (default: 100)")
    p.add_argument("--output-dir", default="question_dataset",
                    help="Output directory (default: question_dataset)")
    p.add_argument("--test", action="store_true",
                    help="Test mode: send only 1 file")
    p.add_argument("--timeout", type=int, default=600,
                    help="HTTP timeout in seconds (default: 600)")
    p.add_argument("--checkpoint", action="store_true", default=True,
                    help="Enable checkpointing (default: True)")
    return p.parse_args()


def discover_parquets(path: str) -> List[str]:
    p = Path(path)
    if p.is_file() and p.suffix == ".parquet":
        return [str(p)]
    if p.is_dir():
        return sorted(str(f) for f in p.glob("*.parquet"))
    raise FileNotFoundError(f"Not a parquet file or directory: {path}")


def check_server(server_url: str) -> dict:
    """Check if server is alive and get model info."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        raise RuntimeError(f"Cannot connect to server at {server_url}")
    except Exception as e:
        raise RuntimeError(f"Server health check failed: {e}")


def upload_single(server_url: str, filepath: str, max_questions: int, timeout: int) -> dict:
    """Upload a single parquet file and return results."""
    fname = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{server_url}/generate",
            files={"file": (fname, f, "application/octet-stream")},
            params={"max_questions": max_questions},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


def upload_batch(server_url: str, filepaths: List[str], max_questions: int, timeout: int) -> dict:
    """Upload multiple parquet files in one request."""
    files = []
    for fp in filepaths:
        fname = os.path.basename(fp)
        fobj = open(fp, "rb")
        files.append(("files", (fname, fobj, "application/octet-stream")))

    try:
        per_file_q = max(5, max_questions // len(filepaths))
        resp = requests.post(
            f"{server_url}/generate_batch",
            files=files,
            params={"max_questions": per_file_q},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    finally:
        for _, (_, fobj, _) in files:
            fobj.close()


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        log.info(f"Resuming: {len(data.get('done', []))} files already sent")
        return data
    return {"done": [], "batches": [], "total_q": 0}


def save_checkpoint(path: str, ckpt: dict):
    with open(path, "w") as f:
        json.dump(ckpt, f)


def main():
    args = parse_args()

    # Discover files
    parquet_files = discover_parquets(args.parquet)
    log.info(f"Found {len(parquet_files)} parquet file(s) in {args.parquet}")

    if args.test:
        parquet_files = parquet_files[:1]
        args.batch_size = 1
        args.max_questions = 10
        log.info("TEST MODE: 1 file, 10 questions")

    # Check server
    log.info(f"Connecting to server: {args.server}")
    info = check_server(args.server)
    log.info(f"Server OK. Model: {info.get('model', 'unknown')}")

    # Setup output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live_csv = out_dir / "question_dataset_live.csv"
    final_csv = out_dir / "question_dataset.csv"
    ckpt_file = out_dir / "client_checkpoint.json"

    # Resume
    ckpt = load_checkpoint(str(ckpt_file))
    done_set = set(ckpt["done"])
    all_stats = ckpt["batches"]
    all_dfs = []

    if live_csv.exists():
        try:
            existing = pd.read_csv(str(live_csv))
            all_dfs.append(existing)
            log.info(f"Loaded {len(existing)} existing rows from live CSV")
        except Exception:
            pass

    remaining = [f for f in parquet_files if f not in done_set]
    batches = [remaining[i:i + args.batch_size] for i in range(0, len(remaining), args.batch_size)]

    log.info(f"\n{'='*60}")
    log.info(f"TO SEND: {len(remaining)} files in {len(batches)} batches")
    log.info(f"{'='*60}\n")

    batch_num = len(all_stats)
    failed = 0

    for batch_files in batches:
        batch_num += 1
        log.info(f"BATCH {batch_num}/{batch_num + len(batches) - 1}  files={len(batch_files)}")

        t0 = time.time()

        try:
            if len(batch_files) == 1:
                result = upload_single(
                    args.server, batch_files[0],
                    args.max_questions, args.timeout,
                )
            else:
                result = upload_batch(
                    args.server, batch_files,
                    args.max_questions, args.timeout,
                )
        except Exception as e:
            log.error(f"  FAILED: {e}")
            failed += 1
            all_stats.append({
                "batch": batch_num, "files": len(batch_files),
                "questions": 0, "error": str(e),
                "seconds": round(time.time() - t0, 1),
            })
            # Mark as done so we don't retry forever
            ckpt["done"] = list(done_set | set(batch_files))
            ckpt["batches"] = all_stats
            save_checkpoint(str(ckpt_file), ckpt)
            done_set.update(batch_files)
            continue

        questions = result.get("questions", [])
        n_q = len(questions)
        exact = result.get("exact", 0)
        norm = result.get("normalised", 0)
        unres = result.get("unresolved", 0)
        dt = round(time.time() - t0, 1)

        log.info(f"  Got {n_q} questions (exact={exact} norm={norm} unresolved={unres})  {dt}s")

        if questions:
            df = pd.DataFrame(questions)
            all_dfs.append(df)
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(str(live_csv), index=False)
            log.info(f"  -> Live CSV: {len(combined)} total rows")

        all_stats.append({
            "batch": batch_num, "files": len(batch_files),
            "questions": n_q, "exact": exact, "normalised": norm,
            "unresolved": unres, "error": None, "seconds": dt,
        })

        ckpt["done"] = list(done_set | set(batch_files))
        ckpt["batches"] = all_stats
        ckpt["total_q"] = sum(s.get("questions", 0) for s in all_stats)
        save_checkpoint(str(ckpt_file), ckpt)
        done_set.update(batch_files)

    # Final
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    final_df.to_csv(str(final_csv), index=False)

    # Report
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Files processed: {len(done_set)}")
    print(f"  Questions:       {len(final_df)}")
    print(f"  Batches:         {len(all_stats)}")
    print(f"  Failed:          {failed}")
    print(f"  Output:          {final_csv}")

    if len(final_df) > 0:
        print(f"\n  Question types:")
        for qt, c in final_df["question_type"].value_counts().items():
            print(f"    {qt}: {c}")
        print(f"\n  Sample:")
        for _, row in final_df.sample(min(5, len(final_df)), random_state=42).iterrows():
            print(f"    Q: {str(row['question'])[:100]}")
            print(f"    A: {str(row['answer_reference'])[:100]}")
            print()

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
