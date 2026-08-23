#!/usr/bin/env python3
"""
RAG Dataset Generation Server (runs on Kaggle)
================================================
Receives parquet files via HTTP, runs Ragas + Gemma generation,
and returns the synthetic questions as JSON.

Setup on Kaggle:
    pip install fastapi uvicorn ragas langchain-core langchain-ollama \
                langchain-huggingface sentence-transformers pyarrow pandas \
                python-multipart requests scikit-learn
    python rag_server.py --model gemma2:9b --port 8000

Then expose via localtunnel:
    npx localtunnel --port 8000
"""

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import unicodedata
import warnings
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_server")


# ──────────────────────────────────────────────────────────────────────────────
# Lazy-loaded global state (initialised once at startup)
# ──────────────────────────────────────────────────────────────────────────────

_STATE = {}


def _norm_ws(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def build_provenance_index(df: pd.DataFrame):
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
            normalised.setdefault(nh, {**info, "text": txt})
    return exact, normalised


def validate_provenance(contexts, exact_idx, norm_idx):
    results = []
    for ctx in (contexts or []):
        if not ctx or not ctx.strip():
            continue
        ctx = ctx.strip()
        h = hashlib.md5(ctx.encode()).hexdigest()
        if h in exact_idx:
            m = exact_idx[h]
            results.append({**m, "match_type": "exact"})
            continue
        nh = hashlib.md5(_norm_ws(ctx).encode()).hexdigest()
        if nh in norm_idx:
            m = norm_idx[nh]
            results.append({**m, "match_type": "normalised_whitespace"})
            continue
        found = False
        for minfo in norm_idx.values():
            chunk_norm = _norm_ws(minfo.get("text", ""))
            if len(ctx) > 50 and _norm_ws(ctx) in chunk_norm:
                results.append({**minfo, "match_type": "substring_contained"})
                found = True
                break
        if found:
            continue
        for minfo in norm_idx.values():
            chunk_norm = _norm_ws(minfo.get("text", ""))
            if len(chunk_norm) > 50 and chunk_norm in _norm_ws(ctx):
                results.append({**minfo, "match_type": "chunk_in_context"})
                found = True
                break
        if found:
            continue
        results.append({"chunk_id": "", "document_id": "", "match_type": "unresolved"})
    exact_c = sum(1 for r in results if r["match_type"] == "exact")
    norm_c = sum(1 for r in results if r["match_type"] not in ("exact", "unresolved"))
    unres = sum(1 for r in results if r["match_type"] == "unresolved")
    return results, exact_c, norm_c, unres


def _extract_quote(text: str) -> str:
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


def chunks_to_lc_docs(df):
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


def load_parquet_bytes(data: bytes) -> pd.DataFrame:
    table = pq.read_table(io.BytesIO(data))
    df = table.to_pandas()

    required = ["id", "document_id", "text", "embedding_text",
                 "chunk_type", "section", "subsection", "breadcrumb",
                 "document_position", "table_id", "figure_id", "row_label", "metadata"]
    for col in required:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))] if col in ("breadcrumb",) else None

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


def generate_for_parquet(
    parquet_bytes: bytes,
    max_questions: int = 100,
    batch_doc_ids: list = None,
) -> dict:
    """
    Core generation: takes raw parquet bytes, returns dict with questions + stats.
    """
    from ragas.testset import TestsetGenerator

    t0 = time.time()
    llm = _STATE["llm"]
    emb = _STATE["emb"]

    # Load parquet
    df = load_parquet_bytes(parquet_bytes)
    n_chunks = len(df)
    n_docs = df["document_id"].nunique()
    log.info(f"Loaded {n_chunks} chunks from {n_docs} doc(s)")

    if batch_doc_ids is None:
        batch_doc_ids = df["document_id"].unique().tolist()

    # Build provenance index
    exact_idx, norm_idx = build_provenance_index(df)

    # Convert to LangChain docs
    lc_docs = chunks_to_lc_docs(df)
    valid = [d for d in lc_docs if d.page_content and len(d.page_content.strip()) > 20]

    if not valid:
        return {
            "questions": [], "n_chunks": n_chunks, "n_docs": n_docs,
            "n_questions": 0, "exact": 0, "normalised": 0, "unresolved": 0,
            "error": "No valid chunks", "seconds": round(time.time() - t0, 1),
        }

    testset_size = min(max_questions, max(5, len(valid)))

    # Generate
    gen = TestsetGenerator(llm=llm, embedding_model=emb)
    try:
        testset = gen.generate_with_chunks(
            chunks=valid,
            testset_size=testset_size,
            raise_exceptions=False,
        )
    except Exception as e:
        log.error(f"generate_with_chunks failed: {e}")
        try:
            log.info("Trying generate_with_langchain_docs fallback...")
            testset = gen.generate_with_langchain_docs(
                documents=valid, testset_size=testset_size, raise_exceptions=False,
            )
        except Exception as e2:
            log.error(f"Fallback also failed: {e2}")
            return {
                "questions": [], "n_chunks": n_chunks, "n_docs": n_docs,
                "n_questions": 0, "exact": 0, "normalised": 0, "unresolved": 0,
                "error": str(e2), "seconds": round(time.time() - t0, 1),
            }

    # Process results
    rows = []
    total_e = total_n = total_u = 0

    for i, sample in enumerate(testset.samples):
        ev = sample.eval_sample
        synth = sample.synthesizer_name or "unknown"
        question = getattr(ev, "user_input", "") or ""
        reference = getattr(ev, "reference", "") or ""
        ref_contexts = getattr(ev, "reference_contexts", []) or []
        persona = getattr(ev, "persona_name", "") or ""

        if "single_hop" in synth.lower():
            qtype = "single_hop"
        elif "multi_hop" in synth.lower():
            qtype = "multi_hop"
        else:
            qtype = "other"

        prov, e, n, u = validate_provenance(ref_contexts, exact_idx, norm_idx)
        total_e += e
        total_n += n
        total_u += u

        v_chunk_ids = list({v["chunk_id"] for v in prov if v["chunk_id"]})
        v_doc_ids = list({v["document_id"] for v in prov if v["document_id"]})

        rows.append({
            "question_id": f"q{i:04d}",
            "question": question,
            "answer_reference": reference,
            "question_type": qtype,
            "query_type": getattr(ev, "query_style", "") or "",
            "synthesizer": synth,
            "evolution_type": "",
            "difficulty": "",
            "persona_name": persona,
            "document_id": "; ".join(batch_doc_ids[:5]),
            "source_document_ids": "; ".join(v_doc_ids),
            "source_chunk_ids": "; ".join(v_chunk_ids),
            "reference_contexts": " || ".join(ref_contexts),
            "evidence_quote": _extract_quote(reference),
            "evidence_match": "; ".join(v["match_type"] for v in prov),
            "evidence_validated": total_u == 0 and len(prov) > 0,
            "chunk_type": "; ".join(sorted({v.get("chunk_type", "") for v in prov if v.get("chunk_type")})),
            "section": "; ".join(sorted({v.get("section", "") for v in prov if v.get("section")})),
            "batch_number": 0,
        })

    dt = time.time() - t0
    log.info(
        f"Generated {len(rows)} questions (exact={total_e} norm={total_n} "
        f"unresolved={total_u}) in {dt:.1f}s"
    )

    return {
        "questions": rows,
        "n_chunks": n_chunks,
        "n_docs": n_docs,
        "n_questions": len(rows),
        "exact": total_e,
        "normalised": total_n,
        "unresolved": total_u,
        "error": None,
        "seconds": round(dt, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

def create_app():
    from fastapi import FastAPI, UploadFile, File, Query
    from fastapi.responses import JSONResponse

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup: load model once."""
        log.info("Initializing models...")
        _init_models()
        log.info("Models ready. Server is live.")
        yield
        log.info("Shutting down.")

    app = FastAPI(
        title="RAG Dataset Generator",
        description="Receives parquet chunks, generates synthetic RAG evaluation questions via Ragas + Gemma.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": _STATE.get("model_name", "unknown")}

    @app.post("/generate")
    async def generate(
        file: UploadFile = File(...),
        max_questions: int = Query(default=100, ge=1, le=500),
        doc_ids: Optional[str] = Query(default=None, description="Comma-separated document IDs (optional)"),
    ):
        """
        Upload a .parquet file. Returns generated questions as JSON.

        The parquet must have columns: id, document_id, text, embedding_text,
        chunk_type, section, metadata.
        """
        try:
            content = await file.read()
            log.info(f"Received file: {file.filename} ({len(content)} bytes)")

            batch_doc_ids = doc_ids.split(",") if doc_ids else None
            result = generate_for_parquet(content, max_questions, batch_doc_ids)

            return JSONResponse(content=result)

        except Exception as e:
            log.error(f"Generation failed: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "questions": []},
            )

    @app.post("/generate_batch")
    async def generate_batch(
        files: List[UploadFile] = File(...),
        max_questions: int = Query(default=100, ge=1, le=500),
    ):
        """
        Upload multiple .parquet files at once. Returns combined results.
        Useful for batching 20 docs at a time from the client.
        """
        all_questions = []
        stats = []
        total_e = total_n = total_u = 0

        for f in files:
            try:
                content = await f.read()
                result = generate_for_parquet(content, max_questions // len(files))
                all_questions.extend(result.get("questions", []))
                total_e += result.get("exact", 0)
                total_n += result.get("normalised", 0)
                total_u += result.get("unresolved", 0)
                stats.append({"file": f.filename, **{k: v for k, v in result.items() if k != "questions"}})
            except Exception as e:
                log.error(f"Failed {f.filename}: {e}")
                stats.append({"file": f.filename, "error": str(e)})

        return JSONResponse(content={
            "questions": all_questions,
            "n_questions": len(all_questions),
            "exact": total_e,
            "normalised": total_n,
            "unresolved": total_u,
            "per_file": stats,
        })

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Model init
# ──────────────────────────────────────────────────────────────────────────────

def _init_models():
    model_name = _STATE.get("model_name", "gemma2:9b")
    base_url = _STATE.get("base_url", "http://localhost:11434")
    embedding_model = _STATE.get("embedding_model", "BAAI/bge-base-en-v1.5")
    embedding_device = _STATE.get("embedding_device", "cpu")

    # LLM
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOllama(
        model=model_name, base_url=base_url,
        temperature=0.7, num_ctx=8192, timeout=180,
    )
    _STATE["llm"] = LangchainLLMWrapper(chat)
    log.info(f"LLM ready: {model_name}")

    # Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    hf_emb = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    _STATE["emb"] = LangchainEmbeddingsWrapper(hf_emb)
    log.info(f"Embeddings ready: {embedding_model}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RAG Dataset Generation Server")
    p.add_argument("--model", default="gemma2:9b", help="Ollama model name")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    _STATE["model_name"] = args.model
    _STATE["base_url"] = args.ollama_url
    _STATE["embedding_model"] = args.embedding_model
    _STATE["embedding_device"] = args.embedding_device

    import uvicorn
    app = create_app()
    log.info(f"Starting server on {args.host}:{args.port}")
    log.info(f"Model: {args.model} via {args.ollama_url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
