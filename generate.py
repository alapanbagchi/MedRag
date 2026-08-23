#!/usr/bin/env python3
"""Scientific RAG Evaluation Dataset Generator."""
import argparse, json, logging, os, re, shutil, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gen")

def cli():
    p = argparse.ArgumentParser(description="Generate RAG questions from PMC papers")
    p.add_argument("--api-key", required=True, help="Cerebras API key")
    p.add_argument("--batch-size", type=int, default=20, help="Papers per batch")
    p.add_argument("--chunks-dir", default="chunks", help="Parquet directory")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--questions-per-paper", type=int, default=8)
    p.add_argument("--cross-paper-ratio", type=float, default=0.1)
    p.add_argument("--model", default="gemma-4-31b", help="Cerebras model name")
    p.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between API calls (default: 0.5)")
    p.add_argument("--test", action="store_true", help="Test: 3 papers only")
    return p.parse_args()

CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"

def create_client(api_key):
    s = requests.Session()
    s.headers.update({
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    })
    return s

def chat_completion(client, model, messages, temperature=0.2, max_tokens=1024, max_retries=5):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "top_p": 1,
        "stream": False,
    }
    for attempt in range(max_retries):
        try:
            resp = client.post(CEREBRAS_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "Too Many" in err_str:
                wait = min(2 ** attempt * 2, 120)
                delay_match = re.search(r"retry in (\d+\.?\d*)s", err_str)
                if delay_match:
                    wait = max(wait, float(delay_match.group(1)) + 1)
                log.warning("  Rate limited, waiting %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded")

REQUIRED_COLS = ["id", "document_id", "text", "embedding_text", "chunk_type", "section",
                 "subsection", "breadcrumb", "document_position", "table_id", "figure_id", "row_label"]

def load_paper(path):
    try:
        table = pq.read_table(path)
        cols = [c for c in REQUIRED_COLS if c in table.column_names]
        for extra in ("parent_id", "object_id", "source_block_ids"):
            if extra in table.column_names and extra not in cols:
                cols.append(extra)
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas()
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = None
        for c in ("breadcrumb", "source_block_ids"):
            if c in df.columns:
                df[c] = df[c].apply(lambda x: x if isinstance(x, list) else [])
        df = df.sort_values("document_position").reset_index(drop=True)
        return df
    except Exception as e:
        log.warning("Failed to load %s: %s", path, e)
        return None

def discover_papers(chunks_dir):
    p = Path(chunks_dir)
    if not p.exists():
        log.error("Directory not found: %s", chunks_dir)
        sys.exit(1)
    return sorted(str(f) for f in p.glob("*.parquet"))

def extract_doc_id(path):
    name = os.path.basename(path)
    return re.sub(r"\.\d+\.parquet$", "", name).replace(".parquet", "")


def format_chunks_for_llm(df):
    parts = []
    for _, row in df.iterrows():
        text = str(row.get("embedding_text") or row.get("text", "")).strip()
        if not text:
            continue
        parts.append("[CHUNK_ID: " + str(row["id"]) + "]\n" + text)
    return "\n\n---\n\n".join(parts)

def build_chunk_index(df):
    idx = {}
    for _, row in df.iterrows():
        text = str(row.get("embedding_text") or row.get("text", ""))
        idx[row["id"]] = text
    return idx

def build_paper_summary(df):
    sections = df["section"].dropna().unique().tolist()
    doc_id = df["document_id"].iloc[0] if len(df) > 0 else ""
    title = ""
    if "metadata" in df.columns:
        meta = df["metadata"].iloc[0]
        if isinstance(meta, dict):
            title = meta.get("title", "")
    return {"document_id": doc_id, "title": title, "total_chunks": len(df), "sections": sections}


SYSTEM_PROMPT = (
    "You are generating evaluation questions for a scientific retrieval-augmented "
    "generation (RAG) system.\n\n"
    "RULES:\n"
    "1. Use ONLY the supplied scientific evidence. Do not rely on pretrained knowledge.\n"
    "2. Every question must be answerable from the supplied evidence.\n"
    "3. Every answer must be directly supported by the evidence.\n"
    "4. Do not invent facts, numbers, entities, methods, or relationships.\n"
    "5. The EVIDENCE field must be copied VERBATIM from the original chunks. Do not paraphrase.\n"
    "6. Each question must include the exact chunk IDs that contain the evidence.\n\n"
    "OUTPUT FORMAT: Return a JSON array. Each element:\n"
    "{\n"
    '    "question": "the question",\n'
    '    "ground_truth": "the answer",\n'
    '    "evidence": "exact verbatim text from the source chunk(s)",\n'
    '    "source_chunk_ids": ["chunk_id_1"],\n'
    '    "question_type": "one of: factual, multi-fact, comparison, methodology, results, quantitative, table-based, multi-chunk, conceptual, cross-section",\n'
    '    "difficulty": "easy|medium|hard"\n'
    "}\n\n"
    "Generate a MIX of question types. Include questions connecting different chunks.\n"
    "Return ONLY the JSON array, no other text."
)

CROSS_PAPER_PROMPT = (
    "You are generating cross-paper evaluation questions for a scientific RAG system.\n\n"
    "You will receive chunks from MULTIPLE papers. Generate questions that require\n"
    "information from at least two different papers to answer.\n\n"
    "RULES:\n"
    "1. Use ONLY the supplied evidence.\n"
    "2. EVIDENCE must be copied verbatim from source chunks.\n"
    "3. Each question must reference chunks from at least 2 different documents.\n\n"
    "OUTPUT FORMAT: JSON array. Each element:\n"
    "{\n"
    '    "question": "...", "ground_truth": "...",\n'
    '    "evidence": "verbatim text (separate with |||)",\n'
    '    "source_chunk_ids": ["id_A", "id_B"], "document_ids": ["A", "B"],\n'
    '    "question_type": "cross_paper", "difficulty": "medium|hard"\n'
    "}\n\nReturn ONLY the JSON array."
)


def build_single_paper_prompt(df, n_questions):
    context = format_chunks_for_llm(df)
    doc_id = df["document_id"].iloc[0]
    user_msg = "Paper: " + doc_id + "\n\n" + context + "\n\n"
    user_msg += "Generate " + str(n_questions) + " evaluation questions for this paper."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]


def build_cross_paper_prompt(dfs, n_questions=2):
    parts = []
    doc_ids = []
    for df in dfs:
        doc_id = df["document_id"].iloc[0]
        doc_ids.append(doc_id)
        sample = df[df["chunk_type"] != "reference"].head(20)
        parts.append("PAPER: " + doc_id + "\n" + format_chunks_for_llm(sample))
    combined = "\n\n===\n\n".join(parts)
    user_msg = (
        "Generate " + str(n_questions) + " CROSS-PAPER questions requiring info from at least two papers:\n"
        + ", ".join(doc_ids) + "\n\nCHUNKS:\n" + combined
    )
    return [{"role": "system", "content": CROSS_PAPER_PROMPT}, {"role": "user", "content": user_msg}]



def parse_json_response(text):
    text = text.strip()
    # Strip markdown code fences
    bt = chr(96) * 3
    if text.startswith(bt):
        text = re.sub(r"^" + bt + r"json\s*", "", text)
        text = re.sub(r"\s*" + bt + "$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
    results = []
    for obj_str in objects:
        try:
            results.append(json.loads(obj_str))
        except json.JSONDecodeError:
            continue
    return results


def validate_question(q, paper_chunks):
    for field in ("question", "ground_truth", "evidence", "source_chunk_ids"):
        if field not in q or not q[field]:
            return False, "missing_field:" + field
    ids = q.get("source_chunk_ids", [])
    if not isinstance(ids, list) or len(ids) == 0:
        return False, "empty_source_chunk_ids"
    evidence = str(q["evidence"]).strip()
    if len(evidence) < 20:
        return False, "evidence_too_short"
    for cid in ids:
        if cid in paper_chunks:
            chunk_text = paper_chunks[cid]
            if evidence in chunk_text:
                return True, "exact"
            norm_ev = re.sub(r"\s+", " ", evidence).strip()
            norm_chunk = re.sub(r"\s+", " ", chunk_text).strip()
            if norm_ev in norm_chunk:
                return True, "normalized_whitespace"
    return False, "evidence_not_in_source_chunks"


OUTPUT_COLUMNS = ["question_id", "question", "ground_truth", "evidence",
                  "document_id", "source_chunk_ids", "question_type", "difficulty", "generation_method"]


def ensure_output_dir(path):
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_processed(output_dir):
    live = output_dir / "questions_live.csv"
    if live.exists():
        try:
            df = pd.read_csv(str(live))
            ids = set()
            for v in df["document_id"].dropna():
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        ids.update(parsed)
                    else:
                        ids.add(str(parsed))
                except (json.JSONDecodeError, TypeError):
                    ids.add(str(v))
            return ids
        except Exception:
            pass
    return set()


def load_live_df(output_dir):
    live = output_dir / "questions_live.csv"
    if live.exists():
        try:
            return pd.read_csv(str(live))
        except Exception:
            pass
    return None


def append_questions(output_dir, existing_df, new_rows):
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        for col in OUTPUT_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = ""
        new_df = new_df[OUTPUT_COLUMNS]
        if existing_df is not None and len(existing_df) > 0:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
    else:
        combined = existing_df if existing_df is not None else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if "source_chunk_ids" in combined.columns:
        combined["source_chunk_ids"] = combined["source_chunk_ids"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else str(x)
        )
    combined.to_csv(str(output_dir / "questions_live.csv"), index=False)
    return combined


def append_failure(output_dir, doc_id, filename, error):
    fail_path = output_dir / "failures.csv"
    row = pd.DataFrame([{"document_id": doc_id, "filename": filename,
                         "error": str(error)[:500], "timestamp": datetime.now(timezone.utc).isoformat()}])
    if fail_path.exists():
        row.to_csv(str(fail_path), mode="a", header=False, index=False)
    else:
        row.to_csv(str(fail_path), index=False)


def finalize(output_dir):
    live = output_dir / "questions_live.csv"
    final = output_dir / "questions.csv"
    if live.exists():
        shutil.copy2(str(live), str(final))
        log.info("Final CSV: %s", final)



def process_single_paper(client, model, paper_path, n_questions):
    filename = os.path.basename(paper_path)
    df = load_paper(paper_path)
    if df is None or len(df) == 0:
        return [], "failed_to_load:" + filename
    doc_id = df["document_id"].iloc[0]
    gen_df = df[df["chunk_type"] != "reference"].copy()
    if len(gen_df) == 0:
        gen_df = df.copy()
    chunk_index = build_chunk_index(df)
    messages = build_single_paper_prompt(gen_df, n_questions)
    try:
        raw = chat_completion(client, model, messages)
    except Exception as e:
        return [], "llm_error:" + str(e)
    parsed = parse_json_response(raw)
    if not parsed:
        return [], "parse_error:empty_response"
    valid = []
    for q in parsed:
        is_valid, _ = validate_question(q, chunk_index)
        if is_valid:
            q["document_id"] = doc_id
            q["generation_method"] = "single_paper"
            valid.append(q)
    return valid, None


def process_cross_paper(client, model, paper_dfs, n_questions=2):
    combined_index = {}
    for df in paper_dfs:
        combined_index.update(build_chunk_index(df))
    messages = build_cross_paper_prompt(paper_dfs, n_questions)
    try:
        raw = chat_completion(client, model, messages)
    except Exception as e:
        return [], "llm_error:" + str(e)
    parsed = parse_json_response(raw)
    if not parsed:
        return [], "parse_error:empty_cross_paper"
    valid = []
    for q in parsed:
        is_valid, _ = validate_question(q, combined_index)
        if is_valid:
            q["generation_method"] = "cross_paper"
            if "document_ids" not in q:
                q["document_ids"] = list(set(df["document_id"].iloc[0] for df in paper_dfs))
            valid.append(q)
    return valid, None


def main():
    args = cli()
    client = create_client(args.api_key)
    model = args.model
    papers = discover_papers(args.chunks_dir)
    total = len(papers)
    if total == 0:
        log.error("No .parquet files found in %s", args.chunks_dir)
        sys.exit(1)
    if args.test:
        papers = papers[:3]
        total = len(papers)
    out = ensure_output_dir(args.output_dir)
    processed_ids = load_processed(out)
    existing_df = load_live_df(out)
    remaining = [p for p in papers if extract_doc_id(p) not in processed_ids]
    total_valid = total_rejected = total_failed = 0

    print()
    print("=" * 60)
    print("QUESTION GENERATION")
    print("=" * 60)
    print("  Papers:     %d" % total)
    print("  Completed:  %d" % len(processed_ids))
    print("  Remaining:  %d" % len(remaining))
    print("  Batch size: %d" % args.batch_size)
    print("  Model:      %s" % model)
    print("=" * 60)
    print()

    log.info("Cerebras client ready. Model: %s", model)
    batch_num = 0
    i = 0

    while i < len(remaining):
        batch = remaining[i:i + args.batch_size]
        batch_num += 1
        i += len(batch)
        print("--- Batch %d (%d papers) ---" % (batch_num, len(batch)))
        batch_questions = []
        batch_generated = batch_valid = batch_rejected = 0

        for paper_path in batch:
            doc_id = extract_doc_id(paper_path)
            filename = os.path.basename(paper_path)
            try:
                questions, error = process_single_paper(client, model, paper_path, args.questions_per_paper)
                if error:
                    log.warning("  %s: %s", doc_id, error)
                    append_failure(out, doc_id, filename, error)
                    total_failed += 1
                    processed_ids.add(doc_id)
                    continue
                batch_generated += len(questions)
                df = load_paper(paper_path)
                if df is not None:
                    chunk_idx = build_chunk_index(df)
                    for q in questions:
                        ok, _ = validate_question(q, chunk_idx)
                        if ok:
                            batch_questions.append(q)
                            batch_valid += 1
                        else:
                            batch_rejected += 1
                else:
                    batch_questions.extend(questions)
                    batch_valid += len(questions)
                processed_ids.add(doc_id)
                log.info("  %s: %d questions", doc_id, len(questions))
                time.sleep(args.rate_limit)
            except Exception as e:
                log.error("  %s: UNEXPECTED: %s", doc_id, e)
                append_failure(out, doc_id, filename, str(e))
                total_failed += 1
                processed_ids.add(doc_id)

        if len(batch) >= 3 and args.cross_paper_ratio > 0:
            n_cross = max(1, int(len(batch) * args.cross_paper_ratio))
            cross_papers = []
            for pp in batch[:min(5, len(batch))]:
                df = load_paper(pp)
                if df is not None:
                    cross_papers.append(df)
            if len(cross_papers) >= 2:
                try:
                    cross_qs, cross_err = process_cross_paper(client, model, cross_papers, n_questions=n_cross)
                    if not cross_err:
                        batch_questions.extend(cross_qs)
                        batch_valid += len(cross_qs)
                        log.info("  Cross-paper: %d questions", len(cross_qs))
                except Exception as e:
                    log.warning("  Cross-paper failed: %s", e)

        if batch_questions:
            existing_df = append_questions(out, existing_df, batch_questions)
            total_valid += batch_valid
            total_rejected += batch_rejected

        done = len(processed_ids)
        print("  Generated: %d  Valid: %d  Rejected: %d" % (batch_generated, batch_valid, batch_rejected))
        print("  Progress: %d/%d papers  |  %d questions" % (done, total, total_valid))

    finalize(out)
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("  Papers processed:  %d" % len(processed_ids))
    print("  Questions total:   %d" % total_valid)
    print("  Rejected:          %d" % total_rejected)
    print("  Failed papers:     %d" % total_failed)
    print("  Output:            %s" % str(out / "questions.csv"))
    print("=" * 60)


if __name__ == "__main__":
    main()
