# MedRAG

Structure-aware RAG pipeline for PMC/JATS biomedical articles.

The code lives in a single `src` package. There is no plugin system; each
pipeline stage is one module and is independently callable from the CLI.

## Agentic RAG pipeline (`python -m src "question"`)

A PydanticAI agent pipeline owns all query logic. Any OpenAI-compatible
endpoint works as the LLM (Kaggle tunnel, Gemini, Ollama, vLLM — see
`.env.example`).

```
src/agents + src/orchestration              any /v1/chat/completions
  planner ──► UMLS terminology enrichment     +-----------------------+
  per-subquery loop (parallel):               | planner · verifier    |
    hybrid retrieve chunks ─► structural      | · evidence · synth    |
    units ─► BATCHED LLM relevance verify     | · rewriter            |
    └─ insufficient? coverage-driven          +-----------------------+
       LLM rewrite (rejection reasons +
       UMLS synonyms) & re-search
  evidence extraction (bounded pool) ─► aggregation ─► synthesis
```

Key architectural properties:

- **Batched verification** – all candidate units of a round go through as few
  LLM calls as possible (`VERIFY_BATCH_MAX_DOCS`, `VERIFY_BATCH_MAX_TOKENS`);
  duplicate units across subqueries are verified once via a memo.
- **Failure ≠ irrelevance** – quota/timeouts/parse errors mark units `unknown`;
  they are retried on later rounds and reported in `warnings`, never silently
  dropped like judged rejections.
- **One shared rate limiter** – every agent draws from a single token bucket
  (`GLOBAL_TOKENS_PER_MIN`); 429 `Retry-After` hints penalize the bucket so
  concurrent callers pause together.
- **Singleton resources** – BM25/corpus/structural-unit indexes load once per
  process, not once per tool call.
- **Coverage-driven rewriting** – when a subquery finds too few DISTINCT
  papers, the next query is an LLM rewrite conditioned on the verifier's
  rejection reasons plus UMLS/MeSH synonyms (`REWRITE_ENABLED`).
- **Balanced synthesis** – evidence is selected round-robin ACROSS subqueries,
  prompts carry full provenance (doc/chunk/section/evidence ids), and citations
  are repaired against real evidence ids.
- **Funnel metrics** – every run returns counters for attrition per stage
  (`funnel` in the result dict, printed by the CLI).

### Run it

```bash
cp .env.example .env        # pick provider + key
uv sync
uv run python -m src "Which surgical repair techniques were \
    associated with early recurrent coarctation, with percentages and p-values?"
```

**Run logging.** The terminal stays quiet by default; a full trace is written
to `logs.txt` (`LOG_FILE` to change). Long chunk/unit texts are truncated to
1500 chars unless `TRACE_FULL_TEXTS=1`.

### Agent architecture

| Component | File | Behaviour |
| --- | --- | --- |
| Query Planner | `src/agents/planner.py` | typed `QueryPlan` (subqueries); UMLS enrichment is deterministic, orchestrator-side |
| Retrieval tool | `src/agents/retriever.py` | shared-service hybrid search (BM25/SPLADE + dense) |
| Verifier | `src/agents/verifier_new.py` | batched unit↔intent classification; `unknown` on failure |
| Evidence Extraction | `src/agents/evidence.py` | per (subquery, unit), tolerant verbatim-quote grounding |
| Evidence Aggregator | `src/agents/evidence.py` | deterministic grouping/dedupe/contradictions |
| Query Rewriter | `src/orchestration/orchestrator.py` | coverage-driven LLM rewrite + deterministic fallback |
| Final Synthesis | `src/agents/synthesizer.py` | answer from balanced verified evidence only |
| Orchestrator | `src/orchestration/orchestrator.py` | control flow, memoization, funnel metrics |
| Shared runner | `src/llm/run.py` | structured output → text+JSON fallback, retries |
| Rate limiting | `src/llm/ratelimit.py` | process-wide token bucket, Retry-After aware |

Run the tests: `uv run pytest -q`.

## Layout

```
src/medrag/
  models.py          shared dataclasses (chunking + retrieval)
  classification.py  section classification rules
  parser.py          JATS/PMC XML -> Document AST
  chunker.py         Document AST -> retrieval chunks
  validators.py      AST and chunk validators
  pipeline.py        chunking pipeline runner (multiprocessing)
  collector.py       PMC OA article downloader (multithreaded)
  embedding.py       MedCPT article-encoder embedding runner
  retrieval/         dense, sparse, hybrid, query, reranker, corpus,
                     index builder, engine, evaluation, and CLI
```

Data and derived artifacts live outside the package in `data/`, `chunks/`,
`embeddings/`, and `index/`.

## Installation

```bash
uv sync                 # install deps + the medrag package (editable)
```

This exposes both `python -m medrag.<module>` and console scripts:
`medrag-collect`, `medrag-chunk`, `medrag-embed`, `medrag-retrieval`.

The project defaults to CPU-only `torch`. To use a CUDA GPU instead:

```bash
uv pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130 --reinstall
```

## Pipeline stages

Each stage is resumable and accepts `--limit N` to cap the number of files
processed (`0` = all). `--workers` enables parallelism where the stage
supports it.

### 1. Collect PMC articles

```bash
medrag-collect --topic cardiology \
    --output-dir data/raw/cardiology --workers 20 --limit 100
```

Uses NCBI E-utilities to find PMC IDs and downloads the public JATS XML from
`pmc-oa-opendata` with a multithreaded S3 client.

### 2. Chunk documents

```bash
medrag-chunk --dir data/raw/cardiology --chunks chunks --workers 8 --limit 100
```

Runs `parse -> AST validate -> chunk -> chunk validate` and writes one
`chunks/{stem}.parquet` per document. Documents that already have a chunk file
are skipped. Only documents that pass validation are written by default; pass
`--save-failed` to persist output for documents whose validation failed.
Prose sizing is tunable via `--max-prose-chars` (target) and
`--hard-max-prose-chars` (absolute cap above which paragraphs are split).

### 3. Generate embeddings

```bash
medrag-embed --input-dir chunks --output-dir embeddings --limit 100
```

Encodes every `retrieval_eligible` chunk with `ncbi/MedCPT-Article-Encoder`
(`[CLS]` last hidden state) and writes `embeddings/{stem}.embeddings.parquet`.
Inference is batched and single-process/GPU-bound, so this stage does not
parallelize across files.

### 4. Build and query the index

```bash
medrag-retrieval build-index \
    --chunks-dir chunks --embeddings-dir embeddings --index-dir index \
    --k1 1.5 --b 0.75

medrag-retrieval search \
    --query "diabetes mellitus treatment" --method hybrid --top-k 10 --show-text \
    --weight-dense 0.7 --weight-sparse 0.3

medrag-retrieval evaluate \
    --eval-file eval/queries.json --index-dir index --report-json eval/report.json
```

The build-index step also accepts `--dense-type` (`flat` or `hnsw`); the
query commands accept `--fusion` (`rrf`/`minmax`), `--rrf-k`, and per-branch
`--weight-dense`/`--weight-sparse` for the fusion step.

For the paper-first, intent-aware retrieval V2 pipeline (planner, per-query
hybrid retrieval, paper selection, local search, MedCPT reranking, intent
scoring and coverage), see `docs/retrieval_v2.md`.

## Tests

```bash
uv run pytest -q
```
