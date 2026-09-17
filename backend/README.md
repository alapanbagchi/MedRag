# MedRAG

> NOTE (2026-09): the agentic backend is now a SINGULAR deepagents flow in
> `src/agents/` (LangGraph: decompose → parallel workers → conflict →
> resolution → gap resolution → synthesis), served at `POST /v1/chat/stream`
> with no engine switch. The v3 pipeline (`src/agentic/`, pydantic-ai agents)
> was retired. Much of the detail below predates that merge and is kept as
> history — see `src/agents/README.md` and the repo-root `MERGE_PLAN.md` for
> the current architecture.

Structure-aware RAG pipeline for PMC/JATS biomedical articles.

The code lives in a single `src` package. There is no plugin system; each
pipeline stage is one module and is independently callable from the CLI.

## Agentic RAG pipeline (`python -m src "question"`)

A PydanticAI agent pipeline owns all query logic. Any OpenAI-compatible
endpoint works as the LLM (Kaggle tunnel, Gemini, Ollama, vLLM, Mistral — see
`.env.example`). Switch providers by changing `LLM_PROVIDER` only.
Different agents can use different providers: `build_model_for(config, role)`
(`src/llm/client.py`) resolves per-role env overrides
(`ORCHESTRATOR_`, `VERIFIER_`, `SYNTHESIZER_`, `PLANNER_`, … — see
`.env.example`) with the global provider as fallback; e.g.
`VERIFIER_PROVIDER=mistral` runs just the critic on Mistral while the rest of
the pipeline stays on the global model (Gemma). Critic verification is a
**TypeSafe System One** call (Jev): one Noul probability per evidence
requirement plus one intent Noul per passage, fanned out as **one System One
request per passage**. Coverage is a code-side threshold on the returned
probability and the verbatim excerpt is selected by a Choice over code-built
spans, so the judge never generates text. See `.env.example` for the
`TYPESAFE_*` knobs.

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

- **Parallel System One verification** – each passage is one System One
  request, fanned out concurrently (bounded by `TYPESAFE_MAX_INFLIGHT`);
  duplicate passages are scored once per process via the verdict cache.
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
| Verifier | `src/tools/verifier.py` | TypeSafe System One (Jev): one Noul per requirement + an intent Noul per passage, span Choice for verbatim; rejects on empty coverage |
| Evidence Extraction | `src/agents/evidence.py` | per (subquery, unit), tolerant verbatim-quote grounding |
| Evidence Aggregator | `src/agents/evidence.py` | deterministic grouping/dedupe/contradictions |
| Query Rewriter | `src/orchestration/orchestrator.py` | coverage-driven LLM rewrite + deterministic fallback |
| Final Synthesis | `src/agents/synthesizer.py` | answer from balanced verified evidence only |
| Orchestrator | `src/orchestration/orchestrator.py` | control flow, memoization, funnel metrics |
| Shared runner | `src/llm/run.py` | structured output → text+JSON fallback, retries |
| Rate limiting | `src/llm/ratelimit.py` | process-wide token bucket, Retry-After aware |

Run the tests: `uv run pytest -q`.

## Agentic V2 — orchestrator-controlled research loop

A NEW flow (`src/agentic_v2/`), kept separate from the existing `src/agentic/`
pipeline (which is left untouched). Instead of a fixed
`decompose → enrich → retrieve → verify` sequence, an **Orchestrator Agent**
drives the research process one decision at a time against a persistent
`ResearchState` (the external research notebook — not conversational context).

```bash
uv run python -m src --agentic-v2 "How do genetic risk variants for essential \
    hypertension identified in African-derived admixed populations influence \
    medial arterial calcification, and what role could predictive modeling of \
    heterogeneous treatment effects play in personalising mTOR-inhibitor therapy?"
```

The orchestrator repeatedly:

1. inspects the current `ResearchState` (question, objectives + status, retrieved
   documents, candidate passages, verified evidence, gaps, contradictions, prior
   actions, budget);
2. chooses the **single next action** with the highest expected value;
3. runs it; folds the result back into the state; repeats;
4. **SYNTHESIZEs** when every major objective has sufficient verified evidence
   (or the gaps are well understood), or **STOPs** honestly when the budget is
   exhausted / the task is not completable.

Available actions (one per turn):

| Action | Purpose |
| --- | --- |
| `DECOMPOSE` | split distinct evidence obligations into research objectives (may recurse) |
| `ENRICH` | expand an objective's query with UMLS/MeSH preferred names + synonyms |
| `GLOBAL_RETRIEVE` | search the corpus with the enriched query; each hit chunk is expanded to its full paragraph / table / figure |
| `READ_DOCUMENT` | read a specific document / chunk / section in depth |
| `FIND_SECTIONS` | list the sections of an already-retrieved document |
| `VERIFY` | check INTENT only — is each retrieved paragraph/table/figure relevant to the objective? |
| `SYNTHESIZE` | produce the final answer from verified evidence only |
| `STOP` | end when the task cannot be completed / budget is exhausted |

Typical per-objective flow: **DECOMPOSE → ENRICH → GLOBAL_RETRIEVE → VERIFY**
(the fetched units go straight to the verifier — there is no separate local
search step), then **SYNTHESIZE** once every objective is covered.

Key properties:

- **Persistent state as memory** — `ResearchState` records everything so the
  loop never repeats failed/identical work (loop avoidance) and never re-fetches
  seen chunks.
- **Evidence-quality labels** — the verifier tags each item as
  `direct / indirect / background / contradictory / absent` and flags
  population / outcome mismatches, so indirect evidence is never presented as
  proof.
- **Clinical/causal distinctions preserved** — association ≠ causation,
  mechanism ≠ efficacy, prediction ≠ treatment effect, heterogeneity ≠
  personalized benefit.
- **Budget-aware** — bounded by `AGENTIC_V2_MAX_ROUNDS` (12) and
  `AGENTIC_V2_MAX_GLOBAL_RETRIEVES` (6); on exhaustion it synthesizes from what
  it has (or stops honestly), never fabricates.

Layout: `state.py` (research notebook), `orchestrator.py` (decision agent),
`actions.py` (executors), `verify.py` (objective verifier), `synthesize.py`
(final synthesis), `pipeline.py` (the run loop), `policy.py` (the runtime
state machine), `events.py` (event log).

### Progress-guaranteed state machine

The loop is **agentic but bounded**. The LLM only *proposes* strategy; the
runtime (`policy.py`) is authoritative:

- **Phase** is derived from state — `EXPLORE → EXCAVATE → ASSESS → ANSWER` —
  and exposed to the orchestrator as advisory context.
- **Legal-action policy** deterministically computes which actions are legal
  (e.g. `VERIFY` needs objectives + material, `SYNTHESIZE` needs evidence or an
  investigated objective, `GLOBAL_RETRIEVE` is capped by its budget). Illegal
  decisions are repaired, never executed.
- **Progress contract** snapshots state before/after each action and marks
  no-progress actions as exhausted (`strategy_history` / `exhausted_strategies`),
  forcing a deterministic pivot instead of thrashing.
- **Strategy keys** normalize queries/targets so semantically-identical attempts
  are recognized and blocked from immediate repetition.
- **Hard budgets** — `AGENTIC_V2_MAX_ROUNDS` and `AGENTIC_V2_MAX_GLOBAL_RETRIEVES`
  are enforced in code, not the prompt.
- **Async timeouts** — `AGENTIC_V2_ORCHESTRATOR_TIMEOUT` (120s) and
  `AGENTIC_V2_ACTION_TIMEOUT` (180s) wrap every top-level await; timeouts emit
  events, record the failure, and pivot/terminate instead of hanging.

New events: `phase_changed`, `policy_repair`, `no_progress`,
`strategy_exhausted`, `timeout`, `progress` (existing events unchanged).


## Agentic V3 - evidence-vetted multi-agent retrieval

A NEW flow (src/agents/), separate from the retired legacy v1/v2 flows
(src/agentic/, src/agentic_v2/ - moved to _trash/). Implements the V1 spec: instead of a flat
Query -> Top-K -> LLM pipeline (and instead of v2's single-agent state
machine), the system decomposes the question, dispatches INDEPENDENT Worker
sub-orchestrators IN PARALLEL, verifies every passage with a CRITIC gate,
requires N independent supporting papers per evidence requirement, performs
deep paper inspection for promising papers, then runs a global Contradiction
Agent and a dedicated Resolution Agent before the final answer.

    QUERY -> MASTER ORCHESTRATOR (decompose + evidence requirements + N + stop criteria)
           -> PARALLEL WORKERS (UMLS pool -> search-term selection -> retriever +
              context expansion -> CRITIC -> threshold loop -> deep paper inspection)
           -> VERIFIED EVIDENCE -> CONTRADICTION AGENT -> RESOLUTION AGENT
           -> FINAL EVIDENCE SET -> FINAL ANSWER

Run it:

    uv run python -m src --agentic-v3 "What are the dietary restrictions people with \
        hypertension should follow, and should they take more vitamin D?"

Key properties (spec section 32):

- Retrieval is iterative - the Worker re-searches with progressively
  different formulations until every evidence requirement has its N
  independent supporting papers or the budget is exhausted.
- Relevance is not evidence - a search hit is candidate evidence only; it
  becomes evidence only when the CRITIC gate judges answers_task = yes on
  the FULL expanded context (paragraph/table/figure), never a snippet.
- Verification precedes counting - only CRITIC-accepted material counts
  toward N; dedup by paper identity means the same paper retrieved by many
  searches counts once (distinct document_ids).
- Deep paper inspection ("the paper might have something in it") - a paper
  whose excerpt did not answer the requirement is read in full; an
  inspection model locates candidate findings (the vision-model role), a
  deterministic GREP/text verification proves the claimed text exists, and
  only then does the CRITIC accept or reject.
- Contradictions are expected and investigated - the Contradiction Agent
  looks ACROSS workers' verified evidence; each flagged conflict opens a
  Resolution Agent with the paper search tool; if it cannot resolve the
  conflict, uncertainty is preserved and the final answer says so
  explicitly - the system never manufactures certainty.
- Structured evidence packages - each Worker returns its task, every
  requirement's coverage (x/N), and the papers behind it (WorkerReport);
  the final answer cites only verified evidence ids (citations are
  deterministically repaired against the evidence set).

### Agentic control loop (adaptive retrieval + explicit state isolation)

The V3 workers run a genuine agentic loop instead of a linear pipeline:

    CRITIC -> sufficient? -> contradiction analysis -> synthesis
           -> insufficient? -> FAILURE ANALYSIS (why it failed, what is
              missing) -> QUERY REPLANNING (replan.py, meaningfully-
              different queries - never the same query again) -> RETRIEVE.

Upgrades over the linear flow:

- **Adaptive retrieval with replanning** - when a requirement is
  insufficiently supported, the replanner receives the FULL scoped context
  (task, requirement, previous queries, retrieved documents, rejected
  candidates + critic reasoning, accepted papers, remaining budget) and
  returns a diagnosis plus NEW queries; "meaningfully different" is
  enforced deterministically (the new query must contain a content token
  absent from every tried query).
- **Explicit evidence state machine** - every candidate walks
  RETRIEVED -> UNDER_REVIEW -> ACCEPTED / REJECTED / CONTRADICTORY, and
  every requirement walks UNSATISFIED -> PARTIALLY_SUPPORTED -> SATISFIED
  (EXHAUSTED when the budget is spent). The orchestrator drives on these
  states, never on implicit behavior.
- **Strict state isolation** - every operation carries run_id / task_id /
  requirement_id / evidence_id / document_id / chunk_id / attempt_id.
  Critic verdicts are scope-stamped BEFORE judgement (a misplaced call
  raises); each worker keeps its own local session state and budget copy;
  the orchestrator merges workers only through explicit structured objects
  (WorkerReport / ResearchTask / VerifiedEvidence).
- **Independently retryable workers** - no worker reads another worker's
  context; any task can be re-run alone with the same scoped ids.
- **Critic separated from contradiction resolution** - the CRITIC judges
  one passage vs one requirement; the Contradiction Agent + Resolution
  Agent operate ONLY on CRITIC-verified evidence (ACCEPTED/CONTRADICTORY)
  and the resolver compares study/context factors (population, exposure,
  outcome, design, baseline, dosage, duration, measurement) before judging
  RESOLVED / PARTIALLY_RESOLVED / UNRESOLVED (unresolved conflicts are
  preserved for the synthesizer).
- **Evidence-gated synthesis** - the synthesizer receives only the verified
  evidence set plus requirement statuses, gaps and contradiction analysis;
  REJECTED material is structurally excluded, every citation is repaired
  against real verified ids, and unsatisfied requirements are stated
  explicitly rather than hallucinated away.
- **Full provenance** - each evidence item retains evidence_id, run_id,
  task_id, requirement_id, attempt_id, document_id, chunk_id, search_query,
  retrieval_method, rank, the critic verdict snapshot, confidence, support
  and terminal state - the complete auditable chain from final claim back
  to search query.
- **Explicit budgets & stopping** - max_searches (query attempts),
  max_retrieval_rounds, max_papers_per_round, max_workers,
  max_deep_inspections, evidence_target (N). A worker stops when its
  requirement is satisfied, the round cap is hit, or the search / deep-
  inspection budget is exhausted - no infinite loops.
- **Observable transitions** - every state change is logged with its full
  scope: [SEARCH:T1:R1:A1], [RETRIEVAL:T1:R1:A1], [CRITIC:T1:R1:A1:E1],
  [REPLAN:T1:R1:A2], [EVIDENCE:T1.R1 -> PARTIALLY_SUPPORTED], ... in the trace log
  (evidence_state, requirement_state, replan transitions).

Layout: state.py (models + state machine), master.py, umls.py, search.py,
retriever.py, critic.py, worker.py (adaptive loop), replan.py (failure
analysis + query replanning), deepinspect.py, contradiction.py,
resolution.py, synthesize.py, pipeline.py, events.py.

Budget knobs (env, defaults in parens): AGENTIC_V3_EVIDENCE_TARGET (3),
AGENTIC_V3_MAX_SEARCHES (5), AGENTIC_V3_MAX_RETRIEVAL_ROUNDS (5),
AGENTIC_V3_PAPERS_PER_SEARCH (5), AGENTIC_V3_MAX_DEEP_INSPECTIONS (3),
AGENTIC_V3_MAX_WORKERS (4), plus per-stage timeouts (AGENTIC_V3_*_TIMEOUT).

### Observability: Pydantic AI → Logfire

By default (when Logfire credentials or a `LOGFIRE_TOKEN` are present) every
every run streams to **Logfire**. Two layers, both automatic:

* **LLM traces (PydanticAI GenAI instrumentation)** — `src/logfire_obs.py`
  calls `logfire.configure()` (once per process) and
  `logfire.instrument_pydantic_ai()` (= `Agent.instrument_all`), so *every*
  agent's model requests emit OpenTelemetry GenAI spans: system instructions,
  input/output messages, model + request parameters, token usage, cost and
  latency. Each `ask_structured` call is additionally wrapped in an
  `llm.ask_structured` span carrying the agent label, and records aggregate
  metrics (`medrag.llm.calls`, `medrag.llm.latency.seconds`,
  `medrag.llm.prompt_tokens`, `medrag.llm.output_chars`).
* **Application trace events** — `src/trace.py` forwards every event it logs
  (stages, agent calls, prompts/responses, retrieval ranks, verdicts, deep
  inspections) to Logfire as spans/logs tagged `medrag`, nested under a
  top-level `medrag.run` span per question (workers additionally carry
  `task_id`/`task_title` context). Metrics: `medrag.retrieval.calls/chunks`,
  `medrag.evidence.accepted/rejected`, `medrag.deep_inspections`.

Setup (one time):

    pip install logfire          # already a dependency (pydantic-ai[logfire])
    logfire auth                 # or: logfire project create
                                 # -> writes .logfire/logfire_credentials.json

Gating: `LOGFIRE_ENABLED` = `auto` (default; on iff credentials/token
present) | `1`/`on` (force on) | `0`/`off` (force off). Other knobs:
`LOGFIRE_SERVICE_NAME` (default `medrag`), `LOGFIRE_ENVIRONMENT`,
`LOGFIRE_CONSOLE` (`true` also prints spans to the terminal), or paste a write
token in `LOGFIRE_TOKEN`. With no token present the app stays fully offline
(no exporting).

## Prompts

Every agent system prompt is a plain-text file under `src/prompts/`, one
file per agent, so prompts can be edited without touching code:

```text
src/prompts/
  agents/        master.txt  search_planner.txt  critic.txt  deep_inspector.txt
                 replanner.txt  contradiction.txt  resolution.txt  synthesize.txt        (--agentic-v3)
```

Modules load their prompts at import time through
`src.prompts.load.load_prompt(subdir, name)`: default root is the shipped
`src/prompts/` tree; set the `PROMPT_DIR` env var to point at a different
tree (e.g. per-deployment prompt variants) - the same file layout applies.
A missing prompt file raises loudly (path included) instead of silently
using a stale prompt. Legacy pipeline reads live from these files per run.

## Layout

```
src/
  __main__.py        agentic query entry: python -m src "<question>"
  config.py          environment configuration
  ingestion/         collector.py  (PMC OA JATS download)
  processing/        jats_to_md.py  parser.py  jatstomd.py  (XML -> Markdown)
  chunking/          documents.py  parsing.py  prose.py  tables.py  sections.py ...
                     (one concern per file; pipeline entry: documents.chunk_document)
  lib/               utils.py  models.py  _torch.py  (shared support)
  retrieval/         dense, sparse, splade, reranker, pgvector, planner
  agents/            agentic_v3 LLM agents (master/worker/critic/...)
  agentic/           agentic_v3 engine (pipeline, domain state, events, worker pipelines)
  tools/             reusable non-LLM capabilities (umls, retrieval) agents call
  prompts/           agent system prompts (plain text, one file per agent)
  llm/  umls/        shared providers
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
    --output-dir data/raw/cardiology --workers 20
medrag-collect --topic cardiology \
    --output-dir data/raw/cardiology --workers 20 --min-year 2015 --max-year 2024
```

Finds PMC IDs through the local edirect binaries (`~/edirect/esearch` +
`~/edirect/efetch`) and downloads the public JATS XML from `pmc-oa-opendata`
with a multithreaded (`boto3`, unsigned) S3 client, showing a tqdm progress
bar. Downloads are resumable by file existence. `--min-year` / `--max-year`
restrict the search to a publication-date range (`[pdat]`, 0 = any). The
collector is a `PMCCollector(Collector)` implementing the
`core.protocols.Collector` protocol (`core/protocols.py`), so any component
that needs a collector can depend on the protocol instead of the class.

### 2. Convert JATS XML to Markdown

```bash
make jats-to-md                                      # data/raw -> data/md
make jats-to-md INPUT=data/raw/cardiology OUTPUT=data/md/cardiology
make jats-to-md LIMIT=100                            # only the first 100 files (0 = all)
# or directly:
python -m src.processing.jats_to_md --input data/raw/cardiology --output data/md/cardiology
python -m src.processing.jats_to_md --input data/raw/cardiology --output data/md/cardiology --limit 100
```

Reuses the structure-aware JATS parser (`src/parser.py`), so tables come out as
real Markdown tables and figures/equations/references keep their structure.
Each article becomes `{stem}.md` with YAML front matter (pmcid, title, journal,
authors, dates, keywords, DOI/PMID). Existing `.md` files are skipped (resumable;
`--overwrite` to re-convert), a tqdm bar shows progress, `--limit N` caps the run.

### 3. Chunk documents

```bash
medrag-chunk --dir data/raw/cardiology --chunks chunks --workers 8 --limit 100
```

Runs `parse -> AST validate -> chunk -> chunk validate` and writes one
`chunks/{stem}.parquet` per document. Documents that already have a chunk file
are skipped. Only documents that pass validation are written by default; pass
`--save-failed` to persist output for documents whose validation failed.
Prose sizing is tunable via `--max-prose-chars` (target) and
`--hard-max-prose-chars` (absolute cap above which paragraphs are split).

### 5. Generate embeddings

```bash
medrag-embed --input-dir chunks --output-dir embeddings --limit 100
```

Encodes every `retrieval_eligible` chunk with `ncbi/MedCPT-Article-Encoder`
(`[CLS]` last hidden state) and writes `embeddings/{stem}.embeddings.parquet`.
Inference is batched and single-process/GPU-bound, so this stage does not
parallelize across files.

### 4. Chunker v2 — Markdown-native (no LLM, SOTA shape)

The XML pipeline (`medrag-chunk`, above) is untouched. A separate **chunker
v2** chunks the *Markdown* files instead (`src/chunking` — one concern per
module, composed by `documents.chunk_document()`), emitting the same `Chunk`
schema so the embedding/retrieval stages work unchanged:

```bash
make chunks                                        # data/md -> chunks + units in medpat (ParadeDB)
make chunks DIR=data/md/cardiology                  # a different directory
# or directly (tuning / overwrite):
python -m src.chunking --input data/md --max-tokens 480 --split-overlap-sentences 2

# Browse what was stored (pgAdmin frontend on the medpat container):
make medpat-up      # postgres + pgAdmin -> http://localhost:5050
```

What it adds over the XML chunker:

- **Pure chunking, no embedding logic**: `embedding_text` is the chunk text
  (encoding is a separate step — `make medpat-embed` — that POSTs the stored
  `embedding_text` to the OpenAI-compatible MedCPT server at
  `EMBEDDING_BASE_URL`).
- **Structure-first on Markdown**: headings → sections (`##`/`###` nesting);
  YAML front matter → per-chunk metadata.
- **Token-budget prose with healthy split overlap**: chunks sized to the
  embedding model's window (~320 soft / 640 hard tokens), oversized
  paragraphs split at sentence boundaries with **abbreviation-aware**
  splitting (`e.g.`, `Fig.`, `No.`, decimals are never cut), and a healthy
  intra-paragraph carry — the last `--split-overlap-sentences` (default 2)
  sentences seed the next piece, token-budgeted to ~25% of `--max-tokens`
  (`--split-overlap-tokens`, capped at half).
- **Late-chunking hooks** (no duplicated vectors): a sentence-split paragraph
  emits a **paragraph unit** (full source text + child piece ids) and every
  piece carries `metadata.sentence_split` (piece_index / piece_count /
  paragraph_unit_id) — an encoder can embed the full paragraph and slice
  piece spans without any LLM.
- **Tables**: summary + per-row + footnotes chunks, ALL parent-linked to the
  per-**table unit** (uniform `parent_id` = "the unit that contains me"), with
  a per-**table unit** holding the full table text.
- **Parent/child units** (medpat.units): one unit per section, table, and
  split paragraph, each with its full text and the ids of its granular
  child chunks — a first-class artifact for "retrieve fine, expand to
  context".
- **Incremental rebuilds**: unchanged documents are skipped on re-run via
  `md_sha256`.

There is no LLM path: table summaries stay structural, and late-chunking is
supported at chunk level (hooks above) rather than with LLM-generated text.
Chunks land in `medpat.chunks` (schema `id, document_id, text,
embedding_text, chunk_type, section, subsection, breadcrumb, parent_id,
table_id, figure_id, document_position, retrieval_eligible`), units in
`medpat.units`; `medpat.chunk_embeddings` is filled by `make medpat-embed`
(OpenAI-compatible MedCPT server, L2-normalized vector(768)).

### 6. Build and query the index

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

### 7. PostgreSQL + pgvector store (embedding_v2)

Put the v2 embeddings + chunk metadata into PostgreSQL for persistent,
ann-indexed retrieval:

```bash
# 1. Start the pgvector Docker container (idempotent, never wipes data)
make pg-init

# 2. Load embeddings_v2 + chunks_v2 (idempotent upsert — safe to re-run)
make pg-load

#    ... or wipe ALL data in the medrag schema first, then load:
make pg-load RESET=1

# Other useful targets
make pg-reset    # drop everything (chunks + embeddings + indexes)
make pg-stats    # current counts + index state
make pg-init --stats-only    # retrieval smoke test
make pg-init --stats-only QUERY="Does high uric acid increase the risk of hypertension?"
```

The loader is `scripts/pg_init.py` and defaults to
`EMBEDDINGS_DIR=embeddings_v2` and `CHUNKS_DIR=chunks_v2`; point them at any
folder (`make pg-load EMBEDDINGS_DIR=/data/embs CHUNKS_DIR=/data/chunks`).

Key behaviours:

- **Reset** (`--reset` / `RESET=1`) runs `DROP SCHEMA medrag CASCADE` and
  reloads from scratch; without it the load is an idempotent upsert.
- **Vectors are already L2-normalized** (FP16 stored, normalized in FP32 at
  generation time), so the loader only upcasts FP16 → FP32 and stores them
  as-is — it does *not* re-normalize. The load summary samples vector norms
  to confirm unit length.
- **Index**: after loading it builds an HNSW index with the inner-product
  opclass (`vector_ip_ops` / `halfvec_ip_ops`), which the store's `<#>`
  cosine queries can use. Falls back to IVFFlat automatically.
- **Storage type**: `vector(768)` (float32) by default;
  `--vector-type halfvec` stores half-precision vectors (2 B/dim, requires
  pgvector ≥ 0.7 — the Docker image has it).
- **FK safety**: embedding rows whose `chunk_id` has no matching chunk row
  are skipped and reported (use `--reset` with both dirs for a full rebuild).

Loading is memory-bounded by design (at most `workers*2` parquet files are
in-flight and COPY batches are flushed frequently). On a low-RAM machine add
`WORKERS=2 FLUSH_ROWS=10000 CHUNK_BATCH=4000` to `make pg-load` to shrink
the working set further.

**Interrupt / re-run behaviour** — `make pg-load` is incremental and
interrupt-safe. Every flushed batch commits its rows and a per-file
checkpoint (`medrag.load_marks`) in one transaction, so:

- re-running after a Ctrl-C **skips** files whose rows are already in the
  DB (`... skipped N files: chunks/embeddings already present in DB`); only
  missing files are loaded;
- the HNSW index is **rebuilt only when data changed** — if it already
  exists and nothing changed, the loader says so and leaves it alone;
- a running index build shows live progress
  (`phase=... blocks=9200/56260 16%` via `pg_stat_progress_create_index`)
  and can be interrupted safely — CREATE INDEX is atomic, so a killed build
  simply leaves no index and the next run rebuilds it;
- `--reset` / `RESET=1` wipes everything (including checkpoints) for a full
  reload, and `FORCE_REINDEX=1` forces an index rebuild without touching
  the data.

Querying: `PyVectorStore` / `PgDenseIndex` in `src/retrieval/pgvector_store.py`
already implement search (`store.search(qvec, top_k=...)`,
`search_filtered(qvec, document_id=...)`), and `scripts/pg_init.py --stats-only` is a
ready-made retrieval check.

### 7b. Hybrid search (BM25 + pgvector)

```bash
make pg-bm25                                    # one-time: rebuild BM25 over chunks_v2
make pg-hybrid QUERY="Does high uric acid increase the risk of hypertension?"
# or with the fresh v2 BM25 index:
make pg-hybrid QUERY="..." BM25_DIR=index/bm25_v2
```

`scripts/pg_hybrid_query.py` runs the two branches the full retrieval
pipeline uses — BM25 (sparse, from the BM25 index) and the MedCPT query
encoder → pgvector (dense) — and fuses them with reciprocal rank fusion
(RRF). The checked-in `index/bm25` was built over the old v1 corpus (its
chunk ids don't exist in pgvector), so run `make pg-bm25` once to rebuild
it over `chunks_v2` (`index/bm25_v2`, built in parallel, same on-disk
format as `BM25Index`).

## Tests

```bash
uv run pytest -q
```

### 7. medpat Postgres (docker) — the corpus store (no disk artifacts)

The medpat pipeline persists everything (documents, chunks, units, references,
citations, embeddings, load marks) into a dedicated Docker Postgres container instead
of parquet files. The existing `medrag` container is left untouched.

```bash
make medpat-up            # start medpat postgres (port 5433) + pgAdmin UI
                          # schema + BM25 index ship EMPTY; you load the corpus
                          # yourself with `make chunks`
make medpat-ui            # pgAdmin frontend -> http://localhost:5050
make chunks DIR=data/md   # data/md -> medpat (documents, units, chunks, refs, citations)
make medpat-embed         # MedCPT Article-Encoder -> medpat.chunk_embeddings (vector(768))
make medpat-psql          # psql shell into the container
make medpat-down          # stop the containers (data volume persists)
```

- **Schema** (`backend/docker/medpat/init/01_schema.sql`, applied on first boot):
  `documents` (body_md = full Markdown source), `units` (section/table/paragraph,
  hierarchical via `parent_unit_id`), `chunks` (column-for-column the chunks_v2
  schema; `parent_id` = enclosing unit, `fingerprint` for exact-dup),
  `chunk_embeddings` (vector(768), HNSW, hash-based change detection),
  `references` + `chunk_citations` (normalized citation graph), `lexicon_terms`,
  `load_marks` (idempotent ingest), `meta`, `v_corpus_stats`.
- **Ingest**: `make chunks DIR=data/md` (or `python -m src.chunking
  --input data/md`). Deterministic ids + ON CONFLICT upserts + md_sha256 skip
  make re-runs idempotent (unchanged docs are skipped). References/citations
  and the FTS `tsv` column are maintained by the writer itself (plus a
  Postgres trigger).
- **Embed**: `python -m src.embedding [--limit N] [--model ...]`
  (MedCPT Article-Encoder, CLS pooled, L2-normalized vector(768)).
- **Connection**: `MEDPAT_DSN` env or `--dsn`, default
  `postgresql://medpat:medpat@localhost:5433/medpat` (password via
  `MEDPAT_PG_PASSWORD` when the container is created). Runtime retrieval reads
  schema `medpat` by default (`MEDPAT_PG_SCHEMA=medrag` points at the legacy
  container).
`make medpat-up` also starts **pgAdmin**, the browser frontend for the
medpat database (pre-registered "medpat (ParadeDB)" server; login defaults
`admin@medpat.io` / `medpat`, overridable via `PGADMIN_EMAIL` /
`PGADMIN_PASSWORD` in `backend/.env`).

