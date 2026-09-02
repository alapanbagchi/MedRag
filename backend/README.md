# MedRAG

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
the pipeline stays on the global model (Gemma). Critic verifications run as **parallel single requests**: every passage
retrieved in a worker round is judged concurrently, one LLM call per passage,
with requests started at most one per second (1 req/s) so free-tier rate
limits are respected - the Mistral free tier cannot submit Batch API jobs, so
batching is not used.

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

- **Parallel paced verification** – every candidate unit of a round is judged
  as its own single request, run concurrently at 1 req/s;
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

## Memory + Context layer (`src/memory`)

Persistent, temporal, provenance-aware research state + context construction.
The governing invariant: **memory is not medical evidence** — the PMC corpus
and the verified-evidence pipeline are the only authority for medical fact;
memory provides continuity, prior research state and personalization, and it
can never silently become a source of unsupported medical truth.

Layers (never blurred): L0 conversation (conversations/messages/summaries),
L1 working research state (active session questions/gaps), L2 persistent
research memory (sessions, claims, contradictions, gaps, preferences), L3 the
evidence corpus — L3 is referenced from memory **only** through
`EvidenceReferenceRecord` (PMCID/PMID/DOI + chunk + verification status,
never a copy of evidence).

Run with memory attached:

    uv run python -m src --memory "Does vitamin D lower blood pressure?"
    # or: MEMORY_ENABLED=1 uv run python -m src "..."

What happens:

* **Before the run** — `prepare_run` resumes/identifies the research session
  and assembles a **bounded, budgeted memory context region** composed from
  typed blocks (conversation / working research state / persistent memory /
  user preferences) with explicit `VERIFIED EVIDENCE` vs `PERSISTENT RESEARCH
  MEMORY` boundary markers. The planner receives it as *advisory context only*
  — it can shape decomposition but is never citable as a source.
* **After the run** — `record_run` persists the research questions, the
  verified evidence references, **evidence-derived claims** (each with a full
  lineage claim → links → evidence → PMCID), the contradictions (both sides,
  dimension-attributed, never collapsed), the gaps and a labeled
  `MODEL_INFERENCE` conclusion. Re-running related questions resumes the same
  session and near-duplicate claims are deduplicated (evidence links unioned,
  never discarded).
* **Write pipeline** — candidate → classification → schema validation →
  **provenance gate** (an `EVIDENCE_DERIVED_CLAIM` requires ≥1 verified
  evidence reference; otherwise it is degraded to a labeled
  `MODEL_INFERENCE`, never silently upgraded) → dedup → relation/temporal
  update → commit (append-only, every write audited in `memory_events`).
* **Background consolidation** — `consolidate()` derives claim statuses from
  evidence links, merges near-duplicate claims (older ones `SUPERSEDED`, kept
  for history), detects contradictions from opposing polarity, and flags
  staleness (`MEMORY_STALENESS_DAYS`, 0 = disabled — staleness then only from
  explicit `mark_stale` and contradicting-evidence triggers).
* **Follow-up continuity** — a completed run marks its session's questions
  `answered` and stores the run's conclusion as the session's rolling
  summary. On the next related question the planner therefore receives prior
  questions as `ALREADY INVESTIGATED — do NOT re-derive`, the established
  conclusion surfaced up front, and mandatory PLANNING RULES ("plan ONLY for
  the NEW question; follow-ups build ON prior findings, they are not the prior
  question re-run"). Short follow-ups ("what about older adults?") also get
  session-question vocabulary expansion + a continuity baseline so prior
  findings are actually recalled (see `tests/test_memory_continuity.py`).

Storage: one PostgreSQL schema `medrag_memory` (same instance as the corpus;
`scripts/memory_init.py` creates it idempotently, pgvector optional for the
embedding columns). `MEMORY_BACKEND=memory` runs fully in-memory (tests/local).

Configuration (see `src/memory/config.py`):

| Env var | Default | Meaning |
| --- | --- | --- |
| `MEMORY_BACKEND` | `auto` | `auto` (Postgres, fallback memory) · `postgres` · `memory` |
| `MEMORY_EMBEDDER` | `hash` | `hash` (offline deterministic) · `medcpt` (768-dim, needs weights) |
| `MEMORY_EMBED_DIM` | `256` | embedding width (must match schema at init) |
| `MEMORY_CONTEXT_TOKENS` | `1800` | total budget of the memory/context region |
| `MEMORY_CLAIM_MIN_SIM` | `0.86` | near-duplicate threshold for dedup/merge |
| `MEMORY_RETRIEVE_MIN_SIM` | `0.30` | semantic floor of vector recall |
| `MEMORY_STALENESS_DAYS` | `0` | revalidation horizon (0 = disabled) |

Tests: `tests/test_memory_*.py` (52 tests) cover the provenance gate
(contamination can never become evidence), store lifecycle + temporal
supersession, hybrid retrieval precision, context budgets/boundaries, the
run-record path and cross-session continuity through the real pipeline.

Frontend linkage (`backend/api.py` + MedPat web): the API keeps one
`MemoryAPI` per process (`auto` backend) and attaches it to every
`/v1/chat/stream` run; `tests/test_api_memory.py` pins the wire contract.
The stream emits first-class `memory` events — `{"type":"memory",
"kind":"prepare", ...}` (research session resumed + prior claims surfaced
into the planner, advisory only) before the run and `{"type":"memory",
"kind":"commit", "stats": {...}}` (what was persisted) afterwards. The
frontend (`lib/rag-client.ts`, `lib/types.ts`, `ChatView`, `AssistantMessage`)
normalizes those events and renders a compact **research-memory strip** under
each response (`MEM · sess … · N prior claims · … recorded N claims`),
explicitly labeled as advisory context — never evidence. `MEMORY_ENABLED=0`
disables the layer on the API side.

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
  chunking/          md_chunker.py  chunker.py  classification.py
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

### 4. Chunker v2 — Markdown-native (optional, no LLM)

The XML pipeline (`medrag-chunk`, above) is untouched. A separate **chunker
v2** chunks the *Markdown* files instead (`src/chunking/md_chunker.py`), emitting the
same `Chunk` schema so the embedding/retrieval stages work unchanged:

```bash
make chunk-md                                        # data/md -> chunks_v2 + units_v2
make chunk-md MD_INPUT=data/md/cardiology CHUNKS_OUT=chunks_v2 UNITS_OUT=units_v2
# or directly (tokens/overlap/lexicon/global-dedup/overwrite):
python -m src.chunking.md_chunker --input data/md --chunks-out chunks_v2 --units-out units_v2 \
    --max-tokens 480 --overlap 0.12 --lexicon entities.json --global-dedup
```

What it adds over the XML chunker:

- **Structure-first on Markdown**: headings → sections (`##`/`###` nesting),
  YAML front matter → per-chunk metadata, header-path + article title/journal/
  date injected into every `embedding_text`.
- **Token-budget prose**: chunks sized to the embedding model's context
  (~480 tokens), oversized paragraphs split at sentence boundaries, and
  **tail overlap** between consecutive prose windows (no LLM).
- **Tables**: summary + per-row + footnotes chunks (parent-linked to the
  summary), plus a per-**table unit** with the full table text.
- **Parent/child units** (`units_v2/{stem}.parquet`): one unit per
  section/table with its full text and the ids of its granular child chunks —
  a first-class artifact for "retrieve fine, expand to context".
- **Entity tags** (`Chunk.concept_ids`): deterministic, word-boundary matches
  against an optional local lexicon JSON (`{"CUI": ["surface forms"]}`) —
  no LLM, no network. UMLS CUIs work well as keys.
- **Exact-duplicate suppression** (per-doc always; `--global-dedup` across
  files) with `dedup_of` provenance; **incremental rebuilds** via an
  `md_sha256` sidecar (unchanged files are skipped); a per-document coverage
  report in `chunks_v2/{stem}.meta.json`.

Skipped by design (need an LLM or new embedding code): LLM-generated table
summaries / question generation, and late-chunking multi-vector indexing.
Chunk parquets stay compatible with the corpus schema
(`id, document_id, text, embedding_text, chunk_type, section, subsection,
breadcrumb, parent_id, table_id, figure_id, document_position,
retrieval_eligible`).

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
