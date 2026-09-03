# Singular DeepAgents Flow — Merge Plan

> STATUS (2026-09-03): IMPLEMENTED. `src/x_deepagents` → `src/agents`,
> v3 retired, all phases below done and verified (337 passed; live run
> reached synthesis with 12 verified items).

Goal: keep `backend/src/x_deepagents` as the **only** agentic layer, fix its
known defects, port the v3-only capabilities worth keeping, then retire the
`agentic/` + `agents/` pipeline and all dual-engine plumbing.

Status: PLAN ONLY. No code changed yet. Two further background studies
(normal-flow detail, shared-infra detail) are still pending; fold their
findings in before implementing.

## 1. Current state (verified)

### 1a. The keeper: `x_deepagents`
- Graph (`src/x_deepagents/graph.py:530-564`): `START → decompose →
  research_worker` (Send fan-out, parallel, semaphore 4) `→ join → conflict →
  resolution → gap_resolution → synthesize → END`. Conditional edge routes to
  `join` when decompose yields no tasks.
- State (`src/x_deepagents/state.py`): `XDeepRunState` + `EvidenceItem` /
  `ResearchRequirement` / `RunBudget` / `Phase` + full enum set. Invariant:
  retrieved text is not evidence until verified (`RETRIEVED → UNDER_REVIEW →
  ACCEPTED / REJECTED / CONTRADICTORY`).
- Agents (`agents/stages.py`): all stage agents built with
  `deepagents.create_deep_agent`, role-routed models (think = qwen/groq/google/
  opencode, verify/contradiction/resolution = paced mistral 1 req/1.5s).
  Reuses prompt texts (`master, search_planner, replanner, critic,
  deep_inspector, contradiction, resolution, synthesize, reliability,
  gap_probe, gap_complete`) via `reuse.load_prompt` — never v3 agent classes.
- Tools: `retrieve` (parquet hybrid), `postgres_search` (PRIMARY, pgFTS +
  pgvector over `medrag.chunks`), `searxng_search` (trust-gated web),
  `umls_lookup`, plus direct-call `fetch_page_text` and `site_reputation`.
- Entry: `POST /v1/chat/stream` with `engine=xdeep` (or `XDEEP_ENGINE=1`) →
  `bridge.stream_xdeep`, same NDJSON wire contract. CLI: `python -m
  src.x_deepagents "<q>" --run [--json]`.

### 1b. The retired: v3 (`agentic/` + `agents/`)
- `AgenticV3Pipeline` (`src/agentic/pipeline.py`): master → parallel workers
  (SearchPipeline + DeepInspectionPipeline in `worker_pipelines.py`) →
  contradiction → resolution → synthesis. Runtime-authoritative budgets,
  per-stage `asyncio.wait_for`, memory prepare/record hooks, `V3Events` queue.
- State (`src/agentic/state.py`) is a near-duplicate of xdeep state
  (`V3RunState`, `CriticRelevance` vs `VerdictRelevance`,
  `RetrievedPaper/CriticVerdict/VerifiedEvidence` vs `EvidenceItem`).
- v3-only capabilities xdeep lacks: **memory layer wiring** (`prepare_run` /
  `record_run`, `memory_prepare/commit` events), LLM-call trace sink
  (`set_llm_event_sink`), per-sentence `_compose_cited` citations.

### 1c. Shared / dual plumbing to unify
- `backend/api.py`: `_engine_selected` + two full streaming paths + duplicated
  citation helpers (`_tokens/_best_source/_chunks/_src/_compose_cited` vs
  `bridge._cited_text/_verified_sources`).
- `graph.py:51-52,87-94`: **process-global `_event_sink/_progress_sink`** —
  concurrent xdeep requests clobber each other's streams. Must fix.
- Frontend already defaults to `engine: "xdeep"` (`store.ts`, `run.ts:96`)
  and already renders `memory` events (never fired by xdeep today).

## 2. Fix list for the keeper (before / during merge)

P0 = correctness; P1 = real defects; P2 = hygiene.

- [ ] P0-1 Per-request sinks. Replace module-global `_event_sink /
  _progress_sink` (`graph.py:51-108`) with `contextvars.ContextVar` (or an
  explicit per-run sink object threaded through `run_research`). Else two
  concurrent UI requests mix traces. Repro: fire two `stream_xdeep` runs in
  parallel, observe cross-talk. (`bridge.py:228-253` attaches/detaches
  globals — racy.)
- [ ] P0-2 Worker web evidence never counts. `agents/worker.py:181-195`
  leaves verified web hits with `document_id=""`, so
  `supporting_papers()` (`state.py:290-296`, requires non-empty doc id)
  ignores them; a worker-satisfied-by-web requirement stays unsatisfied.
  Fix: stamp `web:<slug>` ids exactly like gap-fill does
  (`gap_fill.py:231-233,456-457,562-563`).
- [ ] P0-3 Resolution evidence orphaned. `stages.py:777-791` verifies new
  resolution passages but never appends them to any requirement (empty
  `run_id`, ids only in `ResolutionOutcome`). Either append to the owning
  requirement or document that resolutions are explanation-only.
- [ ] P1-1 `paper_inspect` phantom. `rules.py:51-52` + docstrings advertise a
  tool with no module and no registry entry. Either implement it (wrap
  `get_paper_retriever` from `reuse.py:37-42` — full-paragraph/table restore
  the worker's deep-inspect path wants) or delete every mention.
- [ ] P1-2 Wire the memory layer into xdeep. Today the xdeep branch
  (`api.py:453-462`, `bridge.py:13-14`) bypasses `get_memory_api` entirely.
  Port `prepare_run` (session resume + bounded context → orchestrator prompt)
  and `record_run` (persist claims/contradictions/gaps) from
  `agentic/pipeline.py:164-253`, emitting the same `memory_prepare/commit`
  shapes the frontend already handles (`frontend/src/lib/run.ts:252`).
- [ ] P1-3 Budget disconnect. `state.RunBudget()` defaults rule everywhere;
  `AppConfig.agentic_v3_*` ignored; worker rebuilds a fresh `RunBudget`
  (`worker.py:258-261`) losing gap counters; only `searches_used`
  aggregates at join. Decide ONE budget owner (recommend `RunBudget` +
  env overrides already in `timeouts.py`) and thread the same object.
- [ ] P1-4 Synthesis guard is advisory. `graph.py:442-447,466-467` logs
  `rules.validate_state` violations but synthesizes anyway. Either block
  (fail loud) or downgrade `rules.py` to documented warnings — no silent
  middle state.
- [ ] P1-5 Gap web-augment starvation. `XDEEP_WEB_AUGMENT=1` + shared
  `gap_web_searches_used` lets augmentation (`gap_fill.py:809-838`) eat the
  budget real gaps need. Split the counters or cap augmentation (e.g. 1
  search per satisfied req).
- [ ] P1-6 `confidence_float` dead attr. `bridge._src:162,176` reads an
  attribute `EvidenceItem` never sets → scores always None. Either populate
  from verifier confidence or drop the field.
- [ ] P2-1 Delete dead code: `agents/hello.py` (`make_research_agent`
  unused), stale `worker.py:1` splice comment (no `_splice_graph.py`
  exists), unused `reuse.get_paper_retriever/get_terminology_enricher`
  (or USE paper retriever for P1-1), `build_gemma_model` alias once
  callers are updated.
- [ ] P2-2 CWD-relative log path (`logging.py:30-31`). Resolve default
  `logs.txt` against `backend/` explicitly; keep `XDEEP_LOG_FILE` override.
- [ ] P2-3 `web:<slug>` 80-char truncation can collide distinct URLs;
  hash the URL (e.g. sha1:12) into the id.
- [ ] P2-4 Contradiction serial cost: per-contradiction resolution is
  serial on paced mistral; bound with max-contradictions cap + timeout
  accounting already present, just verify under load.

## 3. Merge steps (order matters)

### Phase A — fix the keeper (no deletion)
1. P0-1 (contextvar sinks) + regression test: two parallel `stream_xdeep`
   runs, assert no cross-talk.
2. P0-2 + P0-3 (evidence counting/orphans) + unit tests on
   `supporting_papers()` and resolution append.
3. P1-2 memory wiring (behind `MEMORY_ENABLED`, same as v3).
4. P1-1 `paper_inspect` decision (implement vs delete).

### Phase B — single wire path
5. `api.py`: delete `_engine_selected`, the v3 `gen()` branch, `STAGE`,
   `_Emitter`, `_compose_cited/_cite_lines/_evidence_sources`; keep ONE
   path → `stream_xdeep(question, conversation_id)`. Keep `/v1/articles`
   proxy untouched.
6. Unify citation helpers into one module (recommend keep
   `bridge._cited_text` marker semantics; port v3's per-sentence fallback
   only if UI shows uncited prose after merge).
7. Unify progress/stage vocabulary (`STAGE_MARKERS` becomes the single map).

### Phase C — retire v3
8. Delete `src/agentic/`, `src/agents/`, `src/tools/` v3-only wrappers
   (`paper_retriever.py` moves only if P1-1 uses it), `src/__main__.py`
   v3 CLI (xdeep `__main__` becomes `python -m src` or keep both names
   pointing at xdeep during transition).
9. Parity checklist — v3 behaviors to verify in xdeep before deleting
   (from normal-flow study; do NOT port the buggy code, verify the
   capability):
   - Deep inspection: v3 `DeepInspector` (quote-grounded, critic re-judge,
     `source=DEEP_INSPECTION`) vs xdeep worker deep-inspect
     (`worker.py:431-465`) — confirm findings get verified ids that count.
   - Citation repair: v3 `_repair_citations` (`synthesize.py:122-130`) is
     itself broken (compares requirement/doc ids against evidence ids,
     strips everything; UI masked it via overlap re-citing). Audit
     `bridge._cited_text` marker→`[n]` resolution for the same class of
     id-mismatch bug instead of porting.
   - Planner writeback: v3 `SearchTermPlanner` writes LLM queries to
     `plan.requirements[0]` regardless of which req was planned
     (`search.py:133-149`) — confirm xdeep planner/replanner binds
     queries by requirement id.
   - Accounting honesty: v3 `WorkerReport.searches_used` counts rounds,
     not queries (`worker.py:131` vs `worker_pipelines.py:71`) —
     under-reports ~3x. xdeep should report per-query counts.
   - Deterministic fallbacks to keep: `_single_hop_plan`
     (`pipeline.py:326-354`), per-worker isolated budget copies
     (`pipeline.py:365`), `_worker_failure` honest-empty reports,
     `detect_contradictions_deterministic` — xdeep already mirrors most
     (DecomposeError loud-fail, UNRESOLVED-with-explanation); confirm
     each has an xdeep equivalent.
10. Tests: delete `test_agentic_v3_*`; port any memory/LLM-sink coverage
    into `test_xdeep_*`. Keep `conftest._isolate_xdeep_log`.
11. Frontend: remove engine selector/state (`store.ts`, `types.ts`,
    `backend.ts`, `run.ts:96`), always-xdeep; `memory` rendering stays
    (now actually fed). Remove "v3 compat" comments.

### Phase D — unify leftovers
12. State: single state module (xdeep's). Delete `agentic/state.py`.
13. Config: `RunBudget` defaults + `XDEEP_*` env become the single budget
    story; remove `agentic_v3_*` from `AppConfig` (or map them).
14. Rules: fix `contradiction_input_ready` enum-vs-string
    (`rules.py:139`), or delete if graph keeps calling conflict
    unconditionally.
15. Docs: update `x_deepagents/README.md` (drop "candidate" language),
    `backend/README.md`, `DESIGN.md`/`PRODUCT.md` if they describe v3.

## 4. Verification
- `uv run pytest tests/test_xdeep_* -q` green; then full suite after Phase C.
  (`tests/test_api_memory.py` 4 tests + `test_memory_continuity.py` 6 tests
  cover the memory frames — port, don't drop.)
- CLI live run: `uv run python -m src.x_deepagents "<q>" --run` retrieves,
  verifies, synthesizes with citations.
- API: `uvicorn api:app`, `POST /v1/chat/stream {"question": ...}` →
  status/sources/token/done + pipeline trace; two concurrent requests,
  no cross-talk (P0-1 test).
- UI: research panel streams, sources render, memory events appear when
  `MEMORY_ENABLED`, no engine switch.
- `backend/logs.txt` shows `[xdeep]` parity for every run.

## 6. Hard constraints (from shared-infra survey — do not break)
- Wire contract byte-stable: `status/sources/token/done/error/memory/
  pipeline` types, `memory{kind:prepare|commit}` first-class frames (NOT
  pipeline-wrapped), `_RESERVED` set, word-aligned `_chunks`, `done.
  timingMs`. `frontend/src/lib/run.ts:95-270` pattern-matches ~20 pipeline
  event names; renaming breaks tool cards. `sources` frames REPLACE
  (merge by id/pmcid/title, `parts.ts:145-163`); steps capped at 260.
- Stage values stay within ResearchStatus (`understanding/decomposing/
  retrieving/reranking/verifying/synthesizing/complete`); lamps derive
  from distinct emitted stages only.
- Citation parity: every prose sentence cited, numbering == `sources`
  order (cap 12, dedupe by document).
- Retrieval: keep `RetrievalService` singleton + query-text cache +
  BM25→PgFTS fallback + `union_rerank_diversify(top_k=max_documents)`;
  never send full documents to the verifier (expand→units).
- Memory: facade-only (`MemoryAPI`), per-chat session boundary, evidence
  boundary (memory advisory-only, MODEL_INFERENCE labeled), append-only
  claims.
- LLM: v3 = PydanticAI agents, xdeep = LangChain/deepagents models — do
  NOT mix call paths without an adapter (opencode Responses-API
  translation lives only on the LangChain side). Preserve choke-point
  semantics: shared token bucket, 429 fail-fast, structured→text
  fallback, `llm_call` sink via contextvar (`src/llm/run.py`,
  `src/llm/ratelimit.py`, `src/llm/client.py:17-22`).
- Deps floors: `deepagents>=0.7.12`, `langgraph>=1.2.11`, py3.12+; lazy
  torch/transformers; `requests` (not httpx) for the PMC proxy (NCBI
  TLS-fingerprinting); `/v1` Vite proxy + `conversation_id` threading +
  CORS + `logs.txt` trace keep working.

## 5. Decision points (defaults assumed)
- Delete v3 outright (git history preserves it) — no archive dir.
- Memory layer is must-keep (frontend already renders it).
- Think-provider routing (`config.py`) stays as-is; opencode path gets a
  runtime smoke test (`src/llm/opencode_client.py` EXISTS — verify import).
