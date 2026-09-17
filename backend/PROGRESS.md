# MedRAG — Agentic Retrieval System: Progress Log

Status: COMPLETE (end-to-end validated)
Last updated: (session write)

This file records the incremental build of the agentic retrieval system, one
step at a time, as requested. Each step was implemented, unit-tested, and (where
the LLM backend was reachable) validated live before moving on.

----------------------------------------------------------------------------
GOAL
----------------------------------------------------------------------------
Rebuild MedRAG retrieval so that:
  1. The LLM decomposes the query into subqueries (intent + evidence_required
     + extracted medical entities).
  2. Each subquery's entities are enriched via UMLS.
  3. Each subquery runs hybrid retrieval + reranking.
  4. Retrieved chunks are restored to their full paragraph and LLM-verified
     (keep / reject).
  5. The retriever + UMLS are exposed as TOOLS the LLM decides to call, looping
     until it finds sufficient evidence or gives up.

Validation query: the radial-artery-vasospasm (IR vs CABG) + KID-ACS/PENK AKI
compound question.

----------------------------------------------------------------------------
ARCHITECTURE (new package: src/agentic/)
----------------------------------------------------------------------------
Legacy deterministic pipeline left untouched.

  src/agentic/planner.py      Step 1 — decompose query -> subqueries
  src/agentic/umls_tool.py    Step 2 — UMLS enrichment folded into queries
  src/agentic/retriever_tool.py  Step 3 — hybrid retrieve + rerank + paragraph restore
  src/agentic/verify.py       Step 4 — LLM keep/reject verification
  src/agentic/loop.py         Step 5 — agentic tool loop (retriever + UMLS tools)
  src/agentic/pipeline.py     End-to-end orchestrator (decompose -> enrich -> loop)
  src/__main__.py             --agentic entry-point flag (python -m src --agentic "<query>")

Tests: tests/test_agentic_*.py  (44 tests at completion)

----------------------------------------------------------------------------
STEP 1 — Robust query decomposition  [DONE]
----------------------------------------------------------------------------
- DecomposePlanner (pydantic_ai Agent, output_type=Decomposition).
- SubQueryPlan carries: id, target, intent, query, focus, evidence_required,
  entities (text + role).
- Post-processing: _clean_text (strips JSON-echo junk), _is_junk_entity,
  _infer_role (lexical role default: anatomy/procedure/biomarker/drug/condition),
  _fallback_entities (deterministic extraction when model emits none),
  _dedupe_subqueries (drops ~80%-overlap near-duplicates, caps at MAX_SUBQUERIES),
  _canonicalize_ids.
- Fallbacks for intent + evidence_required when Gemma truncates its JSON after a
  big <thought> block (added during diagnosis round).
- Verified: test query decomposed into 3-4 clean subqueries (H1 IR access,
  H2 CABG harvest, H3 KID-ACS/PENK, H4 PENK-vs-creatinine).
- Config: MAX_SUBQUERIES (4), MAX_AGENT_ROUNDS (6), AGENT_MIN_EVIDENCE (3).

STEP 2 — UMLS entity enrichment  [DONE]
----------------------------------------------------------------------------
- UMLSEnricher reuses cached, async, transport-injectable UMLSClient.
- enrich_subquery / enrich / apply_to_query fold preferred-names + synonyms into
  each subquery's retrieval query (capped to keep BM25 meaningful).
- _usable_term() filter drops garbage: code-terms ("AKI 001"), off-domain
  dictionary matches ("Active Site", "Social Stratification"), MeSH-inverted
  forms, over/under-sized terms, exact surface-form echoes.
- umls_lookup_tool() is the Step-5 hook (invocable as an agent tool).
- Verified: H3 got "pro-enkephalin" folded in; "AKI 001" / "Active Site" etc.
  correctly discarded.

STEP 3 — Hybrid retrieval + rerank + paragraph restore (tool)  [DONE]
----------------------------------------------------------------------------
- HybridRetrieverTool.search(sub, top_k, exclude_chunk_ids, restore_paragraphs).
- Adapts SubQueryPlan -> legacy SubQuery; calls shared RetrievalService (BM25 +
  dense union -> intent rerank -> paper diversification).
- Restores each chunk to its FULL containing structural unit via
  StructuralUnitIndex (whole paragraph / table / figure), async (to_thread),
  deduped, truncated at MAX_PAPER_TOKENS.
- Returns serializable RetrievalResult (rank, ids, section, unit_kind, score,
  methods, paragraph_text).
- Honors exclude_chunk_ids for later rounds.
- Verified live: H1-H4 surfaced correct paragraphs incl. the proenkephalin ACS
  cohort paper (PMC11695896); pgvector down -> FAISS fallback worked.

STEP 4 — LLM verification (keep/reject) as a tool  [DONE]
----------------------------------------------------------------------------
- VerifyTool.verify(sub, results, base_query) wraps batched VerifierAgent.
- Maps relevant/partially_relevant -> keep; not_relevant -> reject (reason kept);
  unknown/missing-verdict/infra-failure -> unknown (retryable, never silent reject).
- VerificationOutcome: kept/rejected/unknown + rejection_reasons + distinct_papers.
- Verified live: kept vasospasm-relevant paragraphs (conf 0.90/0.80), rejected
  irrelevant units with concrete reasons.

STEP 5 — Agentic tool loop (retriever + UMLS tools)  [DONE]
----------------------------------------------------------------------------
- AgenticLoop (pydantic_ai Agent, deps_type=AgentDeps, output_type=EvidenceReport).
- Tools registered via agent.tool(...): search_subquery (retrieve->restore->
  verify in one call) and umls_lookup (UMLS preferred-name + synonyms).
- The LLM owns the loop: issues queries, reads rejection reasons, umls_lookups,
  revises queries, stops when enough DISTINCT papers are kept or gives up.
- Shared state in AgentDeps (seen chunks, kept evidence, round count) prevents
  re-fetching the same chunk.
- Verified live: H3 loop succeeded (6 search refinements, citations PMC11695896),
  sharing the proenkephalin/KID-ACS findings with an honest CABG-cohort caveat.

END-TO-END + WIRING  [DONE]
----------------------------------------------------------------------------
- AgenticPipeline.answer(query): decompose -> enrich_all -> per-subquery loop ->
  merge into {question_type, num_subqueries, subqueries[], succeeded_subqueries,
  citations[], evidence_excerpts[]}.
- --agentic flag wired in src/__main__.py.
- Full live run on the test query: 2/4 subqueries succeeded (H1 vasospasm-IR,
  H3 KID-ACS/PENK); H2/H4 hit Gemini free-tier 429 quota and failed honestly
  (no fabrication). Citations: PMC11695660, PMC11876165, PMC11980825,
  PMC11695896, PMC11942029.

----------------------------------------------------------------------------
VERIFICATION SUMMARY
----------------------------------------------------------------------------
- Unit tests: 44 agentic tests passing; full suite 131 passed (excluding 5
  pre-existing broken retrieval_v2 tests whose source package is absent).
- Live runs exercised the real LLM + real index + real UMLS.

KNOWN CAVEATS / NEXT STEPS (environment, not code)
----------------------------------------------------------------------------
1. Gemini free-tier quota (16,000 input tokens/min for gemma-4-31b) rate-limits
   multi-subquery runs (HTTP 429). Pace loops or upgrade quota.
2. pgvector not running (Connection refused) -> dense retrieval falls back to
   FAISS. Start Postgres (docker-compose.yml) for pgvector dense search.
3. Gemma emits <thought> blocks that consume output budget and truncate JSON;
   mitigated by deterministic intent/evidence fallbacks in the planner.
4. (Observation only) "AKI" resolves to a UMLS drug-code concept and is
   discarded rather than remapped to "Acute Kidney Injury" — conservative, but
   acronym-expansion could be revisited.

HOW TO RUN
----------------------------------------------------------------------------
  PYTHONPATH=. .venv/bin/python -m src --agentic "<query>"

----------------------------------------------------------------------------
AGENTIC V2 — ORCHESTRATOR-CONTROLLED RESEARCH LOOP (NEW, separate flow)
----------------------------------------------------------------------------
Added a new flow (`src/agentic_v2/`) alongside the existing `src/agentic/`
pipeline (which is left untouched). It implements the Orchestrator Agent
design: a persistent `ResearchState` (research notebook) + an LLM orchestrator
that decides ONE action per turn + deterministic executors for eight actions
(DECOMPOSE, GLOBAL_RETRIEVE, SEARCH_RETRIEVED_DOCUMENTS, READ_DOCUMENT,
FIND_SECTIONS, VERIFY, SYNTHESIZE, STOP).

Files:
  src/agentic_v2/state.py        persistent research state + objective/evidence models
  src/agentic_v2/orchestrator.py ActionDecision + the decision agent
  src/agentic_v2/actions.py      executors for the eight actions (+ local BM25)
  src/agentic_v2/verify.py       objective-level verifier (evidence-quality labels)
  src/agentic_v2/synthesize.py   final synthesis from verified evidence only
  src/agentic_v2/pipeline.py     the run loop + budget/terminal handling
  src/__main__.py                --agentic-v2 entry flag
  src/config.py                  AGENTIC_V2_MAX_ROUNDS / AGENTIC_V2_MAX_GLOBAL_RETRIEVES

Tests: tests/test_agentic_v2_*.py (33 tests). Full suite now 164 passing
(excluding the same 5 pre-existing broken retrieval_v2 tests).

Run:
  PYTHONPATH=. .venv/bin/python -m src --agentic-v2 "<query>"

----------------------------------------------------------------------------
AGENTIC V2 — TRACE UI (rough visualizer)
----------------------------------------------------------------------------
Added a small stdlib-only web UI to watch the orchestrator loop live.

  src/agentic_v2/events.py   structured JSONL event stream (optional, off by default)
  scripts/v2_ui.py           tiny HTTP server (serves ui/agentic_v2.html + /events + /run)
  ui/agentic_v2.html         static page: timeline + state panel, 1s polling
  src/config.py              AGENTIC_V2_EVENTS_FILE env
  pipeline.py / actions.py   emit run_start/decision/agent_spawn/agent_output/
                             step/action_done/failure/state/run_end

Run:
  python scripts/v2_ui.py    # then open http://127.0.0.1:8090
  (or) AGENTIC_V2_EVENTS_FILE=agentic_v2_events.jsonl python -m src --agentic-v2 "<q>"

----------------------------------------------------------------------------
AGENTIC V2 — UMLS ENRICHMENT + FULL-SECTION RETRIEVAL + INTENT-ONLY VERIFY
----------------------------------------------------------------------------
Added a UMLS enrichment step to the workflow and changed the retrieve/verify
contract per request:

  * NEW action ENRICH (src/agentic_v2/actions.py:_enrich) — resolves an
    objective's entities to UMLS/MeSH preferred names + synonyms via
    src.agentic.umls_tool.UMLSEnricher, storing synonyms + enriched_query on the
    ResearchObjective (new fields: entities/synonyms/enriched_query).
  * GLOBAL_RETRIEVE now searches with the objective's enriched_query (falls back
    to the plain query) and loads the ENTIRE SECTION each hit chunk belongs to
    (_load_section_text), so VERIFY sees full context, not just the chunk.
  * VERIFY is now INTENT-ONLY: src/agentic_v2/verify.py classifies each section
    relevant / partially_relevant / not_relevant (confidence + note); the
    executor maps relevance -> evidence quality/support for the UI/state.
  * Orchestrator prompt updated to sequence DECOMPOSE -> ENRICH -> GLOBAL_RETRIEVE
    -> VERIFY.

Tests: +4 (37 agentic_v2). Full suite now 168 passing (excl. the same 5 broken
retrieval_v2 tests).

----------------------------------------------------------------------------
AGENTIC V2 — PROGRESS-GUARANTEED STATE MACHINE (refactor)
----------------------------------------------------------------------------
Refactored the loop into a bounded, runtime-enforced state machine while keeping
the agentic architecture (LLM proposes -> policy validates -> executor runs ->
progress evaluator -> state machine pivots/terminates).

  src/agentic_v2/policy.py      NEW: phase, legal_actions, strategy_key,
                                progress_snapshot/evaluate_progress,
                                validate_or_repair, fallback_action
  src/agentic_v2/state.py       + StrategyAttempt, strategy_history,
                                exhausted_strategies, sections_seen, phase
  src/agentic_v2/orchestrator.py prompt now receives phase/legal/strategy/budget;
                                _repair delegates to policy
  src/agentic_v2/pipeline.py    loop: phase -> decide(timeout) -> policy validate ->
                                snapshot -> execute(timeout) -> progress ->
                                record/exhaust -> pivot; hard budgets + wait_for
  src/agentic_v2/actions.py     FIND_SECTIONS records sections_seen; ENRICH only
                                marks enriched_query when terms found
  src/config.py                 AGENTIC_V2_ORCHESTRATOR_TIMEOUT / _ACTION_TIMEOUT
  ui/agentic_v2.html            renders phase + policy_repair/no_progress/
                                strategy_exhausted/timeout/progress events

Key invariants now enforced (never trusted to the LLM):
  * every non-terminal turn makes progress, exhausts a strategy, or terminates;
  * illegal / exhausted decisions are deterministically repaired;
  * retrieval + iteration budgets are hard boundaries;
  * top-level awaits are timeout-bounded.

Tests: +25 (test_agentic_v2_policy.py 16, test_agentic_v2_state_machine.py 9).
Full suite now 193 passing (excl. the same 5 broken retrieval_v2 tests).

----------------------------------------------------------------------------
AGENTIC V2 — N8N-STYLE WORKFLOW UI + LLM THOUGHT CAPTURE
----------------------------------------------------------------------------
Replaced the board UI with an n8n-style node graph and captured each LLM call's
prompt / thought / response.

  src/llm/run.py               + optional LLM observer: set_llm_observer /
                               reset_llm_observer; extracts <think>/<thought>
                               reasoning and notifies {label,input,thought,output}
  src/agentic_v2/llm_observer.py  NEW: bridges run.py -> event stream as llm_call
                               events (iteration via contextvar, label->agent alias)
  src/agentic_v2/pipeline.py   installs observer + sets iteration per turn
  src/agentic_v2/events.py     documents llm_call + state-machine event types
  ui/agentic_v2.html           n8n-style canvas: node per agent + bezier arrows;
                               click node -> modal (question / thought / response);
                               Ctrl+wheel & +/- zoom
  tests/conftest.py            resets the LLM observer between tests

Tests: +4 (test_agentic_v2_llm_observer.py). Full suite now 197 passing
(excl. the same 5 broken retrieval_v2 tests).

----------------------------------------------------------------------------
AGENTIC V2 — REMOVE LOCAL SEARCH; RETRIEVE -> UNIT -> VERIFY
----------------------------------------------------------------------------
Removed the SEARCH_RETRIEVED_DOCUMENTS action entirely (enum, policy, executor,
prompt, UI) and reverted retrieval to send each hit's full structural unit
(whole paragraph / table / figure) straight to the intent verifier — no
local-search step. Removed the in-memory local_bm25 helper and the section
loader; GLOBAL_RETRIEVE now keeps ``r.paragraph_text`` (unit) instead of loading
whole sections. The n8n UI's retriever node now lists fetched paragraphs
line-by-line (and the modal shows them in full).

Flow is now: DECOMPOSE -> ENRICH -> GLOBAL_RETRIEVE (unit) -> VERIFY -> SYNTHESIZE.

Tests: -6 (removed search/BM25/section-loader tests). Full suite now 191 passing
(excl. the same 5 broken retrieval_v2 tests).

----------------------------------------------------------------------------
AGENTIC V2 — FIX VERIFIER TIMEOUT (prompt-size bound)
----------------------------------------------------------------------------
The intent verifier was sending the FULL passage text (whole paragraphs/tables/
figures) with no truncation, up to 10 passages, so the prompt could exceed the
model's context / stall until the 180s action timeout. Other agents truncated
their input; the verifier was the only one that did not.

Fix (src/agentic_v2/verify.py):
  * _passage_text now truncates each passage to _MAX_PASSAGE_CHARS=2400 chars
    (with a "...truncated" marker);
  * _verify_prompt caps at _MAX_PASSAGES=6 and notes omitted passages;
  * verifier output budget raised to max(2048, agent_max_tokens) so the JSON
    verdict never truncates/retries.
  * actions.py _MAX_VERIFY_PASSAGES 10 -> 6 (aligned).

Tests: +3 (test_agentic_v2_verify.py). Full suite now 194 passing
(excl. the same 5 broken retrieval_v2 tests).
----------------------------------------------------------------------------
AGENTIC V2 — VERIFIER: ONE PASSAGE PER LLM CALL
----------------------------------------------------------------------------
The intent verifier packed up to 6 candidate passages into ONE prompt and asked
for a combined verdict with one assessment per passage. Small models mixed
passages up and skipped assessments in long lists.

Change (src/agentic_v2/verify.py): verify ONE document at a time.
  * new PassageVerdict output (relevance/confidence/note) + singular
    VERIFIER_SYSTEM_PROMPT; _passage_prompt() builds a one-passage prompt;
  * ObjectiveVerifier.verify() loops the passages sequentially — one LLM call,
    one validated JSON verdict per document — then _aggregate() folds them into
    the same ObjectiveVerdict (status supported / partially_supported /
    unresolved, confidence = max of kept passages, gap composed from rejection
    notes when unresolved);
  * per-passage calls keep the _MAX_PASSAGE_CHARS=2400 truncation and the
    _MAX_PASSAGES=6 cap; trace now logs verifier_passage events and labels each
    LLM call objective_verifier:<i>:<doc>;
  * FAILURE != IRRELEVANCE kept: a failed passage call raises (VERIFY fails ->
    policy retries) instead of silently recording not_relevant.

actions.py unchanged (same ObjectiveVerdict contract); UI verdict events and
evidence items behave exactly as before.

Tests: test_agentic_v2_verify.py rewritten for _passage_prompt/_aggregate (+3).
agentic_v2 suite: 67 passing.

----------------------------------------------------------------------------
AGENTIC V2 — VERIFIER: FULL PASSAGE TEXT + BUDGET THAT FITS SEQUENTIAL CALLS
----------------------------------------------------------------------------
Two problems after switching VERIFY to one-paragraph-per-call:
  * each prompt still truncated the passage at _MAX_PASSAGE_CHARS=2400, so big
    tables were judged from a clipped excerpt;
  * pipeline._execute wraps EVERY action in a flat asyncio.wait_for of
    AGENTIC_V2_ACTION_TIMEOUT (180s). Sequential full-text verification of up
    to 6 passages legitimately exceeds that -> "VERIFY failed: timeout after
    180.0s" with all completed verdicts discarded.

Fix:
  * src/agentic_v2/verify.py: _passage_text now returns the FULL unit text
    (strip only — newlines kept so tables keep their rows); truncation removed.
    Verifier calls use max_attempts=2 so one stalled passage fails fast instead
    of eating the whole action budget. One paragraph per sequential LLM call
    unchanged; per-passage trace events unchanged.
  * src/agentic_v2/actions.py: new ActionExecutor.n_verify_candidates() exposes
    how many passages VERIFY would judge.
  * src/agentic_v2/pipeline.py: _action_timeout(decision, state) scales the
    VERIFY budget to base_timeout x candidate_count (each passage gets its own
    base-timeout slot); every other action keeps the flat budget.

Tests: test_agentic_v2_verify.py updated (full-text assertions) + timeout-
scaling tests (+3 net). agentic_v2 suite: 70 passing.

----------------------------------------------------------------------------
AGENTIC V2 — ALWAYS SYNTHESIZE AFTER STOP; ONLY RELEVANT/PARTIAL EVIDENCE
----------------------------------------------------------------------------
When the orchestrator decided STOP, _stop() just set terminal=True — only the
BUDGET exit path ever ran SYNTHESIZE, so a normal stop ended the run with
answer=None. Also, VERIFY stored EVERY assessment as evidence, including
not_relevant (BACKGROUND) and failed (UNKNOWN) passages, which then leaked
into the synthesis prompt as "verified evidence".

Fix:
  * src/agentic_v2/synthesize.py: new usable_evidence(state) = evidence whose
    quality is DIRECT (relevant) or INDIRECT (partially_relevant);
    _synthesis_prompt now feeds ONLY those lines ("VERIFIED EVIDENCE (relevant
    / partially relevant only)").
  * src/agentic_v2/actions.py _verify: rejected/unverifiable assessments no
    longer become evidence items at all (per-chunk UI verdict events are still
    emitted for every passage). Side benefit: an all-rejected verify now
    correctly counts as no-progress.
  * src/agentic_v2/pipeline.py: _finalize_on_budget -> _finalize_answer; called
    after EVERY loop exit. If the loop already synthesized, it is a no-op;
    otherwise it runs SYNTHESIZE from usable evidence with a contextual reason
    ("orchestrator stopped; ..." vs "budget exhausted; ..."). No usable
    evidence: budget stops hard-stop honestly; terminal STOPs keep the
    orchestrator's own reason (e.g. "cannot answer with available tools").

Tests: +5 (test_agentic_v2_synthesize.py; stop-then-synthesize pipeline test;
rejected-passages-are-not-evidence action test). agentic_v2 suite: 75 passing.

----------------------------------------------------------------------------
UI (ui/agentic_v2.html) — TRAJECTORY TRACE, EVERYTHING WIRED TO LIVE EVENTS
----------------------------------------------------------------------------
Rebuilt from the node-graph into a DeepSeek-Harness/DevTools-Network style
dark trace: multi-lane horizontal timeline (Input / Model / Tools / Subtool)
on top, chronological expandable event log below, run bar + live stats at the
bottom. Nothing is static — every control reads the live event stream:

  * Timeline: one lane per category; each event is a colored tick positioned
    at its ts; failed events are red; clicking a tick jumps to + expands the
    matching log row.
  * Log rows (click to expand): per event with type tag, relative time,
    inline preview and a status chip. LLM calls open tabs [Prompt (full raw
    text) / Thinking / Response]; verifier calls additionally show the exact
    "Passage text" extracted from the prompt and a ✓ relevant / ✗ not_relevant
    + confidence verdict chip; GLOBAL_RETRIEVE rows show the fetched docs'
    full chunk text.
  * Verifier live spinner: llm_call_start events render a "VERIFYING · passage
    N · <doc>" row with an animated spinner; the paired llm_call event turns it
    green (relevant / partially_relevant + confidence at the end) or red
    (not_relevant).
  * Controls wired to real endpoints: Run (POST /run), Clear (POST /clear),
    Follow (auto-scroll), and a working text search over event titles/previews/
    contents. Dead elements removed (Chat tab, filter chips, attach button,
    mode dropdown, invented stats) — stats now show turns / steps / LLM time /
    tool time / elapsed computed from event timestamps.

Supporting backend changes:
  * src/llm/run.py: ask_structured now notifies the observer at call START
    (_notify_observer_start) in addition to the existing end event, so the UI
    gets a real timing span + live spinner per LLM call.
  * src/agentic_v2/llm_observer.py: emits llm_call_start / llm_call; agent is
    aliased from the label PREFIX (objective_verifier:1:P -> verifier) while
    the full label is preserved for the UI.
  * scripts/v2_ui.py: /events supports ?after=<idx> incremental polling.

Verified: node --check on the inline JS; pure-logic checks for the verifier
keep/reject + passage extraction + document rows; server smoke test with a
synthetic event stream (35 events) incl. llm_call_start pairs. Full suite:
203 passing (+2 observer tests: llm_call_start ordering, prefix aliasing).

Timeline follow-ups:
  * Retrieved documents are nested inside the retrieve step: document events no
    longer render as standalone log rows; they group by (iteration, action) and
    appear as a collapsible "Documents (N)" list (doc id / chunk / section +
    full chunk text) inside the retriever step row.
  * Timeline shows real DURATION bars instead of instant blips: LLM calls span
    llm_call_start -> llm_call on the Model lane (verifier bars turn green
    relevant / red not_relevant), steps span their first event -> action_done /
    failure on the Tools lane, and running work extends to "now" with a pulse.
    Instantaneous events (decisions, fetches, verdicts, failures, state) remain
    thin ticks.
  * Clicking any bar/tick jumps to and expands the matching log row; document
    ticks resolve to their parent retrieve step row.

----------------------------------------------------------------------------
RATE LIMIT RETRY — FIX INFINITE HANG ON OVERSIZED CALLS
----------------------------------------------------------------------------
A verifier call stalled after hitting a TPM rate limit. Root cause in the
shared TokenBucket (src/llm/ratelimit.py): when a single request's token
estimate exceeded the whole per-minute window, acquire() never admitted it —
_window_left computed to 0.0 (tokens_used was 0), so it looped forever
sleeping ~0.3s, no forward progress, run wedged. (Full untruncated verifier
passages can easily estimate >15k tokens, exactly crossing the Gemini TPM.)

Fixes:
  * TokenBucket.acquire: a request larger than the whole TPM window is now
    admitted immediately (logged) — the bucket is an aggregate backstop; the
    provider's own quotas + the 429-aware retry loop handle it. All waits are
    capped (<=120s + jitter) so nothing can spin unboundedly.
  * retry_after_seconds: the HTTP Retry-After header is now capped at 120s
    like the inline "retry in Ns" form — a huge header can no longer penalize
    the shared bucket for minutes.
  * ask_structured (src/llm/run.py): on a rate-limit failure after retries it
    FAILS FAST instead of falling through to the text path, which previously
    ran a second full retry round against the same 429 (doubling the wait).

Tests: new tests/test_ratelimit.py (11 tests) incl. regression
test_oversized_request_does_not_hang with a 1s timeout guard. Full suite:
214 passing.

----------------------------------------------------------------------------
NEVER SEND THE FULL DOCUMENT TO THE VERIFIER
----------------------------------------------------------------------------
Observed: objective H2's verify call sent the whole paper. Root cause chain:
  * In the EXCAVATE phase READ_DOCUMENT is a legal action (policy.py) and the
    orchestrator may pick it with only a document_id (no chunk_id).
  * actions._read_document's no-chunk branch called _document_text() which
    concatenated EVERY chunk of the paper into one unit ->
    RetrievedDocument(unit_kind="document", section="(full document)",
    text=<entire paper>) pushed into state.documents.
  * _candidates_for fed ALL state.documents rows (incl. that unit) to the
    verifier; _passage_text sends it verbatim (no cap since truncation was
    removed).

Fix (src/agentic_v2/actions.py):
  * READ_DOCUMENT by document_id no longer reads the whole paper. New
    _document_units() restores each chunk's containing structural unit via the
    StructuralUnitIndex, ranks units by lexical overlap with the objective's
    intent/statement/evidence/synonyms (+ decision query/instructions), and
    stores the top-N (<=4) as separate RetrievedDocument rows (proper
    unit_kind paragraph/table/figure, per-unit score). Never a "document" unit.
  * _candidates_for backstop: any document row with unit_kind "document" or
    section "(full document)" is skipped, so even a stray full-paper unit can
    never reach verification.
  * Removed the now-dead _document_text() concatenator.

Tests: +2 in test_agentic_v2_actions.py (read-by-id returns ranked units, not
full paper; candidates filter drops stray full-document unit); removed the
obsolete _document_text test. Full suite: 215 passing.

REVISED - NEVER DROP A FULL-DOCUMENT PASSAGE - PASS ITS RELEVANT PART:
The first fix filtered full-document units OUT of verification, which the
user (correctly) rejected as losing PMC11878906's evidence wholesale. New
behavior (src/agentic_v2/actions.py):
  * NEW _verify_candidates(): before VERIFY, any document/candidate unit
    flagged as a whole paper (unit_kind "document" or section "(full document)")
    is EXPANDED into its structural units (paragraph / table / figure) via
    _document_units() (corpus + StructuralUnitIndex, ranked by objective
    relevance), top-N per document. When the corpus/index is unreachable, a
    paragraph-split fallback (_split_paragraph_units) extracts the most
    relevant bounded paragraphs straight from the blob text (sentence-bounded
    ~400-600 words). Merged with the normal candidates; the document itself
    STAYS in state (never dropped).
  * _candidates_for still skips full-document rows for the base list;
    _verify_candidates re-embraces and expands them afterwards.
  * Verifier safety net (src/agentic_v2/verify.py) no longer drops: it CLIPS a
    stray full-document passage to its most relevant paragraph
    (_clip_full_document) so even a leak through the net can never send a
    50k-char blob, and never discards evidence.

Tests: +2 (expand full document -> relevant units, not dropped; empty-text
full doc gracefully skipped). Full suite: 219 passing.

DIAGNOSIS - "UI STILL SHOWS (full document)" WAS A STALE SERVER:
A fresh run at 21:23 still emitted a (full document) unit with 8720 tokens.
Verification showed the events had ZERO llm_call_start events (feature shipped
19:14) - i.e. the process executed code from BEFORE the fix, cached in
sys.modules by a long-lived server started before the code changes. The fix on
disk is correct (proven: READ_DOCUMENT for the same PMC id now stores
paragraph units Results/Methods, never a full doc; full suite 219 passing).
Actions taken:
  * AgenticV2Pipeline.run_start now stamps code_rev="no-full-document@1" so the
    UI/events identify which code produced a run at a glance. A run without the
    stamp (or without llm_call_start events) is stale.
  * scripts/v2_ui.py purges src.agentic_v2 / src.llm from sys.modules on every
    /run so servers started AFTER this change never serve stale code (an
    already-running older server must be restarted once).
  * Reminder for the user: restart any already-running v2_ui / pipeline process
    before re-running; then check the new run_start event for the revision tag.

UI (ui/agentic_v2.html) — OBJECTIVES ARE THE MASTER TRACE ELEMENT:
The log is no longer a flat chronological stream. Each research objective
(H1, H2, ...) is now a master card; everything related to that H — its
GLOBAL_RETRIEVE step (with the nested Documents list), VERIFY step (with
per-passage verdict rows / LLM calls), ENRICH, and any failures — is nested
under it and revealed on click:
  * buildIterObj(): maps iteration->objective from decision/action_start/
    agent_spawn/action_done/document events so llm_call*/verdict/etc. events
    resolve to the right H even though they only carry an iteration.
  * objOfEvent(): event with objective_id uses it; otherwise falls back to the
    iteration map, then "global".
  * renderLog() groups entries into master cards (first-seen order, "global"
    last); each card header shows id, statement (from VERIFY spawn input),
    counts (LLM calls / docs / failures) and a status chip; clicking toggles
    the body. .obj-master[data-open] CSS hides/shows.
  * bindLog() handles master toggles; jumpToLog()/openMasterOf() auto-open the
    containing master when you click a timeline bar, then scroll to the row.
  * Search filters entries before grouping; full-document safety nets unchanged.
  Verified: node --check on inline JS; grouping algorithm (H1/H2/global
  assignment incl. llm_call-by-iteration) asserted in a python replica; full
  suite 219 passing.

DIAGNOSIS+FIX — "QUERY FINDS 2 RELEVANT CHUNKS BUT THE ORCHESTRATION LAYER
LOSES THEM" (what is cardiac arrest)
---------------------------------------------------------------------------
Observed (live run "What is a cardiac arrest?", captured in
agentic_v2_events_fixed.jsonl / prior log): GLOBAL_RETRIEVE -> VERIFY works
perfectly — 8 units retrieved, the intent verifier judges the two
PMC11704390 DEFINITIONS TABLE rows relevant (quality=direct conf=1.00), 2
evidence items added, H1 status supported. Then SYNTHESIZE answers "No
definition of cardiac arrest was found in the provided verified evidence."
The evidence WAS in state; both downstream consumers silently head-sliced it:

  * the supporting unit is a WHOLE TABLE (ehae724-T2); the literal definition
    row ("Cardiac arrest is defined as a verified sudden cessation of cardiac
    activity...") sits ~400+ chars into the unit, after the header + AKI row;
  * synthesize._evidence_line collapsed + sliced each evidence excerpt to
    400 chars -> the prompt ended at "...haemodialysis or perito" — right
    before the "Row: Cardiac arrest" definition;
  * state.summarize truncated evidence to 320 chars for the orchestrator too,
    which is why the orchestrator at the SYNTHESIZE decision said the verifier
    "incorrectly flagged" the tables — it could not SEE the definition (it
    trusted the labels but distrusted the invisible content).

Fix — anchor the evidence excerpt on the objective's own terms at capture
time, and stop silent head-slicing everywhere downstream:
  * src/agentic_v2/actions.py: NEW _evidence_excerpt(text, objective) — finds
    the objective's longest matching phrase in the unit (evidence_required /
    statement / intent / synonyms / entities), centers a ~1400-char window on
    the FIRST occurrence (snapped to a row boundary so tables keep whole
    rows), marks clipped edges with "..."; _verify now stores this instead of
    the raw unit head. No phrase match -> bounded head slice WITH a marker.
  * src/agentic_v2/synthesize.py: _evidence_line cap 400 -> _display_excerpt
    (1600 chars) with an explicit "... [excerpt truncated]" marker; a clipped
    excerpt is never silently cut.
  * src/agentic_v2/state.py: summarize() gets a separate max_evidence cap
    (900) for EVIDENCE lines (documents stay at 320) so the orchestrator can
    see anchored evidence and stops misjudging verified items as false
    positives.

Tests: +5 (anchored excerpt survives verify; _evidence_excerpt deep-phrase +
head-fallback markers; synthesis prompt keeps a >400-char-deep definition;
_display_excerpt truncation marker; summarize shows deep anchored evidence).
Full suite 229 passing (excl. the same 5 broken retrieval_v2 tests).

Verified live end-to-end (agentic_v2_events_fixed.jsonl): the same query now
synthesizes a real answer from the verified definition —
"Cardiac arrest is the sudden cessation of cardiac activity leading to
unresponsiveness, absence of normal breathing, and lack of circulation,
primarily caused by rhythms such as ventricular fibrillation, asystole, or
pulseless electrical activity." (confidence 1.0, grounded in PMC11704390's
definitions table). The synthesizer prompt now literally contains the anchored
row "Cardiac arrest is defined as a verified sudden cessation of cardiac
activity..." — previously the excerpt ended at "...haemodialysis or perito",
right before that row.

COLLECTOR — YEARS + TQDM (per request)
---------------------------------------------------------------------------
src/collector.py (still the exact PMCCollector(Collector) architecture):
  * __init__ gains min_year / max_year (0 = any); when set, collect() folds a
    publication-date interval into the esearch query:
    "... AND (2015/01/01[pdat] : 2024/12/31[pdat])".
  * tqdm download bar unchanged (desc="Downloading", unit="article").
  * CLI now accepts --min-year / --max-year (validated); interactive script
    prompts for "Year range (e.g. 2015-2024, blank = all years)" again.
Verified: mocked run shows the exact [pdat] clause in the esearch argv and the
tqdm bar; full suite 229 passing.

JATS -> MARKDOWN CONVERTER (per request)
---------------------------------------------------------------------------
New scripts/jats_to_md.py (+ Makefile target `make jats-to-md`, INPUT/OUTPUT
vars, default data/raw -> data/md): converts collected JATS XML into Markdown
by REUSING the structure-aware parser (src.parser.PMCASTParser) instead of
raw XML string-munging:
  * YAML front matter (pmcid, title, journal, authors, published, keywords,
    categories, doi, pmid — YAML-safe quoting);
  * sections -> ##/### headings (recursive for subsections);
  * paragraphs/lists/administrative as plain Markdown;
  * tables -> real Markdown tables from the structured AST (label/caption
    italics, header + separator + padded rows, pipe-escaped cells, footnotes,
    image-table fallback);
  * figures -> ![label](image_ref) + **label caption** (blank line added so
    parsers don't merge image+caption into one paragraph);
  * equations -> $$ latex $$ blocks (inline $...$ already preserved by the
    parser);
  * references -> bullet list under ## References with DOI/PMID links.
  * resumable: existing .md files skipped unless --overwrite; --limit N caps
    the run; tqdm progress bar; per-file failures logged, never aborting.
  * input can be a single .xml file or a directory (tree mirrored, .md per
    article, {stem}.md).
Verified: converted real data/chunked files incl. tables/figures/references;
skip (Skipped: 3) + overwrite (Converted: 3) verified; make jats-to-md runs;
full suite 229 passing (excl. the same 5 broken retrieval_v2 tests).

CHUNKER V2 — MARKDOWN-NATIVE, P1+P2 WITHOUT LLM (per request)
---------------------------------------------------------------------------
Keep v1 (src/chunker.py XML path) untouched. New src/md_chunker.py (chunker
v2) consumes the .md files and emits the SAME Chunk schema, so embedding and
retrieval code are unchanged. `make chunk-md` (MD_INPUT/CHUNKS_OUT/UNITS_OUT)
or `python -m src.md_chunker`.

P1 parent/child:
  * units_v2/{stem}.parquet: one UnitRecord per section AND per table with
    the FULL unit text + the ids of its granular child chunks; chunks carry
    unit_id metadata (table rows/footnotes parent_id -> table summary).
  * tables: summary + per-row + footnotes chunks (same 9 chunk types as v1).
P2 (everything that needs no LLM):
  * YAML front matter -> per-chunk metadata; article title/journal/date +
    full header path injected into every embedding_text as prefix;
  * chunk-level entity tags (Chunk.concept_ids) via LexiconTagger: local
    JSON lexicon {CUI: [surface forms]}, word-boundary, no LLM/network;
  * exact-duplicate suppression (per-doc, --global-dedup across files) with
    dedup_of provenance;
  * incremental rebuilds: md_sha256 sidecar in chunks_v2/{stem}.meta.json,
    unchanged files skipped; per-doc coverage report (chunks_by_type,
    eligibility, units, entities, dedup).
Structure-first basics: token-budget prose (~480 tok default, tiktoken or
chars/4), oversized paragraphs split at sentence boundaries, tail overlap
between consecutive prose windows, figures/equations/lists/references/
administrative atomic as before.
Skipped (need LLM or new embedding code): LLM table summaries / question
generation; late-chunking multi-vector.

Parser notes: fences ($$...$$) extracted pre-markdown-it and re-injected by
line so equations can never leak into prose; italic/bold caption and footnote
lines attached deterministically from inline children (em_open/strong_open);
reference DOI/PMID extracted from raw markdown links.
Tests: tests/test_md_chunker.py (16: front matter, sections, types/
eligibility, embedding prefix, table summary/rows/footnotes + table units,
figure captions/image_ref, equation fences, references+DOI, prose overlap,
entity tagging, exact-dedup, determinism, unit coverage, CLI skip + corpus-
compatible columns). Full suite 245 passing (excl. the same 5 broken
retrieval_v2 tests).

CHUNKER V2 — CRITIQUE-REVIEW FIXES (analysis of an external review)
----------------------------------------------------------------------------
Review verdict per claim: 13 valid bugs, 2 fair tradeoffs, 2 false
("file won't run" was against a truncated paste — the on-disk file ran 245
tests; the chunk_type-Literal doubt settled — VALID_CHUNK_TYPES has
administrative/table_footnotes). Fixes applied (+ tests, 16 -> 22):
  * _extract_fences: single-line $$...$$ strips the trailing $$ AND advances
    i (missing i += 1 was a PRE-EXISTING infinite loop, caught by the new
    test); an UNCLOSED $$ fence no longer swallows the document — the opening
    line degrades to ordinary text.
  * _tail_tokens: token-accurate now (words accumulated until ~n tokens) —
    previously n*4 WORDS over-delivered ~4-5x the configured overlap.
  * _prose_chunks: overlap tail computed from the window's OWN text, never
    from the already-overlapped chunk text (was compounding).
  * hard_max_tokens now enforced: absolute per-piece ceiling in
    _split_long_paragraph (pathological giant sentences clamped with "[...]").
  * build_section_tree: heading-less documents now chunk into an anonymous
    section (was silently dropping all content).
  * _drop_doc_title_root: preserves the h1's direct blocks by prepending
    them to the first child (no silent drop of unheaded intros).
  * --global-dedup: racy shared-cache check-and-set replaced by a
    DETERMINISTIC sorted post-pass over the written parquets (first file in
    sorted order wins; dedup_of recorded; no cross-thread races).
  * _reference_chunk: receives unit_id (refs no longer carry "").
  * administrative sections: figure/equation blocks no longer dropped.
  * figure captions: extracted by image-regex SPANS (text after the LAST
    image) — robust to ']' inside captions and multi-image paragraphs.
  * _report: removed dead table_rows_ok; _units_to_df: real lists for
    breadcrumb + chunk_ids (no JSON strings, no no-op line).
  New tests: single-line eq strip, unclosed-fence safety, heading-less doc,
  h1 direct blocks, reference unit_id, deterministic global dedup (P1 wins).
  Full suite 251 passing (excl. the same 5 broken retrieval_v2 tests).


"PERFECT CONVERTER" — LOSS-AWARE, STRUCTURE-FIRST JATS->MD (big refactor)
----------------------------------------------------------------------------
scripts/jats_to_md.py rebuilt from scratch: RE-PARSES JATS XML via lxml
instead of the flattened PMCASTParser AST, so XML structure survives (the
old heuristic citation repair — and its "P = .[04]" corruption — is deleted
entirely; citations are structural now).

What it does (per the 9-point review):
  1. Structure preservation: sup/sub/italic/bold, xrefs (bibr -> [n]
     resolving to the NUMBERED reference index; fig/table/supp -> labels),
     inline formulas $latex$, links, footnotes — no flattening.
  2. Tables: parsed as grids (rowspan/colspan expanded exactly once), thead
     via .iter("tr"), rectangular pipe output + caption + footnotes; math
     inside cells kept as $latex$.
  3. Figures: label + caption + alt-text + long-desc + href; nested
     figure-supplements (elife <fig><p><fig>) rendered; figures inside
     <p>/<list-item>/<fig-group>/<abstract>/<back>/<sub-article>/<app> all
     recovered.
  4. Equations: verbatim tex-math (no whitespace flattening), $$ blocks,
     stray $$ inside bodies sanitized so fences stay balanced.
  5. Lists: ordered/bullet/alpha markers + nesting (incl. list-in-p).
  6. Title: only emitted when present; empty title -> no bare "#".
  7. LOSS DETECTION: source element counts (tables/figs/refs/equations/supp)
     vs rendered counters; per-file [warn] + run summary — nothing is
     silently dropped: residual misses (0.23% of 14,893 files) are flagged.
  8. Atomic writes (.tmp -> fsync -> rename) + md_sha256 sidecar resumability.
  9. Stateless per-file parse (no shared parser instance).

References are NUMBERED ([1] Author. Title. J Source YYYY;Vol:pages. IDs)
and in-text <xref ref-type="bibr"> resolves to [1,3]-style linked numbers;
year-only junk refs are filtered so numbering never misaligns. Abstract:
structured abstracts render ## Abstract + ### subsections; video/graphical
abstracts and sub-article front figures render too. Supplementary material
gets its own "## Supplementary Material" section with download links.

Corpus validation: 14,893 data/chunked files scanned — files with any
reported drop fell from 246 (1.65%) to 35 (0.23%) after the container
fixes; every remaining gap is WARNED, none silent. Tests rewritten:
tests/test_jats_to_md.py (15) + md chunker regression (22). Full suite
266 passing (excl. the same 5 broken retrieval_v2 tests).

EMBEDDING-FREE EVAL GATE (Tier-1) — scripts/benchmark_chunking.py + report
----------------------------------------------------------------------------
Tier-1 of a two-tier gate. Zero encoders: BM25 (src.retrieval.sparse, k1=1.5
b=0.75) over eligible chunk text; self-supervised queries (chunk-as-gold:
verbatim 40-word prefix = upper bound + mid-paragraph needle = fragmentation
probe); hit = gold TEXT contained in a retrieved chunk, so merge/split/dedup
never bias the comparison. Fidelity per variant: chunk/type mix, dedup
suppression, citation harvest+resolution (100%: every bracketed citation
resolves to a ref id), embedding-budget violations, truncation markers,
unit orphans, empty-chunk guard, determinism. Benchmarks prose_strategy /
max_tokens / global-dedup A/B against a baseline; writes an in-depth markdown
report (eval/reports/chunking_benchmark.md) with failure examples + deltas.
Sample run (150 docs, 282 queries): paragraph-320 100% recall@10 (baseline);
window-320 100% but -9% eligible chunks & +27% avg tokens; paragraph-160
70.6% (-29.4pt — fragmentation cost, doc-level unchanged); global-dedup
+188 suppressed without recall loss (0.0pt) but -0.3pt doc-level. Caveat:
lexical only — Tier-2 (dense/hybrid + end-to-end) still required before
locking the config.

TABLE-FAITHFULNESS FIXES + SINGLE-FILE TEST MODE (per request)
----------------------------------------------------------------------------
Converter (scripts/jats_to_md.py):
  * cells with stacked runs ("16631<break/>17818<break/>13793") are now
    space-joined -> "16631 17818 13793" (was fused "166311781813793");
  * full-width colspan rows render as single-cell context rows (group
    subheaders) instead of data rows padded with empty values;
  * thead-less JATS tables promote the first row with >=2 distinct cells to
    the md header (kills "Column i" fallback + header-row-as-data leak).
Chunker (src/md_chunker.py _table_chunks row loop):
  * group subheaders (one non-empty cell: "Sex", "Median (IQR)", panel
    labels "a. Stepwise ...") become group_path context, not data rows;
  * repeated header rows inside multi-panel tables are consumed as context;
  * first-cell-empty column-context rows ("n1 = 1971 | n2 = 1900") are
    consumed instead of surfacing as "Row: —" junk chunks.
New CLI test mode: python -m src.md_chunker --file <pmc-id|xml|md>  --input
no longer required. Resolves data/chunked/<id>.xml (converts first if xml),
chunks with the current config, prints every chunk with tables row-by-row
(label/group/text), writes parquet to .single_test/chunks/.
Verified on PMC11727332.1 + PMC10327125.4: empty-label rows 0, "Column i"
rows 0, numbers spaced, group_path now carries Sex/Median/panel context.
Full suite 267 passing.

MEMORY + CONTEXT LAYER (src/memory) — persistent, temporal, provenance-aware research state
-------------------------------------------------------------------------------------------
New package src/memory/ implements the MedPat Memory + Context layer design
(L0 conversation / L1 working research state / L2 persistent research memory /
L3 evidence corpus referenced, never copied). Governing invariant enforced
mechanically: MEMORY IS NOT MEDICAL EVIDENCE.

  src/memory/enums.py         provenance classes, claim/relation/contradiction
                              statuses, memory event vocabulary
  src/memory/models.py        typed records (claim, evidence-ref, session,
                              contradiction, gap, preference, audit events, ...)
                              with valid_from/valid_to (bi-temporal, append-only)
  src/memory/embed.py         HashEmbedder (offline deterministic) + optional
                              MedCPTEmbedder (768d) + NullEmbedder
  src/memory/validation.py    provenance gate (EVIDENCE_DERIVED_CLAIM requires
                              >=1 verified evidence ref else DEGRADED to
                              MODEL_INFERENCE), classifier, injection guard,
                              dedup + near-duplicate detection
  src/memory/store.py         MemoryStore contract + InMemory + Postgres
                              (schema medrag_memory, CREATE TABLE idempotently,
                              pgvector optional, FKs + CHECK constraints on
                              provenance/roles, memory_events audit table)
  src/memory/retrieval.py     hybrid recall: lexical + entity + semantic +
                              session + graph proximity, provenance-aware
                              penalties, per-session diversity, staleness
  src/memory/context.py       typed context blocks, token budgets, boundary
                              markers (VERIFIED EVIDENCE vs PERSISTENT RESEARCH
                              MEMORY), contradictions always rendered both
                              sides, hard whole-block shedding on overflow
  src/memory/pipeline.py      candidate -> classify -> validate -> dedup ->
                              commit (atomic, audited); run-result extraction
                              (only CRITIC-verified evidence -> claims)
  src/memory/consolidate.py   background: derive statuses from links, merge
                              near-dups (SUPERSEDED kept), detect
                              contradictions, temporal refresh, staleness
  src/memory/api.py           MemoryAPI facade (prepare_run/record_run,
                              compose_context, session lifecycle, lineage,
                              preference/user-text handling, consolidate)
  src/memory/hooks.py         AgenticV3Pipeline integration adapter
  scripts/memory_init.py      create the medrag_memory schema in PostgreSQL

Integration (opt-in, evidence boundary intact): AgenticV3Pipeline accepts
memory=MemoryAPI(...). Before the run the planner receives a BOUNDED memory
context region labeled ADVISORY ONLY (never citable); after the run the
verified evidence, claims, contradictions, gaps and the labeled conclusion
are persisted. Run with:  python -m src --memory "<query>"  (or MEMORY_ENABLED=1).
MasterOrchestratorAgent.plan gained an optional memory_context kwarg (default
'' — behavior unchanged when absent). result["memory"] reports the wiring.

Key behaviors verified by tests:
  * gate: unsupported statements can never become evidence-backed claims --
    EVIDENCE_DERIVED without verified refs is DEGRADED, user assertions never
    upgraded, prompt-injection text refused for persistence;
  * provenance: every evidence-derived claim renders with its lineage
    claim -> links -> evidence -> PMCID; dedup UNIONs evidence links, never
    discards them;
  * temporal: supersession closes valid_to, history kept (as_of), claims never
    overwritten;
  * contradictions: preserved with BOTH sides + dimension scaffolding;
  * context: typed blocks, budget enforced (whole blocks shed, never silent
    mid-item cut), memory never injected into the evidence channel;
  * cross-session continuity: a follow-up question resumes the same research
    session and the planner receives prior research state.

Tests: tests/test_memory_{validation,store,retrieval,context,pipeline,
integration}.py (52 tests). Full suite: 216 passing, 2 pre-existing failures
(test_md_chunker embedding prefix + test_retrieval_service missing-index,
both fail identically on the pristine tree), 1 skipped.
PostgreSQL smoke-validated live against the dev database (schema created,
record_run -> retrieval -> lineage -> resume -> consolidate, rows cleaned up).

MEMORY + CONTEXT LAYER — FRONTEND LINKAGE (MedPat web ↔ backend/api.py)
-----------------------------------------------------------------------
The memory layer is now linked end-to-end with the frontend:

Backend (backend/api.py):
  * one per-process MemoryAPI singleton (get_memory_api(), lazy, auto
    backend: Postgres medrag_memory when reachable, in-memory fallback;
    MEMORY_ENABLED=0/off disables);
  * /v1/chat/stream attaches memory to the pipeline and forwards the
    frontend's conversation_id (AgenticV3Pipeline.answer gained an optional
    conversation_id kwarg, threaded into prepare_run/record_run);
  * streams FIRST-CLASS memory events (not wrapped as pipeline trace):
      {"type":"memory","kind":"prepare","session_id","session_title",
       "prior_claims","prior_contradictions","prior_gaps"}
      {"type":"memory","kind":"commit","session_id","stats":{...}}
    emitted by the pipeline via new V3Events.memory_prepare/memory_commit;
  * store.ensure_conversation() (in-memory + Postgres) so frontend
    conversation ids are preserved when turns are recorded.

Frontend (frontend/):
  * lib/types.ts: MemoryInfo (sessionId/title, prior counts, committed
    stats) on Message + the memory StreamEvent;
  * lib/rag-client.ts: normalizeEvent handles the backend's snake_case
    memory frame -> camelCase event;
  * lib/mock-rag.ts: mock engine emits realistic memory prepare/commit
    events so the UI shows the strip in mock mode too;
  * components/chat/ChatView.tsx: onEvent stores memory prepare/commit
    into the assistant message;
  * components/chat/AssistantMessage.tsx: compact RESEARCH-MEMORY strip
    under each response (MEM led · sess id · prior claims/contradictions/
    gaps · recorded stats), tooltip-labeled "advisory context only, never
    evidence".

Verified: tests/test_api_memory.py (4 tests, TestClient stream asserts the
memory prepare/commit contract + no trace duplication + conversation_id
threading + MEMORY_ENABLED gating); frontend tsc --noEmit clean; live boot
of api:app streamed {"type":"memory","kind":"prepare",...} against the
dev Postgres before any LLM work (residue cleaned afterwards).
Full backend suite: 220 passing (same 2 pre-existing failures), 1 skipped.

MEMORY UTILIZATION BY THE AGENTS — FOLLOW-UP CONTINUITY FIX
-----------------------------------------------------------
Reported: asking "how does hypertension affect life expectancy" then
"how does hypertension lead to diseases" re-processed the FIRST query —
memory was persisted but not utilized by the planner. Root causes (seen in
prepare_run output for the follow-up):
  1) record_run never marked questions answered -> prior questions stayed
     "[open]" and the planner saw them as outstanding obligations to redo;
  2) the master prompt only said "shape which questions to investigate"
     while showing only the prior open questions -> small models mirrored
     Q1's tasks;
  3) the run's CONCLUSION claim (what was established) was never retrieved
     into context, so there was no "build on this" anchor.

Fixes:
  * api.record_run: on a terminal answer, mark every session question
    answered (store.mark_question_answered) and store the answer summary as
    the session's rolling summary (new store.update_session_summary in both
    backends) — used by session identification + context rendering;
  * context (PersistentMemoryContext.render): "ALREADY INVESTIGATED — do NOT
    re-derive:" header; prior conclusions rendered separately and
    prominently ("prior conclusions (labeled inference, not evidence)");
    ResearchContext.render shows "[answered]" vs "[open]" per question plus
    "last conclusion: ...";
  * agents/master._plan_prompt: mandatory PLANNING RULES — plan ONLY for the
    NEW question; do NOT recreate [answered]/ALREADY-INVESTIGATED tasks; use
    prior findings as background only; follow-ups build ON prior findings;
  * memory retrieval: lower min_score floor (0.12 -> 0.08) and semantic
    floor (0.30 -> 0.15), full session-continuity baseline within the
    resumed session, conclusion-claim boost, and session-question vocabulary
    expansion for short follow-up queries (len(tokens) < 4).

Tests: tests/test_memory_continuity.py (6 tests) — prior questions answered
in follow-up context, conclusion surfaced, short-followup recall via session
questions, run2 answers recorded, end-to-end planner receives the answered
framing, planning rules present only when memory attached. PostgreSQL
smoke-validated (4 questions all answered, summary set, follow-up resumes
same session, [answered] framing in context; rows cleaned).
Full backend suite: 226 passing (same 2 pre-existing failures), 1 skipped.

----------------------------------------------------------------------------
SINGULAR DEEPAGENTS MERGE (2026-09-03)
----------------------------------------------------------------------------

Retired the v3 pipeline (src/agentic/, pydantic-ai src/agents/*); the
deepagents flow is now the only runtime, renamed src/x_deepagents ->
src/agents. Fixes landed with the merge: per-request stream sinks
(contextvars, no cross-talk), worker web evidence counts toward coverage
(stable web: document ids), resolution findings fold into the owning
requirement, memory layer wired into the flow (prepare/record + planner
context), budgets sourced from AppConfig, paper_inspect phantom removed,
gap web-augment capped per requirement, verifier confidence surfaced.
Shared tools kept working via src/tools/models.py (pure-data contracts
moved out of the retired v3 state). API is a single path (no engine
switch); frontend engine reduced to "xdeep". Verified: full pytest suite
green (2 pre-existing unrelated failures), tsc + vite build clean, live
end-to-end run (qwen think + mistral verify + pg corpus + firecrawl) reached
synthesis with 12 verified items. See MERGE_PLAN.md.

CHUNKER V2 — SOTA GAP CLOSURE, NO LLM (per request)
----------------------------------------------------------------------------
md_chunker.py now closes the SOTA gap map entirely WITHOUT any LLM:
  * Contextual embeddings ON by default: embedding_text = Document: {title} +
    Section: {breadcrumb} + object label + evidence. embedding_context mode
    (document|section|object) exposed; object = legacy label-only for eval A/B.
  * Healthy split overlap: --split-overlap-sentences default 2, carry
    token-budgeted to ~25% of max_tokens (--split-overlap-tokens, capped at
    half) so the artificial mid-paragraph boundary never orphans context.
  * Abbreviation-aware sentence splitter: e.g./Fig./No./et al./cap initials/
    decimals never cut mid-value (deterministic, no LLM).
  * Late-chunking hooks: sentence-split paragraphs emit a PARAGRAPH unit
    (full source text + child piece ids) and every piece carries
    metadata.sentence_split (piece_index/piece_count/paragraph_unit_id) so an
    encoder can embed the full paragraph and slice spans.
  * Report now records paragraphs_sentence_split, split_pieces, paragraph_
    units and the full chunker_config (incl. embedding_prefix_mode +
    late_chunking_hooks).
Removed: the 'skipped, need an LLM' framing — LLM table summaries / question
generation are explicitly out of scope; there is no LLM path.
Tests: test_md_chunker.py updated to the pinned parent_id contract (table
chunks -> TABLE unit) + new regressions (contextual modes, healthy carry,
abbreviation splitter, paragraph units). 27 passing in test_md_chunker.py;
chunking suites 61 passed / 1 skipped (real-file test needs corpus file).


MEDPAT POSTGRES STORE (docker) - schema-approved + built
----------------------------------------------------------------------------
New container medpat-postgres (backend/docker/medpat/docker-compose.yml, port
5433; the medrag container is untouched). Schema medpat (init/01_schema.sql):
documents (body_md full source), units (section/table/paragraph, hierarchical
via parent_unit_id - reviewer fix: chunker now emits the tree), chunks
(column-identical to chunks_v2; parent_id = enclosing unit; fingerprint col;
row_index functional index; CASCADE parent FKs; tsv via trigger, 'simple'),
chunk_embeddings (vector(768) + HNSW vector_ip_ops + embedding_text_hash),
references + chunk_citations (UNIQUE (document_id, position)), lexicon_terms,
load_marks, meta, v_corpus_stats. units.chunk_ids dropped (chunks.parent_id is
the single source of truth).
Pipeline: python -m src.chunking.pg_store --input data/md (upsert + md_sha256
skip, parents-first unit order, SQL global dedup via --global-dedup/--dedup-only)
and python -m src.embedding (MedCPT -> vector(768)). Retrieval: SCHEMA env
MEDPAT_PG_SCHEMA defaults to medpat (pgvector_store/tools/retriever/dense_pgvector).
Verified live end-to-end against real docs, then RESET to all-empty: the
container is intentionally INFRASTRUCTURE ONLY (schema + ParadeDB BM25 index,
zero corpus rows). Ingestion is user-run: make medpat-ingest / medpat-embed.
make medpat-up/ingest/embed/dedup/psql/down.

