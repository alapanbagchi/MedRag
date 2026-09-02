# src.x_deepagents - autonomous retrieval system (deepagents + LangGraph)

Candidate **main** retrieval system, living inside backend/src so it shares the
backend venv, src.tools, src.config, src.prompts and data directly.

Reuses ONLY prompts + non-LLM tools (src/tools, src/umls, src/prompts), via
the single boundary in x_deepagents/reuse.py. All agents, state, graph,
rules and budgets are new (deepagents + LangGraph).

## Status (steps 1-4 of the plan)

- [x] Package inside backend/src + venv deps (deepagents, langgraph)
- [x] Reuse boundary: src.tools / src.config importable directly
- [x] deepagents hello agent constructs
- [x] Hello-world LangGraph compiles (one deep-agent node)
- [x] State models (src/x_deepagents/state.py) - evidence/requirement/run
      state machines with explicit transitions
- [x] Workflow rules (src/x_deepagents/rules.py) - the mandatory gates:
      chunk->verifier, contradiction+resolution on the final set,
      evidence-gated synthesis + citation repair, hard budgets, phase spine
- [x] Tool adapters (src/x_deepagents/tools/) - langchain @tools wrapping
      the reused HybridRetrieverTool / UMLSEnricher, plus the postgres_search
      future stub; wired into the registry the deep agent selects from. The
      research agent autonomously drives the retrieve tool loop (live-verified).
- [x] Real research graph (src/x_deepagents/graph.py): decompose ->
      retrieve+verify (mandatory gate) -> contradiction -> resolution ->
      evidence-gated synthesis. Stage agents in agents/stages.py. Live: the
      graph retrieves real candidates, verifies each, detects+resolves
      contradictions and synthesizes from verified evidence only.
- [x] Full architecture: orchestrator (master.txt, gemma) -> parallel
      research workers (gemma) running the adaptive v3 loop: UML enrich ->
      SEARCH PLANNER (search_planner.txt) -> retrieve/verify -> REPLANNER
      (replanner.txt) rounds -> DEEP INSPECTOR (deep_inspector.txt) ->
      verifier critic (critic.txt, mistral paced) -> CONTRADICTION
      (contradiction.txt) + RESOLUTION (resolution.txt, search tools) ->
      evidence-gated SYNTHESIS (synthesize.txt, gemma). Live-verified.
- [x] Logging: run progress + tool calls (web searches, retrievals,
      verifier verdicts) append to backend/logs.txt ([xdeep] events).

## Architecture & model routing

```text
query -> ORCHESTRATOR (think: qwen/alibaba) -> tasks
       -> RESEARCH WORKERS per task (parallel, think model)
            - UML enrichment (umls_lookup)
            - LOCAL retrieval first: postgres (medrag.chunks) primary, then
              hybrid (parquet); every candidate MUST pass the VERIFIER
            - only if still unsatisfied: informed WEB gap-fill (searxng,
              trust-gated) + RELIABILITY critic (the "another agent" for
              websites)
       -> CONFLICT agent (mistral) checks verified set
       -> RESOLUTION (mistral + real search tools + verifier re-check)
       -> GAP RESOLUTION: close HARD gaps (unsatisfied requirements) and
            LATENT gaps (content holes in satisfied requirements found by
            the gap probe: mechanisms, components, comparators, durations,
            subgroups) - MORE local retrievals (pg primary -> hybrid), then
            trust-gated web, then LIST what still cannot be closed
            (contradictions re-checked on the new evidence)
       -> SYNTHESIS (think model) - verified evidence only + citation repair
            (unresolved gaps / limitations rendered as '##' headings)
```

- THINK = the "good agent" for thinking + long-running tasks (decompose,
  research workers, synthesis, conflict). Provider auto-selects:
    * QWEN (Alibaba MaaS) when QWEN_API_KEY / QWEN_BASE_URL are set, or the
      shared GENERAL_LLM_BASE_URL points at Alibaba (aliyuncs/dashscope) -
      the DEFAULT of this deployment. Model from QWEN_MODEL /
      GENERAL_LLM_MODEL (default qwen3.8-flash).
    * GROQ when GROQ_API_KEY is set (alternative; model default
      gemma2-9b-it via GROQ_MODEL).
    * else the shared GENERAL_LLM_* endpoint with a Gemma fallback (never the
      shared AppConfig gemini-2.0-flash default).
  Force with XDEEP_THINK_PROVIDER=qwen|alibaba|aliyun|groq|google.
- MISTRAL (mistral-medium-latest) = short decisions, paced 1 req / 1.5s:
  verification, contradiction checks, resolution, reliability judgement.
- Web search: searxng URL via XDEEP_SEARXNG_URL / SEARXNG_URL (default
  http://127.0.0.1:8888). If unreachable the agent falls back to retriever.

## Evidence policy (pg-first, verified-only, informed web-after-local)

1. The PRIMARY evidence source is the LOCAL Postgres corpus (medrag.chunks)
   via postgres_search (full-text + pgvector) - the research agent uses it
   first and heavily. postgres_search reports "available"/"empty_corpus" so
   the agent can tell a corpus gap from a query miss.
2. Web search (searxng_search) is a HELP and - with XDEEP_WEB_AUGMENT (default
   on) - ALWAYS fires after local verification: per-gap for unsatisfied
   requirements, per probed content hole, and per satisfied requirement to
   bring current guidelines / extra info. Queries are informed by the local
   phase (rejection notes, missing evidence, probe sub-questions); the ORDER
   (local first, web after) is enforced by the worker, not left to the LLM.
3. DEEP WEB EVIDENCE: every web result that passes the site-legitimacy gate
   is FETCHED (tools/fetch.py) and the ACTUAL page text (up to 8k chars) is
   what the verifier and reliability critic judge - not the 500-char search
   snippet. Anti-bot challenge pages (reCAPTCHA/JS gates) fall back to the
   snippet cleanly; verified pages keep their fetched body as the evidence.
4. Every web result passes the site-legitimacy gate (tools/site_reputation.py):
   - TRUSTED = medical journals + official medical sites (NEJM, Lancet, BMJ,
     JAMA, PubMed/PMC, WHO, CDC, NIH, Mayo, Medscape, UpToDate, ...)
   - CONSUMER = WebMD / Drugs.com / RxList / Everyday Health etc. - returned
     but explicitly NOT evidence-grade (reliability critic must judge them)
   - BLOCKED = social media + forums (Reddit, X/Twitter, Facebook, YouTube,
     Quora, Medium, ...) - NEVER returned, even when opted in
   - UNKNOWN = dropped by default; only surfaced labeled "unverified" when
     the agent explicitly opts in (trusted_only=False)
5. The verifier critic judges RELEVANCE; the separate RELIABILITY critic
   judges AUTHORITY/recency/conflicts of the FETCHED web page (not just the
   snippet). Both must pass before a web page counts as evidence (reliability
   lands on EvidenceItem.reliability and feeds the evidence hierarchy).
6. Contradiction RESOLUTION runs real searches (local first, then web) and
   re-verifies every new passage through the verifier before it may appear in
   a resolution reason.

SearXNG web-search troubleshooting
-----------------------------------
* If the log shows \"HTTP 403 / json_format_disabled\": the instance serves
  HTML but format=json is not whitelisted in searxng/settings.yml. Fix:

    sudo chown -R $(id -u):$(id -g) searxng        # take ownership once
    cp searxng-settings-fixed.yml searxng/settings.yml
    docker compose -f searxng.docker-compose.yml up -d --force-recreate

  backend/searxng-settings-fixed.yml ships the corrected settings.

* Log hygiene: pytest NEVER writes to backend/logs.txt - tests redirect the
  xdeep log to a per-test temp file (conftest._isolate_xdeep_log). If you
  still see "base=http://127.0.0.1:1" in logs.txt, it is stale bytes from
  before the isolation fixture, or a real run inheriting XDEEP_LOG_FILE /
  SEARXNG_URL from the shell. Sanity-check with:
      env | grep -E "XDEEP|SEARXNG"   # should show nothing in a clean shell

SearXNG config (env, all optional - defaults shown, see backend/.env.example):

- SEARXNG_URL / XDEEP_SEARXNG_URL = http://127.0.0.1:8888
- SEARXNG_TIMEOUT / XDEEP_SEARXNG_TIMEOUT = 15.0 (seconds)
- SEARXNG_TOP_K / XDEEP_SEARXNG_TOP_K = 8 (default results per search)

## Gap-resolution budget (env, optional)

The post-contradiction gap pass closes unsatisfied requirements with more
local retrievals, then web, then lists the rest:

- XDEEP_TIMEOUT_GAP = 240 (s) - whole pass timeout
- RunBudget.max_gap_local_rounds default 3, max_gap_web_searches default 3
- XDEEP_GAP_PROBE = 0/off disables latent (content-hole) probing; default on
- XDEEP_GAP_PROBE_MAX = max probe sub-questions per run (default 3)
- XDEEP_WEB_AUGMENT = 0/off reverts web to fallback-only; default ON so a
  trust-gated web round fires for every gap AND for satisfied requirements
  (brings current guidelines / extra verified info after local retrieval)

Honest "resolved" accounting
---------------------------
A probed gap is only stamped RESOLVED when a COMPLETENESS GATE confirms the
sub-question is fully answered by the gathered evidence. If important facets
remain (safety, hard outcomes, duration, comparisons, mechanisms, subgroups),
the gap is PARTIALLY_RESOLVED and each remaining facet is listed in
run.gaps. After synthesis, the synthesizer's OWN unresolved_gaps are
reconciled into run.gaps (gaps_reconciled event) and matching
gap_resolutions downgraded - so the UI trace never claims "0 listed" while
the answer actually lists open facets.

## Inline resolved contradictions + highlightable web sources

- RESOLVED contradictions are never shown under a separate header: the
  synthesizer weaves the evidence-supported resolution INLINE into the section
  body that discusses the claim (with citations), so the reader meets the
  answer in the relevant paragraph. UNRESOLVED contradictions still get an
  explicit "Unresolved Contradictions" block.
- WEB sources come with a highlightable open-website button: the source URL
  carries a browser-native Text Fragment (#:~:text=<quoted passage>) and the
  frontend shows "Open website — highlight passage", which opens the actual
  site scrolled to and highlighting the exact cited sentence. PMC sources keep
  the in-app article view + "Open article".

## Synthesis fallback (no dead answers)

If the LLM synthesis step is unavailable (provider timeout / 429 / malformed
reply), the pipeline does NOT return "(no answer - synthesis failed)". A
DETERMINISTIC fallback answer is built from the verified evidence (summary +
per-item extraction with 〔cite:doc〕 markers, all resolvable to source
numbers). The trace emits a "synthesis_fallback" warning so the UI shows the
answer came from fallback mode. The completeness gate likewise never
fabricates missing facets when it fails - real facets are folded in after
synthesis via reconciliation.

## Rich UI trace

The web bridge streams STRUCTURED events (not just progress lines) so the
frontend think-log renders real titles/details for:

- decompose_done (tasks planned)
- search_round / query_start / retrieved (corpus searches + candidate counts)
- web_search_started / web_search_done (broad web search, surfaced URLs,
  dropped-blocked/unverified counts)
- web_fetch (which site was opened, chars fetched, or snippet fallback)
- verify verdicts (accepted / rejected / contradictory + critic note)
- reliability_verdict (high / medium / low + authority note per site)
- gap_probe / gap_resolution (content-hole questions and their outcome)
- contradiction / resolution
- synthesis_start / synthesis_done

Every event also mirrors to logs.txt as "[xdeep] trace:<event>" for parity.

## Run

```bash
cd backend
uv sync                 # ensures deepagents + langgraph are in .venv
uv run pytest tests/test_xdeep_* -q
uv run python -m src.x_deepagents "<question>" --run
```

Provider credentials come from backend/.env (LLM_PROVIDER / GENERAL_LLM_* /
UMLS_API_KEY ...) - one source of truth.

## Web UI (MedPat)

The MedPat frontend talks to this system through the same `/v1/chat/stream`
endpoint it always used - the xdeep engine speaks the identical wire contract
(status / sources / token / done / pipeline trace). To drive the UI with the
xdeep graph:

1. Start the backend API:  `uv run uvicorn api:app --port 8000` (backend/)
2. Point the UI at it: `NEXT_PUBLIC_RAG_API_URL=http://127.0.0.1:8000`
   (in frontend/.env.local; unset NEXT_PUBLIC_USE_MOCK)
3. Choose the engine:
   - **env switch** (no UI change): set `XDEEP_ENGINE=1` before starting the
     API - every request runs xdeep;
   - **per-request** (frontend `engine: \"xdeep\"` param, already wired in
     lib/rag-client.ts) - add an engine selector in the UI whenever you want.

Every run still logs to backend/logs.txt ([xdeep] events) and streams live
status into the UI's research-pipeline panel.
