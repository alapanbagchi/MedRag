.PHONY: download-articles jats-to-md chunks pg-init pg-reset pg-stats pg-hybrid retrieval retrieval-tests judge firecrawl-status firecrawl-test scrape xdeep xdeep-json xdeep-auto xdeep-auto-json xdeep-log xdeep-tests plan subagent api api-dev medpat-up medpat-down medpat-psql medpat-reset medpat-migrate medpat-embed medpat-embed-bulk medpat-index medpat-config medpat-fullvec medpat-ui help

# Download all PMC Open Access articles for a term (interactive CLI).
download-articles:
	.venv/bin/python -m src.ingestion.download_articles

# Convert collected JATS XML to Markdown (default data/raw -> data/md).
INPUT ?= data/raw
OUTPUT ?= data/md
LIMIT ?=
jats-to-md:
	.venv/bin/python -m src.processing.jats_to_md --input "$(INPUT)" --output "$(OUTPUT)" $(if $(LIMIT),--limit $(LIMIT))

# Chunker: chunk every Markdown file in a directory -> chunks + units in the
# medpat ParadeDB container. Point at another dir with DIR="<path>".
DIR ?= data/md
chunks: medpat-up
	.venv/bin/python -m src.chunking --input "$(DIR)" --workers "$(or $(MEDPAT_WORKERS),1)"

# ====================================================================
# PostgreSQL + pgvector targets
# ====================================================================

EMBEDDINGS_DIR ?= embeddings_v2
CHUNKS_DIR ?= chunks_v2

pg-init:
	.venv/bin/python scripts/pg_init.py 		--embeddings-dir "$(EMBEDDINGS_DIR)" 		--chunks-dir "$(CHUNKS_DIR)" 		$(if $(filter 1 yes true on,$(RESET)),--reset) 		$(if $(filter 1 yes true on,$(VECTORS_ONLY)),--vectors-only) 		$(if $(filter 1 yes true on,$(CHUNKS_ONLY)),--chunks-only) 		$(if $(filter 1 yes true on,$(FORCE_REINDEX)),--force-reindex)

pg-reset:
	.venv/bin/python scripts/pg_init.py --reset-only

pg-stats:
	.venv/bin/python scripts/pg_init.py --stats-only

QUERY ?=
BM25_DIR ?=
CROSS_ENCODER ?= 0
MAX_PER_PAPER ?= 2
pg-hybrid:
	.venv/bin/python scripts/pg_hybrid_query.py "$(QUERY)" 		$(if $(BM25_DIR),--bm25-dir "$(BM25_DIR)") 		$(if $(filter 1 yes true on,$(CROSS_ENCODER)),--cross-encoder) 		$(if $(MAX_PER_PAPER),--max-per-paper "$(MAX_PER_PAPER)")

# Time the hybrid retrieval tool end to end (BM25 + MedCPT dense -> RRF ->
# MedCPT cross-encoder -> boundary expansion). Prints per-stage timings plus a
# per-run total so a change can be measured before/after; run 1 is cold.
#   make retrieval query="hypertension thresholds" R=5 SHOW=3
#   make retrieval query="..." JSON=1 EXPECT="chunkA,chunkB"
R ?= 3
PER_LEG ?= 60
TOP_K ?= 10
SHOW ?= 5
EXPECT ?=
retrieval:
	.venv/bin/python scripts/retrieval_check.py "$(query)" --runs "$(R)" --per-leg "$(PER_LEG)" --top-k "$(TOP_K)" --show "$(SHOW)" $(if $(EXPECT),--expect "$(EXPECT)") $(if $(filter 1 yes true on,$(JSON)),--json) $(ARGS)

# Offline retriever tests: RRF fusion, passage/table assembly, the local_search
# tool contract, and the retrieval service. No DB or models needed.
# test_missing_index_fails_clearly is deselected: it asserts the legacy
# local-index failure path, but the service now falls back to pgvector/BM25,
# so it depends on what the surrounding environment can reach.
retrieval-tests:
	.venv/bin/python -m pytest tests/test_retrieval_tool.py tests/test_retrieval_service.py -q \
		--deselect tests/test_retrieval_service.py::test_missing_index_fails_clearly

# A/B the evidence judge's reasoning effort on ONE identical batch of the
# top-k retrieved passages: latency plus the hidden-reasoning vs JSON token
# split, and how many passages the keep gate still accepts (quality check).
#   make judge query="hypertension thresholds" EFFORTS=default,medium,low
EFFORTS ?= default,medium,low
JUDGE_RUNS ?= 2
JUDGE_BATCH ?= 10
judge:
	.venv/bin/python scripts/judge_check.py "$(query)" --efforts "$(EFFORTS)" --runs "$(JUDGE_RUNS)" --batch "$(JUDGE_BATCH)" --top-k "$(TOP_K)" --per-leg "$(PER_LEG)" $(ARGS)

BM25_OUT ?= index/bm25_v2
BM25_WORKERS ?=
pg-bm25:
	.venv/bin/python -m src.retrieval.sparse --chunks-dir "$(CHUNKS_DIR)" --out-dir "$(BM25_OUT)" $(if $(BM25_WORKERS),--workers $(BM25_WORKERS))

# ====================================================================
# Firecrawl cloud API (web search + page scrape for xdeep)
# ====================================================================
# Endpoint + key come from backend/.env (same file the API server uses).

FC_ENV_URL := $(shell grep -E "^FIRECRAWL_URL=" .env 2>/dev/null | cut -d= -f2-)
FC_ENV_KEY := $(shell grep -E "^FIRECRAWL_API_KEY=" .env 2>/dev/null | cut -d= -f2-)

firecrawl-status:
	@curl -s -o /dev/null -w "POST v2/search -> %{http_code}\n" -X POST "$(or $(FC_ENV_URL),https://api.firecrawl.dev)/v2/search" -H "Content-Type: application/json" -H "Authorization: Bearer $(FC_ENV_KEY)" -d '{"query":"test","limit":1}' || echo "POST v2/search -> unreachable"

# Standalone scrape check through the real pipeline (waitFor, wall
# detect + retry). Usage: make scrape URL=https://example.com/page
# (one URL) or make scrape URLS="https://a https://b" (batch path,
# the one the agent uses).
URL ?= https://pmc.ncbi.nlm.nih.gov/articles/PMC11975635/
URLS ?=
scrape:
	.venv/bin/python scripts/scrape_check.py $(if $(URLS),$(URLS),"$(URL)")

# Full Firecrawl tool check through the REAL agent path: v2 search ->
# MyBib credibility gate -> v2 batch scrape. Exit 0 only when usable
# pages flow end to end.
# Usage: make firecrawl-test FC_QUERY="latest hypertension guidelines"
#   FC_LIMIT=5 (listings), FULL=1 (print full page texts).
FC_QUERY ?= latest hypertension guidelines
FC_LIMIT ?= 5
firecrawl-test:
	.venv/bin/python scripts/firecrawl_check.py "$(FC_QUERY)" --limit "$(FC_LIMIT)" $(if $(filter 1 yes true on,$(FULL)),--full)

# ====================================================================
# xdeep (deepagents + LangGraph research pipeline)
# ====================================================================

query ?= Does vitamin D supplementation lower blood pressure?

# Run the research pipeline; full event log lands in backend/logs.txt

# (progress, web firecrawl calls, corpus retrievals, verifier verdicts, ...).
# Autonomous runtime: the model decides the workflow per query (planner DAG
# + tool-choosing workers + per-tool guardrails). Each run appends its full
# event log to backend/logs.txt (progress, searches, verdicts, gaps, ...).
xdeep xdeep-auto:
	.venv/bin/python -m src.agents "$(query)" --run

xdeep-json xdeep-auto-json:
	.venv/bin/python -m src.agents "$(query)" --run --json

xdeep-log:
	@tail -60 logs.txt

xdeep-tests:
	.venv/bin/python -m pytest tests/test_agents_smoke.py tests/test_agents_state_rules.py tests/test_agents_graph.py tests/test_agents_tools.py tests/test_agents_routing.py tests/test_agents_reputation.py -q

# Manual planner-agent test/tweak: query='...' MODEL=... TEMP=...
# SEED=... TOP_P=... PROMPT_FILE=... (extra args via ARGS="--json", e.g.).
PE_MODEL ?=
PE_TEMP ?= 0.0
PE_SEED ?= 43
PE_TOP_P ?= 0.1
PE_PROMPT_FILE ?=
plan:
	.venv/bin/python scripts/test_planner.py "$(if $(QUERY_FILE),$(shell cat $(QUERY_FILE)),$(query))" --temperature "$(PE_TEMP)" --seed "$(PE_SEED)" --top-p "$(PE_TOP_P)" $(if $(PE_MODEL),--model "$(PE_MODEL)") $(if $(PE_PROMPT_FILE),--prompt-file "$(PE_PROMPT_FILE)") $(ARGS)

# Manually spawn ONE subagent leg and watch it think. Long text is safest
# from files (multi-line pastes break shell quoting): QUERY_FILE=... and
# REQ_FILE=... read via cat; inline query='...' / REQ="..." still work for
# short values. (query='...' DEPTH=deep|shallow TASK_ID=...
# REQ="E1: a; E2: b" CONTEXT="..." SA_MODEL=... SA_TEMP=... SA_SEED=...
# SA_TOP_P=... SA_MAX_TOOLS=... extra args via ARGS="--json", e.g.).
SA_DEPTH ?= deep
TASK_ID ?= T1
SA_REQ ?=
SA_CONTEXT ?=
SA_MODEL ?=
SA_TEMP ?= 0.0
SA_SEED ?= 43
SA_TOP_P ?= 0.1
SA_MAX_TOOLS ?=
SA_TIMEOUT ?= 600
QUERY_FILE ?=
REQ_FILE ?=
SA_QUERY = $(if $(QUERY_FILE),$(shell cat $(QUERY_FILE)),$(query))
SA_REQS = $(if $(REQ_FILE),$(shell cat $(REQ_FILE)),$(SA_REQ))
subagent:
	.venv/bin/python scripts/test_subagent.py "$(SA_QUERY)" --depth "$(SA_DEPTH)" --task-id "$(TASK_ID)" $(if $(SA_REQS),--reqs "$(SA_REQS)") $(if $(SA_CONTEXT),--context "$(SA_CONTEXT)") $(if $(SA_MODEL),--model "$(SA_MODEL)") --temperature "$(SA_TEMP)" --seed "$(SA_SEED)" --top-p "$(SA_TOP_P)" $(if $(SA_MAX_TOOLS),--max-tools "$(SA_MAX_TOOLS)") --timeout "$(SA_TIMEOUT)" $(ARGS)

# API server (FastAPI + uvicorn). api-dev enables autoreload on code changes.
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

api:
	.venv/bin/python -m uvicorn api:app --host "$(API_HOST)" --port "$(API_PORT)"

api-dev:
	.venv/bin/python -m uvicorn api:app --host "$(API_HOST)" --port "$(API_PORT)" --reload

help:
	@echo "Targets:"
	@echo "  download-articles  Download PMC articles (interactive)"
	@echo "  jats-to-md         Convert JATS XML -> Markdown (INPUT=... OUTPUT=... LIMIT=0)"
	@echo "  chunks DIR=...     Markdown dir -> chunks + units in the medpat ParadeDB container (MEDPAT_WORKERS=N to parallelize)"
	@echo "  pg-init            Load embeddings + chunks into pgvector (RESET=1 to wipe first)"
	@echo "  pg-reset           Drop all pgvector data"
	@echo "  pg-stats           Show chunk/embedding counts and index state"
	@echo "  pg-hybrid          BM25 + pgvector + RRF search (QUERY='...') [stale: script missing]"
	@echo "  retrieval          Time the hybrid retrieval tool live (query='...' R=... PER_LEG=... TOP_K=... SHOW=... EXPECT=... JSON=1)"
	@echo "  retrieval-tests    Offline retriever tests (RRF, assembly, tool contract; no DB/models)"
	@echo "  judge              A/B the judge's reasoning effort (query='...' EFFORTS=default,medium,low JUDGE_RUNS=2 JUDGE_BATCH=10)"
	@echo "  pg-bm25            Rebuild BM25 over chunks_v2"
	@echo "  firecrawl-status   Show Firecrawl endpoint status"
	@echo "  firecrawl-test     Test the Firecrawl tool end to end (FC_QUERY=... FC_LIMIT=... FULL=1)"
	@echo "  scrape URL=..      Scrape URL(s) via the real pipeline (URLS=\"a b\" for batch)"
	@echo "  xdeep              Run xdeep research pipeline (query='...')"
	@echo "  xdeep-json         Same, with full run-state JSON"
	@echo "  xdeep-auto         Run autonomous runtime (query='...', logs append to logs.txt)"
	@echo "  xdeep-auto-json    Same, with full run-state JSON"
	@echo "  xdeep-log          Tail backend/logs.txt (progress + web searches)"
	@echo "  xdeep-tests        Run the offline xdeep test suite"
	@echo "  plan               Manually test/tweak the planner agent (query='...' PE_MODEL=... PE_TEMP=... PE_SEED=... PE_TOP_P=... PE_PROMPT_FILE=... ARGS=...)"
	@echo "  subagent           Spawn one subagent leg live (query='...' DEPTH=... TASK_ID=... REQ=... CONTEXT=... SA_MODEL=... SA_TEMP=... SA_SEED=... SA_TOP_P=... SA_MAX_TOOLS=... SA_TIMEOUT=... ARGS=...)"
	@echo "  api                Run the API server (no autoreload)"
	@echo "  api-dev            Run the API server with autoreload (API_HOST=... API_PORT=...)"
	@echo ""
	@echo "medpat (ParadeDB corpus):"
	@echo "  medpat-up          Start the medpat postgres + pgAdmin frontend (http://localhost:5050)"
	@echo "  medpat-down        Stop medpat containers"
	@echo "  medpat-psql        psql into medpat"
	@echo "  medpat-reset       Wipe all corpus data, schema kept (prompts; YES=1 to skip)"
	@echo "  medpat-migrate     Rebuild schema from init/*.sql (DESTRUCTIVE; KEEP_DATA=1 to preserve rows, YES=1 to skip prompt)"
	@echo "  medpat-embed       Embed eligible chunks via the OpenAI-compatible EMBEDDING_BASE_URL server (safe incremental; MEDPAT_EMBED=\"--drop-index ...\" for extra flags)"
	@echo "  medpat-embed-bulk  Same, tuned for million-row bulk loads: --drop-index + bulk commits (EMBED_BATCH=.. EMBED_WORKERS=.. EMBED_COMMIT_EVERY=.. EMBED_LIMIT=.. MEDPAT_EMBED=..)"
	@echo "  medpat-index       Build missing retrieval indexes (IDX_TYPE=hnsw|ivfflat, LISTS=.., CHECK=1, FORCE=1, VECTOR_ONLY=1, BM25_ONLY=1, MWM=.., IDX_WORKERS=..)"
	@echo "  medpat-config      Show host + container resources and running Postgres settings"
	@echo "  medpat-fullvec     Widen chunk_embeddings halfvec(768) -> vector(768) (fp32) + rebuild HNSW (RESET_EMBEDDINGS=1 to re-embed)"
	@echo "  medpat-ui          Bring up / print the pgAdmin frontend URL"

.DEFAULT_GOAL := help

# ====================================================================
# medpat Postgres (docker) - the corpus storage container
# ====================================================================
MEDPAT_COMPOSE ?= docker/medpat
MEDPAT_DSN ?= postgresql://medpat:medpat@localhost:5433/medpat
MEDPAT_EMBED ?=
# Container /dev/shm size. pgvector's PARALLEL HNSW build allocates a POSIX
# shared-memory segment sized ~maintenance_work_mem, so this must be >= MWM
# (see `make medpat-index`). Exported so the index script can preflight it.
export MEDPAT_SHM ?= 4gb

medpat-up:
	docker compose -f $(MEDPAT_COMPOSE)/docker-compose.yml up -d --wait

medpat-down:
	docker compose -f $(MEDPAT_COMPOSE)/docker-compose.yml down

medpat-psql:
	docker exec -it medpat-postgres psql -U medpat -d medpat

# Wipe all corpus data, keep the schema (tables stay, empty).
# Asks first unless YES=1 is given.
medpat-reset: medpat-up
	.venv/bin/python scripts/medpat_reset.py --dsn "$(MEDPAT_DSN)" $(if $(filter 1 yes true on,$(YES)),--yes)

# Rebuild the schema from docker/medpat/init/*.sql (DESTRUCTIVE - wipes data
# unless KEEP_DATA=1 parks rows and copies surviving tables back).
# Asks first unless YES=1 is given.
medpat-migrate: medpat-up
	.venv/bin/python scripts/medpat_migrate.py --dsn "$(MEDPAT_DSN)" $(if $(filter 1 yes true on,$(YES)),--yes) $(if $(filter 1 yes true on,$(KEEP_DATA)),--keep-data)

# pgAdmin frontend for the medpat ParadeDB container (browse chunks/units).
medpat-ui:
	@docker compose -f $(MEDPAT_COMPOSE)/docker-compose.yml up -d pgadmin
	@echo "pgAdmin: http://localhost:5050"
	@echo "  email:    $(or $(PGADMIN_EMAIL),admin@medpat.io)"
	@echo "  password: $(or $(PGADMIN_PASSWORD),medpat)"

# Encode eligible chunks via the OpenAI-compatible embedding server.
# EMBEDDING_BASE_URL / MEDPAT_EMBED_MODEL are read from backend/.env (the
# script loads it automatically) or from the environment. Override per run:
#   make medpat-embed MEDPAT_EMBED="--base-url https://host/v1 --model <id>"
medpat-embed:
	.venv/bin/python -m src.embedding $(MEDPAT_EMBED) --dsn "$(MEDPAT_DSN)"

# Bulk-load variant: drops the HNSW index for the run and rebuilds it once at
# the end, with multi-row INSERTs + one COMMIT per EMBED_COMMIT_EVERY rows.
# Fastest for million-row loads, but vector search is degraded until the
# rebuild finishes. For small incremental top-ups, plain `medpat-embed` (which
# already uses bulk commits + session tuning) is enough — no index downtime.
# Override per run, e.g.:
#   make medpat-embed-bulk EMBED_WORKERS=8 EMBED_BATCH=128 EMBED_LIMIT=10000
#   make medpat-embed-bulk MEDPAT_EMBED="--base-url https://host/v1 --model <id>"
EMBED_BATCH ?= 64
EMBED_WORKERS ?= 4
EMBED_COMMIT_EVERY ?= 2048
EMBED_LIMIT ?=
medpat-embed-bulk: medpat-up
	.venv/bin/python -m src.embedding --dsn "$(MEDPAT_DSN)" --batch "$(EMBED_BATCH)" --workers "$(EMBED_WORKERS)" --commit-every "$(EMBED_COMMIT_EVERY)" --drop-index $(if $(EMBED_LIMIT),--limit $(EMBED_LIMIT)) $(MEDPAT_EMBED)

# Widen medpat.chunk_embeddings.embedding halfvec(768) -> vector(768) (fp32).
# Existing vectors keep their fp16-rounded values; RESET_EMBEDDINGS=1 truncates
# so `make medpat-embed` regenerates true fp32 vectors.
# Build the retrieval indexes the hybrid retriever needs: the pgvector HNSW
# index over chunk_embeddings (missing after a bulk --drop-index embed run or
# an interrupted build) and the ParadeDB BM25 index over chunks. Idempotent:
# existing indexes are skipped unless FORCE=1.
#   make medpat-index                 # create whatever is missing
#   make medpat-index FORCE=1         # drop + rebuild both
#   make medpat-index VECTOR_ONLY=1   # only the vector index
#   make medpat-index BM25_ONLY=1     # only the ParadeDB BM25 index
#   make medpat-index IDX_TYPE=ivfflat  # fast build when the HNSW index will
#                                       # not fit RAM (no in-memory graph)
# Large corpora: raise/lower the build memory with MWM (default 3GB) and cap
# parallel workers with IDX_WORKERS (default 4; 0 disables). A parallel HNSW
# build puts its graph buffer in /dev/shm, so keep MWM <= MEDPAT_SHM (default
# 4gb) or the build dies with 'could not resize shared memory segment'. Raise
# both only when the host has RAM to spare:
#   MEDPAT_SHM=8gb make medpat-index MWM=6GB      # needs ~6GB free RAM
MWM ?= 3GB
IDX_WORKERS ?= 4
IDX_TYPE ?= hnsw
LISTS ?=
medpat-index: medpat-up
	.venv/bin/python scripts/medpat_indexes.py --dsn "$(MEDPAT_DSN)" \
		--type "$(IDX_TYPE)" $(if $(LISTS),--lists "$(LISTS)") \
		--maintenance-work-mem "$(MWM)" --parallel-workers "$(IDX_WORKERS)" \
		$(if $(filter 1 yes true on,$(FORCE)),--force) \
		$(if $(filter 1 yes true on,$(CHECK)),--check) \
		$(if $(filter 1 yes true on,$(VECTOR_ONLY)),--vector-only) \
		$(if $(filter 1 yes true on,$(BM25_ONLY)),--bm25-only)

medpat-config: medpat-up
	@echo "== host =="
	@nproc | sed 's/^/  cpus: /'
	@free -h | sed -n '1,3p' | sed 's/^/  /'
	@echo "== container =="
	@docker inspect medpat-postgres --format '  memory limit: {{.HostConfig.Memory}} (0 = unlimited)'
	@docker inspect medpat-postgres --format '  shm_size: {{.HostConfig.ShmSize}} bytes'
	@docker exec medpat-postgres df -h /dev/shm | sed 's/^/  /'
	@echo "== postgres settings =="
	@docker exec medpat-postgres psql -U medpat -d medpat -P pager=off -c \
		"SELECT name, setting, unit FROM pg_settings WHERE name IN ('shared_buffers','effective_cache_size','work_mem','maintenance_work_mem','max_wal_size','max_worker_processes','max_parallel_workers','max_parallel_workers_per_gather','max_parallel_maintenance_workers','dynamic_shared_memory_type') ORDER BY name;"

medpat-fullvec: medpat-up
	.venv/bin/python scripts/medpat_fullvec_migrate.py --dsn "$(MEDPAT_DSN)" $(if $(filter 1 yes true on,$(RESET_EMBEDDINGS)),--reset-embeddings)

