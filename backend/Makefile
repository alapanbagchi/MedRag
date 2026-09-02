.PHONY: download-articles jats-to-md chunk-md pg-init pg-reset pg-stats pg-hybrid searxng-up searxng-down searxng-fix-json searxng-logs searxng-status xdeep xdeep-tests help

# Download all PMC Open Access articles for a term (interactive CLI).
download-articles:
	.venv/bin/python -m src.ingestion.download_articles

# Convert collected JATS XML to Markdown (default data/raw -> data/md).
INPUT ?= data/raw
OUTPUT ?= data/md
LIMIT ?=
jats-to-md:
	.venv/bin/python -m src.processing.jats_to_md --input "$(INPUT)" --output "$(OUTPUT)" $(if $(LIMIT),--limit $(LIMIT))

# Chunker v2: structure-first Markdown chunking (no LLM) -> chunks_v2 + units_v2.
MD_INPUT ?= data/md
CHUNKS_OUT ?= chunks_v2
UNITS_OUT ?= units_v2
chunk-md:
	.venv/bin/python -m src.chunking.md_chunker --input "$(MD_INPUT)" --chunks-out "$(CHUNKS_OUT)" --units-out "$(UNITS_OUT)"

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

BM25_OUT ?= index/bm25_v2
BM25_WORKERS ?=
pg-bm25:
	.venv/bin/python -m src.retrieval.sparse --chunks-dir "$(CHUNKS_DIR)" --out-dir "$(BM25_OUT)" $(if $(BM25_WORKERS),--workers $(BM25_WORKERS))

# ====================================================================
# SearXNG (web search for xdeep)
# ====================================================================

SEARXNG_COMPOSE ?= searxng.docker-compose.yml
SEARXNG_PORT ?= 8888

searxng-up:
	@mkdir -p searxng
	@if ! docker compose version >/dev/null 2>&1; then \
		echo "!! 'docker compose' (v2 plugin) not available; trying plain 'docker run'"; \
		docker run -d --name searxng --restart unless-stopped \
			-p $(SEARXNG_PORT):8080 \
			-v "$$(pwd)/searxng:/etc/searxng" \
			-e SEARXNG_BASE_URL="http://localhost:$(SEARXNG_PORT)" \
			-e SEARXNG_SECRET="$$(openssl rand -hex 32)" \
			searxng/searxng; \
	else \
		docker compose -f $(SEARXNG_COMPOSE) up -d; \
	fi
	@echo "SearXNG up on http://127.0.0.1:$(SEARXNG_PORT) (JSON API: /search?q=...&format=json)"
	@echo "Test: curl 'http://127.0.0.1:$(SEARXNG_PORT)/search?q=vitamin+D&format=json' | head -c 200"

searxng-down:
	-@if docker compose version >/dev/null 2>&1; then docker compose -f $(SEARXNG_COMPOSE) down; else docker stop searxng 2>/dev/null; docker rm searxng 2>/dev/null || true; fi
	@echo "SearXNG stopped"

# Modern SearXNG serves JSON only when whitelisted in settings.yml. Apply
# the corrected settings (backend/searxng-settings-fixed.yml) and recreate.
searxng-fix-json:
	@echo "Fixing SearXNG JSON API (403: format=json not whitelisted)..."
	sudo chown -R $$(id -u):$$(id -g) searxng
	cp searxng-settings-fixed.yml searxng/settings.yml
	docker compose -f $(SEARXNG_COMPOSE) up -d --force-recreate
	@echo "Done. Tech: curl 'http://127.0.0.1:$(SEARXNG_PORT)/search?q=vitamin+D&format=json' | head -c 200"

searxng-logs:
	-@if docker compose version >/dev/null 2>&1; then docker compose -f $(SEARXNG_COMPOSE) logs --tail 50; else docker logs --tail 50 searxng; fi

searxng-status:
	@docker ps --filter name=searxng --format "{{.Names}}  {{.Status}}  {{.Ports}}"
	@curl -s -o /dev/null -w "GET /search -> %{http_code}\n" "http://127.0.0.1:$(SEARXNG_PORT)/search?q=test&format=json" || echo "GET /search -> unreachable"

# ====================================================================
# xdeep (deepagents + LangGraph research pipeline)
# ====================================================================

query ?= Does vitamin D supplementation lower blood pressure?

# Run the research pipeline; full event log lands in backend/logs.txt

# (progress, web searxng calls, corpus retrievals, verifier verdicts, ...).
xdeep:
	.venv/bin/python -m src.x_deepagents "$(query)" --run

xdeep-json:
	.venv/bin/python -m src.x_deepagents "$(query)" --run --json

xdeep-log:
	@tail -60 logs.txt

xdeep-tests:
	.venv/bin/python -m pytest tests/test_xdeep_smoke.py tests/test_xdeep_state_rules.py tests/test_xdeep_graph.py tests/test_xdeep_tools.py tests/test_xdeep_routing.py tests/test_xdeep_reputation.py -q

help:
	@echo "Targets:"
	@echo "  download-articles  Download PMC articles (interactive)"
	@echo "  jats-to-md         Convert JATS XML -> Markdown (INPUT=... OUTPUT=... LIMIT=0)"
	@echo "  chunk-md           Markdown -> chunks_v2 + units_v2"
	@echo "  pg-init            Load embeddings + chunks into pgvector (RESET=1 to wipe first)"
	@echo "  pg-reset           Drop all pgvector data"
	@echo "  pg-stats           Show chunk/embedding counts and index state"
	@echo "  pg-hybrid          BM25 + pgvector + RRF search (QUERY='...')"
	@echo "  pg-bm25            Rebuild BM25 over chunks_v2"
	@echo "  searxng-up         Start SearXNG web search (docker compose or docker run)"
	@echo "  searxng-down       Stop SearXNG"
	@echo "  searxng-logs       Tail SearXNG logs"
	@echo "  searxng-status     Show SearXNG container/endpoint status"
	@echo "  xdeep              Run xdeep research pipeline (query='...')"
	@echo "  xdeep-json         Same, with full run-state JSON"
	@echo "  xdeep-log          Tail backend/logs.txt (progress + web searches)"
	@echo "  xdeep-tests        Run the offline xdeep test suite"

.DEFAULT_GOAL := help
