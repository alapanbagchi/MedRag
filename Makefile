.PHONY: download-articles jats-to-md chunk-md help pg-up pg-load pg-reset pg-stats pg-smoke pg-hybrid pg-bm25

# Download all PMC Open Access articles for a term (interactive CLI).
# Prompts for the title and an optional year range; a BLANK year range
# searches all years indexed by NCBI.
download-articles:
	.venv/bin/python scripts/download_articles.py

# Convert collected JATS XML to Markdown (default data/raw -> data/md).
# Override with INPUT=... OUTPUT=... LIMIT=... (e.g.:
#   make jats-to-md INPUT=data/raw/cardiology LIMIT=100)
INPUT ?= data/raw
OUTPUT ?= data/md
LIMIT ?=
jats-to-md:
	.venv/bin/python scripts/jats_to_md.py --input "$(INPUT)" --output "$(OUTPUT)" $(if $(LIMIT),--limit $(LIMIT))

# Chunker v2: structure-first Markdown chunking (no LLM) -> chunks_v2 + units_v2.
# MD_INPUT defaults to the Markdown output of jats-to-md (data/md).
MD_INPUT ?= data/md
CHUNKS_OUT ?= chunks_v2
UNITS_OUT ?= units_v2
chunk-md:
	.venv/bin/python -m src.md_chunker --input "$(MD_INPUT)" --chunks-out "$(CHUNKS_OUT)" --units-out "$(UNITS_OUT)"

# ====================================================================
# PostgreSQL + pgvector targets
# ====================================================================

# Input folders for pg-load (override freely, e.g.
#   make pg-load EMBEDDINGS_DIR=/data/embs CHUNKS_DIR=/data/chunks).
# ?= keeps the project defaults unless an env var or command-line value
# overrides them.
EMBEDDINGS_DIR ?= embeddings_v2
CHUNKS_DIR ?= chunks_v2

# Start the pgvector container if it is not already running (idempotent;
# never destroys data). Uses scripts/pg_up.sh.
pg-up:
	./scripts/pg_up.sh

# Full load of v2 embeddings + chunk metadata into pgvector.
#   make pg-load               incremental: skips files already in the DB
#                              (per-file checkpoints in medrag.load_marks);
#                              killing it mid-run is safe, just re-run.
#   make pg-load RESET=1       drop ALL data in the medrag schema, then load
#   make pg-load FORCE_REINDEX=1   rebuild the HNSW index even if unchanged
#   make pg-load VECTORS_ONLY=1  (or CHUNKS_ONLY=1) for partial loads
# PG connection comes from .env (PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE).
RESET ?= 0
VECTORS_ONLY ?= 0
CHUNKS_ONLY ?= 0
FORCE_REINDEX ?= 0
pg-load:
	.venv/bin/python scripts/pg_load_v2.py \
		--embeddings-dir "$(EMBEDDINGS_DIR)" \
		--chunks-dir "$(CHUNKS_DIR)" \
		$(if $(filter 1 yes true on,$(RESET)),--reset) \
		$(if $(filter 1 yes true on,$(VECTORS_ONLY)),--vectors-only) \
		$(if $(filter 1 yes true on,$(CHUNKS_ONLY)),--chunks-only) \
		$(if $(filter 1 yes true on,$(FORCE_REINDEX)),--force-reindex)

# Drop ALL data in the medrag schema (chunks + embeddings + indexes).
# Put data back with: make pg-load
pg-reset:
	.venv/bin/python scripts/pg_load_v2.py --reset-only

# Print current row counts / index state.
pg-stats:
	.venv/bin/python scripts/pg_load_v2.py --stats-only

# Retrieval smoke test (works right after pg-load):
#   make pg-smoke                     random stored-vector self-hit probes
#   make pg-smoke QUERY="...text..."  embed a real query and search
QUERY ?=
pg-smoke:
	.venv/bin/python scripts/pg_smoke.py $(if $(QUERY),--query "$(QUERY)")

# Hybrid query: sparse (Postgres FTS by default) + dense pgvector, intent
# rerank + paper diversification, full logs in logs/.
#   make pg-hybrid QUERY="..."
#   make pg-hybrid QUERY="..." CROSS_ENCODER=1      # MedCPT cross-encoder rerank
#   make pg-hybrid QUERY="..." MAX_PER_PAPER=0      # turn paper dedup OFF
QUERY ?=
BM25_DIR ?=
CROSS_ENCODER ?= 0
MAX_PER_PAPER ?= 2
pg-hybrid:
	.venv/bin/python scripts/pg_hybrid_query.py "$(QUERY)" \
		$(if $(BM25_DIR),--bm25-dir "$(BM25_DIR)") \
		$(if $(filter 1 yes true on,$(CROSS_ENCODER)),--cross-encoder) \
		$(if $(MAX_PER_PAPER),--max-per-paper "$(MAX_PER_PAPER)")

# Rebuild the BM25 sparse index over the v2 chunks (chunks_v2 -> index/bm25_v2).
# Needed once so hybrid search's BM25 hits resolve to pgvector chunk ids
# (the checked-in index/bm25 was built over the old v1 corpus).
BM25_OUT ?= index/bm25_v2
BM25_WORKERS ?=
pg-bm25:
	.venv/bin/python scripts/build_bm25_v2.py --chunks-dir "$(CHUNKS_DIR)" --out-dir "$(BM25_OUT)" $(if $(BM25_WORKERS),--workers $(BM25_WORKERS))

help:
	@echo "Targets:"
	@echo "  download-articles   Download articles for a term (interactive, year range)"
	@echo "  jats-to-md          Convert JATS XML to Markdown (INPUT=data/raw OUTPUT=data/md LIMIT=0)"
	@echo "  chunk-md            Chunker v2: Markdown -> chunks_v2 + units_v2 (MD_INPUT=data/md)"
	@echo "  pg-up               Start the pgvector Docker container (idempotent, data-safe)"
	@echo "  pg-load             Load embeddings_v2 + chunks_v2 into pgvector (RESET=1 to wipe first)"
	@echo "  pg-reset            Drop ALL pgvector data (medrag schema)"
	@echo "  pg-stats            Show chunk/embedding counts and index state"
	@echo "  pg-smoke            Retrieval smoke test (QUERY='text' to search real queries)"
	@echo "  pg-hybrid           Hybrid search: BM25 + pgvector + RRF (QUERY='text')"
	@echo "  pg-bm25             Rebuild BM25 over chunks_v2 (index/bm25_v2) for hybrid search"
	@echo "  help                Show this help"

.DEFAULT_GOAL := help