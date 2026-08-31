.PHONY: download-articles jats-to-md chunk-md pg-init pg-reset pg-stats pg-hybrid help

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

.DEFAULT_GOAL := help
