.PHONY: help pgvector stop-pgvector init-db load-db chunk embed \
        build-index build-index-pgvector search eval benchmark \
        expand run test lint clean v2-query v2-benchmark v2-test

PYTHON  := .venv/bin/python3
PIP     := .venv/bin/pip3

# ── Default variables ──────────────────────────────────────────────
CHUNKS_DIR    ?= chunks
EMBED_DIR     ?= embeddings
INDEX_DIR     ?= index
DATA_DIR      ?= data/raw/cardiology
PGHOST        ?= localhost
PGPORT        ?= 5432
PGUSER        ?= postgres
PGPASSWORD    ?= medrag
PGDATABASE    ?= medrag

# ── Help ────────────────────────────────────────────────────────────

help: ## Show this help
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Infrastructure ──────────────────────────────────────────────────

pgvector: ## Start PostgreSQL + pgvector in Docker
	bash scripts/setup_pgvector.sh

stop-pgvector: ## Stop the pgvector container
	docker rm -f medrag-pgvector 2>/dev/null || true

init-db: ## Initialize pgvector database schema
	PGHOST=$(PGHOST) PGPORT=$(PGPORT) PGUSER=$(PGUSER) \
	PGPASSWORD=$(PGPASSWORD) PGDATABASE=$(PGDATABASE) \
	$(PYTHON) scripts/init_pgvector.py

load-db: ## Bulk-load embeddings + corpus into pgvector (parallel)
	PGHOST=$(PGHOST) PGPORT=$(PGPORT) PGUSER=$(PGUSER) \
	PGPASSWORD=$(PGPASSWORD) PGDATABASE=$(PGDATABASE) \
	$(PYTHON) scripts/load_embeddings.py \
		--embeddings-dir $(EMBED_DIR) \
		--corpus $(INDEX_DIR)/corpus.parquet \
		--workers 8 --batch-size 50000

load-vectors: ## Load only embedding vectors into pgvector
	PGHOST=$(PGHOST) PGPORT=$(PGPORT) PGUSER=$(PGUSER) \
	PGPASSWORD=$(PGPASSWORD) PGDATABASE=$(PGDATABASE) \
	$(PYTHON) scripts/load_embeddings.py --vectors-only

load-corpus: ## Load only corpus metadata into pgvector
	PGHOST=$(PGHOST) PGPORT=$(PGPORT) PGUSER=$(PGUSER) \
	PGPASSWORD=$(PGPASSWORD) PGDATABASE=$(PGDATABASE) \
	$(PYTHON) scripts/load_embeddings.py --corpus-only

# ── Pipeline ────────────────────────────────────────────────────────

chunk: ## Chunk XML files into retrieval chunks
	$(PYTHON) -m medrag.pipeline \
		--dir $(DATA_DIR) \
		--chunks $(CHUNKS_DIR) \
		--json chunking_report.json

embed: ## Generate MedCPT embeddings for chunks
	$(PYTHON) -m medrag.embedding \
		--input-dir $(CHUNKS_DIR) \
		--output-dir $(EMBED_DIR)

# ── Indexing ────────────────────────────────────────────────────────

build-index: ## Build corpus + dense (FAISS) + sparse (BM25) indexes
	$(PYTHON) -m medrag.retrieval build-index \
		--chunks-dir $(CHUNKS_DIR) \
		--embeddings-dir $(EMBED_DIR) \
		--index-dir $(INDEX_DIR)

build-index-pgvector: ## Build corpus + dense (pgvector) + sparse (BM25) indexes
	PGHOST=$(PGHOST) PGPORT=$(PGPORT) PGUSER=$(PGUSER) \
	PGPASSWORD=$(PGPASSWORD) PGDATABASE=$(PGDATABASE) \
	$(PYTHON) -m medrag.retrieval build-index \
		--chunks-dir $(CHUNKS_DIR) \
		--embeddings-dir $(EMBED_DIR) \
		--index-dir $(INDEX_DIR) \
		--pgvector

# ── Search / Retrieval ──────────────────────────────────────────────

search: ## Run a single search query  (QUERY="your question")
	$(PYTHON) -m medrag.retrieval search \
		--query "$(QUERY)" \
		--method hybrid \
		--top-k 10 \
		--candidate-k 50 \
		--rerank \
		--show-text

query: ## Full pipeline: LLM expand → retrieve → rerank → save  (QUERY="your question")
	KAGGLE_URL="$(KAGGLE_URL)" \
	$(PYTHON) scripts/query.py "$(QUERY)" \
		$(if $(OUTPUT),--output $(OUTPUT))

expand: ## Expand query via Kaggle /expand endpoint  (QUERY="your question")
	KAGGLE_URL="$(KAGGLE_URL)" \
	$(PYTHON) -c "\
import os, sys, json, requests; \
sys.path.insert(0, 'src'); \
url = os.environ.get('KAGGLE_URL','').rstrip('/') + '/expand'; \
r = requests.post(url, json={'query': '$(QUERY)'}, timeout=120); \
print(json.dumps(r.json(), indent=2))"

run: ## Run orchestrator with a plan file  (PLAN=path/to/plan.json)
	$(PYTHON) -m medrag.retrieval.orchestrator \
		--plan $(PLAN) \
		--output retrieval_runs/$(shell date +%Y%m%d_%H%M%S)_run

# ── Evaluation / Benchmarks ─────────────────────────────────────────

eval: ## Run retrieval evaluation
	$(PYTHON) -m medrag.retrieval evaluate \
		--eval-file eval/queries.json \
		--methods bm25,hybrid,bm25+rerank \
		--report-json eval/report.json

benchmark: ## Run the full retrieval benchmark (50-doc sample)
	$(PYTHON) scripts/benchmark_retrieval.py \
		--methods bm25+rerank \
		--out-dir eval

benchmark-qa: ## Run QA benchmark (requires --queries-file)
	$(PYTHON) scripts/benchmark_retrieval.py \
		--queries-file $(or $(QUERIES_FILE),eval/qa_benchmark_queries.json) \
		--methods bm25+rerank \
		--out-dir eval

# ── Developer ───────────────────────────────────────────────────────

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

v2-test: ## Run retrieval V2 tests
	$(PYTHON) -m pytest tests/test_v2_*.py -v

v2-query: ## Run retrieval V2 end-to-end  (QUERY="your question")
	$(PYTHON) -m medrag.retrieval_v2 "$(QUERY)" \
		$(if $(OUTPUT),--output $(OUTPUT))

v2-benchmark: ## Run the retrieval V2 regression benchmark (known multi-hop question)
	$(PYTHON) scripts/benchmark_v2.py \
		$(if $(OUT),--out $(OUT))

lint: ## Run linter
	$(PYTHON) -m ruff check src/ scripts/

install: ## Install package in editable mode
	$(PIP) install -e ".[dev]"

# ── Cleanup ─────────────────────────────────────────────────────────

clean: ## Remove generated indexes, caches, and reports
	rm -rf index/ __pycache__ .pytest_cache
	rm -f chunking_report.json eval/report.json eval/benchmark_report.json eval/benchmark_report.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Full pipeline (one-shot) ───────────────────────────────────────

pipeline: pgvector chunk embed build-index-pgvector init-db load-db ## Run full pipeline: start DB → chunk → embed → index → load
	@echo ""
	@echo "Pipeline complete. Run 'make search QUERY=\"...\"' to test."
