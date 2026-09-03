"""Application configuration."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_env_file(path: Optional[Path] = None) -> None:
    """Load .env file into process environment."""
    dotenv_path = path or (Path(__file__).resolve().parent.parent / ".env")
    if dotenv_path.is_file():
        load_dotenv(dotenv_path, override=True)


class AppConfig:
    """Application configuration from environment variables."""

    def __init__(self) -> None:
        load_env_file()

        # Provider
        self.provider: str = _env("LLM_PROVIDER", "kaggle").strip().lower()
        self.kaggle_base_url: str = _env("KAGGLE_BASE_URL", "http://127.0.0.1:8083/v1").strip().rstrip("/")
        self.kaggle_api_key: str = _env("KAGGLE_API_KEY", "dummy").strip()
        self.kaggle_model: str = _env("KAGGLE_MODEL", "medgemma")
        self.openai_base_url: str = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_api_key: str = _env("OPENAI_API_KEY", "")
        self.openai_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")
        self.ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        self.ollama_model: str = _env("OLLAMA_MODEL", "medgemma")
        self.vllm_base_url: str = _env("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.vllm_model: str = _env("VLLM_BASE_MODEL", "medgemma")
        self.gemini_base_url: str = _env(
            "GENERAL_LLM_BASE_URL", _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        ).strip().rstrip("/")
        self.gemini_api_key: str = _env("GENERAL_LLM_API_KEY", _env("GEMINI_API_KEY", "")).strip()
        self.gemini_model: str = _env("GENERAL_LLM_MODEL", _env("GEMINI_MODEL", "gemini-2.0-flash"))
        # OpenCode (Muse Spark) — OpenAI-compatible endpoint
        self.opencode_base_url: str = _env(
            "OPENCODE_BASE_URL", _env("GENERAL_LLM_BASE_URL", "")
        ).strip().rstrip("/")
        self.opencode_api_key: str = _env("OPENCODE_API_KEY", _env("GENERAL_LLM_API_KEY", "")).strip()
        self.opencode_model: str = _env("OPENCODE_MODEL", _env("GENERAL_LLM_MODEL", "muse-spark-1.3-contributor-free"))
        self.mistral_base_url: str = _env(
            "MISTRAL_BASE_URL", "https://api.mistral.ai/v1"
        ).strip().rstrip("/")
        self.mistral_api_key: str = _env("MISTRAL_API_KEY", "").strip()
        self.mistral_model: str = _env("MISTRAL_MODEL", "mistral-large-latest")

        # UMLS
        self.umls_api_key: str = _env("UMLS_API_KEY", "").strip()
        self.umls_sabs: str = _env("UMLS_SABS", "MSH")
        self.umls_base_url: str = _env("UMLS_BASE_URL", "https://uts-ws.nlm.nih.gov/rest")
        self.umls_max_synonyms: int = _int("UMLS_MAX_SYNONYMS", 5)
        self.umls_timeout: float = float(_env("UMLS_TIMEOUT", "30"))

        # Retrieval
        self.index_dir: Path = Path(_env("INDEX_DIR", "index"))
        self.corpus_path: Path = self.index_dir / "corpus.parquet"
        # local_mode: use only local FAISS + parquets, ignore pgvector.
        # Defaults to True when VECTOR_DB_URL is not set.
        self.local_mode: bool = _bool("LOCAL_MODE", not bool(_env("VECTOR_DB_URL", "")))
        self.vector_db_url: str = _env("VECTOR_DB_URL", "") if not self.local_mode else ""
        self.embedding_model: str = _env("EMBEDDING_MODEL", "ncbi/MedCPT-Query-Encoder")
        self.enable_dense: bool = _bool("ENABLE_DENSE", False)
        self.retrieval_primary: str = _env("RETRIEVAL_PRIMARY", "bm25").strip().lower()  # bm25 | splade
        self.bm25_depth: int = _int("BM25_DEPTH", 100)
        self.dense_depth: int = _int("DENSE_DEPTH", 100)
        self.rrf_k: float = float(_env("RRF_K", "60"))

        self.max_documents: int = _int("MAX_DOCUMENTS", 8)
        self.max_paper_tokens: int = _int("MAX_PAPER_TOKENS", 12000)  # 0 = unlimited full paper


        self.cross_encoder_model: str = _env("CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder")


        # Research budgets (singular agents flow; stage timeouts and worker
        # fan-out live in src/agents/timeouts.py as XDEEP_* env knobs).
        self.agentic_v3_evidence_target: int = _int("AGENTIC_V3_EVIDENCE_TARGET", 3)
        self.agentic_v3_max_searches: int = _int("AGENTIC_V3_MAX_SEARCHES", 5)
        self.agentic_v3_max_retrieval_rounds: int = _int("AGENTIC_V3_MAX_RETRIEVAL_ROUNDS", 5)
        self.agentic_v3_papers_per_search: int = _int("AGENTIC_V3_PAPERS_PER_SEARCH", 5)

        # Logfire observability (PydanticAI GenAI tracing + app trace events).
        # LOGFIRE_ENABLED: auto (default; enabled iff credentials/token are
        # present) | 1/on (force on) | 0/off (force off). PydanticAI's own env
        # vars (LOGFIRE_TOKEN, LOGFIRE_CONSOLE, ...) are read directly by the
        # Logfire SDK.
        self.logfire_mode: str = _env("LOGFIRE_ENABLED", "auto").strip().lower()
        self.logfire_service_name: str = _env("LOGFIRE_SERVICE_NAME", "medrag")
        self.logfire_environment: str = _env("LOGFIRE_ENVIRONMENT", "development")

        # Memory + Context layer (src/memory). Never a source of medical
        # fact — only persistent research state + context construction.
        # MEMORY_BACKEND: auto (try Postgres, fall back to in-memory) |
        # postgres (fail hard if unconnected) | memory (no DB).
        # MEMORY_EMBEDDER: hash (deterministic, offline, default) |
        # medcpt (optional ncbi/MedCPT-Query-Encoder, dim must match schema).
        self.memory_backend: str = _env("MEMORY_BACKEND", "auto").strip().lower()
        self.memory_embedder: str = _env("MEMORY_EMBEDDER", "hash").strip().lower()
        self.memory_embed_dim: int = _int("MEMORY_EMBED_DIM", 256)
        self.memory_schema: str = _env("MEMORY_SCHEMA", "medrag_memory").strip()
        # Total token budget for the memory/context region composed into
        # downstream calls (evidence-vs-memory split happens in context.py).
        self.memory_context_tokens: int = _int("MEMORY_CONTEXT_TOKENS", 1800)
        # Dedup / retriave thresholds for memory objects (cosine-similarity
        # space of the configured embedder).
        self.memory_claim_min_sim: float = float(_env("MEMORY_CLAIM_MIN_SIM", "0.86"))
        self.memory_retrieve_min_sim: float = float(_env("MEMORY_RETRIEVE_MIN_SIM", "0.15"))
        # Data-driven revalidation: 0 disables the horizon check entirely
        # (staleness is then only explicit mark_stale + contradicting-evidence
        # triggers — no arbitrary expiration periods).
        self.memory_staleness_days: int = _int("MEMORY_STALENESS_DAYS", 0)
        self.memory_consolidation_batch: int = _int("MEMORY_CONSOLIDATION_BATCH", 200)

        # Prompts
        self.prompt_dir: Path = Path(__file__).resolve().parent / "prompts"

    def model_settings(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "timeout": self.request_timeout}


@lru_cache(maxsize=1)
def config() -> AppConfig:
    return AppConfig()