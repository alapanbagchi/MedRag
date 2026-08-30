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

        # Budget
        default_tokens = 2048 if self.provider in ("gemini", "mistral") else 512
        self.agent_max_tokens: int = _int("AGENT_MAX_TOKENS", default_tokens)
        self.verifier_max_groups: int = _int("VERIFIER_MAX_GROUPS", 5)
        self.synthesizer_max_evidence: int = _int("SYNTHESIZER_MAX_EVIDENCE", 6)
        self.max_documents: int = _int("MAX_DOCUMENTS", 8)
        self.max_paper_tokens: int = _int("MAX_PAPER_TOKENS", 12000)  # 0 = unlimited full paper
        self.max_workers: int = _int("MAX_WORKERS", 8)
        self.max_retries: int = _int("MAX_RETRIES", 3)
        self.request_timeout: float = float(_env("REQUEST_TIMEOUT", "180"))
        self.temperature: float = float(_env("TEMPERATURE", "0.0"))
        self.max_plan_retries: int = _int("MAX_PLAN_RETRIES", 2)
        # Legacy: let the planner call search_umls itself. Off by default —
        # terminology enrichment is deterministic (orchestrator-side) because
        # tool loops made small models spiral.
        self.planner_use_umls_tool: bool = _bool("PLANNER_USE_UMLS_TOOL", False)

        # Shared LLM rate limiting (all agents draw from ONE bucket).
        # 0 disables accounting (fine for local/unlimited providers).
        default_tpm = 15000 if self.provider == "gemini" else 0
        legacy_budget = _env("VERIFIER_TOKEN_BUDGET", "").strip()
        legacy_val = int(legacy_budget) if legacy_budget.isdigit() else None
        self.tokens_per_minute: int = _int("GLOBAL_TOKENS_PER_MIN", legacy_val if legacy_val is not None else default_tpm)
        self.max_llm_retries: int = _int("MAX_LLM_RETRIES", 4)
        self.llm_retry_base_s: float = float(_env("LLM_RETRY_BASE_S", "2"))

        # Retrieval / verification loop
        self.max_query_rounds: int = _int("MAX_QUERY_ROUNDS", 3)
        self.min_papers: int = _int("MIN_PAPERS", 3)          # DISTINCT papers per subquery
        self.verify_batch_max_docs: int = _int("VERIFY_BATCH_MAX_DOCS", 6)
        self.verify_batch_max_tokens: int = _int("VERIFY_BATCH_MAX_TOKENS", 8000)
        self.rewrite_enabled: bool = _bool("REWRITE_ENABLED", True)
        self.enable_cross_encoder: bool = _bool("ENABLE_CROSS_ENCODER", False)
        self.cross_encoder_model: str = _env("CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder")

        # Trace verbosity: keep full untruncated chunk/unit texts in logs.txt.
        self.full_text_trace: bool = _bool("TRACE_FULL_TEXTS", False)

        # Agentic decomposition (src/agentic): max distinct subqueries kept.
        self.max_subqueries: int = _int("MAX_SUBQUERIES", 4)
        self.max_agent_rounds: int = _int("MAX_AGENT_ROUNDS", 6)
        self.agent_min_evidence: int = _int("AGENT_MIN_EVIDENCE", 3)

        # Agentic v2 (src/agentic_v2): orchestrator research loop budget.
        self.agentic_v2_max_rounds: int = _int("AGENTIC_V2_MAX_ROUNDS", 12)
        self.agentic_v2_max_global_retrieves: int = _int("AGENTIC_V2_MAX_GLOBAL_RETRIEVES", 6)
        # Per-call async timeouts (seconds) — a hanging decide/execute must not
        # stall the whole run; the state machine falls back / pivots instead.
        self.agentic_v2_orchestrator_timeout: float = float(_env("AGENTIC_V2_ORCHESTRATOR_TIMEOUT", "120"))
        self.agentic_v2_action_timeout: float = float(_env("AGENTIC_V2_ACTION_TIMEOUT", "180"))
        # Structured event stream for the agentic v2 UI (JSONL; empty = off).
        self.agentic_v2_events_file: str = _env("AGENTIC_V2_EVENTS_FILE", "").strip()

        # Agentic v3 (src/agentic_v3): master -> parallel workers ->
        # contradiction -> resolution -> final answer (V1 spec).
        self.agentic_v3_evidence_target: int = _int("AGENTIC_V3_EVIDENCE_TARGET", 3)
        self.agentic_v3_max_searches: int = _int("AGENTIC_V3_MAX_SEARCHES", 5)
        self.agentic_v3_max_retrieval_rounds: int = _int("AGENTIC_V3_MAX_RETRIEVAL_ROUNDS", 5)
        self.agentic_v3_papers_per_search: int = _int("AGENTIC_V3_PAPERS_PER_SEARCH", 5)
        self.agentic_v3_max_deep_inspections: int = _int("AGENTIC_V3_MAX_DEEP_INSPECTIONS", 3)
        self.agentic_v3_max_workers: int = _int("AGENTIC_V3_MAX_WORKERS", 4)
        # Per-stage async timeouts (seconds); a stalled stage must not hang the run.
        self.agentic_v3_master_timeout: float = float(_env("AGENTIC_V3_MASTER_TIMEOUT", "120"))
        self.agentic_v3_worker_timeout: float = float(_env("AGENTIC_V3_WORKER_TIMEOUT", "360"))
        self.agentic_v3_contradiction_timeout: float = float(
            _env("AGENTIC_V3_CONTRADICTION_TIMEOUT", "120"))
        self.agentic_v3_resolution_timeout: float = float(
            _env("AGENTIC_V3_RESOLUTION_TIMEOUT", "180"))
        self.agentic_v3_synthesis_timeout: float = float(
            _env("AGENTIC_V3_SYNTHESIS_TIMEOUT", "180"))
        # Structured event stream for the agentic v3 run (JSONL; empty = off).
        self.agentic_v3_events_file: str = _env("AGENTIC_V3_EVENTS_FILE", "").strip()

        # Mistral Batch API for verification (critic) calls.
        # VERIFIER_BATCH_ENABLED: auto (on when the verifier provider is
        # Mistral) | 1/on | 0/off. Batch requires a Mistral plan with billing
        # enabled; on failure the critic falls back to sequential calls.
        self.verifier_batch_enabled: str = _env("VERIFIER_BATCH_ENABLED", "auto").strip().lower()
        self.verifier_batch_max_requests: int = _int("VERIFIER_BATCH_MAX_REQUESTS", 100)
        self.verifier_batch_poll_seconds: float = float(_env("VERIFIER_BATCH_POLL_SECONDS", "5"))
        self.verifier_batch_timeout_seconds: float = float(_env("VERIFIER_BATCH_TIMEOUT_SECONDS", "900"))

        # Logfire observability (PydanticAI GenAI tracing + app trace events).
        # LOGFIRE_ENABLED: auto (default; enabled iff credentials/token are
        # present) | 1/on (force on) | 0/off (force off). PydanticAI's own env
        # vars (LOGFIRE_TOKEN, LOGFIRE_CONSOLE, ...) are read directly by the
        # Logfire SDK.
        self.logfire_mode: str = _env("LOGFIRE_ENABLED", "auto").strip().lower()
        self.logfire_service_name: str = _env("LOGFIRE_SERVICE_NAME", "medrag")
        self.logfire_environment: str = _env("LOGFIRE_ENVIRONMENT", "development")

        # Prompts
        self.prompt_dir: Path = Path(__file__).resolve().parent / "prompts"

    def model_settings(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "timeout": self.request_timeout}


@lru_cache(maxsize=1)
def config() -> AppConfig:
    return AppConfig()