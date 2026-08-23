"""Stage configuration (V2.4).

Every PageIndex knob lives here, driven by environment variables - no model is
hardcoded. The intended reasoning model (MedGemma) is configured through
PAGEINDEX_LLM_BASE_URL / PAGEINDEX_LLM_API_KEY and PAGEINDEX_CHAT_MODEL.

Indexing prefers heuristic/Flash-style tree generation (pageindex md_to_tree -
no LLM); only query-time navigation uses a reasoning model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MD_DIR = "index/pageindex_md"
DEFAULT_TREE_DIR = "index/pageindex_trees"
DEFAULT_SDK_STORAGE = "index/pageindex_sdk"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class StageConfig:
    """Configuration for the PageIndex stages (indexing + navigation)."""

    # locations
    md_dir: Path = Path(DEFAULT_MD_DIR)              # existing converted Markdown corpus
    tree_dir: Path = Path(DEFAULT_TREE_DIR)          # persisted native PageIndex trees
    sdk_storage: Path = Path(DEFAULT_SDK_STORAGE)    # SDK local document store

    # indexing (heuristic; no LLM)
    index_model: str = ""                            # PAGEINDEX_INDEX_MODEL (unused when md_to_tree)

    # query-time navigation (reasoning model)
    chat_model: str = ""                             # PAGEINDEX_CHAT_MODEL
    temperature: float = 0.0                         # PAGEINDEX_LLM_TEMPERATURE (or PAGEINDEX_TEMPERATURE)
    max_steps: int = 8                               # PAGEINDEX_MAX_TURNS (agent max turns)
    max_nodes: int = 40                              # PAGEINDEX_MAX_NODES (payload cap)
    timeout: int = 180                               # PAGEINDEX_LLM_TIMEOUT (seconds)

    # custom LLM backend (MedGemma behind our HTTP endpoint)
    llm_base_url: str = ""                           # PAGEINDEX_LLM_BASE_URL ends at /v1
    llm_api_key: str = ""                            # PAGEINDEX_LLM_API_KEY

    # behavior
    rebuild: bool = False                            # force re-index even if cached
    allow_structural_fallback: bool = False          # NEVER auto-lexical-fallback for the MedGemma experiment

    def has_llm(self) -> bool:
        return bool(self.llm_base_url and self.chat_model)

    def as_dict(self) -> dict:
        return {
            "md_dir": str(self.md_dir),
            "tree_dir": str(self.tree_dir),
            "sdk_storage": str(self.sdk_storage),
            "index_model": self.index_model,
            "chat_model": self.chat_model,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "max_nodes": self.max_nodes,
            "timeout": self.timeout,
            "llm_base_url": self.llm_base_url,
            "llm_api_key": "***" if self.llm_api_key else "",
            "allow_structural_fallback": self.allow_structural_fallback,
        }


def config_from_env(overrides: dict | None = None) -> StageConfig:
    """Build StageConfig from environment variables + explicit overrides."""
    cfg = StageConfig(
        md_dir=Path(_env("PAGEINDEX_MD_DIR", DEFAULT_MD_DIR)),
        tree_dir=Path(_env("PAGEINDEX_TREE_DIR", DEFAULT_TREE_DIR)),
        sdk_storage=Path(_env("PAGEINDEX_SDK_STORAGE", DEFAULT_SDK_STORAGE)),
        index_model=_env("PAGEINDEX_INDEX_MODEL"),
        chat_model=_env("PAGEINDEX_CHAT_MODEL"),
        temperature=_env_float("PAGEINDEX_LLM_TEMPERATURE",
                             _env_float("PAGEINDEX_TEMPERATURE", 0.0)),
        max_steps=_env_int("PAGEINDEX_MAX_TURNS", _env_int("PAGEINDEX_MAX_STEPS", 8)),
        max_nodes=_env_int("PAGEINDEX_MAX_NODES", 40),
        timeout=_env_int("PAGEINDEX_LLM_TIMEOUT", _env_int("PAGEINDEX_TIMEOUT", 180)),
        llm_base_url=_env("PAGEINDEX_LLM_BASE_URL", _env("LLM_BASE_URL")),
        llm_api_key=_env("PAGEINDEX_LLM_API_KEY"),
        rebuild=_env("PAGEINDEX_REBUILD", "0").strip().lower() in ("1", "true", "yes"),
        allow_structural_fallback=_env("PAGEINDEX_STRUCTURAL_FALLBACK", "0").strip().lower()
        in ("1", "true", "yes"),
    )
    if overrides:
        for key, val in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
    return cfg
