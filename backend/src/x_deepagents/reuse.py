"""Reuse boundary: the ONLY place x_deepagents reaches into the existing
pipeline. Everything reused here is a non-LLM capability (tools) or prompt
text - never an agent. Agents, state, graph and rules are what deepagents +
LangGraph replace, so they are NOT re-exported.
"""

from __future__ import annotations

from typing import Any

from src.config import AppConfig  # shared provider / UMLS / pgvector config

__all__ = [
    "AppConfig",
    "load_prompt",
    "get_hybrid_retriever",
    "get_paper_retriever",
    "get_umls_enricher",
    "get_terminology_enricher",
]


def load_prompt(subdir: str, name: str) -> str:
    """Load a reusable prompt text from the backend prompts tree."""
    from src.prompts.load import load_prompt as _load

    return _load(subdir, name)


def get_hybrid_retriever(config: Any = None) -> Any:
    """HybridRetrieverTool (BM25/dense + pgvector + full-unit restore)."""
    from src.tools.retrieval import HybridRetrieverTool

    return HybridRetrieverTool(config=config)


def get_paper_retriever(config: Any = None) -> Any:
    """PaperRetrieverTool (expand hits to full paragraph/table/figure)."""
    from src.tools.paper_retriever import PaperRetrieverTool

    return PaperRetrieverTool(config=config)


def get_umls_enricher(config: Any = None) -> Any:
    """UMLSEnricher (UMLS/MeSH preferred names + synonyms)."""
    from src.tools.umls import UMLSEnricher

    return UMLSEnricher(config=config)


def get_terminology_enricher(config: Any = None) -> Any:
    """TerminologyEnricher (per-task terminology pool)."""
    from src.tools.terminology import TerminologyEnricher

    return TerminologyEnricher(config=config)
