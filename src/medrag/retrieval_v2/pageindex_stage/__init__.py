"""PageIndex paper-local retrieval stages (V2.4): validation, indexing, navigation.

The first three stages of the PageIndex-based paper-local retrieval system:

    1. existing Markdown corpus validation
    2. Markdown -> PageIndex tree indexing (native, heuristic, no LLM)
    3. PageIndex query-time navigation over a selected paper

Only these three stages live here. BM25 / pgvector / RRF / global retrieval /
paper selection / MedGemma / UMLS / MedCPT / reranking / coverage / repair /
final answer generation are NOT touched (the global retriever connects later).

This package lives under medrag.retrieval_v2 (NOT a top-level 'pageindex'
directory) so it never shadows the installed pageindex SDK package.

Pipeline:

    ORIGINAL XML -> (jats_convert) -> Markdown -> (pageindex md_to_tree, native
    heuristic, no LLM) -> persistent tree -> (PageIndex local store) ->
    (SDK managed chat agent / LLM navigation) -> relevant document nodes
"""

from medrag.retrieval_v2.pageindex_stage.config import StageConfig, config_from_env
from medrag.retrieval_v2.pageindex_stage.models import (
    PaperDocument,
    SelectedNode,
    NavigationResult,
    NavigationError,
    MEDGEMMA_ENDPOINT_ERROR,
    PAGEINDEX_CLIENT_ERROR,
    DOCUMENT_REGISTRATION_ERROR,
    TREE_LOAD_ERROR,
    NAVIGATION_ERROR,
    NAVIGATION_EMPTY_RESULT,
)
from medrag.retrieval_v2.pageindex_stage.indexer import (
    validate_markdown,
    scan_corpus,
    build_index,
    load_index,
    print_tree,
)
from medrag.retrieval_v2.pageindex_stage.navigator import (
    navigation_objective,
    navigate,
    navigation_trace,
    verify_medgemma_endpoint,
    verify_document_registration,
)

__all__ = [
    "StageConfig",
    "config_from_env",
    "PaperDocument",
    "SelectedNode",
    "NavigationResult",
    "NavigationError",
    "validate_markdown",
    "scan_corpus",
    "build_index",
    "load_index",
    "print_tree",
    "navigation_objective",
    "navigate",
    "navigation_trace",
]
