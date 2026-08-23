"""MedRag Retrieval V2 — intent-aware, paper-first retrieval architecture
(V2.1: PageIndex paper-local navigation + table-aware evidence context).

Architecture (see the V2.1 specification):

    question → structured query understanding (MedGemma / deterministic)
        → clinical_entities / query_targets / requested_fields / populations
        → requirements H1..Hn (ONE per genuine evidence obligation)
        → retrieval variants V0..Vn per requirement
        → per-query hybrid (BM25 + pgvector)      [distribution: per variant]
        → query-local RRF                         [no global chunk-level fusion]
        → paper-level aggregation                 [paper = discovery unit]
        → per-branch shortlists ∪ cross-query rank  [branch preservation]
        → top papers → PageIndex paper-local navigation (precomputed trees)
        → logical document index → local hybrid inside papers (per variant)
        → bounded parent/child structural expansion (table assembly)
        → MedCPT cross-encoder (requirement ↔ contextual evidence only)
        → intent-aware contextual scoring with explicit penalties
        → greedy requirement-coverage selection → focused repair → coverage

Core invariants:

1. retrieval happens per requirement branch / variant and is never globally
   fused at the chunk level before paper-level evidence is preserved;
2. a paper that is essential to ONE branch is guaranteed to survive paper
   selection (per-branch shortlist union);
3. PageIndex runs only AFTER paper discovery, over precomputed per-paper tree
   artifacts - never over the global corpus, never as a vector replacement;
4. table rows are scored WITH their assembled table context (header + row +
   footnotes), and requested fields are detected from table headers;
5. the expensive cross-encoder runs only on a small, structurally-constrained
   candidate set; coverage is honest - no generic substitution for missing
   evidence.
"""

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG, config_from_env
from medrag.retrieval_v2.models import (
    Terminology,
    ClinicalEntity,
    RetrievalVariant,
    PageIndexHit,
    Requirement,
    SearchQuery,
    RetrievalEvent,
    QueryLocalResult,
    PaperQueryScore,
    PaperSelection,
    EvidenceCandidate,
    V2Plan,
    CoverageReport,
)

__all__ = [
    "V2Config",
    "DEFAULT_CONFIG",
    "Terminology",
    "ClinicalEntity",
    "RetrievalVariant",
    "PageIndexHit",
    "Requirement",
    "SearchQuery",
    "RetrievalEvent",
    "QueryLocalResult",
    "PaperQueryScore",
    "PaperSelection",
    "EvidenceCandidate",
    "V2Plan",
    "CoverageReport",
]
