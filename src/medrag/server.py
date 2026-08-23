"""MedRAG v2 Kaggle Server — /expand structured query-intelligence endpoint (V2.1).

POST /expand {"query": "..."}
    → {
        "query",
        "clinical_entities",     # [{surface_form, base_concept, modifiers, role,
                                 #    preferred_name, ontology, cui, synonyms, abbreviations}]
        "query_targets",         # [str]
        "requested_fields",      # ["percentage","p-value",...]
        "relationships",         # [str]
        "populations",           # [str]
        "retrieval_variants",    # [{id, text, source}]  (informational)
        "reranker_intent",       # {question_type, focus, target, outcome, requested_fields, evidence_types}
        "umls"                   # normalized ontology records for the base concepts
        "timings"
    }

The server runs the expensive MedGemma structured query understanding +
UMLS/MeSH/BioPortal enrichment on Kaggle. The LOCAL pipeline then builds the
requirement graph itself (one requirement per genuine evidence obligation) —
the server exposes query INTELLIGENCE, never a decomposed requirement list.

MedGemma runs locally on Kaggle (Ollama or vLLM). Env vars:
    LLM_BASE_URL      (default: http://127.0.0.1:11434)
    LLM_MODEL         (default: deepseek)
    BIOPORTAL_API_KEY (optional; BioPortal ontology search)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

app = FastAPI(title="MedRAG v2 Expand", version="2.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────

class ExpandRequest(BaseModel):
    query: str


class ExpandResponse(BaseModel):
    query: str
    clinical_entities: List[Dict[str, Any]]
    query_targets: List[str]
    requested_fields: List[str]
    relationships: List[str]
    populations: List[str]
    retrieval_variants: List[Dict[str, Any]]
    reranker_intent: Dict[str, Any]
    umls: List[Dict[str, Any]]
    timings: Dict[str, float]


# ── Global state (loaded once at startup) ─────────────────────────

_llm = None
_bioportal = None


def _load():
    global _llm, _bioportal
    from medrag.llm_client import LLMClient
    from medrag.ontology import BioPortalEnricher

    _llm = LLMClient()
    print(f"LLM: {_llm.base_url} / {_llm.model}")
    _bioportal = BioPortalEnricher()
    if _bioportal.api_key:
        print("BioPortal: configured")
    else:
        print("BioPortal: no API key (limited enrichment)")


@app.on_event("startup")
def startup():
    _load()


# ── Query intelligence builders ───────────────────────────────────

def _suggest_retrieval_variants(query: str, entities: List[Dict[str, Any]],
                                targets: List[str], fields: List[str]) -> List[Dict[str, Any]]:
    """Small set of compact retrieval variants (part 6), informational: the
    local planner regenerates the authoritative variants from its own state."""
    cond = next((e.get("surface_form", "") for e in entities if e.get("role") == "condition"), "")
    base = next((e.get("base_concept", "") for e in entities if e.get("role") == "condition"), cond)
    target = targets[0] if targets else ""
    variants: List[Dict[str, Any]] = []
    seen: set = set()

    def add(text: str, source: str) -> None:
        text = " ".join((text or "").split()).strip()
        low = text.lower()
        if not text or low in seen or len(variants) >= 5:
            return
        seen.add(low)
        variants.append({"id": f"V{len(variants)}", "text": text, "source": source})

    add(query, "original")
    add(f"{target} {cond}", "target+concept")
    if base and base != cond:
        add(f"{target} {base}", "base_concept")
    if fields:
        add(f"{target} {cond} {' '.join(fields)}", "focus+fields")
    return variants


def build_query_intelligence_response(
    query: str,
    llm: Any = None,
    bioportal: Any = None,
) -> Dict[str, Any]:
    """Produce the NEW structured /expand payload (V2.1 part 3).

    MedGemma structured query understanding is attempted first; the
    deterministic extractor is the backbone and always wins on surface forms.
    Ontology normalization follows MedGemma (part 5): BASE CONCEPT lookup
    first, modifiers preserved.
    """
    from medrag.query_intelligence import (
        extract_query_intelligence,
        merge_intelligence,
        normalize_entities,
        structured_query_understanding,
    )

    qi = extract_query_intelligence(query)
    llm_qi = structured_query_understanding(query, llm) if llm is not None else None
    if llm_qi is not None:
        qi = merge_intelligence(qi, llm_qi.to_dict())

    # ontology normalization (base concept first; fail-soft)
    normalized = normalize_entities(qi.clinical_entities, enricher=bioportal)

    entities = [e.to_dict() for e in normalized]
    umls = [
        {
            "surface_form": e.surface_form,
            "base_concept": e.base_concept,
            "modifiers": list(e.modifiers),
            "preferred_name": e.preferred_name,
            "ontology": e.ontology,
            "cui": e.cui,
            "synonyms": list(e.synonyms),
        }
        for e in normalized
        if e.preferred_name or e.ontology or e.synonyms
    ]

    reranker_intent = dict(qi.reranker_intent)
    reranker_intent.setdefault("evidence_types", ["table_row", "table_summary", "paragraph", "figure"])

    return {
        "query": qi.query,
        "clinical_entities": entities,
        "query_targets": list(qi.query_targets),
        "requested_fields": list(qi.requested_fields),
        "relationships": list(qi.relationships),
        "populations": list(qi.populations),
        "retrieval_variants": _suggest_retrieval_variants(
            qi.query, entities, qi.query_targets, qi.requested_fields),
        "reranker_intent": reranker_intent,
        "umls": umls,
    }


# ── /expand endpoint ──────────────────────────────────────────────

@app.post("/expand")
def expand(request: ExpandRequest) -> ExpandResponse:
    """Structured query understanding for one question (new contract)."""
    timings: Dict[str, float] = {}
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    t0 = time.perf_counter()
    payload = build_query_intelligence_response(query, _llm, _bioportal)
    timings["understand_ms"] = round((time.perf_counter() - t0) * 1000)
    timings["total_ms"] = round(sum(v for v in timings.values()), 1)

    return ExpandResponse(**payload, timings=timings)


# ── Health check ──────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "medrag-expand", "contract": "v2.1-structured-query-intelligence"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
