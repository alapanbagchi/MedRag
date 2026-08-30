"""Planner agent with UMLS tool."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from typing import Any, List

from pydantic import BaseModel, Field

logger = logging.getLogger("src.planner")

# Template junk small models echo into string fields (observed verbatim from
# gemma: an entity whose text was literally 'terminology:true,text:').
_JUNK_RE = re.compile(r"[,:\{\}\[\]]")
_JUNK_WORDS = {"true", "false", "null", "none", "text", "role", "terminology",
               "id", "focus", "query", "target", "string"}


# --- Models ---

class Entity(BaseModel):
    text: str
    role: str = "condition"
    terminology: bool = True


class SubQuery(BaseModel):
    id: str
    target: str
    focus: str = "evidence"
    query: str = ""  # natural language query for retrieval
    evidence_required: List[str] = Field(default_factory=list)
    terminology: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)  # UMLS-enriched terms


class QueryPlan(BaseModel):
    original_query: str = ""
    question_type: str = "factual"
    entities: List[Entity] = Field(default_factory=list)
    targets: List[str] = Field(default_factory=list)
    subqueries: List[SubQuery] = Field(default_factory=list)


class ClinicalEntity(BaseModel):
    surface_form: str
    base_concept: str = ""
    role: str = "condition"
    preferred_name: str = ""
    ontology: str = ""
    cui: str = ""
    synonyms: List[str] = Field(default_factory=list)


class EnrichedPlan:
    def __init__(self, plan: QueryPlan, clinical_entities: List[ClinicalEntity] = None, warnings: List[str] = None):
        self.plan = plan
        self.clinical_entities = clinical_entities or []
        self.warnings = warnings or []

    def __getattr__(self, name):
        return getattr(self.plan, name)


# --- UMLS Tool ---

_UMLS_TOOL_CACHE: dict = {}


async def search_umls(term: str, transport: Any = None) -> dict:
    """Search UMLS/MeSH for a term. Returns preferred name, CUI, and synonyms.

    Cached per process: models tend to re-look-up identical terms across
    retries, and every miss costs one or two HTTP round-trips.
    ``transport`` injects an httpx transport (tests).
    """
    from src.config import AppConfig
    from src.trace import get_trace
    import httpx

    trace = get_trace()
    key = " ".join((term or "").split()).lower()
    if not key:
        return {"term": term, "found": False}
    if key in _UMLS_TOOL_CACHE:
        cached = dict(_UMLS_TOOL_CACHE[key])
        cached["cached"] = True
        return cached

    trace.tool("search_umls", {"term": term})
    cfg = AppConfig()
    if not cfg.umls_api_key:
        result = {"term": term, "found": False}
        _UMLS_TOOL_CACHE[key] = dict(result)
        trace.log("tool_result", tool="search_umls", result=result)
        return result
    base_url = cfg.umls_base_url.rstrip("/")
    kwargs = {"timeout": 30.0}
    if transport is not None:
        kwargs["transport"] = transport
    async with httpx.AsyncClient(**kwargs) as client:
        try:
            resp = await client.get(f"{base_url}/search/current", params={
                "string": term, "apiKey": cfg.umls_api_key, "sabs": cfg.umls_sabs,
                "returnIdType": "concept", "searchType": "words", "pageSize": 5,
            })
            resp.raise_for_status()
            results = (resp.json().get("result") or {}).get("results") or []
            if not results:
                result = {"term": term, "found": False}
                _UMLS_TOOL_CACHE[key] = dict(result)
                trace.log("tool_result", tool="search_umls", result=result)
                return result

            selected = results[0]
            cui = str(selected.get("ui", ""))
            preferred = str(selected.get("name", ""))

            synonyms = []
            if cui:
                resp2 = await client.get(f"{base_url}/content/current/CUI/{cui}/atoms", params={
                    "apiKey": cfg.umls_api_key, "sabs": cfg.umls_sabs, "language": "ENG", "pageSize": 10,
                })
                if resp2.is_success:
                    seen = {preferred.lower()}
                    for atom in resp2.json().get("result") or []:
                        name = str(atom.get("name", "")).strip()
                        if name and name.lower() not in seen:
                            seen.add(name.lower())
                            synonyms.append(name)
                            if len(synonyms) >= 5:
                                break

            result = {"term": term, "found": True, "cui": cui, "preferred_name": preferred, "synonyms": synonyms}
            _UMLS_TOOL_CACHE[key] = dict(result)
            trace.log("tool_result", tool="search_umls",
                      result={"term": term, "found": True, "cui": cui,
                              "preferred_name": preferred, "n_synonyms": len(synonyms)})
            return result
        except Exception as exc:
            logger.warning("UMLS search failed: %s", exc)
            result = {"term": term, "found": False, "error": str(exc)}
            trace.log("tool_result", tool="search_umls", result=result)
            return result


def reset_umls_tool_cache() -> None:
    """Drop the tool cache (tests)."""
    _UMLS_TOOL_CACHE.clear()


# --- Agent ---

class PlannerAgent:
    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model
        from pydantic_ai import Agent

        from src.prompts.load import load_prompt

        self.config = config or AppConfig()
        self.model = model or build_model(self.config)
        self._prompt = load_prompt("legacy", "planner.txt")
        # ONE agent; structured output is chosen per-run by ask_structured.
        #
        # The UMLS tool is OFF by default. Live lookups inside the planning
        # loop made small models spiral (repeated lookups, junk entities like
        # bare acronyms) and multiplied LLM turns; terminology enrichment now
        # happens DETERMINISTICALLY in the orchestrator after planning
        # (UMLSClient.enrich_entities), where it cannot loop. Set
        # PLANNER_USE_UMLS_TOOL=1 to restore the legacy tool-using planner.
        tools = [search_umls] if getattr(self.config, "planner_use_umls_tool", False) else []
        self.use_tool = bool(tools)
        self.agent = Agent(
            self.model,
            system_prompt=self._prompt,
            name="planner",
            tools=tools,
        )

    async def plan(self, query: str) -> EnrichedPlan:
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        plan = await ask_structured(
            self.agent, query, QueryPlan,
            label="planner",
            max_tokens=self.config.agent_max_tokens,
            max_attempts=getattr(self.config, "max_llm_retries", 4),
        )

        if not plan.original_query:
            plan = plan.model_copy(update={"original_query": query})

        # Small models sometimes emit a plan with NO subqueries (observed with
        # gemma: entities/targets but an empty list). That silently no-ops the
        # whole pipeline; fall back to ONE single-hop subquery covering the
        # entire question instead.
        if not plan.subqueries:
            logger.warning("planner returned 0 subqueries; using single-hop fallback")
            trace.bullet("planner produced no subqueries -> deterministic single-hop fallback")
            fallback = SubQuery(
                id="H1",
                target=query.strip()[:120],
                query=query.strip(),
                focus="evidence",
                evidence_required=[],
            )
            plan = plan.model_copy(update={"subqueries": [fallback]})

        clinical_entities = [
            ClinicalEntity(surface_form=e.text, base_concept=e.text, role=e.role)
            for e in plan.entities if e.terminology
        ]

        return EnrichedPlan(plan, clinical_entities)


if __name__ == "__main__":
    from src.config import AppConfig

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is COPD exacerbation?"
    planner = PlannerAgent(config=AppConfig())

    async def _run():
        print(f"Query: {query}\n")
        enriched = await planner.plan(query)
        print(json.dumps(enriched.plan.model_dump(), indent=2))

    asyncio.run(_run())
