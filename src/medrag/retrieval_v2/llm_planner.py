"""LLM-assisted query planning for retrieval V2 (V2.1 architecture repair).

NEW /expand contract (part 3). The /expand response is a structured
query-intelligence object:

    {
      "query": "...",
      "clinical_entities": [...],
      "query_targets": [...],
      "requested_fields": [...],
      "relationships": [...],
      "populations": [...],
      "retrieval_variants": [...],
      "reranker_intent": {...},
      "umls": [...]
    }

The old /expand -> plan_from_expand contract that turned the server's
'evidence_requirements' list into H1/H2/H3/H4 statistical branches is REMOVED.
plan_from_expand now maps the structured object onto the deterministic
planner's QueryIntelligence and lets the planner build ONE requirement per
genuine evidence obligation.

MedGemma's job (part 4) is STRUCTURED QUERY UNDERSTANDING (clinical entities
with surface_form/base_concept/modifiers/role, query targets, requested
fields, relationships, populations, question type, search concepts) - never
free decomposition into requirements.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from medrag.query_intelligence import (
    QueryIntelligence,
    intelligence_from_expand_payload,
    structured_query_understanding,
)
from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from medrag.retrieval_v2.models import V2Plan


def _lower(s: str) -> str:
    return (s or "").lower()


def call_expand(url: str, question: str, timeout: int = 180) -> Dict[str, Any]:
    """POST the question to the Kaggle /expand endpoint (MedGemma server)."""
    import requests
    url = url.rstrip("/")
    expand_url = f"{url}/expand"
    print(f"  Calling /expand: {expand_url}")
    resp = requests.post(expand_url, json={"query": question}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    n_entities = len(data.get("clinical_entities", data.get("concepts", [])))
    print(f"  /expand: clinical_entities={n_entities}, keys={sorted(data.keys())}")
    return data


def plan_from_expand(question: str, expand_data: Dict[str, Any],
                     config: Optional[V2Config] = None) -> V2Plan:
    """Build a V2Plan from the NEW structured /expand response.

    The server's structured query understanding is folded into the
    deterministic planner; the server NEVER gets to dictate the requirement
    graph. A question with percentages + p-values and one condition stays ONE
    requirement.
    """
    from medrag.retrieval_v2.planner import plan_question

    cfg = config or DEFAULT_CONFIG
    qi = intelligence_from_expand_payload(expand_data)
    if not qi.query:
        qi.query = question
    plan = plan_question(question, cfg, intelligence=qi)
    plan.planner_method = "expand+plan"
    plan.reranker_intent = qi.reranker_intent or dict(plan.reranker_intent)
    return plan


def structured_plan_with_llm(question: str, llm: Any,
                             config: Optional[V2Config] = None) -> V2Plan:
    """Plan via MedGemma structured query understanding (direct LLM path).

    Falls back to the deterministic planner when the LLM is unavailable or
    returns unusable output. The LLM only enriches query intelligence; the
    requirement graph is always built by the deterministic obligation logic.
    """
    from medrag.retrieval_v2.planner import plan_question

    qi = structured_query_understanding(question, llm)
    if qi is None:
        plan = plan_question(question, config)
        plan.warnings.append("LLM structured understanding unavailable, using deterministic")
        return plan
    plan = plan_question(question, config, intelligence=qi)
    plan.planner_method = "llm_structured"
    return plan


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# Backwards-compatible alias (a few callers may still use the old name).
def decompose_plan_with_llm(question: str, llm: Any,
                            config: Optional[V2Config] = None) -> V2Plan:
    """Deprecated alias of structured_plan_with_llm (same semantics)."""
    return structured_plan_with_llm(question, llm, config)
