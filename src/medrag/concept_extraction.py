"""LLM-based biomedical concept extraction.

Extracts structured biomedical concepts from a user query using an LLM.
Each concept includes name, type, and description.
"""

from __future__ import annotations

from typing import Any, Dict, List

from medrag.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are a biomedical NLP expert. Extract biomedical concepts from the query. "
    "Return ONLY a valid JSON array."
)

USER_PROMPT = """Extract biomedical concepts from this medical question.

Question: {query}

Return a JSON array where each element is an object with:
- "name": the concept name (e.g. "CHA2DS2-VASc score", "atrial fibrillation")
- "type": one of "disease", "drug", "procedure", "symptom", "lab_test", "biomarker", "score", "anatomy", "gene", "other"
- "description": a brief 1-sentence description of what this concept means clinically

Focus on:
- Named entities (diseases, drugs, procedures, scales)
- Clinical measures (lab values, scores, thresholds)
- Anatomical/physiological terms

Do NOT include generic words like "patient", "study", "result", "treatment" unless they are part of a specific named entity.

Return ONLY the JSON array, no explanation."""


def extract_concepts(
    query: str,
    llm: LLMClient,
) -> List[Dict[str, Any]]:
    """Extract biomedical concepts from the original user query.

    Returns a list of dicts, each with 'name', 'type', 'description'.
    Falls back to basic NER if the LLM fails.
    """
    if not llm.is_available():
        return _extract_concepts_rules(query)

    try:
        prompt = USER_PROMPT.format(query=query)
        result = llm.generate_json(prompt, system=SYSTEM_PROMPT)

        if isinstance(result, list):
            # Validate structure
            validated = []
            for item in result:
                if isinstance(item, dict) and "name" in item:
                    validated.append({
                        "name": item.get("name", ""),
                        "type": item.get("type", "other"),
                        "description": item.get("description", ""),
                    })
            if validated:
                return validated
    except Exception:  # noqa: BLE001
        pass

    # Fallback
    return _extract_concepts_rules(query)


def _extract_concepts_rules(query: str) -> List[Dict[str, Any]]:
    """Rule-based concept extraction fallback."""
    import re

    concepts: List[Dict[str, Any]] = []

    # Find capitalized phrases (potential named entities)
    caps = re.findall(
        r'[A-Z][A-Za-z0-9-]+(?:[\s/-][A-Z][A-Za-z0-9-]+)*',
        query,
    )
    seen = set()
    for phrase in caps:
        low = phrase.lower()
        if low not in seen and len(phrase) > 2:
            seen.add(low)
            concepts.append({
                "name": phrase,
                "type": "other",
                "description": f"Medical entity mentioned in the query.",
            })

    return concepts[:10]
