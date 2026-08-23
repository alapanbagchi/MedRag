"""LLM-based biomedical query expansion.

Takes the original query, extracted concepts, and BioPortal context,
then generates an expanded search query optimized for BM25/dense retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from medrag.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are a biomedical search query optimizer. "
    "Return ONLY the expanded query string, nothing else."
)

USER_PROMPT = """Given this medical question and the extracted biomedical context, generate an optimized search query for retrieving relevant passages from a biomedical literature database.

Original question: {query}

Extracted biomedical concepts:
{concepts_str}

BioPortal ontology terms (synonyms and related terms):
{bioportal_str}

Create an expanded query that:
1. Preserves the specific clinical intent of the original question
2. Adds relevant synonyms and alternative terms from BioPortal
3. Includes key MeSH/ontology terms that papers would use
4. Uses natural language suitable for both BM25 and dense retrieval
5. Does NOT change the meaning — only broadens the vocabulary

Return ONLY the expanded query string (no quotes, no explanation)."""


def expand_query(
    query: str,
    concepts: List[Dict[str, Any]],
    bioportal_terms: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[LLMClient] = None,
) -> str:
    """Generate an expanded search query from the original + context.

    The expanded query is used for BM25 and dense retrieval.
    The original query is preserved for intent extraction and MedCPT.
    """
    from medrag.trace import get_trace
    trace = get_trace()

    if not llm or not llm.is_available():
        expanded = _expand_query_rules(query, concepts, bioportal_terms)
        trace.log(
            "query_expand_rules",
            params={"query": query, "n_concepts": len(concepts), "n_bioportal": len(bioportal_terms or [])},
            result={"expanded_query": expanded},
        )
        return expanded

    concepts_str = _format_concepts(concepts)
    bioportal_str = _format_bioportal(bioportal_terms or [])

    try:
        prompt = USER_PROMPT.format(
            query=query,
            concepts_str=concepts_str,
            bioportal_str=bioportal_str,
        )
        result = llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=300)
        if result and len(result) > 5:
            trace.log(
                "query_expand_llm",
                params={"query": query, "n_concepts": len(concepts), "n_bioportal": len(bioportal_terms or [])},
                result={"expanded_query": result.strip()},
            )
            return result.strip()
    except Exception:  # noqa: BLE001
        pass

    expanded = _expand_query_rules(query, concepts, bioportal_terms)
    trace.log(
        "query_expand_fallback",
        params={"query": query},
        result={"expanded_query": expanded},
    )
    return expanded


def _expand_query_rules(
    query: str,
    concepts: List[Dict[str, Any]],
    bioportal_terms: Optional[List[Dict[str, Any]]],
) -> str:
    """Rule-based query expansion fallback."""
    parts = [query]

    # Add concept names and synonyms
    for concept in concepts:
        name = concept.get("name", "")
        syns = concept.get("synonyms", [])
        if name and name.lower() not in query.lower():
            parts.append(name)
        for syn in syns[:3]:
            if syn.lower() not in query.lower():
                parts.append(syn)

    # Add BioPortal pref_labels
    if bioportal_terms:
        for term in bioportal_terms[:5]:
            label = term.get("pref_label", "")
            if label and label.lower() not in query.lower():
                parts.append(label)

    # Deduplicate while preserving order
    seen = set()
    unique_parts = []
    for p in parts:
        low = p.lower().strip()
        if low and low not in seen:
            seen.add(low)
            unique_parts.append(p.strip())

    return " ".join(unique_parts)


def _format_concepts(concepts: List[Dict[str, Any]]) -> str:
    lines = []
    for c in concepts:
        name = c.get("name", "")
        ctype = c.get("type", "other")
        desc = c.get("description", "")
        syns = c.get("synonyms", [])
        line = f"- {name} (type: {ctype})"
        if desc:
            line += f": {desc}"
        if syns:
            line += f" [synonyms: {', '.join(syns[:5])}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"


def _format_bioportal(terms: List[Dict[str, Any]]) -> str:
    lines = []
    for t in terms:
        label = t.get("pref_label", "")
        onto = t.get("ontology", "")
        sem = t.get("semantic_type", "")
        syns = t.get("synonyms", [])
        line = f"- {label} ({onto}, {sem})"
        if syns:
            line += f" [synonyms: {', '.join(syns[:3])}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"
