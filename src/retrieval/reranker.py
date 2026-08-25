"""Deterministic intent reranking over the BM25+dense union.

Flow (as designed):

    BM25 top 30 ──┐
                  ├── UNION + dedupe (by chunk_id)
    Dense top 30 ─┘
                  ↓
          ONE intent reranker  (scores the whole union ONCE)
                  ↓
          paper diversification (spread top-K across distinct papers)
                  ↓
                  top K
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from src.agents.planner import SubQuery

def rerank_candidates(sub: SubQuery, documents: Sequence[Any]) -> List[Any]:
    """Compat shim (legacy probe scripts): sort docs by rrf_score."""
    scored = sorted(enumerate(documents), key=lambda t: -getattr(t[1], "rrf_score", 0.0))
    for rank, (idx, doc) in enumerate(scored, 1):
        doc.rank = rank
    return [doc for _, doc in scored]



PERCENT_RE = re.compile(r"\b(?:percent|percentage|proportion|rate)\b", re.IGNORECASE)
PVALUE_RE = re.compile(r"\bp-?values?\b|\bstatistically significant\b|confidence interval", re.IGNORECASE)
ODDS_RE = re.compile(r"\bodds ratio\b|\bhazard ratio\b|\brisk ratio\b", re.IGNORECASE)

FIELD_PATTERNS = {"percentage": PERCENT_RE, "p-value": PVALUE_RE, "odds ratio": ODDS_RE}

# evidence-type priors applied by the intent reranker
TYPE_BONUS = {
    "table_summary": 0.10,
    "table_row": 0.07,
    "figure": 0.05,
    "paragraph": 0.0,
}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _tokens(text: str) -> set:
    return set(_norm(text).split())


def _field_bonus(text: str) -> float:
    hits = sum(1 for p in FIELD_PATTERNS.values() if p.search(text or ""))
    return 0.05 * min(hits, 2)


def _type_bonus(doc: Any) -> float:
    node = (doc.node_type or "paragraph").lower()
    bonus = TYPE_BONUS.get(node, 0.0)
    if node == "paragraph" and "results" in (doc.section or "").lower():
        bonus = 0.04
    return bonus


def intent_score(sub: SubQuery, doc: Any) -> float:
    """Score ONE candidate against the subquery intent (union-wide, same call).

    Components:
      - overlap of the doc text with the subquery query + target +
        evidence_required terms (the topically most important part),
      - plus evidence-type bonus (table/figure/results) and statistical-field
        bonus (percent / p-value / odds-ratio mentions).
    Returns a float in roughly [0, 1].
    """
    text = doc.text or ""
    text_norm = _norm(text)
    text_tokens = set(text_norm.split())

    # build the union of terms we care about: query + target + evidence_required + terminology
    terms: List[str] = []
    for s in [sub.query, sub.target, *sub.evidence_required]:
        if s:
            terms.append(s)
    term_tokens = _tokens(" ".join(terms))
    for term in sub.terminology:
        if term:
            term_tokens |= _tokens(term)

    if not term_tokens:
        overlap = 0.0
    else:
        overlap = len(text_tokens & term_tokens) / max(len(term_tokens), 1)

    type_b = _type_bonus(doc)
    field_b = _field_bonus(text)

    # bonus if any evidence_required phrase appears ~verbatim in the chunk
    phrase_hits = 0
    for req in sub.evidence_required:
        if _norm(req) and _norm(req) in text_norm:
            phrase_hits += 1
    phrase_b = 0.05 * min(phrase_hits, 3)

    return min(1.0, overlap + type_b + field_b + phrase_b)


def rerank_union(
    sub: SubQuery,
    union_docs: List[Any],
    *,
    top_k: int = 60,
) -> List[Dict[str, Any]]:
    """Score the WHOLE union once, return list sorted by intent score.

    Each entry: {doc, intent_score, topical, type_bonus, field_bonus}
    """
    scored: List[Dict[str, Any]] = []
    for doc in union_docs:
        text = doc.text or ""
        scored.append({
            "doc": doc,
            "intent_score": round(intent_score(sub, doc), 5),
        })
    scored.sort(key=lambda d: d["intent_score"], reverse=True)
    return scored[:top_k]


def diversify_papers(
    reranked: List[Dict[str, Any]],
    *,
    top_k: int = 8,
    max_per_paper: int = 2,
) -> List[Any]:
    """Greedy paper diversification over the intent-reranked list.

    Walk the list in intent-score order; keep a doc unless that paper already
    contributed max_per_paper docs.  Guarantees top-K is not one-paper-heavy.
    """
    picked: List[Any] = []
    counts: Dict[str, int] = {}
    for entry in reranked:
        if len(picked) >= top_k:
            break
        doc = entry["doc"]
        paper = doc.document_id or "unknown"
        if counts.get(paper, 0) >= max_per_paper:
            continue
        counts[paper] = counts.get(paper, 0) + 1
        picked.append(doc)
    return picked


def union_rerank_diversify(
    sub: SubQuery,
    bm25_docs: Sequence[Any],
    dense_docs: Sequence[Any],
    *,
    top_k: int = 8,
    max_per_paper: int = 2,
    score_cap: int = 60,
) -> List[Any]:
    """The full flow the user wants.

    BM25 top N + Dense top N -> union + dedupe -> ONE intent reranker ->
    paper diversification -> top K.
    """
    # union, keeping per-method provenance
    seen: Dict[str, Any] = {}
    for doc in [*bm25_docs, *dense_docs]:
        cid = doc.chunk_id
        if cid in seen:
            # merge method provenance
            for m in doc.methods:
                if m not in seen[cid].methods:
                    seen[cid].methods = [*seen[cid].methods, m]
            continue
        seen[cid] = doc

    union = list(seen.values())
    reranked = rerank_union(sub, union, top_k=score_cap)
    return diversify_papers(reranked, top_k=top_k, max_per_paper=max_per_paper)
