"""Deterministic retrieval-query construction."""

from __future__ import annotations

from typing import List

from src.retrieval.plans import SubQuery


def build_subquery_variants(sub: SubQuery, max_variants: int = 5) -> List[str]:
    """Generate retrieval variants for one subquery."""
    variants: List[str] = []
    seen = set()

    def add(text: str) -> None:
        text = " ".join((text or "").split()).strip()
        key = text.lower()
        if not text or key in seen or len(variants) >= max_variants:
            return
        seen.add(key)
        variants.append(text)

    # V0 - original target
    add(sub.target)

    # V1..Vn - terminology variants
    for term in sub.terminology[:max(max_variants - len(variants) - 1, 1)]:
        add(" ".join(p for p in (sub.target, term) if p))

    return variants[:max_variants]
