"""Deterministic section classification shared by the parser and chunker.

Sections are classified as one of three kinds:

- ``content``        : scientific body text (retrieval eligible)
- ``administrative`` : funding, acknowledgements, ethics, etc.
- ``references``     : bibliography entries

Both the JATS parser (when classifying a section title) and the AST chunker
(when falling back to title-based classification) use this single source of
truth so the rules can never drift apart.
"""

from __future__ import annotations

# Section titles classified as administrative (case/whitespace insensitive).
ADMIN_TITLES = frozenset({
    "funding",
    "funding sources",
    "acknowledgements",
    "acknowledgments",
    "author contributions",
    "authors contributions",
    "data availability",
    "data availability statement",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "supplementary material",
    "supplementary materials",
    "notes",
})

REF_TITLES = frozenset({
    "references",
    "reference",
    "bibliography",
})

# Titles that contain "reference" but are *not* reference lists.
REF_SUBSTRING_EXCLUSIONS = frozenset({
    "reference value",
    "reference values",
    "reference range",
    "reference ranges",
    "reference interval",
    "reference standard",
})


def classify_section_title(title: str) -> str:
    """Classify a section title as ``content``, ``administrative`` or ``references``."""
    norm = " ".join(title.lower().split())
    if not norm:
        return "content"

    for exclusion in REF_SUBSTRING_EXCLUSIONS:
        if exclusion in norm:
            return "content"

    if norm in REF_TITLES:
        return "references"
    if "reference" in norm:
        return "references"

    if norm in ADMIN_TITLES:
        return "administrative"
    if (
        "funding" in norm
        or "conflict" in norm
        or "competing interest" in norm
        or "data availability" in norm
        or "acknowledg" in norm
        or "supplementary" in norm
        or ("author" in norm and "contribution" in norm)
    ):
        return "administrative"

    return "content"
