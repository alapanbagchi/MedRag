"""Token estimation + abbreviation-aware sentence splitting (no LLM).

Two jobs only:
  1. size prose against the chunk budget (chars/4 - a rough token estimate,
     exact enough for budgeting; no encoder dependency lives here),
  2. split paragraphs into sentences without breaking abbreviations
     ("Fig. 3", "e.g.", "No. of", decimals).

The chunk size knobs live here so every module agrees on the numbers.
"""

from __future__ import annotations

import re
from typing import List


# Chunk size knobs (sized for a downstream MedCPT 512-token encoder, but the
# chunker itself never loads one - encoding is a separate step).
_SPLIT_OVERLAP_FRACTION = 0.25          # default carry budget = 25% of max_tokens
_DEFAULT_SPLIT_OVERLAP_SENTENCES = 2    # last N sentences seed the next piece


def _estimate_tokens(text: str) -> int:
    """Rough token count (chars divided by 4). Good enough to budget by."""
    return max(1, (len(text or "") + 3) // 4)


def _truncate_tokens(text: str, limit: int) -> str:
    """Truncate text to roughly limit tokens (chars/4). Never raises."""
    if limit <= 0:
        return ""
    if _estimate_tokens(text) <= limit:
        return text
    return text[: limit * 4].strip()


# Protected abbreviations: a period after one of these is NOT a sentence end.
_ABBREV_RE = re.compile(
    r"(?i)(?:^|\s)(?:e\.g|i\.e|et al|vs|viz|cf|ca|approx|figs?|tabs?|refs?|no|"
    r"st|mt|dr|mr|mrs|ms|prof|vol|pp|suppl|sp|spp|var|al|inc|co|ltd|etc)\.$"
)


def _ends_with_abbrev(part: str) -> bool:
    """True when part ends in a protected abbreviation or a capital initial
    ("P."), so it is re-joined with the following part."""
    s = (part or "").rstrip()
    if not s:
        return False
    if _ABBREV_RE.search(s):
        return True
    return bool(re.search(r"\b([A-Z])\.$", s))


def _split_sentences(text: str) -> List[str]:
    """Split prose into sentences without breaking abbreviations, initials,
    decimals or unit values.

    Boundary = sentence-ending punctuation + whitespace + a sentence-ish start
    (uppercase letter, digit, opening quote/bracket). The abbreviation check
    re-merges the remaining false boundaries. Deterministic, no LLM.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\u201c])", text)
    merged: List[str] = []
    for part in parts:
        if merged and _ends_with_abbrev(merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged
