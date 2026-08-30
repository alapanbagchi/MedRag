"""Shared utilities: JSON parsing, think-block stripping, quote grounding."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Tuple, Type

COMPACTION_NUDGE = (
    "\n\nIMPORTANT: your previous output was invalid or truncated. Reply with "
    "ONLY a single valid JSON object. No markdown code fences, no explanations."
)

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def strip_think(text: str) -> str:
    """Remove <think>/<thinking>/<thought> reasoning blocks models emit.

    Gemma-family models frequently prepend chain-of-thought wrapped in such
    tags before the actual JSON payload; leaving them in wastes parser effort
    and leaks reasoning into logs.
    """
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _THOUGHT_RE.sub("", cleaned)
    return cleaned.strip()


def decode_structured(text: str, output_type: Type) -> Any:
    """Parse JSON from (possibly fenced / think-wrapped) text into a model."""
    if not text:
        raise ValueError("empty model response")

    cleaned = strip_think(text)
    cleaned = _FENCE_OPEN_RE.sub("", cleaned.strip())
    cleaned = _FENCE_CLOSE_RE.sub("", cleaned).strip()
    if not cleaned:
        raise ValueError("empty after fence stripping")

    last_err = None
    try:
        return output_type.model_validate(json.loads(cleaned))
    except Exception as exc:
        last_err = exc

    for obj in _extract_json_objects(cleaned):
        try:
            return output_type.model_validate(json.loads(obj))
        except Exception as exc:
            last_err = exc

    raise ValueError(
        f"no valid JSON for {getattr(output_type, '__name__', '?')}: {str(last_err)[:200]}"
    )


def _extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start : i + 1])
                start = -1
    objects.sort(key=len, reverse=True)
    return objects


# ---------------------------------------------------------------------------
# Quote grounding (verbatim-quote validation for extracted evidence)
# ---------------------------------------------------------------------------

_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u200b": "",
}


def normalize_for_match(text: str) -> str:
    """Canonical form for substring matching: NFC, punctuation folding,
    whitespace collapse, casefold."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return " ".join(text.split()).casefold()


def _squash(text: str) -> str:
    """Whitespace-free form (last-resort exact comparison)."""
    return "".join(normalize_for_match(text).split())


def plural(n: int, word: str) -> str:
    """Natural pluralization for trace narration: 1 item, 2 items."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def quote_grounded(quote: str, document_text: str, fuzzy_threshold: float = 0.90) -> Tuple[bool, str]:
    """Check a model-quoted ``quote`` is actually present in ``document_text``.

    Tolerant ladder (LLM quotes drift in whitespace/unicode/table spacing):
      1. normalized substring
      2. whitespace-squashed substring   (handles table reflow)
      3. fuzzy containment               (best common block covers >= threshold)

    Returns ``(grounded, method)`` where method is one of
    ``exact|squashed|fuzzy|none``.
    """
    q_norm = normalize_for_match(quote or "")
    d_norm = normalize_for_match(document_text or "")
    if not q_norm or not d_norm:
        return False, "none"
    if q_norm in d_norm:
        return True, "exact"
    # Edge-punctuation tolerant: quotes often start/end mid-sentence
    # ("...poor prognosis." in the source is "...poor prognosis, including").
    q_edge = q_norm.strip(" .,;:!?'\"()-")
    if len(q_edge) >= 12 and q_edge in d_norm:
        return True, "exact"
    q_squash, d_squash = _squash(quote), _squash(document_text)
    if q_squash and q_squash in d_squash:
        return True, "squashed"
    q_edge_sq = q_squash.strip(".,;:!?\"'()-")
    if len(q_edge_sq) >= 12 and q_edge_sq in d_squash:
        return True, "squashed"
    # Fuzzy containment: longest common block between the squashed forms must
    # cover most of the quote. Bounded work: both strings are short-ish and
    # SequenceMatcher is C-implemented.
    matcher = SequenceMatcher(None, q_squash[:2000], d_squash, autojunk=False)
    match = matcher.find_longest_match(0, len(q_squash[:2000]), 0, len(d_squash))
    if match.size >= max(20, int(len(q_squash[:2000]) * fuzzy_threshold)):
        return True, "fuzzy"
    return False, "none"


__all__ = [
    "COMPACTION_NUDGE",
    "decode_structured",
    "normalize_for_match",
    "quote_grounded",
    "strip_think",
]
