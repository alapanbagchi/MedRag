"""Deterministic citation guarantee for synthesized answers.

The synthesizer is prompted to cite a [Pn] marker on every claim, but a
model miss can never ship: this module enforces it mechanically.

* :func:`enforce_citations` appends the turn's verified refs to any answer
  block that carries no marker (headings, tables, and the References
  section are left alone), then
* :func:`build_references` rebuilds the ``## References`` section from the
  ledger for exactly the refs actually cited in the final text — deduped,
  titled, url-linked — replacing whatever the model wrote.

The input/output contract is plain text with GFM blocks separated by
blank lines (what the synthesizer emits).
"""

from __future__ import annotations

import re
from typing import Any

_MARKER = re.compile(r"\[(?:[Pp])\d[\w-]*\]")
_BLOCK_SEP = re.compile(r"(\n\s*\n)")
_HEADING = re.compile(r"^\s*#{1,6}\s")
_TABLE_OR_FENCE = re.compile(r"^\s*\|.*\|\s*$|^\s*```")
_REF_HEADING = re.compile(r"^\s*#{1,6}\s*references\s*$")


def cited_refs(text: str) -> list[str]:
    """Refs cited in the answer, first-appearance order, normalized (P1, …)."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _MARKER.finditer(text):
        ref = m.group(0)[1:-1]
        ref = f"P{ref[1:]}" if ref[:1].lower() == "p" else ref
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def _ref_num(ref: str) -> int:
    digits = ref[1:] if isinstance(ref, str) and ref[:1].lower() == "p" else ""
    return int(digits) if digits.isdigit() else 0


def sort_refs(refs: list[str]) -> list[str]:
    return sorted(refs, key=_ref_num)


def _choose_attach_refs(text: str, verified: list[str]) -> list[str]:
    """Refs to attach to a marker-less block: the model's own cited set
    first (it chose those for nearby claims), then remaining verified
    refs, capped — never empty when verified refs exist."""
    pool = [r for r in verified if r] or []
    if not pool:
        return []
    have = set(pool)
    cited = [r for r in cited_refs(text) if r in have]
    rest = sort_refs([r for r in pool if r not in cited])
    return (cited + rest)[:5]


def _guard_block(block: str, attach: list[str]) -> str:
    """Append [Pn] markers to a block that has none and is not structure.

    Score (heading only), tables and code fences are left alone — they are
    not claim-bearing prose; anything else without a marker gets the
    markers appended to its last line, so the citation always renders in
    the visible text.
    """
    first = block.lstrip()
    if _HEADING.match(first) or _TABLE_OR_FENCE.match(first):
        return block
    if _MARKER.search(block):
        return block
    if not attach:
        return block
    suffix = "".join(f"[{r}]" for r in attach)
    newline = "\n" if block.endswith("\n") else ""
    return f"{block.rstrip()}{suffix}{newline}"


def build_references(records: list[dict[str, Any]]) -> str:
    """Render the References section from ledger records (deduped, in
    citation order). Each entry: ``[Pn] Title — url`` (url empty → the
    document id / ref alone)."""
    if not records:
        return ""
    lines = ["## References", ""]
    for r in records:
        label = (r.get("title") or "").strip() or (r.get("document_id") or "").strip() or r.get("ref", "")
        url = (r.get("url") or "").strip()
        ref = f'[{r.get("ref")}]'
        lines.append(f"{ref} {label} — {url}" if url else f"{ref} {label}")
    return "\n".join(lines)


def enforce_citations(text: str, records: list[dict[str, Any]]) -> str:
    """Guarantee every claim block carries a citation, then rebuild the
    References section for exactly the refs actually cited.

    ``records``: ledger evidence records as dicts with ref/title/url/
    document_id (verified, judge-kept passages). Returns the corrected
    full text.
    """
    verified = [str(r.get("ref") or "") for r in records if r.get("ref")]
    attach = _choose_attach_refs(text, verified)
    # Cut the model-written References section (heading may share a block
    # with its first entry — split at the heading line, not just at
    # blank lines) — it is rebuilt from the ledger below.
    ref_match = re.search(r"(?im)^\s*#{1,6}\s*references\s*$", text or "")
    if ref_match:
        text = text[: ref_match.start()]
    out: list[str] = []
    for block in re.split(_BLOCK_SEP, text or ""):
        if not block.strip():
            out.append(block)
            continue
        out.append(_guard_block(block, attach))
    head = "".join(out).rstrip()
    if not head:
        return (text or "").rstrip()
    by_ref = {str(r.get("ref")): r for r in records if r.get("ref")}
    used = [by_ref[r] for r in cited_refs(head) if r in by_ref]
    section = build_references(used)
    return f"{head}\n\n{section}" if section else head
