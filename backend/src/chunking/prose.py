"""Prose chunking: sentence-split with a healthy carry, hard cap, emit pieces.

The only custom chunking logic: when a paragraph exceeds the budget it is
split at sentence boundaries and the last N sentences of the flushed piece
seed the next piece (the "overlap", applied ONLY inside a split paragraph -
boundaries between distinct paragraphs are never overlapped).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.chunking.ids import DocumentState
from src.chunking.models import UnitRecord
from src.chunking.parsing import RawBlock
from src.chunking.tokens import _SPLIT_OVERLAP_FRACTION, _estimate_tokens, _split_sentences, _truncate_tokens


def split_long_paragraph_pieces(
    text: str,
    max_tokens: int,
    hard_max_tokens: int,
    split_overlap_sentences: int,
    overlap_budget: int,
) -> List[str]:
    """Split one paragraph into <= max_tokens pieces at sentence boundaries.

    The last split_overlap_sentences sentence(s) of the flushed piece seed
    the next piece, so the artificial mid-paragraph boundary never orphans
    local context. The carry is token-budgeted (overlap_budget) and shrinks
    oldest-first. This is the ONE place overlap is applied.
    """
    sentences = _split_sentences(text)
    pieces: List[str] = []
    cur_text: List[str] = []
    cur_tokens = 0
    for sent in sentences:
        t = _estimate_tokens(sent)
        if cur_text and cur_tokens + t > max_tokens:
            pieces.append(" ".join(cur_text))
            if split_overlap_sentences > 0:
                carry = list(cur_text[-split_overlap_sentences:])
                while carry and _estimate_tokens(" ".join(carry)) > overlap_budget:
                    carry = carry[1:]
                cur_text = carry
                cur_tokens = _estimate_tokens(" ".join(cur_text)) if cur_text else 0
            else:
                cur_text = []
                cur_tokens = 0
        cur_text.append(sent)
        cur_tokens += t
    if cur_text:
        pieces.append(" ".join(cur_text))
    return [hard_cap_piece_text(p, hard_max_tokens) for p in pieces]


def hard_cap_piece_text(text: str, hard_max_tokens: int) -> str:
    """Clamp a pathological piece to hard_max_tokens TOKENS (never words)."""
    if _estimate_tokens(text) <= hard_max_tokens:
        return text
    marker = " [...]"
    limit = max(1, hard_max_tokens - _estimate_tokens(marker))
    kept = _truncate_tokens(text, limit)
    return (kept + marker).strip()


def split_overlap_budget(max_tokens: int,
                         split_overlap_tokens: Optional[int] = None) -> int:
    """Token budget for the intra-paragraph sentence carry: default ~25% of
    max_tokens, capped at half (so the carry never dominates a piece)."""
    if split_overlap_tokens is not None and split_overlap_tokens <= 0:
        split_overlap_tokens = None
    return min(
        split_overlap_tokens or max(1, int(max_tokens * _SPLIT_OVERLAP_FRACTION)),
        max(1, max_tokens // 2),
    )


def prose_chunks(paragraphs: List[RawBlock], path: List[str],
                 section_unit_id: str, state: DocumentState, factory, *,
                 max_tokens: int, hard_max_tokens: int,
                 split_overlap_sentences: int,
                 split_overlap_effective: int) -> Tuple[List["Chunk"], List[UnitRecord]]:
    """Chunk one section's prose. Returns (chunks, paragraph_units).

    Paragraph-first: a paragraph that fits the budget is one chunk; an
    oversized paragraph is sentence-split (with the healthy carry) into one
    chunk per piece. Split paragraphs additionally emit a PARAGRAPH unit (full
    source text + child piece ids) and tag every piece with
    metadata.sentence_split - so an encoder can embed the full paragraph and
    slice the piece spans (late-chunking hook, no duplicated vectors).
    """
    from src.lib.models import Chunk

    if not paragraphs:
        return [], []
    # (piece, split_meta_or_None); split_meta carries sentence_split info.
    pieces: List[Tuple[RawBlock, Optional[Dict[str, Any]]]] = []
    full_texts: Dict[str, str] = {}   # paragraph_unit_id -> full source text
    for p in paragraphs:
        if _estimate_tokens(p.text) > max_tokens:
            sp = split_long_paragraph(p, max_tokens, hard_max_tokens,
                                      split_overlap_sentences,
                                      split_overlap_effective)
            if len(sp) > 1:
                pu_id = state.unique(f"{state.doc_id}_unit_{state.unit_counter}")
                state.unit_counter += 1
                full_texts[pu_id] = p.text
                state.split_paragraphs += 1
                state.split_pieces += len(sp)
                for i, piece in enumerate(sp):
                    pieces.append((piece, {
                        "sentence_split": {
                            "piece_index": i,
                            "piece_count": len(sp),
                            "paragraph_unit_id": pu_id,
                        },
                    }))
            else:
                pieces.append((sp[0] if sp else p, None))
        else:
            pieces.append((p, None))

    chunks = [factory.paragraph(piece.text, path, section_unit_id) for piece, _ in pieces]
    # carry the split metadata onto the emitted chunks
    for chunk, (_, meta) in zip(chunks, pieces):
        if meta:
            chunk.metadata.update(meta)

    # Assemble paragraph units for split paragraphs (deterministic order).
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for chunk, (_, meta) in zip(chunks, pieces):
        ss = (meta or {}).get("sentence_split")
        if ss:
            grouped.setdefault(ss["paragraph_unit_id"], []).append(
                (ss["piece_index"], chunk.id))
    p_units: List[UnitRecord] = []
    for pid in sorted(grouped):
        ordered = [cid for _, cid in sorted(grouped[pid])]
        p_units.append(UnitRecord(
            pid, state.doc_id, "paragraph",
            path[-1] if path else "", path,
            full_texts.get(pid, ""), ordered,
            parent_unit_id=section_unit_id,  # the enclosing section unit
        ))
    return chunks, p_units


def split_long_paragraph(paragraph: RawBlock, max_tokens: int,
                         hard_max_tokens: int, split_overlap_sentences: int,
                         overlap_budget: int) -> List[RawBlock]:
    """Sentence-split one oversized paragraph into RawBlock pieces."""
    pieces = split_long_paragraph_pieces(
        paragraph.text, max_tokens, hard_max_tokens,
        split_overlap_sentences, overlap_budget,
    )
    return [RawBlock(kind="paragraph", text=t) for t in pieces] or [paragraph]
