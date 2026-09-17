"""The document pipeline: one Markdown article -> (chunks, units, report).

Composes the per-concern modules for one document. Encoding is a separate
step (python -m src.embedding) that reads the stored chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.lib.models import Chunk
from src.chunking.chunks import ChunkConfig, ChunkFactory
from src.chunking.ids import DocumentState, digest_id
from src.chunking.models import UnitRecord
from src.chunking.parsing import (
    _attach_figure_context, _attach_table_context, _document_metadata,
    _extract_fences, _parse_front_matter, build_section_tree,
)
from src.chunking.prose import split_overlap_budget
from src.chunking.sections import drop_doc_title_root, section_chunks
from src.chunking.tokens import _DEFAULT_SPLIT_OVERLAP_SENTENCES


def _make_config(*, max_tokens: int, hard_max_tokens: int,
                 split_overlap_sentences: int,
                 split_overlap_tokens: Optional[int]) -> ChunkConfig:
    """Validate the public knobs once and normalize into a ChunkConfig."""
    if max_tokens <= 0 or hard_max_tokens < max_tokens:
        raise ValueError("need 0 < max_tokens <= hard_max_tokens")
    return ChunkConfig(
        max_tokens=int(max_tokens),
        hard_max_tokens=int(hard_max_tokens),
        split_overlap_sentences=max(0, int(split_overlap_sentences)),
        split_overlap_effective=split_overlap_budget(int(max_tokens), split_overlap_tokens),
    )


def chunk_document(text: str, doc_id: Optional[str] = None, *,
                   max_tokens: int = 320, hard_max_tokens: int = 640,
                   split_overlap_sentences: int = _DEFAULT_SPLIT_OVERLAP_SENTENCES,
                   split_overlap_tokens: Optional[int] = None
                   ) -> Tuple[List[Chunk], List[UnitRecord], Dict[str, Any]]:
    """Chunk one Markdown article into (chunks, units, report).

    doc_id defaults to a content digest. The pipeline: parse the Markdown into
    a heading tree, recurse into every section and chunk each block, then
    resolve bracketed citation numbers against the parsed reference list.
    """
    config = _make_config(
        max_tokens=max_tokens, hard_max_tokens=hard_max_tokens,
        split_overlap_sentences=split_overlap_sentences,
        split_overlap_tokens=split_overlap_tokens,
    )
    doc_id = doc_id or digest_id(text)
    state = DocumentState(doc_id)
    front, body = _parse_front_matter(text)
    state.meta = _document_metadata(front, doc_id)

    body_clean, fences = _extract_fences(body)
    roots = drop_doc_title_root(build_section_tree(body_clean, fences))
    factory = ChunkFactory(state)

    chunks: List[Chunk] = []
    units: List[UnitRecord] = []
    seen_blocks = 0

    def process(node, breadcrumb: List[str],
                parent_unit_id: Optional[str] = None) -> None:
        nonlocal seen_blocks
        # breadcrumb = the path INCLUDING this heading (v1 semantics:
        # section() <- breadcrumb[0], subsection() <- breadcrumb[-1])
        path = breadcrumb + [node.title]
        _attach_table_context(node.blocks)
        _attach_figure_context(node.blocks)
        seen_blocks += len(node.blocks)
        unit = section_chunks(node, path, parent_unit_id, state, factory,
                              config, chunks, units)
        if unit:
            units.append(unit)
        unit_id = unit.unit_id if unit else parent_unit_id
        # every subsection hangs off its enclosing section unit;
        # paragraph/table units hang off the section unit too
        for child in node.children:
            process(child, path, unit_id)

    for root in roots:
        process(root, [])

    # Resolve bracketed citation numbers -> reference ids now that the whole
    # document (including its reference list) has been chunked. Numbers
    # outside the parsed reference range are dropped (years, list numbering).
    ref_count = state.reference_counter
    for c in chunks:
        nums = c.metadata.get("citation_numbers")
        if not nums:
            continue
        valid = [n for n in nums if 1 <= n <= ref_count] if ref_count else []
        if valid:
            c.citation_refs = [f"ref_{n - 1}" for n in valid]
            c.metadata["citation_numbers"] = valid
        else:
            del c.metadata["citation_numbers"]

    return chunks, units, build_report(state, config, chunks, units, seen_blocks)


def build_report(state: DocumentState, config: ChunkConfig,
                 chunks: List[Chunk], units: List[UnitRecord],
                 source_blocks: int) -> Dict[str, Any]:
    """Per-document audit: counts, split stats, config, document metadata.
    Stored in medpat.documents.chunker_report."""
    by_type: Dict[str, int] = {}
    eligible = 0
    cited = 0
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
        if c.retrieval_eligible:
            eligible += 1
        if c.citation_refs:
            cited += 1
    return {
        "document_id": state.doc_id,
        "source_blocks": source_blocks,
        "chunks": len(chunks),
        "retrieval_eligible": eligible,
        "chunks_by_type": dict(sorted(by_type.items())),
        "units": len(units),
        "paragraph_units": sum(1 for u in units if u.kind == "paragraph"),
        "paragraphs_sentence_split": state.split_paragraphs,
        "split_pieces": state.split_pieces,
        "chunks_with_citations": cited,
        "chunker_config": {
            "prose_strategy": "paragraph",
            "max_tokens": config.max_tokens,
            "hard_max_tokens": config.hard_max_tokens,
            "split_overlap_sentences": config.split_overlap_sentences,
            "split_overlap_tokens": config.split_overlap_effective,
            "parent_id_semantics": "enclosing_unit",
        },
        "meta": dict(state.meta),
    }
