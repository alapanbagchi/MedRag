"""Section processing: one heading node -> its chunks + the section unit.

The document walk (documents.py) recurses over the parsed heading tree and
calls section_chunks per node; this module owns the content->chunk dispatch
(references / administrative / content) and the section unit records.
"""

from __future__ import annotations

from typing import List, Optional

from src.lib.models import Chunk
from src.chunking.classification import classify_section_title
from src.chunking.chunks import ChunkConfig, ChunkFactory
from src.chunking.ids import DocumentState
from src.chunking.models import UnitRecord
from src.chunking.parsing import MDNode, RawBlock, node_plain
from src.chunking.prose import prose_chunks
from src.chunking.tables import table_chunks


def drop_doc_title_root(roots: List[MDNode]) -> List[MDNode]:
    """The generator emits # {title} as the only h1. That is the document
    title, not a section: drop it, promote its children, and preserve any
    direct blocks (an unheaded abstract / citation line) by prepending them
    to the first child."""
    if len(roots) == 1 and roots[0].level == 1:
        root = roots[0]
        if not root.children:
            return [root]
        if root.blocks:
            root.children[0].blocks = root.blocks + root.children[0].blocks
        return root.children
    return roots


def section_chunks(node: MDNode, path: List[str],
                   parent_unit_id: Optional[str], state: DocumentState,
                   factory: ChunkFactory, config: ChunkConfig,
                   chunks: List[Chunk],
                   units: List[UnitRecord]) -> Optional[UnitRecord]:
    """Chunk one node's direct blocks; return its section unit.

    parent_unit_id is the enclosing section unit (None for roots).
    """
    unit_id = state.unique(f"{state.doc_id}_unit_{state.unit_counter}")
    state.unit_counter += 1
    unit_children: List[str] = []
    kind = classify_section_title(node.title)

    if kind == "references":
        for b in node.blocks:
            if b.kind == "list":
                for item, raw in zip(b.items, b.raw_items):
                    c = factory.reference(item, path, raw=raw, unit_id=unit_id)
                    chunks.append(c)
                    unit_children.append(c.id)
            elif b.text.strip():
                c = factory.reference(b.text, path, raw=b.text, unit_id=unit_id)
                chunks.append(c)
                unit_children.append(c.id)
        unit_text = "\n".join(node_plain(node.blocks))
        if not unit_children:
            return None
        return UnitRecord(unit_id, state.doc_id, "section", node.title,
                          path, unit_text or "References", unit_children,
                          parent_unit_id=parent_unit_id)

    if kind == "administrative":
        for b in node.blocks:
            if b.kind == "table":
                sub = table_chunks(b, path, unit_id, state, factory, units)
                chunks.extend(sub)
                unit_children.extend(c.id for c in sub)
                continue
            if b.kind == "figure":
                c = factory.figure(b, path, unit_id)
                chunks.append(c)
                unit_children.append(c.id)
                continue
            if b.kind == "equation" and b.text.strip():
                c = factory.equation(b, path, unit_id)
                chunks.append(c)
                unit_children.append(c.id)
                continue
            text = b.text if b.kind in ("paragraph", "code") else None
            if text and text.strip():
                c = factory.administrative(text, path, unit_id)
                chunks.append(c)
                unit_children.append(c.id)
            elif b.kind == "list":
                text = "\n".join(f"- {it}" for it in b.items)
                if text:
                    c = factory.administrative(text, path, unit_id)
                    chunks.append(c)
                    unit_children.append(c.id)
        if not unit_children:
            return None
        return UnitRecord(unit_id, state.doc_id, "section", node.title,
                          path, "\n".join(node_plain(node.blocks)), unit_children,
                          parent_unit_id=parent_unit_id)

    # content
    prose: List[RawBlock] = []
    for b in node.blocks:
        if b.kind == "paragraph":
            if b.text.strip():
                prose.append(b)
            continue
        if b.kind == "list":
            c = factory.list_chunk(b, path, unit_id)
            if c:
                chunks.append(c)
                unit_children.append(c.id)
            continue
        if b.kind == "table":
            sub = table_chunks(b, path, unit_id, state, factory, units)
            chunks.extend(sub)
            unit_children.extend(c.id for c in sub)
            continue
        if b.kind == "figure":
            c = factory.figure(b, path, unit_id)
            chunks.append(c)
            unit_children.append(c.id)
            continue
        if b.kind == "equation":
            if b.text.strip():
                c = factory.equation(b, path, unit_id)
                chunks.append(c)
                unit_children.append(c.id)
            continue
        if b.kind == "code":
            if b.text.strip():
                c = factory.paragraph(b.text, path, unit_id,
                                      extra_metadata={"original_block_type": "code"})
                chunks.append(c)
                unit_children.append(c.id)
            continue
    pc, p_units = prose_chunks(
        prose, path, unit_id, state, factory,
        max_tokens=config.max_tokens,
        hard_max_tokens=config.hard_max_tokens,
        split_overlap_sentences=config.split_overlap_sentences,
        split_overlap_effective=config.split_overlap_effective,
    )
    chunks.extend(pc)
    unit_children.extend(c.id for c in pc)
    for pu in p_units:
        units.append(pu)
    unit_text = "\n".join(node_plain(node.blocks))
    return UnitRecord(unit_id, state.doc_id, "section", node.title, path,
                      unit_text, unit_children, parent_unit_id=parent_unit_id)
