"""Property-based tests for chunking invariants.

These generate synthetic ASTs (deterministic, pseudo-random) and verify the
invariants from specification section 79.
"""

from __future__ import annotations

import random
from typing import List

from src.lib.models import Block, Document, Section
from src.chunking.chunker import ASTChunker


def random_section(rng: random.Random, depth: int = 0) -> Section:
    title = f"Section_{rng.randint(0, 1000)}"
    blocks: List[Block] = []
    for _ in range(rng.randint(0, 6)):
        choice = rng.randint(0, 7)
        if choice == 0:
            blocks.append(Block("paragraph", f"Sentence {rng.randint(0, 100)}. "
                                             f"More words {rng.randint(0, 100)}."))
        elif choice == 1:
            blocks.append(Block("list", "- a\n- b"))
        elif choice == 2:
            blocks.append(Block("table", "t", {"table": {
                "table_id": f"tbl{rng.randint(0, 1000)}",
                "label": "Table 1", "caption": "Cap.",
                "columns": [{"index": 0, "name": "c"}],
                "categories": [],
                "rows": [{"row_label": "r", "group_path": [], "values": {"c": "1"},
                          "source_block_ids": ["x"]}],
                "footnotes": [],
                "structure_degraded": False,
            }}))
        elif choice == 3:
            blocks.append(Block("figure", "Figure 1", {"figure_id": f"fig{rng.randint(0, 1000)}"}))
        elif choice == 4:
            blocks.append(Block("formula", "x^2", {"equation_id": f"eq{rng.randint(0, 1000)}"}))
        elif choice == 5:
            blocks.append(Block("reference", "Ref text", {"reference_id": f"bib{rng.randint(0, 1000)}"}))
        elif choice == 6:
            blocks.append(Block("paragraph", "admin text"))
        else:
            blocks.append(Block("paragraph", "plain paragraph"))

    children = []
    if depth < 2:
        for _ in range(rng.randint(0, 2)):
            children.append(random_section(rng, depth + 1))

    section_type = "content"
    if "fund" in title.lower() or "acknowledg" in title.lower():
        section_type = "administrative"
    return Section(title, depth + 1, [title], blocks=blocks, children=children,
                   section_type=section_type, metadata={"section_type": section_type})


def random_document(seed: int) -> Document:
    rng = random.Random(seed)
    return Document(
        pmcid=f"PMC{seed}",
        title="Random Article",
        sections=[random_section(rng) for _ in range(rng.randint(1, 4))],
    )


def collect_blocks(document: Document) -> List[Block]:
    out = []

    def walk(sections):
        for s in sections:
            out.extend(s.blocks)
            walk(s.children)

    walk(document.sections)
    return out


def test_never_emits_duplicate_ids():
    for seed in range(50):
        chunks = ASTChunker().chunk(random_document(seed))
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), f"duplicate ids for seed {seed}"


def test_table_rows_always_point_to_existing_summary():
    for seed in range(50):
        chunks = ASTChunker().chunk(random_document(seed))
        id_to_chunk = {c.id: c for c in chunks}
        for c in chunks:
            if c.chunk_type == "table_row":
                assert c.parent_id in id_to_chunk, f"seed {seed}"
                assert c.table_id


def test_never_loses_blocks():
    for seed in range(50):
        document = random_document(seed)
        chunks = ASTChunker().chunk(document)
        chunker = ASTChunker()
        chunker.chunk(document)
        report = chunker._report
        assert report["source_blocks_unhandled"] == 0, f"seed {seed}"


def test_same_input_gives_same_output():
    for seed in range(20):
        document = random_document(seed)
        c1 = ASTChunker().chunk(document)
        c2 = ASTChunker().chunk(document)
        assert [x.id for x in c1] == [x.id for x in c2]
        assert [x.text for x in c1] == [x.text for x in c2]


def test_order_is_stable():
    for seed in range(20):
        document = random_document(seed)
        chunks = ASTChunker().chunk(document)
        positions = [c.document_position for c in chunks]
        assert positions == sorted(positions)
        assert len(positions) == len(set(positions))


def test_structured_objects_are_hard_boundaries():
    document = Document(pmcid="PMC1", title="T", sections=[
        Section("S", 1, ["S"], blocks=[
            Block("paragraph", "A."),
            Block("table", "t", {"table": {
                "table_id": "t1", "label": "Table 1", "caption": "",
                "columns": [], "categories": [],
                "rows": [{"row_label": "r", "group_path": [], "values": {},
                          "source_block_ids": ["t1"]}],
                "footnotes": [], "structure_degraded": False,
            }}),
            Block("paragraph", "B."),
        ], section_type="content"),
    ])
    chunks = ASTChunker().chunk(document)
    types = [c.chunk_type for c in chunks]
    # the table is never packed into prose
    assert "table_summary" in types
    assert "A." not in next(c.text for c in chunks if c.chunk_type == "table_summary")
