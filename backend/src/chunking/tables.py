"""Table chunking: summary + per-row + footnotes chunks and the TABLE unit.

The section walk (sections.py) routes RawBlock(kind="table") here. Every
chunk points its parent_id at a first-class TABLE unit that carries the full
table text plus the ids of all its child chunks.
"""

from __future__ import annotations

from typing import List, Tuple

from src.chunking.chunks import ChunkFactory
from src.chunking.ids import DocumentState
from src.chunking.models import UnitRecord
from src.chunking.parsing import RawBlock


def clean_cell(text: str) -> str:
    """One display cell: whitespace collapsed onto a single line."""
    return " ".join((text or "").replace("\n", " ").split())


def table_id(block: RawBlock, state: DocumentState) -> Tuple[str, str]:
    """Return (doc_local_table_id, display_label)."""
    if block.label:
        return state.unique(DocumentState.slug(block.label)), block.label
    n = state.table_counter
    state.table_counter += 1
    return state.unique(f"table_{n}"), f"Table {n}"


def table_chunks(block: RawBlock, path: List[str], section_unit_id: str,
                 state: DocumentState, factory: ChunkFactory,
                 units: List[UnitRecord]) -> List["Chunk"]:
    """Chunk one table: summary + rows + footnotes, all parent-linked to a
    first-class TABLE unit. Returns the chunk list (unit appended to units)."""
    from src.lib.models import Chunk

    table_id_, display = table_id(block, state)
    # allocate the TABLE unit id up front so every chunk this table produces
    # (summary, rows, footnotes) can point at it via parent_id. One uniform
    # meaning for parent_id across all chunk types: "the unit that contains me".
    table_unit_id = state.unique(f"{state.doc_id}_unit_{state.unit_counter}")
    state.unit_counter += 1

    columns = list(block.header)
    if not columns and block.rows:
        columns = [f"column_{i}" for i in range(len(block.rows[0]))]

    summary_parts = []
    if block.label or block.caption:
        summary_parts.append(f"{block.label + ': ' if block.label else ''}{block.caption}".strip())
    if columns:
        summary_parts.append("Columns:\n" + "\n".join(f"- {c}" for c in columns))
    summary_text = "\n\n".join(p for p in summary_parts if p) or display

    summary_id = state.unique(f"{state.doc_id}_{table_id_}_summary")
    summary = factory.make(
        summary_id, summary_text, "table_summary", path,
        position=state.next_pos(), parent_id=table_unit_id,
        object_id=table_id_, table_id=table_id_,
        extra_metadata={"unit_id": section_unit_id, "table_id": table_id_,
                        "columns": columns},
    )
    out: List[Chunk] = [summary]

    # current spanning-subheader group (see row loop): carries over to every
    # following row's group_path
    current_group: List[str] = []
    # rowspan flattening: a row whose FIRST cell is blank continues the previous
    # row's first-column value ("| | Felodipine | C | ..." belongs to the
    # "Calcium channelblockers" category above). carried_label recovers that.
    carried_label = ""
    for i, row in enumerate(block.rows):
        cells = [clean_cell(c) for c in row]
        if not any(cells):
            continue  # fully empty row -> no chunk
        non_empty = [c for c in cells if c]
        if (len(cells) > 1 and len(non_empty) > 1
                and len(set(non_empty)) == 1
                and " " in non_empty[0]):
            # Spanning subheader row: the MD generator flattens a JATS colspan
            # header by repeating its label into every cell. That is group
            # context, not evidence — emitting it as a chunk would put the same
            # sentence in the index N times. The " " guard keeps all-identical
            # NUMERIC data rows (e.g. "100 100 100") from being misread.
            label = non_empty[0].rstrip(":").strip()
            current_group = [label] if label else []
            continue
        if len(non_empty) == 1 and non_empty[0].strip() and len(columns) > 1:
            # One populated cell (whatever the padding): a JATS colspan
            # subheader ("Sex", "Median (IQR)", panel labels...). Group context
            # - it qualifies the FOLLOWING rows via group_path, never becomes a
            # data row full of em-dashes.
            current_group = [non_empty[0].rstrip(":").strip()]
            continue
        if columns and all(
                (a or "").strip() == (b or "").strip()
                for a, b in zip(cells, columns)):
            # Repeated header row inside a multi-panel table: identical to the
            # column headings -> context, not evidence.
            current_group = [(cells[0] or "").rstrip(":").strip()]
            continue
        # Data row. A blank first cell means the row continues the previous
        # row's first-column value (JATS rowspan flattening), so its label is
        # carried forward - NEVER swallowed as group context.
        if cells[0].strip():
            row_label = cells[0]
            carried_label = row_label
        else:
            row_label = carried_label
            carried_label = carried_label
        text_bits: List[str] = []
        if columns:
            first_header = columns[0].strip() or "Row"
            text_bits.append(
                f"{first_header}: {row_label if row_label.strip() else '—'}"
            )
        elif row_label:
            text_bits.append(f"Row: {row_label}")
        for j in range(1, max(len(columns), len(cells))):
            header = (
                columns[j].strip()
                if j < len(columns) and columns[j].strip()
                else f"column_{j}"
            )
            val = cells[j] if j < len(cells) else ""
            # Empty cells are explicit so "not reported" is distinguishable
            # from a missing structural cell.
            text_bits.append(f"{header}: {val if val.strip() else '—'}")
        row_text = "\n".join(text_bits) or (row_label or "table row")
        rid = state.unique(f"{state.doc_id}_{table_id_}_row_{i}")
        out.append(factory.make(
            rid, row_text, "table_row", path,
            position=state.next_pos(), parent_id=table_unit_id,
            object_id=table_id_, table_id=table_id_,
            row_label=row_label or None,
            group_path=list(current_group),
            extra_metadata={"unit_id": section_unit_id, "table_id": table_id_,
                            "row_index": i},
        ))

    if block.footnotes:
        fn_text = "\n".join(
            f"[{m}] {t}" if m else t for m, t in block.footnotes if t
        )
        if fn_text:
            fid = state.unique(f"{state.doc_id}_{table_id_}_footnotes")
            out.append(factory.make(
                fid, fn_text, "table_footnotes", path,
                position=state.next_pos(), parent_id=table_unit_id,
                object_id=table_id_, table_id=table_id_,
                extra_metadata={"unit_id": section_unit_id, "table_id": table_id_},
            ))

    # A first-class TABLE unit (parent context for its child chunks). Every
    # table chunk's parent_id points here, and this unit carries the full
    # table text plus every child chunk id.
    unit_lines: List[str] = []
    head = f"{block.label + ': ' if block.label else ''}{block.caption}".strip()
    if head:
        unit_lines.append(head)
    if block.header:
        unit_lines.append(" | ".join(block.header))
    unit_lines.extend(" | ".join(row) for row in block.rows)
    unit_lines.extend(f"[{m}] {t}" if m else t for m, t in block.footnotes)
    units.append(UnitRecord(
        unit_id=table_unit_id,
        document_id=state.doc_id, kind="table", title=display,
        breadcrumb=path,
        text="\n".join(unit_lines),
        chunk_ids=[c.id for c in out],
        parent_unit_id=section_unit_id,  # the enclosing section unit
    ))
    return out
