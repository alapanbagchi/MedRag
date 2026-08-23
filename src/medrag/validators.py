"""AST and chunk validators.

ASTValidator validates the parsed ``Document`` AST before chunking.
ChunkValidator validates the chunk list after chunking. Both produce
structured diagnostics rather than raising on every problem.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from medrag.models import (
    CANONICAL_BLOCK_TYPES,
    NORMALIZED_BLOCK_TYPES,
    VALID_CHUNK_TYPES,
    Block,
    Chunk,
    Document,
    Section,
    ValidationReport,
)


class ASTValidator:
    """Validate a parsed ``Document`` AST."""

    def __init__(self) -> None:
        self._section_source_ids: Dict[str, int] = {}

    def validate(self, document: Document) -> ValidationReport:
        report = ValidationReport()
        self._section_source_ids = {}

        if not document.pmcid:
            report.add(
                "ERROR",
                "MISSING_DOCUMENT_ID",
                "Document has no PMCID and no deterministic fallback.",
            )

        if not document.title:
            report.add(
                "WARNING",
                "MISSING_TITLE",
                "Document has no title.",
            )

        for section in document.sections:
            self._validate_section(section, report)

        return report

    # ==================================================================
    # SECTIONS
    # ==================================================================

    def _validate_section(
        self,
        section: Section,
        report: ValidationReport,
        parent_level: int = 0,
    ) -> None:
        # Invalid hierarchy: a child must be strictly deeper than its parent.
        if section.level <= parent_level:
            report.add(
                "ERROR",
                "INVALID_SECTION_HIERARCHY",
                "Section level is not deeper than its parent.",
                section=section.title,
                details={"level": section.level, "parent_level": parent_level},
            )

        if not section.title:
            report.add(
                "WARNING",
                "UNTITLED_SECTION",
                "Section has no title.",
                section=section.title,
            )

        source_id = section.source_id or section.metadata.get("source_id")
        if source_id:
            if source_id in self._section_source_ids:
                report.add(
                    "WARNING",
                    "DUPLICATE_SECTION_ID",
                    "Duplicate section source id.",
                    section=section.title,
                    details={"source_id": source_id},
                )
            self._section_source_ids[source_id] = (
                self._section_source_ids.get(source_id, 0) + 1
            )

        if not section.blocks and not section.children:
            report.add(
                "WARNING",
                "EMPTY_SECTION",
                "Section contains no blocks and no children.",
                section=section.title,
            )

        for block in section.blocks:
            self._validate_block(block, section, report)

        for child in section.children:
            self._validate_section(child, report, parent_level=section.level)

    # ==================================================================
    # BLOCKS
    # ==================================================================

    def _validate_block(
        self,
        block: Block,
        section: Section,
        report: ValidationReport,
    ) -> None:
        block_type = block.block_type

        if block_type not in NORMALIZED_BLOCK_TYPES and block_type not in CANONICAL_BLOCK_TYPES:
            report.add(
                "WARNING",
                "UNKNOWN_BLOCK_TYPE",
                "Block type is not a known canonical/normalized type.",
                block_id=self._block_id(block),
                section=section.title,
                details={"block_type": block_type},
            )

        metadata = block.metadata or {}

        # A list label attached to a paragraph block is an orphan label
        # (the labelled-list recognition should have consumed it). Figures
        # and supplementary blocks legitimately carry a "label" key.
        if metadata.get("label") and block_type == "paragraph":
            report.add(
                "WARNING",
                "ORPHAN_LIST_LABEL",
                "List label attached to a paragraph block.",
                block_id=self._block_id(block),
                section=section.title,
            )

        if block_type == "paragraph":
            self._validate_text_block(block, section, report)

        elif block_type == "list":
            self._validate_text_block(block, section, report)
            if not (block.content or "").strip():
                report.add(
                    "WARNING",
                    "EMPTY_LIST",
                    "List block has no items.",
                    block_id=self._block_id(block),
                    section=section.title,
                )

        elif block_type == "table":
            self._validate_table(block, section, report)

        elif block_type == "figure":
            self._validate_figure(block, section, report)

        elif block_type in ("formula", "equation"):
            if not (block.content or "").strip():
                report.add(
                    "WARNING",
                    "EMPTY_EQUATION",
                    "Equation block has no content.",
                    block_id=self._block_id(block),
                    section=section.title,
                )

        elif block_type == "reference":
            if not metadata.get("reference_id") and not self._block_id(block):
                report.add(
                    "WARNING",
                    "REFERENCE_NO_ID",
                    "Reference block has no identifier.",
                    section=section.title,
                )
            # A reference block inside a content section is suspicious.
            if section.classification not in ("references", "administrative"):
                report.add(
                    "WARNING",
                    "REFERENCE_IN_CONTENT_SECTION",
                    "Reference block found outside a references section.",
                    block_id=self._block_id(block),
                    section=section.title,
                )

    def _validate_text_block(
        self,
        block: Block,
        section: Section,
        report: ValidationReport,
    ) -> None:
        if not (block.content or "").strip():
            report.add(
                "WARNING",
                "EMPTY_SEMANTIC_BLOCK",
                "Semantic block has empty content.",
                block_id=self._block_id(block),
                section=section.title,
                details={"block_type": block.block_type},
            )

    def _validate_table(
        self,
        block: Block,
        section: Section,
        report: ValidationReport,
    ) -> None:
        metadata = block.metadata or {}
        table = metadata.get("table")
        if not isinstance(table, dict):
            report.add(
                "ERROR",
                "TABLE_MALFORMED",
                "Table block is missing structured table metadata.",
                block_id=self._block_id(block),
                section=section.title,
            )
            return

        table_id = table.get("table_id") or self._block_id(block)
        if not table_id:
            report.add(
                "WARNING",
                "TABLE_NO_ID",
                "Table has no source identity; a document-local fallback will be used.",
                section=section.title,
            )

        # Raw HTML / XML markup must never be the table representation.
        raw = (table.get("caption", "") or "") + " " + str(table.get("rows", ""))
        if self._has_raw_markup(block.content) or self._has_raw_markup(raw):
            report.add(
                "ERROR",
                "TABLE_RAW_HTML",
                "Table content contains raw HTML/XML markup.",
                block_id=self._block_id(block),
                section=section.title,
            )

        rows = table.get("rows", [])
        if isinstance(rows, list) and not rows and not table.get("structure_degraded"):
            report.add(
                "ERROR",
                "TABLE_NO_ROWS",
                "Structured table exists but has zero data rows.",
                block_id=self._block_id(block),
                section=section.title,
                details={"table_id": table_id},
            )

        if not table.get("caption"):
            report.add(
                "WARNING",
                "TABLE_NO_CAPTION",
                "Table has no caption.",
                block_id=self._block_id(block),
                section=section.title,
            )

        columns = table.get("columns", [])
        if not columns:
            report.add(
                "WARNING",
                "TABLE_NO_COLUMNS",
                "Table has no explicit column names.",
                block_id=self._block_id(block),
                section=section.title,
            )

    def _validate_figure(
        self,
        block: Block,
        section: Section,
        report: ValidationReport,
    ) -> None:
        metadata = block.metadata or {}
        if not metadata.get("figure_id") and not self._block_id(block):
            report.add(
                "ERROR",
                "FIGURE_NO_ID",
                "Figure has no identity.",
                section=section.title,
            )

        if not metadata.get("caption"):
            report.add(
                "WARNING",
                "FIGURE_NO_CAPTION",
                "Figure has no caption.",
                block_id=self._block_id(block),
                section=section.title,
            )

        if not metadata.get("image_ref"):
            report.add(
                "WARNING",
                "FIGURE_NO_IMAGE",
                "Figure has no image reference.",
                block_id=self._block_id(block),
                section=section.title,
            )

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _block_id(self, block: Block) -> Optional[str]:
        metadata = block.metadata or {}
        value = (
            metadata.get("block_id")
            or metadata.get("id")
            or metadata.get("xml_id")
        )
        return str(value) if value is not None else None

    @staticmethod
    def _has_raw_markup(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        markers = ("<table", "<tr", "<td", "<th", "</tbody", "</table>", "<thead")
        return any(marker in lowered for marker in markers)



class ChunkValidator:
    """Validate a list of ``Chunk`` objects."""

    RAW_MARKUP_MARKERS = (
        "<table", "<tr", "<td", "<th", "</tbody", "</table>", "<thead",
    )

    def validate(
        self,
        chunks: List[Chunk],
        document_id: Optional[str] = None,
    ) -> ValidationReport:
        report = ValidationReport()

        if not chunks:
            report.add("ERROR", "NO_CHUNKS", "Chunker produced no chunks.")
            return report

        id_to_chunk: Dict[str, Chunk] = {}
        seen_ids: Dict[str, int] = {}
        seen_texts: Dict[str, str] = {}

        for chunk in chunks:
            # ----------------------------------------------------------
            # Unique IDs
            # ----------------------------------------------------------
            if chunk.id in seen_ids:
                report.add(
                    "ERROR",
                    "DUPLICATE_CHUNK_ID",
                    "Duplicate chunk id.",
                    chunk_id=chunk.id,
                )
            seen_ids[chunk.id] = seen_ids.get(chunk.id, 0) + 1
            id_to_chunk[chunk.id] = chunk

            # ----------------------------------------------------------
            # Document IDs
            # ----------------------------------------------------------
            if not chunk.document_id:
                report.add(
                    "ERROR",
                    "MISSING_DOCUMENT_ID",
                    "Chunk has no document id.",
                    chunk_id=chunk.id,
                )
            elif document_id and chunk.document_id != document_id:
                report.add(
                    "ERROR",
                    "DOCUMENT_ID_MISMATCH",
                    "Chunk document id does not match the source document.",
                    chunk_id=chunk.id,
                    details={
                        "expected": document_id,
                        "actual": chunk.document_id,
                    },
                )

            # ----------------------------------------------------------
            # Chunk type
            # ----------------------------------------------------------
            if chunk.chunk_type not in VALID_CHUNK_TYPES:
                report.add(
                    "ERROR",
                    "INVALID_CHUNK_TYPE",
                    "Chunk has an invalid chunk_type.",
                    chunk_id=chunk.id,
                    details={"chunk_type": chunk.chunk_type},
                )

            # ----------------------------------------------------------
            # Text / embedding text
            # ----------------------------------------------------------
            if chunk.retrieval_eligible and not (chunk.text or "").strip():
                report.add(
                    "ERROR",
                    "EMPTY_RETRIEVAL_TEXT",
                    "Retrieval-eligible chunk has empty text.",
                    chunk_id=chunk.id,
                )

            if chunk.retrieval_eligible and not (chunk.embedding_text or "").strip():
                report.add(
                    "ERROR",
                    "EMPTY_EMBEDDING_TEXT",
                    "Retrieval-eligible chunk has empty embedding_text.",
                    chunk_id=chunk.id,
                )

            # ----------------------------------------------------------
            # Retrieval eligibility policy
            # ----------------------------------------------------------
            if chunk.chunk_type == "reference" and chunk.retrieval_eligible:
                report.add(
                    "ERROR",
                    "REFERENCE_RETRIEVAL_ELIGIBLE",
                    "Reference chunk is marked retrieval-eligible.",
                    chunk_id=chunk.id,
                )

            if chunk.chunk_type == "administrative" and chunk.retrieval_eligible:
                report.add(
                    "ERROR",
                    "ADMIN_RETRIEVAL_ELIGIBLE",
                    "Administrative chunk is marked retrieval-eligible.",
                    chunk_id=chunk.id,
                )

            # ----------------------------------------------------------
            # Table invariants
            # ----------------------------------------------------------
            if chunk.chunk_type == "table_row":
                if not chunk.table_id:
                    report.add(
                        "ERROR",
                        "TABLE_ROW_NO_TABLE_ID",
                        "table_row chunk has no table_id.",
                        chunk_id=chunk.id,
                    )
                if not chunk.parent_id:
                    report.add(
                        "ERROR",
                        "TABLE_ROW_NO_PARENT",
                        "table_row chunk has no parent_id.",
                        chunk_id=chunk.id,
                    )
                if chunk.parent_id and chunk.parent_id not in id_to_chunk:
                    report.add(
                        "ERROR",
                        "PARENT_NOT_FOUND",
                        "table_row parent does not exist.",
                        chunk_id=chunk.id,
                        details={"parent_id": chunk.parent_id},
                    )

            if chunk.chunk_type == "table_footnotes":
                if not chunk.table_id:
                    report.add(
                        "ERROR",
                        "TABLE_FOOTNOTES_NO_TABLE_ID",
                        "table_footnotes chunk has no table_id.",
                        chunk_id=chunk.id,
                    )
                if chunk.parent_id and chunk.parent_id not in id_to_chunk:
                    report.add(
                        "ERROR",
                        "PARENT_NOT_FOUND",
                        "table_footnotes parent does not exist.",
                        chunk_id=chunk.id,
                        details={"parent_id": chunk.parent_id},
                    )

            # ----------------------------------------------------------
            # Figure invariant
            # ----------------------------------------------------------
            if chunk.chunk_type == "figure" and not chunk.figure_id:
                report.add(
                    "ERROR",
                    "FIGURE_NO_ID",
                    "figure chunk has no figure_id.",
                    chunk_id=chunk.id,
                )

            # ----------------------------------------------------------
            # Raw HTML / XML leakage in retrieval representations
            # ----------------------------------------------------------
            if chunk.chunk_type.startswith("table"):
                if self._has_raw_markup(chunk.text) or self._has_raw_markup(
                    chunk.embedding_text
                ):
                    report.add(
                        "ERROR",
                        "TABLE_RAW_HTML",
                        "Table chunk contains raw HTML/XML markup.",
                        chunk_id=chunk.id,
                    )

            if chunk.chunk_type in ("paragraph", "list") and self._has_raw_markup(
                chunk.text
            ):
                report.add(
                    "WARNING",
                    "HTML_XML_LEAKAGE",
                    "Prose chunk contains raw HTML/XML markup.",
                    chunk_id=chunk.id,
                )

            # ----------------------------------------------------------
            # Source provenance
            # ----------------------------------------------------------
            if not chunk.source_block_ids:
                report.add(
                    "WARNING",
                    "MISSING_SOURCE_BLOCK_ID",
                    "Chunk has no source block provenance.",
                    chunk_id=chunk.id,
                )

            # ----------------------------------------------------------
            # Duplicate content detection
            # ----------------------------------------------------------
            key = chunk.text.strip()
            if key and key in seen_texts:
                report.add(
                    "WARNING",
                    "DUPLICATE_CONTENT",
                    "Chunk text is identical to another chunk.",
                    chunk_id=chunk.id,
                    details={"duplicate_of": seen_texts[key]},
                )
            else:
                seen_texts[key] = chunk.id

        # --------------------------------------------------------------
        # Ordering invariant
        # --------------------------------------------------------------
        positions = [c.document_position for c in chunks]
        if positions != sorted(positions):
            report.add(
                "ERROR",
                "ORDERING_VIOLATION",
                "document_position is not monotonic.",
            )
        if len(set(positions)) != len(positions):
            report.add(
                "ERROR",
                "DUPLICATE_POSITION",
                "document_position values are not unique.",
            )

        return report

    @staticmethod
    def _has_raw_markup(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in ChunkValidator.RAW_MARKUP_MARKERS)

