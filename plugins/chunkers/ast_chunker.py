from __future__ import annotations

import re
from typing import Any, List, Optional

from core.models import Block, Chunk, Document, Section


class ASTChunker:
    """
    Structure-aware chunker for PMC articles.

    Strategy
    --------
    1. Document structure is the strongest boundary.
    2. Block/object type determines the chunking strategy.
    3. Prose is grouped only within the same structural section.
    4. Tables become:
           table_summary
           table_row*
           table_footnotes
    5. Figures are atomic semantic objects.
    6. Lists are kept together and may have a semantic label.
    7. Equations are independent structured objects.
    8. References are structured and excluded from the main vector index.
    9. Administrative material is preserved but excluded from the main
       semantic retrieval index.
    10. Size is a secondary constraint, primarily for prose.

    ID strategy
    -----------
    Prose/list/admin:
        PMC123_0
        PMC123_1
        PMC123_2

    Tables:
        PMC123_table_0_summary
        PMC123_table_0_row_0
        PMC123_table_0_row_1
        PMC123_table_0_footnotes

    Figures:
        PMC123_figure_0

    Equations:
        PMC123_equation_0

    References:
        PMC123_ref_1

    Parser-provided IDs are preserved whenever available. Missing table
    and figure IDs are replaced with document-local deterministic IDs.
    """

    CHUNK_VERSION = "1.0"

    def __init__(
        self,
        max_prose_chars: int = 1600,
        hard_max_prose_chars: int = 3200,
    ) -> None:
        if max_prose_chars <= 0:
            raise ValueError(
                "max_prose_chars must be greater than 0"
            )

        if hard_max_prose_chars < max_prose_chars:
            raise ValueError(
                "hard_max_prose_chars must be >= max_prose_chars"
            )

        self.max_prose_chars = max_prose_chars
        self.hard_max_prose_chars = hard_max_prose_chars

        self._prose_counter = 0
        self._position_counter = 0
        self._table_counter = 0
        self._figure_counter = 0
        self._equation_counter = 0
        self._reference_counter = 0

        self._used_ids: set[str] = set()

        # Coverage / reporting state (reset per document).
        self._report: dict[str, Any] = {}
        self._warnings: List[dict[str, Any]] = []
        self._table_coverage: dict[str, dict[str, Any]] = {}
        self._all_source_blocks: List[Block] = []
        self._block_to_index: dict[int, int] = {}
        self._consumed_source_blocks: set[int] = set()
        self._source_type_counts: dict[str, int] = {}

    # ==================================================================
    # PUBLIC
    # ==================================================================

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Convert a parsed Document AST into chunks.
        """
        self._reset_state()
        self._collect_source_blocks(document)

        chunks: List[Chunk] = []

        document_id = document.pmcid
        document_metadata = self._build_document_metadata(document)

        for section in document.sections:
            self._process_section(
                section=section,
                chunks=chunks,
                document_id=document_id,
                document_metadata=document_metadata,
                breadcrumb=self._initial_breadcrumb(section),
            )

        self._report = self._build_report(document, chunks)

        return chunks

    # ==================================================================
    # STATE / IDS
    # ==================================================================

    def _reset_state(self) -> None:
        self._prose_counter = 0
        self._position_counter = 0
        self._table_counter = 0
        self._figure_counter = 0
        self._equation_counter = 0
        self._reference_counter = 0
        self._used_ids = set()

        self._report = {}
        self._warnings = []
        self._table_coverage = {}
        self._all_source_blocks = []
        self._block_to_index = {}
        self._consumed_source_blocks = set()
        self._source_type_counts = {}

    def _warn(
        self,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        self._warnings.append({
            "code": code,
            "message": message,
            **details,
        })

    def _next_position(self) -> int:
        position = self._position_counter
        self._position_counter += 1
        return position

    def _make_unique_id(self, candidate: str) -> str:
        """
        Guarantee uniqueness even if malformed or duplicated parser IDs
        appear in a document.
        """
        if candidate not in self._used_ids:
            self._used_ids.add(candidate)
            return candidate

        suffix = 1
        while f"{candidate}_{suffix}" in self._used_ids:
            suffix += 1

        unique_id = f"{candidate}_{suffix}"
        self._used_ids.add(unique_id)

        return unique_id

    def _next_prose_id(
        self,
        document_id: str,
    ) -> tuple[str, int]:
        position = self._next_position()

        candidate = (
            f"{document_id}_{self._prose_counter}"
        )

        self._prose_counter += 1

        return (
            self._make_unique_id(candidate),
            position,
        )

    # ==================================================================
    # DOCUMENT METADATA
    # ==================================================================

    def _build_document_metadata(
        self,
        document: Document,
    ) -> dict[str, Any]:
        metadata = document.metadata or {}

        article_meta = metadata.get(
            "article_meta",
            {},
        )

        journal_meta = metadata.get(
            "journal_meta",
            {},
        )

        article_ids = article_meta.get(
            "article_ids",
            {},
        )

        publication_dates = article_meta.get(
            "publication_dates",
            {},
        )

        issue_info = article_meta.get(
            "issue_info",
            {},
        )

        authors = article_meta.get(
            "authors",
            [],
        )

        funding = article_meta.get(
            "funding",
            [],
        )

        author_names: List[str] = []

        for author in authors:
            if isinstance(author, dict):
                name = author.get("name")
                if name:
                    author_names.append(str(name))
            elif isinstance(author, str):
                author_names.append(author)

        funding_sources: List[str] = []

        for source in funding:
            if isinstance(source, dict):
                value = source.get("source")
                if value:
                    funding_sources.append(str(value))
            elif isinstance(source, str):
                funding_sources.append(source)

        first_page = issue_info.get("fpage", "")
        last_page = issue_info.get("lpage", "")

        pages = ""
        if first_page and last_page:
            pages = f"{first_page}-{last_page}"
        elif first_page:
            pages = str(first_page)

        return {
            "pmcid": article_ids.get(
                "pmcid",
                document.pmcid,
            ),
            "title": article_meta.get(
                "title",
                document.title,
            ),
            "journal": journal_meta.get(
                "title",
                "",
            ),
            "doi": article_ids.get(
                "doi",
                "",
            ),
            "publication_date": (
                publication_dates.get("epub")
                or publication_dates.get("collection")
                or ""
            ),
            "keywords": article_meta.get(
                "keywords",
                [],
            ),
            "categories": article_meta.get(
                "categories",
                [],
            ),
            "authors": author_names,
            "funding_sources": funding_sources,
            "volume": issue_info.get(
                "volume",
                "",
            ),
            "issue": issue_info.get(
                "issue",
                "",
            ),
            "pages": pages,
        }

    # ==================================================================
    # SECTION PROCESSING
    # ==================================================================

    def _process_section(
        self,
        section: Section,
        chunks: List[Chunk],
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> None:
        """
        Process direct blocks only, then recursively process children.

        This prevents parent/child sections from accidentally sharing
        prose chunks.
        """
        section_kind = self._classify_section(section)

        if section.blocks:
            self._process_blocks(
                blocks=section.blocks,
                chunks=chunks,
                document_id=document_id,
                document_metadata=document_metadata,
                breadcrumb=breadcrumb,
                section_kind=section_kind,
            )

        for child in section.children:
            self._process_section(
                section=child,
                chunks=chunks,
                document_id=document_id,
                document_metadata=document_metadata,
                breadcrumb=breadcrumb + [child.title],
            )

    # ==================================================================
    # BLOCK ROUTING
    # ==================================================================

    def _process_blocks(
        self,
        blocks: List[Block],
        chunks: List[Chunk],
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
        section_kind: str,
    ) -> None:
        prose_buffer: List[Block] = []

        def flush_prose() -> None:
            if not prose_buffer:
                return

            chunks.extend(
                self._chunk_prose_blocks(
                    blocks=list(prose_buffer),
                    document_id=document_id,
                    document_metadata=document_metadata,
                    breadcrumb=breadcrumb,
                )
            )

            prose_buffer.clear()

        for block in blocks:
            self._mark_source_consumed(block)

            # ----------------------------------------------------------
            # REFERENCES SECTION
            # ----------------------------------------------------------

            if section_kind == "references":
                flush_prose()

                chunks.append(
                    self._build_reference_chunk(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

                continue

            # ----------------------------------------------------------
            # ADMINISTRATIVE SECTION
            # ----------------------------------------------------------

            if section_kind == "administrative":
                flush_prose()

                chunks.append(
                    self._build_administrative_chunk(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

                continue

            block_type = self._normalize_block_type(block)

            # ----------------------------------------------------------
            # PARAGRAPH
            # ----------------------------------------------------------

            if block_type == "paragraph":
                prose_buffer.append(block)
                continue

            # Every structured object is a hard prose boundary.
            flush_prose()

            # ----------------------------------------------------------
            # LIST
            # ----------------------------------------------------------

            if block_type == "list":
                chunks.extend(
                    self._build_list_chunks(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

            # ----------------------------------------------------------
            # TABLE
            # ----------------------------------------------------------

            elif block_type == "table":
                chunks.extend(
                    self._build_table_chunks(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

            # ----------------------------------------------------------
            # FIGURE
            # ----------------------------------------------------------

            elif block_type == "figure":
                chunks.append(
                    self._build_figure_chunk(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

            # ----------------------------------------------------------
            # EQUATION
            # ----------------------------------------------------------

            elif block_type in {
                "formula",
                "equation",
            }:
                chunks.append(
                    self._build_equation_chunk(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

            # ----------------------------------------------------------
            # REFERENCE BLOCK
            # ----------------------------------------------------------

            elif block_type == "reference":
                chunks.append(
                    self._build_reference_chunk(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

            # ----------------------------------------------------------
            # ADMIN BLOCK
            # ----------------------------------------------------------

            elif block_type == "administrative":
                chunks.append(
                    self._build_administrative_chunk(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

            # ----------------------------------------------------------
            # HEADING
            # ----------------------------------------------------------

            elif block_type == "heading":
                # Heading is represented through breadcrumb/context.
                continue

            # ----------------------------------------------------------
            # UNKNOWN
            # ----------------------------------------------------------

            else:
                chunks.append(
                    self._build_fallback_chunk(
                        block=block,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

        flush_prose()

    # ==================================================================
    # PROSE
    # ==================================================================

    def _chunk_prose_blocks(
        self,
        blocks: List[Block],
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        current_blocks: List[Block] = []
        current_length = 0

        for block in blocks:

            text = self._clean_text(
                block.content
            )

            if not text:
                continue

            text_length = len(text)

            # ----------------------------------------------------------
            # Single paragraph is enormous
            # ----------------------------------------------------------

            if text_length > self.hard_max_prose_chars:

                if current_blocks:
                    chunks.append(
                        self._build_prose_chunk(
                            blocks=current_blocks,
                            document_id=document_id,
                            document_metadata=document_metadata,
                            breadcrumb=breadcrumb,
                        )
                    )

                    current_blocks = []
                    current_length = 0

                split_blocks = (
                    self._split_oversized_paragraph(
                        block
                    )
                )

                for split_block in split_blocks:
                    chunks.append(
                        self._build_prose_chunk(
                            blocks=[split_block],
                            document_id=document_id,
                            document_metadata=document_metadata,
                            breadcrumb=breadcrumb,
                        )
                    )

                continue

            # ----------------------------------------------------------
            # Regular paragraph
            # ----------------------------------------------------------

            separator_length = (
                2
                if current_blocks
                else 0
            )

            would_exceed = (
                bool(current_blocks)
                and
                current_length
                + separator_length
                + text_length
                > self.max_prose_chars
            )

            if would_exceed:
                chunks.append(
                    self._build_prose_chunk(
                        blocks=current_blocks,
                        document_id=document_id,
                        document_metadata=document_metadata,
                        breadcrumb=breadcrumb,
                    )
                )

                current_blocks = [block]
                current_length = text_length

            else:
                current_blocks.append(block)

                current_length += (
                    separator_length
                    + text_length
                )

        if current_blocks:
            chunks.append(
                self._build_prose_chunk(
                    blocks=current_blocks,
                    document_id=document_id,
                    document_metadata=document_metadata,
                    breadcrumb=breadcrumb,
                )
            )

        return chunks

    def _build_prose_chunk(
        self,
        blocks: List[Block],
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> Chunk:

        text_parts: List[str] = []
        source_block_ids: List[str] = []
        citation_refs: List[str] = []
        footnote_refs: List[str] = []

        for block in blocks:

            text = self._clean_text(
                block.content
            )

            if text:
                text_parts.append(text)

            block_id = self._get_block_id(
                block
            )

            if block_id:
                source_block_ids.append(
                    block_id
                )

            for ref in self._get_refs(
                block,
                "citation_refs",
            ):
                if ref not in citation_refs:
                    citation_refs.append(ref)

            for ref in self._get_refs(
                block,
                "footnote_refs",
            ):
                if ref not in footnote_refs:
                    footnote_refs.append(ref)

        text = "\n\n".join(
            text_parts
        )

        chunk_id, position = (
            self._next_prose_id(
                document_id
            )
        )

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            embedding_text=(
                self._build_embedding_text(
                    text=text,
                    breadcrumb=breadcrumb,
                )
            ),
            chunk_type="paragraph",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            source_block_ids=source_block_ids,
            citation_refs=citation_refs,
            footnote_refs=footnote_refs,
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
            },
            document_position=position,
            retrieval_eligible=True,
            chunk_version=self.CHUNK_VERSION,
        )

    # ==================================================================
    # LISTS
    # ==================================================================

    def _build_list_chunks(
        self,
        block: Block,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> List[Chunk]:

        metadata = block.metadata or {}

        label = self._clean_text(
            metadata.get(
                "label",
                "",
            )
        )

        text = self._clean_text(
            block.content
        )

        if label:
            text = (
                f"{label}\n\n{text}"
                if text
                else label
            )

        chunk_id, position = (
            self._next_prose_id(
                document_id
            )
        )

        return [
            Chunk(
                id=chunk_id,
                document_id=document_id,
                text=text,
                embedding_text=(
                    self._build_embedding_text(
                        text=text,
                        breadcrumb=breadcrumb,
                    )
                ),
                chunk_type="list",
                section=(
                    breadcrumb[0]
                    if breadcrumb
                    else None
                ),
                subsection=(
                    breadcrumb[-1]
                    if len(breadcrumb) > 1
                    else None
                ),
                breadcrumb=list(
                    breadcrumb
                ),
                source_block_ids=(
                    self._single_block_id(
                        block
                    )
                ),
                citation_refs=self._get_refs(
                    block,
                    "citation_refs",
                ),
                footnote_refs=self._get_refs(
                    block,
                    "footnote_refs",
                ),
                metadata={
                    **document_metadata,
                    "full_breadcrumb": (
                        " > ".join(
                            breadcrumb
                        )
                    ),
                    "list_label": (
                        label
                        if label
                        else None
                    ),
                },
                document_position=position,
                retrieval_eligible=True,
                chunk_version=self.CHUNK_VERSION,
            )
        ]

    # ==================================================================
    # TABLE IDS
    # ==================================================================

    def _resolve_table_id(
        self,
        block: Block,
        document_id: str,
    ) -> str:

        metadata = block.metadata or {}

        nested_table = metadata.get(
            "table",
            {},
        )

        if not isinstance(
            nested_table,
            dict,
        ):
            nested_table = {}

        # --------------------------------------------------------------
        # 1. Parser-provided table ID
        # --------------------------------------------------------------

        provided_id = (
            nested_table.get(
                "table_id"
            )
            or metadata.get(
                "table_id"
            )
        )

        if provided_id:
            return str(
                provided_id
            )

        # --------------------------------------------------------------
        # 2. Source block ID
        # --------------------------------------------------------------

        block_id = self._get_block_id(
            block
        )

        if block_id:
            return f"table_{block_id}"

        # --------------------------------------------------------------
        # 3. Guaranteed document-local fallback
        # --------------------------------------------------------------

        table_id = (
            f"table_{self._table_counter}"
        )

        self._table_counter += 1

        return table_id

    # ==================================================================
    # TABLES
    # ==================================================================

    def _build_table_chunks(
        self,
        block: Block,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> List[Chunk]:

        metadata = block.metadata or {}

        nested_table = metadata.get(
            "table",
            {},
        )

        if not isinstance(
            nested_table,
            dict,
        ):
            nested_table = {}

        table_id = self._resolve_table_id(
            block=block,
            document_id=document_id,
        )

        caption = self._clean_text(
            nested_table.get(
                "caption",
                metadata.get(
                    "caption",
                    "",
                ),
            )
        )

        label = self._clean_text(
            nested_table.get(
                "label",
                metadata.get(
                    "label",
                    "",
                ),
            )
        )

        # Human-facing table label (e.g. "Table 1") for self-contained
        # row/embedding context. ``table_id`` remains the stable identity.
        display_label = (
            label
            if label
            else f"Table {table_id}"
        )

        columns = nested_table.get(
            "columns",
            metadata.get(
                "columns",
                [],
            ),
        )

        categories = nested_table.get(
            "categories",
            metadata.get(
                "categories",
                [],
            ),
        )

        rows = nested_table.get(
            "rows",
            metadata.get(
                "rows",
                [],
            ),
        )

        footnotes = nested_table.get(
            "footnotes",
            metadata.get(
                "footnotes",
                [],
            ),
        )

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        summary_text = (
            self._build_table_summary_text(
                label=label,
                caption=caption,
                columns=columns,
                categories=categories,
                fallback_content=block.content,
            )
        )

        # A table summary must never be empty (section 93). Fall back to
        # the table identity when no label/caption/columns are available.
        if not summary_text:
            summary_text = display_label

        summary_id = self._make_unique_id(
            f"{document_id}_{table_id}_summary"
        )

        summary_position = (
            self._next_position()
        )

        summary_chunk = Chunk(
            id=summary_id,
            document_id=document_id,
            text=summary_text,
            embedding_text=(
                self._build_embedding_text(
                    text=summary_text,
                    breadcrumb=breadcrumb,
                    prefix=display_label,
                )
            ),
            chunk_type="table_summary",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            object_id=table_id,
            table_id=table_id,
            source_block_ids=(
                self._single_block_id(
                    block
                )
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
                "table_id": table_id,
                "columns": columns,
                "categories": categories,
            },
            document_position=summary_position,
            retrieval_eligible=True,
            chunk_version=self.CHUNK_VERSION,
        )

        result: List[Chunk] = [
            summary_chunk
        ]

        # --------------------------------------------------------------
        # Rows
        # --------------------------------------------------------------

        for row_index, row in enumerate(
            rows
        ):
            result.append(
                self._build_table_row_chunk(
                    row=row,
                    row_index=row_index,
                    table_id=table_id,
                    table_label=display_label,
                    parent_id=summary_id,
                    document_id=document_id,
                    document_metadata=document_metadata,
                    breadcrumb=breadcrumb,
                )
            )

        # --------------------------------------------------------------
        # Footnotes
        # --------------------------------------------------------------

        if footnotes:
            result.append(
                self._build_table_footnotes_chunk(
                    footnotes=footnotes,
                    table_id=table_id,
                    table_label=display_label,
                    parent_id=summary_id,
                    document_id=document_id,
                    document_metadata=document_metadata,
                    breadcrumb=breadcrumb,
                    source_block=block,
                )
            )

        rows_source = len(rows) if isinstance(rows, list) else 0
        footnotes_source = (
            len(footnotes) if isinstance(footnotes, list) else 0
        )
        rows_chunks = sum(
            1 for c in result if c.chunk_type == "table_row"
        )
        footnotes_chunks = sum(
            1 for c in result if c.chunk_type == "table_footnotes"
        )

        # A single table_footnotes chunk carries *all* footnotes, so
        # preservation is measured as (has source) <=> (has chunk).
        footnotes_preserved = (
            (footnotes_source > 0) == (footnotes_chunks > 0)
        )
        self._table_coverage[table_id] = {
            "rows_source": rows_source,
            "rows_chunks": rows_chunks,
            "footnotes_source": footnotes_source,
            "footnotes_chunks": footnotes_chunks,
            "footnotes_preserved": footnotes_preserved,
            "pass": (
                rows_source == rows_chunks
                and footnotes_preserved
            ),
        }

        return result

    def _build_table_summary_text(
        self,
        caption: str,
        columns: Any,
        categories: Any,
        fallback_content: Any,
        label: str = "",
    ) -> str:

        parts: List[str] = []

        if label and caption:
            parts.append(f"{label}: {caption}")
        elif caption:
            parts.append(caption)
        elif label:
            parts.append(label)

        if columns:
            formatted_columns = [
                self._format_column(column)
                for column in columns
            ]

            formatted_columns = [
                value
                for value in formatted_columns
                if value
            ]

            if formatted_columns:
                parts.append(
                    "Columns:\n"
                    + "\n".join(
                        f"- {value}"
                        for value in formatted_columns
                    )
                )

        if categories:
            formatted_categories = [
                self._format_category(category)
                for category in categories
            ]

            formatted_categories = [
                value
                for value in formatted_categories
                if value
            ]

            if formatted_categories:
                parts.append(
                    "Categories:\n"
                    + "\n".join(
                        f"- {value}"
                        for value in formatted_categories
                    )
                )

        if parts:
            return "\n\n".join(parts)

        # Fallback only when structured metadata is missing.
        # Remove HTML/XML markup so raw table markup does not become
        # embedding content.
        return self._strip_markup(
            fallback_content
        )

    def _build_table_row_chunk(
        self,
        row: Any,
        row_index: int,
        table_id: str,
        parent_id: str,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
        table_label: str = "",
    ) -> Chunk:

        if isinstance(
            row,
            dict,
        ):
            row_text = self._clean_text(
                row.get(
                    "text",
                    "",
                )
            )

            row_label = self._clean_text(
                row.get(
                    "row_label",
                    "",
                )
            )

            group_path = row.get(
                "group_path",
                [],
            )

            values = row.get(
                "values",
                row.get(
                    "cells",
                    {},
                ),
            )

            source_block_ids = row.get(
                "source_block_ids",
                [],
            )

        else:
            row_text = self._clean_text(
                row
            )

            row_label = ""
            group_path = []
            values = {}
            source_block_ids = []

        if not row_text:
            row_text = (
                self._format_table_row_text(
                    row_label=row_label,
                    group_path=group_path,
                    values=values,
                )
            )

        row_id = self._make_unique_id(
            f"{document_id}_{table_id}"
            f"_row_{row_index}"
        )

        position = self._next_position()

        return Chunk(
            id=row_id,
            document_id=document_id,
            text=row_text,
            embedding_text=(
                self._build_embedding_text(
                    text=row_text,
                    breadcrumb=breadcrumb,
                    prefix=(
                        table_label
                        if table_label
                        else f"Table {table_id}"
                    ),
                )
            ),
            chunk_type="table_row",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            parent_id=parent_id,
            object_id=table_id,
            table_id=table_id,
            row_label=(
                row_label
                if row_label
                else None
            ),
            group_path=list(
                group_path or []
            ),
            source_block_ids=list(
                source_block_ids
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
                "table_id": table_id,
                "row_index": row_index,
            },
            document_position=position,
            retrieval_eligible=True,
            chunk_version=self.CHUNK_VERSION,
        )

    def _build_table_footnotes_chunk(
        self,
        footnotes: Any,
        table_id: str,
        parent_id: str,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
        source_block: Block,
        table_label: str = "",
    ) -> Chunk:

        text = self._format_footnotes(
            footnotes
        )

        chunk_id = self._make_unique_id(
            f"{document_id}_{table_id}"
            "_footnotes"
        )

        position = self._next_position()

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            embedding_text=(
                self._build_embedding_text(
                    text=text,
                    breadcrumb=breadcrumb,
                    prefix=(
                        f"{table_label} footnotes"
                        if table_label
                        else f"Table {table_id} footnotes"
                    ),
                )
            ),
            chunk_type="table_footnotes",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            parent_id=parent_id,
            object_id=table_id,
            table_id=table_id,
            source_block_ids=(
                self._single_block_id(
                    source_block
                )
            ),
            citation_refs=self._get_refs(
                source_block,
                "citation_refs",
            ),
            footnote_refs=self._get_refs(
                source_block,
                "footnote_refs",
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
                "table_id": table_id,
            },
            document_position=position,
            retrieval_eligible=True,
            chunk_version=self.CHUNK_VERSION,
        )

    # ==================================================================
    # FIGURES
    # ==================================================================

    def _resolve_figure_id(
        self,
        block: Block,
        document_id: str,
    ) -> str:

        metadata = block.metadata or {}

        provided_id = metadata.get(
            "figure_id"
        )

        if provided_id:
            return str(
                provided_id
            )

        block_id = self._get_block_id(
            block
        )

        if block_id:
            return f"figure_{block_id}"

        figure_id = (
            f"figure_{self._figure_counter}"
        )

        self._figure_counter += 1

        return figure_id

    def _build_figure_chunk(
        self,
        block: Block,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> Chunk:

        metadata = block.metadata or {}

        figure_id = (
            self._resolve_figure_id(
                block=block,
                document_id=document_id,
            )
        )

        text = self._clean_text(
            block.content
        )

        if not text:
            text = self._build_figure_text(
                metadata
            )

        # A figure must never produce an empty retrieval chunk. Fall back
        # to its identity when no label/caption/description is available.
        if not text:
            text = (
                self._clean_text(
                    metadata.get(
                        "label"
                    )
                )
                or f"Figure {figure_id}"
            )

        chunk_id = self._make_unique_id(
            f"{document_id}_{figure_id}"
        )

        position = self._next_position()

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            embedding_text=(
                self._build_embedding_text(
                    text=text,
                    breadcrumb=breadcrumb,
                    prefix=f"Figure {figure_id}",
                )
            ),
            chunk_type="figure",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            object_id=figure_id,
            figure_id=figure_id,
            source_block_ids=(
                self._single_block_id(
                    block
                )
            ),
            citation_refs=self._get_refs(
                block,
                "citation_refs",
            ),
            footnote_refs=self._get_refs(
                block,
                "footnote_refs",
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
                "figure_id": figure_id,
                "image_ref": metadata.get(
                    "image_ref"
                ),
            },
            document_position=position,
            retrieval_eligible=True,
            chunk_version=self.CHUNK_VERSION,
        )

    # ==================================================================
    # EQUATIONS
    # ==================================================================

    def _resolve_equation_id(
        self,
        block: Block,
        document_id: str,
    ) -> str:

        metadata = block.metadata or {}

        provided_id = metadata.get(
            "equation_id"
        )

        if provided_id:
            return str(
                provided_id
            )

        block_id = self._get_block_id(
            block
        )

        if block_id:
            return f"equation_{block_id}"

        equation_id = (
            f"equation_{self._equation_counter}"
        )

        self._equation_counter += 1

        return equation_id

    def _build_equation_chunk(
        self,
        block: Block,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> Chunk:

        equation_id = (
            self._resolve_equation_id(
                block=block,
                document_id=document_id,
            )
        )

        text = self._clean_text(
            block.content
        )

        chunk_id = self._make_unique_id(
            f"{document_id}_{equation_id}"
        )

        position = self._next_position()

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            embedding_text=(
                self._build_embedding_text(
                    text=text,
                    breadcrumb=breadcrumb,
                    prefix=(
                        f"Equation {equation_id}"
                    ),
                )
            ),
            chunk_type="equation",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            object_id=equation_id,
            equation_id=equation_id,
            source_block_ids=(
                self._single_block_id(
                    block
                )
            ),
            citation_refs=self._get_refs(
                block,
                "citation_refs",
            ),
            footnote_refs=self._get_refs(
                block,
                "footnote_refs",
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
                "equation_id": equation_id,
            },
            document_position=position,
            retrieval_eligible=True,
            chunk_version=self.CHUNK_VERSION,
        )

    # ==================================================================
    # REFERENCES
    # ==================================================================

    def _resolve_reference_id(
        self,
        block: Block,
    ) -> str:

        metadata = block.metadata or {}

        provided_id = (
            metadata.get(
                "reference_id"
            )
            or metadata.get(
                "id"
            )
            or metadata.get(
                "xml_id"
            )
        )

        if provided_id:
            return str(
                provided_id
            )

        reference_id = (
            f"ref_{self._reference_counter}"
        )

        self._reference_counter += 1

        return reference_id

    def _build_reference_chunk(
        self,
        block: Block,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> Chunk:

        metadata = block.metadata or {}

        reference_id = (
            self._resolve_reference_id(
                block
            )
        )

        text = self._clean_text(
            block.content
        )

        chunk_id = self._make_unique_id(
            f"{document_id}_{reference_id}"
        )

        position = self._next_position()

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            embedding_text="",
            chunk_type="reference",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            object_id=reference_id,
            reference_id=reference_id,
            source_block_ids=(
                self._single_block_id(
                    block
                )
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
                "reference_id": reference_id,
                "doi": metadata.get(
                    "doi"
                ),
                "pmid": metadata.get(
                    "pmid"
                ),
                "pmcid": metadata.get(
                    "pmcid"
                ),
            },
            document_position=position,
            retrieval_eligible=False,
            chunk_version=self.CHUNK_VERSION,
        )

    # ==================================================================
    # ADMINISTRATIVE
    # ==================================================================

    def _build_administrative_chunk(
        self,
        block: Block,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> Chunk:

        text = self._clean_text(
            block.content
        )

        chunk_id, position = (
            self._next_prose_id(
                document_id
            )
        )

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            embedding_text="",
            chunk_type="administrative",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            source_block_ids=(
                self._single_block_id(
                    block
                )
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
            },
            document_position=position,
            retrieval_eligible=False,
            chunk_version=self.CHUNK_VERSION,
        )

    # ==================================================================
    # FALLBACK
    # ==================================================================

    def _build_fallback_chunk(
        self,
        block: Block,
        document_id: str,
        document_metadata: dict[str, Any],
        breadcrumb: List[str],
    ) -> Chunk:

        text = self._clean_text(
            block.content
        )

        chunk_id, position = (
            self._next_prose_id(
                document_id
            )
        )

        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            embedding_text=(
                self._build_embedding_text(
                    text=text,
                    breadcrumb=breadcrumb,
                )
            ),
            chunk_type="paragraph",
            section=(
                breadcrumb[0]
                if breadcrumb
                else None
            ),
            subsection=(
                breadcrumb[-1]
                if len(breadcrumb) > 1
                else None
            ),
            breadcrumb=list(
                breadcrumb
            ),
            source_block_ids=(
                self._single_block_id(
                    block
                )
            ),
            metadata={
                **document_metadata,
                "full_breadcrumb": (
                    " > ".join(
                        breadcrumb
                    )
                ),
                "original_block_type": (
                    block.block_type
                ),
            },
            document_position=position,
            retrieval_eligible=True,
            chunk_version=self.CHUNK_VERSION,
        )

    # ==================================================================
    # SECTION CLASSIFICATION
    # ==================================================================

    def _classify_section(
        self,
        section: Section,
    ) -> str:

        # The AST parser records an explicit deterministic classification.
        # Prefer it when present; fall back to title-based classification
        # for ASTs produced by other builders.
        explicit = (section.metadata or {}).get(
            "section_type"
        )
        if explicit in {
            "content",
            "administrative",
            "references",
        }:
            return explicit

        title = self._normalize_title(
            section.title
        )

        if not title:
            return "content"

        # References (exclude "reference value/range" style sections)
        if title in {
            "reference",
            "references",
            "bibliography",
        }:
            return "references"

        reference_exclusions = {
            "reference value",
            "reference values",
            "reference range",
            "reference ranges",
            "reference interval",
            "reference standard",
        }
        if any(term in title for term in reference_exclusions):
            return "content"

        if "reference" in title:
            return "references"

        # Administrative
        administrative_titles = {
            "funding",
            "funding sources",
            "acknowledgements",
            "acknowledgments",
            "author contributions",
            "authors contributions",
            "data availability",
            "conflict of interest",
            "conflicts of interest",
            "competing interests",
            "supplementary material",
            "supplementary materials",
            "notes",
        }

        if title in administrative_titles:
            return "administrative"

        if "funding" in title:
            return "administrative"

        if "conflict" in title:
            return "administrative"

        if "competing interest" in title:
            return "administrative"

        if "data availability" in title:
            return "administrative"

        if "acknowledg" in title:
            return "administrative"

        if "supplementary" in title:
            return "administrative"

        if (
            "author" in title
            and "contribution" in title
        ):
            return "administrative"

        return "content"

    # ==================================================================
    # BLOCK TYPE NORMALIZATION
    # ==================================================================

    def _normalize_block_type(
        self,
        block: Block,
    ) -> str:

        metadata = block.metadata or {}

        candidates = [
            metadata.get(
                "chunk_type"
            ),
            metadata.get(
                "semantic_type"
            ),
            block.block_type,
        ]

        candidate = ""

        for value in candidates:
            if value:
                candidate = (
                    str(value)
                    .strip()
                    .lower()
                )
                break

        aliases = {
            "paragraph": "paragraph",

            "list": "list",
            "ordered_list": "list",
            "unordered_list": "list",
            "list_item_group": "list",

            "table": "table",

            "figure": "figure",
            "fig": "figure",

            "formula": "equation",
            "equation": "equation",
            "eq": "equation",

            "reference": "reference",
            "ref": "reference",

            "administrative": "administrative",
            "admin": "administrative",

            "heading": "heading",
        }

        return aliases.get(
            candidate,
            candidate,
        )

    # ==================================================================
    # OVERSIZED PROSE
    # ==================================================================

    def _split_oversized_paragraph(
        self,
        block: Block,
    ) -> List[Block]:

        text = self._clean_text(
            block.content
        )

        sentences = self._split_sentences(
            text
        )

        if len(sentences) <= 1:
            return [block]

        result: List[Block] = []

        current: List[str] = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(
                sentence
            )

            additional_length = (
                sentence_length
                if not current
                else sentence_length + 1
            )

            if (
                current
                and
                current_length
                + additional_length
                > self.max_prose_chars
            ):
                result.append(
                    Block(
                        block_type="paragraph",
                        content=" ".join(
                            current
                        ),
                        metadata=dict(
                            block.metadata
                            or {}
                        ),
                    )
                )

                current = [sentence]
                current_length = sentence_length

            else:
                current.append(
                    sentence
                )
                current_length += additional_length

        if current:
            result.append(
                Block(
                    block_type="paragraph",
                    content=" ".join(
                        current
                    ),
                    metadata=dict(
                        block.metadata
                        or {}
                    ),
                )
            )

        return result

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text.strip(),
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ==================================================================
    # TABLE FORMATTING
    # ==================================================================

    def _format_table_row_text(
        self,
        row_label: str,
        group_path: Any,
        values: Any,
    ) -> str:

        parts: List[str] = []

        if group_path:

            if isinstance(
                group_path,
                list,
            ):
                group_text = (
                    " > ".join(
                        str(value)
                        for value in group_path
                    )
                )
            else:
                group_text = str(
                    group_path
                )

            parts.append(
                f"Group: {group_text}"
            )

        if row_label:
            parts.append(
                f"Row: {row_label}"
            )

        if isinstance(
            values,
            dict,
        ):
            for key, value in values.items():
                parts.append(
                    f"{key}: {value}"
                )

        elif isinstance(
            values,
            list,
        ):
            for value in values:
                parts.append(
                    str(value)
                )

        elif values:
            parts.append(
                str(values)
            )

        return "\n".join(parts)

    def _format_column(
        self,
        column: Any,
    ) -> str:

        if isinstance(
            column,
            dict,
        ):
            name = column.get(
                "name",
                column.get(
                    "label",
                    "",
                ),
            )

            unit = column.get(
                "unit",
                "",
            )

            if name and unit:
                return (
                    f"{name} ({unit})"
                )

            return str(
                name or ""
            )

        return str(column)

    def _format_category(
        self,
        category: Any,
    ) -> str:

        if isinstance(
            category,
            dict,
        ):
            return str(
                category.get(
                    "name",
                    category.get(
                        "label",
                        "",
                    ),
                )
            )

        return str(category)

    def _format_footnotes(
        self,
        footnotes: Any,
    ) -> str:

        if isinstance(
            footnotes,
            str,
        ):
            return self._clean_text(
                footnotes
            )

        if isinstance(
            footnotes,
            list,
        ):
            result: List[str] = []

            for footnote in footnotes:

                if isinstance(
                    footnote,
                    dict,
                ):
                    marker = footnote.get(
                        "marker",
                        "",
                    )

                    text = footnote.get(
                        "text",
                        footnote.get(
                            "content",
                            "",
                        ),
                    )

                    text = self._clean_text(
                        text
                    )

                    if marker:
                        result.append(
                            f"[{marker}] {text}"
                        )
                    elif text:
                        result.append(
                            text
                        )

                elif footnote:
                    result.append(
                        self._clean_text(
                            footnote
                        )
                    )

            return "\n".join(result)

        return self._clean_text(
            footnotes
        )

    # ==================================================================
    # FIGURE / EMBEDDING
    # ==================================================================

    def _build_figure_text(
        self,
        metadata: dict[str, Any],
    ) -> str:

        parts: List[str] = []

        caption = self._clean_text(
            metadata.get(
                "caption",
                "",
            )
        )

        description = self._clean_text(
            metadata.get(
                "description",
                "",
            )
        )

        notes = self._clean_text(
            metadata.get(
                "notes",
                "",
            )
        )

        if caption:
            parts.append(
                caption
            )

        if description:
            parts.append(
                description
            )

        if notes:
            parts.append(
                notes
            )

        return "\n\n".join(
            parts
        )

    def _build_embedding_text(
        self,
        text: str,
        breadcrumb: List[str],
        prefix: Optional[str] = None,
    ) -> str:

        parts: List[str] = []

        if prefix:
            prefix = prefix.strip()
            if prefix:
                parts.append(prefix)

        if breadcrumb:
            breadcrumb_text = (
                " > ".join(
                    part.strip()
                    for part in breadcrumb
                    if part
                    and part.strip()
                )
            )

            if breadcrumb_text:
                parts.append(
                    breadcrumb_text
                )

        text = text.strip()

        if text:
            parts.append(text)

        return "\n\n".join(parts)

    # ==================================================================
    # PROVENANCE
    # ==================================================================

    def _get_block_id(
        self,
        block: Block,
    ) -> Optional[str]:

        metadata = block.metadata or {}

        value = (
            metadata.get(
                "block_id"
            )
            or metadata.get(
                "id"
            )
            or metadata.get(
                "xml_id"
            )
        )

        if value is None:
            return None

        return str(value)

    def _single_block_id(
        self,
        block: Block,
    ) -> List[str]:

        block_id = (
            self._get_block_id(block)
        )

        return (
            [block_id]
            if block_id
            else []
        )

    def _get_refs(
        self,
        block: Block,
        key: str,
    ) -> List[str]:

        value = (
            block.metadata or {}
        ).get(
            key,
            [],
        )

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            return [value]

        if isinstance(
            value,
            list,
        ):
            return [
                str(item)
                for item in value
                if item is not None
            ]

        return []

    # ==================================================================
    # TEXT HELPERS
    # ==================================================================

    def _clean_text(
        self,
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = str(value)

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = [
            line.strip()
            for line in text.split("\n")
        ]

        return "\n".join(
            line
            for line in lines
            if line
        ).strip()

    def _strip_markup(
        self,
        value: Any,
    ) -> str:
        """
        Fallback cleaning for raw XML/HTML.
        Structured table data should normally avoid reaching this.
        """
        text = self._clean_text(value)

        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _normalize_title(
        self,
        title: str,
    ) -> str:

        return " ".join(
            title.strip()
            .lower()
            .split()
        )

    def _initial_breadcrumb(
        self,
        section: Section,
    ) -> List[str]:

        if section.breadcrumb:
            return list(
                section.breadcrumb
            )

        return [
            section.title
        ]


    # ==================================================================
    # SOURCE COVERAGE / REPORTING
    # ==================================================================

    def _collect_source_blocks(self, document: Document) -> None:
        """Walk the AST and record every semantic block for coverage."""
        blocks: List[Block] = []
        type_counts: dict[str, int] = {}

        def walk(sections: List[Section]) -> None:
            for section in sections:
                for block in section.blocks:
                    blocks.append(block)
                    key = self._normalize_block_type(block)
                    type_counts[key] = type_counts.get(key, 0) + 1
                walk(section.children)

        walk(document.sections)

        self._all_source_blocks = blocks
        self._block_to_index = {
            id(block): index
            for index, block in enumerate(blocks)
        }
        self._consumed_source_blocks = set()
        self._source_type_counts = type_counts

    def _mark_source_consumed(self, block: Block) -> None:
        index = self._block_to_index.get(id(block))
        if index is not None:
            self._consumed_source_blocks.add(index)

    def _build_report(
        self,
        document: Document,
        chunks: List[Chunk],
    ) -> dict[str, Any]:
        total_source = len(self._all_source_blocks)
        consumed = len(self._consumed_source_blocks)
        unhandled = total_source - consumed

        type_counts: dict[str, int] = {}
        retrieval_eligible = 0
        for chunk in chunks:
            type_counts[chunk.chunk_type] = (
                type_counts.get(chunk.chunk_type, 0) + 1
            )
            if chunk.retrieval_eligible:
                retrieval_eligible += 1

        report: dict[str, Any] = {
            "document_id": document.pmcid,
            "source_blocks": total_source,
            "source_blocks_consumed": consumed,
            "source_blocks_unhandled": unhandled,
            "chunks": len(chunks),
            "retrieval_eligible": retrieval_eligible,
            "chunks_by_type": dict(sorted(type_counts.items())),
            "source_blocks_by_type": dict(sorted(self._source_type_counts.items())),
            "paragraphs": type_counts.get("paragraph", 0),
            "lists": type_counts.get("list", 0),
            "tables": type_counts.get("table_summary", 0),
            "table_rows": type_counts.get("table_row", 0),
            "table_footnotes": type_counts.get("table_footnotes", 0),
            "figures": type_counts.get("figure", 0),
            "equations": type_counts.get("equation", 0),
            "references": type_counts.get("reference", 0),
            "administrative": type_counts.get("administrative", 0),
            "coverage": (
                round(consumed / total_source, 6)
                if total_source
                else 1.0
            ),
            "table_coverage": dict(sorted(self._table_coverage.items())),
            "warnings": list(self._warnings),
        }
        return report

    def chunk_with_report(
        self,
        document: Document,
    ) -> tuple[List[Chunk], dict[str, Any]]:
        """Chunk and return ``(chunks, report)`` for diagnostics/statistics."""
        chunks = self.chunk(document)
        return chunks, self._report
