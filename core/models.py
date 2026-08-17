from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional


# ======================================================================
# Canonical / normalized block types
# ======================================================================
# The parser may emit any of these. Unknown block types are tolerated by
# the chunker (they fall back to a generic path) and recorded so content
# never disappears silently.
CANONICAL_BLOCK_TYPES = (
    "paragraph",
    "list",
    "table",
    "figure",
    "formula",
    "heading",
)

NORMALIZED_BLOCK_TYPES = (
    "paragraph",
    "list",
    "table",
    "figure",
    "equation",
    "reference",
    "administrative",
    "heading",
)

# Valid chunk types emitted by the chunker.
VALID_CHUNK_TYPES = (
    "paragraph",
    "list",
    "table_summary",
    "table_row",
    "table_footnotes",
    "figure",
    "equation",
    "reference",
    "administrative",
)


@dataclass
class MedicalConcept:
    id: str
    name: str
    description: str
    identifiers: dict[str, str] = field(default_factory=dict)


@dataclass
class Block:
    """A semantic AST block.

    ``block_type`` is intentionally a plain string rather than a strict
    Literal so that unknown/novel block types from new parsers never crash
    the pipeline. The chunker normalizes known types and routes unknown
    types through a safe, logged fallback path.
    """
    block_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    title: str
    level: int
    breadcrumb: List[str]
    blocks: List[Block] = field(default_factory=list)
    children: List["Section"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Classification hint: "content" | "administrative" | "references"
    # The chunker prefers this when present but always falls back to
    # deterministic title-based classification.
    section_type: str = "content"

    # Stable source identity (e.g. the JATS <sec id="..."> attribute).
    source_id: Optional[str] = None

    @property
    def classification(self) -> str:
        return (self.metadata.get("section_type") or self.section_type or "content")


@dataclass
class Document:
    pmcid: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    document_id: str

    # --------------------------------------------------------------
    # Content
    # --------------------------------------------------------------

    text: str
    embedding_text: str

    # --------------------------------------------------------------
    # Chunk type
    # --------------------------------------------------------------

    chunk_type: Literal[
        "paragraph",
        "list",
        "table_summary",
        "table_row",
        "table_footnotes",
        "figure",
        "equation",
        "reference",
        "administrative",
    ]

    # --------------------------------------------------------------
    # Structure
    # --------------------------------------------------------------

    section: Optional[str] = None
    subsection: Optional[str] = None
    breadcrumb: List[str] = field(default_factory=list)

    # --------------------------------------------------------------
    # Hierarchy / provenance
    # --------------------------------------------------------------

    parent_id: Optional[str] = None
    object_id: Optional[str] = None
    source_block_ids: List[str] = field(default_factory=list)

    # --------------------------------------------------------------
    # Structured objects
    # --------------------------------------------------------------

    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    equation_id: Optional[str] = None
    reference_id: Optional[str] = None

    # --------------------------------------------------------------
    # Table-specific
    # --------------------------------------------------------------

    row_label: Optional[str] = None
    group_path: List[str] = field(default_factory=list)

    # --------------------------------------------------------------
    # References
    # --------------------------------------------------------------

    citation_refs: List[str] = field(default_factory=list)
    footnote_refs: List[str] = field(default_factory=list)

    # --------------------------------------------------------------
    # Retrieval / validation
    # --------------------------------------------------------------

    embedding_token_count: int = 0
    document_position: int = 0
    retrieval_eligible: bool = True

    # --------------------------------------------------------------
    # Concepts
    # --------------------------------------------------------------

    concept_ids: List[str] = field(default_factory=list)

    # --------------------------------------------------------------
    # Miscellaneous
    # --------------------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------
    # Version
    # --------------------------------------------------------------

    chunk_version: str = "1.0"


# ======================================================================
# Structured AST objects (serialized into Block.metadata by the parser)
# ======================================================================
# These dataclasses document the canonical AST contract. The parser
# attaches their dict form (via dataclasses.asdict) to Block.metadata
# under the keys "table", "figure", "equation", and "reference" so the
# chunker consumes plain JSON-serializable structures.


@dataclass
class TableColumn:
    index: int
    name: str
    unit: str = ""
    colspan: int = 1
    rowspan: int = 1


@dataclass
class TableRow:
    row_label: str = ""
    group_path: List[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)
    cells: List[str] = field(default_factory=list)
    is_group_row: bool = False
    source_id: Optional[str] = None


@dataclass
class TableFootnote:
    marker: str = ""
    text: str = ""


@dataclass
class TableData:
    table_id: str = ""
    label: str = ""
    caption: str = ""
    columns: List[TableColumn] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    rows: List[TableRow] = field(default_factory=list)
    footnotes: List[TableFootnote] = field(default_factory=list)
    structure_degraded: bool = False
    source_block_id: Optional[str] = None


@dataclass
class FigureData:
    figure_id: str = ""
    label: str = ""
    caption: str = ""
    image_ref: str = ""
    panels: List[dict[str, Any]] = field(default_factory=list)
    description: str = ""
    notes: str = ""


@dataclass
class EquationData:
    equation_id: str = ""
    latex: str = ""
    text: str = ""


@dataclass
class ReferenceData:
    reference_id: str = ""
    citation_number: Optional[int] = None
    text: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""


# ======================================================================
# Validation diagnostics
# ======================================================================


@dataclass
class Diagnostic:
    severity: Literal["ERROR", "WARNING", "INFO"]
    code: str
    message: str
    block_id: Optional[str] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.block_id is not None:
            data["block_id"] = self.block_id
        if self.section is not None:
            data["section"] = self.section
        if self.chunk_id is not None:
            data["chunk_id"] = self.chunk_id
        if self.details:
            data["details"] = self.details
        return data


@dataclass
class ValidationReport:
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                severity=severity,  # type: ignore[arg-type]
                code=code,
                message=message,
                **kwargs,
            )
        )

    @property
    def errors(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "ERROR")

    @property
    def warnings(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "WARNING")

    @property
    def infos(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "INFO")

    @property
    def passed(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }