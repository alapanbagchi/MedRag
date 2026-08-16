"""
MedicalConcept
    identify the medical concept
    store its basic name/label
    store identifiers from external sources
    store useful descriptive information
"""
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Any


@dataclass
class MedicalConcept:
    id: str
    name: str
    description: str
    """
    Holds the identifiers for different biomedical sources. 
    
    Example:
    {
        "mondo": "MONDO:0005252",
        "omim": "209850",
        "orphanet": "106"
    }
    """
    identifiers: dict[str, str] = field(default_factory=dict)


@dataclass
class Block:
    """Represents a leaf node in the AST (Paragraph, Table, Figure, List)."""
    block_type: Literal["paragraph", "table", "figure", "list", "formula"]
    content: str  # Markdown text for paragraphs, HTML for tables
    metadata: dict = field(default_factory=dict)  # e.g., table_id, figure_id


@dataclass
class Section:
    """Represents a node in the document hierarchy."""
    title: str
    level: int  # 1, 2, 3 (from H1, H2, H3)
    breadcrumb: List[str]  # e.g., ["EMPA-KIDNEY Trial", "Results", "Baseline characteristics"]
    blocks: List[Block] = field(default_factory=list)
    children: List['Section'] = field(default_factory=list)


@dataclass
class Document:
    """The root AST node."""
    pmcid: str
    title: str
    metadata: dict = field(default_factory=dict)  # The YAML frontmatter
    sections: List[Section] = field(default_factory=list)


@dataclass
class Chunk:
    """The final unit of text sent to the vector database."""
    id: str
    text: str
    chunk_type: Literal["paragraph", "table_row", "table_summary", "figure_caption", "section_summary"]
    metadata: dict = field(default_factory=dict)  # pmcid, breadcrumb, table_id, etc.

    # This is where your existing model shines!
    # A downstream NER plugin can populate this with identified entities.
    concepts: List[MedicalConcept] = field(default_factory=list)