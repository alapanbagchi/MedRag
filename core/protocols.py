from pathlib import Path
from typing import Protocol, List

from core.models import Document, Chunk


class Collector(Protocol):
    def collect(self):
        ...

class Parser(Protocol):
    def parse(self, source: Path):
        ...

class ASTBuilder(Protocol):
    """Converts Markdown into a structured Document AST."""
    def build(self, markdown_text: str, metadata: dict) -> Document:
        ...

class Chunker(Protocol):
    """Breaks the Document AST into vector-ready Chunks (handles Table Normalization here)."""
    def chunk(self, document: Document) -> List[Chunk]:
        ...

