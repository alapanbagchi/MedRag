"""Chunking package: Markdown -> chunks + units, one concern per file.

Public entry:
    python -m src.chunking --input data/md          # CLI -> medpat (ParadeDB)
    from src.chunking import chunk_document         # library API

Modules: documents (pipeline), parsing, tokens, prose, tables, sections,
chunks, ids, models, classification, store_pg, cli.
"""

from src.chunking.cli import main
from src.chunking.documents import chunk_document

__all__ = ["chunk_document", "main"]
