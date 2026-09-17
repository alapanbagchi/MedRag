"""Embedding package: encode medpat chunks via an OpenAI-compatible server.

Public entry:
    python -m src.embedding                         # CLI -> medpat.chunk_embeddings
    from src.embedding import EmbedClient           # library API

Modules: client (OpenAI-compatible HTTP), store (pending-chunk read + vector
upsert), cli (argparse + orchestration).
"""

from src.embedding.cli import main
from src.embedding.client import EmbedClient

__all__ = ["EmbedClient", "main"]
