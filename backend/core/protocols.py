"""Protocols: structural contracts implemented by pipeline components.

Using ``typing.Protocol`` (not ABCs) keeps the contracts implicit — any class
with the right shape satisfies them, and implementations stay plain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Collector(Protocol):
    """Anything that discovers and downloads articles for a topic.

    Implementations are expected to expose ``topic``, ``output_dir`` and
    ``max_workers`` and to make ``collect()`` idempotent/resumable (files
    that already exist are skipped).
    """

    topic: str
    output_dir: Path
    max_workers: int

    def collect(self) -> None:
        """Discover + download all matching articles into ``output_dir``."""
        ...