"""Prompt loader - the single place system prompts come from.

All agent system prompts live as plain-text files under src/prompts/ (or a
PROMPT_DIR override), one file per agent, so they can be edited without
touching code:

    src/prompts/
      legacy/        evidence.txt planner.txt rewriter.txt synthesizer.txt verifier.txt
      agentic_v1/    planner.txt loop.txt
      agentic_v2/    orchestrator.txt verifier.txt synthesize.txt
      agentic_v3/    master.txt search_planner.txt critic.txt deep_inspector.txt
                     replanner.txt contradiction.txt resolution.txt synthesize.txt

Env override: PROMPT_DIR=/path/to/prompts swaps the whole tree (useful for
per-deployment prompt variants). Resolution order: PROMPT_DIR env, else the
src/prompts directory shipped with the package.

Cache: prompts are read once per path and cached for the process lifetime.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Root of the shipped prompt tree (this module lives in src/prompts/).
DEFAULT_ROOT = Path(__file__).resolve().parent


def root_dir() -> Path:
    """The prompt tree root (PROMPT_DIR env override, else the package tree)."""
    override = os.environ.get("PROMPT_DIR", "").strip()
    return Path(override) if override else DEFAULT_ROOT


@lru_cache(maxsize=256)
def load_prompt(subdir: str, name: str) -> str:
    """Load one system prompt file from the prompt tree.

    Args:
        subdir: sub-folder name, e.g. "agentic_v3".
        name:   file name, e.g. "master.txt".

    Raises:
        FileNotFoundError: with the expected absolute path so a typo is loud
        instead of silently using a stale built-in prompt.
    """
    path = root_dir() / subdir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"prompt file not found: {path} "
            f"(expected under the prompt root {root_dir()})")
    return path.read_text(encoding="utf-8")


def prompt_path(subdir: str, name: str) -> Path:
    """Absolute path of a prompt file (for tooling / docs)."""
    return root_dir() / subdir / name


def reset_cache() -> None:
    load_prompt.cache_clear()
