"""Prompt tree tests.

Every agent system prompt lives as a plain-text file under src/prompts/
(one file per agent), loaded at call time via src.prompts.load.load_prompt -
so editing a .txt file changes behavior without code changes.

Also verifies the PROMPT_DIR env override (custom prompt tree) and the
loud file-not-found error.
"""

from pathlib import Path

import pytest

from src.prompts.load import load_prompt, reset_cache, root_dir

ROOT = Path(__file__).resolve().parent.parent / "src" / "prompts"

# every prompt file the singular agents flow loads (stages/gap_fill agents)
PROMPTS = [
    "agents/master.txt",
    "agents/search_planner.txt",
    "agents/replanner.txt",
    "agents/critic.txt",
    "agents/deep_inspector.txt",
    "agents/contradiction.txt",
    "agents/resolution.txt",
    "agents/synthesize.txt",
    "agents/reliability.txt",
    "agents/gap_probe.txt",
    "agents/gap_complete.txt",
]

# legacy pipeline prompts are read as files too (no module constant)
LEGACY_FILES: list = []


def test_all_prompt_files_exist_and_are_nonempty():
    for rel in LEGACY_FILES + PROMPTS:
        path = ROOT / rel
        assert path.is_file(), f"missing prompt file: {path}"
        assert path.read_text(encoding="utf-8").strip(), f"empty prompt: {path}"


def test_prompt_constants_load_from_files():
    for rel in PROMPTS:
        sub, name = rel.split("/")
        assert load_prompt(sub, name).strip(), f"empty prompt: {rel}"


def test_legacy_prompt_files_load():
    for rel in LEGACY_FILES:
        sub, name = rel.split("/")
        assert load_prompt(sub, name).strip()


def test_missing_prompt_file_raises_with_path():
    reset_cache()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_prompt("agents", "does_not_exist.txt")
    assert "does_not_exist.txt" in str(excinfo.value)
    assert str(root_dir()) in str(excinfo.value)
    reset_cache()


def test_prompt_dir_override_swaps_prompts(tmp_path, monkeypatch):
    custom = tmp_path / "custom_prompts" / "agents"
    custom.mkdir(parents=True)
    (custom / "master.txt").write_text(
        "You are a CUSTOM master prompt.", encoding="utf-8")

    monkeypatch.setenv("PROMPT_DIR", str(tmp_path / "custom_prompts"))
    reset_cache()
    try:
        assert load_prompt("agents", "master.txt") ==             "You are a CUSTOM master prompt."
    finally:
        reset_cache()

    # after restoring the env the default tree is used again
    monkeypatch.delenv("PROMPT_DIR")
    reset_cache()
    loaded = load_prompt("agents", "master.txt")
    assert "CUSTOM" not in loaded
