"""Prompt tree tests.

Every agent system prompt lives as a plain-text file under src/prompts/
(one file per agent), and every prompt constant loads from those files - so
editing a .txt file changes behavior without code changes.

Also verifies the PROMPT_DIR env override (custom prompt tree) and the
loud file-not-found error.
"""

import importlib
from pathlib import Path

import pytest

from src.prompts.load import load_prompt, reset_cache, root_dir

ROOT = Path(__file__).resolve().parent.parent / "src" / "prompts"

# (constant-holding module, constant name, relative prompt file)
CONSTANTS = [
    ("src.agents.master", "MASTER_SYSTEM_PROMPT", "agents/master.txt"),
    ("src.agents.search", "SEARCH_PLANNER_SYSTEM_PROMPT", "agents/search_planner.txt"),
    ("src.agents.critic", "CRITIC_SYSTEM_PROMPT", "agents/critic.txt"),
    ("src.agents.deepinspect", "INSPECTOR_SYSTEM_PROMPT", "agents/deep_inspector.txt"),
    ("src.agents.replan", "REPLANNER_SYSTEM_PROMPT", "agents/replanner.txt"),
    ("src.agents.contradiction", "CONTRADICTION_SYSTEM_PROMPT", "agents/contradiction.txt"),
    ("src.agents.resolution", "RESOLUTION_SYSTEM_PROMPT", "agents/resolution.txt"),
    ("src.agents.synthesize", "SYNTHESIS_SYSTEM_PROMPT", "agents/synthesize.txt"),
]

# legacy pipeline prompts are read as files too (no module constant)
LEGACY_FILES: list = []


def test_all_prompt_files_exist_and_are_nonempty():
    for rel in LEGACY_FILES + [c[2] for c in CONSTANTS]:
        path = ROOT / rel
        assert path.is_file(), f"missing prompt file: {path}"
        assert path.read_text(encoding="utf-8").strip(), f"empty prompt: {path}"


def test_prompt_constants_load_from_files():
    for mod, const, rel in CONSTANTS:
        module = importlib.import_module(mod)
        loaded = load_prompt(rel.split("/")[0], rel.split("/")[1])
        assert getattr(module, const) == loaded, f"{mod}.{const} != {rel}"
        assert getattr(module, const).strip(), f"{mod}.{const} empty"


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
