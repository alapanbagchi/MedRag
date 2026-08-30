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
    ("src.agentic.planner", "PLANNER_SYSTEM_PROMPT", "agentic_v1/planner.txt"),
    ("src.agentic.loop", "LOOP_SYSTEM_PROMPT", "agentic_v1/loop.txt"),
    ("src.agentic_v2.orchestrator", "ORCHESTRATOR_SYSTEM_PROMPT", "agentic_v2/orchestrator.txt"),
    ("src.agentic_v2.verify", "VERIFIER_SYSTEM_PROMPT", "agentic_v2/verifier.txt"),
    ("src.agentic_v2.synthesize", "SYNTHESIS_SYSTEM_PROMPT", "agentic_v2/synthesize.txt"),
    ("src.agentic_v3.master", "MASTER_SYSTEM_PROMPT", "agentic_v3/master.txt"),
    ("src.agentic_v3.search", "SEARCH_PLANNER_SYSTEM_PROMPT", "agentic_v3/search_planner.txt"),
    ("src.agentic_v3.critic", "CRITIC_SYSTEM_PROMPT", "agentic_v3/critic.txt"),
    ("src.agentic_v3.deepinspect", "INSPECTOR_SYSTEM_PROMPT", "agentic_v3/deep_inspector.txt"),
    ("src.agentic_v3.replan", "REPLANNER_SYSTEM_PROMPT", "agentic_v3/replanner.txt"),
    ("src.agentic_v3.contradiction", "CONTRADICTION_SYSTEM_PROMPT", "agentic_v3/contradiction.txt"),
    ("src.agentic_v3.resolution", "RESOLUTION_SYSTEM_PROMPT", "agentic_v3/resolution.txt"),
    ("src.agentic_v3.synthesize", "SYNTHESIS_SYSTEM_PROMPT", "agentic_v3/synthesize.txt"),
]

# legacy pipeline prompts are read as files too (no module constant)
LEGACY_FILES = [
    "legacy/evidence.txt", "legacy/planner.txt", "legacy/rewriter.txt",
    "legacy/synthesizer.txt", "legacy/verifier.txt",
]


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
        load_prompt("agentic_v3", "does_not_exist.txt")
    assert "does_not_exist.txt" in str(excinfo.value)
    assert str(root_dir()) in str(excinfo.value)
    reset_cache()


def test_prompt_dir_override_swaps_prompts(tmp_path, monkeypatch):
    custom = tmp_path / "custom_prompts" / "agentic_v3"
    custom.mkdir(parents=True)
    (custom / "master.txt").write_text(
        "You are a CUSTOM master prompt.", encoding="utf-8")

    monkeypatch.setenv("PROMPT_DIR", str(tmp_path / "custom_prompts"))
    reset_cache()
    try:
        assert load_prompt("agentic_v3", "master.txt") ==             "You are a CUSTOM master prompt."
    finally:
        reset_cache()

    # after restoring the env the default tree is used again
    monkeypatch.delenv("PROMPT_DIR")
    reset_cache()
    loaded = load_prompt("agentic_v3", "master.txt")
    assert "CUSTOM" not in loaded
