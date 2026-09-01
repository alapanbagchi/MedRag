"""src.tools canonical modules: registered, importable, no stale duplicates.

The non-LLM capabilities (umls, terminology, retrieval, paper_retriever)
live ONLY in src.tools; src.agents holds LLM agents + orchestration.
"""

import importlib
import importlib.util

import pytest

import src.tools.paper_retriever  # noqa: F401  (registers on import)
import src.tools.retrieval  # noqa: F401
import src.tools.terminology  # noqa: F401
import src.tools.umls  # noqa: F401


@pytest.mark.parametrize("mod", [
    "src.tools.umls",
    "src.tools.terminology",
    "src.tools.retrieval",
    "src.tools.paper_retriever",
])
def test_tool_modules_importable(mod):
    importlib.import_module(mod)


def test_tools_registered():
    from src.tools import all_tools

    names = set(all_tools())
    assert {"umls_enricher", "terminology_enricher",
            "hybrid_retriever", "paper_retriever"} <= names


def test_no_tools_left_in_agents():
    """Moved modules must not be re-created under src.agents."""
    for stale in ("src.agents.umls_enricher", "src.agents.umls",
                  "src.agents.search_engine", "src.agents.retriever"):
        assert importlib.util.find_spec(stale) is None, (
            f"{stale} must live in src.tools")
