"""Tool registry: the table the deep agent may select from (autonomy B).

Wires the real reuse adapters (retrieve / umls_lookup) plus the PRIMARY
pg-chunk source (postgres_search over medrag.chunks). The registry keeps
"which tool the agent may call" explicit and testable, and is the single place
a capability is flipped on/off without touching the graph or rules.
"""

from __future__ import annotations

from typing import Any

_TOOLS: dict[str, dict] = {}


def register(name: str, tool=None, *, enabled: bool = True, future: bool = False) -> None:
    """Declare a selectable tool.

    future=True marks a capability that is KNOWN (the agent may choose it)
    but not yet implemented (enabled=False) - e.g. postgres_search.
    """
    _TOOLS[name] = {
        "name": name,
        "tool": tool,
        "enabled": enabled,
        "future": future,
    }


def get(name: str) -> dict | None:
    return _TOOLS.get(name)


def all_tools() -> dict[str, dict]:
    return dict(_TOOLS)


def selectable() -> list[str]:
    """Tools the agent may actually call today (enabled + implemented)."""
    return [n for n, t in _TOOLS.items() if t["enabled"]]


def tools_for_agent() -> list[Any]:
    """The LangChain tool objects handed to create_deep_agent."""
    return [t["tool"] for n, t in _TOOLS.items() if t["enabled"] and t["tool"] is not None]


# --- register the reusable tool adapters ----------------------------------

def _register_all() -> None:
    from src.agents.tools.retrieve import retrieve
    from src.agents.tools.searxng import searxng_search
    from src.agents.tools.umls import umls_lookup
    # PRIMARY pg-chunk source: enabled by default (real implementation)
    from src.agents.tools.postgres_search import postgres_search

    register("retrieve", tool=retrieve, enabled=True)
    register("searxng_search", tool=searxng_search, enabled=True)
    register("umls_lookup", tool=umls_lookup, enabled=True)
    register("postgres_search", tool=postgres_search, enabled=True, future=False)


_register_all()
