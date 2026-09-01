"""Tool registry: tools self-register on import and are looked up by name."""

from __future__ import annotations

from typing import Any

_TOOLS: dict[str, type] = {}


def register(name: str):
    """Class decorator: register a tool class under ``name``."""

    def deco(cls: type) -> type:
        _TOOLS[name] = cls
        return cls

    return deco


def get(name: str) -> Any:
    """Return the registered tool class for ``name``, or None."""
    return _TOOLS.get(name)


def all_tools() -> dict[str, type]:
    """All registered tools, as ``{name: class}``."""
    return dict(_TOOLS)
