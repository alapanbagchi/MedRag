# registry.py

from typing import Any, Dict, List


class PluginNotFoundError(KeyError):
    """Raised when a requested plugin is not registered."""
    pass


class PluginRegistry:
    """Keeps track of the plugins as a key value pair."""

    def __init__(self):
        self.plugins: Dict[str, Any] = {}

    def register(self, name: str, plugin: Any) -> None:
        """Register a plugin instance under a specific name."""
        self.plugins[name] = plugin

    def get(self, name: str) -> Any:
        """Retrieve a plugin by name."""
        if name not in self.plugins:
            raise PluginNotFoundError(f"Plugin '{name}' is not registered. Available: {self.list()}")
        return self.plugins[name]

    def list(self) -> List[str]:
        """Return a list of all registered plugin names."""
        return list(self.plugins.keys())

    def __contains__(self, name: str) -> bool:
        """Allow using `if 'name' in registry:` syntax."""
        return name in self.plugins
