# Keeps track of the plugins as a key value pair

class PluginRegistry:
    def __init__(self):
        self.plugins = {}
    def register(self, name: str, plugin):
        self.plugins[name] = plugin
    def get(self, name: str):
        return self.plugins[name]
    def list(self):
        return list(self.plugins.keys())
