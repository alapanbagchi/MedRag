from typing import Protocol

class Collector(Protocol):
    def collect(self):
        ...
