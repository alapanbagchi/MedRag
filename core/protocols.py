from pathlib import Path
from typing import Protocol

class Collector(Protocol):
    def collect(self):
        ...

class Parser(Protocol):
    def parse(self, source: Path):
        ...