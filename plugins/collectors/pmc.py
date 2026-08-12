from typing import Protocol

from core.protocols import Collector


class PMCCollector(Collector):
    def __init__(self, topic: str):
        self.topic = topic
    def collect(self):
        print("PMC ARTICLE COLLECTIION HAPPENS HERE")
