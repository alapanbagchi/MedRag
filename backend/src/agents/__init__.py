"""Agentic v3 - LLM agents only.

Each module here defines one agent role (an LLM-driven actor):

  master.py         MasterOrchestratorAgent - decompose the query into tasks
  search.py         SearchTermPlanner       - search-term selection per round
  replan.py         ReplannerAgent          - failure analysis + replanning
  critic.py         CriticAgent             - the verification gate
  deepinspect.py    DeepInspector           - deep paper inspection
  contradiction.py  ContradictionAgent      - cross-evidence analysis
  resolution.py     ResolutionAgent         - resolves contradictions
  synthesize.py     FinalSynthesizer        - evidence-gated final answer
  worker.py         WorkerAgent             - the worker sub-orchestrator

Non-agent code lives elsewhere: the run pipeline, domain state, events and
worker pipelines in src.agentic; reusable non-LLM capabilities in src.tools.
"""

from src.agents.contradiction import ContradictionAgent
from src.agents.critic import CriticAgent
from src.agents.deepinspect import DeepInspector, verify_quote
from src.agents.master import MasterOrchestratorAgent
from src.agents.replan import ReplannerAgent, meaningfully_different
from src.agents.resolution import ResolutionAgent
from src.agents.search import SearchTermPlanner
from src.agents.synthesize import FinalSynthesizer
from src.agents.worker import WorkerAgent

__all__ = [
    "ContradictionAgent", "CriticAgent", "DeepInspector",
    "FinalSynthesizer", "MasterOrchestratorAgent", "ReplannerAgent",
    "ResolutionAgent", "SearchTermPlanner", "WorkerAgent",
    "meaningfully_different", "verify_quote",
]
