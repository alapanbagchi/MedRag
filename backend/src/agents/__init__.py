"""MedRag agentic runtime: the singular deepagents + LangGraph pipeline.

Query -> orchestrator (decompose) -> parallel research workers ->
conflict -> resolution -> gap resolution -> evidence-gated synthesis.
Only prompts + non-LLM tools are reused from the shared backend
(see agents.reuse); agents, state, graph, rules and budgets live here.
"""

__version__ = "1.0.0"
