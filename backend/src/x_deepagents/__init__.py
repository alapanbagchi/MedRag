"""x_deepagents - standalone autonomous retrieval system.

Lives INSIDE backend/src so it shares the backend venv, tools, config and
prompts directly (imports are plain "from src.tools..."). Built on deepagents
+ LangGraph as the candidate *main* retrieval system, isolated here for
testing. All agents, state, graph and rules are new; only prompts + non-LLM
tools are reused (see x_deepagents.reuse).
"""

__version__ = "0.1.0"
