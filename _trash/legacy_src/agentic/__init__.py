"""Agentic retrieval pipeline.

A fresh, LLM-driven retrieval design where the agent has authority over its
tools (hybrid retrieval + UMLS) and loops autonomously until it finds
sufficient evidence or gives up.

Implemented incrementally:

  Step 1  src/agentic/planner.py        robust query decomposition
  Step 2  src/agentic/umls_tool.py      per-subquery entity enrichment
  Step 3  src/agentic/retriever_tool.py hybrid retrieval + rerank tool
  Step 5  src/agentic/loop.py           agentic tool loop (LLM drives tools)
"""
