"""Shared LLM adapter: the OpenCode (Muse Spark) LangChain chat model.

The singular agents flow builds all its models through src/agents/config.py;
this package holds the one non-standard adapter (OpenCode speaks the
Responses API, not Chat Completions).
"""

from src.llm.opencode_client import OpenCodeChatModel, build_opencode_model

__all__ = ["OpenCodeChatModel", "build_opencode_model"]
