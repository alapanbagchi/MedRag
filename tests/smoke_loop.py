"""Smoke test: build the agentic loop agent and verify tool registration."""
import asyncio

from src.config import AppConfig
from src.agentic.planner import SubQueryPlan
from src.agentic.retriever_tool import HybridRetrieverTool
from src.agentic.verify import VerifyTool
from src.agentic.loop import AgenticLoop


async def main() -> None:
    cfg = AppConfig()
    loop = AgenticLoop(config=cfg)
    agent = loop._build_agent()
    print("agent name:", agent.name)
    print("deps_type:", getattr(agent, "deps_type", None))
    # list registered tools
    tools = agent._function_tools if hasattr(agent, "_function_tools") else None
    if tools is None:
        tools = getattr(agent, "_dynamic_tools", None)
        print("_dynamic_tools:", tools)
        tools = getattr(agent, "_function_tools", None)
    # try the modern accessor
    try:
        from pydantic_ai import Tool
        names = [t.name for t in agent._function_tools.values()] if hasattr(agent, "_function_tools") and hasattr(agent._function_tools, "values") else []
        print("tool names:", names)
    except Exception as e:
        print("tool introspection error:", e)
    print("build OK")


if __name__ == "__main__":
    asyncio.run(main())
