"""Quick driver: run agentic pipeline with trace stream open, verify logs.txt."""
import asyncio
import os

os.environ["LOG_FILE"] = "logs_agentic_demo.txt"

from src.trace import get_trace
from src.agentic.pipeline import AgenticPipeline

TEST_QUERY = (
    "What pharmacological agents prevent radial artery vasospasm during "
    "transradial access for interventional procedures?"
)


async def main() -> None:
    trace = get_trace()
    trace.open_stream(os.environ["LOG_FILE"], query=TEST_QUERY)
    try:
        pipeline = AgenticPipeline()
        result = await pipeline.answer(TEST_QUERY)
        print("succeeded_subqueries:", result["succeeded_subqueries"], "/", result["num_subqueries"])
    finally:
        trace.close_stream()


if __name__ == "__main__":
    asyncio.run(main())
