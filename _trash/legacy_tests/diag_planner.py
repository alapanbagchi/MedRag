"""Diagnose the planner: capture raw MedGemma output for the TEST QUERY."""
import asyncio
import json

from src.config import AppConfig
from src.agentic.planner import DecomposePlanner, PLANNER_SYSTEM_PROMPT

TEST_QUERY = (
    "Compare the pharmacological and imaging strategies used to prevent and "
    "manage radial artery vasospasm when the artery is used as an access site "
    "for interventional radiology procedures (such as PAE) versus when it is "
    "harvested as a conduit for coronary artery bypass grafting (CABG). "
    "Furthermore, if a patient undergoing CABG is at high risk for "
    "post-operative acute kidney injury (AKI), explain how the KID-ACS score "
    "utilizes proenkephalin (PENK) to improve early risk stratification "
    "compared to traditional serum creatinine monitoring"
)


async def main() -> None:
    cfg = AppConfig()
    planner = DecomposePlanner(config=cfg)

    # 1) first capture the RAW text output of the model (what medgemma returns)
    from src.llm import build_model
    model = build_model(cfg)
    agent = planner.agent

    print("=== RAW MODEL OUTPUT (structured path) ===")
    try:
        result = await agent.run(TEST_QUERY, output_type=str, model_settings={"max_tokens": cfg.agent_max_tokens})
        raw = str(result.output)
        print(repr(raw[:3000]))
    except Exception as exc:
        print("structured text run failed:", repr(exc))

    print()
    print("=== DecomposePlanner.plan() result ===")
    try:
        d = await planner.plan(TEST_QUERY)
        print(json.dumps(d.model_dump(), indent=2))
    except Exception as exc:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
