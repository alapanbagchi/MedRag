"""Full end-to-end agentic pipeline driver (query hardcoded to avoid shell quoting)."""
import asyncio
import json

from src.agentic.pipeline import AgenticPipeline

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
    pipeline = AgenticPipeline()
    result = await pipeline.answer(TEST_QUERY)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
