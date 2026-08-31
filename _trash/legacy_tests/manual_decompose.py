"""Manual driver: run the decompose planner against the TEST QUERY."""
import asyncio
import json

from src.agentic.planner import DecomposePlanner

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
    planner = DecomposePlanner()
    d = await planner.plan(TEST_QUERY)
    print(json.dumps(d.model_dump(), indent=2))
    print("\n--- SUMMARY ---")
    print("question_type:", d.question_type)
    print("num_subqueries:", len(d.subqueries))
    for s in d.subqueries:
        print(f"  {s.id}: target={s.target!r}")
        print(f"       intent={s.intent!r}")
        print(f"       query={s.query!r}")
        print(f"       focus={s.focus!r} evidence_required={s.evidence_required}")
        print(f"       entities={[(e.text, e.role) for e in s.entities]}")


if __name__ == "__main__":
    asyncio.run(main())
