"""Manual driver: agentic loop with explicit model override (bypass .env)."""
import asyncio

from src.config import AppConfig
from src.agentic.planner import DecomposePlanner
from src.agentic.umls_tool import UMLSEnricher
from src.agentic.loop import AgenticLoop

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


def make_config() -> AppConfig:
    cfg = AppConfig()
    # .env currently points at an invalid model (medgemma-27b-it -> 404);
    # override to the proven-working model for this test only.
    cfg.gemini_model = "gemma-4-31b-it"
    return cfg


async def main() -> None:
    cfg = make_config()
    planner = DecomposePlanner(config=cfg)
    enricher = UMLSEnricher(config=cfg)
    loop = AgenticLoop(config=cfg, umls_enricher=enricher)

    decomposition = await planner.plan(TEST_QUERY)
    print(f"decomposed into {len(decomposition.subqueries)} subqueries\n")

    target_sub = None
    for s in decomposition.subqueries:
        blob = (s.target + " " + s.query).lower()
        if "kid-acs" in blob or "proenkephalin" in blob or "penk" in blob or "creatinine" in blob:
            target_sub = s
            break
    if target_sub is None:
        target_sub = decomposition.subqueries[-1]

    print(f"=== AGENTIC LOOP for {target_sub.id}: {target_sub.target!r} ===")
    print(f"    query={target_sub.query!r}\n")
    report = await loop.run(target_sub, base_query=TEST_QUERY)

    print("--- EVIDENCE REPORT ---")
    print("subquery_id:", report.subquery_id)
    print("succeeded:", report.succeeded)
    print("summary:", report.summary)
    print("searches performed:")
    for q in report.searches_performed:
        print("   *", q)
    print("evidence excerpts:")
    for e in report.evidence_excerpts:
        print("   -", " ".join(e.split())[:160])
    print("citations:", report.citations)
    if report.notes:
        print("notes:", report.notes)


if __name__ == "__main__":
    asyncio.run(main())
