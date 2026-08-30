"""Manual driver: decompose (Step 1) then UMLS-enrich (Step 2) the TEST QUERY."""
import asyncio
import json

from src.agentic.planner import DecomposePlanner
from src.agentic.umls_tool import UMLSEnricher

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
    decomposition = await planner.plan(TEST_QUERY)
    enricher = UMLSEnricher()

    print("=== DECOMPOSITION ===")
    print("question_type:", decomposition.question_type, "| subqueries:", len(decomposition.subqueries))
    print()
    for sub in decomposition.subqueries:
        terms = await enricher.enrich_subquery(sub)
        expanded = enricher.apply_to_query(sub, terms)
        sub.enriched_query = expanded
        print(f"{sub.id}: {sub.target!r}")
        print(f"   entities       : {[e.text for e in sub.entities]}")
        print(f"   base query     : {sub.query!r}")
        print(f"   UMLS enabled   : {enricher.enabled}")
        for t in terms:
            print(f"     • {t.surface_form!r} -> preferred={t.preferred_name!r} cui={t.cui} synonyms={t.synonyms}")
        print(f"   ENRICHED QUERY : {expanded!r}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
