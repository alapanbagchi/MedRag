"""Manual driver: decompose + enrich + hybrid-retrieve (Step 3) the TEST QUERY."""
import asyncio

from src.agentic.planner import DecomposePlanner
from src.agentic.umls_tool import UMLSEnricher
from src.agentic.retriever_tool import HybridRetrieverTool

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
    enricher = UMLSEnricher()
    retriever = HybridRetrieverTool()

    decomposition = await planner.plan(TEST_QUERY)
    print(f"=== decomposed into {len(decomposition.subqueries)} subqueries ===\n")

    for sub in decomposition.subqueries:
        terms = await enricher.enrich_subquery(sub)
        expanded = enricher.apply_to_query(sub, terms)
        print(f"{sub.id}: {sub.target!r}")
        print(f"   enriched query: {expanded!r}")
        results = await retriever.search(sub, top_k=4)
        print(f"   -> {len(results)} restored paragraphs:")
        for r in results:
            head = " ".join(r.paragraph_text.split())[:140]
            print(f"      #{r.rank} {r.document_id}/{r.chunk_id} [{r.unit_kind}] "
                  f"score={r.rrf_score:.3f}")
            print(f"         {head}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
