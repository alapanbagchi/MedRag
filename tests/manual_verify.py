"""Manual driver: retrieve + verify (Step 3+4) one subquery of the TEST QUERY."""
import asyncio

from src.agentic.planner import DecomposePlanner
from src.agentic.umls_tool import UMLSEnricher
from src.agentic.retriever_tool import HybridRetrieverTool
from src.agentic.verify import VerifyTool

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


def clip(text, n=90):
    return " ".join(text.split())[:n]


async def main() -> None:
    planner = DecomposePlanner()
    enricher = UMLSEnricher()
    retriever = HybridRetrieverTool()
    verifier = VerifyTool()

    decomposition = await planner.plan(TEST_QUERY)
    print(f"decomposed into {len(decomposition.subqueries)} subqueries\n")

    # run retrieve+verify on the FIRST subquery only (keeps LLM cost bounded)
    sub = decomposition.subqueries[0]
    terms = await enricher.enrich_subquery(sub)
    expanded = enricher.apply_to_query(sub, terms)
    sub.enriched_query = expanded

    print(f"SUBQUERY {sub.id}: {sub.target!r}")
    print(f"  enriched query: {expanded!r}")
    print(f"  intent: {sub.intent!r}")
    print(f"  evidence_required: {sub.evidence_required}\n")

    results = await retriever.search(sub, top_k=6)
    print(f"  retrieved {len(results)} restored units; verifying...\n")

    outcome = await verifier.verify(sub, results, base_query=TEST_QUERY)
    print(f"  KEPT ({len(outcome.kept)}):")
    for u in outcome.kept:
        print(f"    [K] {u.document_id}/{u.chunk_id} [{u.unit_kind}] conf={u.confidence:.2f} :: {clip(u.paragraph_text)}")
    print(f"  REJECTED ({len(outcome.rejected)}):")
    for u in outcome.rejected:
        print(f"    [R] {u.document_id}/{u.chunk_id} reason={u.reason[:70]!r} :: {clip(u.paragraph_text)}")
    print(f"  UNKNOWN ({len(outcome.unknown)}):")
    for u in outcome.unknown:
        print(f"    [?] {u.document_id}/{u.chunk_id} reason={u.reason[:70]!r}")
    print(f"  distinct papers kept: {outcome.distinct_papers}")


if __name__ == "__main__":
    asyncio.run(main())
