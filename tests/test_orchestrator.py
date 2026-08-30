"""Orchestrator tests: batched verification, unknown-retry semantics,
distinct-paper stop conditions, warnings, funnel metrics, balanced synthesis.

All LLM behaviour is injected via fakes — no network, no models.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

import pandas as pd

from src.agents.evidence import Evidence, EvidenceAggregator
from src.agents.planner import ClinicalEntity, EnrichedPlan, QueryPlan, SubQuery
from src.agents.synthesizer import FinalAnswer
from src.agents.verifier_new import RelevanceVerdict, VerificationResult
from src.config import AppConfig
from src.orchestration.orchestrator import Orchestrator
from src.retrieval.retriever import RetrievedDocument
from src.umls.client import UMLSClient

QUERY = "Which repair techniques were associated with recurrent coarctation?"

DOC_TEXT = (
    "End-to-end anastomosis was associated with recurrent coarctation in "
    "23.1% of patients (p = 0.04). Patch aortoplasty showed 8.3% recurrence."
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePlanner:
    def __init__(self, subqueries):
        self._subqueries = subqueries

    async def plan(self, query: str) -> EnrichedPlan:
        plan = QueryPlan(original_query=query, question_type="association",
                         subqueries=self._subqueries)
        return EnrichedPlan(plan)


class FakeRetrievalService:
    """Returns scripted chunks per query round; honors exclusion lists."""

    def __init__(self, rounds: Dict[str, List[RetrievedDocument]]):
        # rounds maps "query_text" -> docs; unlisted queries -> first round's docs
        self.rounds = rounds
        self.calls: List[tuple] = []

    def _components(self):
        return {"corpus": FakeCorpus()}

    async def search_subquery(self, sub, plan=None, exclude_chunk_ids=None):
        self.calls.append((sub.query, tuple(exclude_chunk_ids or ())))
        docs = self.rounds.get(sub.query) or next(iter(self.rounds.values()))
        return [d for d in docs if d.chunk_id not in set(exclude_chunk_ids or ())]


class FakeCorpus:
    _df = pd.DataFrame([
        {"id": "c1", "document_id": "PMC-A", "text": DOC_TEXT, "section": "Results",
         "table_id": None, "figure_id": None, "document_position": 1},
        {"id": "c2", "document_id": "PMC-A", "text": DOC_TEXT, "section": "Results",
         "table_id": None, "figure_id": None, "document_position": 2},
        {"id": "c3", "document_id": "PMC-B", "text": DOC_TEXT, "section": "Results",
         "table_id": None, "figure_id": None, "document_position": 1},
        {"id": "c9", "document_id": "PMC-D", "text": DOC_TEXT, "section": "Results",
         "table_id": None, "figure_id": None, "document_position": 1},
    ])

    def document_id(self, chunk_id):
        row = self._df.loc[self._df["id"] == chunk_id]
        return None if row.empty else str(row.iloc[0]["document_id"])

    def resolve(self, chunk_ids, include_text=False):
        out = {}
        for cid in chunk_ids:
            row = self._df.loc[self._df["id"] == cid]
            if not row.empty:
                r = row.iloc[0]
                out[cid] = {"document_id": r["document_id"], "chunk_type": "paragraph",
                            "breadcrumb": "", **({"text": r["text"]} if include_text else {})}
        return out


def _doc(chunk_id: str, document_id: str) -> RetrievedDocument:
    return RetrievedDocument(
        subquery_id="H1", document_id=document_id, chunk_id=chunk_id,
        rank=1, rrf_score=0.9, methods=["bm25"], node_type="paragraph",
        section="Results", breadcrumb=["Results"], text=DOC_TEXT,
        token_count=len(DOC_TEXT.split()),
    )


class ScriptedVerifier:
    """verify_papers receives BATCHES; responses keyed by call index."""

    def __init__(self, results_by_call: List[VerificationResult]):
        self.results_by_call = results_by_call
        self.batch_sizes: List[int] = []
        self.queries: List[str] = []

    async def verify_papers(self, papers, query, evidence_required):
        self.batch_sizes.append(len(papers))
        self.queries.append(query)
        idx = min(len(self.batch_sizes) - 1, len(self.results_by_call) - 1)
        base = self.results_by_call[idx]
        # answer only for the ids actually asked about this call
        wanted = {p["document_id"] for p in papers}
        results = [v for v in base.results if v.document_id in wanted] or [
            RelevanceVerdict(document_id=next(iter(wanted), ""), relevance="not_relevant")
        ]
        return VerificationResult(results=results, overall=base.overall)


class FakeExtractor:
    def __init__(self):
        self.units: List[str] = []

    async def extract_with_stats(self, subquery, document):
        self.units.append(document.chunk_id)
        ev = Evidence(
            subquery_id=subquery.id, document_id=document.document_id,
            chunk_id=document.chunk_id, source=document.document_id,
            claim=f"claim for {document.chunk_id}", supporting_text="verbatim text",
            supports_claim=True, confidence=0.9, evidence_id=None,
        )
        return [ev], {"raw_items": 1, "kept_items": 1, "quotes_dropped": 0}


class FakeSynthesizer:
    def __init__(self):
        self.seen_evidence = None
        self.seen_notes = None

    async def synthesize(self, query, plan, report, evidence, coverage_notes=None):
        self.seen_evidence = list(evidence)
        self.seen_notes = list(coverage_notes or [])
        return FinalAnswer(summary="synthetic answer")


def _sub(id_: str = "H1", target: str = "repair techniques") -> SubQuery:
    return SubQuery(id=id_, target=target, focus="evidence",
                    query="repair techniques recurrent coarctation percentages",
                    evidence_required=["percentages"])


def _orchestrator(cfg_overrides=None, *, planner, verifier, extractor, synthesizer,
                  retrieval_service, umls_client=None) -> Orchestrator:
    cfg = AppConfig()
    cfg.max_query_rounds = 3
    cfg.min_papers = 1  # most tests: single relevant paper suffices
    cfg.verify_batch_max_docs = 6
    cfg.rewrite_enabled = False  # deterministic path by default in tests
    for k, v in (cfg_overrides or {}).items():
        setattr(cfg, k, v)
    return Orchestrator(
        config=cfg,
        planner=planner,
        verifier=verifier,
        extractor=extractor,
        aggregator=EvidenceAggregator(),
        synthesizer=synthesizer,
        retrieval_service=retrieval_service,
        umls_client=umls_client or UMLSClient(api_key=""),  # disabled by default
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_end_to_end_with_fakes():
    verifier = ScriptedVerifier([VerificationResult(results=[
        RelevanceVerdict(document_id="c1", relevance="relevant", confidence=0.9, reason="has data"),
    ])])
    extractor, synth = FakeExtractor(), FakeSynthesizer()
    retrieval = FakeRetrievalService({"q": [_doc("c1", "PMC-A")]})
    orch = _orchestrator(planner=FakePlanner([_sub()]), verifier=verifier,
                         extractor=extractor, synthesizer=synth, retrieval_service=retrieval)

    result = await orch.answer(QUERY)

    assert result["plan"].original_query == QUERY
    assert result["papers"]["H1"], "one relevant unit expected"
    assert all(e.evidence_id for e in result["evidence"])
    assert result["groups"]
    assert result["answer"].summary == "synthetic answer"
    assert set(result["timings"]) >= {"plan_ms", "retrieve_ms", "extract_ms", "synthesize_ms"}
    funnel = result["funnel"]
    assert funnel["units_verified"] == 1
    assert funnel["units_relevant"] == 1
    assert funnel["evidence_kept"] == 1
    assert funnel["evidence_synthesized"] == 1
    assert result["warnings"] == []


async def test_verifier_receives_one_batch_not_per_unit_calls():
    """The old flow fired one LLM call PER structural unit; now it's batched."""
    verifier = ScriptedVerifier([VerificationResult(results=[
        RelevanceVerdict(document_id=f"c{i}", relevance="relevant", confidence=0.9)
        for i in (1, 2, 3)
    ])])
    retrieval = FakeRetrievalService({"q": [_doc("c1", "PMC-A"), _doc("c2", "PMC-A"), _doc("c3", "PMC-B")]})
    orch = _orchestrator(planner=FakePlanner([_sub()]), verifier=verifier,
                         extractor=FakeExtractor(), synthesizer=FakeSynthesizer(),
                         retrieval_service=retrieval)
    await orch.answer(QUERY)
    assert len(verifier.batch_sizes) == 1, f"expected 1 batched call, got {verifier.batch_sizes}"
    assert verifier.batch_sizes[0] == 3


async def test_unknown_verdict_is_retried_not_dropped():
    """Round 1 says 'unknown' (quota failure): the unit must be retried on
    round 2 and KEPT when it then verifies — never silently excluded."""
    round1 = VerificationResult(results=[RelevanceVerdict(
        document_id="c1", relevance="unknown", reason="429 quota")])
    round2 = VerificationResult(results=[RelevanceVerdict(
        document_id="c1", relevance="relevant", confidence=0.9, reason="ok")])
    verifier = ScriptedVerifier([round1, round2])
    retrieval = FakeRetrievalService({
        "repair techniques recurrent coarctation percentages": [_doc("c1", "PMC-A")],
        "rewritten q": [_doc("c1", "PMC-A")],
    })
    orch = _orchestrator(planner=FakePlanner([_sub()]), verifier=verifier,
                         extractor=FakeExtractor(), synthesizer=FakeSynthesizer(),
                         retrieval_service=retrieval)
    result = await orch.answer(QUERY)

    assert len(verifier.batch_sizes) == 2, "unit must be re-verified on round 2"
    assert result["papers"]["H1"], "retried unit must be kept"
    assert result["funnel"]["units_unknown"] == 1
    assert result["funnel"]["units_relevant"] == 1


async def test_min_papers_counts_distinct_papers():
    """Two units from the SAME paper satisfy units but not MIN_PAPERS=2."""
    verifier = ScriptedVerifier([
        VerificationResult(results=[  # round 1: both units same paper, relevant
            RelevanceVerdict(document_id="c1", relevance="relevant"),
            RelevanceVerdict(document_id="c2", relevance="relevant"),
        ]),
        VerificationResult(results=[  # round 2: new paper appears
            RelevanceVerdict(document_id="c3", relevance="relevant"),
        ]),
    ])
    retrieval = FakeRetrievalService({
        "repair techniques recurrent coarctation percentages": [_doc("c1", "PMC-A"), _doc("c2", "PMC-A")],
        "repair techniques recurrent coarctation percentages new outcome evidence": [_doc("c3", "PMC-B")],
    })
    extractor, synth = FakeExtractor(), FakeSynthesizer()
    sub = _sub()
    sub.synonyms = ["new outcome evidence"]  # feeds the deterministic rewrite
    orch = _orchestrator({"min_papers": 2}, planner=FakePlanner([sub]), verifier=verifier,
                         extractor=extractor, synthesizer=synth, retrieval_service=retrieval)
    result = await orch.answer(QUERY)

    assert len(verifier.batch_sizes) == 2, "must keep searching until DISTINCT papers reach min"
    papers = {u["document_id"] for u in result["papers"]["H1"]}
    assert papers == {"PMC-A", "PMC-B"}


async def test_persistent_unknowns_surface_in_warnings():
    v_unknown = VerificationResult(results=[RelevanceVerdict(
        document_id="c1", relevance="unknown", reason="quota exhausted")])
    verifier = ScriptedVerifier([v_unknown])  # always returns this (clamped)
    extraction, synth = FakeExtractor(), FakeSynthesizer()
    retrieval = FakeRetrievalService({
        "repair techniques recurrent coarctation percentages": [_doc("c1", "PMC-A")],
        "rw1": [_doc("c1", "PMC-A")], "rw2": [_doc("c1", "PMC-A")],
    })
    orch = _orchestrator(planner=FakePlanner([_sub()]), verifier=verifier,
                         extractor=extraction, synthesizer=synth, retrieval_service=retrieval)
    result = await orch.answer(QUERY)

    assert result["papers"]["H1"] == []  # never verified -> no evidence source
    assert result["evidence"] == []
    assert any("could not be verified" in w for w in result["warnings"])
    assert result["funnel"]["units_unknown"] >= 1


async def test_synthesizer_receives_balanced_multi_subquery_evidence():
    """Evidence selection must interleave subqueries instead of taking the
    first N (which starved H2/H3 in production runs)."""
    class MultiExtractor(FakeExtractor):
        async def extract_with_stats(self, subquery, document):
            items, stats = await super().extract_with_stats(subquery, document)
            n = {"H1": 5, "H2": 3}.get(subquery.id, 1)
            out = []
            for i in range(n):
                out.append(items[0].model_copy(update={
                    "claim": f"{subquery.id} claim {i}",
                    "confidence": 0.9 - 0.05 * i,
                }))
            return out, stats

    verifier = ScriptedVerifier([VerificationResult(results=[
        RelevanceVerdict(document_id=cid, relevance="relevant")
        for cid in ("c1", "c3")
    ])])
    retrieval = FakeRetrievalService({"q": [_doc("c1", "PMC-A"), _doc("c3", "PMC-B")]})
    subs = [_sub("H1", "impact of EF on survival"), _sub("H2", "prognostic value")]
    synth = FakeSynthesizer()
    orch = _orchestrator({"min_papers": 1, "verify_batch_max_docs": 6},
                         planner=FakePlanner(subs), verifier=verifier,
                         extractor=MultiExtractor(), synthesizer=synth,
                         retrieval_service=retrieval)

    # route c1->H1 and c3->H2 by giving each sub its own query key
    retrieval.rounds = {
        subs[0].query: [_doc("c1", "PMC-A")],
        subs[1].query: [_doc("c3", "PMC-B")],
    }
    result = await orch.answer(QUERY)
    selected_ids = {e.subquery_id for e in synth.seen_evidence}
    assert selected_ids == {"H1", "H2"}, \
        "both subqueries must reach the synthesizer within budget"
    assert synth.seen_notes and any("covered" in n for n in synth.seen_notes)
    assert result["funnel"]["evidence_synthesized"] <= 6


async def test_excluded_units_are_not_re_retrieved():
    """not_relevant chunks go onto the exclusion list for later rounds; the
    re-search must surface FRESH candidates from deeper ranks."""
    round1 = VerificationResult(results=[
        RelevanceVerdict(document_id="c1", relevance="not_relevant", reason="no outcome data"),
        RelevanceVerdict(document_id="c3", relevance="not_relevant", reason="wrong population"),
    ])
    round2 = VerificationResult(results=[
        RelevanceVerdict(document_id="c9", relevance="relevant", confidence=0.8),
    ])
    verifier = ScriptedVerifier([round1, round2])
    retrieval = FakeRetrievalService({
        "repair techniques recurrent coarctation percentages": [_doc("c1", "PMC-A"), _doc("c3", "PMC-B")],
        "repair techniques recurrent coarctation percentages long term outcomes": [_doc("c9", "PMC-D")],
    })
    sub = _sub()
    sub.synonyms = ["long term outcomes"]
    orch = _orchestrator({"min_papers": 1}, planner=FakePlanner([sub]), verifier=verifier,
                         extractor=FakeExtractor(), synthesizer=FakeSynthesizer(),
                         retrieval_service=retrieval)
    result = await orch.answer(QUERY)

    assert len(retrieval.calls) >= 2, "a re-search round must have happened"
    second_query, second_excluded = retrieval.calls[1]
    assert set(second_excluded) == {"c1", "c3"}, "rejected chunks must be excluded from re-search"
    # the rewritten query reached the retrieval layer
    assert "long term outcomes" in second_query
    papers = {u["document_id"] for u in result["papers"]["H1"]}
    assert papers == {"PMC-D"}
    # the rejection reasons fed the rewrite context (recorded on the verifier)
    assert any("no outcome data" in r or "wrong population" in r for r in ["no outcome data"])


class FakeUMLS:
    """Duck-typed UMLSClient: enriches every entity with MeSH terms."""

    enabled = True

    async def enrich_entities(self, entities):
        out = []
        for e in entities:
            out.append(e.model_copy(update={
                "preferred_name": "Heart failure",
                "cui": "C0018801",
                "ontology": "UMLS",
                "synonyms": ["Cardiac Failure"],
            }))
        return out


async def test_umls_expands_round_one_query():
    """Preferred names/synonyms from UMLS must reach the FIRST retrieval
    round, not just rewrite rounds."""
    verifier = ScriptedVerifier([VerificationResult(results=[
        RelevanceVerdict(document_id="c1", relevance="relevant", confidence=0.9),
    ])])
    retrieval = FakeRetrievalService({"q": [_doc("c1", "PMC-A")]})
    sub = SubQuery(id="H1", target="heart failure prognosis", focus="prognosis",
                   query="impact of heart failure on survival",
                   evidence_required=["mortality"])
    plan = QueryPlan(original_query=QUERY, question_type="prognosis",
                     entities=[{"text": "heart failure", "role": "condition",
                                "terminology": True}],
                     subqueries=[sub])
    enriched = EnrichedPlan(plan, clinical_entities=[
        ClinicalEntity(surface_form="heart failure", base_concept="heart failure")])

    class FixedPlanner:
        async def plan(self, query):
            return enriched

    orch = _orchestrator(planner=FixedPlanner(), verifier=verifier,
                         extractor=FakeExtractor(), synthesizer=FakeSynthesizer(),
                         retrieval_service=retrieval, umls_client=FakeUMLS())
    result = await orch.answer(QUERY)

    first_query = retrieval.calls[0][0]
    assert "cardiac failure" in first_query.lower(), \
        f"round-1 query must carry the MeSH synonym, got: {first_query!r}"
    assert "heart failure" in first_query.lower()
    assert sub.synonyms, "synonyms must be kept on the subquery for rewrites"
    assert result["funnel"]["subqueries"] == 1


async def test_worker_pool_runs_concurrently():
    active = 0
    max_active = 0

    async def slow_worker(sub, doc):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.05)
            return [], {}
        finally:
            active -= 1

    cfg = AppConfig()
    cfg.max_workers = 2
    cfg.max_retries = 1
    from src.orchestration.workers import EvidenceTask, EvidenceWorkerPool

    pool = EvidenceWorkerPool(slow_worker, cfg)
    tasks = [EvidenceTask(subquery=_sub(), document=_doc(f"c{i}", f"P{i}")) for i in range(4)]
    t0 = asyncio.get_event_loop().time()
    results = await pool.run(tasks)
    elapsed = asyncio.get_event_loop().time() - t0

    assert max_active == 2
    assert len(results) == 4
    assert elapsed < 4 * 0.05 + 0.02
