"""Unit tests for agentic v2 action executors (fakes, no live LLM/corpus)."""
import pytest

from src.agentic.planner import Decomposition, SubQueryPlan
from src.agentic.retriever_tool import RetrievalResult
from src.agentic_v2.actions import ActionExecutor, ActionResult
from src.agentic_v2.orchestrator import ActionDecision
from src.agentic_v2.state import (
    ActionType,
    CandidatePassage,
    EvidenceQuality,
    ObjectiveStatus,
    ResearchObjective,
    ResearchState,
    RetrievedDocument,
)
from src.agentic_v2.verify import ObjectiveVerdict, PassageAssessment
from src.agentic_v2.synthesize import SynthesisReport
from src.config import AppConfig


# -- fakes --------------------------------------------------------------

class FakePlanner:
    def __init__(self, subs):
        self.subs = subs

    async def plan(self, query):
        return Decomposition(question_type="comprehensive", subqueries=self.subs)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def search(self, sub, top_k=6, exclude_chunk_ids=None):
        self.calls.append((sub.query, exclude_chunk_ids))
        return [r for r in self.results if r.chunk_id not in (exclude_chunk_ids or [])]


class FakeVerifier:
    def __init__(self, verdict):
        self.verdict = verdict

    async def verify(self, objective, passages):
        return self.verdict


class FakeSynthesizer:
    async def synthesize(self, state):
        return SynthesisReport(summary="an answer", confidence=0.7)


def _executor(**deps):
    # corpus=None keeps tests off the real index (section loading is skipped).
    return ActionExecutor(config=AppConfig(), corpus=None, **deps)


def _sub():
    return SubQueryPlan(id="H1", target="hypertension variants and calcification",
                        intent="link variants to calcification",
                        query="hypertension variants medial arterial calcification",
                        evidence_required=["variant-calcification link"])


@pytest.mark.asyncio
async def test_decompose_creates_objectives():
    ex = _executor(planner=FakePlanner([_sub()]))
    state = ResearchState(question="q")
    res = await ex.execute(ActionDecision(action=ActionType.DECOMPOSE), state)
    assert res.status == "done"
    assert len(state.objectives) == 1
    assert state.objectives[0].id == "H1"
    assert state.objectives[0].status == ObjectiveStatus.OPEN


@pytest.mark.asyncio
async def test_decompose_recursive_children():
    parent = ResearchObjective(id="H1", statement="variants -> calcification")
    state = ResearchState(question="q")
    state.upsert_objective(parent)
    ex = _executor(planner=FakePlanner([_sub()]))
    await ex.execute(ActionDecision(action=ActionType.DECOMPOSE, objective_id="H1"), state)
    assert state.objective("H1.1") is not None


@pytest.mark.asyncio
async def test_global_retrieve_adds_new_docs_and_avoids_repeats():
    r = RetrievalResult(rank=1, chunk_id="c1", document_id="PMC1", section="Results",
                        paragraph_text="calcification text", rrf_score=0.9, unit_kind="paragraph")
    ex = _executor(retriever=FakeRetriever([r]))
    state = ResearchState(question="q")
    state.upsert_objective(ResearchObjective(id="H1", statement="calcification"))
    d = ActionDecision(action=ActionType.GLOBAL_RETRIEVE, objective_id="H1",
                       query="medial arterial calcification hypertension")
    res = await ex.execute(d, state)
    assert res.status == "done"
    assert len(state.documents) == 1
    assert state.documents[0].document_id == "PMC1"

    # repeating the SAME query must not re-run retrieval
    res2 = await ex.execute(d, state)
    assert res2.status == "repeated"
    assert len(state.documents) == 1


@pytest.mark.asyncio
async def test_verify_folds_verdict_into_state():
    state = ResearchState(question="q")
    obj = ResearchObjective(id="H1", statement="variants -> calcification")
    state.upsert_objective(obj)
    state.add_documents([
        RetrievedDocument(chunk_id="c1", document_id="PMC1", section="Results",
                          text="variants associated with calcification", objective_id="H1"),
    ])
    verdict = ObjectiveVerdict(
        objective_id="H1",
        status=ObjectiveStatus.SUPPORTED_WITH_CAVEAT,
        confidence=0.8,
        gap="direct variant-to-calcification evidence limited",
        caveats=["population mismatch"],
        population_match="partial",
        assessments=[PassageAssessment(document_id="PMC1", chunk_id="c1",
                                       section="Results", quality=EvidenceQuality.INDIRECT,
                                       support="supports", confidence=0.8)],
    )
    ex = _executor(verifier=FakeVerifier(verdict))
    res = await ex.execute(ActionDecision(action=ActionType.VERIFY, objective_id="H1"), state)
    assert res.status == "done"
    assert state.objective("H1").status == ObjectiveStatus.SUPPORTED_WITH_CAVEAT
    assert len(state.evidence) == 1
    assert state.evidence[0].quality == EvidenceQuality.INDIRECT
    assert state.gaps


@pytest.mark.asyncio
async def test_verify_rejected_passages_are_not_evidence():
    """not_relevant assessments get verdict events but never become evidence."""
    from src.agentic_v2.verify import PassageAssessment as PA

    state = ResearchState(question="q")
    obj = ResearchObjective(id="H1", statement="variants -> calcification")
    state.upsert_objective(obj)
    state.add_documents([
        RetrievedDocument(chunk_id="c-good", document_id="PMC1", section="Results",
                          text="relevant", objective_id="H1"),
        RetrievedDocument(chunk_id="c-bad", document_id="PMC2", section="Intro",
                          text="off topic", objective_id="H1"),
    ])
    verdict = ObjectiveVerdict(
        objective_id="H1",
        status=ObjectiveStatus.SUPPORTED,
        confidence=0.9,
        assessments=[
            PA(document_id="PMC1", chunk_id="c-good", relevance="relevant",
               quality=EvidenceQuality.DIRECT, support="supports", confidence=0.9),
            PA(document_id="PMC2", chunk_id="c-bad", relevance="not_relevant",
               quality=EvidenceQuality.BACKGROUND, support="neutral", confidence=0.7),
        ],
    )
    ex = _executor(verifier=FakeVerifier(verdict))
    res = await ex.execute(ActionDecision(action=ActionType.VERIFY, objective_id="H1"), state)
    assert res.status == "done"
    # only the relevant unit is answer material for the synthesizer
    assert [e.chunk_id for e in state.evidence] == ["c-good"]
    assert state.evidence[0].quality == EvidenceQuality.DIRECT
    assert state.objective("H1").status == ObjectiveStatus.SUPPORTED


@pytest.mark.asyncio
async def test_verify_without_objective_is_graceful():
    ex = _executor()
    state = ResearchState(question="q")
    res = await ex.execute(ActionDecision(action=ActionType.VERIFY), state)
    assert res.status == "done"
    assert "DECOMPOSE" in res.summary


# -- evidence excerpt anchoring (regression: definition lost in orchestration) --

_TABLE_WITH_DEFINITION = (
    "[Table ehae724-T2]\n"
    "Table 2: Acute coronary syndrome/percutaneous coronary intervention—clinical "
    "outcomes and their definitions\n"
    "Row: Acute coronary syndrome/PCI: Level 1 variables\n"
    "column_1: Acute coronary syndrome/PCI: Level 1 variables\n"
    "Row: Acute kidney injury requiring renal replacement therapy\n"
    "column_1: Renal replacement therapy includes ultrafiltration (haemofiltration), "
    "haemodialysis or peritoneal dialysis.50\n"
    "Row: Cardiac arrest\n"
    "column_1: Cardiac arrest is defined as a verified sudden cessation of cardiac "
    "activity causing unresponsiveness, absence of normal breathing and no signs of "
    "circulation (excluding syncope or profound vagally mediated bradycardia) with "
    "ventricular fibrillation, rapid ventricular tachycardia or bradycardia resulting "
    "in loss of consciousness, pulseless electrical activity, or asystole as the "
    "major causes.\n"
    "Row: Heart failure hospitalisation\n"
    "column_1: Hospital admission primarily due to heart failure.\n"
)


def _cardiac_arrest_objective() -> ResearchObjective:
    return ResearchObjective(
        id="H1", statement="definition of cardiac arrest",
        intent="define cardiac arrest and explain its nature",
        evidence_required=["definition of cardiac arrest",
                           "pathophysiology of cardiac arrest"],
        synonyms=["Cardiac Arrest", "Heart Arrest", "Asystole"],
        entities=["cardiac arrest"],
    )


@pytest.mark.asyncio
async def test_verify_anchors_evidence_excerpt_on_objective_terms():
    """A relevant TABLE whose definition row sits deep inside the unit must store
    an excerpt ANCHORED on that row — never a head slice that truncates the
    definition away before the synthesizer sees it."""
    from src.agentic_v2.verify import PassageAssessment as PA

    # the definition must really sit past the old 400-char head slice
    assert _TABLE_WITH_DEFINITION.find("Cardiac arrest is defined as") > 400

    state = ResearchState(question="What is a cardiac arrest?")
    state.upsert_objective(_cardiac_arrest_objective())
    state.add_documents([RetrievedDocument(
        chunk_id="c-def", document_id="PMC11704390", section="Results",
        unit_kind="table", text=_TABLE_WITH_DEFINITION, objective_id="H1",
    )])
    verdict = ObjectiveVerdict(
        objective_id="H1", status=ObjectiveStatus.SUPPORTED, confidence=1.0,
        assessments=[PA(document_id="PMC11704390", chunk_id="c-def", section="Results",
                        relevance="relevant", quality=EvidenceQuality.DIRECT,
                        support="supports", confidence=1.0)],
    )
    ex = _executor(verifier=FakeVerifier(verdict))
    res = await ex.execute(ActionDecision(action=ActionType.VERIFY, objective_id="H1"),
                           state)
    assert res.status == "done"
    assert len(state.evidence) == 1
    # the literal definition (buried past the old cap) is now part of the excerpt
    assert "Cardiac arrest is defined as" in state.evidence[0].excerpt
    # ...and the excerpt stays bounded (not the whole table)
    assert len(state.evidence[0].excerpt) <= 1600


def test_evidence_excerpt_anchors_deep_phrase_and_marks_head_fallback():
    """_evidence_excerpt() anchors mid-text phrases and never silently cuts."""
    from src.agentic_v2.actions import _evidence_excerpt

    obj = _cardiac_arrest_objective()
    text = ("intro filler. " * 50) + _TABLE_WITH_DEFINITION
    ex = _evidence_excerpt(text, obj)
    assert ex.startswith("...")                     # anchored mid-text -> lead marker
    assert "Cardiac arrest is defined as" in ex     # definition survives
    assert len(ex) <= 1450

    # no objective phrase present -> bounded head slice WITH a marker
    ex2 = _evidence_excerpt("plain text without any objective terms " * 200, obj)
    assert "objective terms" in ex2
    assert ex2.endswith("...")
    assert len(ex2) <= 1410


@pytest.mark.asyncio
async def test_synthesize_sets_terminal():
    ex = _executor(synthesizer=FakeSynthesizer())
    state = ResearchState(question="q")
    res = await ex.execute(ActionDecision(action=ActionType.SYNTHESIZE), state)
    assert state.terminal is True
    assert state.stop_reason == "synthesized"
    assert state.final_answer["summary"] == "an answer"


@pytest.mark.asyncio
async def test_stop_sets_terminal():
    ex = _executor()
    state = ResearchState(question="q")
    await ex.execute(ActionDecision(action=ActionType.STOP, rationale="budget exhausted"), state)
    assert state.terminal is True
    assert "budget exhausted" in state.stop_reason


def test_map_relevance():
    from src.agentic_v2.actions import _map_relevance
    q, s = _map_relevance("relevant", EvidenceQuality.UNKNOWN, "neutral")
    assert q == EvidenceQuality.DIRECT and s == "supports"
    q, s = _map_relevance("partially_relevant", EvidenceQuality.UNKNOWN, "neutral")
    assert q == EvidenceQuality.INDIRECT and s == "supports"
    q, s = _map_relevance("not_relevant", EvidenceQuality.UNKNOWN, "neutral")
    assert q == EvidenceQuality.BACKGROUND and s == "neutral"
    # empty relevance -> keep the provided fallback
    q, s = _map_relevance("", EvidenceQuality.DIRECT, "supports")
    assert q == EvidenceQuality.DIRECT and s == "supports"


@pytest.mark.asyncio
async def test_enrich_stores_synonyms_and_enriched_query():
    from src.agentic.umls_tool import EnrichedTerm

    class FakeUMLS:
        def __init__(self):
            self.calls = []

        async def enrich_subquery(self, sub):
            self.calls.append(sub)
            return [EnrichedTerm(surface_form=sub.entities[0].text,
                                 preferred_name="medial artery calcification",
                                 synonyms=["vascular calcification"])]

        def apply_to_query(self, sub, terms):
            return sub.query + " vascular calcification"

    ex = _executor(umls_enricher=FakeUMLS())
    state = ResearchState(question="q")
    state.upsert_objective(ResearchObjective(
        id="H1", statement="medial arterial calcification",
        entities=["medial arterial calcification"],
    ))
    res = await ex.execute(ActionDecision(action=ActionType.ENRICH, objective_id="H1"), state)
    assert res.status == "done"
    o = state.objective("H1")
    assert o.enriched_query == "medial arterial calcification vascular calcification"
    assert "vascular calcification" in o.synonyms


@pytest.mark.asyncio
async def test_global_retrieve_uses_enriched_query_and_paragraph_text():
    class Ret:
        def __init__(self):
            self.calls = []

        async def search(self, sub, top_k=6, exclude_chunk_ids=None):
            self.calls.append((sub.query, exclude_chunk_ids))
            return [RetrievalResult(rank=1, chunk_id="c1", document_id="PMC1",
                                    section="Results", paragraph_text="chunk only",
                                    rrf_score=0.9, unit_kind="paragraph")]

    ret = Ret()
    ex = ActionExecutor(config=AppConfig(), retriever=ret, corpus=None)
    state = ResearchState(question="q")
    state.upsert_objective(ResearchObjective(
        id="H1", statement="plain", enriched_query="plain with synonyms",
    ))
    await ex.execute(ActionDecision(action=ActionType.GLOBAL_RETRIEVE,
                                    objective_id="H1", query="plain"), state)
    assert ret.calls[0][0] == "plain with synonyms"          # enriched query used
    assert len(state.documents) == 1
    assert state.documents[0].text == "chunk only"           # full structural unit
    assert state.documents[0].unit_kind == "paragraph"


# -- full-document must never reach the verifier ---------------------------

class SimpleCorpus:
    """Duck-typed corpus akin to src.retrieval.corpus (df + document helpers)."""

    def __init__(self, df):
        self._df = df

    def document_id(self, chunk_id):
        row = self._df[self._df["id"] == chunk_id]
        return str(row.iloc[0]["document_id"]) if not row.empty else None

    def document_ids(self):
        return self._df["document_id"].drop_duplicates()


def _fake_corpus_with_df(doc_id="DOC1"):
    """A minimal corpus exposing a chunk dataframe for unit restoration."""
    import pandas as pd

    df = pd.DataFrame([
        {"id": "c1", "document_id": doc_id,
         "text": "para one about trauma severity scoring for wearable patients",
         "section": "Results", "table_id": None, "figure_id": None, "document_position": 1},
        {"id": "c2", "document_id": doc_id,
         "text": "para two about unrelated vitals monitoring details",
         "section": "Methods", "table_id": None, "figure_id": None, "document_position": 2},
        {"id": "c3", "document_id": doc_id,
         "text": "para three off-topic background material",
         "section": "Intro", "table_id": None, "figure_id": None, "document_position": 3},
    ])
    return SimpleCorpus(df)


@pytest.mark.asyncio
async def test_read_document_by_id_returns_relevant_units_not_full_paper():
    """READ_DOCUMENT with only document_id restores structural units ranked by
    objective relevance - never a (full document) unit for the verifier."""
    from src.retrieval.fullpaper import reset_unit_index

    reset_unit_index()
    ex = _executor()
    ex._corpus_loaded = True
    ex._corpus = _fake_corpus_with_df()
    state = ResearchState(question="q")
    obj = ResearchObjective(id="H2", statement="trauma severity scoring for wearables")
    state.upsert_objective(obj)
    dec = ActionDecision(action=ActionType.READ_DOCUMENT, document_id="DOC1",
                         objective_id="H2", query="trauma severity scoring")
    result = await ex.execute(dec, state)
    assert result.status == "done"
    # Every stored unit is a structural unit - never the whole paper
    assert state.documents, "expected units to be stored"
    for d in state.documents:
        assert d.unit_kind != "document"
        assert d.section != "(full document)"
    # The most relevant unit (trauma scoring) is ranked first
    assert "trauma severity scoring" in state.documents[0].text
    # The verifier candidate intake still keeps only units
    cands = ex._candidates_for(state, "H2")
    assert cands
    for c in cands:
        assert c.text != "THE ENTIRE PAPER"
        assert "para one" in c.text or "para two" in c.text or "para three" in c.text


@pytest.mark.asyncio
async def test_candidates_for_filters_stray_full_document_unit():
    """Even if a (full document) unit somehow lands in state.documents, the
    candidate intake must drop it so the verifier never receives it."""
    state = ResearchState(question="q")
    obj = ResearchObjective(id="H2", statement="s")
    state.upsert_objective(obj)
    state.add_documents([
        RetrievedDocument(chunk_id="par-1", document_id="P1", section="Results",
                          unit_kind="paragraph", text="a paragraph unit",
                          objective_id="H2"),
        RetrievedDocument(document_id="P1", section="(full document)",
                          unit_kind="document",
                          text="THE ENTIRE PAPER WOULD GO HERE",
                          objective_id="H2"),
    ])
    ex = _executor()
    cands = ex._candidates_for(state, "H2")
    texts = [c.text for c in cands]
    assert "a paragraph unit" in texts
    assert "THE ENTIRE PAPER WOULD GO HERE" not in texts



# -- full-document expansion (never drop, pass the relevant part) ------------

@pytest.mark.asyncio
async def test_verify_candidates_expands_full_document_not_drops():
    """A (full document) unit must NOT be dropped: its relevant paragraphs are
    extracted and passed to the verifier instead."""
    state = ResearchState(question="trauma severity scoring")
    obj = ResearchObjective(id="H2", statement="trauma severity scoring parameters")
    state.upsert_objective(obj)
    full_doc = RetrievedDocument(
        document_id="PMC11878906", section="(full document)", unit_kind="document",
        text=("We developed a novel trauma severity scoring system for wearable "
              "continuous monitoring. " * 800),
        objective_id="H2",
    )
    real_unit = RetrievedDocument(
        chunk_id="c1", document_id="PMC11878944", section="Materials and methods",
        unit_kind="paragraph",
        text="The score uses heart rate variability, respiratory rate, oxygen saturation.",
        objective_id="H2",
    )
    state.add_documents([full_doc, real_unit])

    ex = _executor()
    dec = ActionDecision(action=ActionType.VERIFY, objective_id="H2",
                         query="trauma severity scoring")
    passages = await ex._verify_candidates(state, obj, dec)

    # Nothing dropped: the full doc remains in state and is represented below
    assert any((d.section or "").strip() == "(full document)" for d in state.documents)
    # Its relevant units reach the verifier - bounded, never the whole paper
    assert any(p.document_id == "PMC11878906" for p in passages)
    for p in passages:
        if p.document_id == "PMC11878906":
            assert p.section != "(full document)"
            assert len(p.text.split()) <= 600


@pytest.mark.asyncio
async def test_verify_candidates_full_document_no_relevance():
    """Full doc with no corpus and empty text: gracefully skipped, not crashed."""
    state = ResearchState(question="q")
    obj = ResearchObjective(id="H2", statement="s")
    state.upsert_objective(obj)
    state.add_documents([
        RetrievedDocument(document_id="P1", section="(full document)",
                          unit_kind="document", text="", objective_id="H2"),
    ])
    ex = _executor()
    dec = ActionDecision(action=ActionType.VERIFY, objective_id="H2")
    passages = await ex._verify_candidates(state, obj, dec)
    assert passages == []
