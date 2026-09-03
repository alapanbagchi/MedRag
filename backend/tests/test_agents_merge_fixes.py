"""Regression tests for the singular-flow merge fixes (offline, no LLM).

Covers: stable web document ids, worker-style web evidence counting toward
coverage, verifier-confidence surfacing, per-context event sinks (no
cross-talk between concurrent runs), resolution evidence folding into the
owning requirement, and the memory record_run translation.
"""

from __future__ import annotations

from typing import Any


def test_web_document_id_stable_and_unique():
    from src.agents.state import web_document_id

    a = web_document_id("https://www.who.int/news-room/fact-sheets/hypertension", "R1.W1")
    b = web_document_id("https://www.who.int/news-room/fact-sheets/hypertension", "R1.W2")
    c = web_document_id("https://www.who.int/news-room/fact-sheets/diabetes", "R1.W3")
    assert a == b  # same URL -> same id (dedup works)
    assert a != c  # distinct URLs never collide
    assert a.startswith("web:")
    assert web_document_id("", "R1.W9") == "web:R1.W9"


def test_verified_web_evidence_counts_toward_coverage():
    from src.agents.state import (
        AnswersTask,
        ResearchRequirement,
        SupportDirection,
        VerdictRelevance,
        VerifierVerdict,
        web_document_id,
    )

    req = ResearchRequirement(id="R1", text="Does X lower BP?", target_n=1)
    from src.agents.state import EvidenceItem

    item = EvidenceItem(
        id="R1.W1", requirement_id="R1", section="web",
        unit_kind="web_result", text="A sufficient guideline passage " * 10,
        source_url="https://www.who.int/x", trust="trusted",
        source_query="X blood pressure", retrieval_method="web:searxng",
    )
    item.document_id = web_document_id(item.source_url, item.id)
    req.add_item(item)
    item.submit_to_verifier()
    item.set_verdict(VerifierVerdict(
        evidence_id=item.id, requirement_id="R1",
        relevance=VerdictRelevance.RELEVANT, answers_task=AnswersTask.YES,
        support=SupportDirection.SUPPORTS, confidence=0.9,
        note="directly answers"))
    assert req.satisfied()
    assert req.coverage() == 1
    assert req.derive_status().value == "satisfied"


def test_confidence_float_surfaces_verdict_confidence():
    from src.agents.state import (
        AnswersTask,
        EvidenceItem,
        SupportDirection,
        VerdictRelevance,
        VerifierVerdict,
    )

    item = EvidenceItem(id="E1", text="t")
    assert item.confidence_float == 0.0
    item.submit_to_verifier()
    item.set_verdict(VerifierVerdict(
        evidence_id="E1", relevance=VerdictRelevance.RELEVANT,
        answers_task=AnswersTask.YES, support=SupportDirection.SUPPORTS,
        confidence=0.77))
    assert item.confidence_float == 0.77


def test_event_sink_does_not_leak_across_threads():
    import threading

    from src.agents import graph as G

    seen_main: list = []
    seen_other: list = []
    G.set_event_sink(lambda e, f: seen_main.append((e, f)))

    def other() -> None:
        G.set_event_sink(lambda e, f: seen_other.append((e, f)))
        G.emit_event("other_event", x=1)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    try:
        G.emit_event("main_event", x=2)
    finally:
        G.set_event_sink(None)
    assert [e for e, _ in seen_main] == ["main_event"]
    assert [e for e, _ in seen_other] == ["other_event"]


def test_resolution_evidence_folds_into_owner(monkeypatch):
    import asyncio

    from src.agents.agents import stages as sm
    from src.agents.state import (
        AnswersTask,
        Contradiction,
        ContradictionKind,
        ResearchRequirement,
        SupportDirection,
        VerdictRelevance,
        VerifierVerdict,
    )

    async def _candidates(query: str, top_k: int = 3):
        return [{
            "text": ("Resolution passage with enough characters to pass "
                     "the length gate. " * 5),
            "retrieval_method": "web:searxng",
            "source_url": "https://example.org/study",
            "trust": "trusted",
            "document_id": "",
            "section": "Results",
        }]

    async def _accept(verifier: Any, req: Any, item: Any) -> None:
        item.submit_to_verifier()
        item.set_verdict(VerifierVerdict(
            evidence_id=item.id, requirement_id=req.id,
            relevance=VerdictRelevance.RELEVANT,
            answers_task=AnswersTask.YES,
            support=SupportDirection.SUPPORTS, confidence=0.8))

    calls = {"n": 0}

    class _Agent:
        async def ainvoke(self, payload: dict) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                return {"messages": [
                    '{"characterization": "dose differs", "queries": ["q1"]}']}
            return {"messages": [
                '{"status": "resolved", "explanation": "dose explains it"}']}

    monkeypatch.setattr(sm, "_resolution_candidates", _candidates)
    monkeypatch.setattr(sm, "verify_item", _accept)

    owner = ResearchRequirement(id="R1", text="Does X lower BP?", target_n=3)
    contradiction = Contradiction(
        id="C1", claim="X lowers BP", requirement_id="R1",
        evidence_a=["A1"], evidence_b=["B1"],
        kind=ContradictionKind.DIRECT_CONFLICT)

    asyncio.run(sm.resolve_contradiction(_Agent(), object(), contradiction, owner))

    assert contradiction.resolution is not None
    assert contradiction.resolution.status.value == "resolved"
    assert contradiction.resolution.new_evidence_ids
    owned_ids = {it.id for it in owner.items}
    assert set(contradiction.resolution.new_evidence_ids) <= owned_ids
    for it in owner.items:
        assert it.requirement_id == "R1"
        assert it.document_id.startswith("web:")


def test_memory_result_translation():
    from types import SimpleNamespace

    from src.agents.bridge import _memory_result
    from src.agents.state import (
        AnswersTask,
        Contradiction,
        ContradictionKind,
        EvidenceItem,
        ResearchRequirement,
        ResolutionOutcome,
        ResolutionStatus,
        SupportDirection,
        VerdictRelevance,
        VerifierVerdict,
        XDeepRunState,
    )

    req = ResearchRequirement(id="R1", text="Does X lower BP?", target_n=1)
    item = EvidenceItem(id="R1.E1", run_id="abc", requirement_id="R1",
                        chunk_id="c1", document_id="PMC1", text="passage text",
                        retrieval_method="pgfts+pgvector", rank=1)
    req.add_item(item)
    item.submit_to_verifier()
    item.set_verdict(VerifierVerdict(
        evidence_id="R1.E1", requirement_id="R1",
        relevance=VerdictRelevance.RELEVANT, answers_task=AnswersTask.YES,
        support=SupportDirection.SUPPORTS, confidence=0.9, note="helps"))
    req.derive_status()
    run = XDeepRunState(run_id="abc", question="Does X lower BP?",
                        requirements=[req], answer="It helps. " * 50,
                        gaps=["long-term data"])
    run.contradictions.append(Contradiction(
        id="C1", claim="X helps", requirement_id="R1",
        evidence_a=["R1.E1"], evidence_b=[],
        kind=ContradictionKind.DIRECT_CONFLICT,
        resolution=ResolutionOutcome(status=ResolutionStatus.RESOLVED,
                                     explanation="ok")))

    out = _memory_result(run, "Does X lower BP?")
    assert out["run_id"] == "abc"
    assert out["evidence"][0]["status"] == "accepted"
    assert out["evidence"][0]["excerpt"] == "passage text"
    assert out["evidence"][0]["support"] == "supports"
    assert out["contradictions"][0]["resolution"]["status"] == "resolved"
    assert out["gaps"] == ["long-term data"]
    assert out["answer"]["summary"]

    # the translation must satisfy the memory extractor's contract
    from src.memory.pipeline import extract_run_candidates

    refs, claims, ctrs, gaps = extract_run_candidates(out, "sess1")
    assert refs and refs[0].verification_status == "verified"
    assert ctrs and ctrs[0].claim == "X helps"
    assert gaps and gaps[0].question == "long-term data"
