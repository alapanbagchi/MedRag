"""Verifier agent tests: batched calls, unknown-on-failure, salvage parsing."""

from __future__ import annotations

import json


from src.agents.verifier_new import VerifierAgent
from src.config import AppConfig

from tests.conftest import native_test_model

PAPERS = [
    {"document_id": "D1", "full_text": "HFrEF patients showed 43% mortality at 5 years (p<0.001)."},
    {"document_id": "D2", "full_text": "MRI findings in coarctation cohorts."},
]

BATCH_JSON = json.dumps({
    "results": [
        {"document_id": "D1", "relevance": "relevant", "confidence": 0.9,
         "reason": "reports EF-outcome data"},
        {"document_id": "D2", "relevance": "not_relevant", "confidence": 0.8,
         "reason": "imaging only"},
    ],
    "overall": "any_relevant",
})


def _agent(model=None) -> VerifierAgent:
    cfg = AppConfig()
    cfg.verify_batch_max_docs = 6
    cfg.verify_batch_max_tokens = 100000
    return VerifierAgent(config=cfg, model=model)


async def test_batched_verification_maps_verdicts():
    agent = _agent(native_test_model(BATCH_JSON))
    result = await agent.verify_papers(PAPERS, "EF and survival", ["mortality"])
    by_id = {v.document_id: v for v in result.results}
    assert by_id["D1"].relevance == "relevant"
    assert by_id["D2"].relevance == "not_relevant"
    assert result.overall == "any_relevant"


async def test_infra_failure_yields_unknown_not_rejection():
    """A verifier crash must produce 'unknown' (retryable), never a silent
    not_relevant that permanently drops the paper."""
    agent = _agent()
    import src.llm.run as run_mod

    async def explode(*a, **k):
        raise RuntimeError("429 quota exceeded")

    orig = run_mod.ask_structured
    run_mod.ask_structured = explode
    try:
        result = await agent.verify_papers(PAPERS, "EF and survival", ["mortality"])
    finally:
        run_mod.ask_structured = orig
    assert all(v.relevance == "unknown" for v in result.results)
    assert all("quota" in v.reason for v in result.results)


async def test_batches_respect_max_docs(monkeypatch):
    cfg = AppConfig()
    cfg.verify_batch_max_docs = 1  # force one doc per call
    agent = VerifierAgent(config=cfg, model=native_test_model(BATCH_JSON))
    batches = agent._batches(PAPERS)
    assert [len(b) for b in batches] == [1, 1]


async def test_skipped_documents_become_unknown():
    """If the model answers for only one document, the rest become unknown."""
    partial = json.dumps({
        "results": [{"document_id": "D1", "relevance": "relevant", "confidence": 0.9}],
        "overall": "any_relevant",
    })
    agent = _agent(native_test_model(partial))
    result = await agent.verify_papers(PAPERS, "EF and survival", ["mortality"])
    by_id = {v.document_id: v for v in result.results}
    assert by_id["D2"].relevance == "unknown"


async def test_text_fallback_salvages_array_output():
    """Structured mode fails on TestModel text that isn't schema-JSON; the
    fallback parser salvages a bare array with boolean/loose relevance."""
    messy = 'Sure! <think>reasoning here</think>\n[{"document_id": "D1", "relevance": "True", "confidence": "0.7"}]'
    agent = _agent(native_test_model(messy))
    result = await agent.verify_papers(PAPERS, "EF and survival", ["mortality"])
    by_id = {v.document_id: v for v in result.results}
    assert by_id["D1"].relevance == "relevant"
