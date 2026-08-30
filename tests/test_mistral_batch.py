"""Mistral batch client + critic batching (offline tests, no network)."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.agentic_v3.critic import CriticAgent, _BATCH_UNAVAILABLE
from src.agentic_v3.state import (
    EvidenceRequirement,
    EvidenceStatus,
    ResearchTask,
    RetrievedPaper,
)
from src.llm.mistral_batch import MistralBatch, MistralBatchError
from tests.conftest import native_test_model


def _task() -> ResearchTask:
    return ResearchTask(
        id="T1", title="eggs and cardiovascular disease",
        objective="does egg intake affect CVD risk", intent="association evidence",
        evidence_requirements=[
            EvidenceRequirement(id="T1.R1", text="egg consumption and CVD",
                                target_n=2)],
    )


def _paper(doc: str, evidence_id: str) -> RetrievedPaper:
    return RetrievedPaper(
        evidence_id=evidence_id, task_id="T1", requirement_id="T1.R1",
        attempt_id="A1", status=EvidenceStatus.RETRIEVED,
        document_id=doc, chunk_id=f"c-{doc}", section="Results",
        score=0.9, text="A passage about eggs and cardiovascular disease outcomes.",
    )


# ---------------------------------------------------------------------------
# Client result parsing (result-shape tolerance)
# ---------------------------------------------------------------------------

def test_extract_outputs_by_custom_id_and_id():
    job = {"outputs": [
        {"custom_id": "E1", "response": {"choices": [
            {"message": {"content": '{"relevance": "relevant"}'}}]}},
        {"id": "E2", "response": {"choices": [
            {"message": {"content": '{"answers_task": "yes"}'}}]}},
    ]}
    out = MistralBatch.extract_outputs(job, ["E1", "E2"])
    assert out["E1"] == '{"relevance": "relevant"}'
    assert out["E2"] == '{"answers_task": "yes"}'


def test_extract_outputs_missing_and_positional_fallback():
    out = MistralBatch.extract_outputs(
        {"outputs": [{"custom_id": "E1", "response": {"choices": [
            {"message": {"content": "x"}}]}}]}, ["E1", "MISSING"])
    assert out["E1"] == "x"
    assert out.get("MISSING") is None
    # entry without ids -> positional fallback
    out2 = MistralBatch.extract_outputs(
        {"outputs": [{"response": {"choices": [{"message": {"content": "y"}}]}}]},
        ["A", "B"])
    assert out2["A"] == "y"
    assert out2.get("B") is None


# ---------------------------------------------------------------------------
# Critic batch request building
# ---------------------------------------------------------------------------

def test_build_batch_requests_shape():
    critic = CriticAgent(model=native_test_model('{"relevance":"relevant"}'))
    papers = [_paper("PMC1", "T1.R1.A1.E1"), _paper("PMC2", "T1.R1.A1.E2")]
    requests, pairs = critic._build_batch_requests(_task(), _task().evidence_requirements[0],
                                                   papers, model="mistral-medium-latest")
    assert len(requests) == 2
    r0 = requests[0]
    assert r0["custom_id"] == "T1.R1.A1.E1"
    assert r0["body"]["model"] == "mistral-medium-latest"
    assert r0["body"]["messages"][0]["role"] == "system"
    assert "CRITIC" in r0["body"]["messages"][0]["content"]
    assert r0["body"]["response_format"]["type"] == "json_schema"
    assert r0["body"]["response_format"]["json_schema"]["name"] == "critic_output"
    assert pairs[0][1]["evidence_id"] == "T1.R1.A1.E1"


def test_batch_disabled_when_verifier_not_mistral():
    critic = CriticAgent(model=native_test_model("{}"))
    # provider default (not mistral) + batch mode auto -> disabled
    assert critic._batching_enabled() is False


def test_batch_forced_on():
    import os
    os.environ["VERIFIER_BATCH_ENABLED"] = "1"
    try:
        critic = CriticAgent(model=native_test_model("{}"))
        assert critic._batching_enabled() is True
    finally:
        os.environ.pop("VERIFIER_BATCH_ENABLED", None)


# ---------------------------------------------------------------------------
# Fallback: batch failure (e.g. 402 billing) -> sequential judge
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_batch_cache():
    _BATCH_UNAVAILABLE.clear()
    yield
    _BATCH_UNAVAILABLE.clear()


def test_batch_failure_falls_back_to_sequential(monkeypatch):
    monkeypatch.setenv("VERIFIER_BATCH_ENABLED", "1")
    monkeypatch.setenv("VERIFIER_PROVIDER", "mistral")
    verdict_json = json.dumps({"relevance": "relevant", "answers_task": "yes",
                               "support": "supports", "confidence": 0.95,
                               "note": "passage answers the requirement"})
    critic = CriticAgent(model=native_test_model(verdict_json))
    attempts = {"n": 0}

    async def boom(*_a, **_k):
        attempts["n"] += 1
        raise MistralBatchError("HTTP 402: enable billing via the console")

    monkeypatch.setattr(critic, "_batch_judge_papers", boom)
    papers = [_paper("PMC1", "T1.R1.A1.E1"), _paper("PMC2", "T1.R1.A1.E2")]
    pairs = asyncio.run(
        critic.judge_papers(_task(), _task().evidence_requirements[0], papers))
    assert len(pairs) == 2
    assert all(verdict is not None for _, verdict in pairs)
    assert attempts["n"] == 1, "second round must skip the batch attempt"
    # second call: batch cache marks it unavailable -> straight sequential
    pairs2 = asyncio.run(
        critic.judge_papers(_task(), _task().evidence_requirements[0], papers))
    assert len(pairs2) == 2
    assert attempts["n"] == 1


def test_sequential_path_returns_none_on_critic_failure():
    class BoomCritic(CriticAgent):
        async def judge(self, *a, **k):
            raise RuntimeError("provider timeout")

    critic = BoomCritic(model=native_test_model("{}"))
    papers = [_paper("PMC1", "T1.R1.A1.E1")]
    pairs = asyncio.run(
        critic.judge_papers(_task(), _task().evidence_requirements[0], papers))
    assert len(pairs) == 1
    assert pairs[0][1] is None