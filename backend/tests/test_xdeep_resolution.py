"""Resolution-loop tests (Gap D): the resolver actually runs searches and
re-verifies any new passage through the critic before it can influence the
judgement. No real LLM calls - all stage functions are faked/monkeypatched.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.x_deepagents.state import (
    AnswersTask,
    Contradiction,
    ContradictionKind,
    ResolutionStatus,
    SupportDirection,
    VerifierVerdict,
)


def make_contradiction():
    return Contradiction(
        id="C1",
        claim="Vitamin D lowers blood pressure",
        requirement_id="R1",
        evidence_a=["E1"],
        evidence_b=["E2"],
        evidence_a_text="Study A: vitamin D reduced SBP by 4 mmHg.",
        evidence_b_text="Study B: no effect of vitamin D on SBP.",
        kind=ContradictionKind.DIRECT_CONFLICT,
    )


class FakeAgent:
    """Records prompts; returns canned JSON per stage."""

    def __init__(self, plan_json=None, judge_json=None):
        self.plan_json = plan_json or {
            "characterization": "differ by baseline vitamin D status",
            "queries": ["vitamin D baseline status blood pressure meta-analysis"],
        }
        self.judge_json = judge_json or {
            "status": "resolved",
            "explanation": "benefit seen only with low baseline vitamin D",
            "characterization": "baseline vitamin D status",
        }
        self.prompts = []

    async def ainvoke(self, input):
        self.prompts.append(str(input))
        content = str(input)
        if "STAGE 1 (CHARACTERISE)" in content:
            text = json.dumps(self.plan_json)
        else:
            text = json.dumps(self.judge_json)
        return {"messages": [SimpleNamespace(content=text, role="assistant")]}


async def _verifier(agent, requirement, item):
    """A critic that ACCEPTS items whose text contains 'meta-analysis'."""
    item.submit_to_verifier()
    accepted = "meta-analysis" in item.text.lower()
    item.set_verdict(VerifierVerdict(
        evidence_id=item.id,
        requirement_id=item.requirement_id,
        relevance="relevant" if accepted else "not_relevant",
        answers_task=AnswersTask.YES if accepted else AnswersTask.NO,
        support=SupportDirection.SUPPORTS if accepted else SupportDirection.NEUTRAL,
        confidence=0.9 if accepted else 0.1,
        note="meta-analysis answers" if accepted else "irrelevant",
    ))
    return item


def test_resolution_runs_search_and_uses_verified_evidence(monkeypatch):
    """The resolver: plans queries, runs local search, keeps only
    VERIFIED new evidence in the judgement prompt, resolves."""
    from src.x_deepagents.agents.stages import resolve_contradiction

    captured = {}

    async def fake_candidates(query, top_k=3):
        captured["query"] = query
        return [{
            "text": "A meta-analysis of RCTs found benefit only in participants "
                    "with low baseline vitamin D.",
            "retrieval_method": "pgfts+pgvector",
            "source_url": "",
            "trust": "local",
            "document_id": "PMC77",
            "section": "Results",
        }]

    async def fake_search_impl(**kw):
        return json.dumps({"available": True, "results": []})

    monkeypatch.setattr("src.x_deepagents.agents.stages._resolution_candidates",
                        fake_candidates)
    # note: _resolution_candidates is what runs the tools; fake it to avoid DB

    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _verifier)
    agent = FakeAgent()
    ctr = make_contradiction()
    out = asyncio.run(resolve_contradiction(agent, None, ctr))

    assert out.resolution is not None
    assert out.resolution.status == ResolutionStatus.RESOLVED
    # the search really ran with the planned query
    assert "vitamin D" in captured["query"]
    # the new VERIFIED evidence id is recorded
    assert len(out.resolution.new_evidence_ids) == 1
    assert out.resolution.search_rounds == 1
    # the judgement prompt contains the verified new evidence
    judge_prompt = agent.prompts[-1]
    assert "NEW VERIFIED EVIDENCE" in judge_prompt
    assert "C1.N1-1" in judge_prompt


def test_resolution_rejects_unverified_web_text(monkeypatch):
    """A web result the critic REJECTS must never become a resolution reason."""
    from src.x_deepagents.agents.stages import resolve_contradiction

    async def fake_candidates(query, top_k=3):
        return [{
            "text": "Random blog claim about vitamin D curing hypertension.",
            "retrieval_method": "web:google",
            "source_url": "https://blog.example.com/post",
            "trust": "unverified",
            "document_id": "",
            "section": "web",
        }]

    monkeypatch.setattr("src.x_deepagents.agents.stages._resolution_candidates",
                        fake_candidates)

    agent = FakeAgent()
    ctr = make_contradiction()
    out = asyncio.run(resolve_contradiction(agent, _verifier, ctr))

    # _verifier rejects (no "meta-analysis" in text) -> no new evidence
    assert out.resolution.new_evidence_ids == []
    judge_prompt = agent.prompts[-1]
    assert "NEW VERIFIED EVIDENCE" not in judge_prompt  # nothing verified to add
    assert out.resolution.status == ResolutionStatus.RESOLVED  # judged on sides only


def test_resolution_unresolved_is_honest(monkeypatch):
    """No new evidence + conflicting sides -> honest unresolved judgement."""
    from src.x_deepagents.agents.stages import resolve_contradiction

    async def fake_candidates(query, top_k=3):
        return []

    monkeypatch.setattr("src.x_deepagents.agents.stages._resolution_candidates",
                        fake_candidates)
    agent = FakeAgent(judge_json={
        "status": "unresolved",
        "explanation": "evidence does not explain the discrepancy",
        "characterization": "genuine conflict",
    })
    ctr = make_contradiction()
    out = asyncio.run(resolve_contradiction(agent, _verifier, ctr))
    assert out.resolution.status == ResolutionStatus.UNRESOLVED
    assert out.resolution.search_rounds == 0
    assert "does not explain" in out.resolution.explanation


def test_resolution_budget_caps_new_evidence(monkeypatch):
    """Even when candidates are plentiful, at most a few are verified."""
    from src.x_deepagents.agents.stages import resolve_contradiction

    async def fake_candidates(query, top_k=3):
        return [
            {"text": f"A meta-analysis result number {i}.", "retrieval_method": "pgfts",
             "source_url": "", "trust": "local", "document_id": f"PMC{i}",
             "section": "Results"}
            for i in range(10)
        ]

    monkeypatch.setattr("src.x_deepagents.agents.stages._resolution_candidates",
                        fake_candidates)
    agent = FakeAgent()
    ctr = make_contradiction()
    out = asyncio.run(resolve_contradiction(agent, _verifier, ctr))
    assert len(out.resolution.new_evidence_ids) <= 4
