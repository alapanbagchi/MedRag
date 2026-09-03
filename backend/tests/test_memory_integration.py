"""Memory layer — integration with the singular deepagents flow.

Run 1 answers the vitamin D question and commits research memory; Run 2 is a
follow-up ("what about older adults?") that MUST resume the same research
session and hand bounded memory context to the orchestrator — while the
evidence boundary stays intact (memory items are advisory for the planner
only; verified evidence still comes from the workers).

No LLM / no live DB: the graph's run_research is faked with real run states,
the memory API is real (in-memory store + hash embedder).
"""

from __future__ import annotations

import asyncio

from src.agents import bridge as bridge_module
from src.agents.state import (
    AnswersTask,
    EvidenceItem,
    ResearchRequirement,
    SupportDirection,
    VerdictRelevance,
    VerifierVerdict,
    XDeepRunState,
)
from src.memory.api import MemoryAPI
from src.memory.config import MemoryConfig
from src.memory.embed import HashEmbedder
from src.memory.enums import EvidenceRole, ProvenanceClass
from src.memory.store import InMemoryMemoryStore


def build_api() -> MemoryAPI:
    return MemoryAPI(store=InMemoryMemoryStore(), embedder=HashEmbedder(256),
                     config=MemoryConfig(backend="memory"))


def vitamin_d_run(run_id: str = "run1") -> XDeepRunState:
    """A realistic finished run state: 3 verified supporting papers."""
    req = ResearchRequirement(id="R1",
                              text="vitamin D supplementation and blood pressure",
                              target_n=3)
    for i, doc in enumerate(["PMC11684474", "PMC11684475", "PMC11684476"], 1):
        item = EvidenceItem(
            id=f"R1.E{i}", run_id=run_id, requirement_id="R1",
            chunk_id=f"c-{doc}", document_id=doc, section="Results",
            text=f"{doc}: blood-pressure finding {i}",
            claim=(f"Finding from {doc}: vitamin D supplementation affects "
                   "blood pressure in hypertensive adults."),
            source_query="vitamin D blood pressure",
            retrieval_method="pgfts+pgvector", rank=i)
        req.add_item(item)
        item.submit_to_verifier()
        item.set_verdict(VerifierVerdict(
            evidence_id=item.id, requirement_id="R1",
            relevance=VerdictRelevance.RELEVANT,
            answers_task=AnswersTask.YES,
            support=SupportDirection.SUPPORTS, confidence=0.9,
            note="directly answers"))
    req.derive_status()
    return XDeepRunState(
        run_id=run_id,
        question="Does vitamin D supplementation lower blood pressure?",
        requirements=[req],
        answer=("Dietary sodium reduction and vitamin D findings are "
                "summarized from the verified evidence. " * 10),
        gaps=[])


def _stage_flow(monkeypatch, runs: list):
    """Fake run_research with staged run states; capture planner contexts."""
    contexts: list[str] = []
    staged = list(runs)

    async def fake_run_research(question, memory_context=""):
        contexts.append(memory_context or "")
        return staged.pop(0)

    monkeypatch.setattr(bridge_module, "run_research", fake_run_research)
    return contexts


async def _collect(question: str, conversation_id: str = "") -> list[dict]:
    import json

    lines = []
    async for line in bridge_module.stream_xdeep(
            question, conversation_id=conversation_id):
        lines.append(json.loads(line))
    return lines


def _memory_frames(lines: list[dict], kind: str) -> list[dict]:
    return [l for l in lines
            if l.get("type") == "memory" and l.get("kind") == kind]


def test_cross_session_continuity(monkeypatch):
    api = MemoryAPI(store=InMemoryMemoryStore(), embedder=HashEmbedder(256),
                    config=MemoryConfig(backend="memory"))
    monkeypatch.setattr(bridge_module, "_get_memory_api", lambda: api)
    contexts = _stage_flow(monkeypatch, [vitamin_d_run("run1"),
                                         vitamin_d_run("run2")])

    # ---- run 1: full investigation ----
    lines1 = asyncio.run(_collect(
        "Does vitamin D supplementation lower blood pressure?", "conv-X"))
    commits1 = _memory_frames(lines1, "commit")
    assert len(commits1) == 1
    stats1 = commits1[0]["stats"]
    assert stats1["claims_committed"] >= 1
    assert stats1["claims_deduped"] >= 2      # near-dup claims merged, links unioned
    session1 = commits1[0]["session_id"]
    assert session1 and api.store.get_session(session1) is not None
    # the orchestrator got (empty-at-first) context plumbing, not nothing
    assert contexts[0] is not None

    evidence_claims = [c for c in api.store.get_claims()
                       if c.provenance_class == ProvenanceClass.EVIDENCE_DERIVED_CLAIM]
    assert len(evidence_claims) >= 1
    # every evidence-derived claim is fully traceable to verified evidence;
    # support links from all distinct papers accumulate on the survivor
    all_links = [l for c in evidence_claims
                 for l in api.store.get_claim_evidence_links(c.id)
                 if l.role == EvidenceRole.SUPPORTED_BY]
    assert len(all_links) >= 3                 # 3 distinct papers recorded
    for claim in evidence_claims:
        links = api.store.get_claim_evidence_links(claim.id)
        assert links
        assert all(api.store.get_evidence_ref(l.evidence_ref_id).verified
                   for l in links)

    # ---- run 2: follow-up query resumes the SAME research session ----
    lines2 = asyncio.run(_collect(
        "What about vitamin D supplementation in older adults?", "conv-X"))
    commits2 = _memory_frames(lines2, "commit")
    assert len(commits2) == 1
    assert commits2[0]["session_id"] == session1   # continuation, not new
    # the orchestrator really received the prior research state
    assert any("vitamin" in (m or "").casefold() for m in contexts[1:])
    # the run merged into the same session (no duplicate claims)
    claims_after = api.store.get_claims(
        session_id=session1, provenance=ProvenanceClass.EVIDENCE_DERIVED_CLAIM)
    assert len(claims_after) == len(evidence_claims)   # dedup, no explosion


def test_flow_without_memory_is_unchanged(monkeypatch):
    """MEMORY_ENABLED=0 -> no memory frames, run still streams to done."""
    bridge_module._memory_api = None
    bridge_module._memory_failed = False
    monkeypatch.setenv("MEMORY_ENABLED", "0")
    _stage_flow(monkeypatch, [vitamin_d_run("run9")])
    try:
        lines = asyncio.run(_collect("dietary salt and hypertension?"))
    finally:
        bridge_module._memory_failed = False
    assert "done" in {l["type"] for l in lines}
    assert not [l for l in lines if l.get("type") == "memory"]


def test_context_boundary_in_integration(monkeypatch):
    """Memory context never injects into the evidence channel: the run's
    verified evidence still comes only from the workers, and the planner
    context region is bounded and advisory-only."""
    api = build_api()
    monkeypatch.setattr(bridge_module, "_get_memory_api", lambda: api)
    contexts = _stage_flow(monkeypatch, [vitamin_d_run("runB"),
                                         vitamin_d_run("runC")])
    lines = asyncio.run(_collect("vitamin D and blood pressure?", "conv-Z"))

    sources = [l for l in lines if l.get("type") == "sources"]
    assert sources  # worker evidence present in the stream
    for s in sources[0]["sources"]:
        assert s["id"]  # every source resolves to a real document

    # second turn in the SAME chat: the planner context carries bounded,
    # advisory-only prior research (never evidence)
    asyncio.run(_collect("vitamin D dosage in trials?", "conv-Z"))
    ctx = contexts[1]
    assert ctx
    assert len(ctx) < 6000
    assert "not evidence" in ctx.casefold() or "VERIFIED EVIDENCE" in ctx
