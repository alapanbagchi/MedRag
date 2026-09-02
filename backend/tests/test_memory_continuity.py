"""Memory + Context layer — follow-up / cross-question continuity.

Regression tests for the reported behavior: asking "how does hypertension
lead to diseases" after "how does hypertension affect life expectancy" must
NOT re-process the first query. The planner gets a context that marks prior
questions answered, surfaces the established conclusion, and instructs it to
plan only for the new question.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agentic.pipeline import AgenticV3Pipeline
from src.agentic.state import (
    EvidenceRequirement,
    EvidenceSource,
    MasterPlan,
    ResearchTask,
    SupportDirection,
    VerifiedEvidence,
    WorkerReport,
)
from src.agents.synthesize import SynthesisReport
from src.config import AppConfig
from src.memory.api import MemoryAPI
from src.memory.config import MemoryConfig
from src.memory.embed import HashEmbedder
from src.memory.retrieval import retrieve_claims
from src.memory.store import InMemoryMemoryStore


def build_api() -> MemoryAPI:
    return MemoryAPI(store=InMemoryMemoryStore(), embedder=HashEmbedder(256),
                     config=MemoryConfig(backend="memory"))


def hypertension_run(run_id: str = "r1") -> dict:
    return {
        "run_id": run_id,
        "question": "How does hypertension affect life expectancy",
        "tasks": [{"id": "T1", "title": "hypertension and life expectancy",
                   "objective": "quantify the effect of hypertension on life expectancy",
                   "evidence_requirements": [
                       {"id": "T1.R1", "text": "hypertension and life expectancy"},
                       {"id": "T1.R2", "text": "hypertension mortality risk"}]}],
        "evidence": [
            {"id": "E1", "document_id": "PMC111", "chunk_id": "c1", "status": "accepted",
             "claim": "Hypertension is associated with reduced life expectancy in adults.",
             "support": "supports", "confidence": 0.9},
            {"id": "E2", "document_id": "PMC222", "chunk_id": "c2", "status": "accepted",
             "claim": "Uncontrolled hypertension lowers life expectancy by several years.",
             "support": "supports", "confidence": 0.85},
        ],
        "gaps": ["no lifetime risk model by age at onset"],
        "contradictions": [],
        "answer": {"summary": "Hypertension shortens life expectancy mainly "
                              "through cardiovascular mortality."},
    }


# ---------------------------------------------------------------------------
# Context-level continuity
# ---------------------------------------------------------------------------

def test_followup_context_marks_prior_questions_answered():
    api = build_api()
    api.record_run(hypertension_run())
    prep = api.prepare_run("how does hypertension lead to diseases")
    rendered = prep.context_text()

    # the prior investigation is DONE, not open — no re-derive signals
    assert "question [answered]" in rendered
    assert "How does hypertension affect life expectancy" in rendered
    open_questions = [line for line in rendered.splitlines()
                      if "question [open]" in line]
    assert open_questions == []

    # the established conclusion + prior findings are surfaced
    assert "Hypertension shortens life expectancy mainly through cardiovascular mortality." in rendered
    assert "ALREADY INVESTIGATED" in rendered
    assert "Hypertension is associated with reduced life expectancy" in rendered
    # memory is still labeled, never evidence
    assert "VERIFIED EVIDENCE" in rendered
    assert "NOT current evidence" in rendered


def test_short_followup_retrieves_prior_claims_via_session_questions():
    """'what about older adults?' shares almost no words with the prior
    findings — session-question expansion must still recall them."""
    api = build_api()
    api.record_run(hypertension_run())
    scored = retrieve_claims(
        api.store, "what about older adults",
        api.embedder, session_ids=[api.store.list_sessions()[0].id], top_k=5)
    texts = [s.claim.text for s in scored]
    assert any("life expectancy" in t for t in texts)


def test_conclusion_claim_surfaces_in_followup_context():
    api = build_api()
    api.record_run(hypertension_run())
    prep = api.prepare_run("how does hypertension lead to diseases")
    lines = [l for l in prep.context_text().splitlines()
             if "prior conclusions" in l or "last conclusion" in l]
    assert lines, "conclusion must be presented to the planner"


def test_new_question_stays_open_after_followup():
    """The follow-up's own questions get recorded and remain open/anwered
    consistently — the session continues, not restarts."""
    api = build_api()
    api.record_run(hypertension_run())
    stats2 = api.record_run({
        "run_id": "r2",
        "question": "how does hypertension lead to diseases",
        "tasks": [{"id": "T2", "title": "hypertension disease pathways",
                   "objective": "pathways from hypertension to end-organ diseases",
                   "evidence_requirements": [
                       {"id": "T2.R1", "text": "hypertension end-organ damage mechanisms"}]}],
        "evidence": [
            {"id": "E3", "document_id": "PMC333", "chunk_id": "c1", "status": "accepted",
             "claim": "Hypertension drives end-organ damage through arterial remodeling.",
             "support": "supports", "confidence": 0.8},
        ],
        "gaps": [], "contradictions": [],
        "answer": {"summary": "Hypertension causes disease via end-organ damage "
                              "and arterial remodeling."},
    })
    session = api.store.get_session(stats2.session_id)
    # the follow-up resumed the SAME session: title intact, rolling summary
    # now reflects the newest conclusion
    assert "life expectancy" in session.title
    assert session.summary == "Hypertension causes disease via end-organ damage " \
        "and arterial remodeling."
    answers = [q for q in api.store.get_questions(stats2.session_id)
               if q.status == "answered"]
    assert len(answers) >= 4   # run1 (3) + run2 (1) all investigated
    assert any("end-organ" in q.question for q in answers)


# ---------------------------------------------------------------------------
# End-to-end: the planner actually receives the continuity framing
# ---------------------------------------------------------------------------

class CapturingMaster:
    """Records every memory context handed to the planner."""

    def __init__(self):
        self.seen: list[str] = []

    async def plan(self, query, budget, memory_context=""):
        self.seen.append(memory_context or "")
        return MasterPlan(question=query, tasks=[
            ResearchTask(
                id="T1", title=f"tasks for: {query[:40]}",
                objective=query[:100], intent="answer the new question",
                evidence_requirements=[
                    EvidenceRequirement(id="T1.R1", text=query[:80],
                                        target_n=budget.evidence_target or 3)],
                stop_criteria=["evidence satisfied", "budget exhausted"],
            )])


class FakeWorker:
    async def run(self, task, budget, run_id=""):
        for req in task.evidence_requirements:
            req.add_evidence(VerifiedEvidence(
                id=f"E-{task.id}-{req.id}-1", task_id=task.id,
                requirement_id=req.id, document_id="PMC999", chunk_id="c1",
                excerpt="finding", claim="A follow-up finding was established.",
                support=SupportDirection.SUPPORTS, confidence=0.9,
                source=EvidenceSource.RETRIEVAL))
        task.finalize()
        return WorkerReport(run_id=run_id, task_id=task.id,
                            task_title=task.title, status=task.status.value,
                            stop_reason="satisfied", requirements=[],
                            evidence=[e for r in task.evidence_requirements
                                      for e in r.accepted])


class NoContradictions:
    async def detect(self, state):
        return []


class NoResolution:
    async def resolve(self, contradiction, state):
        return None


class FakeSynthesizer:
    async def synthesize(self, state):
        return SynthesisReport(
            summary="Follow-up answer.", sections=[], limitations=[],
            unresolved_gaps=[], unresolved_contradictions=[],
            resolved_contradictions=[], confidence=0.7, citations=[])


def _pipeline(api, master):
    return AgenticV3Pipeline(
        config=AppConfig(), master=master, worker=FakeWorker(),
        contradiction_agent=NoContradictions(),
        resolution_agent=NoResolution(),
        synthesizer=FakeSynthesizer(), memory=api)


def test_planner_receives_answered_framing_on_followup():
    api = build_api()
    api.record_run(hypertension_run())          # simulate completed run 1
    master = CapturingMaster()
    asyncio.run(_pipeline(api, master).answer("how does hypertension lead to diseases"))

    ctx = master.seen[-1]                       # the follow-up planner context
    assert ctx
    assert "question [answered]" in ctx
    assert "ALREADY INVESTIGATED" in ctx
    assert "do NOT re-derive" in ctx or "do NOT recreate" in ctx
    assert "Hypertension is associated with reduced life expectancy" in ctx


def test_planner_context_contains_planning_rules_only_when_memory():
    from src.agents.master import MasterOrchestratorAgent
    # no memory -> historical prompt, no rules block
    plain = MasterOrchestratorAgent._plan_prompt("q", memory_context="")
    assert "PLANNING RULES" not in plain
    # with memory -> rules present
    with_mem = MasterOrchestratorAgent._plan_prompt("q", memory_context="[PRIOR…]")
    assert "PLANNING RULES" in with_mem
    assert "do NOT recreate" in with_mem
    assert "Plan tasks ONLY for the NEW USER QUESTION" in with_mem


# ---------------------------------------------------------------------------
# Per-chat memory boundary: memory persists WITHIN a chat, never across chats
# ---------------------------------------------------------------------------

def _mem_api():
    from src.memory.api import MemoryAPI
    from src.memory.config import MemoryConfig

    return MemoryAPI.build(MemoryConfig(
        backend="memory", embedder="hash", embed_dim=256,
        context_tokens=1800))


def _sample_result(question="follow-up question?"):
    return {
        "question": question,
        "answer": {"summary": "summary", "sections": []},
        "tasks": [{"objective": "obj", "title": "t",
                   "evidence_requirements": [{"text": "requirement"}]}],
        "evidence": [],
    }


def test_same_conversation_resumes_same_research_session():
    """Follow-up turns in the SAME chat keep the same research session."""
    api = _mem_api()
    a1 = api.prepare_run("Does vitamin D lower blood pressure?",
                         conversation_id="conv-A")
    a2 = api.prepare_run("What dose of vitamin D is used in trials?",
                         conversation_id="conv-A")
    assert a2.session_id == a1.session_id


def test_new_conversation_gets_clean_slate():
    """A DIFFERENT chat gets a FRESH session even for a near-identical query -
    the global similarity resume must never cross conversations."""
    api = _mem_api()
    a = api.prepare_run("Does vitamin D lower blood pressure?",
                        conversation_id="conv-A")
    b = api.prepare_run("Does vitamin D lower blood pressure?",
                        conversation_id="conv-B")
    assert b.session_id != a.session_id
    # chat B has zero of chat A's committed memory
    api.record_run(_sample_result(), session_id=a.session_id,
                   conversation_id="conv-A")
    assert len(api.store.get_claims(session_id=a.session_id)) > 0
    assert len(api.store.get_claims(session_id=b.session_id)) == 0


def test_records_land_in_the_binding_conversation():
    """record_run persists into the session of the SAME chat only."""
    api = _mem_api()
    a = api.prepare_run("question one", conversation_id="conv-A")
    b = api.prepare_run("question one similar", conversation_id="conv-B")
    api.record_run(_sample_result(), session_id=a.session_id,
                   conversation_id="conv-A")
    api.record_run(_sample_result(), session_id=b.session_id,
                   conversation_id="conv-B")
    assert len(api.store.get_claims(session_id=a.session_id)) >= 1
    assert len(api.store.get_claims(session_id=b.session_id)) >= 1
    # memory is strictly session-scoped: reading A never sees B's claims
    a_ids = {cl.id for cl in api.store.get_claims(session_id=a.session_id)}
    b_ids = {cl.id for cl in api.store.get_claims(session_id=b.session_id)}
    assert a_ids.isdisjoint(b_ids)

def test_legacy_resume_without_conversation_still_works():
    """Callers that pass no conversation_id keep the historical behavior
    (global similarity resume) - CLI / non-chat entrypoints are unaffected."""
    api = _mem_api()
    s1 = api.prepare_run("Does vitamin D lower blood pressure?",
                         conversation_id=None)
    s2 = api.prepare_run("Does vitamin D lower blood pressure?",
                         conversation_id=None)
    assert s2.session_id == s1.session_id   # global resume by similarity
