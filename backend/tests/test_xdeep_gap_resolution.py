"""Gap-resolution tests (post-contradiction gap pass + UI headers).

Covers:
  * compute_gaps finds unsatisfied requirements and describes what is missing;
  * resolve_requirement_gaps closes a gap with additional LOCAL retrievals
    (nothing reaches synthesis unverified);
  * a gap that survives the local budget goes to WEB, then is LISTED when
    still unsatisfied (recorded on state.gaps + GapResolution UNRESOLVED);
  * the graphical spine places gap_resolution between resolution and
    synthesis;
  * _answer_to_prose renders gaps/limitations/contradictions as '##' headers.
No real LLM calls - all stage/tool functions are faked.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.x_deepagents.state import (
    AnswersTask,
    EvidenceItem,
    GapResolutionStatus,
    Phase,
    ResearchRequirement,
    RunBudget,
    SupportDirection,
    VerifierVerdict,
    XDeepRunState,
)


def make_req(n_target=2, doc_id="PMC1"):
    req = ResearchRequirement(id="R1",
                              text="Does vitamin D lower blood pressure?",
                              target_n=n_target)
    if doc_id:
        it = EvidenceItem(id="R1.E1", requirement_id="R1", document_id=doc_id,
                          text="RCT found a reduction",
                          status="accepted",  # terminal, verified
                          verification="verified")
    return req


def accept_item(verdict=None, doc_id="PMC1", text="A randomized controlled "
                                                  "trial found a reduction."):
    item = EvidenceItem(id="x", requirement_id="R1", document_id=doc_id,
                        text=text)
    from src.x_deepagents.state import EvidenceStatus

    item.status = EvidenceStatus.ACCEPTED
    item.verdict = VerifierVerdict(
        evidence_id=item.id, requirement_id="R1", relevance="relevant",
        answers_task=AnswersTask.YES, support=SupportDirection.SUPPORTS,
        confidence=0.9, note="ok")
    return item


def make_state(*reqs):
    return XDeepRunState(run_id="r", question="q", requirements=list(reqs),
                         phase=Phase.RESOLUTION)


def make_satisfied_req(req_id="R1", n=1):
    """A requirement with n real verified SUPPORTS items (document_id set)."""
    req = ResearchRequirement(id=req_id, text="a", target_n=n)
    for i in range(n):
        req.add_item(accept_item(doc_id=f"PMC{i+1}"))
    req.derive_status()
    assert req.satisfied(), f"helper bug: target {n}"
    return req


async def _fake_verify(agent, requirement, item):
    item.submit_to_verifier()
    item.set_verdict(VerifierVerdict(
        evidence_id=item.id, requirement_id=item.requirement_id,
        relevance="relevant", answers_task=AnswersTask.YES,
        support=SupportDirection.SUPPORTS, confidence=0.9, note="accepted"))
    return item


def _fake_local(results, method="pgfts+pgvector"):
    async def _impl(payload, query, seen_chunks, top_k=5):
        return results, method, {"available": True, "empty_corpus": False,
                                 "stats": {"chunks": 10}}
    return _impl


# ---------------------------------------------------------------------------
# compute_gaps
# ---------------------------------------------------------------------------

def test_compute_gaps_lists_unsatisfied_requirements():
    from src.x_deepagents.agents.gap_fill import compute_gaps

    sat = make_satisfied_req("R1", n=1)
    unsat = ResearchRequirement(id="R2", text="b", target_n=3)
    state = make_state(sat, unsat)
    gaps = compute_gaps(state)
    assert len(gaps) == 1
    req, text = gaps[0]
    assert req.id == "R2"
    assert "R2" in text and "3" in text


def test_compute_gaps_empty_when_all_satisfied():
    from src.x_deepagents.agents.gap_fill import compute_gaps

    req = make_satisfied_req("R1", n=1)
    assert compute_gaps(make_state(req)) == []


# ---------------------------------------------------------------------------
# resolve_requirement_gaps: local close
# ---------------------------------------------------------------------------

def test_gap_resolution_closes_via_local_retrieval(monkeypatch):
    """A gap closes with more LOCAL retrievals - all new evidence verified."""
    from src.x_deepagents.agents import gap_fill as gf

    hits = [{
        "rank": 1, "chunk_id": "g1", "document_id": "PMC2",
        "section": "Results", "unit_kind": "paragraph", "score": 0.8,
        "methods": ["pgfts", "pgvector"], "retrieval_method": "pgfts+pgvector",
        "text": "A randomized controlled trial found a reduction in SBP.",
    }]
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve",
                        _fake_local([hits[0]]))
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item",
                        _fake_verify)
    async def _plan(*a, **k):
        return ["new query"]
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._plan_gap_queries",
                        _plan)

    req = ResearchRequirement(id="R1", text="Does vitamin D lower BP?",
                              target_n=1)
    state = make_state(req)
    resolutions = asyncio.run(gf.resolve_requirement_gaps(state))

    assert len(resolutions) == 1
    res = resolutions[0]
    assert res.status is GapResolutionStatus.RESOLVED_LOCAL
    assert res.evidence_ids  # the new item is recorded
    assert req.verified_items()          # new evidence is verified
    assert state.gaps == []              # nothing listed


def test_gap_resolution_lists_when_unresolved(monkeypatch):
    """A gap that local AND web cannot close is LISTED, not fabricated."""
    from src.x_deepagents.agents import gap_fill as gf

    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve",
                        _fake_local([]))                       # no local hits
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item",
                        _fake_verify)
    async def _plan(*a, **k):
        return ["q1", "q2"]
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._plan_gap_queries",
                        _plan)

    async def _empty_web(query, top_k=None, trusted_only=True):
        return json.dumps({"query": query, "available": True, "results": [],
                           "dropped_blocked": 0, "dropped_unverified": 0})

    monkeypatch.setattr(
        "src.x_deepagents.tools.searxng.searxng_search_impl", _empty_web)

    req = ResearchRequirement(id="R1", text="Does X work?", target_n=2)
    state = make_state(req)
    asyncio.run(gf.resolve_requirement_gaps(state))

    assert len(state.gap_resolutions) == 1
    res = state.gap_resolutions[0]
    assert res.status is GapResolutionStatus.UNRESOLVED
    assert len(state.gaps) == 1            # the gap is listed
    assert "R1" in state.gaps[0]


def test_gap_resolution_web_closes_when_local_empty(monkeypatch):
    """Local finds nothing -> web closes the gap (RESOLVED_WEB)."""
    from src.x_deepagents.agents import gap_fill as gf

    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve",
                        _fake_local([]))
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item",
                        _fake_verify)
    async def _plan(*a, **k):
        return ["web query"]
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._plan_gap_queries",
                        _plan)

    async def _web_hit(query, top_k=None, trusted_only=True):
        return json.dumps({"query": query, "available": True,
                           "dropped_blocked": 0, "dropped_unverified": 0,
                           "results": [{
                               "title": "WHO",
                               "url": "https://www.who.int/x",
                               "snippet": "WHO guidance: vitamin D reduces "
                                          "blood pressure in adults.",
                               "engine": "google", "trust": "trusted",
                           }]})

    monkeypatch.setattr(
        "src.x_deepagents.tools.searxng.searxng_search_impl", _web_hit)
    monkeypatch.setattr("src.x_deepagents.agents.stages.judge_site_reliability",
                        lambda *a, **k: {"reliability": "high"})

    req = ResearchRequirement(id="R1", text="Does X work?", target_n=1)
    state = make_state(req)
    asyncio.run(gf.resolve_requirement_gaps(state))

    res = state.gap_resolutions[0]
    assert res.status is GapResolutionStatus.RESOLVED_WEB
    assert state.gaps == []
    web_items = [i for i in req.verified_items()
                 if i.retrieval_method.startswith("web:")]
    assert web_items


def test_gap_budget_limits_local_rounds(monkeypatch):
    """gap_local_exhausted stops the local loop even with more queries."""
    from src.x_deepagents.agents import gap_fill as gf

    calls = {"n": 0}

    async def _counting_local(payload, query, seen_chunks, top_k=5):
        calls["n"] += 1
        return [], "pgfts", {"available": True, "empty_corpus": False,
                             "stats": {}}

    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve",
                        _counting_local)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item",
                        _fake_verify)
    async def _plan(*a, **k):
        return ["q1"]
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._plan_gap_queries",
                        _plan)
    async def _empty_web(*a, **k):
        return json.dumps({"available": True, "results": []})
    monkeypatch.setattr(
        "src.x_deepagents.tools.searxng.searxng_search_impl", _empty_web)

    req = ResearchRequirement(id="R1", text="Q?", target_n=3)
    state = make_state(req)
    state.budget = RunBudget(max_gap_local_rounds=2, max_gap_web_searches=0)
    asyncio.run(gf.resolve_requirement_gaps(state))
    assert state.budget.gap_local_rounds_used == 2   # local budget respected
    assert state.budget.gap_web_searches_used == 0   # web budget 0 -> no web


# ---------------------------------------------------------------------------
# Graph spine
# ---------------------------------------------------------------------------

def test_graph_places_gap_resolution_between_resolution_and_synthesis():
    from src.x_deepagents.graph import build_research_graph

    g = build_research_graph()
    edges = {(e.source, e.target) for e in g.get_graph().edges}
    assert ("resolution", "gap_resolution") in edges
    assert ("gap_resolution", "synthesize") in edges


# ---------------------------------------------------------------------------
# UI headers
# ---------------------------------------------------------------------------

def test_answer_prose_renders_gap_and_limitation_headers():
    """Gap/limitation/unresolved blocks render as headers; RESOLVED
    contradictions do NOT get a separate header (they are inline in the
    section body, per the user contract)."""
    from src.x_deepagents.agents.stages import _answer_to_prose

    prose = _answer_to_prose({
        "summary": "summary",
        "sections": [{"heading": "BP", "body": "text"}],
        "unresolved_gaps": ["gap one", "gap two"],
        "limitations": ["only observational"],
        "resolved_contradictions": ["resolved one"],  # must NOT be a header
        "unresolved_contradictions": ["unresolved one"],
    })
    for header in ("## Unresolved Gaps", "## Limitations",
                   "## Unresolved Contradictions"):
        assert header in prose, header
    # resolved contradictions are inline in sections, never a separate header
    assert "## Resolved Contradictions" not in prose
    assert "- gap one" in prose
    assert "- only observational" in prose



# ---------------------------------------------------------------------------
# LATENT gap probing: satisfied requirements with content holes must still
# probe + search (the DASH-diet scenario the user reported: 0 gaps logged,
# no web search, but synthesis listed unresolved gaps).
# ---------------------------------------------------------------------------

def make_satisfied_holey_req(target_n=3):
    """A requirement that is numerically SATISFIED but whose evidence leaves
    content-level holes (e.g. mechanisms / individual foods unanswered)."""
    from src.x_deepagents.state import EvidenceStatus

    req = ResearchRequirement(id="R1",
                              text="Effect of dietary patterns on blood pressure",
                              target_n=target_n)
    for i in range(target_n):
        it = EvidenceItem(id=f"E{i}", requirement_id="R1",
                          document_id=f"PMC{i}",
                          text="DASH diet reduces blood pressure in RCTs.")
        it.status = EvidenceStatus.ACCEPTED
        it.verdict = VerifierVerdict(
            evidence_id=it.id, requirement_id="R1", relevance="relevant",
            answers_task=AnswersTask.YES, support=SupportDirection.SUPPORTS,
            confidence=0.9, note="ok")
        req.add_item(it)
    req.derive_status()
    assert req.satisfied()
    return req


async def _probe_gaps(agent, req):
    return [
        {"question": "What are the mechanisms by which the DASH diet lowers BP?",
         "queries": ["DASH diet mechanisms blood pressure reduction"]},
        {"question": "Which individual foods and nutrients lower blood pressure?",
         "queries": ["individual foods nutrients blood pressure effects"]},
    ]




async def _complete_gate(agent, question, evidence):
    return []   # fully answered


async def _partial_gate(agent, question, evidence):
    return ["safety/harms of the intervention",
            "long-term durability beyond follow-up"]

def test_satisfied_requirement_still_probes_latent_gaps(monkeypatch):
    """compute_gaps returns [] for a satisfied requirement (hard path skips),
    but resolve_requirement_gaps STILL probes content holes and runs
    searches - the user's reported bug: '0 gaps' yet listable gaps."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps

    assert compute_gaps_for(make_satisfied_holey_req()) == []

    captured = {"local": [], "web": []}

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        captured["local"].append(query)
        if "mechanisms" in query:
            return [{
                "rank": 1, "chunk_id": "m1", "document_id": "PMC9",
                "section": "Discussion", "unit_kind": "paragraph", "score": 0.8,
                "methods": ["pgfts"], "retrieval_method": "pgfts",
                "text": "DASH mechanisms involve RAS suppression and sodium handling.",
            }], "pgfts", {}
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        captured["web"].append(query)
        return json.dumps({"available": True, "results": [],
                           "dropped_blocked": 0, "dropped_unverified": 0})

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps",
                        _probe_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    async def _rel(*a, **k):
        return {"reliability": "high"}
    monkeypatch.setattr("src.x_deepagents.agents.stages.judge_site_reliability", _rel)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._gap_completeness_check",
                        _complete_gate)

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="How does diet affect hypertension?",
                          requirements=[req], phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=2,
                                           max_gap_web_searches=2))
    resolutions = asyncio.run(resolve_requirement_gaps(state))

    # the probe actually searched: local fired for both, AND web fired even
    # for the gap local closed (web augmentation; not fallback-only)
    assert captured["local"], "latent probe must run local retrieval"
    assert len(captured["web"]) >= 2, captured["web"]
    # mechanism closed locally, foods gap listed
    def _status_for(gaps_res, needle):
        for gr in gaps_res:
            if gr.gap.startswith(needle):
                return gr.status.value
        return None
    assert _status_for(resolutions, "What are the mechanisms") == "resolved_local"
    assert _status_for(resolutions, "Which individual foods") == "unresolved"
    assert len(state.gaps) == 1          # the food gap is listed
    # new VERIFIED probe items are on the requirement
    assert any(i.retrieval_method.endswith(":probe") for i in req.verified_items())


def compute_gaps_for(req):
    from src.x_deepagents.agents.gap_fill import compute_gaps

    from src.x_deepagents.state import XDeepRunState
    st = XDeepRunState(run_id="r", question="q", requirements=[req])
    return compute_gaps(st)


def test_probe_env_knobs_gate_latent_search(monkeypatch):
    """XDEEP_GAP_PROBE=0 disables latent probing entirely."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps

    monkeypatch.setenv("XDEEP_GAP_PROBE", "0")
    monkeypatch.setenv("XDEEP_WEB_AUGMENT", "0")
    assert gf.probe_enabled() is False
    assert gf.web_augment_enabled() is False

    captured = {"web": []}

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        captured["web"].append(query)
        return json.dumps({"available": True, "results": []})

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps",
                        _probe_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=1))
    resolutions = asyncio.run(resolve_requirement_gaps(state))
    assert resolutions == []              # probe off -> nothing to do
    assert captured["web"] == []          # no web search fired



# ---------------------------------------------------------------------------
# WEB AUGMENTATION: web search must run even when local closes the gap, and
# even when the probe finds no gaps at all (XDEEP_WEB_AUGMENT).
# ---------------------------------------------------------------------------

def test_web_fires_even_when_local_closes_the_gap(monkeypatch):
    """Per-gap: after local closes a content hole, a trust-gated web round STILL
    runs (bounded) - the user contract 'web search after local retrieval'."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps

    web_calls = []

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        return [{"rank": 1, "chunk_id": "m1", "document_id": "PMC9",
                 "section": "Discussion", "unit_kind": "paragraph",
                 "score": 0.8, "methods": ["pgfts"], "retrieval_method": "pgfts",
                 "text": "DASH mechanisms involve RAS suppression and sodium handling."}], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        web_calls.append(query)
        return json.dumps({"available": True, "results": [], "dropped_blocked": 0,
                           "dropped_unverified": 0})

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps", _probe_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._gap_completeness_check", _complete_gate)
    async def _rel(*a, **k):
        return {"reliability": "high"}
    monkeypatch.setattr("src.x_deepagents.agents.stages.judge_site_reliability", _rel)

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=3))
    resolutions = asyncio.run(resolve_requirement_gaps(state))
    # THE USER'S COMPLAINT: web must fire even when local closed the gap.
    # (The stub returns empty results, so no new IDs are credited, but the
    # searches themselves are what matter here.)
    assert web_calls, "web must fire even when local closed the gap"


def test_web_augmentation_on_no_probe_gaps(monkeypatch):
    """Even when the probe finds NO content holes, a bounded web round still
    fires per satisfied requirement to bring current guidelines / extra info."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps

    web_calls = []

    async def _no_gaps(agent, req):
        return []                       # probe: evidence already complete

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        web_calls.append(query)
        return json.dumps({"available": True, "results": [], "dropped_blocked": 0,
                           "dropped_unverified": 0})

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps", _no_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    monkeypatch.setattr("src.x_deepagents.agents.stages.judge_site_reliability",
                        lambda *a, **k: {"reliability": "high"})

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=2))
    resolutions = asyncio.run(resolve_requirement_gaps(state))
    assert web_calls, "web augmentation must fire even with no probe gaps"


def test_web_augment_knob_suppresses_web(monkeypatch):
    """XDEEP_WEB_AUGMENT=0 makes web fallback-only (no gratuitous searches)."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps

    monkeypatch.setenv("XDEEP_WEB_AUGMENT", "0")
    web_calls = []

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        # local closes the mechanism gap
        if "mechanisms" in query:
            return [{"rank": 1, "chunk_id": "m1", "document_id": "PMC9",
                     "section": "D", "unit_kind": "paragraph", "score": 0.8,
                     "methods": ["pgfts"], "retrieval_method": "pgfts",
                     "text": "DASH mechanisms involve RAS suppression."}], "pgfts", {}
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        web_calls.append(query)
        return json.dumps({"available": True, "results": [], "dropped_blocked": 0,
                           "dropped_unverified": 0})

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps", _probe_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._gap_completeness_check",
                        _complete_gate)

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=3))
    resolutions = asyncio.run(resolve_requirement_gaps(state))
    # XDEEP_WEB_AUGMENT=0 -> no GRATUITOUS web for the gap local closed
    assert not any("mechanisms" in q for q in web_calls), web_calls
    # the open 'foods' gap may still use web as a FALLBACK - that is correct
    assert any(g.status.value == "resolved_local" for g in resolutions)



# ---------------------------------------------------------------------------
# DIVE DEEP: web evidence uses the FETCHED page text, not just the snippet.
# ---------------------------------------------------------------------------

def test_web_evidence_uses_fetched_page_text(monkeypatch):
    """A web hit whose page fetch succeeds must be verified + reliability-judged
    against the REAL page text; the EvidenceItem.text is the fetched body, not
    the 500-char search snippet."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps

    seen_texts = []

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        return json.dumps({"available": True, "dropped_blocked": 0,
                           "dropped_unverified": 0, "results": [{
                               "title": "WHO",
                               "url": "https://www.who.int/diet",
                               "snippet": "WHO fact sheet: DASH diet snippet only.",
                               "engine": "google", "trust": "trusted",
                           }]})

    async def _page(agent, req):
        return [{"question": "Current WHO diet guidance?",
                 "queries": ["WHO dietary guidance hypertension"]}]

    # the page FETCH returns the real article text (this is the deep dive)
    async def _fetch(url, max_chars=10000, timeout_s=12.0):
        return ("The DASH diet reduced systolic blood pressure by 11 mmHg in "
                "hypertensive adults over an 8-week randomized trial, per the "
                "full WHO guideline article body.")

    async def _verify(agent, requirement, item):
        seen_texts.append(item.text)
        item.submit_to_verifier()
        item.set_verdict(VerifierVerdict(
            evidence_id=item.id, requirement_id=item.requirement_id,
            relevance="relevant", answers_task=AnswersTask.YES,
            support=SupportDirection.SUPPORTS, confidence=0.9, note="ok"))
        return item

    import src.x_deepagents.tools.fetch as fetch_mod
    monkeypatch.setattr(fetch_mod, "fetch_page_text", _fetch)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps", _page)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._gap_completeness_check", _complete_gate)
    async def _rel(*a, **k):
        return {"reliability": "high"}
    monkeypatch.setattr("src.x_deepagents.agents.stages.judge_site_reliability", _rel)

    req = make_satisfied_holey_req(target_n=2)
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=1))
    resolutions = asyncio.run(resolve_requirement_gaps(state))

    # verification received the FULL page body (deep dive), not the snippet
    assert any("11 mmHg" in t and "snippet only" not in t for t in seen_texts), seen_texts
    # and a verified web item carried the fetched page text as its evidence
    assert any("11 mmHg" in it.text for it in req.verified_items()
               if it.retrieval_method.startswith("web:"))



# ---------------------------------------------------------------------------
# COMPLETENESS GATE + POST-SYNTHESIS RECONCILIATION (honest "resolved")
# ---------------------------------------------------------------------------

def test_gate_marks_partially_resolved_with_remaining_facets(monkeypatch):
    """Evidence found but important facets missing -> PARTIALLY_RESOLVED and
    each missing facet appended to state.gaps (the old code over-claimed
    'resolved, 0 listed' for exactly this case - potassium safety, duration)."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps
    from src.x_deepagents.state import GapResolutionStatus

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        if "mechanisms" in query:
            return [{"rank": 1, "chunk_id": "m1", "document_id": "PMC9",
                     "section": "D", "unit_kind": "paragraph", "score": 0.8,
                     "methods": ["pgfts"], "retrieval_method": "pgfts",
                     "text": "DASH mechanisms involve RAS suppression."}], "pgfts", {}
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        return json.dumps({"available": True, "results": [], "dropped_blocked": 0,
                           "dropped_unverified": 0})

    # gate says: evidence found but safety + duration remain unanswered
    async def _partial(agent, question, evidence):
        return ["safety/harms of the intervention",
                "long-term durability beyond follow-up"]

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps", _probe_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._gap_completeness_check", _partial)

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=1))
    resolutions = asyncio.run(resolve_requirement_gaps(state))
    mech = next(g for g in resolutions if "mechanisms" in g.gap)
    assert mech.status is GapResolutionStatus.PARTIALLY_RESOLVED
    # each remaining facet is LISTED in state.gaps - honest "listed" count
    assert any("safety" in g for g in state.gaps)
    assert any("durability" in g for g in state.gaps)
    assert len(state.gaps) >= 2


def test_fully_answered_gap_is_resolved(monkeypatch):
    """Gate confirming full answer -> status stays RESOLVED_* and no listing."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps
    from src.x_deepagents.state import GapResolutionStatus

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        if "mechanisms" in query:
            return [{"rank": 1, "chunk_id": "m1", "document_id": "PMC9",
                     "section": "D", "unit_kind": "paragraph", "score": 0.8,
                     "methods": ["pgfts"], "retrieval_method": "pgfts",
                     "text": "DASH mechanisms fully explained incl. RAS and endothelium."}], "pgfts", {}
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        return json.dumps({"available": True, "results": [], "dropped_blocked": 0,
                           "dropped_unverified": 0})

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps", _probe_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._gap_completeness_check",
                        _complete_gate)

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=3))
    resolutions = asyncio.run(resolve_requirement_gaps(state))
    mech = next(g for g in resolutions if "mechanisms" in g.gap)
    assert mech.status is GapResolutionStatus.RESOLVED_LOCAL
    # no partial listing for the fully-answered gap
    assert not any("still missing" in g for g in state.gaps)


def test_synthesis_reconciles_unresolved_gaps_into_run(monkeypatch):
    """The synthesizer's own unresolved_gaps become the FINAL run.gaps even if
    the gap pass under-listed them (the user's report: '2 resolved' yet the
    answer lists holes). synthesize_node must fold them in + emit event."""
    import asyncio

    import src.x_deepagents.graph as G
    from src.x_deepagents.agents import stages as sm

    run = XDeepRunState(run_id="r", question="How does diet affect hypertension?",
                        phase=Phase.SYNTHESIS,
                        requirements=[make_satisfied_holey_req()],
                        budget=RunBudget())
    # gap pass previously said resolved; synthesizer lists real holes
    from src.x_deepagents.state import GapResolution

    gr = GapResolution(requirement_id="R1",
                       gap="Potassium-based salt substitutes: benefit and safety",
                       status="resolved_web", evidence_ids=["E1"])
    run.gap_resolutions = [gr]

    async def fake_synth(agent, state):
        # returns (prose, JSON payload with unresolved_gaps)
        return ("answer prose \u3014cite:PMC1\u3015", {
            "summary": "s",
            "sections": [{"heading": "h", "body": "b"}],
            "unresolved_gaps": [
                "Potassium salt-substitute safety in renal patients unknown",
                "Portfolio-type diets not studied",
            ],
            "limitations": [],
            "citations": [],
        })

    monkeypatch.setattr(sm, "make_synthesizer",
                        lambda model=None: SimpleNamespace())
    monkeypatch.setattr(sm, "synthesize_answer_with_data", fake_synth)

    events = []
    G.set_event_sink(lambda e, f: events.append(e))
    try:
        out = asyncio.run(G.synthesize_node({"state": run}))
    finally:
        G.set_event_sink(None)
    run = out["state"]
    # synthesizer-listed gaps are now in run.gaps
    assert any("Potassium salt-substitute safety" in g for g in run.gaps), run.gaps
    assert any("Portfolio-type" in g for g in run.gaps)
    # the previously-claimed resolved gap is downgraded to PARTIALLY_RESOLVED
    # (evidence was found, but the synthesizer still lists facets of it)
    assert run.gap_resolutions[0].status.value == "partially_resolved"
    assert "gaps_reconciled" in events
    assert "synthesis_done" in events
    # summary still rendered
    assert run.answer



# ---------------------------------------------------------------------------
# SYNTHESIS FALLBACK: provider failure must yield an evidence-grounded answer,
# never a dead "(no answer - synthesis failed)" placeholder.
# ---------------------------------------------------------------------------

def _verified_run(text="The DASH diet lowered SBP by 11 mmHg in hypertensive adults."):
    from src.x_deepagents.state import EvidenceStatus

    req = ResearchRequirement(id="R1", text="diet and BP", target_n=1)
    it = EvidenceItem(id="R1.E1", requirement_id="R1", document_id="PMC10262995",
                      text=text)
    it.status = EvidenceStatus.ACCEPTED
    it.verdict = VerifierVerdict(evidence_id=it.id, requirement_id="R1",
                                 relevance="relevant", answers_task=AnswersTask.YES,
                                 support=SupportDirection.SUPPORTS,
                                 confidence=1.0, note="ok")
    req.add_item(it)
    req.derive_status()
    return XDeepRunState(run_id="r", question="How does diet affect hypertension?",
                         requirements=[req])


def test_fallback_synthesize_grounds_answer_in_evidence():
    from src.x_deepagents.agents.stages import fallback_synthesize

    fb = fallback_synthesize(_verified_run())
    assert "11 mmHg" in fb
    assert "\u3014cite:PMC10262995\u3015" in fb   # bridge resolves to [n]
    assert "Summary" in fb
    assert "faithful extraction" in fb.lower()


def test_synthesize_with_data_falls_back_when_agent_raises(monkeypatch):
    """If the LLM synthesizer raises, the user gets the deterministic
    evidence-grounded answer - not a dead placeholder."""
    import asyncio

    from src.x_deepagents.agents.stages import synthesize_answer_with_data

    class _Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("provider 429")

    prose, data = asyncio.run(
        synthesize_answer_with_data(_Boom(), _verified_run()))
    assert prose
    assert "11 mmHg" in prose
    assert data is None
    assert "(no answer" not in prose


def test_gate_failure_does_not_fabricate_facet(monkeypatch):
    """A failing completeness gate must not inject a fake 'compl
    failed...' facet into run.gaps - the synthesizer reconciles real ones."""
    from src.x_deepagents.agents import gap_fill as gf
    from src.x_deepagents.agents.gap_fill import resolve_requirement_gaps

    async def _boom_gate(agent, question, evidence):
        raise RuntimeError("gate provider down")

    async def _fake_local(payload, query, seen_chunks, top_k=5):
        if "mechanisms" in query:
            return [{"rank": 1, "chunk_id": "m1", "document_id": "PMC9",
                     "section": "D", "unit_kind": "paragraph", "score": 0.8,
                     "methods": ["pgfts"], "retrieval_method": "pgfts",
                     "text": "DASH mechanisms involve RAS suppression."}], "pgfts", {}
        return [], "pgfts", {}

    async def _fake_web(query, top_k=None, trusted_only=True):
        return json.dumps({"available": True, "results": [], "dropped_blocked": 0,
                           "dropped_unverified": 0})

    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._probe_latent_gaps", _probe_gaps)
    monkeypatch.setattr("src.x_deepagents.agents.stages.verify_item", _fake_verify)
    monkeypatch.setattr("src.x_deepagents.agents.worker._local_retrieve", _fake_local)
    monkeypatch.setattr("src.x_deepagents.tools.searxng.searxng_search_impl", _fake_web)
    monkeypatch.setattr("src.x_deepagents.agents.gap_fill._gap_completeness_check",
                        _boom_gate)

    req = make_satisfied_holey_req()
    state = XDeepRunState(run_id="r", question="q", requirements=[req],
                          phase=Phase.RESOLUTION,
                          budget=RunBudget(max_gap_local_rounds=1,
                                           max_gap_web_searches=1))
    resolutions = asyncio.run(resolve_requirement_gaps(state))
    # no fabricated "completeness check failed" facet in run.gaps
    assert not any("completeness check failed" in g for g in state.gaps)
    # evidence found; gate failed but gap is not lying about a fake facet
    mech = next(g for g in resolutions if "mechanisms" in g.gap)
    assert mech.status.value in ("resolved_local", "resolved_web")
