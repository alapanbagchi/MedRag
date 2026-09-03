"""x_deepagents state + rules tests (offline, no LLM).

Covers the mandatory workflow gates:
  * Rule 2: retrieved chunk -> verifier (unverified is never evidence).
  * Rule 3: contradiction analysis + resolution on the final set.
  * Rule 4: evidence-gated synthesis + deterministic citation repair.
  * Rule 5: hard budgets.
  * Phase ordering: the spine cannot be skipped.
  * Tool autonomy: the agent chooses the tool (retrieve/postgres_search),
    the order is fixed.
"""

from __future__ import annotations

from src.agents import rules
from src.agents.rules import (
    RuleViolation,
    answer_citations_are_verified,
    budget_exhausted,
    can_advance_to,
    contradiction_input_ready,
    every_contradiction_resolved_or_reported,
    repair_citations,
    synthesis_input,
    validate_state,
)
from src.agents.state import (
    AnswersTask,
    Contradiction,
    ContradictionKind,
    EvidenceItem,
    ResolutionOutcome,
    ResolutionStatus,
    ResearchRequirement,
    RunBudget,
    SupportDirection,
    VerifierVerdict,
    XDeepRunState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_item(evidence_id="E1", req_id="R1", status=None):
    item = EvidenceItem(
        id=evidence_id,
        run_id="RUN",
        requirement_id=req_id,
        chunk_id="C1",
        document_id="PMC1",
        text="Full expanded unit text ...",
        source_query="noise reduction",
        retrieval_method="hybrid:bm25+dense",
    )
    if status is not None:
        item.status = status
    return item


def verify_and_accept(item, support=SupportDirection.SUPPORTS):
    """Push an item through the verifier gate to ACCEPTED."""
    item.submit_to_verifier()
    verdict = VerifierVerdict(
        evidence_id=item.id,
        requirement_id=item.requirement_id,
        relevance="relevant",
        answers_task=AnswersTask.YES,
        support=support,
        confidence=0.9,
        note="answers the requirement",
    )
    item.set_verdict(verdict)
    return item


def make_state(*items):
    req = ResearchRequirement(id="R1", text="association X -> Y", target_n=2)
    for it in items:
        req.add_item(it)
    req.derive_status()
    return XDeepRunState(run_id="RUN", question="q", requirements=[req])


# ---------------------------------------------------------------------------
# Rule 2: chunk -> verifier
# ---------------------------------------------------------------------------

def test_retrieved_item_must_pass_verifier_before_evidence():
    item = make_item()
    state = make_state(item)
    # a freshly retrieved candidate is NOT evidence and flags a violation
    assert rules.every_retrieved_item_was_verified(state.requirements[0]) is False
    assert validate_state(state)  # non-empty violations

    # once verified + accepted it becomes evidence
    verify_and_accept(item)
    assert item.verified is True
    assert rules.every_retrieved_item_was_verified(state.requirements[0]) is True
    assert validate_state(state) == []


def test_unverified_never_reaches_synthesis():
    item = make_item()
    # only verified items may be synthesised/cited
    assert synthesis_input(make_state(item)) == []
    verify_and_accept(item)
    assert [i.id for i in synthesis_input(make_state(item))] == ["E1"]


def test_set_verdict_transitions_derived():
    item = make_item()
    item.submit_to_verifier()
    item.set_verdict(VerifierVerdict(
        evidence_id="E1", requirement_id="R1",
        answers_task=AnswersTask.NO, support=SupportDirection.NEUTRAL))
    # answers_task=NO -> rejected
    assert item.status.value == "rejected"
    assert item.verified is False


def test_contradictory_verdict_is_kept_as_evidence():
    # ANSWERS_TASK=YES + CONTRADICTS -> CONTRADICTORY (real evidence,
    # must reach contradiction analysis, not be dropped)
    item = make_item("E1")
    item.submit_to_verifier()
    item.set_verdict(VerifierVerdict(
        evidence_id="E1", requirement_id="R1",
        answers_task=AnswersTask.YES, support=SupportDirection.CONTRADICTS,
        confidence=0.8))
    assert item.status.value == "contradictory"
    assert item.verified is True
    assert [i.id for i in synthesis_input(make_state(item))] == ["E1"]


def test_verdict_scope_mismatch_rejected():
    item = make_item("E1")
    item.submit_to_verifier()
    try:
        item.set_verdict(VerifierVerdict(
            evidence_id="E_OTHER", requirement_id="R1",
            answers_task=AnswersTask.YES, support=SupportDirection.SUPPORTS))
        assert False, "scope mismatch must raise"
    except ValueError:
        pass


def test_verified_item_requires_non_neutral_support():
    item = make_item()
    verify_and_accept(item, support=SupportDirection.NEUTRAL)
    # NEUTRAL support after acceptance is a rule violation
    assert validate_state(make_state(item)) != []


# ---------------------------------------------------------------------------
# Rule 3: contradiction analysis + resolution on the final set
# ---------------------------------------------------------------------------

def test_contradiction_input_ready_only_on_final_set():
    state = make_state()
    # no requirements / no items -> not ready
    assert contradiction_input_ready(state) is False
    item = verify_and_accept(make_item())
    state = make_state(item)
    state.requirements[0].derive_status()  # coverage 1 < target 2 -> partially
    assert contradiction_input_ready(state) is False  # still gathering
    # satisfy it: a SECOND, DISTINCT SUPPORTS paper (independence) brings
    # coverage to 2 >= target 2, so the requirement becomes SATISFIED
    other = verify_and_accept(make_item("E2", "R1"), support=SupportDirection.SUPPORTS)
    other.document_id = "PMC2"
    state.requirements[0].add_item(other)
    state.requirements[0].derive_status()  # coverage 2 -> satisfied
    assert state.requirements[0].status.value == "satisfied"
    assert contradiction_input_ready(state) is True


def test_every_contradiction_resolved_or_reported():
    state = make_state(verify_and_accept(make_item()))
    state.contradictions.append(Contradiction(id="C1", claim="X lowers Y"))
    # no resolution -> flagged
    assert every_contradiction_resolved_or_reported(state) == ["C1"]

    c = state.contradictions[0]
    c.resolution = ResolutionOutcome(status=ResolutionStatus.RESOLVED, explanation="context-dependent")
    assert every_contradiction_resolved_or_reported(state) == []

    # unresolved without explanation -> still flagged (must be reported)
    c.resolution = ResolutionOutcome(status=ResolutionStatus.UNRESOLVED, explanation="")
    assert every_contradiction_resolved_or_reported(state) == ["C1"]
    # unresolved WITH explanation -> acceptably reported
    c.resolution = ResolutionOutcome(status=ResolutionStatus.UNRESOLVED,
                                     explanation="conflicting RCTs, no synthesis possible yet")
    assert every_contradiction_resolved_or_reported(state) == []


# ---------------------------------------------------------------------------
# Rule 4: evidence-gated synthesis + citation repair
# ---------------------------------------------------------------------------

def test_repair_citations_drops_unverified_and_hallucinated():
    verified = verify_and_accept(make_item("E1"))
    state = make_state(verified)
    repaired = repair_citations(state, ["E1", "E_HALLUCINATED", "E2"])
    assert repaired == ["E1"]
    assert answer_citations_are_verified(state, ["E1"]) is True
    assert answer_citations_are_verified(state, ["E1", "FAKE"]) is False


def test_repair_citations_preserves_order_and_dedupes():
    a = verify_and_accept(make_item("E1"))
    b = verify_and_accept(make_item("E2"), support=SupportDirection.CONTRADICTS)
    state = make_state(a, b)
    assert repair_citations(state, ["E2", "E1", "E2", "E1"]) == ["E2", "E1"]


# ---------------------------------------------------------------------------
# Rule 1 + phase ordering: the spine cannot be skipped
# ---------------------------------------------------------------------------

def test_phase_order_is_sequential():
    assert can_advance_to("decompose", "retrieve") is True
    assert can_advance_to("decompose", "verify") is False        # skip
    assert can_advance_to("decompose", "synthesis") is False     # skip far
    assert can_advance_to("retrieve", "verify") is True
    assert can_advance_to("verify", "contradiction") is True
    assert can_advance_to("contradiction", "resolution") is True
    assert can_advance_to("resolution", "gap_resolution") is True
    assert can_advance_to("gap_resolution", "synthesis") is True
    assert can_advance_to("resolution", "synthesis") is False    # must pass gap resolution
    assert can_advance_to("verify", "resolution") is False       # must pass contradiction


# ---------------------------------------------------------------------------
# Tool autonomy (B): the agent picks the tool, the order is fixed
# ---------------------------------------------------------------------------

def test_tool_choice_is_agent_sided():
    # both retrieval paths are known AND wired today (agent may choose either)
    assert rules.is_known_tool("retrieve")
    assert rules.is_known_tool("postgres_search")
    assert rules.tool_available_now("retrieve") is True
    assert rules.tool_available_now("postgres_search") is True


# ---------------------------------------------------------------------------
# Rule 5: hard budgets
# ---------------------------------------------------------------------------

def test_budget_is_hard():
    b = RunBudget(max_searches=3, max_retrieval_rounds=3)
    assert budget_exhausted(b) is False
    b.searches_used = 3
    assert budget_exhausted(b) is True


def test_validate_state_detects_mixed_violations():
    # rejected + accepted (leaves REJECTED as terminal, fine)
    rej = make_item("E1")
    rej.submit_to_verifier()
    rej.set_verdict(VerifierVerdict(
        evidence_id="E1", requirement_id="R1",
        answers_task=AnswersTask.NO, support=SupportDirection.NEUTRAL))
    acc = verify_and_accept(make_item("E2"))
    state = make_state(rej, acc)
    # both verdict-terminal -> no rule-2 violation
    assert validate_state(state) == []

    # add an unhandled contradiction -> flagged
    state.contradictions.append(Contradiction(id="C1", claim="c"))
    assert validate_state(state) != []


# --- replanner meaningfully-different guard --------------------------------

def test_meaningfully_different_drops_exact_repeats():
    from src.agents.agents.stages import meaningfully_different

    tried = ["Does vitamin D lower blood pressure?"]
    out = meaningfully_different(["Does vitamin D lower blood pressure?"], tried)
    assert out == []


def test_meaningfully_different_keeps_new_angle():
    from src.agents.agents.stages import meaningfully_different

    tried = ["vitamin d blood pressure"]
    out = meaningfully_different(
        ["cholecalciferol supplementation hypertension trial"], tried)
    assert out  # new content tokens (cholecalciferol, hypertension...)


def test_meaningfully_different_drops_subset_tokens():
    from src.agents.agents.stages import meaningfully_different

    tried = ["vitamin d supplementation blood pressure"]
    # all tokens already covered by the tried query -> dropped
    out = meaningfully_different(["vitamin d blood pressure"], tried)
    assert out == []


def test_meaningfully_different_caps_at_three():
    from src.agents.agents.stages import meaningfully_different

    out = meaningfully_different(
        ["alpha one", "beta two", "gamma three", "delta four"], [])
    assert len(out) <= 3


# --- semantic-query sanitization --------------------------------------------

def test_sanitize_strips_boolean_operators():
    from src.agents.agents.stages import _sanitize_semantic_query

    assert _sanitize_semantic_query("Cardiac arrest AND hypertension") == "Cardiac arrest hypertension"
    assert _sanitize_semantic_query("hypertension OR high blood pressure") == "hypertension high blood pressure"
    assert _sanitize_semantic_query(
        '\"vitamin D\" AND \"blood pressure\"') == "vitamin D blood pressure"


def test_sanitize_keeps_natural_language():
    from src.agents.agents.stages import _sanitize_semantic_query

    q = "does vitamin D supplementation lower systolic blood pressure in older adults?"
    assert _sanitize_semantic_query(q) == q


def test_semantic_queries_dedup_and_limit():
    from src.agents.agents.stages import _semantic_queries

    out = _semantic_queries([
        "Cardiac arrest AND hypertension",
        "Cardiac arrest AND hypertension",
        "does vitamin D lower BP",
        "",
    ])
    assert "Cardiac arrest hypertension" in out
    assert len(out) <= 4


def test_planner_prompt_has_semantic_notice():
    from src.agents.agents.stages import (
        _SEMANTIC_NOTICE,
        search_plan_prompt,
    )

    assert "SEMANTIC" in _SEMANTIC_NOTICE
    assert "AND" not in _SEMANTIC_NOTICE.split() or "boolean" in _SEMANTIC_NOTICE
    # the notice is actually attached to the planner prompt
    prompt = search_plan_prompt("task", "requirement", pool=[])
    assert "SEMANTIC" in prompt