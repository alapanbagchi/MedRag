"""WorkflowRules - the enforceable guardrails (layer A of the autonomy model).

The agreed contract, in code, as a single deterministic validator the graph
consults after every LLM decision:

  * The deep agent chooses the TOOL and HOW (retrieve now, postgres_search
    later)  ->  this module never dictates the tool.
  * The WORKFLOW ORDER is fixed and mandatory -> this module DOES enforce it.

Rules encoded here:
  1. DECOMPOSE before any retrieval.
  2. Every retrieved chunk must pass the VERIFIER (chunk -> verifier);
     unverified material is structurally excluded from contradiction +
     synthesis.
  3. Contradiction analysis runs on the FINAL verified set, and every
     detected contradiction goes through RESOLUTION (unresolved is reported,
     never fabricated away).
  4. Synthesis is evidence-gated: it may receive ONLY verified ids, and
     citations are repaired against the verified set.
  5. Budgets are hard.

Nothing here calls an LLM and nothing mutates state (pure predicates +
repair helpers), so it is unit-testable against arbitrary agent output.
"""

from __future__ import annotations

from typing import Iterable

from src.agents.state import (
    EvidenceItem,
    EvidenceStatus,
    Phase,
    RequirementStatus,
    ResearchRequirement,
    ResolutionStatus,
    RunBudget,
    SupportDirection,
    XDeepRunState,
)


# ---------------------------------------------------------------------------
# Tool-selection domain (autonomy B)
# ---------------------------------------------------------------------------

# Known tools. postgres_search is the PRIMARY pg-chunk retrieval capability
# (over medrag.chunks) and is live - the agent may choose it, exactly like
# retrieve (parquet hybrid). The workflow order is unchanged.
# (paper_inspect was removed: no implementation ever existed; full-paper
# inspection lives in the worker's deep-inspect path, not in the registry.)
KNOWN_TOOLS = frozenset({"retrieve", "umls_lookup", "postgres_search"})
TOOLS_AVAILABLE_NOW = frozenset({"retrieve", "umls_lookup", "postgres_search"})


def is_known_tool(name: str) -> bool:
    return name in KNOWN_TOOLS


def tool_available_now(name: str) -> bool:
    """True if the tool has a real implementation attached today.

    postgres_search (pg-chunk retrieval) is live and selectable.
    """
    return name in TOOLS_AVAILABLE_NOW


# ---------------------------------------------------------------------------
# Phase ordering (the mandatory workflow)
# ---------------------------------------------------------------------------

# The phase order is the fixed spine of the graph. The LLM chooses actions
# inside a phase; it can never jump the spine.
_ORDER = {p: i for i, p in enumerate(Phase.ordered())}


def phase_index(phase: Phase) -> int:
    return _ORDER[phase]


def can_advance_to(current: Phase, nxt: Phase) -> bool:
    """True if `nxt` is the exact next phase (no skipping allowed)."""
    i, j = _ORDER[current], _ORDER[nxt]
    return j == i + 1


# ---------------------------------------------------------------------------
# Rule 2: chunk -> verifier
# ---------------------------------------------------------------------------

def every_retrieved_item_was_verified(requirement: ResearchRequirement) -> bool:
    """Every non-REJECTED candidate has a verifier verdict attached.

    RETRIEVED / UNDER_REVIEW items are NOT yet verified and must never be
    treated as evidence.
    """
    for item in requirement.items:
        if item.status in (EvidenceStatus.RETRIEVED, EvidenceStatus.UNDER_REVIEW):
            return False
        if item.verified and item.verdict is None:
            return False
    return True


def verified_evidence_items(state: XDeepRunState) -> list[EvidenceItem]:
    """ACCEPTED + CONTRADICTORY only - the sole synthesis/contradiction input.

    REJECTED / RETRIEVED / UNDER_REVIEW are structurally excluded here.
    """
    return state.verified_items()


# ---------------------------------------------------------------------------
# Rule 3: contradiction analysis on the final set + resolution
# ---------------------------------------------------------------------------

def contradiction_input_ready(state: XDeepRunState) -> bool:
    """Contradiction analysis only runs once the verified set is final.

    It is ready once every requirement has stopped gathering (satisfied or
    exhausted) and there is at least one verified item.
    """
    if not state.requirements:
        return False
    if not state.verified_items():
        return False
    for req in state.requirements:
        if req.status not in (RequirementStatus.SATISFIED,
                              RequirementStatus.EXHAUSTED):
            return False
    return True


def every_contradiction_resolved_or_reported(state: XDeepRunState) -> list[str]:
    """Every detected contradiction must have a resolution (unresolved counts
    as *reported*, never silently dropped). Returns the list of offending ids
    (empty == all contradictions are either resolved or explicitly reported)."""
    unhandled: list[str] = []
    for c in state.contradictions:
        if c.resolution is None:
            unhandled.append(c.id)
        elif c.resolution.status is ResolutionStatus.UNRESOLVED                 and not c.resolution.explanation:
            # unresolved is acceptable ONLY if it is reported (has an
            # explanation saying why it could not be resolved)
            unhandled.append(c.id)
    return unhandled


# ---------------------------------------------------------------------------
# Rule 4: evidence-gated synthesis + citation repair
# ---------------------------------------------------------------------------

def synthesis_input(state: XDeepRunState) -> list[EvidenceItem]:
    """The synthesizer may cite ONLY verified items."""
    return verified_evidence_items(state)


def repair_citations(state: XDeepRunState, cited_ids: Iterable[str]) -> list[str]:
    """Deterministic citation repair: keep only ids that exist and are
    verified; drop hallucinated / unverified ones. Preserves input order.
    """
    verified = state.verified_ids()
    out: list[str] = []
    for cid in cited_ids:
        if cid in verified and cid not in out:
            out.append(cid)
    return out


def answer_citations_are_verified(state: XDeepRunState, cited_ids: Iterable[str]) -> bool:
    """True if every cited id is present AND verified (no holes)."""
    verified = state.verified_ids()
    return all(cid in verified for cid in cited_ids)


# ---------------------------------------------------------------------------
# Rule 5: hard budgets
# ---------------------------------------------------------------------------

def budget_exhausted(budget: RunBudget) -> bool:
    return budget.exhausted()


# ---------------------------------------------------------------------------
# One-stop validation the graph calls after each node
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
    """Raised when a workflow rule would be broken. The graph treats this as
    a hard stop / repair signal - the LLM can never produce a state that
    violates the rules."""


def validate_state(state: XDeepRunState) -> list[str]:
    """Run every rule against the state; return a list of human-readable
    violations (empty == state is rule-compliant). Pure check, no mutation.
    """
    problems: list[str] = []

    # Rule 2: no unverified item may count as evidence.
    for req in state.requirements:
        if not every_retrieved_item_was_verified(req):
            problems.append(
                f"requirement {req.id}: unverified item(s) present "
                f"(status not verdict-terminal)"
            )
        for item in req.items:
            if item.verified and item.verdict is None:
                problems.append(
                    f"item {item.id}: verified but has no verifier verdict"
                )

    # Rule 3: contradictions must be resolved or reported.
    unhandled = every_contradiction_resolved_or_reported(state)
    if unhandled:
        problems.append(
            f"unhandled contradiction(s): {', '.join(unhandled)}"
        )

    # Rule 4: synthesis input is verified-only (checked at use, see
    # synthesis_input / repair_citations; also ensure verified items always
    # carry a supporting verdict direction).
    for item in state.verified_items():
        if item.verdict is not None                 and item.verdict.support is SupportDirection.NEUTRAL:
            problems.append(
                f"item {item.id}: verified with NEUTRAL support - must be "
                "SUPPORTS or CONTRADICTS after verification"
            )

    return problems
