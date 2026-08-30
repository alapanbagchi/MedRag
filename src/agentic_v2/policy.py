"""Agentic v2 — runtime policy (the deterministic state machine).

The orchestrator proposes ONE action per turn; this module is the AUTHORITATIVE
layer that:

  * derives the current research PHASE from ``ResearchState``,
  * computes the LEGAL action set for the current state,
  * REPAIRS illegal / exhausted decisions deterministically (never trusting the
    LLM to enforce prerequisites),
  * snapshots state BEFORE/AFTER an action and evaluates whether it made
    meaningful PROGRESS,
  * produces a stable STRATEGY KEY so semantically-identical attempts can be
    recognized and blocked from immediate repetition.

It depends only on ``state.py`` (no LLM, no corpus), so it is fully unit-testable.
"""

from __future__ import annotations

import enum
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.agentic_v2.state import ActionType, ObjectiveStatus, ResearchState


class ResearchPhase(str, enum.Enum):
    EXPLORE = "explore"      # no objectives, or objectives still lack documents
    EXCAVATE = "excavate"    # documents exist; candidate evidence not yet located
    ASSESS = "assess"        # candidate passages exist / need verification
    ANSWER = "answer"        # objectives sufficiently investigated -> synthesize


_TERMINAL_ACTIONS = (ActionType.SYNTHESIZE, ActionType.STOP)

# Preferred action order per phase (used only for deterministic fallback/pivot).
_PHASE_ORDER: Dict[ResearchPhase, List[ActionType]] = {
    ResearchPhase.EXPLORE: [
        ActionType.DECOMPOSE, ActionType.ENRICH, ActionType.GLOBAL_RETRIEVE,
        ActionType.STOP,
    ],
    ResearchPhase.EXCAVATE: [
        ActionType.READ_DOCUMENT, ActionType.FIND_SECTIONS,
        ActionType.GLOBAL_RETRIEVE, ActionType.ENRICH, ActionType.VERIFY,
        ActionType.STOP,
    ],
    ResearchPhase.ASSESS: [
        ActionType.VERIFY, ActionType.GLOBAL_RETRIEVE, ActionType.ENRICH,
        ActionType.STOP,
    ],
    ResearchPhase.ANSWER: [
        ActionType.SYNTHESIZE, ActionType.STOP,
    ],
}


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------

def determine_phase(state: ResearchState) -> ResearchPhase:
    """Derive the current phase from the research state (monotonic-ish)."""
    if not state.objectives:
        return ResearchPhase.EXPLORE
    if all(o.status != ObjectiveStatus.OPEN for o in state.objectives):
        return ResearchPhase.ANSWER
    if not state.documents and not state.candidates:
        return ResearchPhase.EXPLORE
    # material exists; nothing verified/located yet -> excavate, else assess
    if not state.evidence and not state.candidates:
        return ResearchPhase.EXCAVATE
    return ResearchPhase.ASSESS


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------

def _synthesizable(state: ResearchState) -> bool:
    if state.evidence:
        return True
    return any(o.status != ObjectiveStatus.OPEN for o in state.objectives)


def legal_actions(state: ResearchState) -> Set[ActionType]:
    """The deterministic set of actions that are legal for the current state."""
    legal: Set[ActionType] = set()
    has_objectives = bool(state.objectives)
    has_material = bool(state.documents or state.candidates)

    legal.add(ActionType.DECOMPOSE)         # always legal (bootstrap + refinement)
    if has_objectives:
        legal.add(ActionType.ENRICH)

    # GLOBAL_RETRIEVE is bounded by its own budget (enforced here, not in prompt).
    if has_objectives and state.global_retrieves_used() < state.max_global_retrieves:
        legal.add(ActionType.GLOBAL_RETRIEVE)

    if has_material:
        legal.add(ActionType.READ_DOCUMENT)
        legal.add(ActionType.FIND_SECTIONS)

    if has_objectives and has_material:
        legal.add(ActionType.VERIFY)

    if _synthesizable(state):
        legal.add(ActionType.SYNTHESIZE)

    legal.add(ActionType.STOP)              # always legal
    return legal


def fallback_action(state: ResearchState, exclude_action: Optional[ActionType] = None) -> ActionType:
    """Deterministic next-best action (used to repair/force a pivot)."""
    legal = legal_actions(state)
    phase = determine_phase(state)
    for action in _PHASE_ORDER[phase]:
        if action in legal and action != exclude_action:
            return action
    return ActionType.STOP


# ---------------------------------------------------------------------------
# Strategy keys (normalized, so semantically-identical attempts are recognized)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _norm_tokens(text: Any) -> Tuple[str, ...]:
    return tuple(sorted(_TOKEN_RE.findall((text or "").lower())))


def _digest(tokens: Iterable[str]) -> str:
    return hashlib.sha1("|".join(tokens).encode("utf-8")).hexdigest()[:12]


def strategy_key(decision: Any, state: ResearchState) -> str:
    """A stable key identifying one strategy attempt (semantic, not literal)."""
    action = getattr(decision, "action", ActionType.STOP)
    obj = (getattr(decision, "objective_id", "") or "").strip()
    query = (getattr(decision, "query", "") or "").strip()
    doc_id = (getattr(decision, "document_id", "") or "").strip()

    if action == ActionType.GLOBAL_RETRIEVE:
        q = query
        if not q and obj:
            o = state.objective(obj)
            q = (o.statement if o else "") or ""
        return f"GLOBAL_RETRIEVE:{obj}:{_digest(_norm_tokens(q))}"

    if action in (ActionType.READ_DOCUMENT, ActionType.FIND_SECTIONS):
        return f"{action.value}:{obj}:{doc_id}"

    if action == ActionType.VERIFY:
        # include the candidate set signature so "verify NEW evidence" is a new
        # strategy, but "re-verify the same evidence" is blocked.
        sig = sorted({(c.chunk_id or c.document_id) for c in state.candidates} |
                     {(d.chunk_id or d.document_id) for d in state.documents})
        return f"VERIFY:{obj}:{_digest(sig)}"

    if action == ActionType.ENRICH:
        return f"ENRICH:{obj}"

    if action == ActionType.DECOMPOSE:
        return f"DECOMPOSE:{obj}"

    return f"{action.value}:{obj}"


# ---------------------------------------------------------------------------
# Progress snapshot + evaluation
# ---------------------------------------------------------------------------

@dataclass
class ProgressOutcome:
    progress: bool
    summary: str = ""


def progress_snapshot(state: ResearchState) -> Dict[str, Any]:
    """A compact, comparable snapshot of everything that counts as progress."""
    return {
        "objectives": tuple(
            (o.id, o.status.value, o.gap, tuple(sorted(o.caveats)),
             o.enriched_query, tuple(sorted(o.synonyms)))
            for o in state.objectives
        ),
        "doc_keys": frozenset(
            d.chunk_id if d.chunk_id else (d.document_id, d.text[:120])
            for d in state.documents
        ),
        "cand_keys": frozenset(
            c.chunk_id if c.chunk_id else (c.document_id, c.text[:120])
            for c in state.candidates
        ),
        "evid_keys": frozenset(
            (e.objective_id, e.chunk_id, e.excerpt[:80]) for e in state.evidence
        ),
        "gaps": tuple(sorted(state.gaps)),
        "contradictions": tuple(sorted(state.contradictions)),
        "sections_seen": frozenset(state.sections_seen),
        "searched": tuple(state.searched_queries),
        "terminal": state.terminal,
        "answer": (state.final_answer or {}).get("summary", ""),
    }


def evaluate_progress(
    decision: Any,
    before: Dict[str, Any],
    after: Dict[str, Any],
    result: Any,
) -> ProgressOutcome:
    """Whether the action moved the state forward in a meaningful way."""
    action = getattr(decision, "action", ActionType.STOP)

    if action in _TERMINAL_ACTIONS:
        if after["terminal"]:
            return ProgressOutcome(True, "terminal reached")
        return ProgressOutcome(False, f"{action.value} did not terminate")

    if action == ActionType.DECOMPOSE:
        if after["objectives"] != before["objectives"]:
            return ProgressOutcome(True, "objectives added/refined")
        return ProgressOutcome(False, "no objective change")

    if action == ActionType.GLOBAL_RETRIEVE:
        new = after["doc_keys"] - before["doc_keys"]
        if new:
            return ProgressOutcome(True, f"{len(new)} new document(s)")
        return ProgressOutcome(False, "no new documents")

    if action == ActionType.READ_DOCUMENT:
        new = after["doc_keys"] - before["doc_keys"]
        if new:
            return ProgressOutcome(True, "new document content read")
        return ProgressOutcome(False, "no new content")

    if action == ActionType.FIND_SECTIONS:
        new = after["sections_seen"] - before["sections_seen"]
        if new:
            return ProgressOutcome(True, f"{len(new)} new section(s) exposed")
        return ProgressOutcome(False, "sections already known")

    if action == ActionType.ENRICH:
        if after["objectives"] != before["objectives"]:
            return ProgressOutcome(True, "objective enriched")
        return ProgressOutcome(False, "no enrichment change")

    if action == ActionType.VERIFY:
        if after["evid_keys"] != before["evid_keys"]:
            return ProgressOutcome(True, "new verified evidence")
        if after["objectives"] != before["objectives"]:
            return ProgressOutcome(True, "objective status/gap changed")
        if after["contradictions"] != before["contradictions"]:
            return ProgressOutcome(True, "contradictions changed")
        return ProgressOutcome(False, "verify produced no change")

    return ProgressOutcome(False, "unknown action")


# ---------------------------------------------------------------------------
# Decision validation / repair
# ---------------------------------------------------------------------------

def validate_or_repair(
    decision: Any,
    state: ResearchState,
    phase: Optional[ResearchPhase] = None,
    legal: Optional[Set[ActionType]] = None,
) -> Tuple[Any, bool, str]:
    """Return (decision, repaired, reason) enforcing legality + no-repeat.

    ``decision`` is returned unchanged when legal; otherwise it is repaired to a
    deterministic fallback action. STOP is the ultimate safe terminal fallback.
    """
    phase = phase or determine_phase(state)
    legal = legal or legal_actions(state)
    original = getattr(decision, "action", ActionType.STOP)
    reasons: List[str] = []
    repaired = False

    action = original
    if action not in legal:
        reasons.append(f"{action.value} illegal in phase {phase.value}")
        action = fallback_action(state, exclude_action=action)
        repaired = True
    elif action != ActionType.STOP:
        key = strategy_key(decision, state)
        if key in state.exhausted_strategies:
            reasons.append(f"strategy exhausted: {key}")
            action = fallback_action(state, exclude_action=action)
            repaired = True

    if repaired:
        decision = decision.model_copy(update={"action": action})
    return decision, repaired, "; ".join(reasons)
