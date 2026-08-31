"""Agentic v2 — orchestrator-controlled research loop.

A NEW flow, separate from ``src/agentic/`` (which is left untouched). Instead
of a fixed decompose -> enrich -> retrieve -> verify pipeline, an Orchestrator
Agent drives the research process one decision at a time against a persistent
``ResearchState`` (the research notebook):

    state -> orchestrator decides ONE action -> executor runs it ->
    state is updated -> repeat -> SYNTHESIZE when evidence is sufficient.

Actions: DECOMPOSE, ENRICH (UMLS/MeSH query enrichment), GLOBAL_RETRIEVE,
READ_DOCUMENT, FIND_SECTIONS, VERIFY (intent only), SYNTHESIZE, STOP.

Layout:
  state.py        persistent research state + objective/evidence models
  orchestrator.py the decision agent (exactly one ActionDecision per turn)
  verify.py       objective-level evidence verifier
  synthesize.py   final synthesis from verified evidence
  actions.py      deterministic executors for the eight actions
  pipeline.py     the run loop + terminal/budget handling
"""

from src.agentic_v2.state import (
    ActionType,
    EvidenceQuality,
    ObjectiveStatus,
    ResearchObjective,
    ResearchState,
    RetrievedDocument,
    CandidatePassage,
    VerifiedEvidence,
    StrategyAttempt,
)
from src.agentic_v2.orchestrator import ActionDecision, OrchestratorAgent
from src.agentic_v2.actions import ActionExecutor, ActionResult
from src.agentic_v2.verify import ObjectiveVerifier, ObjectiveVerdict
from src.agentic_v2.synthesize import FinalSynthesizer, SynthesisReport
from src.agentic_v2.policy import (
    ResearchPhase,
    determine_phase,
    legal_actions,
    fallback_action,
    strategy_key,
    progress_snapshot,
    evaluate_progress,
    validate_or_repair,
    ProgressOutcome,
)
from src.agentic_v2.pipeline import AgenticV2Pipeline

__all__ = [
    "ActionType",
    "EvidenceQuality",
    "ObjectiveStatus",
    "ResearchObjective",
    "ResearchState",
    "RetrievedDocument",
    "CandidatePassage",
    "VerifiedEvidence",
    "StrategyAttempt",
    "ActionDecision",
    "OrchestratorAgent",
    "ActionExecutor",
    "ActionResult",
    "ObjectiveVerifier",
    "ObjectiveVerdict",
    "FinalSynthesizer",
    "SynthesisReport",
    "ResearchPhase",
    "determine_phase",
    "legal_actions",
    "fallback_action",
    "strategy_key",
    "progress_snapshot",
    "evaluate_progress",
    "validate_or_repair",
    "ProgressOutcome",
    "AgenticV2Pipeline",
]
