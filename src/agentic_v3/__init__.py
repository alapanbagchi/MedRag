"""Agentic v3 - evidence-vetted multi-agent RAG retrieval.

A NEW flow, kept separate from src/agentic/ (v1) and src/agentic_v2/ (the
orchestrator research loop - both left untouched). Implements the V1 spec:

    QUERY
      -> MASTER ORCHESTRATOR          (decompose + evidence requirements +
                                       N thresholds + stop criteria)
      -> PARALLEL WORKERS             (UMLS enrichment, search-term
                                       selection, retrieval + context
                                       expansion, CRITIC verification gate,
                                       N independent papers per requirement
                                       with dedup, deep paper inspection,
                                       iterative retrieval until satisfied
                                       or budget exhausted)
      -> VERIFIED EVIDENCE
      -> CONTRADICTION AGENT          (cross-evidence conflicts/anomalies)
      -> RESOLUTION AGENT             (paper search tool; resolved or
                                       explicitly unresolved)
      -> FINAL EVIDENCE SET -> FINAL ANSWER (honest synthesis; gaps and
                                       unresolved conflicts are reported)

Layout:
  state.py        pipeline models + evidence-threshold/dedup logic
  master.py       the Master Orchestrator (Stage 2)
  umls.py         UMLS terminology enrichment (Stage 4)
  search.py       search-term selection (Stage 5)
  retriever.py    retriever tool + context expansion (Stages 6-7)
  critic.py       the CRITIC verification gate (Stage 8)
  worker.py       the Worker sub-orchestrator loop (Stages 3, 10-13, 17-18)
  deepinspect.py  deep paper inspection (Stage 11/14)
  contradiction.py  global contradiction agent (Stage 14)
  resolution.py     contradiction resolution agent (Stage 15)
  synthesize.py     final answer (Stage 24)
  pipeline.py       AgenticV3Pipeline - full run (Stages 1-24)
  events.py         structured event stream (v2-compatible JSONL)
"""

from src.agentic_v3.state import (
    AnswersTask,
    Contradiction,
    ContradictionKind,
    CriticRelevance,
    CriticVerdict,
    EvidenceRequirement,
    EvidenceSource,
    EvidenceStatus,
    FinalEvidenceSet,
    MasterPlan,
    RequirementReport,
    RequirementStatus,
    ResearchTask,
    ResolutionOutcome,
    ResolutionStatus,
    RetrievedPaper,
    RunBudget,
    SupportDirection,
    TaskStatus,
    TermConcept,
    V3RunState,
    VerifiedEvidence,
    WorkerReport,
)
from src.agentic_v3.master import MasterOrchestratorAgent, build_plan, fallback_task
from src.agentic_v3.umls import TerminologyEnricher
from src.agentic_v3.search import SearchTermPlanner, TaskSearchPlan
from src.agentic_v3.retriever import PaperRetrieverTool
from src.agentic_v3.critic import CriticAgent, is_promising_for_deep_inspection
from src.agentic_v3.worker import WorkerAgent
from src.agentic_v3.deepinspect import DeepInspector, verify_quote
from src.agentic_v3.replan import (
    FailureAnalysis,
    ReplanContext,
    ReplannerAgent,
    meaningfully_different,
)
from src.agentic_v3.contradiction import ContradictionAgent
from src.agentic_v3.resolution import ResolutionAgent
from src.agentic_v3.synthesize import FinalSynthesizer, SynthesisReport
from src.agentic_v3.pipeline import AgenticV3Pipeline
from src.agentic_v3.events import V3Events

__all__ = [
    "AnswersTask",
    "Contradiction",
    "ContradictionKind",
    "CriticRelevance",
    "CriticVerdict",
    "EvidenceRequirement",
    "EvidenceSource",
    "EvidenceStatus",
    "FailureAnalysis",
    "FinalEvidenceSet",
    "MasterPlan",
    "RequirementReport",
    "RequirementStatus",
    "ResearchTask",
    "ResolutionOutcome",
    "ResolutionStatus",
    "RetrievedPaper",
    "RunBudget",
    "SupportDirection",
    "TaskStatus",
    "TermConcept",
    "V3RunState",
    "VerifiedEvidence",
    "WorkerReport",
    "MasterOrchestratorAgent",
    "build_plan",
    "fallback_task",
    "TerminologyEnricher",
    "SearchTermPlanner",
    "TaskSearchPlan",
    "PaperRetrieverTool",
    "CriticAgent",
    "is_promising_for_deep_inspection",
    "WorkerAgent",
    "DeepInspector",
    "verify_quote",
    "ReplanContext",
    "ReplannerAgent",
    "meaningfully_different",
    "ContradictionAgent",
    "ResolutionAgent",
    "FinalSynthesizer",
    "SynthesisReport",
    "AgenticV3Pipeline",
    "V3Events",
]
