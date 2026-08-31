"""Agentic v2 — intent verification (the VERIFY action's agent).

The verifier's ONLY job is to check INTENT: for each candidate passage (a full
structural unit — paragraph / table / figure), is it relevant /
partially_relevant / not_relevant to the objective's intent? It does NOT assess
population match, causal strength, clinical efficacy, or contradiction — those
are out of scope by design.

Design: ONE PARAGRAPH PER LLM CALL, SEQUENTIALLY. Each candidate unit is
verified on its own — the FULL unit text goes into the prompt (no truncation),
one call at a time, one validated JSON verdict per call — and the per-passage
verdicts are aggregated into one ObjectiveVerdict. The pipeline scales this
action's time budget by the number of candidates precisely because the calls
are sequential (see AgenticV2Pipeline._action_timeout).

The output is an ObjectiveVerdict which the executor folds back into the
ResearchState (updating the objective status and any gaps).
"""

from __future__ import annotations
from src.prompts.load import load_prompt

import logging
from typing import Any, List

from pydantic import BaseModel, Field

from src.agentic_v2.state import (
    CandidatePassage,
    EvidenceQuality,
    ObjectiveStatus,
    ResearchObjective,
)

logger = logging.getLogger("src.agentic_v2.verify")


class PassageAssessment(BaseModel):
    document_id: str = ""
    chunk_id: str = ""
    section: str = ""
    relevance: str = ""             # relevant | partially_relevant | not_relevant
    quality: EvidenceQuality = EvidenceQuality.UNKNOWN   # derived from relevance
    support: str = "neutral"        # supports | neutral (derived from relevance)
    confidence: float = 0.0
    note: str = ""                  # why (intent match / mismatch)


class PassageVerdict(BaseModel):
    """Per-passage LLM output (one paragraph per call)."""

    relevance: str = "not_relevant"  # relevant | partially_relevant | not_relevant
    confidence: float = 0.0
    note: str = ""                   # why the passage matches / misses


class ObjectiveVerdict(BaseModel):
    objective_id: str = ""
    status: ObjectiveStatus = ObjectiveStatus.UNRESOLVED
    confidence: float = 0.0
    gap: str = ""                   # what remains unsupported
    caveats: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    population_match: str = "unknown"   # yes | no | partial | unknown
    outcome_match: str = "unknown"      # yes | no | partial | unknown
    assessments: List[PassageAssessment] = Field(default_factory=list)


VERIFIER_SYSTEM_PROMPT = load_prompt('agentic_v2', 'verifier.txt')


# Cap on how many passages one VERIFY action verifies (actions.py bounds its
# candidate list too); each cap unit is one full-text LLM call.
_MAX_PASSAGES = 6

_RELEVANCES = ("relevant", "partially_relevant", "not_relevant")


def _passage_text(p: CandidatePassage) -> str:
    """The FULL unit text — no truncation, layout preserved (tables keep rows)."""
    return (p.text or "").strip()


def _is_full_document(p: CandidatePassage) -> bool:
    """True when a candidate passage is actually a WHOLE PAPER, not a unit.

    Recognizes the explicit markers (section '(full document)' or a
    document-kind flag) plus a defensive heuristic: a passage with no chunk id
    whose text is implausibly large for one unit is treated as a whole
    document and must never be sent to the verifier.
    """
    if (p.section or "").strip() == "(full document)":
        return True
    if getattr(p, "unit_kind", "") == "document":
        return True
    if not p.chunk_id and (p.text or "").strip():
        # Structural units are restored per-chunk; an id-less blob of many
        # thousands of tokens is almost certainly a concatenated paper.
        if len((p.text or "").split()) > 1200:
            return True
    return False


def _clip_full_document(p: CandidatePassage) -> CandidatePassage:
    """Last-resort: reduce a whole-paper passage to its most relevant paragraph.

    Never discards the document outright - it keeps the single most relevant
    paragraph so the evidence still reaches the model (bounded, not a 50k-char
    blob). Returns the original passage unchanged when nothing can be clipped.
    """
    import re

    if not _is_full_document(p):
        return p
    text = (p.text or "").strip()
    parts = [
        part.strip()
        for part in re.split(r"\n{2,}|(?<=\.)\n", text)
        if part and part.strip() and len(part.split()) >= 5
    ]
    if not parts:
        return p
    # Heuristic: pick the first sufficiently substantive paragraph (no query
    # context here), capped at 600 words to stay a bounded unit.
    best = max(parts, key=len)
    if len(best.split()) > 600:
        words = best.split()
        best = " ".join(words[:600]) + " [...clipped to first 600 words]"
    return p.model_copy(update={
        "text": best,
        "section": (p.section or "").replace("(full document)", "(relevant paragraph)"),
        "chunk_id": p.chunk_id or f"{p.document_id}:clip",
    })


def _objective_block(objective: ResearchObjective) -> str:
    return (
        f"OBJECTIVE {objective.id}: {objective.statement}\n"
        f"INTENT: {objective.intent or '(unspecified)'}\n"
        f"EVIDENCE REQUIRED: {', '.join(objective.evidence_required) if objective.evidence_required else '(unspecified)'}"
    )


def _passage_prompt(objective: ResearchObjective, passage: CandidatePassage) -> str:
    """Prompt for exactly ONE candidate document, with its FULL text."""
    return (
        _objective_block(objective)
        + f"\n\nCANDIDATE PASSAGE (doc={passage.document_id}, chunk={passage.chunk_id}, "
          f"section={passage.section}):\n{_passage_text(passage)}"
        + "\n\nIs this single passage relevant to the objective's intent? "
          "Return the JSON intent verdict for this passage."
    )


def _aggregate(
    objective: ResearchObjective,
    assessments: List[PassageAssessment],
) -> ObjectiveVerdict:
    """Fold per-passage verdicts into the overall ObjectiveVerdict."""
    relevant = [a for a in assessments if a.relevance == "relevant"]
    partial = [a for a in assessments if a.relevance == "partially_relevant"]
    kept = relevant + partial

    if relevant:
        status = ObjectiveStatus.SUPPORTED
    elif partial:
        status = ObjectiveStatus.PARTIALLY_SUPPORTED
    else:
        status = ObjectiveStatus.UNRESOLVED

    confidence = max((a.confidence for a in kept), default=0.0)

    gap = ""
    if status is ObjectiveStatus.UNRESOLVED and assessments:
        gap = (
            f"none of the {len(assessments)} retrieved passage(s) matched this "
            "objective's intent"
        )
        reasons: List[str] = []
        for a in assessments:
            note = (a.note or "").strip()
            if note and note not in reasons:
                reasons.append(note)
        if reasons:
            gap += ": " + "; ".join(reasons[:3])
        gap = gap[:240]

    return ObjectiveVerdict(
        objective_id=objective.id,
        status=status,
        confidence=round(confidence, 4),
        gap=gap,
        assessments=list(assessments),
    )


class ObjectiveVerifier:
    """Checks whether candidate passages match the objective's intent.

    ONE paragraph per LLM call, verified sequentially in candidate order;
    per-passage verdicts are aggregated into a single ObjectiveVerdict.
    """

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="verifier")
        self.agent = Agent(
            self.model,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            name="objective_verifier",
        )

    async def verify(
        self,
        objective: ResearchObjective,
        passages: List[CandidatePassage],
    ) -> ObjectiveVerdict:
        """Verify each candidate passage individually (sequential), then aggregate."""
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        # SAFETY NET: the executor expands full-document units into their
        # relevant paragraphs before they reach the verifier, so verify()
        # should only ever see structural units. If one still slips through,
        # NEVER drop it - clip it to its most relevant paragraph so evidence
        # survives the backstop too.
        chosen = [
            _clip_full_document(p) if _is_full_document(p) else p
            for p in passages
        ][:_MAX_PASSAGES]
        clipped = sum(1 for p in passages if _is_full_document(p))
        if clipped:
            logger.warning(
                "verifier: clipped %d full-document passage(s) for %s to their "
                "most relevant paragraph", clipped, objective.id,
            )
            trace.log("verifier_clipped_full_documents", objective=objective.id,
                      count=clipped)
        trace.agent("objective_verifier", output_type="ObjectiveVerdict",
                    meta={"objective": objective.id, "passages": len(passages),
                          "mode": "one-paragraph-per-call"})
        trace.bullet(
            f"verifier: {len(chosen)} passage(s), one full-text LLM call each"
        )

        assessments: List[PassageAssessment] = []
        for i, p in enumerate(chosen, start=1):
            prompt = _passage_prompt(objective, p)
            try:
                v = await ask_structured(
                    self.agent,
                    prompt,
                    PassageVerdict,
                    label=f"objective_verifier:{i}:{p.document_id or p.chunk_id}",
                    # Tiny JSON output, but keep headroom so pre-JSON chatter
                    # never truncates the verdict into a retry loop.
                    max_tokens=max(2048, self.config.agent_max_tokens),
                    # Fail a stalled passage faster than the default 4 attempts
                    # so one bad call cannot eat the whole action budget.
                    max_attempts=2,
                )
            except Exception as exc:
                # FAILURE != IRRELEVANCE: a failed call must never be recorded
                # as not_relevant (that would silently drop evidence). Surface
                # it so VERIFY reports failure and the policy can retry later.
                logger.warning(
                    "verifier failed for %s/%s (%s): %s",
                    p.document_id or "?", p.section or "?", p.chunk_id or "?", exc,
                )
                raise
            relevance = v.relevance if v.relevance in _RELEVANCES else "not_relevant"
            try:
                confidence = max(0.0, min(1.0, float(v.confidence or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            assessments.append(PassageAssessment(
                document_id=p.document_id,
                chunk_id=p.chunk_id,
                section=p.section,
                relevance=relevance,
                support="supports" if relevance == "relevant" else "neutral",
                confidence=confidence,
                note=(v.note or "").strip(),
            ))
            trace.log("verifier_passage", index=i, objective=objective.id,
                      document_id=p.document_id, chunk_id=p.chunk_id,
                      section=p.section, relevance=relevance,
                      confidence=confidence)

        return _aggregate(objective, assessments)
