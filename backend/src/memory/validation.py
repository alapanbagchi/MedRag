"""Memory + Context layer — the provenance gate and write validation.

This is the safety core of the layer. Its job is to mechanically guarantee
the MedPat invariant:

    Memory is not medical evidence.

Concretely it decides, for every candidate memory:

  1. CLASSIFY  — what kind of information is this? (claim / preference /
                 transient / question; which ProvenanceClass?)
  2. GATE      — can it be persisted as an EVIDENCE_DERIVED_CLAIM? Only with
                 >=1 validated verified-evidence reference.
  3. SANITIZE  — prompt-injection / control text defense-in-depth; stored
                 text is always rendered as quoted data by context.py, this
                 is an extra refusal layer.
  4. DEDUP     — near-duplicate suppression (deterministic key + embedding).

Rules (non-negotiable):
  * USER_ASSERTION can never be upgraded to an evidence class.
  * MODEL_INFERENCE / UNVERIFIED_INFORMATION are persisted — if useful — but
    always labeled, never evidence-backed.
  * EVIDENCE_DERIVED_CLAIM without a verified evidence ref is DEGRADED to
    MODEL_INFERENCE (never silently dropped, never silently upgraded).
  * Stored text that looks like the user is trying to inject instructions is
    refused for persistence as a claim.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.lib.utils import normalize_for_match
from src.memory.enums import (
    CandidateState,
    MemoryObjectType,
    ProvenanceClass,
)
from src.memory.models import (
    CandidateMemory,
    ClaimRecord,
    ValidationResult,
    new_id,
    stable_key,
)

logger = logging.getLogger("src.memory.validation")

# ---------------------------------------------------------------------------
# Classification: what should / should not become memory
# ---------------------------------------------------------------------------

_PREFERENCE_PATTERNS = re.compile(
    r"\b(i|we|please|always|usually|only)\b.{0,24}"
    r"\b(prefer|preference|want|use|style|format|like|avoid|never use|"
    r"do not use|don't use|cite|sources?)\b",
    re.IGNORECASE,
)
_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye|excuse me)"
    r"([\s,!.?]|$)",
    re.IGNORECASE,
)
_QUESTION_PATTERNS = re.compile(r"\?\s*$|^(what|how|why|does|do|is|are|can|"
                                r"should|which|who|when|where)\b",
                                re.IGNORECASE)
_MEDICAL_CLAIM_HINTS = re.compile(
    r"\b(associated|reduces?|increases?|decreases?|risk|mortality|survival|"
    r"efficacy|effect|benefit|harm|supplement|intervention|therapy|treatment|"
    r"dosage|outcomes?|prevalence|incidence)\b",
    re.IGNORECASE,
)

# Prompt-injection / instruction-leak defense-in-depth patterns. Stored text
# is ALWAYS rendered as quoted data by context.py; these patterns additionally
# refuse persistence of blatant control text.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions?|prompts?|context)", re.I),
    re.compile(r"disregard (all |any |the )?(previous|prior|above) (instructions?|prompts?)", re.I),
    re.compile(r"you are now (?!a doctor|an assistant)", re.I),
    re.compile(r"from now on,? (you|respond|answer|act)", re.I),
    re.compile(r"\b(system|assistant|developer|user)\s*:\s*(you|respond|ignore|pretend)", re.I),
    re.compile(r"<\|?(system|im_start|im_end)\|?>", re.I),
    re.compile(r"###\s*(system|instructions?)", re.I),
    re.compile(r"\bDO NOT (mention|tell|reveal|state|say)", re.I),
    re.compile(r"\b(reveal|print|output) your (system prompt|instructions|prompt)\b", re.I),
    re.compile(r"\bfake (evidence|citation|source|study|pmcid|data)\b", re.I),
]
_INJECT_ABUSE_TERMS = re.compile(r"\b(bomb|exploit|jailbreak|prompt injection)\b", re.I)


@dataclass
class Classification:
    """What kind of information is this, and should it persist?"""
    object_type: MemoryObjectType = MemoryObjectType.CLAIM
    provenance_class: ProvenanceClass = ProvenanceClass.MODEL_INFERENCE
    persist: bool = True
    reason: str = ""
    key: str = ""                        # preference key when object_type==USER_PREFERENCE


def classify_text(text: str) -> Classification:
    """Deterministic first-pass classification of free text.

    Conservative by design: when in doubt, persist as labeled
    MODEL_INFERENCE or drop. Nothing here can upgrade a class; the gate in
    ``evidence_gate`` is the only authority for EVIDENCE_DERIVED_CLAIM.
    """
    t = (text or "").strip()
    if not t or len(t) < 4:
        return Classification(persist=False, reason="too short / empty")
    if len(t) > 6000:
        return Classification(persist=False, reason="oversized for a memory unit")
    if _GREETING_PATTERNS.match(t):
        return Classification(persist=False, reason="transient greeting")
    if _INJECTION_PATTERNS and _looks_injected(t):
        return Classification(persist=False, reason="possible instruction-injection text")

    # Explicit durable preference -> preference record.
    if _PREFERENCE_PATTERNS.search(t):
        key = _preference_key(t)
        return Classification(
            object_type=MemoryObjectType.USER_PREFERENCE,
            provenance_class=ProvenanceClass.USER_ASSERTION,
            persist=True,
            reason=f"explicit durable preference ({key})",
            key=key,
        )

    # Questions are research seeds, not claims.
    if _QUESTION_PATTERNS.search(t) or t.endswith("?"):
        return Classification(
            object_type=MemoryObjectType.RESEARCH_QUESTION,
            provenance_class=ProvenanceClass.MODEL_INFERENCE,
            persist=True,
            reason="question (research seed, not a claim)",
        )

    # A user-stated factual assertion about medicine — never evidence.
    if _MEDICAL_CLAIM_HINTS.search(t):
        return Classification(
            provenance_class=ProvenanceClass.USER_ASSERTION,
            persist=True,
            reason="user medical assertion (labeled, never evidence)",
        )

    return Classification(
        provenance_class=ProvenanceClass.USER_ASSERTION,
        persist=True,
        reason="user statement (labeled)",
    )


def _preference_key(text: str) -> str:
    lowered = text.casefold()
    if re.search(r"\b(evidence|meta|rct|randomized|systematic|review|peer|pmc)\b", lowered):
        return "evidence_type"
    if re.search(r"\b(cite|citation|source|references?)\b", lowered):
        return "citation_style"
    if re.search(r"\b(recent|up.to.date|latest|current)\b", lowered):
        return "recency_bias"
    if re.search(r"\b(format|bullets|table|summary|concise|detailed|length)\b", lowered):
        return "format"
    if re.search(r"\b(interested|research|topic|focus|study)\b", lowered):
        return "topic_interest"
    return "format"


def _looks_injected(text: str) -> bool:
    """True when the text is blatant control/instruction text. This is NOT the
    security boundary by itself (rendering-as-data is); it refuses persistence."""
    return any(p.search(text) for p in _INJECTION_PATTERNS) or bool(
        _INJECT_ABUSE_TERMS.search(text))


def injection_flag(text: str) -> tuple[bool, str]:
    """Public helper: (is_injected, matched pattern description)."""
    for p in _INJECTION_PATTERNS:
        m = p.search(text or "")
        if m:
            return True, f"pattern: {m.group(0)[:60]}"
    if _INJECT_ABUSE_TERMS.search(text or ""):
        return True, "abuse-term leak"
    return False, ""


def sanitize_stored_text(text: str) -> str:
    """Strip control characters and collapse whitespace for storage.

    Rendering-as-data (context.py) is the real injection defense; this only
    keeps the store clean.
    """
    if not text:
        return ""
    cleaned = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
    return " ".join(cleaned.split())


# ---------------------------------------------------------------------------
# The provenance gate
# ---------------------------------------------------------------------------

EvidenceRefLookup = Callable[[str], Any | None]


@dataclass
class GateRule:
    """One decision rule of the provenance gate (kept declarative for tests)."""
    applies: Callable[[CandidateMemory], bool]
    check: Callable[[CandidateMemory], tuple[bool, str]]   # (ok, reason)


def evidence_gate(
    candidate: CandidateMemory,
    lookup_ref: EvidenceRefLookup,
    *,
    degrade: ProvenanceClass | None = ProvenanceClass.MODEL_INFERENCE,
) -> ValidationResult:
    """The central provenance gate for a candidate claim.

    * EVIDENCE_DERIVED_CLAIM requires >=1 evidence ref id that resolves to a
      reference with ``verification_status == 'verified'``. Otherwise it is
      DEGRADED (default -> MODEL_INFERENCE) — never dropped, never upgraded.
    * USER_ASSERTION is never upgraded to any evidence class.
    * MODEL_INFERENCE / UNVERIFIED pass as labeled (their class is kept).
    * Non-claim object types (preferences, questions) pass; they carry their
      own semantics and are never evidence-backed.
    """
    result = ValidationResult(candidate=candidate)
    if candidate.state == CandidateState.REJECTED:
        result.ok = False
        result.reasons.append("candidate was already rejected")
        return result

    if candidate.object_type != MemoryObjectType.CLAIM:
        result.ok = True
        return result

    cls = candidate.provenance_class

    if cls == ProvenanceClass.EVIDENCE_DERIVED_CLAIM:
        verified = [rid for rid in candidate.evidence_ref_ids
                    if _ref_is_verified(rid, lookup_ref)]
        if not verified:
            if degrade is not None:
                result.degraded_to = degrade
                result.reasons.append(
                    f"evidence gate: no verified evidence ref for "
                    f"EVIDENCE_DERIVED_CLAIM -> degraded to {degrade.value}")
                # degrade in place so the caller can commit the labeled claim
                candidate.provenance_class = degrade
            else:
                result.ok = False
                result.reasons.append(
                    "evidence gate: EVIDENCE_DERIVED_CLAIM without verified "
                    "evidence ref rejected (degrade=False)")
            return result
        result.reasons.append(
            f"evidence gate: {len(verified)} verified evidence ref(s)")

    elif cls == ProvenanceClass.USER_ASSERTION:
        result.reasons.append("user assertion stays labeled; never evidence")

    elif cls in (ProvenanceClass.MODEL_INFERENCE,
                 ProvenanceClass.UNVERIFIED_INFORMATION):
        result.reasons.append(f"{cls.value} persists labeled (not evidence)")

    result.ok = True
    return result


def _ref_is_verified(ref_id: str, lookup_ref: EvidenceRefLookup) -> bool:
    if not ref_id:
        return False
    ref = lookup_ref(ref_id)
    if ref is None:
        return False
    status = getattr(ref, "verification_status", None) or \
        getattr(ref, "get", lambda k: None)("verification_status")
    return status == "verified"


# ---------------------------------------------------------------------------
# Dedup + relationship heuristics
# ---------------------------------------------------------------------------

def claim_dedup_key(claim: ClaimRecord) -> str:
    """Deterministic dedup key for a claim (independent of embeddings)."""
    return stable_key(claim.normalized_text or claim.text)


def normalized_claim_text(text: str) -> str:
    return normalize_for_match(text or "")


def embedded_similarity(text_a: str, text_b: str, embedder) -> float:
    """Cosine similarity of two texts in the configured embedder space.
    Returns 0.0 when embeddings are unavailable."""
    try:
        [va, vb] = embedder.embed([text_a, text_b])
    except Exception:
        return 0.0
    if not va or not vb:
        return 0.0
    dot = sum(a * b for a, b in zip(va, vb))
    return float(dot)


def is_near_duplicate(claim: ClaimRecord, existing: ClaimRecord,
                      embedder, min_sim: float) -> bool:
    """Near-duplicate test: same normalized text OR same dedup key OR high
    embedding similarity. Used to avoid committing the same claim twice."""
    if claim.dedup_key and claim.dedup_key == existing.dedup_key:
        return True
    if embedded_similarity(claim.text, existing.text, embedder) >= min_sim:
        return True
    return False


__all__ = [
    "Classification",
    "ClaimRecord",
    "classify_text",
    "claim_dedup_key",
    "embedded_similarity",
    "evidence_gate",
    "injection_flag",
    "is_near_duplicate",
    "normalized_claim_text",
    "sanitize_stored_text",
]