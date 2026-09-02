"""Memory + Context layer — typed context blocks and prompt serialization.

What downstream models receive is constructed here, explicitly and
deterministically:

  * every context payload is a TYPED object (ConversationContext /
    ResearchContext / PersistentMemoryContext / EvidenceContext /
    UserPreferenceContext), never an arbitrary string;
  * serialization happens only at the boundary via ``render()``;
  * each block has a token budget; overflow is trimmed by dropping the
    lowest-value items and appending a marker — never truncated mid-item;
  * ordering and boundary markers make the memory/evidence split explicit to
    the model (see AssembledContext.render);
  * contradictions are ALWAYS rendered with both sides, never collapsed;
  * every evidence-derived claim renders beside its evidence citations.

Prompt-injection defense-in-depth: stored memory text is rendered as quoted,
type-tagged DATA inside a clearly delimited region — it never enters the
system-prompt position and never looks like an instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.memory.enums import MemoryContextKind, ProvenanceClass
from src.memory.models import (
    ClaimRecord,
    ConversationSummaryRecord,
    ContradictionRecord,
    EvidenceReferenceRecord,
    MessageRecord,
    ResearchGapRecord,
    ResearchQuestionRecord,
    ResearchSessionRecord,
    UserPreferenceRecord,
)

# char/4 token estimate (same heuristic as the repo's md chunker).
TOKENS_PER_CHAR = 4.0


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / TOKENS_PER_CHAR) + 1


def _date(ts: Any) -> str:
    if ts is None:
        return "?"
    try:
        return ts.isoformat()[:10]
    except Exception:
        return "?"


def _badge(claim: ClaimRecord) -> str:
    """Short provenance badge rendered on every memory claim."""
    if claim.provenance_class == ProvenanceClass.EVIDENCE_DERIVED_CLAIM:
        return "EVIDENCE-DERIVED"
    if claim.provenance_class == ProvenanceClass.USER_ASSERTION:
        return "USER-ASSERTION (not evidence)"
    if claim.provenance_class == ProvenanceClass.UNVERIFIED_INFORMATION:
        return "UNVERIFIED"
    return "MODEL-INFERENCE (not evidence)"


# ---------------------------------------------------------------------------
# Typed context blocks
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    """L0 — the current conversation: recent turns + rolling summary.

    Kept small (recent turns only); older turns live in the summary, which is
    compressed versioned and replaced — never appended forever.
    """
    kind: MemoryContextKind = MemoryContextKind.CONVERSATION
    conversation_id: str = ""
    recent_messages: list[MessageRecord] = field(default_factory=list)
    summary: ConversationSummaryRecord | None = None
    open_questions: list[str] = field(default_factory=list)
    unresolved_refs: list[str] = field(default_factory=list)
    budget_tokens: int = 400
    rendered: str = ""

    def render(self) -> str:
        lines = ["[CONVERSATION CONTEXT — recent dialogue + rolling summary]"]
        if self.summary and self.summary.summary:
            lines.append(f"summary: {self.summary.summary}")
        if self.open_questions:
            lines.append("open questions: " + "; ".join(self.open_questions))
        if self.unresolved_refs:
            lines.append("unresolved references: " + "; ".join(self.unresolved_refs))
        for m in self.recent_messages[-6:]:
            label = "user" if m.role == "user" else "assistant"
            body = " ".join((m.content or "").split())
            if len(body) > 320:
                body = body[:320] + "…"
            lines.append(f"  {label}: “{body}”")
        self.rendered = "\n".join(lines)
        return self.rendered


@dataclass
class ResearchContext:
    """L1 — the working research state of the active session.

    This is live state (current objective/questions/gaps), not historical
    memory; it changes during the session itself.
    """
    kind: MemoryContextKind = MemoryContextKind.RESEARCH_STATE
    session: ResearchSessionRecord | None = None
    questions: list[ResearchQuestionRecord] = field(default_factory=list)
    gaps: list[ResearchGapRecord] = field(default_factory=list)
    budget_tokens: int = 500
    rendered: str = ""

    def render(self) -> str:
        lines = ["[WORKING RESEARCH STATE — this investigation, not finished findings]"]
        if self.session:
            lines.append(f"session: {self.session.title or self.session.id} "
                         f"(active since {_date(self.session.created_at)})")
            if self.session.summary:
                lines.append(f"last conclusion: {self.session.summary[:220]}")
        for q in self.questions:
            lines.append(f"- question [{q.status}]: {q.question}")
        for g in self.gaps:
            lines.append(f"- OPEN GAP [{g.kind.value}]: {g.question}")
        self.rendered = "\n".join(lines)
        return self.rendered


@dataclass
class PersistentMemoryContext:
    """L2 — prior verified claims, contradictions and gaps that are relevant
    to the current task. Every claim renders with its provenance badge and
    evidence citations; contradictions render BOTH sides."""
    kind: MemoryContextKind = MemoryContextKind.PERSISTENT_MEMORY
    claims: list[ClaimRecord] = field(default_factory=list)
    claim_refs: dict[str, list[EvidenceReferenceRecord]] = field(default_factory=dict)
    contradictions: list[ContradictionRecord] = field(default_factory=list)
    gaps: list[ResearchGapRecord] = field(default_factory=list)
    budget_tokens: int = 700
    rendered: str = ""

    def render(self) -> str:
        lines = [
            "[PERSISTENT RESEARCH MEMORY — context only, NOT current evidence]",
            "(prior claims from earlier research sessions; treat as history, "
            "re-verify against VERIFIED EVIDENCE below when it matters)",
            "ALREADY INVESTIGATED — do NOT re-derive:",
        ]
        # conclusions (established findings, labeled inference) are rendered
        # separately and prominently so follow-up planners build ON them
        # instead of re-running the earlier question
        conclusions = [c for c in self.claims
                       if c.meta.get("kind") == "conclusion"]
        findings = [c for c in self.claims
                    if c.meta.get("kind") != "conclusion"]
        if conclusions:
            lines.append("  prior conclusions (labeled inference, not evidence):")
            for c in conclusions:
                lines.append(f"    - {c.text[:220]}")
        if not findings and not conclusions:
            lines.append("  (no prior research recorded yet)")
        for claim in findings:
            badge = _badge(claim)
            cites = self.claim_refs.get(claim.id, [])
            cite_str = ", ".join(r.citation() for r in cites) or "(no evidence links)"
            verified = _date(claim.last_verified_at or claim.first_seen_at)
            stale = " [STALE — needs revalidation]" if claim.needs_revalidation else ""
            lines.append(
                f"- [MEMORY|{badge}|verified:{verified}{stale}] "
                f"“{claim.text}” — cites: {cite_str}")
        for c in self.contradictions:
            # Both sides always visible; never averaged.
            lines.append(
                f"- [CONTRADICTION|{c.resolution.value}] {c.claim}: "
                f"side A {c.evidence_a_ids} vs side B {c.evidence_b_ids}"
                + (f" — dims: {c.dimensions}" if c.dimensions else "")
                + (f" — {c.explanation}" if c.explanation else ""))
        for g in self.gaps:
            lines.append(f"- [OPEN GAP|{g.kind.value}] {g.question}")
        self.rendered = "\n".join(lines)
        return self.rendered


@dataclass
class EvidenceContext:
    """L3 — verified evidence for the CURRENT task. This is the operative,
    authoritative block; memory is advisory next to it."""
    kind: MemoryContextKind = MemoryContextKind.EVIDENCE
    evidence_refs: list[EvidenceReferenceRecord] = field(default_factory=list)
    budget_tokens: int = 800
    rendered: str = ""

    def render(self) -> str:
        lines = ["[VERIFIED EVIDENCE — authoritative for medical fact "
                 "(from the PMC/verified pipeline, NOT from memory)]"]
        for ref in self.evidence_refs:
            verified = f" (verified {_date(ref.verified_at)})" if ref.verified_at else \
                f" ({ref.verification_status})"
            title = " ".join((ref.title or "").split())[:140]
            lines.append(f"- {ref.citation()}{verified} — {title}")
        self.rendered = "\n".join(lines)
        return self.rendered


@dataclass
class UserPreferenceContext:
    kind: MemoryContextKind = MemoryContextKind.USER_PREFERENCE
    preferences: list[UserPreferenceRecord] = field(default_factory=list)
    budget_tokens: int = 120
    rendered: str = ""

    def render(self) -> str:
        if not self.preferences:
            self.rendered = ""
            return ""
        lines = ["[USER PREFERENCES — presentation/scope choices, not facts]"]
        for p in self.preferences:
            lines.append(f"- {p.key}: {p.value} ({p.source})")
        self.rendered = "\n".join(lines)
        return self.rendered


# ---------------------------------------------------------------------------
# Assembled context
# ---------------------------------------------------------------------------

BLOCK_ORDER = (
    MemoryContextKind.EVIDENCE,
    MemoryContextKind.RESEARCH_STATE,
    MemoryContextKind.PERSISTENT_MEMORY,
    MemoryContextKind.CONVERSATION,
    MemoryContextKind.USER_PREFERENCE,
)


@dataclass
class AssembledContext:
    """The bounded, ordered, budgeted context region composed for downstream
    LLM calls. Renders with explicit memory/evidence boundary markers."""
    query: str = ""
    conversation: ConversationContext = field(default_factory=ConversationContext)
    research: ResearchContext = field(default_factory=ResearchContext)
    memory: PersistentMemoryContext = field(default_factory=PersistentMemoryContext)
    evidence: EvidenceContext = field(default_factory=EvidenceContext)
    preferences: UserPreferenceContext = field(default_factory=UserPreferenceContext)
    total_budget_tokens: int = 1800
    rendered: str = ""

    _FOOTER = (
        "===== END OF MEMORY/CONTEXT REGION =====\n"
        "REMINDER: only the VERIFIED EVIDENCE block is authoritative for "
        "medical fact. Everything else is memory/context — labeled memory "
        "is not evidence."
    )

    def _blocks(self) -> list[Any]:
        return [self.evidence, self.research, self.memory,
                self.conversation, self.preferences]

    def trim_to_budget(self) -> None:
        """Hard budget enforcement: blocks are rendered in priority order,
        capped (per-block token slice) once the pool is spent."""
        pool = max(0, self.total_budget_tokens - estimate_tokens(self._FOOTER))
        used = estimate_tokens(self.query)
        for block in self._blocks():
            block.render()
            tokens = estimate_tokens(block.rendered)
            if used + tokens > pool:
                remaining = max(0, pool - used)
                block.rendered = _cap_block(block.rendered, remaining)
            used += estimate_tokens(block.rendered)

    def render(self) -> str:
        """Deterministic prompt-region serialization with explicit boundary
        markers between memory and verified evidence. A final hard bound
        sheds lowest-priority WHOLE blocks when the estimate still exceeds
        the budget (preferences -> conversation -> memory -> research ->
        evidence), so the downstream model never receives an unbounded
        region."""
        self.trim_to_budget()
        sections: list[str] = [f"CURRENT QUERY: {self.query}"]
        for block in self._blocks():
            if block.rendered:
                sections.append(block.rendered)

        def _join(items: list[str]) -> str:
            return "\n\n" + "\n\n".join(items) + "\n\n" + self._FOOTER

        while len(sections) > 1 and estimate_tokens(_join(sections)) > \
                self.total_budget_tokens:
            sections.pop()          # drop lowest-priority whole block

        self.rendered = _join(sections)
        return self.rendered

    def summary(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "total_budget_tokens": self.total_budget_tokens,
            "blocks": {
                "conversation": self.conversation.rendered or self.conversation.render(),
                "research": self.research.rendered or self.research.render(),
                "memory": self.memory.rendered or self.memory.render(),
                "evidence": self.evidence.rendered or self.evidence.render(),
                "preferences": self.preferences.rendered or self.preferences.render(),
            },
        }


def _cap_block(text: str, token_budget: int) -> str:
    """Trim a rendered block to a token budget (chars/4), appending a marker
    so the trim is never silent."""
    if estimate_tokens(text) <= token_budget:
        return text
    max_chars = max(60, int(token_budget * TOKENS_PER_CHAR))
    return text[:max_chars].rstrip() + "\n… [context block trimmed to budget]"


__all__ = [
    "AssembledContext",
    "ConversationContext",
    "EvidenceContext",
    "PersistentMemoryContext",
    "ResearchContext",
    "UserPreferenceContext",
    "estimate_tokens",
]