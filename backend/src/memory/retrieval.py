"""Memory + Context layer — hybrid retrieval over persistent memory.

Not "conversation -> embedding -> top-k". The memory layer retrieves
structured records with multiple coordinated signals:

  Stage 0  HARD FILTERS   SQL/list-level: session scope, provenance classes,
                          current-versions (valid_to IS NULL) or as_of(t),
                          stale exclusion (unless explicitly wanted).
  Stage 1  CANDIDATES     lexical (normalized token overlap + entity
                          overlap), semantic (embedder cosine when enabled),
                          session relevance, graph proximity (1-hop via
                          claim_relations).
  Stage 2  SCORING        weighted fusion + recency + provenance-aware
                          penalties (USER_ASSERTION/MODEL_INFERENCE score
                          lower than evidence-derived), staleness penalty.
  Stage 3  SELECTION      per-type caps + global budget; diversity by
                          session; rerank by fused score; return.

The same code path drives both store backends (in-memory + Postgres):
candidates are SQL/list-filtered, then scored uniformly in Python so tests
are backend-independent. pgvector columns exist in the schema for a future
scale-out dense path; correctness first, scale later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from src.memory.enums import (
    ClaimRelationKind,
    ProvenanceClass,
)
from src.memory.models import ClaimRecord, EvidenceReferenceRecord
from src.memory.store import MemoryStore

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def query_entities(query: str) -> set[str]:
    """Exact-terms from the query — the lexical/entity-overlap signal."""
    return set(_TOKEN_RE.findall((query or "").casefold()))


def token_overlap(text: str, tokens: set[str]) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").casefold())) & tokens


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot)


@dataclass
class ScoredMemory:
    """One retrieved memory candidate with its fused score and signals."""
    claim: ClaimRecord | None = None
    session_score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def score(self) -> float:
        return sum(self.signals.values())


@dataclass
class RetrievalConfig:
    """Weights for the fused scorer. Defaults are deliberately conservative:
    lexical + entity overlap dominate for a medical research memory (precision
    on vocabulary-dense terms), semantics support paraphrase recall."""
    w_lexical: float = 0.45
    w_semantic: float = 0.25
    w_entity: float = 0.20
    w_session: float = 0.10
    w_graph: float = 0.05
    min_score: float = 0.08
    staleness_penalty: float = 0.10       # applied per year since last_verified_at
    provenance_penalty: float = 0.10      # applied to non-evidence-backed classes
    contradiction_boost: float = 0.03     # conflicts matter — surface them
    conclusion_boost: float = 0.05        # established findings matter — surface them


def retrieve_claims(
    store: MemoryStore,
    query: str,
    embedder: Any | None = None,
    *,
    session_ids: Sequence[str] | None = None,
    include_stale: bool = False,
    provenance: Sequence[ProvenanceClass] | None = None,
    top_k: int = 10,
    min_sim: float = 0.30,
    cfg: RetrievalConfig | None = None,
    now: datetime | None = None,
) -> list[ScoredMemory]:
    """Hybrid recall of the most relevant claims for ``query``.

    ``session_ids`` scopes recall to one (or several) research session(s) —
    the cross-session continuity path. ``provenance`` restricts to the given
    classes (e.g. only evidence-backed claims for context assembly).
    """
    cfg = cfg or RetrievalConfig()
    now = now or datetime.now(timezone.utc)
    tokens = query_entities(query)

    # Short follow-up queries ("what about older adults?", "how does
    # hypertension lead to diseases") share very few lexemes with prior
    # findings. When the raw query is thin, expand the matching vocabulary
    # with the session's own questions so continuity recall still fires.
    if len(tokens) < 4 and session_ids:
        for sid in session_ids:
            for q_ in store.get_questions(sid):
                tokens |= query_entities(q_.question)

    candidates = store.scan_claims(
        session_ids=session_ids,
        provenance=provenance,
        include_stale=include_stale,
    )
    if not candidates:
        return []

    q_emb: list[float] | None = None
    if embedder is not None and getattr(embedder, "dim", 0) > 0:
        try:
            q_emb = embedder.embed([query])[0]
        except Exception:
            q_emb = None

    scored: list[ScoredMemory] = []
    for claim in candidates:
        signals: dict[str, float] = {}

        # --- lexical (+ entity) overlap -------------------------------
        text_tokens = token_overlap(claim.text, tokens) | \
            token_overlap(claim.normalized_text, tokens)
        lex = len(text_tokens) / max(1.0, len(tokens))
        signals["lexical"] = lex * cfg.w_lexical

        entity_tokens = set()
        for e in list(claim.entity_ids) + list(claim.topic_ids):
            entity_tokens |= set(_TOKEN_RE.findall((e or "").casefold()))
        ent = len(entity_tokens & tokens) / max(1.0, len(tokens) or 1)
        signals["entity"] = ent * cfg.w_entity

        # --- semantic ------------------------------------------------
        if q_emb:
            try:
                c_emb = embedder.embed([claim.text])[0]
                sim = _cosine(q_emb, c_emb)
                signals["semantic"] = (sim if sim > min_sim else 0.0) * cfg.w_semantic
            except Exception:
                signals["semantic"] = 0.0

        # --- session relevance (continuity) --------------------------------
        if session_ids and claim.session_id in session_ids:
            # within the resumed session every claim gets a continuity
            # baseline so prior findings always surface for follow-ups
            signals["session"] = cfg.w_session

        # --- established conclusions ------------------------------------
        if claim.meta.get("kind") == "conclusion":
            signals["conclusion"] = cfg.conclusion_boost

        # --- graph proximity (1-hop) -----------------------------------
        if signals.get("lexical", 0) > 0 or signals.get("semantic", 0) > 0:
            rels = store.get_claim_relations(claim.id, ClaimRelationKind.RELATED_TO)
            if rels:
                signals["graph"] = cfg.w_graph

        # --- provenance awareness -------------------------------------
        if not claim.provenance_class.evidence_backed:
            signals["provenance_penalty"] = -cfg.provenance_penalty

        # --- staleness --------------------------------------------------
        age_years = 0.0
        last_verified = claim.last_verified_at or claim.first_seen_at
        if last_verified:
            age_years = max(0.0, (now - last_verified).total_seconds()
                            / (365.25 * 86400))
        if age_years > 0.5:
            signals["staleness_penalty"] = -min(cfg.staleness_penalty,
                                                age_years * cfg.staleness_penalty)

        score = sum(signals.values())
        if score <= 0:
            continue
        if score < cfg.min_score:
            continue

        reason_bits = [k for k, v in signals.items() if v > 0]
        scored.append(ScoredMemory(claim=claim, signals=signals,
                                   reason="+".join(reason_bits) or "lexical"))

    scored.sort(key=lambda s: s.score, reverse=True)

    # --- per-session diversity -----------------------------------------
    picked: list[ScoredMemory] = []
    seen_sessions: set[str] = set()
    for sm in scored:
        if len(picked) >= top_k:
            break
        sid = sm.claim.session_id or ""
        if sid and sid in seen_sessions and len(picked) >= 4:
            # avoid flooding the context with one session when others match
            continue
        seen_sessions.add(sid)
        picked.append(sm)
    return picked


def unresolved_contradictions(
    store: MemoryStore,
    session_ids: Sequence[str] | None = None,
) -> list[Any]:
    """Unresolved contradictions — surfaced explicitly, never collapsed."""
    out = []
    for session_id in session_ids or [None]:
        out.extend(store.get_contradictions(
            session_id=session_id, unresolved_only=True))
    return out


def open_gaps(store: MemoryStore,
              session_ids: Sequence[str] | None = None,
              limit: int = 8) -> list[Any]:
    out: list[Any] = []
    for session_id in session_ids or [None]:
        for gap in store.get_gaps(session_id=session_id, open_only=True):
            out.append(gap)
    return out[:limit]


def evidence_refs_for_claims(
    store: MemoryStore,
    claims: Sequence[ClaimRecord],
) -> dict[str, list[EvidenceReferenceRecord]]:
    """Resolve the evidence lineage for a set of claims: claim_id -> refs.

    This is the citation/provenance preservation path — assembled context
    always renders a claim next to its evidence citations, never bare.
    """
    out: dict[str, list[EvidenceReferenceRecord]] = {}
    ref_ids: set[str] = set()
    links_by_claim: dict[str, list] = {}
    for claim in claims:
        links = store.get_claim_evidence_links(claim.id)
        links_by_claim[claim.id] = links
        ref_ids.update(link.evidence_ref_id for link in links)
    refs = {r.id: r for r in store.get_evidence_refs(sorted(ref_ids))}
    for claim in claims:
        links = links_by_claim.get(claim.id, [])
        out[claim.id] = [
            refs[link.evidence_ref_id]
            for link in links
            if link.evidence_ref_id in refs
        ]
    return out


__all__ = [
    "RetrievalConfig",
    "ScoredMemory",
    "open_gaps",
    "query_entities",
    "retrieve_claims",
    "token_overlap",
    "unresolved_contradictions",
]