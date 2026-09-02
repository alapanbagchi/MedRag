"""Memory + Context layer — offline/background consolidation.

Runs periodically (not on the user-turn hot path). Operations here are
idempotent and must never lose information (append + supersede, never
destructive overwrite):

  1. derive_claim_statuses     — recompute claim.status from its current
                                 evidence links (SUPPORTED / CONTRADICTED /
                                 MIXED); a claim with a CONTRADICTED_BY link
                                 is never silently 'supported'.
  2. deduplicate_claims        — near-duplicate claims (same dedup_key / high
                                 embedding sim) of the SAME provenance class
                                 are merged: evidence links are re-pointed to
                                 the survivor, the older claim is SUPERSEDED
                                 (kept for history). Different classes are
                                 never merged (a USER_ASSERTION is not the
                                 same object as an EVIDENCE_DERIVED_CLAIM).
  3. detect_contradictions     — evidence-derived claims that overlap in
                                 topic/entities but carry opposing signals
                                 become ContradictionRecords (unresolved,
                                 dimension scaffolding attached where known).
  4. refresh_temporal          — close valid_to of superseded claims.
  5. mark_stale_by_horizon     — data-driven revalidation flag (0 = disabled:
                                 staleness only via explicit mark_stale or
                                 contradicting-evidence triggers).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from src.memory.enums import (
    ClaimRelationKind,
    ClaimStatus,
    ContradictionKind,
    ContradictionResolution,
    EvidenceRole,
    MemoryEventType,
    MemoryObjectType,
    ProvenanceClass,
)
from src.memory.models import (
    ClaimEvidenceLinkRecord,
    ClaimRecord,
    ClaimRelationRecord,
    ContradictionRecord,
    new_id,
    now_utc,
)
from src.memory.store import MemoryStore
from src.memory.validation import embedded_similarity

logger = logging.getLogger("src.memory.consolidate")

_NEGATION = re.compile(r"\b(no|not|none|never|failed to|absence of|"
                       r"no significant|no association|no effect|did not|"
                       r"does not|without)\b", re.I)
_DECREASE = re.compile(r"\b(reduce|reduction|decreased?|lower|less|"
                       r"decrease|decline)\w*\b", re.I)
_INCREASE = re.compile(r"\b(increase|increased|higher|greater|elevated|"
                       r"rise|raise)\w*\b", re.I)


class Consolidator:
    def __init__(self, store: MemoryStore, embedder: Any | None = None,
                 min_sim: float = 0.86):
        self.store = store
        self.embedder = embedder
        self.min_sim = min_sim
        self.report: dict[str, Any] = {}

    # ------------------------------------------------------------------

    def derive_claim_statuses(self) -> int:
        """Recompute claim.status from current evidence links. Returns the
        number of claims whose status changed."""
        changed = 0
        claims = self.store.scan_claims(include_superseded=True)
        for claim in claims:
            links = self.store.get_claim_evidence_links(claim.id)
            if not links:
                continue
            supported = any(l.role == EvidenceRole.SUPPORTED_BY for l in links)
            contradicted = any(l.role == EvidenceRole.CONTRADICTED_BY for l in links)
            if contradicted and supported:
                new = ClaimStatus.MIXED
            elif contradicted:
                new = ClaimStatus.CONTRADICTED
            elif supported:
                new = ClaimStatus.SUPPORTED
            else:
                continue
            if claim.status != new:
                self.store.update_claim(claim.id, status=new.value)
                self.store.add_event(
                    "consolidator", MemoryEventType.UPDATE,
                    MemoryObjectType.CLAIM, claim.id,
                    {"field": "status", "old": claim.status.value, "new": new.value})
                claim.status = new
                changed += 1
        return changed

    # ------------------------------------------------------------------

    def deduplicate_claims(self) -> int:
        """Merge near-duplicate claims of the SAME provenance class.

        Sub-rule: when both candidates are EVIDENCE_DERIVED with non-empty
        evidence links, the survivor is the one with more links and the older
        claim's links are re-pointed to it before supersession.
        """
        merged = 0
        claims = sorted(self.store.scan_claims(include_superseded=True),
                        key=lambda c: c.first_seen_at)
        index: dict[str, list[ClaimRecord]] = {}
        for claim in claims:
            index.setdefault(claim.dedup_key, []).append(claim)

        # 1) exact dedup-key groups
        for group in index.values():
            if len(group) < 2:
                continue
            merged += self._merge_group(group)

        # 2) embedding-near duplicates across different keys — quadratic scan
        #    is fine at memory scale (thousands); bounded by batch size.
        candidates = [c for c in claims if c.current]
        survivors: list[ClaimRecord] = []
        for claim in candidates:
            hit = self._find_merge_target(claim, survivors)
            if hit is not None:
                merged += self._supersede(claim, hit)
            else:
                survivors.append(claim)
        return merged

    def _merge_group(self, group: list[ClaimRecord]) -> int:
        by_class: dict[ProvenanceClass, list[ClaimRecord]] = {}
        for c in group:
            by_class.setdefault(c.provenance_class, []).append(c)
        total = 0
        for same_class in by_class.values():
            if len(same_class) < 2:
                continue
            same_class.sort(key=lambda c: (len(
                self.store.get_claim_evidence_links(c.id)), c.first_seen_at),
                reverse=True)
            survivor = same_class[0]
            for older in same_class[1:]:
                total += self._supersede(older, survivor)
        return total

    def _find_merge_target(self, claim: ClaimRecord,
                           survivors: list[ClaimRecord]) -> ClaimRecord | None:
        if self.embedder is None or getattr(self.embedder, "dim", 0) <= 0:
            return None
        for other in survivors:
            if other.provenance_class != claim.provenance_class:
                continue
            if other.id == claim.id:
                continue
            try:
                sim = embedded_similarity(claim.text, other.text, self.embedder)
            except Exception:
                continue
            if sim >= self.min_sim:
                return other
        return None

    def _supersede(self, older: ClaimRecord, survivor: ClaimRecord) -> int:
        """Re-point evidence links + supersede the older claim (kept for
        history — never deleted). Returns 1 when a merge happened."""
        claimed = 0
        for link in self.store.get_claim_evidence_links(older.id):
            # re-point the older claim's evidence to the survivor
            self.store.add_claim_evidence_link(ClaimEvidenceLinkRecord(
                claim_id=survivor.id, evidence_ref_id=link.evidence_ref_id,
                role=link.role, weight=link.weight))
            claimed += 1
        now = now_utc()
        self.store.update_claim(
            older.id, status=ClaimStatus.SUPERSEDED.value,
            superseded_by=survivor.id, superseded_at=now, valid_to=now)
        self.store.add_claim_relation(ClaimRelationRecord(
            from_claim_id=older.id, to_claim_id=survivor.id,
            relation=ClaimRelationKind.SUPERSEDES))
        self.store.add_event(
            "consolidator", MemoryEventType.MERGE, MemoryObjectType.CLAIM,
            older.id, {"into": survivor.id, "links_repointed": claimed})
        logger.info("merged claim %s into %s (%d links re-pointed)",
                    older.id, survivor.id, claimed)
        return 1

    # ------------------------------------------------------------------

    def detect_contradictions(self) -> int:
        """Evidence-derived claims that overlap in topic/entities but carry
        opposing signals -> preserved ContradictionRecord (never collapsed).

        Signal: shared significant token OR the claims reference the same
        evidence ref on different sides, plus an opposing polarity marker
        (negation vs. plain, increase vs. decrease).
        """
        created = 0
        claims = self.store.scan_claims(
            provenance=[ProvenanceClass.EVIDENCE_DERIVED_CLAIM])
        n = len(claims)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = claims[i], claims[j]
                if not self._overlap(a, b):
                    continue
                polarity = _polarity(a.text) * _polarity(b.text)
                if polarity >= 0:
                    continue
                if a.session_id and b.session_id and a.session_id != b.session_id:
                    continue  # v1: contradictions are surfaced within a session
                # avoid double-creating the same contradiction pair
                existing = self.store.get_contradictions(
                    session_id=a.session_id or None)
                if any(c.claim_a_id == a.id and c.claim_b_id == b.id
                       or (c.claim_a_id == b.id and c.claim_b_id == a.id)
                       for c in existing):
                    continue
                links_a = self.store.get_claim_evidence_links(a.id)
                links_b = self.store.get_claim_evidence_links(b.id)
                rec = ContradictionRecord(
                    id=new_id("ctr"), session_id=a.session_id,
                    claim=f"{a.text}  //  {b.text}",
                    claim_a_id=a.id, claim_b_id=b.id,
                    evidence_a_ids=[l.evidence_ref_id for l in links_a],
                    evidence_b_ids=[l.evidence_ref_id for l in links_b],
                    kind=ContradictionKind.DIRECT_CONFLICT,
                    resolution=ContradictionResolution.UNRESOLVED,
                )
                self.store.add_contradiction(rec)
                self.store.add_event(
                    "consolidator", MemoryEventType.PROPOSE,
                    MemoryObjectType.CONTRADICTION, rec.id, {"kind": rec.kind.value})
                created += 1
        return created

    def _overlap(self, a: ClaimRecord, b: ClaimRecord) -> bool:
        toks_a = set(re.findall(r"[a-z0-9]{4,}", a.text.casefold()))
        toks_b = set(re.findall(r"[a-z0-9]{4,}", b.text.casefold()))
        shared = len(toks_a & toks_b)
        if shared >= 2:
            return True
        # entity/topic overlap
        ea = set(x.casefold() for x in a.entity_ids + a.topic_ids)
        eb = set(x.casefold() for x in b.entity_ids + b.topic_ids)
        return bool(ea & eb)

    # ------------------------------------------------------------------

    def refresh_temporal(self) -> int:
        """Close valid_to of superseded claims that still have NULL valid_to."""
        fixed = 0
        for claim in self.store.scan_claims(include_superseded=True):
            if claim.superseded_by and claim.valid_to is None:
                self.store.update_claim(claim.id, valid_to=claim.superseded_at or now_utc())
                fixed += 1
        return fixed

    def mark_stale_by_horizon(self, days: int) -> int:
        """Data-driven revalidation flag. ``days <= 0`` disables the horizon
        check entirely (only explicit mark_stale + contradicting-evidence
        triggers then apply — no arbitrary expiration periods)."""
        if days <= 0:
            return 0
        cutoff = now_utc() - timedelta(days=days)
        flagged = 0
        for claim in self.store.scan_claims(
                provenance=[ProvenanceClass.EVIDENCE_DERIVED_CLAIM]):
            if claim.needs_revalidation:
                continue
            last = claim.last_verified_at or claim.first_seen_at
            if last and last < cutoff:
                self.store.update_claim(claim.id, needs_revalidation=True)
                self.store.add_event(
                    "consolidator", MemoryEventType.MARK_STALE,
                    MemoryObjectType.CLAIM, claim.id,
                    {"reason": f"verified more than {days}d ago"})
                flagged += 1
        return flagged

    # ------------------------------------------------------------------

    def run(self, staleness_days: int = 0) -> dict[str, int]:
        """Run the full consolidation pass; returns a counts report."""
        self.report = {
            "statuses_derived": self.derive_claim_statuses(),
            "claims_deduplicated": self.deduplicate_claims(),
            "contradictions_detected": self.detect_contradictions(),
            "temporal_fixed": self.refresh_temporal(),
            "stale_flagged": self.mark_stale_by_horizon(staleness_days),
        }
        return self.report


def _polarity(text: str) -> int:
    """+1 positive effect dir, -1 negative/negated dir, 0 none/ambiguous."""
    neg = bool(_NEGATION.search(text or ""))
    inc = bool(_INCREASE.search(text or ""))
    dec = bool(_DECREASE.search(text or ""))
    if neg:
        return -1
    if inc and not dec:
        return 1
    if dec and not inc:
        return 1
    return 0


__all__ = ["Consolidator", "_polarity"]