"""MedCPT cross-encoder reranking, requirement-aware (spec section 25, V2.1 part 31).

The cross-encoder is expensive, so it runs ONLY after paper selection, local
retrieval and structural filtering - typically on 20-100 strong candidates.
Each candidate is scored AGAINST THE REQUIREMENT IT IS MEANT TO SATISFY:

    score(H4, evidence_context)   instead of   score(giant_query, evidence)

The passage scored is the TABLE-AWARE context when the candidate is a table
member (part 31): summary + headers + row + footnotes + breadcrumb - never
the isolated row text, so "End-to-end | 2 (8) | 2 (50) | 0.04" is judged as
part of Table 2's "Characteristics of patients with and without early re-CoA".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG
from medrag.retrieval_v2.models import EvidenceCandidate, Requirement
from medrag.retrieval_v2.retriever import NullTrace


def build_passage(candidate: EvidenceCandidate) -> str:
    """Passage representation scored by the cross-encoder.

    Prefers the table-aware context_text (part 31) over the raw chunk text;
    section/table metadata is prepended for both.
    """
    parts: List[str] = []
    if candidate.breadcrumb:
        parts.append("Section: " + " > ".join(candidate.breadcrumb))
    if candidate.node_type:
        parts.append(f"Type: {candidate.node_type}")
    if candidate.table_id:
        parts.append(f"Table: {candidate.table_id}")
    if candidate.figure_id:
        parts.append(f"Figure: {candidate.figure_id}")
    if candidate.context_text:
        parts.append("Passage: " + candidate.context_text)
    elif candidate.text:
        parts.append("Passage: " + candidate.text)
    return "\n".join(parts)


class RequirementReranker:
    """Cross-encoder scoring of (requirement, evidence) pairs."""

    def __init__(
        self,
        cross_encoder: Any,
        config: Optional[V2Config] = None,
        trace: Any = None,
    ) -> None:
        self.cross_encoder = cross_encoder
        self.config = config or DEFAULT_CONFIG
        self.trace = trace or NullTrace()

    def rerank(
        self,
        candidates_by_requirement: Dict[str, List[EvidenceCandidate]],
        requirements: Sequence[Requirement],
    ) -> None:
        """Score every (candidate, requirement) pair and normalize per requirement.

        Mutates candidates in place: 'medcpt_scores' holds raw logits per
        requirement, 'medcpt_norm' the min-max normalized values.
        """
        if self.cross_encoder is None:
            for req in requirements:
                pool = candidates_by_requirement.get(req.id, [])
                for c in pool:
                    c.medcpt_scores[req.id] = c.local_scores.get("rrf", 0.0)
            self._normalize(candidates_by_requirement, requirements)
            return

        pairs: List[tuple[str, str, str, str]] = []
        per_req_budget = max(1, self.config.rerank_pair_cap // max(1, len(requirements)))
        for req in requirements:
            pool = candidates_by_requirement.get(req.id, [])
            for c in pool[:per_req_budget]:
                pairs.append((req.id, c.chunk_id, req.rerank_query_text(), build_passage(c)))
        if not pairs:
            return
        if len(pairs) > self.config.rerank_pair_cap:
            pairs = pairs[: self.config.rerank_pair_cap]

        self.trace.log(
            "medcpt_rerank_start",
            params={
                "n_pairs": len(pairs),
                "n_requirements": len(requirements),
                "cap": self.config.rerank_pair_cap,
            },
        )
        queries = [p[2] for p in pairs]
        passages = [p[3] for p in pairs]
        scores = self.cross_encoder._score_pairs(
            list(zip(queries, passages)),
            batch_size=self.config.rerank_batch_size,
        )
        self.trace.log(
            "medcpt_rerank_done",
            params={"n_pairs": len(pairs)},
            result={
                "mean_score": round(float(scores.mean()), 5) if len(scores) else 0,
                "score_range": [
                    round(float(scores.min()), 5), round(float(scores.max()), 5)
                ] if len(scores) else [],
            },
        )
        for (req_id, chunk_id, _q, _p), score in zip(pairs, scores):
            pool = candidates_by_requirement.get(req_id, [])
            for c in pool:
                if c.chunk_id == chunk_id:
                    c.medcpt_scores[req_id] = float(score)
                    break
        self._normalize(candidates_by_requirement, requirements)

    def _normalize(
        self,
        candidates_by_requirement: Dict[str, List[EvidenceCandidate]],
        requirements: Sequence[Requirement],
    ) -> None:
        for req in requirements:
            pool = candidates_by_requirement.get(req.id, [])
            vals = [c.medcpt_scores.get(req.id, 0.0) for c in pool]
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            rng = hi - lo if hi > lo else 1.0
            for c in pool:
                raw = c.medcpt_scores.get(req.id, 0.0)
                c.medcpt_norm[req.id] = (raw - lo) / rng if rng else 0.0
