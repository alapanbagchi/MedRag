"""Evidence extraction agent with verbatim-quote grounding."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.lib import quote_grounded

logger = logging.getLogger("src.evidence")


# --- Models ---

class Evidence(BaseModel):
    subquery_id: str = ""
    document_id: str = ""
    chunk_id: Optional[str] = None
    source: Optional[str] = None
    claim: str = Field(description="The claim the evidence supports")
    supporting_text: str = Field(description="VERBATIM quote from document")
    supports_claim: bool = Field(description="True if quote supports the claim")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_type: str = "paragraph"
    section: str = ""
    subsection: Optional[str] = ""
    breadcrumb: List[str] = Field(default_factory=list)
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    is_inference: bool = Field(default=False)
    contradiction_note: str = Field(default="")
    evidence_id: Optional[str] = None


class EvidenceSet(BaseModel):
    document_id: str = ""
    subquery_id: str = ""
    items: List[Evidence] = Field(default_factory=list)


class EvidenceGroup(BaseModel):
    group_id: str
    subquery_id: str = ""
    claim: str = ""
    supporting_evidence: List[Evidence] = Field(default_factory=list)
    contradicting_evidence: List[Evidence] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    max_confidence: float = 0.0
    mean_confidence: float = 0.0


# --- Agent ---

class EvidenceExtractor:
    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model
        from pydantic_ai import Agent

        from src.prompts.load import load_prompt

        self.config = config or AppConfig()
        self.model = model or build_model(self.config)
        self._prompt = load_prompt("legacy", "evidence.txt")
        # ONE plain agent reused across calls; structured output per-run.
        self.agent = Agent(self.model, system_prompt=self._prompt,
                           name="evidence_extractor")

    def _prompt_for(self, subquery: Any, document: Any) -> str:
        return (
            f"SUBQUERY {subquery.id}:\n"
            f"target: {subquery.target}\n"
            f"focus: {subquery.focus}\n"
            f"evidence_required: {', '.join(subquery.evidence_required)}\n\n"
            f"DOCUMENT: {document.document_id} (chunk {document.chunk_id})\n"
            f"section: {document.section}\n\n"
            f"DOCUMENT TEXT:\n{document.text}"
        )

    def _finalize(self, evset: EvidenceSet, subquery: Any, document: Any) -> Tuple[List[Evidence], Dict[str, int]]:
        """Attach provenance + enforce quote grounding (with drift tolerance)."""
        items: List[Evidence] = []
        dropped = 0
        methods: Dict[str, int] = {}
        for item in evset.items:
            grounded, method = quote_grounded(item.supporting_text, document.text or "")
            if not grounded:
                dropped += 1
                logger.warning(
                    "dropping ungrounded quote from %s (len=%d): %.80s…",
                    document.document_id, len(item.supporting_text or ""), item.supporting_text or "",
                )
                continue
            methods[method] = methods.get(method, 0) + 1
            items.append(item.model_copy(update={
                "subquery_id": subquery.id,
                "document_id": document.document_id,
                "chunk_id": document.chunk_id,
                "source": document.document_id,
                "section": item.section or getattr(document, "section", ""),
                "subsection": item.subsection or getattr(document, "subsection", "") or "",
                "breadcrumb": item.breadcrumb or list(getattr(document, "breadcrumb", []) or []),
                "table_id": item.table_id or getattr(document, "table_id", None),
                "figure_id": item.figure_id or getattr(document, "figure_id", None),
                "evidence_type": item.evidence_type or getattr(document, "node_type", "paragraph"),
            }))
        stats = {
            "raw_items": len(evset.items),
            "kept_items": len(items),
            "quotes_dropped": dropped,
            "grounding_methods": methods,
        }
        return items, stats

    async def extract_with_stats(self, subquery: Any, document: Any) -> Tuple[List[Evidence], Dict[str, int]]:
        from src.llm.run import ask_structured
        from src.trace import get_trace

        trace = get_trace()
        trace.agent("evidence_extractor", output_type="EvidenceSet",
                    meta={"subquery": subquery.id, "document": getattr(document, "document_id", ""),
                          "chunk": getattr(document, "chunk_id", "")})
        prompt = self._prompt_for(subquery, document)

        try:
            evset = await ask_structured(
                self.agent, prompt, EvidenceSet,
                label="evidence_extractor",
                max_tokens=self.config.agent_max_tokens,
            )
        except Exception:
            raise
        return self._finalize(evset, subquery, document)

    async def extract(self, subquery: Any, document: Any) -> List[Evidence]:
        """Back-compat wrapper (no stats)."""
        items, _stats = await self.extract_with_stats(subquery, document)
        return items


# --- Aggregator (deterministic) ---

class EvidenceAggregator:
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold

    def aggregate(self, evidence: List[Evidence]) -> List[EvidenceGroup]:
        if not evidence:
            return []

        deduped = []
        seen = set()
        for item in evidence:
            key = (item.subquery_id, self._normalize(item.claim))
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        by_subquery: dict[str, list[Evidence]] = {}
        for item in deduped:
            by_subquery.setdefault(item.subquery_id, []).append(item)

        groups = []
        for subquery_id in sorted(by_subquery):
            for cluster in self._cluster(by_subquery[subquery_id]):
                groups.append(self._build_group(subquery_id, cluster, len(groups) + 1))
        return groups

    def _cluster(self, items: List[Evidence]) -> list[list[Evidence]]:
        clusters: list[list[Evidence]] = []
        for item in items:
            placed = False
            for cluster in clusters:
                # Compare against EVERY member's claim, not just the head, so
                # paraphrases still join.
                if any(
                    self._similar(item.claim, member.claim) >= self.similarity_threshold
                    for member in cluster
                ):
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])
        return clusters

    def _build_group(self, subquery_id: str, items: List[Evidence], idx: int) -> EvidenceGroup:
        supporting = [e for e in items if not e.contradiction_note.strip()]
        contradicting = [e for e in items if e.contradiction_note.strip()]
        confidences = [e.confidence for e in supporting]
        doc_ids = list(dict.fromkeys(e.document_id for e in items))
        primary = max(supporting, key=lambda e: e.confidence) if supporting else items[0]
        return EvidenceGroup(
            group_id=f"G{idx}", subquery_id=subquery_id, claim=primary.claim,
            supporting_evidence=supporting, contradicting_evidence=contradicting,
            document_ids=doc_ids,
            max_confidence=round(max(confidences), 4) if confidences else 0.0,
            mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        )

    def _normalize(self, text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())

    def _similar(self, a: str, b: str) -> float:
        ta = set(self._normalize(a).split())
        tb = set(self._normalize(b).split())
        return len(ta & tb) / len(ta | tb) if ta and tb else 0.0
