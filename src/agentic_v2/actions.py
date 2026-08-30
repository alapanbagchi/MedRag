"""Agentic v2 — action executors (the "what should happen next" is run here).

Each orchestrator action has a deterministic executor. The orchestrator only
DECIDES; this module EXECUTES and folds the result back into the
``ResearchState`` (the research notebook). Executors are plain async methods so
they can be unit-tested with fake dependencies.

Reuses the existing building blocks without modifying them:
  * DECOMPOSE            -> src.agentic.planner.DecomposePlanner
  * ENRICH               -> src.agentic.umls_tool.UMLSEnricher (UMLS/MeSH)
  * GLOBAL_RETRIEVE      -> src.agentic.retriever_tool.HybridRetrieverTool
  * READ_DOCUMENT / FIND_SECTIONS -> corpus + StructuralUnitIndex
  * VERIFY               -> src.agentic_v2.verify.ObjectiveVerifier (intent only)
  * SYNTHESIZE           -> src.agentic_v2.synthesize.FinalSynthesizer

Retrieval uses the objective's UMLS-enriched query (when present) and sends each
hit's FULL structural unit (whole paragraph / table / figure) straight to the
verifier — there is no separate local-search step.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agentic_v2.orchestrator import ActionDecision, ActionType
from src.agentic_v2.state import (
    CandidatePassage,
    EvidenceQuality,
    ObjectiveStatus,
    ResearchObjective,
    ResearchState,
    RetrievedDocument,
    VerifiedEvidence,
)
from src.agentic_v2.verify import ObjectiveVerifier
from src.agentic_v2.synthesize import FinalSynthesizer

logger = logging.getLogger("src.agentic_v2.actions")

_MAX_VERIFY_PASSAGES = 6

# Intent relevance (the verifier's ONLY job) -> evidence-quality label + support.
_RELEVANCE_QUALITY = {
    "relevant": (EvidenceQuality.DIRECT, "supports"),
    "partially_relevant": (EvidenceQuality.INDIRECT, "supports"),
    "not_relevant": (EvidenceQuality.BACKGROUND, "neutral"),
}


def _map_relevance(relevance: str, quality: EvidenceQuality, support: str):
    """Derive (quality, support) from intent relevance when present."""
    mapped = _RELEVANCE_QUALITY.get((relevance or "").strip())
    if mapped is not None:
        return mapped
    return quality, support


# ---------------------------------------------------------------------------
# Evidence excerpt capture
# ---------------------------------------------------------------------------
# A retrieved unit can be a WHOLE TABLE whose relevant row (e.g. a literal
# definition of cardiac arrest) sits hundreds of characters past the head.
# Storing the raw head of the unit as the evidence excerpt silently truncates
# the supporting text away before the orchestrator / synthesizer ever see it
# (demonstrated failure: "No definition of cardiac arrest was found in the
# provided verified evidence" despite 2 relevant units). Anchor the excerpt on
# the objective's own terms instead.

_EVIDENCE_EXCERPT_CHARS = 1400   # window around the anchored phrase
_EVIDENCE_EXCERPT_LEAD = 180     # context chars before the phrase


def _excerpt_phrases(objective: Any) -> List[str]:
    """Search phrases for anchoring, longest first (most specific wins)."""
    raw: List[str] = []
    raw += list(getattr(objective, "evidence_required", []) or [])
    raw += [getattr(objective, "statement", "") or "",
            getattr(objective, "intent", "") or ""]
    raw += list(getattr(objective, "synonyms", []) or [])
    raw += list(getattr(objective, "entities", []) or [])
    out: List[str] = []
    for chunk in raw:
        s = " ".join(str(chunk).split())
        if s and len(s) >= 4 and s not in out:
            out.append(s)
    out.sort(key=len, reverse=True)
    return out


def _best_matching_phrase(objective: Any, text: str) -> Optional[str]:
    """The objective's longest phrase that actually appears in the unit text."""
    low = text.lower()
    for phrase in _excerpt_phrases(objective):
        if phrase.lower() in low:
            return phrase
    return None


def _clip_excerpt(text: str, start: int, max_chars: int) -> str:
    start = max(0, start)
    end = min(len(text), start + max_chars)
    head = "..." if start > 0 else ""
    tail = "..." if end < len(text) else ""
    return head + text[start:end] + tail


def _evidence_excerpt(text: str, objective: Any,
                      max_chars: int = _EVIDENCE_EXCERPT_CHARS) -> str:
    """A bounded evidence excerpt anchored on the objective's best term.

    Falls back to the unit head when no objective phrase is present; the head
    is clipped with an explicit marker, never silently cut.
    """
    text = (text or "").strip()
    if not text:
        return ""
    phrase = _best_matching_phrase(objective, text)
    if phrase is None:
        return _clip_excerpt(text, 0, max_chars)
    idx = text.lower().find(phrase.lower())
    start = max(0, idx - _EVIDENCE_EXCERPT_LEAD)
    # snap the start onto a row/line boundary so tables keep whole rows
    nl = text.rfind("\n", 0, start)
    if nl >= 0 and start - nl <= 120:
        start = nl + 1
    return _clip_excerpt(text, start, max_chars)


_UNSET = object()   # sentinel: "corpus not injected -> load the shared service lazily"


# ---------------------------------------------------------------------------
# Action result
# ---------------------------------------------------------------------------

class ActionResult(BaseModel):
    action: ActionType
    summary: str = ""
    status: str = "done"                 # done | failed | repeated
    data: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ActionExecutor:
    """Runs the action chosen by the orchestrator, mutating the state."""

    def __init__(
        self,
        config: Any = None,
        *,
        planner: Any = None,
        retriever: Any = None,
        verifier: Any = None,
        synthesizer: Any = None,
        umls_enricher: Any = None,
        corpus: Any = _UNSET,
        events: Any = None,
    ):
        from src.config import AppConfig

        self.config = config or AppConfig()
        self._planner = planner
        self._retriever = retriever
        self._verifier = verifier
        self._synthesizer = synthesizer
        self._umls_enricher = umls_enricher
        # ``corpus`` injection: pass an object (or None) to skip the lazy load
        # of the shared retrieval service (used by tests / no-corpus contexts).
        if corpus is _UNSET:
            self._corpus: Any = None
            self._corpus_loaded: bool = False
        else:
            self._corpus = corpus
            self._corpus_loaded = True
        self._unit_index: Any = None
        self.events = events   # optional EventEmitter for the UI

    def _emit(self, type_: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(type_, **fields)

    # -- lazy deps ------------------------------------------------------

    def _planner_obj(self):
        if self._planner is None:
            from src.agentic.planner import DecomposePlanner
            self._planner = DecomposePlanner(config=self.config)
        return self._planner

    def _retriever_obj(self):
        if self._retriever is None:
            from src.agentic.retriever_tool import HybridRetrieverTool
            self._retriever = HybridRetrieverTool(config=self.config)
        return self._retriever

    def _verifier_obj(self):
        if self._verifier is None:
            self._verifier = ObjectiveVerifier(config=self.config)
        return self._verifier

    def _synthesizer_obj(self):
        if self._synthesizer is None:
            self._synthesizer = FinalSynthesizer(config=self.config)
        return self._synthesizer

    def _umls_enricher_obj(self):
        if self._umls_enricher is None:
            from src.agentic.umls_tool import UMLSEnricher
            self._umls_enricher = UMLSEnricher(config=self.config)
        return self._umls_enricher

    def _corpus_obj(self) -> Any:
        """Lazily load the corpus (None when injected None / unavailable)."""
        if self._corpus_loaded:
            return self._corpus
        try:
            from src.retrieval.retriever import get_retrieval_service
            service = get_retrieval_service(self.config)
            self._corpus = service._components()["corpus"]
        except Exception as exc:
            logger.warning("corpus unavailable: %s", exc)
            self._corpus = None
        self._corpus_loaded = True
        return self._corpus

    def _unit_index_obj(self, corpus: Any) -> Any:
        if self._unit_index is not None:
            return self._unit_index
        try:
            from src.retrieval.fullpaper import get_unit_index
            self._unit_index = get_unit_index(corpus)
        except Exception as exc:
            logger.warning("unit index unavailable: %s", exc)
            self._unit_index = None
        return self._unit_index

    # -- dispatch -------------------------------------------------------

    async def execute(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        handlers = {
            ActionType.DECOMPOSE: self._decompose,
            ActionType.GLOBAL_RETRIEVE: self._global_retrieve,
            ActionType.ENRICH: self._enrich,
            ActionType.READ_DOCUMENT: self._read_document,
            ActionType.FIND_SECTIONS: self._find_sections,
            ActionType.VERIFY: self._verify,
            ActionType.SYNTHESIZE: self._synthesize,
            ActionType.STOP: self._stop,
        }
        handler = handlers.get(decision.action)
        if handler is None:
            return ActionResult(action=decision.action, status="failed",
                                summary=f"unknown action {decision.action}")
        return await handler(decision, state)

    # -- DECOMPOSE ------------------------------------------------------

    async def _decompose(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        planner = self._planner_obj()

        # Recursive decomposition of one objective, or top-level decomposition.
        parent_id = (decision.objective_id or "").strip()
        parent = state.objective(parent_id) if parent_id else None
        if parent is not None:
            target_text = parent.statement or state.question
        elif not state.objectives:
            target_text = state.question
        else:
            # Re-decompose WITHOUT a target -> refine the first objective that
            # still needs work, rather than resetting already-completed ones.
            parent = state.non_terminal_objectives()[0] if state.non_terminal_objectives() else None
            target_text = (parent.statement if parent else state.question)

        self._emit("agent_spawn", iteration=state.iteration, action="DECOMPOSE",
                   agent="planner", input={"target": target_text, "parent": parent_id})
        try:
            decomposition = await planner.plan(target_text)
        except Exception as exc:
            self._emit("failure", iteration=state.iteration, action="DECOMPOSE",
                       error=str(exc)[:160])
            return ActionResult(action=ActionType.DECOMPOSE, status="failed",
                                summary=f"decomposition failed: {str(exc)[:160]}")

        new_objs: List[ResearchObjective] = []
        for i, sub in enumerate(decomposition.subqueries, start=1):
            obj_id = sub.id
            if parent is not None:
                obj_id = f"{parent.id}.{i}"
            new_objs.append(ResearchObjective(
                id=obj_id,
                statement=sub.target or sub.query or target_text,
                intent=sub.intent,
                evidence_required=list(sub.evidence_required or []),
                status=ObjectiveStatus.OPEN,
                entities=[e.text for e in sub.entities if e and e.text],
            ))

        if not new_objs:
            # deterministic single-hop fallback (mirrors the planner's own guard)
            obj_id = f"{parent.id}.1" if parent is not None else "O1"
            new_objs = [ResearchObjective(
                id=obj_id, statement=target_text[:200],
                intent="answer the objective as posed", status=ObjectiveStatus.OPEN,
            )]

        for obj in new_objs:
            state.upsert_objective(obj)
        labels = ", ".join(f"{o.id}" for o in new_objs)
        scope = f"objective {parent.id}" if parent is not None else "question"
        self._emit("agent_output", iteration=state.iteration, action="DECOMPOSE",
                   agent="planner", status="done",
                   output={"objectives": [o.id for o in new_objs],
                           "statements": {o.id: o.statement for o in new_objs}})
        return ActionResult(
            action=ActionType.DECOMPOSE,
            summary=f"decomposed {scope} into {len(new_objs)} objective(s): {labels}",
            data={"objectives": [o.id for o in new_objs]},
        )

    # -- ENRICH (UMLS/MeSH query expansion) --------------------------------

    async def _enrich(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        from src.agentic.planner import PlannedEntity, SubQueryPlan

        obj = state.objective(decision.objective_id) if decision.objective_id else None
        if obj is None:
            for o in state.objectives:
                if o.status not in (ObjectiveStatus.SUPPORTED, ObjectiveStatus.CONTRADICTED):
                    obj = o
                    break
        if obj is None:
            return ActionResult(action=ActionType.ENRICH, status="done",
                                summary="no objective to enrich (run DECOMPOSE first)")

        base = " ".join((decision.query or "").split()) or obj.statement or state.question
        entities = list(obj.entities)
        if not entities:
            from src.agentic.planner import _fallback_entities
            entities = [e.text for e in _fallback_entities(base)]

        sub = SubQueryPlan(
            id=obj.id,
            target=obj.statement or base,
            query=base,
            focus="evidence",
            evidence_required=list(obj.evidence_required or []),
            entities=[PlannedEntity(text=t) for t in entities],
        )
        enricher = self._umls_enricher_obj()
        self._emit("agent_spawn", iteration=state.iteration, action="ENRICH",
                   agent="umls", input={"query": base, "entities": entities})
        try:
            terms = await enricher.enrich_subquery(sub)
            enriched = enricher.apply_to_query(sub, terms) or base
        except Exception as exc:
            self._emit("failure", iteration=state.iteration, action="ENRICH",
                       error=str(exc)[:160])
            return ActionResult(action=ActionType.ENRICH, status="failed",
                                summary=f"enrichment failed: {str(exc)[:160]}")

        syns: List[str] = []
        for t in terms:
            if t.preferred_name:
                syns.append(t.preferred_name)
            syns.extend(t.synonyms or [])
        # dedupe, drop exact echoes of the base query terms
        seen = set(base.casefold().split())
        clean: List[str] = []
        for s in syns:
            s = " ".join((s or "").split())
            if not s or s.casefold() in seen:
                continue
            seen.add(s.casefold())
            clean.append(s)

        obj.synonyms = clean
        if clean:
            obj.enriched_query = enriched   # only counts as enrichment when terms found
        self._emit("agent_output", iteration=state.iteration, action="ENRICH",
                   agent="umls", status="done",
                   output={"enriched_query": obj.enriched_query, "synonyms": clean})
        return ActionResult(
            action=ActionType.ENRICH,
            summary=f"enriched {obj.id}: +{len(clean)} UMLS/MeSH term(s) -> {enriched[:140]}",
            data={"objective": obj.id, "synonyms": clean, "enriched_query": enriched},
        )

    # -- GLOBAL_RETRIEVE -------------------------------------------------

    async def _global_retrieve(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        from src.agentic.planner import SubQueryPlan

        query = " ".join((decision.query or "").split())
        obj = state.objective(decision.objective_id) if decision.objective_id else None
        if not query:
            query = (obj.statement if obj else state.question) or ""
        if not query:
            return ActionResult(action=ActionType.GLOBAL_RETRIEVE, status="done",
                                summary="no search query available")

        # Prefer the UMLS-enriched query (with synonyms) when ENRICH already ran
        # for this objective; otherwise fall back to the plain query.
        search_query = query
        if obj and obj.enriched_query and obj.enriched_query != query:
            search_query = obj.enriched_query

        # Loop avoidance: never re-run an already-searched query unchanged.
        if search_query.casefold() in {q.casefold() for q in state.searched_queries}:
            return ActionResult(action=ActionType.GLOBAL_RETRIEVE, status="repeated",
                                summary=f"query already searched; no new documents: {search_query!r}")

        sub = SubQueryPlan(
            id="GR",
            target=search_query,
            intent=obj.intent if obj else "",
            query=search_query,
            focus="evidence",
            evidence_required=list(obj.evidence_required if obj else []),
        )
        exclude = sorted(state.seen_chunk_ids())
        self._emit("agent_spawn", iteration=state.iteration, action="GLOBAL_RETRIEVE",
                   agent="retriever", input={"query": search_query, "exclude_seen": len(exclude),
                                             "objective": decision.objective_id,
                                             "enriched": search_query != query})
        try:
            results = await self._retriever_obj().search(
                sub, top_k=self.config.max_documents, exclude_chunk_ids=exclude
            )
        except Exception as exc:
            self._emit("failure", iteration=state.iteration, action="GLOBAL_RETRIEVE",
                       error=str(exc)[:160])
            return ActionResult(action=ActionType.GLOBAL_RETRIEVE, status="failed",
                                summary=f"global retrieval failed: {str(exc)[:160]}")

        # Each hit chunk is ALREADY restored by the retriever to its full
        # containing structural unit (whole paragraph / table / figure), so it
        # can go straight to the verifier — no local search step is needed.
        docs: List[RetrievedDocument] = []
        for r in results:
            docs.append(RetrievedDocument(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                section=r.section,
                unit_kind=r.unit_kind,
                score=r.rrf_score,
                text=r.paragraph_text,
                objective_id=decision.objective_id,
                source_action="GLOBAL_RETRIEVE",
                source_query=search_query,
            ))
        added = state.add_documents(docs)
        state.searched_queries.append(search_query)
        doc_ids = ", ".join((d.document_id or d.chunk_id) for d in docs[:6]) or "(none)"
        # One structured event per retrieved unit (paragraph/table/figure) so the
        # UI can render the fetched text line-by-line.
        for d in docs:
            self._emit("document", iteration=state.iteration, action="GLOBAL_RETRIEVE",
                       objective_id=d.objective_id,
                       document_id=d.document_id, chunk_id=d.chunk_id, section=d.section,
                       unit_kind=d.unit_kind, text=d.text, source="global_retrieve",
                       score=d.score)
        self._emit("agent_output", iteration=state.iteration, action="GLOBAL_RETRIEVE",
                   agent="retriever", status="done",
                   output={"retrieved": len(results), "new": added,
                           "query": search_query,
                           "docs": [(d.document_id or d.chunk_id, d.section, d.unit_kind)
                                    for d in docs[:6]]})
        return ActionResult(
            action=ActionType.GLOBAL_RETRIEVE,
            summary=f"retrieved {len(results)} unit(s) ({added} new): {doc_ids}",
            data={"retrieved": len(results), "new": added, "query": search_query},
        )

    # -- READ_DOCUMENT ----------------------------------------------------

    async def _read_document(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        doc_id = (decision.document_id or "").strip()
        chunk_id = (decision.chunk_id or "").strip()
        corpus = self._corpus_obj()
        if not doc_id and not chunk_id:
            return ActionResult(action=ActionType.READ_DOCUMENT, status="done",
                                summary="need document_id (or chunk_id) to read")
        if corpus is None:
            return ActionResult(action=ActionType.READ_DOCUMENT, status="failed",
                                summary="corpus unavailable; cannot read document")

        self._emit("step", iteration=state.iteration, action="READ_DOCUMENT",
                   input={"document_id": doc_id, "chunk_id": chunk_id},
                   output={"reading": True}, status="running")
        try:
            if chunk_id:
                text, kind, section = await self._read_chunk(corpus, chunk_id)
                if not text:
                    return ActionResult(action=ActionType.READ_DOCUMENT, status="done",
                                        summary=f"chunk {chunk_id} has no text")
                doc = RetrievedDocument(
                    chunk_id=chunk_id,
                    document_id=doc_id or (corpus.document_id(chunk_id) or ""),
                    section=section, unit_kind=kind, text=text,
                    objective_id=decision.objective_id,
                    source_action="READ_DOCUMENT",
                )
            else:
                # NEVER read a whole paper into one unit: READ_DOCUMENT by
                # document_id restores the paper's structural units (paragraph/
                # table/figure) and keeps only the ones most relevant to the
                # objective's evidence needs.
                units = await self._document_units(corpus, doc_id, state, decision)
                if not units:
                    return ActionResult(
                        action=ActionType.READ_DOCUMENT, status="done",
                        summary=f"no readable units for document {doc_id} "
                                "(try GLOBAL_RETRIEVE or pick a chunk_id)",
                    )
                docs = [
                    RetrievedDocument(
                        chunk_id=cu["chunk_id"],
                        document_id=doc_id,
                        section=cu["section"], unit_kind=cu["kind"], text=cu["text"],
                        score=cu["score"],
                        objective_id=decision.objective_id,
                        source_action="READ_DOCUMENT",
                    )
                    for cu in units
                ]
                added = state.add_documents(docs)
                for i, d in enumerate(docs, start=1):
                    self._emit("document", iteration=state.iteration,
                               action="READ_DOCUMENT", objective_id=decision.objective_id,
                               document_id=d.document_id, chunk_id=d.chunk_id,
                               section=d.section, text=d.text, source="read_document")
                self._emit("step", iteration=state.iteration, action="READ_DOCUMENT",
                           input={"document_id": doc_id, "chunk_id": chunk_id},
                           output={"documents": len(docs),
                                   "unit_ids": [d.chunk_id for d in docs]},
                           status="done")
                return ActionResult(
                    action=ActionType.READ_DOCUMENT,
                    summary=(f"read {len(docs)} relevant unit(s) of {doc_id} "
                             f"({sum(len(d.text.split()) for d in docs)} tokens; {added} new)"),
                    data={"document_id": doc_id, "units": len(docs), "new": added},
                )
        except Exception as exc:
            return ActionResult(action=ActionType.READ_DOCUMENT, status="failed",
                                summary=f"read failed: {str(exc)[:160]}")

        added = state.add_documents([doc])
        self._emit("document", iteration=state.iteration, action="READ_DOCUMENT",
                   objective_id=decision.objective_id,
                   document_id=doc.document_id, chunk_id=doc.chunk_id, section=doc.section,
                   text=doc.text, source="read_document")
        self._emit("step", iteration=state.iteration, action="READ_DOCUMENT",
                   input={"document_id": doc_id, "chunk_id": chunk_id},
                   output={"document_id": doc.document_id, "chunk_id": doc.chunk_id,
                           "section": doc.section, "tokens": len(doc.text.split())},
                   status="done")
        return ActionResult(
            action=ActionType.READ_DOCUMENT,
            summary=f"read {doc.document_id or doc.chunk_id} ({len(doc.text.split())} tokens; {added} new)",
            data={"document_id": doc.document_id, "chunk_id": doc.chunk_id, "new": added},
        )

    async def _document_units(self, corpus: Any, doc_id: str,
                              state: ResearchState, decision: ActionDecision,
                              top_n: int = 4) -> List[dict]:
        """Restore a document structural units and rank by objective relevance.

        Returns up to top_n mapped units shaped as dict(chunk_id, section, kind,
        text, score). The full document is NEVER returned as one unit - each
        item is a containing structural unit (paragraph / table / figure) so
        the verifier only ever sees the relevant piece of the paper.
        """
        import asyncio

        df = getattr(corpus, "_df", None)
        if df is None or "id" not in df.columns or "document_id" not in df.columns:
            return []
        sub = df[df["document_id"] == doc_id]
        if sub.empty:
            return []
        idx = self._unit_index_obj(corpus)
        if idx is None:
            return []

        # Objective evidence terms used to rank units (fall back to decision).
        obj = state.objective(decision.objective_id) if decision.objective_id else None
        terms = self._unit_query_terms(obj, decision)
        max_units = max(1, min(top_n, len(sub)))

        units: List[dict] = []
        seen: set = set()
        for _, row in sub.iterrows():
            cid = str(row.get("id") or "")
            if not cid or cid in seen:
                continue
            text = ""
            kind = "paragraph"
            try:
                text = await asyncio.to_thread(idx.get, cid)
                kind = await asyncio.to_thread(idx.unit_kind, cid)
            except Exception:
                pass
            if not text or text in seen:
                # fall back to the chunk text itself on index miss
                text = str(row.get("text") or "").strip()
                if not text or text in seen:
                    continue
            seen.add(text)
            sec = str(row.get("section") or "")
            units.append({
                "chunk_id": cid,
                "section": sec,
                "kind": kind,
                "text": text,
                "score": self._score_unit(text, terms) if terms else 0.0,
            })

        # Best-relevant first; stable for ties (insertion = document order).
        units = sorted(
            enumerate(units),
            key=lambda pair: (-pair[1]["score"], pair[0]),
        )
        return [pair[1] for pair in units[:max_units]]

    @staticmethod
    def _split_paragraph_units(text: str, terms: List[str], doc_id: str,
                               top_n: int = 4) -> List[dict]:
        """Best-effort fallback: split a whole-paper blob into paragraph-like
        units and keep the ones most relevant to the objective terms.

        Used only when the structural-unit index cannot be reached for a
        full-document unit, so the ACTUAL RELEVANT PART is still verified
        instead of silently dropping the document entirely.
        """
        import re

        parts = re.split(r"\n{2,}|(?<=\.)\n", (text or "").strip())
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            return []
        # A single "paragraph" may still be huge (e.g. minified full text);
        # break any unit above ~600 words at sentence boundaries so each
        # candidate stays bounded and relevant.
        units = []
        for i, p in enumerate(parts):
            if len(p.split()) >= 5 and len(p.split()) <= 600:
                units.append((f"{doc_id}:para{i}", p))
            elif len(p.split()) > 600:
                sentences = re.split(r"(?<=[\.!?])\s+", p)
                bucket: List[str] = []
                for sent in sentences:
                    bucket.append(sent)
                    if sum(len(s.split()) for s in bucket) >= 400:
                        merged = " ".join(bucket)
                        if len(merged.split()) <= 600:
                            units.append((f"{doc_id}:para{i}", merged))
                        bucket = []
                if bucket:
                    merged = " ".join(bucket)
                    if len(merged.split()) >= 5:
                        units.append((f"{doc_id}:para{i}", merged))
        out: List[dict] = []
        for cid, p in units:
            out.append({
                "chunk_id": cid,
                "section": "",
                "kind": "paragraph",
                "text": p,
                "score": ActionExecutor._score_unit(p, terms) if terms else 0.0,
            })
        units = sorted(
            enumerate(out),
            key=lambda pair: (-pair[1]["score"], pair[0]),
        )
        return [pair[1] for pair in units[:top_n]]

    @staticmethod
    def _score_unit(text: str, terms: List[str]) -> float:
        """Coarse lexical relevance: how many objective terms appear in unit."""
        low = text.lower()
        return sum(1.0 for t in terms if t and t in low)

    @staticmethod
    def _unit_query_terms(obj: Any, decision: ActionDecision) -> List[str]:
        import re

        raw = []
        if obj is not None:
            raw += [obj.intent or "", obj.statement or ""]
            raw += list(getattr(obj, "evidence_required", []) or [])
            raw += list(getattr(obj, "synonyms", []) or [])
        raw += [decision.query or "", decision.instructions or ""]
        terms: List[str] = []
        seen: set = set()
        for chunk in raw:
            for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(chunk).lower()):
                if tok not in seen:
                    seen.add(tok)
                    terms.append(tok)
        return terms
    async def _read_chunk(self, corpus: Any, chunk_id: str):
        idx = self._unit_index_obj(corpus)
        if idx is not None:
            text = await asyncio.to_thread(idx.get, chunk_id)
            kind = await asyncio.to_thread(idx.unit_kind, chunk_id)
            if text:
                return text, kind, ""
        # fall back to the raw chunk text + section
        df = getattr(corpus, "_df", None)
        if df is not None:
            row = df[df["id"] == chunk_id]
            if not row.empty:
                r = row.iloc[0]
                return (str(r.get("text") or ""), str(r.get("chunk_type") or "paragraph"),
                        str(r.get("section") or ""))
        return "", "paragraph", ""

    # -- FIND_SECTIONS ----------------------------------------------------

    async def _find_sections(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        doc_id = (decision.document_id or "").strip()
        if not doc_id:
            return ActionResult(action=ActionType.FIND_SECTIONS, status="done",
                                summary="need document_id to list sections")
        corpus = self._corpus_obj()
        if corpus is None:
            return ActionResult(action=ActionType.FIND_SECTIONS, status="failed",
                                summary="corpus unavailable; cannot list sections")

        df = getattr(corpus, "_df", None)
        if df is None:
            return ActionResult(action=ActionType.FIND_SECTIONS, status="failed",
                                summary="corpus DataFrame not loaded")
        sub = df[df["document_id"] == doc_id]
        if sub.empty:
            return ActionResult(action=ActionType.FIND_SECTIONS, status="done",
                                summary=f"document {doc_id} not in corpus")

        sections: List[Dict[str, Any]] = []
        if {"section", "subsection"}.issubset(sub.columns):
            grouped = sub.groupby(["section", "subsection"], dropna=False)
            for (sec, subsec), grp in grouped:
                sections.append({
                    "section": "" if sec is None or (isinstance(sec, float) and sec != sec) else str(sec),
                    "subsection": "" if subsec is None or (isinstance(subsec, float) and subsec != subsec) else str(subsec),
                    "n_chunks": int(len(grp)),
                })
        else:
            sections.append({"section": "(unknown)", "subsection": "", "n_chunks": int(len(sub))})

        tables = int(sub["table_id"].notna().sum()) if "table_id" in sub.columns else 0
        figures = int(sub["figure_id"].notna().sum()) if "figure_id" in sub.columns else 0
        brief = "; ".join(f"{s['section'] or s['subsection'] or '(untitled)'} ({s['n_chunks']})"
                          for s in sections[:12])
        # record what this call exposed so repeated listings count as no-progress.
        for s in sections:
            key = f"{doc_id}:{s['section']}:{s['subsection']}"
            if key not in state.sections_seen:
                state.sections_seen.append(key)
        self._emit("step", iteration=state.iteration, action="FIND_SECTIONS",
                   input={"document_id": doc_id},
                   output={"sections": sections, "tables": tables, "figures": figures},
                   status="done")
        return ActionResult(
            action=ActionType.FIND_SECTIONS,
            summary=f"document {doc_id}: {len(sections)} section(s), tables={tables}, figures={figures} — {brief}",
            data={"document_id": doc_id, "sections": sections, "tables": tables, "figures": figures},
        )

    # -- VERIFY -----------------------------------------------------------

    async def _verify(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        obj = state.objective(decision.objective_id) if decision.objective_id else None
        if obj is None:
            for o in state.objectives:
                if o.status not in (ObjectiveStatus.SUPPORTED, ObjectiveStatus.CONTRADICTED):
                    obj = o
                    break
        if obj is None:
            return ActionResult(action=ActionType.VERIFY, status="done",
                                summary="no objective to verify (run DECOMPOSE first)")

        passages = await self._verify_candidates(state, obj, decision)
        self._emit("agent_spawn", iteration=state.iteration, action="VERIFY",
                   agent="verifier",
                   input={"objective": obj.id, "statement": obj.statement,
                          "passages": [(p.document_id or p.chunk_id, p.section) for p in passages]})
        try:
            verdict = await self._verifier_obj().verify(obj, passages)
        except Exception as exc:
            self._emit("failure", iteration=state.iteration, action="VERIFY",
                       error=str(exc)[:160])
            return ActionResult(action=ActionType.VERIFY, status="failed",
                                summary=f"verification failed: {str(exc)[:160]}")

        text_by_chunk = {p.chunk_id: p.text for p in passages if p.chunk_id}
        items: List[VerifiedEvidence] = []
        for i, a in enumerate(verdict.assessments, start=1):
            quality, support = _map_relevance(a.relevance, a.quality, a.support)
            if quality not in (EvidenceQuality.DIRECT, EvidenceQuality.INDIRECT):
                # not_relevant / unknown passages are NOT answer material:
                # they still get their per-chunk verdict event below (UI), but
                # only relevant + partially relevant units become evidence the
                # synthesizer may cite.
                continue
            unit_text = text_by_chunk.get(a.chunk_id) or ""
            # Anchor the excerpt on the objective's terms: the unit may be a
            # whole table whose relevant row is deep inside (e.g. a literal
            # definition), so a head slice would drop the evidence before the
            # synthesizer ever sees it.
            excerpt = _evidence_excerpt(unit_text, obj) or a.section or ""
            items.append(VerifiedEvidence(
                id=f"E-{obj.id}-{i}",
                objective_id=obj.id,
                document_id=a.document_id,
                chunk_id=a.chunk_id,
                section=a.section,
                excerpt=excerpt,
                quality=quality,
                support=support,
                confidence=a.confidence,
                note=a.note or a.relevance,
            ))
        added = state.add_evidence(items)

        obj.status = verdict.status
        obj.gap = verdict.gap
        obj.caveats = list(verdict.caveats)
        for c in verdict.contradictions:
            if c and c not in state.contradictions:
                state.contradictions.append(c)
        if verdict.gap and verdict.gap not in state.gaps:
            state.gaps.append(verdict.gap)
        state.confidence = self._overall_confidence(state)

        # Per-chunk verdicts so the UI can flip an "unverified" chunk card to
        # its verified state (green/amber/red/rejected).
        for a in verdict.assessments:
            quality, support = _map_relevance(a.relevance, a.quality, a.support)
            self._emit("verdict", iteration=state.iteration, action="VERIFY",
                       objective_id=obj.id, document_id=a.document_id, chunk_id=a.chunk_id,
                       section=a.section, relevance=a.relevance,
                       quality=quality.value, support=support,
                       confidence=a.confidence)
        self._emit("agent_output", iteration=state.iteration, action="VERIFY",
                   agent="verifier", status="done",
                   output={"objective": obj.id, "status": verdict.status.value,
                           "confidence": verdict.confidence,
                           "gap": verdict.gap,
                           "assessments": [{"document_id": a.document_id,
                                            "chunk_id": a.chunk_id,
                                            "relevance": a.relevance,
                                            "quality": _map_relevance(a.relevance, a.quality, a.support)[0].value,
                                            "support": _map_relevance(a.relevance, a.quality, a.support)[1],
                                            "confidence": a.confidence}
                                           for a in verdict.assessments]})
        return ActionResult(
            action=ActionType.VERIFY,
            summary=(f"verified {obj.id}: {verdict.status.value} "
                     f"(conf={verdict.confidence:.2f}); {added} evidence item(s) added"),
            data={"objective": obj.id, "status": verdict.status.value, "evidence": added},
        )

    def n_verify_candidates(self, state: ResearchState, objective_id: str) -> int:
        """How many passages VERIFY would judge (bounds its time budget)."""
        return len(self._candidates_for(state, objective_id))

    async def _verify_candidates(self, state: ResearchState, obj: Any,
                                 decision: ActionDecision) -> List[CandidatePassage]:
        """Candidate passages for the verifier, expanding any full-document unit.

        A whole-paper unit (section "(full document)" / unit_kind "document") is
        NEVER sent to the verifier as-is. Instead its containing structural
        units (paragraph / table / figure) are restored and ranked by
        relevance to the objective, so the ACTUAL RELEVANT PART of that
        document is verified (never the whole paper, never silently dropped).
        """
        candidates = self._candidates_for(state, obj.id)
        pending = [
            d for d in state.documents
            if d.objective_id in ("", obj.id)
            and (d.unit_kind == "document" or (d.section or "").strip() == "(full document)")
        ]
        pending += [
            c for c in state.candidates
            if c.objective_id in ("", obj.id)
            and (getattr(c, "unit_kind", "") == "document"
                 or (c.section or "").strip() == "(full document)")
        ]
        if not pending:
            return candidates

        corpus = self._corpus_obj()
        extra: List[CandidatePassage] = []
        seen_units: set = set()
        for doc in pending:
            did = (doc.document_id or "").strip()
            if not did:
                continue
            units = (
                await self._document_units(corpus, did, state, decision, top_n=4)
                if corpus is not None else []
            )
            if not units and (doc.text or "").strip():
                # No structural index reachable: fall back to extracting the
                # relevant paragraphs straight out of the full-document text.
                terms = self._unit_query_terms(
                    state.objective(decision.objective_id) if decision.objective_id else None,
                    decision,
                )
                units = self._split_paragraph_units(
                    doc.text, terms, did, top_n=4,
                )
            for u in units:
                cid = u.get("chunk_id") or ""
                if cid and cid in seen_units:
                    continue
                if cid:
                    seen_units.add(cid)
                extra.append(CandidatePassage(
                    chunk_id=cid,
                    document_id=did,
                    section=u.get("section", ""),
                    text=u.get("text", ""),
                    objective_id=obj.id,
                    origin="read_document",
                ))
            if not units:
                msg = "[verify] no units restored for full-document " + did + " - skipped"
                print(msg, flush=True)

        if not extra:
            return candidates
        seen = {c.chunk_id or (c.document_id, c.text[:80]) for c in candidates}
        merged = list(candidates)
        for e in extra:
            key = e.chunk_id or (e.document_id, e.text[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append(e)
        merged.sort(key=lambda p: -(p.score or 0.0))
        return merged[:_MAX_VERIFY_PASSAGES]

    def _candidates_for(self, state: ResearchState, objective_id: str) -> List[CandidatePassage]:
        """Candidate passages to verify for one objective (bounded)."""
        out: List[CandidatePassage] = []
        seen = set()
        for d in state.documents:
            if d.objective_id not in ("", objective_id):
                continue
            # NEVER verify a whole paper: only structural units (paragraph /
            # table / figure) may become candidate passages. A stray
            # (full document) unit must never reach the verifier.
            if d.unit_kind == "document" or (d.section or "").strip() == "(full document)":
                continue
            key = d.chunk_id or (d.document_id, d.text[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append(CandidatePassage(
                chunk_id=d.chunk_id, document_id=d.document_id, section=d.section,
                text=d.text, objective_id=objective_id, origin="global_retrieve",
            ))
        for c in state.candidates:
            if c.objective_id not in ("", objective_id):
                continue
            # Same full-paper backstop as documents: never verify a whole
            # paper, only structural units.
            if c.unit_kind == "document" or (c.section or "").strip() == "(full document)":
                continue
            key = c.chunk_id or (c.document_id, c.text[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        # best-first by score when available, then bound the count.
        out.sort(key=lambda p: -(p.score or 0.0))
        return out[:_MAX_VERIFY_PASSAGES]

    @staticmethod
    def _overall_confidence(state: ResearchState) -> float:
        if not state.evidence:
            return 0.0
        confs = [e.confidence for e in state.evidence]
        return round(sum(confs) / len(confs), 4)

    # -- SYNTHESIZE / STOP -------------------------------------------------

    async def _synthesize(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        self._emit("agent_spawn", iteration=state.iteration, action="SYNTHESIZE",
                   agent="synthesizer",
                   input={"evidence": len(state.evidence),
                          "objectives": [(o.id, o.status.value) for o in state.objectives],
                          "gaps": list(state.gaps)})
        try:
            report = await self._synthesizer_obj().synthesize(state)
        except Exception as exc:
            self._emit("failure", iteration=state.iteration, action="SYNTHESIZE",
                       error=str(exc)[:160])
            return ActionResult(action=ActionType.SYNTHESIZE, status="failed",
                                summary=f"synthesis failed: {str(exc)[:160]}")
        state.final_answer = report.model_dump()
        state.terminal = True
        state.stop_reason = "synthesized"
        state.confidence = report.confidence
        self._emit("agent_output", iteration=state.iteration, action="SYNTHESIZE",
                   agent="synthesizer", status="done",
                   output={"summary": report.summary,
                           "sections": [s.heading for s in report.sections],
                           "limitations": report.limitations,
                           "unresolved": report.unresolved,
                           "confidence": report.confidence})
        return ActionResult(
            action=ActionType.SYNTHESIZE,
            summary=f"synthesized final answer (confidence={report.confidence:.2f})",
            data={"answer": report.model_dump()},
        )

    async def _stop(self, decision: ActionDecision, state: ResearchState) -> ActionResult:
        state.terminal = True
        state.stop_reason = (decision.rationale or decision.instructions or "orchestrator stopped").strip()
        return ActionResult(action=ActionType.STOP, status="done", summary=state.stop_reason)
