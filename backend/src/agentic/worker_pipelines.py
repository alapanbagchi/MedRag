"""Agentic v3 - the Worker's two inner pipelines.

Two self-contained units extracted from WorkerAgent.run(), one per worker
stage:

  * SearchPipeline        - one retrieval round for one requirement
                            (Stage 6 retrieve -> Stage 7 context -> Stage 8
                            CRITIC gate -> verdict folding -> promising
                            bookkeeping).
  * DeepInspectionPipeline - one bounded full-paper inspection of a
                            "promising" paper whose excerpts did not answer
                            (Stages 11/14).

Both are pure logic over injected dependencies (retriever / critic /
deep_inspector + events) and mutate ONLY the per-run WorkerRunContext and
the task/requirement they are handed - no module state, no shared globals.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.critic import (
    CriticVerdict,
    is_promising_for_deep_inspection,
)
from src.agentic.state import (
    EvidenceRequirement,
    EvidenceSource,
    EvidenceStatus,
    ResearchTask,
    RetrievedPaper,
    SupportDirection,
    VerifiedEvidence,
)
from src.lib.utils import plural

logger = logging.getLogger("src.agentic.worker_pipelines")


class SearchPipeline:
    """One search attempt: retrieve -> stamp scope -> CRITIC gate -> fold.

    Every candidate paper is stamped with its OWN scoped id
    (evidence_id = task.requirement.attempt.seq) BEFORE retrieval output is
    inspected, so a critic result can never migrate to another requirement
    (requirement 3).
    """

    def __init__(self, retriever: Any, critic: Any, events: Any = None):
        self._retriever = retriever
        self._critic = critic
        self.events = events

    async def run(
        self,
        task: ResearchTask,
        req: EvidenceRequirement,
        query: str,
        round_no: int,
        attempt_id: str,
        ctx: Any,
        *,
        run_id: str = "",
    ) -> None:
        """One query: retrieve -> context -> CRITIC -> record, in ctx."""
        # one SEARCH attempt = one executed query (budget accounting per
        # query, not per round - so adaptive replanning can always run its
        # first query before the search budget is checked again)
        ctx.budget.searches_used += 1
        papers = await self._retriever.search(
            task, req, query,
            exclude_chunk_ids=sorted(ctx.seen_chunks),
            top_k=ctx.budget.max_papers_per_round,
            round_no=round_no,
        )
        new_papers = [p for p in papers if p.chunk_id not in ctx.seen_chunks]
        if papers:
            self._bullet(
                f"Retrieved {plural(len(papers), 'candidate passage')} for "
                f"{req.id} (query: “{query}”): "
                + ", ".join(
                    f"{p.document_id} ({p.section}"
                    + (f", score {p.score:.3f}" if p.score is not None else "")
                    + ")" for p in papers[:5])
                + ("…" if len(papers) > 5 else "") + ".",
                task=task)
        if self.events is not None:
            self.events.retrieved(
                task.id, req.id,
                [{"document_id": p.document_id, "section": p.section,
                  "score": round(float(p.score or 0.0), 4)} for p in papers],
                query=query, attempt_id=attempt_id)

        notes: list[str] = []
        # stamp scope BEFORE any judgement (requirement 3): pool AND batch
        # both judge only papers whose scope was fixed here first.
        stamped: list[RetrievedPaper] = []
        for idx, paper in enumerate(new_papers, start=1):
            if paper.chunk_id:
                ctx.seen_chunks.add(paper.chunk_id)
            paper = paper.model_copy(update={
                "evidence_id": f"{req.id}.{attempt_id}.E{idx}",
                "task_id": task.id,
                "requirement_id": req.id,
                "attempt_id": attempt_id,
                "status": EvidenceStatus.RETRIEVED,
                "source_query": query,
                "round_no": round_no,
            })
            self._emit_evidence_state(task, req, paper.evidence_id,
                                      EvidenceStatus.RETRIEVED.value,
                                      EvidenceStatus.UNDER_REVIEW.value)
            stamped.append(paper)

        # Stage 8: CRITIC gate — all passages of this query in ONE call
        # (Mistral batch when enabled, otherwise sequential). A None verdict
        # means critic-failed (cannot verify), NOT a rejection.
        judge_papers = getattr(self._critic, "judge_papers", None)
        if judge_papers is not None:
            verdict_pairs = await judge_papers(
                task, req, stamped, run_id=run_id, attempt_id=attempt_id)
        else:
            # duck-typed / test critics with only the per-paper judge()
            verdict_pairs = []
            for paper in stamped:
                try:
                    verdict_pairs.append((paper, await self._critic.judge(
                        task, req, paper, run_id=run_id, attempt_id=attempt_id)))
                except Exception as exc:
                    logger.warning("critic failed %s/%s (%s)", task.id, req.id, exc)
                    verdict_pairs.append((paper, None))
        for paper, verdict in verdict_pairs:
            if verdict is None:
                notes.append(f"critic-failed: {paper.document_id}")
                continue
            self._record_verdict(task, req, paper, verdict, run_id=run_id)
            if verdict.accepted:
                if verdict.support == SupportDirection.SUPPORTS:
                    notes.append(f"accepted(support): {paper.document_id}")
                elif verdict.support == SupportDirection.CONTRADICTS:
                    notes.append(f"accepted(contradicts): {paper.document_id}")
            if is_promising_for_deep_inspection(verdict):
                existing = {d["doc"] for d in ctx.promising.get(req.id, [])}
                if paper.document_id and paper.document_id not in existing:
                    ctx.promising.setdefault(req.id, []).append({
                        "doc": paper.document_id,
                        "hint": (verdict.note or "")[:240],
                    })
            if req.satisfied():
                break

        ctx.attempts.setdefault(req.id, []).append({
            "query": query,
            "round": round_no,
            "attempt_id": attempt_id,
            "papers": len(papers),
            "notes": " | ".join(notes)[:800],
        })
        for p in papers:
            ctx.attempts.setdefault(req.id + ":docs", []).append({
                "document_id": p.document_id,
                "section": p.section,
                "excerpt": (p.text or "")[:300],
            })

    # ------------------------------------------------------------------
    def _record_verdict(self, task: ResearchTask, req: EvidenceRequirement,
                        paper: RetrievedPaper, verdict: CriticVerdict,
                        *, run_id: str = "") -> None:
        """Fold one scoped verdict into the requirement.

        The verdict carries its own scope; the requirement that records it
        MUST be the same requirement it was judged against (a reviewer
        mismatch raises - the log can never be polluted cross-requirement).
        """
        if verdict.requirement_id and verdict.requirement_id != req.id:
            raise ValueError(
                f"verdict scope mismatch: verdict.requirement_id="
                f"{verdict.requirement_id} != req.id={req.id}")
        entry = {
            "evidence_id": paper.evidence_id or verdict.evidence_id or "",
            "document_id": paper.document_id,
            "chunk_id": paper.chunk_id,
            "attempt_id": verdict.attempt_id or paper.attempt_id or "",
            "relevance": verdict.relevance.value,
            "answers_task": verdict.answers_task.value,
            "support": verdict.support.value,
            "confidence": verdict.confidence,
            "note": verdict.note,
            "accepted": verdict.accepted,
        }
        req.record_review(entry)
        if self.events is not None:
            # stream the critic verdict with its reasoning so the UI shows
            # exactly what was verified and WHY (accept or reject)
            self.events.verdict(
                task.id, req.id, paper.document_id, paper.chunk_id,
                verdict.relevance.value, verdict.answers_task.value,
                verdict.support.value, verdict.confidence, verdict.accepted,
                note=verdict.note or "")
        from src.lib import logfire_obs as lf
        lf.record_metric("evidence_accepted" if verdict.accepted
                         else "evidence_rejected", 1,
                         task_id=task.id, requirement_id=req.id,
                         document=paper.document_id)
        old = req.status
        if verdict.accepted:
            item = self._evidence_from_verdict(
                task, req, paper, verdict, run_id=run_id)
            terminal = (EvidenceStatus.CONTRADICTORY
                        if item.support == SupportDirection.CONTRADICTS
                        else EvidenceStatus.ACCEPTED)
            item.status = terminal
            if req.add_evidence(item):
                self._emit_evidence_state(task, req, item.id,
                                          EvidenceStatus.UNDER_REVIEW.value,
                                          terminal.value)
                if self.events is not None:
                    self.events.evidence_added(
                        task.id, req.id, item.id, item.document_id,
                        item.support.value, item.source.value,
                        attempt_id=item.attempt_id)
                self._bullet(
                    f"✓ Verified {item.document_id} ({item.section}) for "
                    f"{req.id} — {item.support.value} the requirement "
                    f"(confidence {item.confidence:.0%}): "
                    f"{(verdict.note or '')[:160]}",
                    task=task, agent="critic")
        else:
            req.rejected += 1
            self._emit_evidence_state(
                task, req, entry["evidence_id"],
                EvidenceStatus.UNDER_REVIEW.value, EvidenceStatus.REJECTED.value)
            self._bullet(
                f"✗ Rejected {paper.document_id} ({paper.section}) for "
                f"{req.id} — {(verdict.note or 'not relevant to the requirement')[:160]}",
                task=task, agent="critic")
        if req.status != old:
            self._emit_requirement_state(task, req, old, run_id=run_id)

    @staticmethod
    def _evidence_from_verdict(task: ResearchTask, req: EvidenceRequirement,
                               paper: RetrievedPaper,
                               verdict: CriticVerdict,
                               *, run_id: str = "") -> VerifiedEvidence:
        from src.agents.critic import evidence_excerpt

        return VerifiedEvidence(
            id=paper.evidence_id or f"E-{task.id}-{req.id}-{len(req.accepted) + 1}",
            run_id=run_id,
            task_id=task.id,
            requirement_id=req.id,
            attempt_id=verdict.attempt_id or paper.attempt_id or "",
            document_id=paper.document_id,
            chunk_id=paper.chunk_id,
            section=paper.section,
            excerpt=evidence_excerpt(paper.text),
            claim=verdict.note or verdict.support.value,
            support=verdict.support,
            confidence=verdict.confidence,
            source=EvidenceSource.RETRIEVAL,
            retrieval_method=paper.retrieval_method,
            rank=paper.rank,
            critic_verdict=verdict.model_dump(mode="json"),
            note=verdict.note,
            search_query=paper.source_query,
        )

    # ------------------------------------------------------------------
    def _emit_evidence_state(self, task: ResearchTask, req: EvidenceRequirement,
                             evidence_id: str, old: str, new: str) -> None:
        if self.events is not None:
            self.events.evidence_state(task.id, req.id, evidence_id, old, new)

    def _emit_requirement_state(self, task: ResearchTask,
                                req: EvidenceRequirement,
                                old: Any, *, run_id: str = "") -> None:
        if self.events is not None:
            self.events.requirement_state(
                task.id, req.id, old.value, req.status.value,
                req.coverage(), req.target_n, run_id=run_id)

    @staticmethod
    def _bullet(msg: str, *, task: ResearchTask, agent: str = "worker") -> None:
        from src.lib.trace import get_trace
        get_trace().bullet(msg, agent=agent)


class DeepInspectionPipeline:
    """One bounded full-paper inspection of a promising document."""

    def __init__(self, deep_inspector: Any, events: Any = None):
        self._deep_inspector = deep_inspector
        self.events = events

    async def run(self, task: ResearchTask, req: EvidenceRequirement,
                  cand: dict[str, str], ctx: Any, *,
                  run_id: str = "", attempt_id: str = "") -> None:
        """Inspect one document; verified passages fold into the requirement."""
        doc_id = cand.get("doc", "")
        if not doc_id:
            return
        ctx.budget.deep_inspections_used += 1
        task.deep_inspections_used += 1
        if self.events is not None:
            self.events.deep_inspection(task.id, req.id, doc_id, "starting",
                                        attempt_id=attempt_id)
        hint = (cand.get("hint") or "").strip()
        from src.lib.trace import get_trace
        get_trace().bullet(
            f"Deep-inspecting {doc_id} for {req.id} — the search excerpt was "
            f"not decisive, reading the full paper now"
            + (f" (why it was flagged: {hint[:140]})" if hint else "")
            + ".",
            agent="worker")
        try:
            from src.lib import logfire_obs as lf
            with lf.span("deep_inspection", document=doc_id, requirement=req.id):
                found = await self._deep_inspector.inspect(
                    task, req, doc_id, context_hint=cand.get("hint", ""),
                    run_id=run_id, attempt_id=attempt_id)
            lf.record_metric("deep_inspections", 1,
                             task_id=task.id, requirement_id=req.id, document=doc_id)
        except Exception as exc:
            logger.warning("deep inspection failed %s: %s", doc_id, exc)
            if self.events is not None:
                self.events.deep_inspection(task.id, req.id, doc_id, "failed",
                                            detail=str(exc)[:160])
            return
        added = 0
        for item in found:
            item.id = f"{req.id}.{attempt_id}.DI{len(req.accepted) + 1}"
            if req.add_evidence(item):
                added += 1
        if self.events is not None:
            self.events.deep_inspection(task.id, req.id, doc_id,
                                        "accepted" if added else "no_evidence",
                                        findings=len(found), verified=added > 0)
        get_trace().bullet(
            f"Deep inspection of {doc_id} for {req.id}: "
            f"{'accepted ' + plural(added, 'verified passage') if added else 'found no verifiable evidence — excerpt-based verdict stands'}.",
            agent="deep_inspector")