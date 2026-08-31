"""Agentic v3 - structured event stream (observability/UI), v2-compatible JSONL.

A small thread-safe JSONL emitter (stdlib only; the emitter was previously
shared from src.agentic_v2.events and is inlined here so agentic v3 has no
legacy package dependency):

    AGENTIC_V3_EVENTS_FILE=agentic_v3_events.jsonl

Event types (all carry a timestamp):
  run_start          {question, budget}
  master_plan        {tasks: [{id, title, objective, evidence_requirements}]}
  task_start         {task_id, title}
  task_done          {task_id, status, summary}
  search_round       {task_id, round_no, queries: {requirement_id: [queries]}}
  retrieved          {task_id, requirement_id, papers: [{document_id, section, score}]}
  verdict            {task_id, requirement_id, document_id, chunk_id, relevance,
                      answers_task, support, confidence, accepted, note}
  evidence_added     {task_id, requirement_id, evidence_id, document_id, support, source}
  deep_inspection    {task_id, requirement_id, document_id, status, findings, verified}
  worker_report      {task_id, status, searches_used, deep_inspections_used, requirements}
  contradiction      {contradiction_id, claim, kind, evidence_a, evidence_b, description}
  resolution         {contradiction_id, status, explanation, additional_papers}
  final_evidence     {evidence: n, gaps: [...], contradictions: n, resolved: n, unresolved: n}
  run_end            {stop_reason, confidence, answer}

For the v2 UI to render common nodes it also emits the legacy shapes
agent_spawn / agent_output where they map 1:1.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional


class EventEmitter:
    """Appends one JSON object per line to ``path`` (thread-safe)."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def emit(self, type_: str, **fields: Any) -> None:
        record = {"ts": time.time(), "type": type_, **fields}
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
        except Exception:
            line = json.dumps({"ts": time.time(), "type": type_,
                               "error": "unserializable"})
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def truncate(self) -> None:
        """Start a fresh event log (the UI calls this before a new run)."""
        with self._lock:
            open(self.path, "w", encoding="utf-8").close()


_SINGLETON: Optional[EventEmitter] = None


def get_emitter(path: Optional[str] = None) -> Optional[EventEmitter]:
    global _SINGLETON
    if path:
        if _SINGLETON is None or _SINGLETON.path != path:
            _SINGLETON = EventEmitter(path)
        return _SINGLETON
    return _SINGLETON


def reset_emitter() -> None:
    global _SINGLETON
    _SINGLETON = None


__all__ = ["EventEmitter", "V3Events", "get_emitter", "reset_emitter"]


class V3Events:
    """Typed convenience wrapper over the shared JSONL EventEmitter."""

    def __init__(self, emitter: Optional[EventEmitter] = None):
        self.emitter = emitter

    @property
    def enabled(self) -> bool:
        return self.emitter is not None

    def emit(self, type_: str, **fields: Any) -> None:
        if self.emitter is not None:
            self.emitter.emit(type_, **fields)

    # -- typed helpers --------------------------------------------------

    def run_start(self, question: str, budget: dict, run_id: str = "") -> None:
        self.emit("run_start", run_id=run_id, question=question, budget=budget)

    def master_plan(self, plan: dict) -> None:
        self.emit("master_plan", tasks=plan.get("tasks", []),
                  rationale=plan.get("rationale", ""))

    def task_start(self, task_id: str, title: str) -> None:
        self.emit("task_start", task_id=task_id, title=title)

    def task_done(self, task_id: str, status: str, summary: str = "",
                  stop_reason: str = "") -> None:
        self.emit("task_done", task_id=task_id, status=status, summary=summary,
                  stop_reason=stop_reason)

    def search_round(self, task_id: str, round_no: int, queries: dict) -> None:
        self.emit("search_round", task_id=task_id, round_no=round_no, queries=queries)

    def retrieved(self, task_id: str, requirement_id: str, papers: list,
                  query: str = "", attempt_id: str = "") -> None:
        self.emit("retrieved", task_id=task_id, requirement_id=requirement_id,
                  query=query, attempt_id=attempt_id, papers=papers)

    def verdict(self, task_id: str, requirement_id: str, document_id: str,
                chunk_id: str, relevance: str, answers_task: str, support: str,
                confidence: float, accepted: bool, note: str = "") -> None:
        self.emit("verdict", task_id=task_id, requirement_id=requirement_id,
                  document_id=document_id, chunk_id=chunk_id,
                  relevance=relevance, answers_task=answers_task,
                  support=support, confidence=round(float(confidence or 0.0), 4),
                  accepted=accepted, note=note)

    def evidence_added(self, task_id: str, requirement_id: str, evidence_id: str,
                       document_id: str, support: str, source: str,
                       attempt_id: str = "") -> None:
        self.emit("evidence_added", task_id=task_id,
                  requirement_id=requirement_id, evidence_id=evidence_id,
                  document_id=document_id, support=support, source=source,
                  attempt_id=attempt_id)

    def deep_inspection(self, task_id: str, requirement_id: str, document_id: str,
                        status: str, findings: int = 0, verified: bool = False,
                        detail: str = "", attempt_id: str = "") -> None:
        self.emit("deep_inspection", task_id=task_id,
                  requirement_id=requirement_id, document_id=document_id,
                  status=status, findings=findings, verified=verified,
                  detail=detail, attempt_id=attempt_id)

    def worker_report(self, task_id: str, status: str, searches_used: int,
                      deep_inspections_used: int, requirements: list,
                      stop_reason: str = "", evidence: Optional[list] = None) -> None:
        self.emit("worker_report", task_id=task_id, status=status,
                  searches_used=searches_used,
                  deep_inspections_used=deep_inspections_used,
                  requirements=requirements, stop_reason=stop_reason,
                  evidence=evidence or [])

    # -- scoped state transitions (requirement 10) ---------------------

    def evidence_state(self, task_id: str, requirement_id: str,
                       evidence_id: str, old: str, new: str) -> None:
        """One evidence item transition, e.g. retrieved -> under_review.

        The scope (task_id, requirement_id, evidence_id) is carried on the
        event itself so logs can never be attributed ambiguously.
        """
        self.emit("evidence_state", task_id=task_id,
                  requirement_id=requirement_id, evidence_id=evidence_id,
                  old_state=old, new_state=new)

    def requirement_state(self, task_id: str, requirement_id: str,
                          old: str, new: str, coverage: int, target_n: int,
                          run_id: str = "") -> None:
        """One requirement transition, e.g. unsatisfied -> partially_supported."""
        self.emit("requirement_state", run_id=run_id, task_id=task_id,
                  requirement_id=requirement_id,
                  old_state=old, new_state=new,
                  coverage=coverage, target_n=target_n)

    def replan(self, task_id: str, requirement_id: str, attempt_id: str,
               diagnosis: str, missing_evidence: str, queries: list,
               run_id: str = "") -> None:
        """The adaptive-retrieval replanning step (requirement 1)."""
        self.emit("replan", run_id=run_id, task_id=task_id,
                  requirement_id=requirement_id, attempt_id=attempt_id,
                  diagnosis=diagnosis, missing_evidence=missing_evidence,
                  queries=queries)

    def contradiction(self, cid: str, claim: str, kind: str,
                      evidence_a: list, evidence_b: list, description: str) -> None:
        self.emit("contradiction", contradiction_id=cid, claim=claim,
                  kind=kind, evidence_a=evidence_a, evidence_b=evidence_b,
                  description=description)

    def resolution(self, cid: str, status: str, explanation: str,
                   additional_papers: list) -> None:
        self.emit("resolution", contradiction_id=cid, status=status,
                  explanation=explanation, additional_papers=additional_papers)

    def final_evidence(self, evidence: int, gaps: list, contradictions: int,
                       resolved: int, unresolved: int) -> None:
        self.emit("final_evidence", evidence=evidence, gaps=gaps,
                  contradictions=contradictions, resolved=resolved,
                  unresolved=unresolved)

    def run_end(self, stop_reason: str, confidence: float, answer: str) -> None:
        self.emit("run_end", stop_reason=stop_reason,
                  confidence=round(float(confidence or 0.0), 4), answer=answer)

    def answer(self, report: dict) -> None:
        """The full final answer report (summary + sections + citations)."""
        self.emit("answer", report=report, summary=report.get("summary", ""))

    # -- legacy shapes (so the v2 UI can render agent nodes) ------------

    def agent_spawn(self, iteration: int, action: str, agent: str, input_data: dict) -> None:
        self.emit("agent_spawn", iteration=iteration, action=action,
                  agent=agent, input=input_data)

    def agent_output(self, iteration: int, action: str, agent: str,
                     output: dict, status: str = "done") -> None:
        self.emit("agent_output", iteration=iteration, action=action,
                  agent=agent, output=output, status=status)
