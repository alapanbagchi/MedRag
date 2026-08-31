"""Agentic v3 - Stage 2: the Master Orchestrator.

The Master is the TOP-LEVEL PLANNING agent. It does NOT retrieve. It:

  1. understands the user's information need,
  2. identifies the distinct questions inside the query,
  3. decomposes the query into independent tasks,
  4. defines, per task, the evidence required,
  5. sets minimum evidence thresholds (N independent papers per
     requirement) and explicit stop criteria,
  6. emits a structured retrieval plan (MasterPlan) whose tasks are
     dispatched to the parallel Worker sub-orchestrators (Stage 3).

ROBUST DECOMPOSITION (accommodates strict/verbose master prompts):

  The system prompt lives in src/prompts/agents/master.txt and may be
  edited freely. Under a very strict prompt some models "think in prose":
  they draft the whole plan (Task N / Objective / Intent / Evidence /
  Entities) inside a <thought> block as markdown bullets and then truncate
  BEFORE emitting the required JSON. plan() therefore never gives up after
  one parse failure:

    1. try the structured JSON path,
    2. salvage the JSON object from the RAW response (including <thought>),
    3. salvage the plan from the PROSE OUTLINE (bullet blocks) if JSON is
       absent,
    4. retry once with a "JSON only, no <thought>" nudge,
    5. only then fall back to the deterministic single task.

  Missing per-task fields are filled deterministically (default intent,
  evidence requirement from the objective, entities extracted from the
  objective text), so downstream workers never receive empty tasks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.agents.state import (
    EvidenceRequirement,
    MasterPlan,
    ResearchTask,
    RunBudget,
)
from src.prompts.load import load_prompt

MASTER_SYSTEM_PROMPT = load_prompt("agents", "master.txt")

logger = logging.getLogger("src.agents.master")

# Junk the smaller models echo into free-text fields (mirrors planner guard).
_JUNK_WORDS = {
    "true", "false", "null", "none", "text", "id", "focus", "query",
    "target", "objective", "task", "requirement", "intent", "evidence",
}

# Appended on the final retry when the model wrote prose but no JSON.
_JSON_ONLY_NUDGE = (
    "\n\nIMPORTANT: your previous response did not contain the "
    "retrieval-plan JSON object. Output ONLY the JSON object now - no "
    "<thought>, no prose, no markdown, no bullets."
)


class PlanTask(BaseModel):
    """LLM-shaped task blueprint (targets + requirements, no thresholds).

    The field mirrors the LLM output schema exactly: the Master emits
    'evidence_required' (the concrete obligations), the deterministic
    builder turns each into an EvidenceRequirement with target_n.
    """
    id: str = "T1"
    title: str = ""
    objective: str = ""
    intent: str = ""
    evidence_required: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)


class MasterDecomposition(BaseModel):
    """Structured output of the Master Orchestrator."""
    rationale: str = ""
    tasks: List[PlanTask] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic field cleaning / entity fallback
# ---------------------------------------------------------------------------

def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split()).strip()
    if not text:
        return ""
    if text.lower() in _JUNK_WORDS:
        return ""
    # Reject punctuation-only / junk shells ("", ",", "[]", ...) that small
    # models echo into free-text fields - they must never become evidence
    # requirements or task titles.
    if len(re.sub(r"[^a-z0-9]", "", text.lower())) < 3:
        return ""
    return text[:400]


def _clean_req(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    # drop template echoes like "evidence_required: ..."
    if re.search(r"\b(evidence|requirement|objective|intent)\s*:", text.lower()):
        return ""
    return text


def _clean_entity(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = text.rstrip(".,;:")   # drop trailing sentence punctuation
    if len(re.sub(r"[^a-z0-9]", "", text.lower())) < 3:
        return ""
    return text


_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "with", "to",
    "which", "what", "how", "whether", "is", "are", "be", "determine",
    "identify", "assess", "evaluate", "supported", "evidence", "question",
    "answer", "people", "patients", "should", "their", "from", "that",
}


def _entities_from_text(text: str, cap: int = 6) -> List[str]:
    """Deterministic entity fallback when the model supplied none.

    Extracts up to ''cap'' content n-grams (1-3 words) from the objective /
    title, skipping generic function words - enough to give the worker a
    terminology pool so search planning never starts empty.
    """
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", (text or "").lower())
    phrases: List[str] = []
    i = 0
    while i < len(toks):
        if toks[i] in _STOP or len(toks[i]) < 3:
            i += 1
            continue
        phrase = [toks[i]]
        j = i + 1
        while j < len(toks) and len(phrase) < 3 \
                and toks[j] not in _STOP and len(toks[j]) >= 3:
            phrase.append(toks[j])
            j += 1
        phrases.append(" ".join(phrase))
        i = j
    out: List[str] = []
    seen: set = set()
    for p in phrases:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# JSON salvage (including <thought>-wrapped responses)
# ---------------------------------------------------------------------------

def unusable(decomposition: Any) -> bool:
    """True when a parse produced no REAL plan (title-only/default fields).

    A task is only "real" when it carries a substantive objective that
    differs from its title AND a concrete evidence requirement - the
    synthetic defaults ("Evidence for: <title>", default intent, title-echo
    objective) indicate the model wrote labels but no plan, so the salvage
    keeps trying (nudge) instead of accepting degraded output.
    """
    tasks = list(getattr(decomposition, "tasks", None) or [])
    if not tasks:
        return True
    for t in tasks:
        obj = (getattr(t, "objective", "") or "").strip()
        title = (getattr(t, "title", "") or "").strip()
        reqs = [str(r or "").strip()
                for r in (getattr(t, "evidence_required", None) or [])]
        real_reqs = [r for r in reqs
                     if r and not r.lower().startswith("evidence for:")]
        real_objective = bool(obj) and (not title or obj != title)
        if real_objective and real_reqs:
            return False
        if real_objective and title and obj != title:
            return False
        # a task whose objective equals its title (or only has defaults) is
        # a label, not a plan
    return True


def first_json(raw: str, output_type: Any) -> Optional[Any]:
    """Extract + validate the first parseable JSON object of the raw text.

    Unlike the strict path, this also inspects text that still contains a
    <thought> block (the model often drafts its plan there).
    """
    from src.lib.utils import _extract_json_objects, strip_think

    if not raw:
        return None
    candidates = [raw, strip_think(raw)]
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue
        try:
            return output_type.model_validate(json.loads(cand))
        except Exception:
            pass
        for obj in _extract_json_objects(cand):
            try:
                return output_type.model_validate(json.loads(obj))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Prose-outline salvage: the model is thinking in bullets, not JSON
# ---------------------------------------------------------------------------

_TASK_LABEL_RE = re.compile(
    r"(?im)^[ \t]*(?:\*\*\s*)?(?:[*-]\s*)?(?:T(\d+)|task\s*(\d+))\s*[:.:\-)\]]+")


def _field_value(segment: str, key: str) -> List[str]:
    """Values of one field (objective/intent/evidence/entities) in a block."""
    out: List[str] = []
    in_field = False
    for ln in segment.splitlines():
        s = ln.strip()
        if not s:
            continue
        head = re.match(r"(?:\*\*\s*)?(?:[*-]\s*)?([a-zA-Z_]+)\s*[:.]", s)
        if head:
            raw_key = head.group(1).lower()
            if raw_key in ("objective", "intent"):
                key2 = raw_key
            elif raw_key.startswith("evidence"):
                key2 = "evidence"
            elif raw_key.startswith("entit"):
                key2 = "entities"
            else:
                key2 = ""
            if key2 == key:
                in_field = True
                text = _clean_text(s[head.end():])
                if text:
                    out.append(text)
                continue
            if in_field:
                break  # a different field ends this one
        elif in_field and s[:1] in ("-", "*", "\u2022"):
            text = _clean_text(s.lstrip("-*\u2022"))
            if text:
                out.append(text)
        elif in_field and s:
            text = _clean_text(s)
            if text and out:
                out[-1] = out[-1] + " " + text
    return out


def _segment_title(segment: str) -> str:
    for ln in segment.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.match(r"(?:\*\*\s*)?(?:[*-]\s*)?T\d+\b", s):
            title = re.sub(r"^[ \t]*(?:\*\*\s*)?(?:[*-]\s*)?T\d+\s*[:.:\-)\]]+\s*", "", s)
            return _clean_text(re.sub(r"\.+$", "", title.lstrip("*").strip()))
        if re.match(r"(?:\*\*\s*)?(?:[*-]\s*)?task\s*\d+\b", s):
            title = re.sub(r"^[ \t]*(?:\*\*\s*)?(?:[*-]\s*)?task\s*\d+\s*[:.:\-)\]]+\s*", "", s)
            return _clean_text(re.sub(r"\.+$", "", title.lstrip("*").strip()))
    return ""


def parse_prose_plan(raw: str) -> Optional[MasterDecomposition]:
    """Salvage the plan from a markdown bullet outline.

    Observed shape under strict master prompts (model thinks aloud):

        * T1: Dietary management of hypertension
            * Objective: ...
            * Intent: ...
            * Evidence: ...
            * Entities: ...
        * T2: ...

    Returns None when the text does not look like an outline at all.
    """
    if not raw:
        return None
    # the outline often sits inside <thought>...</thought> - the tags are
    # structure, not content, so strip them before parsing the bullets
    raw = re.sub(r"</?[a-z][a-z0-9_]*>", " ", raw)
    labels = list(_TASK_LABEL_RE.finditer(raw))
    if not labels:
        return None
    tasks: List[PlanTask] = []
    for i, m in enumerate(labels):
        end = labels[i + 1].start() if i + 1 < len(labels) else len(raw)
        segment = raw[m.start():end]
        title = _segment_title(segment)
        objective = " ".join(_field_value(segment, "objective"))
        intent = " ".join(_field_value(segment, "intent"))
        # the model often writes several items on ONE line ("A, B, C") -
        # split on commas / semicolons / " and" so each becomes an item
        evidence = []
        for _e in _field_value(segment, "evidence"):
            for _part in re.split(r"\s*;\s*", _e):
                _part = _clean_req(_part)
                if _part and _part not in evidence:
                    evidence.append(_part)
        entities = []
        for _e in _field_value(segment, "entities"):
            for _part in re.split(r"\s*[,;]\s*|\s+and\s+", _e):
                _part = _clean_entity(_part)
                if _part and _part not in entities:
                    entities.append(_part)
        label_id = m.group(1) or m.group(2) or str(len(tasks) + 1)
        if not title and not objective and not evidence:
            continue
        if not objective:
            objective = title
        if not title:
            title = objective[:80] or f"Task {label_id}"
        reqs = evidence or [f"Evidence for: {objective}"]
        tasks.append(PlanTask(
            id=f"T{len(tasks) + 1}",
            title=title,
            objective=objective,
            intent=intent or "find evidence addressing the objective",
            evidence_required=reqs[:3],
            entities=entities[:6],
        ))
        if len(tasks) >= 4:
            break
    if not tasks:
        return None
    return MasterDecomposition(
        rationale="salvaged from the orchestrator's outline (JSON absent)",
        tasks=tasks,
    )


def _field_value(segment: str, key: str) -> List[str]:
    """Values of one field (objective/intent/evidence/entities) in a block."""
    out: List[str] = []
    in_field = False
    for ln in segment.splitlines():
        s = ln.strip()
        if not s:
            continue
        # strip leading bullets AND bold markers ("* **Objective:** x")
        had_bullet = bool(re.match(r"^(?:\*|\-|\u2022)", s))
        body = re.sub(r"^(?:(?:\*|\-)\s*|\*\*\s*)+", "", s)
        m = re.match(r"([a-zA-Z_]+)\s*[:.]\s*(?:\*\*\s*)?", body)
        if m:
            key2 = m.group(1).lower()
            if key2 in ("objective", "task", "description", "goal", "aim",
                        "intent"):
                key2 = "objective" if key2 != "intent" else "intent"
            elif key2.startswith("evidence"):
                key2 = "evidence"
            elif key2.startswith("entit"):
                key2 = "entities"
            else:
                key2 = ""
            if key2 == key:
                in_field = True
                text = _clean_text(body[m.end():])
                if text:
                    out.append(text)
                continue
            if in_field:
                break  # a different field ends this one
        elif in_field:
            text = _clean_text(body)
            if text:
                if had_bullet:
                    out.append(text)
                elif out:
                    out[-1] = out[-1] + " " + text
    return out


# ---------------------------------------------------------------------------
# Canonicalization + plan building
# ---------------------------------------------------------------------------

def _canonicalize_tasks(blueprints: List[PlanTask], fallback_title: str) -> List[PlanTask]:
    """Dedupe near-duplicate tasks, fill every field, enforce max of 4."""
    kept: List[PlanTask] = []
    for i, b in enumerate(blueprints, start=1):
        title = _clean_text(b.title) or f"Task {i}"
        objective = _clean_text(b.objective) or title
        intent = _clean_text(b.intent) or "find evidence addressing the objective"
        reqs: List[str] = []
        for r in b.evidence_required:
            r = _clean_req(r)
            if r and r.casefold() not in {x.casefold() for x in reqs}:
                reqs.append(r)
        if not reqs:
            reqs.append(f"Evidence for: {objective or title}")
        entities: List[str] = []
        for e in b.entities:
            e = _clean_entity(e)
            if e and e.casefold() not in {x.casefold() for x in entities}:
                entities.append(e)
        if not entities:
            entities = _entities_from_text(f"{objective} {title}")
        # near-duplicate guard: same objective tokens as an already-kept task
        dup = False
        for prior in kept:
            if _norm_tokens(prior.objective) and _norm_tokens(objective):
                inter = len(_norm_tokens(prior.objective) & _norm_tokens(objective))
                if inter >= 0.8 * min(len(_norm_tokens(prior.objective)),
                                      len(_norm_tokens(objective))):
                    dup = True
                    break
        if dup:
            continue
        kept.append(PlanTask(
            id=f"T{len(kept) + 1}",
            title=title,
            objective=objective,
            intent=intent,
            evidence_required=reqs[:3],
            entities=entities[:6],
        ))
        if len(kept) >= 4:
            break
    return kept


def _norm_tokens(text: str) -> set:
    return set(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def build_plan(
    query: str,
    blueprints: List[PlanTask],
    *,
    budget: Optional[RunBudget] = None,
    rationale: str = "",
) -> MasterPlan:
    """Deterministic MasterPlan construction from LLM-shaped blueprints.

    Thresholds (target_n) and stop criteria are set HERE - the LLM never gets
    to negotiate its own stopping conditions (spec section 17).
    """
    budget = budget or RunBudget()
    tasks: List[ResearchTask] = []
    for i, b in enumerate(_canonicalize_tasks(blueprints, query), start=1):
        reqs = [
            EvidenceRequirement(id=f"{b.id}.R{j}", text=text,
                                target_n=budget.evidence_target or 3)
            for j, text in enumerate(b.evidence_required, start=1)
        ]
        tasks.append(ResearchTask(
            id=b.id,
            title=b.title,
            objective=b.objective,
            intent=b.intent,
            evidence_requirements=reqs,
            stop_criteria=[
                "required evidence satisfied (N independent papers per "
                "requirement)",
                "retrieval budget exhausted",
            ],
            entities=list(b.entities),
        ))
    if not tasks:
        tasks = [fallback_task(query, budget)]
    return MasterPlan(
        question=query,
        tasks=tasks,
        global_stop_criteria=[
            "all tasks satisfied",
            "retrieval budget exhausted",
            "literature insufficient and/or irreconcilably contradictory",
        ],
        rationale=rationale or "master decomposition",
    )


def fallback_task(query: str, budget: Optional[RunBudget] = None) -> ResearchTask:
    """Deterministic last-resort fallback: one honest task for the question."""
    budget = budget or RunBudget()
    return ResearchTask(
        id="T1",
        title="Evidence for the question",
        objective=query.strip()[:300],
        intent="answer the question as posed from verified evidence",
        evidence_requirements=[
            EvidenceRequirement(id="T1.R1", text=query.strip()[:200],
                                target_n=budget.evidence_target or 3),
        ],
        stop_criteria=[
            "required evidence satisfied (N independent papers per requirement)",
            "retrieval budget exhausted",
        ],
        entities=[],
    )


class MasterOrchestratorAgent:
    """The Master: turns a question into a structured retrieval plan."""

    def __init__(self, config: Any = None, model: Any = None):
        from src.config import AppConfig
        from src.llm import build_model_for
        from pydantic_ai import Agent

        self.config = config or AppConfig()
        self.model = model or build_model_for(self.config, role="orchestrator")
        self.agent = Agent(
            self.model,
            system_prompt=MASTER_SYSTEM_PROMPT,
            name="master_orchestrator",
        )

    async def plan(self, query: str, budget: Optional[RunBudget] = None) -> MasterPlan:
        """Decompose the question with RAW-first, salvage-after strategy."""
        from src.lib.trace import get_trace

        trace = get_trace()
        budget = budget or RunBudget()
        trace.agent("master_orchestrator", output_type="MasterPlan",
                    meta={"budget": budget.model_dump()})
        prompt = f"USER QUESTION:\n{query}\n\nProduce the retrieval plan."
        decomposition = await self._decompose(prompt, label="master_orchestrator")
        plan = build_plan(query, decomposition.tasks, budget=budget,
                          rationale=decomposition.rationale or "")
        trace.bullet(
            "Divided the question into " + _plural(len(plan.tasks), "task") + ": "
            + ", ".join(
                f"{t.id} — “{t.title[:90]}” ({len(t.evidence_requirements)} "
                f"{'evidence requirement' if len(t.evidence_requirements) == 1 else 'evidence requirements'})"
                for t in plan.tasks)
            + ".",
            agent="master",
        )
        return plan

    async def _decompose(self, prompt: str, label: str) -> MasterDecomposition:
        """The full decomposition pipeline (never drops a salvageable plan).

        RAW-FIRST strategy: under pydantic-ai's structured (json_schema)
        request some models answer with a SKELETON (valid JSON whose
        objective/evidence/entities fields are empty), while their raw text
        contains the fully populated plan (often after a thought block).
        So the master reads the RAW text and recovers the plan from it
        (JSON -> prose outline -> JSON-only nudge), and only then falls back
        to the structured call. Every result passes the unusable() guard, so
        a skeleton can never enter the pipeline.
        """
        from src.llm.run import ask_structured

        # 1. raw-text salvage (JSON incl. thought, then prose outline), with
        #    a JSON-only nudge retry when the model only drafted prose.
        for nudge in (None, _JSON_ONLY_NUDGE):
            try:
                raw = await self._raw_run(prompt + (nudge or ""), label)
            except Exception as exc:
                logger.warning("master raw run failed (%s)", exc)
                continue
            parsed = first_json(raw, MasterDecomposition)
            if parsed is not None and not unusable(parsed):
                return parsed
            parsed = parse_prose_plan(raw)
            if parsed is not None and not unusable(parsed):
                logger.info("master: salvaged %d task(s) from the prose outline",
                            len(parsed.tasks))
                return parsed
            # skeleton JSON / label-only outline: unusable -> keep trying

        # 2. structured path as a last resort (endpoints that only give JSON
        #    in structured mode), still guarded against skeletons.
        try:
            parsed = await ask_structured(
                self.agent,
                prompt,
                MasterDecomposition,
                label=label,
                max_tokens=min(1500, getattr(self.config, "agent_max_tokens", 2048)),
            )
            if parsed is not None and not unusable(parsed):
                return parsed
            logger.warning("master structured decomposition came back unusable")
        except Exception as exc:
            logger.warning("master structured decomposition failed (%s)", exc)

        return MasterDecomposition(
            rationale="master output unusable; deterministic fallback",
            tasks=[],
        )

    async def _raw_run(self, prompt: str, label: str) -> str:
        """One unconstrained text call (rate-limited + observed like the rest)."""
        from src.llm.ratelimit import estimate_tokens, get_bucket
        from src.llm.run import _notify_observer, _notify_observer_start

        bucket = get_bucket()
        await bucket.acquire(estimate_tokens(prompt))
        _notify_observer_start(label, prompt)
        try:
            result = await self.agent.run(
                prompt,
                output_type=str,
                model_settings={
                    "max_tokens": min(2000,
                                      getattr(self.config, "agent_max_tokens", 2048)),
                    "temperature": 0.0,
                },
            )
            raw = str(result.output)
        except BaseException:
            _notify_observer(label, prompt, "", None)
            raise
        _notify_observer(label, prompt, raw, None)
        return raw
