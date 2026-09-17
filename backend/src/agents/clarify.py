"""Human-in-the-loop clarification: the ask_user tool.

The master orchestrator AND every deep sub-agent carry ask_user. Calling
it parks the run (no sub-agent is ever spawned for clarification - the
calling agent handles it natively):

1. The questions are normalized, persisted to the chat ledger, and
   surfaced to the UI as a "question" stream event.
2. The tool awaits the user's answer; the run resumes when the UI posts
   to POST /v1/chats/{chat}/questions/{qid}/answer (resolved here).
3. The tool returns a compact summary the agent reasons over.

Every question always carries TWO implicit UI options in addition to the
model-generated ones:

* "Find all you can find" - the user wants exhaustive coverage; the agent
  must not narrow further, it should broaden the research scope.
* "Other" - a free-text box for anything the options missed.

Options are multi-select (checkboxes) unless the model marks
multi_select=false for a mutually exclusive either/or choice (radio).

An unanswered question can never hang a run: the tool waits up to
HITL_TIMEOUT_S (default 600 s) and on timeout tells the agent to proceed
with its best interpretation and to state its assumptions as gaps.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import RunContext

from src.tools.umls import DeepDeps

_FALLBACK_TIMEOUT = float(os.environ.get("HITL_TIMEOUT_S", "600"))


class ClarifyQuestion(BaseModel):
    """One question the agent asks the user.

    id: short stable id (q1, q2, ...) echoed on the wire and in answers.
    text: the question, in user-facing language.
    options: concrete answer choices the agent generated (the UI appends
        "Find all you can find" and "Other" automatically).
    multi_select: True = checkboxes (pick several); False = radio
        buttons, ONLY for mutually exclusive either/or choices.
    allow_other: whether the user may type a free-text answer.
    allow_find_all: whether the "Find all you can find" escape hatch is
        offered (always True in practice).
    context: where the question came from (e.g. the sub-agent task), for
        the UI and the ledger.
    """

    id: str = ""
    text: str = ""
    options: list[str] = Field(default_factory=list)
    multi_select: bool = True
    allow_other: bool = True
    allow_find_all: bool = True
    context: str = ""

    def to_event(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "options": self.options,
            "multi_select": self.multi_select,
            "allow_other": self.allow_other,
            "allow_find_all": self.allow_find_all,
            "context": self.context,
        }


def normalize_questions(raw: Any) -> list[ClarifyQuestion]:
    """Leniently coerce a tool-call payload into ClarifyQuestion objects.

    Accepts a list of dicts, a JSON-encoded list, a dict with a
    "questions" key, or already-built ClarifyQuestion instances. Missing
    ids are assigned q1, q2, ... in order so the wire and the ledger can
    correlate answers without trusting the model to be careful.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw)
        except ValueError:
            return []
    if isinstance(raw, dict):
        inner = raw.get("questions", raw.get("question", None))
        raw = inner if inner is not None else []
    if not isinstance(raw, list):
        return []
    out: list[ClarifyQuestion] = []
    for i, item in enumerate(raw):
        if isinstance(item, ClarifyQuestion):
            q = item
        elif isinstance(item, dict):
            try:
                q = ClarifyQuestion(**{
                    k: v for k, v in item.items()
                    if k in ClarifyQuestion.model_fields
                })
            except ValidationError:
                # Ragged option/multi_select types are the model's fault, not
                # the user's: keep the question, drop the fields that won't
                # parse. normalize_questions must never raise on tool payload.
                q = ClarifyQuestion(id=str(item.get("id") or ""),
                                    text=str(item.get("text") or ""))
        else:
            continue
        if not q.id:
            q.id = f"q{i + 1}"
        if not q.text.strip():
            continue
        q.options = [str(o).strip() for o in q.options if str(o).strip()]
        q.context = str(q.context or "").strip()
        out.append(q)
    return out


def validate_answer(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce + validate an answer payload from the UI.

    At least one of selections / other / find_all must be present.
    """
    selections = payload.get("selections") if isinstance(payload.get("selections"), list) else []
    selections = [str(s).strip() for s in selections if str(s).strip()]
    other = str(payload.get("other") or "").strip()
    find_all = bool(payload.get("find_all"))
    if not selections and not other and not find_all:
        return {}
    return {"selections": selections, "other": other, "find_all": find_all}


# --- broker: (chat_id, question_id) -> Future -------------------------------
# The tool registers a future and awaits it; the API endpoint resolves it
# when the user answers. Futures live only while a run is parked on them.
# A plain threading lock guards the dict: both the parked run and the
# resolution endpoint share one process/loop, so no cross-loop handoff is
# needed and submit_answer stays synchronous (truthful return value).
_broker: dict[tuple[str, str], "asyncio.Future[dict[str, Any]]"] = {}
_broker_lock = threading.Lock()


def _register(chat_id: str, question: ClarifyQuestion) -> "asyncio.Future[dict[str, Any]]":
    """Create (or supersede) the broker future for one parked question."""
    import asyncio

    with _broker_lock:
        key = (chat_id, question.id)
        prev = _broker.pop(key, None)
        if prev is not None and not prev.done():
            # A later ask with the same id supersedes the earlier wait;
            # deliver a benign skip so it never hangs.
            prev.set_result({"selections": [], "other": "", "find_all": False,
                             "superseded": True})
        fut: "asyncio.Future[dict[str, Any]]" = asyncio.get_running_loop().create_future()
        _broker[key] = fut
        return fut


def has_pending(chat_id: str, question_id: str) -> bool:
    """True when a run is parked on this question right now."""
    with _broker_lock:
        fut = _broker.get((chat_id, question_id))
    return fut is not None and not fut.done()


def submit_answer(chat_id: str, question_id: str, payload: dict[str, Any]) -> bool:
    """Resolve a waiting ask_user call (called by the API endpoint).

    Persists the answer to the chat ledger (so GET /v1/chats shows it
    even if the run already moved on), then releases the parked tool.
    Returns True only when a parked question was actually resolved.
    """
    import asyncio

    answer = validate_answer(payload)
    if not answer:
        return False
    with _broker_lock:
        fut = _broker.pop((chat_id, question_id), None)
    if fut is None or fut.done():
        return False
    fut.set_result(answer)
    # Ledger write is best-effort and must not hold up the run.
    try:
        from src.runstate.store import get_store

        store = get_store()
    except Exception:  # noqa: BLE001 - resolution must never throw at the API
        store = None
    if store is not None:
        try:
            asyncio.create_task(
                store.record_question_answer(chat_id, question_id, answer))
        except Exception:  # noqa: BLE001 - best-effort persistence
            pass
    return True


async def _wait_answers(
    chat_id: str,
    questions: list[ClarifyQuestion],
    timeout: float,
) -> dict[str, dict[str, Any]]:
    """Register every question and wait for the answers.

    Answers are folded in as they arrive and the whole call is bounded by
    ONE shared deadline. Waiting question-by-question (the old shape) meant
    an unanswered q1 held the run to the full deadline and then discarded
    the answers the user did submit for q2..qN.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    futures: dict[str, "asyncio.Future[dict[str, Any]]"] = {}
    try:
        for q in questions:
            futures[q.id] = _register(chat_id, q)
        answers: dict[str, dict[str, Any]] = {}
        pending = dict(futures)
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            done, _ = await asyncio.wait(
                set(pending.values()), timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED)
            if not done:
                break
            for qid, fut in list(pending.items()):
                if fut not in done:
                    continue
                pending.pop(qid)
                try:
                    answers[qid] = fut.result()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — a failed ask is a no-answer
                    continue
        return answers
    finally:
        for qid, fut in futures.items():
            if not fut.done():
                fut.cancel()
            with _broker_lock:  # unanswered questions must not leak brokers
                if _broker.get((chat_id, qid)) is fut:
                    _broker.pop((chat_id, qid), None)


def _format_answers(questions: list[ClarifyQuestion],
                    answers: dict[str, dict[str, Any]]) -> str:
    lines = ["<user-clarification-answers>"]
    got_any = False
    for q in questions:
        ans = answers.get(q.id)
        if ans is None or ans.get("superseded"):
            continue
        got_any = True
        bits: list[str] = []
        if ans.get("find_all"):
            bits.append("Find all you can find (do NOT narrow: research the full topic exhaustively)")
        if ans.get("selections"):
            bits.append("selected: " + "; ".join(ans["selections"]))
        if ans.get("other"):
            bits.append("other: " + str(ans["other"]))
        lines.append(f"[{q.id}] " + (", ".join(bits) if bits else "(no answer)")
                     + f" - question was: {q.text}")
    if not got_any:
        lines.append("(no user answers delivered within the wait budget - proceed with your"
                     " best interpretation and clearly list assumptions as gaps in the final answer)")
    lines.append("</user-clarification-answers>")
    return "\n".join(lines)


async def ask_user(
    ctx: RunContext[DeepDeps],
    questions: list[dict[str, Any]] | str | None = None,
) -> str:
    """Ask the USER for clarification (human-in-the-loop).

    Call ONLY when the question or the evidence needed is genuinely
    ambiguous - never to offload work the agent should do itself, and
    never as a replacement for normal research. This tool parks the run,
    shows your question(s) to the user, and returns their answers.

    CLEAR QUESTION CONTRACT (STRICT):
    * questions: a list of objects:
        { id: "q1",
          text: "The question you need answered",
          options: ["concrete option 1", "..."],
          multi_select: true,   // true = checkboxes; false = radio - ONLY
                                // for mutually exclusive either/or choices
          allow_other: true,
          allow_find_all: true }
    * Options MUST be concrete and distinguishable, 3-7 of them. A question
      without options (e.g. "What would you like to know?") is FORBIDDEN -
      if you cannot think of concrete alternatives, you are not ready to
      ask: draft the alternatives first. This tool is the ONLY way your
      clarification reaches the user as a form; never clarify in plain text.
    * The UI ALWAYS appends "Find all you can find" (user wants exhaustive
      coverage - then do NOT narrow; broaden) and "Other" (free text).
    * Ask ONE well-formed question per call unless several are truly
      independent - then batch them in one call (same turn, fewer pauses).
    * Do not ask questions whose answers the research itself will reveal.
    * If the user picks "Find all you can find", stop clarifying and
      proceed with exhaustive coverage. If they type "Other", use that as
      the authoritative instruction.
    * Timeout: if no answer arrives, this returns a notice to proceed
      with your best interpretation and state assumptions as gaps.
    """
    chat_id = str(getattr(ctx.deps, "chat_id", "") or "") or ""
    qs = normalize_questions(questions)
    if not qs or not chat_id:
        return ("[ask_user unavailable: no active chat] Proceed with your best "
                "interpretation and clearly list assumptions as gaps.")
    from src.runstate.store import get_store

    store = get_store()
    for q in qs:
        try:
            from src.runstate.models import QuestionRecord

            await store.record_question(chat_id, QuestionRecord(
                id=q.id, text=q.text, options=q.options,
                multi_select=q.multi_select, allow_other=q.allow_other,
                allow_find_all=q.allow_find_all, context=q.context,
                by_task=str(getattr(ctx.deps, "task_id", "") or "")))
        except Exception:  # noqa: BLE001 - ledger write never blocks the ask
            pass
    answers = await _wait_answers(chat_id, qs, _FALLBACK_TIMEOUT)
    return _format_answers(qs, answers)


__all__ = ["ClarifyQuestion", "ask_user", "normalize_questions",
           "submit_answer", "validate_answer", "has_pending"]

