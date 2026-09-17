"""LLM-as-a-judge verification middleware.

Sits between the deep agent's evidence tools and the agent itself: tool
input in, tool runs, output verified against the caller's evidence
requirements, tool output plus verdicts returned to the agent.
Non-evidence tools pass through, as does anything with no judgeable output.

The judge never breaks a tool call: any verifier failure degrades to the
raw tool output plus a short [EVIDENCE JUDGMENT FAILED] note.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any
from contextlib import nullcontext
from contextvars import ContextVar

from pydantic_ai.capabilities import AbstractCapability

from src.lib import narrate
from src.lib.trace import get_trace
from src.tools.verifier import verify_passages

EVIDENCE_TOOLS = ("local_search", "web_search")

# Local-corpus tools return exact source passages already cited by id, so they
# skip the verbatim-excerpt contract entirely (see `verify_passages`).
LOCAL_SEARCH_TOOLS = frozenset({"local_search"})

# The judge lane gets ONE batched verifier call per tool invocation: all
# passages are scored in a single prompt, and the judge returns bounded
# VERBATIM excerpts (deterministically checked) instead of megabyte bodies
# — so one call stays within the judge window without truncation.
# Evidence text returned to the AGENT is clipped here. The judge scores the
# full (3000-char-clipped) passages and the ledger stores them; the agent only
# needs the ids, verdicts, and readable excerpts to reason and cite — a
# megabyte blob sent up the agent lane blows qwen\u2019s input window (400).
AGENT_TEXT_CAP = 4000
AGENT_JSON_CAP = 400_000

# Keys whose string values are page/passage bodies — the only fields clipped.
_TEXT_KEYS = frozenset({"text", "markdown", "snippet", "content"})

# Progress channel: fired the moment an evidence tool's raw output is in
# (passages found), before the judge runs — so the UI can show the
# passages table while verdicts are still pending. The streaming adapter
# installs the sink per run; with none installed this is a no-op.
# Signature: (tool_call_id, tool_name, raw_output).
_ProgressSink = Callable[[str, str, str], None]
_progress_sink: ContextVar[_ProgressSink | None] = ContextVar(
    "medrag_retrieval_progress", default=None)


def install_retrieval_progress_sink(sink: _ProgressSink) -> object:
    """Install the per-run passages-found sink; pass the token to reset."""
    return _progress_sink.set(sink)


# Thinking channel retained for the UI thought stream. The TypeSafe verifier
# returns decisions rather than a token stream, so nothing emits into it
# today; the installer stays so the streaming adapter wiring is unchanged.
# Signature: (tool_call_id, thinking_delta).
_ThinkingSink = Callable[[str, str], None]
_thinking_sink: ContextVar[_ThinkingSink | None] = ContextVar(
    "medrag_judge_thinking", default=None)


def install_judge_thinking_sink(sink: _ThinkingSink) -> object:
    """Install the per-run judge-thinking sink; pass the token to reset."""
    return _thinking_sink.set(sink)




def _clip_agent_texts(obj: Any, cap: int = AGENT_TEXT_CAP) -> Any:
    """Clip long text fields in a nested JSON payload (structural, valid)."""
    if isinstance(obj, dict):
        return {
            key: (_clip_agent_text(value, cap) if key in _TEXT_KEYS
                  else _clip_agent_texts(value, cap))
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_clip_agent_texts(value, cap) for value in obj]
    return obj


def _clip_agent_text(text: str, cap: int) -> str:
    if isinstance(text, str) and len(text) > cap:
        return text[:cap] + " … [clipped for agent — full text in ledger]"
    return text


def _agent_evidence_view(output: str, report, ask_verbatim: bool = True) -> str:
    """Build the bounded evidence block returned to the agent.

    Instead of shipping the raw tool output (megabyte passages) or an
    arbitrary character clip, rebuild the original JSON shape with each
    passage/page text replaced by its judge-verified VERBATIM excerpt
    (the relevant sentences, deterministically checked against the source).
    Unjudged/irrelevant items keep their identity but carry no long text, so
    the agent sees the evidence it can actually quote, bounded by relevance
    rather than by a truncation cutoff. Falls back to the structural
    compactor when the tool output is unparseable.
    """
    try:
        data = json.loads(output) if isinstance(output, str) else output
    except (ValueError, TypeError):
        return compact_evidence_output(output)
    by_pid = {str(v.passage_id): v for v in (report.evidence_results or [])}

    def _replace(item):
        if not isinstance(item, dict):
            return item
        pid = str(item.get("id") or item.get("passage_id") or item.get("chunk_id")
                  or item.get("url") or "")
        verdict = by_pid.get(pid)
        excerpt = (verdict.verbatim or "").strip() if verdict is not None else ""
        out = dict(item)
        if excerpt:
            # The verbatim excerpt IS the evidence; drop the megabyte bodies.
            for key in ("text", "markdown", "snippet", "content"):
                out.pop(key, None)
            out["verbatim"] = excerpt
        elif verdict is not None and (verdict.coverage or []):
            if ask_verbatim:
                # A relevant page whose excerpt failed the verbatim check:
                # never ship the megabyte body, mark it visibly instead.
                for key in ("text", "markdown", "snippet", "content"):
                    out.pop(key, None)
                out["verbatim"] = " (excerpt failed verbatim check)"
            else:
                # Local passages carry no verbatim contract — keep a clipped
                # body so the agent can still quote and cite it.
                for key in ("text", "content"):
                    if isinstance(out.get(key), str):
                        out[key] = _clip_agent_text(out[key], AGENT_TEXT_CAP)
        return out

    if isinstance(data, dict):
        if isinstance(data.get("local"), list) or isinstance(data.get("web"), dict):
            local = [ _replace(i) for i in (data.get("local") or []) ]
            web = dict(data.get("web") or {})
            if isinstance(web.get("pages"), list):
                web["pages"] = [_replace(p) for p in web["pages"]]
            if isinstance(web.get("results"), list):
                web["results"] = [_replace(r) for r in web["results"]]
            data = {**data, "local": local, "web": web}
        elif isinstance(data.get("pages"), list):
            data = {**data, "pages": [_replace(p) for p in data["pages"]]}
        elif isinstance(data.get("results"), list):
            data = {**data, "results": [_replace(r) for r in data["results"]]}
    elif isinstance(data, list):
        data = [_replace(i) for i in data]
    return json.dumps(data, ensure_ascii=False)

def compact_evidence_output(output: str) -> str:
    """Return a bounded tool result for the agent lane: every passage/page
    text clipped, ids/urls/verdicts intact, JSON always valid. Falls back to
    a hard character cap on unparseable output.
    """
    try:
        data = json.loads(output) if isinstance(output, str) else output
        compact = json.dumps(_clip_agent_texts(data), ensure_ascii=False)
        if len(compact) > AGENT_JSON_CAP:
            return compact[:AGENT_JSON_CAP] + " … [clipped for agent]"
        return compact
    except (ValueError, TypeError):
        # Truncated mid-values (the live 400 case) cannot parse whole. Try
        # salvaging a valid prefix so the agent still gets structured data;
        # a final hard fallback emits a well-formed JSON error envelope
        # (never a raw mid-JSON slice that would break both the model and
        # the UI).
        salvaged = _salvage_json_prefix(output
                                        if isinstance(output, str) else "")
        if salvaged is not None:
            return salvaged
        return json.dumps({"clipped": True,
                           "note": "tool output truncated past parseability",
                           "head": str(output)[:2000]})


def _salvage_json_prefix(text: str) -> str | None:
    """Best-effort recovery of a parseable JSON prefix from truncated text.

    Walks back from the end dropping bytes until ``json.loads`` succeeds
    (JSON must close all braces/strings, so a real prefix survives) then
    runs the structural clipper so the result is bounded AND valid. Returns
    None when nothing salvageable remains.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = text
    # Upper bound on retries: a valid prefix of a megabyte blob is found in
    # a handful of steps; this caps pathological inputs.
    for _ in range(16):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            cut = candidate.rstrip()
            cut = cut[: cut.rfind(",") if "," in cut else len(cut)]
            if not cut or len(cut) >= len(candidate):
                return None
            candidate = cut
            continue
        compact = json.dumps(_clip_agent_texts(data), ensure_ascii=False)
        return compact[:AGENT_JSON_CAP] if len(compact) > AGENT_JSON_CAP else compact
    return None


def _sequential() -> bool:
    """Debug mode: serialize judge runs so console output reads top-to-bottom."""
    return os.environ.get("SEQUENTIAL", "").strip().lower() in ("1", "true", "yes")


_JUDGE_LOCK: asyncio.Lock | None = None


def _judge_lock() -> asyncio.Lock:
    global _JUDGE_LOCK
    if _JUDGE_LOCK is None:
        _JUDGE_LOCK = asyncio.Lock()
    return _JUDGE_LOCK


class LLMAsJudge(AbstractCapability):
    """Verify evidence-tool outputs before they reach the agent."""

    def __init__(self, tools=EVIDENCE_TOOLS, judge_client=None) -> None:
        self._tools = tuple(tools)
        self._judge = judge_client

    async def on_tool_execute_error(self, ctx, *, call, tool_def, args, error):
        # An evidence-tool failure must not kill the run: hand the agent an
        # explicit error message so it can retry or try another tool.
        if call.tool_name not in self._tools:
            raise error
        msg = f"[TOOL FAILED: {call.tool_name}: {error}]"
        narrate.say(f"  [judge:{call.tool_name}] {msg}")
        get_trace().log("tool_failed", tool=call.tool_name, error=str(error))
        return msg

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        output = await handler(args)
        if call.tool_name not in self._tools:
            return output
        ask_verbatim = call.tool_name not in LOCAL_SEARCH_TOOLS
        _report_progress(call, output)
        question = args.get("query", "") if isinstance(args, dict) else ""
        requirements = _requirements_from_args(args)
        if not str(question).strip() or not requirements:
            # Fetch-style tools carry urls, not the question/requirements, and
            # an agent sometimes forgets the pre-planned requirements. Fall
            # back to the task's ledger so judged passages attribute to the
            # requirements they were actually scored against.
            ledger_q, ledger_reqs = _ledger_context(ctx)
            if not str(question).strip():
                question = ledger_q
            requirements = requirements or ledger_reqs
        items = _evidence_from_output(call.tool_name, output)
        if not str(question).strip() or not items:
            return output
        if not requirements:
            # Nothing planned and the agent passed none: judge generically.
            requirements = [{"id": "GENERAL", "description": str(question).strip()}]
        try:
            # The System One client is created inside `verify_passages`; an
            # injected client is used as-is. SEQUENTIAL=1 holds one
            # process-wide lock across the whole judge pass so concurrent
            # tool calls cannot interleave.
            gate = _judge_lock() if _sequential() else nullcontext()
            async with gate:
                report = await verify_passages(
                    question, requirements, items,
                    client=self._judge,
                    label=f"judge:{call.tool_name}",
                    ask_verbatim=ask_verbatim,
                )
        except Exception as exc:  # noqa: BLE001 — the judge must never break tools
            return (f"{compact_evidence_output(output)}\n\n"
                    f"[EVIDENCE JUDGMENT FAILED: {exc}]")
        if not report.judged:
            # verify_passages degrades failures to an empty verdict; surface
            # that here so a down judge is visible, not silent. An empty
            # result with `judged` set means every passage was rejected —
            # a real outcome the UI must show, not a failure.
            get_trace().log("evidence_judgment_failed", tool=call.tool_name)
            return (f"{compact_evidence_output(output)}\n\n"
                    f"[EVIDENCE JUDGMENT FAILED: judge returned no verdicts]")
        get_trace().log(
            "evidence_judgment",
            tool=call.tool_name,
            kept=[v.passage_id for v in report.evidence_results],
            verdicts={v.passage_id: {"intent": round(v.intent_score, 2),
                                     "coverage": v.coverage}
                      for v in report.evidence_results},
        )
        _accumulate_kept(ctx, items,
                         [v.passage_id for v in report.evidence_results])
        # Stateful ledger: valid (judge-kept) evidence lands in the run
        # JSON as the tool call returns it — the orchestrator later steers
        # from the stored counts, not by re-reading this output.
        try:
            from src.runstate.store import record_evidence as _record_evidence

            await _record_evidence(ctx, call.tool_name, items,
                                   list(report.evidence_results))
        except Exception:  # noqa: BLE001 — recording never breaks tools
            pass
        return (f"{_agent_evidence_view(output, report, ask_verbatim)}\n\n"
                f"[EVIDENCE JUDGMENT]\n{report.model_dump_json()}")


def _ledger_context(ctx) -> tuple[str, list[dict]]:
    """(task question, planned requirements) from the run ledger.

    Lets fetch-style evidence tools that carry no query/requirements still be
    judged against the task's contract; ("", []) on any failure, so callers
    fall through to a generic judgment or pass the output through unjudged.
    """
    try:
        deps = getattr(ctx, "deps", None)
        chat_id = getattr(deps, "chat_id", None)
        if not chat_id:
            return "", []
        from src.runstate.store import get_store

        chat = get_store().get_chat(chat_id)
        turn = chat.current if chat is not None else None
        if turn is None:
            return "", []
        task = turn.task(getattr(deps, "task_id", None) or "")
        if task is None:
            return (turn.question or "").strip(), [
                {"id": r.id, "description": r.description}
                for r in (turn.run_requirements or [])]
        return (task.task or "").strip(), [
            {"id": r.id, "description": r.description}
            for r in task.evidence_requirements]
    except Exception:  # noqa: BLE001 — fallback must never break tools
        return "", []


# Hard bound so pathological runs can't grow the shared collection
# without limit (the gap checker only ever looks at the newest 25).
MAX_KEPT_PASSAGES = 200

_PASSAGE_ID_KEYS = ("id", "passage_id", "chunk_id", "citation_id", "url")


def _accumulate_kept(ctx, items: list, kept_ids: list[str]) -> None:
    """Stash judge-kept passage dicts on the run deps for the gap checker.

    Best-effort and silent: no deps (or no kept list) means nothing to do.
    """
    kept = getattr(getattr(ctx, "deps", None), "kept_passages", None)
    if not isinstance(kept, list) or not kept_ids:
        return
    by_id: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in _PASSAGE_ID_KEYS:
            value = item.get(key)
            if value:
                by_id.setdefault(str(value), item)
    have = set()
    for existing in kept:
        if isinstance(existing, dict):
            for key in _PASSAGE_ID_KEYS:
                value = existing.get(key)
                if value:
                    have.add(str(value))
    for pid in kept_ids:
        item = by_id.get(str(pid))
        if item is not None and str(pid) not in have:
            kept.append(item)
            have.add(str(pid))
    del kept[:-MAX_KEPT_PASSAGES]


def _report_progress(call, output) -> None:
    """Fire the passages-found progress sink (best-effort, never breaks tools)."""
    sink = _progress_sink.get()
    if sink is None:
        return
    try:
        text = output if isinstance(output, str) else json.dumps(output)
        sink(getattr(call, "tool_call_id", None) or "", call.tool_name, text)
    except Exception:  # noqa: BLE001 — progress must never break a tool call
        pass


def _requirements_from_args(args) -> list:
    # The agent passes what it is looking for alongside the query:
    # [{"id": "E1", "description": "..."}] (bare strings tolerated).
    # Anything ragged degrades to "no requirements" rather than failing.
    if not isinstance(args, dict):
        return []
    raw = args.get("evidence_requirements", [])
    if raw is None:
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    expanded = []
    for r in raw:
        if isinstance(r, str):
            # Agents often pass the whole list JSON-encoded in one string.
            try:
                parsed = json.loads(r)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                expanded.append(parsed)
                continue
            if isinstance(parsed, list):
                expanded.extend(x for x in parsed if isinstance(x, (str, dict)))
                continue
        if isinstance(r, (str, dict)):
            expanded.append(r)
    return expanded


def _evidence_from_output(tool_name: str, output) -> list[dict]:
    # Normalize a tool's JSON output into judge-ready evidence items.
    # The web-search tool returns listings under "results" plus scraped
    # page markdown under "pages" — only the pages carry quotable text,
    # so they are what the judge scores (same as document retrieval).
    try:
        data = json.loads(output) if isinstance(output, str) else output
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        if isinstance(data.get("local"), list) or isinstance(data.get("web"), dict):
            # Parallel fan-out shape: {"local": [...], "web": {...}}.
            # Judge the union of local passages and scraped web pages
            # (falling back to web snippets when no scrape survived),
            # so one combined call is scored exactly like the two
            # separate tools would be.
            local = data.get("local")
            local = local if isinstance(local, list) else []
            web = data.get("web")
            web = web if isinstance(web, dict) else {}
            pages = web.get("pages")
            if isinstance(pages, list) and any(isinstance(p, dict) for p in pages):
                usable = [p for p in pages
                          if isinstance(p, dict) and p.get("success") is not False]
                web_items = usable if usable else web.get("results", [])
            else:
                web_items = web.get("results", [])
            web_items = web_items if isinstance(web_items, list) else []
            data = list(local) + list(web_items)
        if isinstance(data, dict):
            pages = data.get("pages")
            if isinstance(pages, list) and any(isinstance(p, dict) for p in pages):
                # Explicit scrape failures carry no quotable text — judging them
                # only yields boilerplate rejects. Worse, a pages list of pure
                # failures would shadow the search-snippet fallback below, so
                # drop them and fall back to the snippets when nothing usable
                # survived the scrape.
                usable = [p for p in pages
                          if isinstance(p, dict) and p.get("success") is not False]
                data = usable if usable else data.get("results", [])
            else:
                data = data.get("results", [])
    if not isinstance(data, list):
        return []
    items = []
    for i, entry in enumerate(data):
        if isinstance(entry, dict):
            items.append(entry)
        elif isinstance(entry, str):
            items.append({"citation_id": f"{tool_name}-{i}", "text": entry})
    return items
