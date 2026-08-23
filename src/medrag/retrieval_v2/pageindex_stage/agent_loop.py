"""MedGemma text-tool-protocol agent loop (V2.5)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.pageindex_stage.config import StageConfig
from medrag.retrieval_v2.pageindex_stage.models import NAVIGATION_ERROR, NavigationError, NavigationStep

_TOOL_NAMES = {"get_document_structure", "get_page_content", "get_document", "browse_documents"}
_LP = chr(92) + chr(40)   # escaped literal (
_NAMES_ALT = "(get_document_structure|get_page_content|get_document|browse_documents)"
_NAKED_TOOL_RE = re.compile(_NAMES_ALT + chr(32) + "*" + _LP)


def _extract_json_blocks(text):
    """Brace-scan JSON objects in the model output (no regex backslash games)."""
    blocks = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start:i + 1])
                start = -1
    return blocks

TOOL_DOCS = (
    "get_document_structure(doc_id) - returns the document node tree.",
    "get_page_content(doc_id, pages) - node text content.",
    "get_document(doc_id) - document metadata.",
    "browse_documents() - list registered documents.",
)

_Q = chr(34)


def _format_structure(envelope, limit=7000):
    """Render the PageIndex structure tool output as a compact title tree."""
    import json as _json
    try:
        obj = _json.loads(envelope)
    except Exception:
        return envelope[:limit]
    structure = obj.get("structure") if isinstance(obj, dict) else None
    if not isinstance(structure, list):
        return envelope[:limit]
    lines = []
    def walk(nodes, depth):
        for node in nodes:
            title = node.get("title") or node.get("node_id") or ""
            nid = node.get("node_id") or ""
            lines.append((chr(32) * (depth * 2)) + str(title) + "  [" + str(nid) + "]")
            walk(node.get("nodes") or [], depth + 1)
    walk(structure, 0)
    return chr(10).join(lines)[:limit]

_SYSTEM = (
    "You are a PageIndex document-tree navigation agent inside a medical RAG "
    "system. Navigate the given biomedical document to locate WHERE the required "
    "evidence lives. Call get_document_structure ONE time, read the returned "
    "title tree (sections and their node ids), then STOP tool calls and respond "
    "with the exact node titles you select for the evidence (e.g. the Results "
    "section, recurrent-coarctation subsection, surgical-technique subsection, "
    "and the relevant tables) plus one reason per node. Do not re-call a tool "
    "you already called. Available tools:\n"
    + "\n".join("- " + d for d in TOOL_DOCS)
    + "\n\nPer turn call EXACTLY ONE tool by responding with a JSON block on "
      "its own line, e.g.\n"
    + json.dumps({"tool": "get_document_structure", "args": {"doc_id": "<id>"}})
    + "\n\nOnce enough evidence is located, stop calling tools and respond with "
      "the exact node titles you select and a one-line reason per node. Do NOT "
      "answer the medical question - only report where the evidence lives."
)
def _all_nodes_and_paths(structure, path=None):
    path = list(path or [])
    for node in structure:
        title = node.get("title") or ""
        nid = str(node.get("node_id", ""))
        text = node.get("text") or ""
        crumbs = path + ([title] if title else [])
        yield (nid, title, crumbs, len(crumbs), text[:160])
        yield from _all_nodes_and_paths(node.get("nodes") or [], crumbs)


def extract_from_response(text, structure, max_nodes):
    tl = (text or "").lower()
    seen = {}
    for nid, title, path, level, text_head in _all_nodes_and_paths(structure, []):
        if not title:
            continue
        t = " ".join(title.lower().split())
        if len(t) >= 3 and t in tl and nid not in seen:
            reason = ""
            for sent in re.split(r"[.!?][ \t\r\n]+", text or ""):
                if t in " ".join(sent.lower().split()):
                    reason = sent.strip()[:220]
                    break
            seen[nid] = {
                "node_id": nid, "title": title, "path": list(path), "level": level,
                "reason": reason or "selected by LLM tree navigation",
                "text_head": text_head, "raw": {"mentioned_in_response": True},
            }
    return list(seen.values())[:max_nodes]
_BT = chr(96)


def _chat_content(resp):
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"])
        c = resp.get("content")
        if c:
            return str(c)
        choices = resp.get("choices") or []
        if choices and isinstance(choices[0], dict):
            m = choices[0].get("message")
            if isinstance(m, dict) and m.get("content"):
                return str(m["content"])
    return str(resp)


# strip markdown code fences (built with chr(96) so the source stays quote-clean)
_FENCE_RE = re.compile(_BT * 3 + "(?:json|tool_code)?")


def _parse_tool_invocations(content):
    text = _FENCE_RE.sub("", content)
    text = text.replace(_BT * 3, "")
    calls = []
    # JSON blocks: {"tool": "...", "args": {...}}
    for block in _extract_json_blocks(text):
        try:
            obj = json.loads(block)
        except Exception:
            obj = {}
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("tool_call") or obj.get("name") or ""
        name = str(name)
        if name not in _TOOL_NAMES:
            continue
        args = obj.get("args") or obj.get("arguments") or {}
        calls.append((name, args if isinstance(args, dict) else {}))
    # bare invocations: name(args...)
    if not calls:
        for m in _NAKED_TOOL_RE.finditer(text):
            name = m.group(1)
            args = {}
            calls.append((name, args))
    seen = set()
    out = []
    for name, args in calls:
        key = (name, json.dumps(args, sort_keys=True))
        if key not in seen:
            seen.add(key)
            out.append((name, args))
    return out


_DOC_TOOLS = ("get_document_structure", "get_page_content", "get_document")


def _run_tool(client, name, args, paper_id):
    from pageindex.agent_tools import call_tool
    normalized = {}
    for k, v in args.items():
        key = {"doc_id": "doc_name", "document_id": "doc_name", "doc": "doc_name",
               "doc_name": "doc_name"}.get(k, k)
        if name == "browse_documents":
            continue
        normalized[key] = v
    envelope, is_error = call_tool(client, name, normalized, doc_ids=[paper_id])
    return envelope, is_error
def run_text_tool_loop(paper_id, objective, cfg, client):
    """Drive MedGemma through the PageIndex tree tools turn by turn.

    Returns {final_content, steps, turns, plan}. Raises
    NavigationError(NAVIGATION_ERROR) on transport failures.
    """
    backend = {"base_url": cfg.llm_base_url.rstrip("/"), "api_key": cfg.llm_api_key,
               "timeout": cfg.timeout}
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": objective},
    ]
    steps = []
    plan = []
    final_content = ""
    turns = 0
    last_call_key = ""
    for step_idx in range(1, cfg.max_steps + 1):
        resp = None
        last_exc = None
        for attempt in range(1, 4):
            try:
                resp = client.chat_completions(
                    messages, model=cfg.chat_model or None,
                    backend=backend, temperature=cfg.temperature,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if "524" in str(exc) or "Timeout" in str(exc) or "timed out" in str(exc).lower():
                    import time as _time
                    _time.sleep(8 * attempt)
                    continue
                break
        if resp is None:
            raise NavigationError(paper_id=paper_id, error_type=NAVIGATION_ERROR,
                                  message="text-tool loop turn %d failed: %s" % (step_idx, last_exc)) from last_exc
        content = _chat_content(resp)
        turns += 1
        final_content = content
        plan.append({"step": step_idx, "turns": turns, "raw": content[:1500],
                     "raw_resp": str(resp)[:900]})
        steps.append(NavigationStep(index=step_idx, action="llm-turn",
                                    node_title="", detail=content[:220]))
        calls = _parse_tool_invocations(content)
        if not calls:
            steps.append(NavigationStep(index=step_idx, action="final-answer",
                                        node_title="",
                                        detail="no tool call; treated as final selection"))
            break
        executed_any = False
        for name, args in calls:
            call_key = name + json.dumps(args, sort_keys=True)
            if call_key == last_call_key:
                # degenerate loop guard: same tool call twice -> nudge to finalize
                steps.append(NavigationStep(index=step_idx, action="tool-skip-duplicate",
                                            node_title="", detail="same tool call repeated; nudging to finalize"))
                messages.append({"role": "user",
                                 "content": "You already have the document structure "
                                            "and called " + name + " before. Provide "
                                            "your final selected node titles now, one "
                                            "line each with a short reason. Do not call "
                                            "tools again."})
                continue
            try:
                envelope, is_error = _run_tool(client, name, args, paper_id)
            except Exception as exc:  # noqa: BLE001
                envelope, is_error = "call_tool raised: " + str(exc), True
            executed_any = True
            last_call_key = call_key
            steps.append(NavigationStep(index=step_idx, action="tool: " + name,
                                        node_title="", detail=envelope[:220]))
            tool_out = envelope
            if name == "get_document_structure" and not is_error:
                tool_out = _format_structure(envelope, limit=2600)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user",
                             "content": "Tool " + name + " result:\n" + tool_out[:2400]
                                        + "\n\nIf you have located the evidence "
                                          "regions, stop tool calls and respond "
                                          "with the exact node titles you select."})
            if is_error:
                steps.append(NavigationStep(index=step_idx, action="tool-error",
                                            node_title="", detail=envelope[:220]))
                break
    return {"final_content": final_content, "steps": steps, "turns": turns, "plan": plan}
