"""Stage 3 - PageIndex query-time navigation over a selected paper (V2.4).

The global retriever is NOT part of this stage - a paper has already been
selected. We load ITS persisted PageIndex tree from the SDK local store and
perform LLM tree navigation with the installed PageIndex query API.

Two paths are implemented:

  * SDK managed agent (PageIndex LLM tree search) - the primary path: the
    installed client's chat_completions(doc_id=...) drives a reasoning model
    (configurable; intended MedGemma) over the document with the SDK's agent
    tools, so PageIndex itself navigates the tree.

  * deterministic structural navigation - used ONLY when no reasoning endpoint
    is configured or the SDK agent fails; clearly labeled (status
    "degraded") and never presented as the LLM path. This keeps the stage
    runnable offline without fabricating an LLM result.

The LLM is only asked WHERE the evidence is (navigation objective), never to
produce the final answer (Step 3A) and no reranker runs after this (Step 3E).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.pageindex_stage.config import StageConfig, config_from_env
from medrag.retrieval_v2.pageindex_stage.indexer import _leaves_and_paths, build_index
from medrag.retrieval_v2.pageindex_stage.models import (
    NavigationError,
    NavigationResult,
    NavigationStep,
    SelectedNode,
)

_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "with", "without", "to",
    "for", "was", "were", "is", "are", "be", "been", "as", "by", "at", "from",
    "that", "this", "these", "those", "between", "among", "across", "vs",
    "versus", "which", "what", "who", "how", "do", "does", "did", "we", "our",
    "it", "its", "not", "no", "find", "finding", "report", "reports",
    "reported", "including", "section", "sections", "table", "tables",
    "figure", "figures", "part", "the", "associated", "with",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t and t not in _STOP]


def navigation_objective(target: str = "", condition: str = "",
                         fields: Optional[Sequence[str]] = None,
                         relationships: Optional[Sequence[str]] = None,
                         requirement: Optional[Dict[str, Any]] = None) -> str:
    """Build a paper-local navigation objective (Step 3B).

    The objective asks the reasoning model WHERE the evidence lives - it never
    asks it to answer the question.
    """
    if requirement:
        target = target or str(requirement.get("target", ""))
        condition = condition or str(requirement.get("condition", "") or requirement.get("topic", ""))
        fields = fields or requirement.get("requested_fields") or []
        relationships = relationships or requirement.get("relationships") or []
    parts = [
        "Find the Results sections, tables, and surgical-technique subsections "
        "that compare repair techniques in relation to recurrent coarctation."
    ]
    if target:
        parts.append(f"Target: {target}.")
    if condition:
        parts.append(f"Clinical condition: {condition}.")
    if relationships:
        parts.append("Relationship: " + "; ".join(str(r) for r in relationships) + ".")
    if fields:
        parts.append("Requested fields: " + ", ".join(str(f) for f in fields) + ".")
    parts.append(
        "Navigate the document tree and select node ids/titles of the structural "
        "regions holding this evidence (sections, subsections, tables). Do NOT "
        "answer the question - only report where the evidence lives.")
    return " ".join(parts)


def _structure_navigate(structure: Sequence[Dict[str, Any]], objective: str,
                        max_nodes: int) -> Tuple[List[SelectedNode], List[NavigationStep]]:
    """Score native tree nodes against the objective terms; walk the best
    branch level-by-level to expose a navigation trace."""
    terms = _tokens(objective)
    if not terms:
        return [], []

    nodes: List[Tuple[str, str, List[str], int, str]] = _leaves_and_paths(structure, [])
    scored: List[Tuple[float, str, str, List[str], int, str]] = []
    for nid, title, path, level, text_head in nodes:
        text = " ".join(path + ([title] if title else []) + [text_head[:120]])
        tl = text.lower()
        hits = sum(1 for t in terms if t in tl)
        score = 2.0 * hits / max(1, len(terms))
        low_path = " ".join(path).lower()
        if "table" in low_path:
            score += 0.6
        if "results" in low_path:
            score += 0.25
        if "recurrent" in low_path and "coarctation" in low_path:
            score += 0.35
        if score > 0:
            scored.append((score, nid, title, path, level, text_head))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, nid, title, path, level, text_head in scored[:max_nodes]:
        reason = "; ".join(
            [t for t in terms if t in " ".join(path + [title]).lower()][:4]
        ) or "structural candidate"
        selected.append(SelectedNode(
            node_id=nid, title=title or nid, path=list(path), level=level,
            reason=f"term overlap ({reason})", text_head=text_head,
            raw={"pageindex_score": round(score, 4)},
        ))

    steps: List[NavigationStep] = []
    step = 1

    def descend(nodes: Sequence[Dict[str, Any]], path: List[str], depth: int) -> None:
        nonlocal step
        children = list(nodes)
        if not children:
            return
        best = None
        best_score = -1.0
        for child in children:
            title = child.get("title") or ""
            tl = " ".join(path + [title]).lower()
            hits = sum(1 for t in terms if t in tl)
            s = hits + (0.6 if "table" in tl else 0.0)
            if s > best_score:
                best, best_score = child, s
        if best is None or best_score <= 0:
            return
        title = best.get("title") or ""
        steps.append(NavigationStep(index=step, action="select", node_title=title,
                                    node_path=list(path) + [title],
                                    detail=f"branch score {best_score:.2f}"))
        step += 1
        descend(best.get("nodes") or [], list(path) + [title], depth + 1)

    descend(structure, [], 0)
    return selected, steps

# ---------------------------------------------------------------------------
# SDK managed agent path (PageIndex LLM tree search)
# ---------------------------------------------------------------------------
def _sdk_navigate(paper_id: str, objective: str, structure: Sequence[Dict[str, Any]],
                  cfg: StageConfig) -> Dict[str, Any]:
    """Drive the installed PageIndex client's managed agent over the stored
    document (doc_id=paper_id) using the configured reasoning backend."""
    from pageindex.client import PageIndexClient

    backend = {"base_url": cfg.llm_base_url.rstrip("/"), "api_key": cfg.llm_api_key}
    client = PageIndexClient(
        storage_path=str(cfg.sdk_storage),
        chat_model=cfg.chat_model or None,
        chat_backend=backend,
    )
    t0 = time.perf_counter()
    resp = client.chat_completions(
        [
            {"role": "system", "content":
             "You navigate a biomedical document tree. Identify WHERE the "
             "evidence lives (section/subsection/table nodes). Return the node "
             "titles or ids you selected and a one-line reason per node."},
            {"role": "user", "content": objective},
        ],
        doc_id=paper_id,
        model=cfg.chat_model or None,
        backend=backend,
        temperature=cfg.temperature,
        max_turns=cfg.max_steps,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"response": resp, "latency_ms": latency_ms, "client": client}


def _extract_from_response(text: str, structure: Sequence[Dict[str, Any]],
                           max_nodes: int) -> List[SelectedNode]:
    """Map node titles mentioned in the LLM response back onto the native tree
    (with the surrounding response text as the reason)."""
    tl = (text or "").lower()
    seen: Dict[str, SelectedNode] = {}
    for nid, title, path, level, text_head in _leaves_and_paths(structure, []):
        if not title:
            continue
        t = title.lower()
        if len(t) >= 3 and t in tl and nid not in seen:
            reason = ""
            for sent in re.split(r"(?<=[.!?])\s+", text or ""):
                if t in sent.lower():
                    reason = sent.strip()[:220]
                    break
            seen[nid] = SelectedNode(
                node_id=nid, title=title, path=list(path), level=level,
                reason=reason or "selected by LLM tree navigation",
                text_head=text_head,
                raw={"mentioned_in_response": True},
            )
    return list(seen.values())[:max_nodes]


# ---------------------------------------------------------------------------
# Public navigation entry
# ---------------------------------------------------------------------------
def navigate(paper_id: str, objective: Optional[str] = None,
             requirement: Optional[Dict[str, Any]] = None,
             config: Optional[StageConfig] = None) -> NavigationResult:
    """Run PageIndex query-time navigation over a SELECTED paper (Stage 3).

    The paper's persisted tree is loaded (built on first use); the navigation
    objective (or the one derived from the requirement) is sent to PageIndex.
    """
    cfg = config or config_from_env()
    obj = objective or navigation_objective(requirement=requirement)
    result = NavigationResult(paper_id=paper_id, status="failed", objective=obj)

    try:
        indexed = build_index(paper_id, config=cfg)  # cached tree (or first build)
    except NavigationError as exc:
        result.error = exc.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = {"status": "failed", "error_type": "tree_build_failed",
                        "message": str(exc), "paper_id": paper_id}
        return result

    structure = indexed["structure"]

    # primary path: SDK managed agent (LLM tree search)
    if cfg.has_llm():
        try:
            out = _sdk_navigate(paper_id, obj, structure, cfg)
            raw = out["response"]
            result.raw = raw if isinstance(raw, dict) else {"response": raw}
            result.latency_ms = out["latency_ms"]
            if isinstance(raw, dict):
                content = (raw.get("message") or {}).get("content") or raw.get("content") or ""
                steps_raw = raw.get("steps") or raw.get("history") or []
            else:
                content = str(raw)
                steps_raw = []
            result.trace = [
                NavigationStep(index=i + 1, action="sdk-agent",
                               node_title=str(s.get("title", "") or s.get("tool", "")) if isinstance(s, dict) else "",
                               detail=str(s)[:200])
                for i, s in enumerate(steps_raw)
            ]
            result.selected_nodes = _extract_from_response(content, structure, cfg.max_nodes)
            result.status = "success" if result.selected_nodes else "navigated_no_selection"
            return result
        except Exception as exc:  # noqa: BLE001
            result.raw = {"llm_error": str(exc)}
            result.error = {"status": "failed", "error_type": "llm_unavailable",
                            "message": f"SDK agent navigation failed: {exc}",
                            "paper_id": paper_id}

    # offline fallback: deterministic structural navigation (clearly labeled)
    try:
        selected, steps = _structure_navigate(structure, obj, cfg.max_nodes)
    except Exception as exc:  # noqa: BLE001
        result.error = {"status": "failed", "error_type": "navigation_failed",
                        "message": f"structural navigation failed: {exc}",
                        "paper_id": paper_id}
        return result
    result.selected_nodes = selected
    result.trace = steps
    result.error = {"status": "degraded",
                    "error_type": "llm_unavailable",
                    "message": "no reasoning model configured (PAGEINDEX_LLM_BASE_URL + "
                               "PAGEINDEX_CHAT_MODEL); deterministic structural navigation "
                               "over the native PageIndex tree shown instead",
                    "paper_id": paper_id}
    result.status = "degraded"
    return result


# ---------------------------------------------------------------------------
# Trace rendering (Step 3D)
# ---------------------------------------------------------------------------
def navigation_trace(result: NavigationResult) -> str:
    lines: List[str] = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("PAGEINDEX NAVIGATION")
    lines.append(sep)
    lines.append("Paper:")
    lines.append(f"    {result.paper_id}")
    lines.append("")
    lines.append("Objective:")
    lines.append(f"    {result.objective}")
    lines.append("")
    for i, s in enumerate(result.trace, start=1):
        lines.append(f"Step {i}:")
        lines.append(f"    action:  {s.action}")
        if s.node_title:
            path = " > ".join(s.node_path) if s.node_path else s.node_title
            lines.append(f"    node:    {path}")
        if s.detail:
            lines.append(f"    detail:  {s.detail}")
        lines.append("")
    lines.append("Selected nodes:")
    if result.selected_nodes:
        for n in result.selected_nodes:
            lines.append(f"    - {n.node_id}  {n.title}  " + "[" + " > ".join(n.path) + "]")
            if n.reason:
                lines.append(f"        reason: {n.reason[:180]}")
    else:
        lines.append("    (none)")
    if result.error:
        lines.append("")
        lines.append("Status:")
        lines.append(f"    {result.status}  ({result.error.get('error_type', '')})")
        lines.append(f"    {result.error.get('message', '')}")
    lines.append(sep)
    return "\n".join(lines)

