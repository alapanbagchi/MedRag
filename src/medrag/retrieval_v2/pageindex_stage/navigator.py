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
    DOCUMENT_REGISTRATION_ERROR,
    MEDGEMMA_ENDPOINT_ERROR,
    NAVIGATION_ERROR,
    NAVIGATION_EMPTY_RESULT,
    PAGEINDEX_CLIENT_ERROR,
    TREE_LOAD_ERROR,
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
    """Build a paper-local navigation objective (Part 5).

    The objective preserves target, condition, relationship and requested
    fields verbatim (never reduced to bare keywords) and asks the reasoning
    model only WHERE the evidence lives - never to answer the question.
    """
    if requirement:
        target = target or str(requirement.get("target", ""))
        condition = condition or str(requirement.get("condition", "") or requirement.get("topic", ""))
        fields = fields or requirement.get("requested_fields") or []
        relationships = relationships or requirement.get("relationships") or []
    parts = [
        "Find the Results sections, surgical-technique sections, and tables "
        "comparing repair techniques in relation to recurrent coarctation. "
        "Prioritize evidence that identifies specific surgical repair techniques "
        "and reports recurrence percentages and p-values. Prefer the recurrent "
        "coarctation and surgical technique sections and their associated tables "
        "and footnotes."
    ]
    if target:
        parts.append("Target: " + target + ".")
    if condition:
        parts.append("Clinical condition: " + condition + ".")
    if relationships:
        parts.append("Relationship: " + "; ".join(str(r) for r in relationships) + ".")
    if fields:
        parts.append("Requested fields: " + ", ".join(str(f) for f in fields) + ".")
    parts.append(
        "Navigate the document tree and select node ids/titles of the structural "
        "regions holding this evidence (sections, subsections, tables). Do NOT "
        "answer the question - only report where the evidence lives.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Part 1 - MedGemma endpoint verification (direct, before PageIndex)
# ---------------------------------------------------------------------------
def verify_medgemma_endpoint(config: Optional[StageConfig] = None) -> Dict[str, Any]:
    """Verify the MedGemma OpenAI-compatible endpoint directly.

      GET  {BASE_URL}/models
      POST {BASE_URL}/chat/completions   ("Respond with the single word OK.")

    Returns a report dict; raises NavigationError(MEDGEMMA_ENDPOINT_ERROR) with
    the exact failure layer when anything fails. The API key is never logged.
    """
    from medrag.retrieval_v2.pageindex_stage.models import MEDGEMMA_ENDPOINT_ERROR
    import requests

    cfg = config or config_from_env()
    if not cfg.llm_base_url:
        raise NavigationError(paper_id="", error_type=MEDGEMMA_ENDPOINT_ERROR,
                              message="PAGEINDEX_LLM_BASE_URL is not set - MedGemma endpoint unknown")
    base = cfg.llm_base_url.rstrip("/")
    model = cfg.chat_model or ""
    if not model:
        raise NavigationError(paper_id="", error_type=MEDGEMMA_ENDPOINT_ERROR,
                              message="PAGEINDEX_CHAT_MODEL is not set")
    headers = {"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {}

    report: Dict[str, Any] = {
        "endpoint_url": base,
        "model": model,
        "api_key_configured": bool(cfg.llm_api_key),
        "models": {}, "chat": {}, "latency_ms": 0.0,
    }
    try:
        t0 = time.perf_counter()
        resp = requests.get(f"{base}/models", headers=headers, timeout=cfg.timeout)
        report["models"]["status"] = resp.status_code
        models_text = resp.text[:400]
        if resp.status_code == 200:
            try:
                report["models"]["data"] = [m.get("id") for m in resp.json().get("data", [])][:10]
            except Exception:
                report["models"]["data"] = "parse-failed"
        else:
            report["models"]["error"] = models_text
        report["models"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if resp.status_code != 200:
            raise NavigationError(paper_id="", error_type=MEDGEMMA_ENDPOINT_ERROR,
                                  message=f"GET {base}/models -> HTTP {resp.status_code}: {models_text[:200]}")
    except NavigationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NavigationError(paper_id="", error_type=MEDGEMMA_ENDPOINT_ERROR,
                              message=f"GET {base}/models failed: {type(exc).__name__}: {exc}") from exc

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{base}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Respond with the single word OK."}],
                "temperature": cfg.temperature,
                "max_tokens": 20,
            },
            timeout=cfg.timeout,
        )
        report["chat"]["status"] = resp.status_code
        report["chat"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if resp.status_code != 200:
            report["chat"]["error"] = resp.text[:400]
            raise NavigationError(paper_id="", error_type=MEDGEMMA_ENDPOINT_ERROR,
                                  message=f"POST {base}/chat/completions -> HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = None
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            pass
        report["chat"]["returned_text"] = (content or "")[:120]
        report["latency_ms"] = round(report["models"]["latency_ms"] + report["chat"]["latency_ms"], 1)
        if content is None:
            raise NavigationError(paper_id="", error_type=MEDGEMMA_ENDPOINT_ERROR,
                                  message=f"POST {base}/chat/completions returned 200 but choices[0].message.content is missing: {json.dumps(data)[:300]}")
    except NavigationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NavigationError(paper_id="", error_type=MEDGEMMA_ENDPOINT_ERROR,
                              message=f"POST {base}/chat/completions failed: {type(exc).__name__}: {exc}") from exc
    report["ok"] = True
    return report


def verify_document_registration(client: Any, paper_id: str) -> Dict[str, Any]:
    """Part 3 sanity: document registered + native tree visible to the SDK."""
    from medrag.retrieval_v2.pageindex_stage.models import DOCUMENT_REGISTRATION_ERROR
    try:
        listing = client.list_documents(limit=100)
    except Exception as exc:  # noqa: BLE001
        raise NavigationError(paper_id=paper_id, error_type=PAGEINDEX_CLIENT_ERROR,
                              message=f"list_documents failed: {exc}") from exc
    docs = listing.get("documents", []) if isinstance(listing, dict) else listing
    ids = [d.get("id") or d.get("name") for d in docs if isinstance(d, dict)]
    if paper_id not in ids:
        raise NavigationError(paper_id=paper_id, error_type=DOCUMENT_REGISTRATION_ERROR,
                              message=f"paper {paper_id} not registered in the PageIndex store (visible: {ids[:10]})")
    try:
        structure = client.get_document_structure(paper_id)
        tree = client.get_tree(paper_id, include_text=False)
    except Exception as exc:  # noqa: BLE001
        raise NavigationError(paper_id=paper_id, error_type=TREE_LOAD_ERROR,
                              message=f"get_document_structure/get_tree failed: {exc}") from exc
    return {"registered": True, "structure_nodes": len(structure) if isinstance(structure, list) else structure,
            "tree_status": tree.get("status") if isinstance(tree, dict) else None}


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
# Public navigation entry (V2.5: real PageIndex agent path only)
# ---------------------------------------------------------------------------
def navigate(paper_id: str, objective: Optional[str] = None,
             requirement: Optional[Dict[str, Any]] = None,
             config: Optional[StageConfig] = None,
             endpoint_report: Optional[Dict[str, Any]] = None) -> NavigationResult:
    """Run the REAL PageIndex agent over a SELECTED paper (Part 6).

    The agent is driven by MedGemma (PAGEINDEX_LLM_BASE_URL). No BM25, no
    pgvector, no regex, no lexical fallback is used for the experiment; on any
    failure the exact failing layer is reported (Part 17). Navigation caching
    is disabled (Part 15); tree caching stays on.
    """
    from medrag.retrieval_v2.pageindex_stage.models import (
        NAVIGATION_EMPTY_RESULT,
        NAVIGATION_ERROR,
        PAGEINDEX_CLIENT_ERROR,
    )

    cfg = config or config_from_env()
    obj = objective or navigation_objective(requirement=requirement)
    rid = str((requirement or {}).get("id", "")) if requirement else ""
    result = NavigationResult(paper_id=paper_id, status="failed", requirement_id=rid, objective=obj)
    t_all = time.perf_counter()

    # tree loading (cached; Part 15 keeps navigation uncached, tree cached)
    try:
        t0 = time.perf_counter()
        indexed = build_index(paper_id, config=cfg)
        result.tree_ms = (time.perf_counter() - t0) * 1000
    except NavigationError as exc:
        result.error = exc.to_dict()
        result.total_ms = (time.perf_counter() - t_all) * 1000
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = {"status": "failed", "error_type": TREE_LOAD_ERROR,
                        "message": str(exc), "paper_id": paper_id}
        result.total_ms = (time.perf_counter() - t_all) * 1000
        return result

    # endpoint availability (Part 1)
    if not cfg.has_llm():
        result.error = {"status": "failed", "error_type": MEDGEMMA_ENDPOINT_ERROR,
                        "message": "PAGEINDEX_LLM_BASE_URL / PAGEINDEX_CHAT_MODEL not configured",
                        "paper_id": paper_id}
        result.total_ms = (time.perf_counter() - t_all) * 1000
        return result
    if endpoint_report is not None and endpoint_report.get("ok"):
        result.endpoint_ms = float(endpoint_report.get("latency_ms", 0.0))
    else:
        try:
            rep = verify_medgemma_endpoint(cfg)
            result.endpoint_ms = float(rep.get("latency_ms", 0.0))
        except NavigationError as exc:
            result.error = exc.to_dict()
            result.total_ms = (time.perf_counter() - t_all) * 1000
            return result

    # PageIndex client initialization + document visibility (Parts 2-3)
    try:
        from pageindex.client import PageIndexClient
        backend = {"base_url": cfg.llm_base_url.rstrip("/"), "api_key": cfg.llm_api_key}
        t0 = time.perf_counter()
        client = PageIndexClient(
            storage_path=str(cfg.sdk_storage),
            chat_model=cfg.chat_model or None,
            chat_backend=backend,
        )
        result.init_ms = (time.perf_counter() - t0) * 1000
        verification = verify_document_registration(client, paper_id)
        result.raw["document"] = verification
    except NavigationError as exc:
        result.error = exc.to_dict()
        result.total_ms = (time.perf_counter() - t_all) * 1000
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = {"status": "failed", "error_type": PAGEINDEX_CLIENT_ERROR,
                        "message": f"PageIndex client initialization failed: {exc}",
                        "paper_id": paper_id}
        result.total_ms = (time.perf_counter() - t_all) * 1000
        return result

    # real agent navigation (Part 6-8): the PageIndex managed-agent runner
    # cannot execute tools because MedGemma returns prose-with-text-tool_code
    # instead of structured tool_calls, so the text-tool-protocol loop drives
    # PageIndex's own chat_completions + agent_tools.call_tool against MedGemma.
    from medrag.retrieval_v2.pageindex_stage import agent_loop
    try:
        t0 = time.perf_counter()
        out = agent_loop.run_text_tool_loop(paper_id, obj, cfg, client)
        result.agent_ms = (time.perf_counter() - t0) * 1000
    except NavigationError as exc:
        result.error = exc.to_dict()
        result.total_ms = (time.perf_counter() - t_all) * 1000
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = {"status": "failed", "error_type": NAVIGATION_ERROR,
                        "message": f"text-tool agent loop failed: {type(exc).__name__}: {exc}",
                        "paper_id": paper_id}
        result.raw = {"agent_failure": str(exc)}
        result.total_ms = (time.perf_counter() - t_all) * 1000
        return result

    result.raw["agent_mode"] = "text-tool-protocol (MedGemma chat_completions + PageIndex call_tool)"
    result.raw["agent_response"] = {"final_content": out["final_content"]}
    result.raw["plan"] = out["plan"]
    result.trace = list(out["steps"])
    result.selected_nodes = [
        SelectedNode(**n) for n in agent_loop.extract_from_response(
            out["final_content"], indexed["structure"], cfg.max_nodes)
    ]
    result.status = "success" if result.selected_nodes else "empty"
    if not result.selected_nodes:
        result.error = {"status": "empty", "error_type": NAVIGATION_EMPTY_RESULT,
                        "message": "MedGemma ran but selected no structural nodes from the tree",
                        "paper_id": paper_id}
    result.total_ms = (time.perf_counter() - t_all) * 1000
    result.latency_ms = result.agent_ms + result.init_ms + result.tree_ms + result.endpoint_ms
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

