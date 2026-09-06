"""Per-task research worker: the agent-decided tool loop.

Runs inside the research graph (graph.py research_worker node). The worker
LLM picks ONE tool per step (corpus, web, terminology, inspection, author
lookup, deep inspection - or finish); code executes the call and enforces
THAT tool's guardrail:

  * corpus tools (postgres_search, retrieve, lookup_by_author,
    inspect_paper) -> every candidate through the VERIFIER critic;
  * searxng_search -> trust-gated fetch + RELIABILITY critic + verifier;
  * umls_lookup -> terminology only, never evidence (no items created);
  * deep_inspect quotes -> verifier.

There is no fixed local-first order: corpus-before-web is a prompted
preference, and the planner's tool list is an allow-list scope, not a
script. Budgets (tool steps, web searches) are hard and code-enforced.
"""

import json
import logging
from typing import Any

from src.agents.graph import emit_event
from src.agents.state import (
    EvidenceItem,
    ResearchRequirement,
    RunBudget,
)
from src.retrieval.plans import PlannedEntity, SubQueryPlan
from src.agents.timeouts import run_with_timeout, timeout_for


async def _llm(coro, label: str):
    """Run one LLM/tool call with the per-call timeout (never hang a worker)."""
    return await run_with_timeout(coro, timeout_for("llm_call"), label)


logger = logging.getLogger("agents.worker")


def _critic_notes(rejected: list) -> str:
    return "\n".join(str(r.get("note", "")) for r in rejected[-10:])


async def _local_retrieve(payload: dict, query: str, seen_chunks: set, top_k: int = 5):
    """PRIMARY local retrieval: postgres (medrag.chunks), falling back to the
    parquet hybrid (retrieve) only when postgres is unreachable/empty.

    Returns (results: list[dict], method: str, health: dict).
    method is the retrieval_method tag stamped on every candidate, e.g.
    "pgfts+pgvector" or "hybrid:bm25+dense"; health carries corpus stats /
    availability so the caller can surface corpus problems (Gap A).
    """
    task_id = payload["task_id"]
    entities = payload.get("entities") or []

    results, method, health = await _pg_fetch(
        task_id, query, entities, seen_chunks, top_k=top_k)
    if health.get("available") and not health.get("empty_corpus"):
        return results, method, health

    # postgres unreachable or empty -> parquet hybrid fallback
    if not health.get("available"):
        from src.agents.graph import progress
        progress(f"    [research:{task_id}] postgres unavailable - falling back to retrieve")
    else:
        from src.agents.graph import progress
        progress(f"    [research:{task_id}] postgres corpus EMPTY {health['stats']} - "
                 "falling back to retrieve; web gap-fill will be critical")

    out, method = await _hybrid_fetch(
        task_id, query, payload["text"], seen_chunks, top_k=top_k)
    return out, method, health


async def _web_fill(payload: dict, req: ResearchRequirement, queries: list[str],
                    verifier, budget: RunBudget) -> None:
    """WEB gap-fill: only called AFTER local retrieval rounds are exhausted and
    the requirement is still unsatisfied (Gap B). Queries are INFORMED by the
    local phase (critic rejection notes + missing evidence are in the prompt
    that produced them). Every web candidate passes the SAME verifier gate.
    """
    from src.agents.graph import progress
    from src.agents.tools.searxng import searxng_base_url, searxng_search_impl

    task_id = payload["task_id"]
    max_searches = budget.max_searches - budget.searches_used
    if max_searches <= 0:
        progress(f"    [research:{task_id}] web budget exhausted - no gap-fill")
        return

    from src.agents.stages import make_reliability_critic
    reliability_critic = make_reliability_critic()

    for query in queries:
        if req.satisfied() or max_searches <= 0:
            break
        progress(f"        [web] {query}")
        emit_event("web_search_started", requirement_id=task_id,
                   query=query, base=searxng_base_url())
        try:
            out = await _llm(searxng_search_impl(query=query, top_k=6, trusted_only=True),
                             f"web-search:{query[:60]}")
            data = json.loads(out)
        except Exception as exc:
            logger.warning("searxng failed for %s: %s", query, exc)
            continue
        if not data.get("available"):
            progress(f"        [web] searxng unreachable - skipping web gap-fill")
            emit_event("web_search_done", requirement_id=task_id, query=query,
                       available=False, count=0, urls=[])
            break
        budget.searches_used += 1
        max_searches -= 1
        progress(f"        [web] {len(data.get('results') or [])} trusted result(s) "
                 f"({data.get('dropped_blocked', 0)} blocked, "
                 f"{data.get('dropped_unverified', 0)} unverified dropped)")
        emit_event("web_search_done", requirement_id=task_id, query=query,
                   count=len(data.get("results") or []),
                   dropped_blocked=data.get("dropped_blocked", 0),
                   dropped_unverified=data.get("dropped_unverified", 0),
                   urls=[h.get("url", "") for h in (data.get("results") or [])][:10])

        for i, hit in enumerate((data.get("results") or [])[:5], 1):
            if req.satisfied():
                break
            # DIVE DEEP: fetch the ACTUAL page and verify/reliability-judge
            # its real text, not the 500-char search snippet.
            from src.agents.stages import fetch_web_hit, verify_web_hit

            text, _fetched, _bundle = await fetch_web_hit(
                hit.get("url", ""), hit.get("snippet", ""), task_id,
                tag="web", extra={"query": query})
            if not text:
                continue
            from src.agents.stages import stamp_web_provenance
            item = EvidenceItem(
                id=f"{task_id}.W{len(req.items) + 1}",
                run_id="",
                requirement_id=task_id,
                chunk_id="",
                document_id="",
                section="web",
                unit_kind="web_result",
                text=text,
                source_url=hit.get("url", ""),
                trust=hit.get("trust", "unverified"),
                source_query=query,
                retrieval_method=("web:" + str(hit.get("engine", "searxng"))),
                rank=i,
            )
            stamp_web_provenance(item, hit, _bundle, query)
            item.derive_evidence_level()
            req.add_item(item)
            # Gap C: RELIABILITY gate - the "another agent" for web sources.
            # Only VERIFIED web candidates are judged, and the verdict is
            # stored on the item (never trusted to the site-gate alone).
            await verify_web_hit(req, item, verifier=verifier,
                                 reliability_critic=reliability_critic,
                                 tag="web")
    req.derive_status()


# Preferred order when the selector output is unusable: cheapest corpus
# first, web last. Only a fallback - the agent decides the real order.
_TOOL_STEP_ORDER = ("postgres_search", "retrieve", "searxng_search",
                    "lookup_by_author", "inspect_paper", "umls_lookup")


def _resolve_default(scope: list[str]) -> str:
    """Deterministic fallback tool (or finish when scope is empty)."""
    for t in _TOOL_STEP_ORDER:
        if t in scope:
            return t
    return "finish"


async def _pg_fetch(task_id: str, query: str, entities: list,
                    seen_chunks: set, top_k: int = 5):
    """Single postgres_search tool call -> (results, method, health)."""
    from src.agents.tools.postgres_search import postgres_search_impl

    try:
        out = await postgres_search_impl(
            requirement_id=task_id,
            query=query,
            entities=entities,
            top_k=top_k,
            exclude_chunk_ids=list(seen_chunks),
        )
        data = json.loads(out)
    except Exception as exc:
        logger.warning("postgres_search failed for %s/%s: %s", task_id, query, exc)
        data = {"available": False, "results": []}

    health = {"available": data.get("available", False),
              "empty_corpus": data.get("empty_corpus", False),
              "stats": data.get("stats", {})}
    results = []
    if data.get("available") and not data.get("empty_corpus"):
        results = data.get("results") or []
        for r in results:
            r["retrieval_method"] = "+".join(r.get("methods") or ["pgfts"]) or "pgfts"
    return results, "pgfts+pgvector", health


async def _hybrid_fetch(task_id: str, query: str, requirement_text: str,
                        seen_chunks: set, top_k: int = 5):
    """Single retrieve (parquet hybrid) tool call -> (results, method)."""
    from src.agents.reuse import get_hybrid_retriever

    sub = SubQueryPlan(
        id=task_id, target=query, intent="evidence", query=query,
        focus="evidence", evidence_required=[requirement_text],
    )
    try:
        results = await get_hybrid_retriever().search(
            sub, top_k=top_k, exclude_chunk_ids=list(seen_chunks),
            restore_paragraphs=True,
        )
    except Exception as exc:
        logger.warning("hybrid retrieval failed for %s/%s: %s", task_id, query, exc)
        return [], "hybrid:bm25+dense"

    out = []
    for res in results:
        out.append({
            "rank": getattr(res, "rank", 0),
            "chunk_id": res.chunk_id,
            "document_id": res.document_id,
            "section": res.section,
            "unit_kind": getattr(res, "unit_kind", "paragraph"),
            "score": float(getattr(res, "rrf_score", 0.0) or 0.0),
            "methods": getattr(res, "methods", None) or ["hybrid"],
            "text": res.paragraph_text,
        })
    for r in out:
        r["retrieval_method"] = "+".join(r.get("methods") or ["hybrid"]) or "hybrid"
    return out, "hybrid:bm25+dense"


async def _verify_candidate(req: ResearchRequirement, item: EvidenceItem,
                            verifier: Any, task_id: str, logs: dict) -> None:
    """GUARDRAIL (corpus): every candidate through the verifier critic.

    Updates the retrieved/rejected logs the selector learns from.
    """
    from src.agents.graph import progress

    try:
        from src.agents.verify_queue import submit_verify
        await _llm(submit_verify(verifier, req, item), f"verify:{item.id}")
        if item.verdict:
            progress(f"            [verify] {item.id} -> "
                     f"{item.status.value} ({item.verdict.support.value})")
            emit_event("verdict",
                       evidence_id=item.id,
                       requirement_id=task_id,
                       document_id=item.document_id or "",
                       status=item.status.value,
                       support=item.verdict.support.value,
                       confidence=round(float(item.verdict.confidence), 3),
                       note=item.verdict.note[:200],
                       section=item.section)
            if item.status.value == "rejected":
                logs["rejected"].append(
                    {"note": item.verdict.note, "evidence_id": item.id})
            else:
                logs["retrieved"].append({
                    "document_id": item.document_id,
                    "section": item.section,
                    "text": item.text[:300],
                })
    except Exception as exc:
        logger.warning("verification failed for %s: %s", item.id, exc)
        item.verdict = None
    req.derive_status()


async def _add_corpus_items(task_id: str, req: ResearchRequirement,
                            results: list, method: str, query: str,
                            prefix: str, verifier: Any, logs: dict,
                            seen_chunks: set) -> None:
    """Shape raw corpus hits into candidates + verify each (guardrail)."""
    for i, res in enumerate(results, 1):
        if req.satisfied():
            break
        if res.get("chunk_id") and res["chunk_id"] in seen_chunks:
            continue
        seen_chunks.add(res.get("chunk_id", f"{prefix}{i}"))
        item = EvidenceItem(
            id=f"{task_id}.{prefix}{len(req.items) + 1}",
            run_id="",
            requirement_id=task_id,
            chunk_id=res.get("chunk_id", ""),
            document_id=res.get("document_id", ""),
            section=res.get("section", ""),
            unit_kind=res.get("unit_kind", "paragraph"),
            text=res.get("text", ""),
            source_query=query,
            retrieval_method=res.get("retrieval_method", method),
            rank=int(res.get("rank", 0) or i),
        )
        item.source_kind = "corpus"
        item.derive_evidence_level()
        req.add_item(item)
        await _verify_candidate(req, item, verifier, task_id, logs)


async def _exec_umls(task_id: str, text: str, entities: list,
                     pool: list[str]) -> list[str]:
    """umls_lookup tool: terminology only - NEVER creates evidence items."""
    from src.agents.graph import progress
    from src.agents.reuse import get_umls_enricher

    try:
        enricher = get_umls_enricher()
        if enricher.enabled and entities:
            sub = SubQueryPlan(
                id=task_id, target=text, query=text, intent="evidence",
                focus="evidence", evidence_required=[text],
                entities=[PlannedEntity(text=e) for e in entities],
            )
            terms = await enricher.enrich_subquery(sub)
            seen = {n.casefold() for n in pool}
            for t in terms:
                for name in ([t.preferred_name] if t.preferred_name else []) + t.synonyms:
                    name = " ".join((name or "").split())
                    key = name.casefold()
                    if name and key not in seen:
                        seen.add(key)
                        pool.append(name)
            del pool[15:]
            progress(f"    [umls:{task_id}] pool: {pool[:8]}")
    except Exception as exc:
        logger.warning("UML enrichment failed for %s: %s", task_id, exc)
    return pool


async def _exec_inspect(task_id: str, req: ResearchRequirement, logs: dict,
                        pmc_id: str, query: str, verifier: Any) -> None:
    """inspect_paper tool: one paper's passages, each verified (guardrail)."""
    from src.agents.tools.lookup import inspect_paper_impl

    try:
        data = await inspect_paper_impl(pmc_id=pmc_id, top_k=12)
    except Exception as exc:
        logger.warning("inspect failed %s: %s", pmc_id, exc)
        return
    for res in (data.get("results", []) or []):
        if req.satisfied() or not (res.get("text", "") or "").strip():
            continue
        item = EvidenceItem(
            id=f"{task_id}.D{len(req.items) + 1}",
            run_id="",
            requirement_id=task_id,
            chunk_id=res.get("chunk_id", ""),
            document_id=res.get("document_id", "") or pmc_id,
            section=res.get("section", ""),
            unit_kind=res.get("unit_kind", "paragraph"),
            text=res.get("text", ""),
            source_query=query,
            retrieval_method=res.get("retrieval_method", "doc_lookup:inspect"),
            rank=int(res.get("rank", 0) or 0),
        )
        item.source_kind = "corpus"
        item.derive_evidence_level()
        req.add_item(item)
        await _verify_candidate(req, item, verifier, task_id, logs)


async def _exec_lookup(task_id: str, req: ResearchRequirement, logs: dict,
                       author: str, topic: str, verifier: Any,
                       seen_chunks: set) -> None:
    """lookup_by_author tool: author's local papers, each verified."""
    from src.agents.tools.lookup import lookup_by_author_impl

    try:
        data = await lookup_by_author_impl(author=author, topic=topic, top_k=5)
    except Exception as exc:
        logger.warning("author lookup failed %s: %s", task_id, exc)
        return
    await _add_corpus_items(task_id, req, data.get("results", []) or [],
                            "lookup", topic, "A", verifier, logs, seen_chunks)


async def _exec_deep_inspect(task_id: str, req: ResearchRequirement,
                             inspector: Any, verifier: Any,
                             deep_inspected: set) -> int:
    """deep_inspect tool: quotes from one verified doc, each verified."""
    from src.agents.graph import progress

    for item in req.verified_items():
        if item.document_id and item.document_id not in deep_inspected:
            break
    else:
        return 0
    deep_inspected.add(item.document_id)
    progress(f"        [deep-inspect] {item.document_id}")
    try:
        findings = await _llm(deep_inspect(inspector, req.text, item.text),
                              f"deep-inspect:{item.document_id}")
    except Exception as exc:
        logger.warning("deep inspect failed: %s", exc)
        return 0
    added = 0
    for j, f in enumerate(findings, 1):
        quote = str(f.get("quote") or f.get("claim") or "").strip()
        if not quote:
            continue
        item2 = EvidenceItem(
            id=f"{task_id}.DI{j}",
            run_id="",
            requirement_id=task_id,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            section=str(f.get("section") or item.section),
            unit_kind=item.unit_kind,
            text=quote,
            claim=str(f.get("claim") or ""),
            source_query=item.source_query,
            retrieval_method="deep_inspection",
            rank=j,
        )
        req.add_item(item2)
        try:
            from src.agents.verify_queue import submit_verify
            await submit_verify(verifier, req, item2)
            added += 1
        except Exception as exc:
            logger.warning("deep verify failed: %s", exc)
        req.derive_status()
    return added


async def research_worker(payload: dict) -> dict:
    """One research worker: the agent picks the tool each step, code guards it.

    The tool selector LLM chooses ONE action per step from the payload's
    tool scope (planner allow-list); each executed tool gets its guardrail
    (corpus -> verifier, web -> reliability + verifier, umls -> never
    evidence). Loop ends on satisfied / finish / step or search budget.
    A deterministic deep-inspect tail still runs when unsatisfied.
    Every candidate MUST pass the verifier (rule 2 - hard gate).
    """
    from src.agents.stages import (
        WORKER_TOOL_SCOPE,
        choose_tool_action,
        deep_inspect,
        make_deep_inspector,
        make_tool_selector,
        make_verifier,
    )
    from src.agents.graph import progress

    task_id = payload["task_id"]
    text = payload["text"]
    entities = payload.get("entities") or []
    progress(f"[research:{task_id}] worker started")

    req = ResearchRequirement(id=task_id, text=text,
                              target_n=payload.get("target_n", 3))
    budget = RunBudget(
        max_searches=payload.get("budget_max_searches", 5),
        max_retrieval_rounds=payload.get("budget_max_rounds", 3),
    )
    verifier = make_verifier()
    inspector = make_deep_inspector()
    selector = make_tool_selector()

    # Planner allow-list scope: the agent chooses WITHIN it, never outside.
    raw_scope = payload.get("tools") or ["postgres_search", "retrieve",
                                         "searxng_search", "umls_lookup"]
    scope = [t for t in raw_scope if t in WORKER_TOOL_SCOPE]
    if "deep_inspect" not in scope:
        scope.append("deep_inspect")  # worker-internal, always available
    max_steps = max(1, int(payload.get("budget_max_steps",
                                       2 * budget.max_retrieval_rounds)))

    pool: list[str] = []          # terminology (umls tool fills it)
    retrieved_log: list[dict] = []
    rejected_log: list[dict] = []
    seen_chunks: set = set()
    tried: list[str] = []         # tool trail for the selector
    deep_inspected: set = set()   # document_ids already deep-inspected

    # --- agent-decided tool loop (no fixed order) ---------------------------
    logs = {"retrieved": retrieved_log, "rejected": rejected_log}
    for step in range(1, max_steps + 1):
        if req.satisfied():
            progress(f"    [research:{task_id}] satisfied after step {step - 1}")
            break
        searches_left = max(0, budget.max_searches - budget.searches_used)
        try:
            action = await _llm(choose_tool_action(
                selector, text, req.text, req.target_n,
                len(req.verified_items()),
                tried=tried,
                critic_notes=_critic_notes(rejected_log),
                searches_left=searches_left,
                steps_left=max_steps - step + 1,
                scope=scope,
                args=dict(payload.get("args", {}) or {}),
            ), f"tool-select:{task_id}")
        except Exception as exc:
            logger.warning("tool selector failed for %s: %s", task_id, exc)
            action = {"tool": "__default__", "query": req.text, "args": {}}
        tool = str(action.get("tool", "") or "")
        query = str(action.get("query", "") or "") or req.text
        if tool == "__default__":
            tool = _resolve_default(scope)
            progress(f"    [research:{task_id}] selector unclear "
                     f"({action.get('reason', '')[:80]}) - defaulting to {tool}")
        if tool == "finish" or tool not in scope:
            if tool not in scope and tool != "finish":
                progress(f"    [research:{task_id}] out-of-scope tool {tool!r} "
                         "rejected - stopping")
            break
        if tool == "searxng_search" and searches_left <= 0:
            progress(f"    [research:{task_id}] web budget exhausted - "
                     "skipping searxng_search")
            tried.append("searxng_search(exhausted)")
            continue
        progress(f"    [research:{task_id}] step {step}/{max_steps}: "
                 f"{tool} ({action.get('reason', '')[:100]})")
        emit_event("tool_step", requirement_id=task_id, step=step,
                   tool=tool, query=query)
        tried.append(tool)
        try:
            if tool == "postgres_search":
                results, method, _health = await _pg_fetch(
                    task_id, query, entities, seen_chunks)
                progress(f"        retrieved {len(results)} candidate(s) [{method}]")
                await _add_corpus_items(task_id, req, results, method,
                                        query, "P", verifier, logs, seen_chunks)
            elif tool == "retrieve":
                results, method = await _hybrid_fetch(
                    task_id, query, req.text, seen_chunks)
                progress(f"        retrieved {len(results)} candidate(s) [{method}]")
                await _add_corpus_items(task_id, req, results, method,
                                        query, "H", verifier, logs, seen_chunks)
            elif tool == "searxng_search":
                await _web_fill(payload, req, [query], verifier, budget)
            elif tool == "umls_lookup":
                await _exec_umls(task_id, text, entities, pool)
            elif tool == "inspect_paper":
                pmc = str((action.get("args", {}) or {}).get("pmc_id", "") or "")
                if not pmc:
                    import re as _re
                    m = _re.search(r"PMC\d+", text, _re.IGNORECASE)
                    pmc = m.group(0).upper() if m else ""
                if not pmc:
                    progress(f"    [research:{task_id}] inspect_paper needs "
                             "a PMCID - skipping")
                    continue
                await _exec_inspect(task_id, req, logs, pmc, query, verifier)
            elif tool == "lookup_by_author":
                author = str((action.get("args", {}) or {}).get("author", "") or "")
                if not author and entities:
                    author = str(entities[0])
                if not author:
                    progress(f"    [research:{task_id}] lookup_by_author needs "
                             "an author - skipping")
                    continue
                await _exec_lookup(task_id, req, logs, author, text,
                                   verifier, seen_chunks)
            elif tool == "deep_inspect":
                await _exec_deep_inspect(task_id, req, inspector, verifier,
                                         deep_inspected)
        except Exception as exc:
            logger.warning("tool %s failed for %s: %s", tool, task_id, exc)
            continue
        req.derive_status()

    if not req.satisfied() and not tried:
        progress(f"    [research:{task_id}] no tool steps ran - "
                 "marking requirement honestly")
        req.derive_status()

    # --- deep inspect promising docs if still unsatisfied -------------------
    if not req.satisfied():
        for item in req.verified_items():
            if req.satisfied():
                break
            progress(f"        [deep-inspect] {item.document_id}")
            try:
                findings = await _llm(deep_inspect(inspector, text, item.text),
                                      f"deep-inspect:{item.document_id}")
            except Exception as exc:
                logger.warning("deep inspect failed: %s", exc)
                continue
            for j, f in enumerate(findings, 1):
                quote = str(f.get("quote") or f.get("claim") or "").strip()
                if not quote:
                    continue
                item2 = EvidenceItem(
                    id=f"{task_id}.DI{j}",
                    run_id="",
                    requirement_id=task_id,
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    section=str(f.get("section") or item.section),
                    unit_kind=item.unit_kind,
                    text=quote,
                    claim=str(f.get("claim") or ""),
                    source_query=item.source_query,
                    retrieval_method="deep_inspection",
                    rank=j,
                )
                req.add_item(item2)
                try:
                    from src.agents.verify_queue import submit_verify
                    await submit_verify(verifier, req, item2)
                except Exception as exc:
                    logger.warning("deep verify failed: %s", exc)
                req.derive_status()

    req.derive_status()
    progress(f"    [research:{task_id}] final status={req.status.value}, "
             f"{len(req.verified_items())} verified, "
             f"searches_used={budget.searches_used}")
    return {"requirements": [req], "searches_used": budget.searches_used}


__all__ = ["research_worker"]
