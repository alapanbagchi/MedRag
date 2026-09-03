"""Per-task research worker: the adaptive retrieve -> verify loop.

Runs inside the research graph (graph.py research_worker node): UML
enrichment, local retrieval rounds (postgres primary, hybrid fallback),
trust-gated web gap-fill, then deep inspection - every candidate through
the verifier critic.
"""

import json
import logging

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
    from src.agents.tools.postgres_search import postgres_search_impl

    task_id = payload["task_id"]
    entities = payload.get("entities") or []
    health: dict = {}

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
    if data.get("available") and not data.get("empty_corpus"):
        results = data.get("results") or []
        for r in results:
            r["retrieval_method"] = "+".join(r.get("methods") or ["pgfts"]) or "pgfts"
        return results, "pgfts+pgvector", health

    # postgres unreachable or empty -> parquet hybrid fallback
    if not data.get("available"):
        from src.agents.graph import progress
        progress(f"    [research:{task_id}] postgres unavailable - falling back to retrieve")
    else:
        from src.agents.graph import progress
        progress(f"    [research:{task_id}] postgres corpus EMPTY {health['stats']} - "
                 "falling back to retrieve; web gap-fill will be critical")

    from src.agents.reuse import get_hybrid_retriever

    sub = SubQueryPlan(
        id=task_id, target=query, intent="evidence", query=query,
        focus="evidence", evidence_required=[payload["text"]],
    )
    try:
        results = await get_hybrid_retriever().search(
            sub, top_k=top_k, exclude_chunk_ids=list(seen_chunks),
            restore_paragraphs=True,
        )
    except Exception as exc:
        logger.warning("hybrid retrieval failed for %s/%s: %s", task_id, query, exc)
        return [], "hybrid:bm25+dense", health

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
    return out, "hybrid:bm25+dense", health


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

    from src.agents.agents.stages import make_reliability_critic
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
            from src.agents.agents.stages import fetch_web_hit, verify_web_hit

            text, _fetched = await fetch_web_hit(
                hit.get("url", ""), hit.get("snippet", ""), task_id,
                tag="web", extra={"query": query})
            if not text:
                continue
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
            item.derive_evidence_level()
            req.add_item(item)
            # Gap C: RELIABILITY gate - the "another agent" for web sources.
            # Only VERIFIED web candidates are judged, and the verdict is
            # stored on the item (never trusted to the site-gate alone).
            await verify_web_hit(req, item, verifier=verifier,
                                 reliability_critic=reliability_critic,
                                 tag="web")
    req.derive_status()


async def research_worker(payload: dict) -> dict:
    """One research worker (qwen): adaptive loop mirroring agentic-v3.

    ORDER (agreed, Gap A + Gap B):
      1. UML enrichment (deterministic) -> terminology pool.
      2. LOCAL retrieval rounds (PRIMARY = postgres medrag.chunks, fallback =
         parquet hybrid), every candidate through the VERIFIER critic.
      3. ONLY when local rounds are exhausted AND still unsatisfied: WEB
         gap-fill via searxng (trust-gated), queries informed by the local
         phase, same verifier gate, counted against the search budget.
      4. Deep inspector on promising docs.
    Every candidate MUST pass the verifier (rule 2 - hard gate).
    """
    from src.agents.agents.stages import (
        deep_inspect,
        make_deep_inspector,
        make_replanner,
        make_search_planner,
        make_verifier,
        plan_queries,
        replan_queries,
        verify_item,
    )
    from src.agents.graph import progress
    from src.agents.reuse import get_umls_enricher

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
    planner = make_search_planner()
    replanner = make_replanner()
    verifier = make_verifier()
    inspector = make_deep_inspector()

    # --- UML terminology pool (for the planner) -----------------------------
    pool: list[str] = []
    try:
        enricher = get_umls_enricher()
        if enricher.enabled and entities:
            sub = SubQueryPlan(
                id=task_id, target=text, query=text, intent="evidence",
                focus="evidence", evidence_required=[text],
                entities=[PlannedEntity(text=e) for e in entities],
            )
            terms = await enricher.enrich_subquery(sub)
            seen: set = set()
            for t in terms:
                for name in ([t.preferred_name] if t.preferred_name else []) + t.synonyms:
                    name = " ".join((name or "").split())
                    key = name.casefold()
                    if name and key not in seen:
                        seen.add(key)
                        pool.append(name)
            pool = pool[:15]
            progress(f"    [umls:{task_id}] pool: {pool[:8]}")
    except Exception as exc:
        logger.warning("UML enrichment failed for %s: %s", task_id, exc)

    previous_queries: list[str] = []
    retrieved_log: list[dict] = []
    rejected_log: list[dict] = []
    seen_chunks: set = set()
    web_queries: list[str] = []

    # --- LOCAL rounds (primary = postgres) ----------------------------------
    local_rounds = max(1, budget.max_retrieval_rounds)
    for round_no in range(1, local_rounds + 1):
        if req.satisfied():
            progress(f"    [research:{task_id}] satisfied after round {round_no - 1}")
            break

        if round_no == 1:
            progress(f"    [research:{task_id}] SEARCH PLANNER (round 1)")
            try:
                queries = await _llm(plan_queries(
                    planner, text, text, pool=pool,
                    previous_queries=previous_queries,
                    critic_notes=_critic_notes(rejected_log),
                ), f"plan:{task_id}")
            except Exception as exc:
                logger.warning("search planner failed for %s: %s", task_id, exc)
                queries = [text]
        else:
            progress(f"    [research:{task_id}] REPLANNER (round {round_no})")
            try:
                queries = await _llm(replan_queries(
                    replanner, text, text, previous_queries,
                    retrieved_log, rejected_log,
                    [i.document_id for i in req.verified_items()],
                ), f"replan:{task_id}")
            except Exception as exc:
                logger.warning("replanner failed for %s: %s", task_id, exc)
                queries = [text] if not previous_queries else []
        if not queries:
            progress(f"    [research:{task_id}] no new queries - stopping "
                     "search rounds")
            break
        previous_queries.extend(queries)
        progress(f"    [research:{task_id}] local queries: {queries}")
        emit_event("search_round", round_no=round_no,
                   requirement_id=task_id, queries=queries,
                   source="corpus")

        for query in queries:
            if req.satisfied():
                break
            progress(f"        retrieve: {query}")
            emit_event("query_start", requirement_id=task_id,
                       query=query, source="corpus")
            try:
                results, method, health = await _local_retrieve(
                    payload, query, seen_chunks, top_k=5)
            except Exception as exc:
                logger.warning("local retrieval failed for %s/%s: %s",
                               task_id, query, exc)
                continue
            seen_chunks.update(r.get("chunk_id") for r in results if r.get("chunk_id"))
            progress(f"        retrieved {len(results)} candidate(s) [{method}]")
            emit_event("retrieved", requirement_id=task_id,
                       query=query, count=len(results),
                       method=method,
                       papers=[{"document_id": r.get("document_id", ""),
                                "section": r.get("section", ""),
                                "score": round(float(r.get("score", 0.0)), 4)}
                               for r in results[:10]])

            for i, res in enumerate(results, 1):
                item = EvidenceItem(
                    id=f"{task_id}.E{round_no}-{i}",
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
                item.derive_evidence_level()   # deterministic hierarchy stamp
                req.add_item(item)
                try:
                    await _llm(verify_item(verifier, req, item), f"verify:{item.id}")
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
                            rejected_log.append(
                                {"note": item.verdict.note, "evidence_id": item.id})
                        else:
                            retrieved_log.append({
                                "document_id": item.document_id,
                                "section": item.section,
                                "text": item.text[:300],
                            })
                except Exception as exc:
                    logger.warning("verification failed for %s: %s", item.id, exc)
                    item.verdict = None
            req.derive_status()

    # --- WEB gap-fill (ONLY after local rounds + still unsatisfied) ---------
    if not req.satisfied():
        progress(f"    [research:{task_id}] local rounds exhausted; "
                 "planning web gap-fill")
        try:
            web_queries = await _llm(replan_queries(
                replanner, text, text, previous_queries,
                retrieved_log, rejected_log,
                [i.document_id for i in req.verified_items()],
            ), f"web-replan:{task_id}")
        except Exception as exc:
            logger.warning("web replan failed for %s: %s", task_id, exc)
            web_queries = []
        if not web_queries:
            # fall back to UUID-distinct semantic queries from the planner
            try:
                web_queries = await _llm(plan_queries(
                    planner, text, text, pool=pool,
                    previous_queries=previous_queries,
                    critic_notes=_critic_notes(rejected_log),
                ), f"web-plan:{task_id}")
            except Exception as exc:
                logger.warning("web plan failed for %s: %s", task_id, exc)
        if web_queries:
            progress(f"    [research:{task_id}] web queries: {web_queries}")
            await _web_fill(payload, req, web_queries, verifier, budget)

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
                    await verify_item(verifier, req, item2)
                except Exception as exc:
                    logger.warning("deep verify failed: %s", exc)
                req.derive_status()

    req.derive_status()
    progress(f"    [research:{task_id}] final status={req.status.value}, "
             f"{len(req.verified_items())} verified, "
             f"searches_used={budget.searches_used}")
    return {"requirements": [req], "searches_used": budget.searches_used}


__all__ = ["research_worker"]
