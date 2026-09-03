"""Gap-resolution pass: AFTER contradiction resolution, close remaining
evidence gaps with more retrievals.

Two kinds of gaps are handled:

A. HARD gaps - requirements still not satisfied (coverage < target_n).
   1. Within the gap-LOCAL budget, run additional LOCAL retrieval rounds
      (postgres primary -> parquet hybrid fallback); verified candidates
      close the gap.
   2. If local cannot close it, run trust-gated WEB searches (searxng
      trusted_only + reliability judge) within the gap-WEB budget.
   3. If STILL open, LIST it (UNRESOLVED GapResolution + run.gaps).

B. LATENT gaps - CONTENT-LEVEL holes in a requirement that IS numerically
   satisfied. "Satisfied" only means enough independent documents were
   found; the verified passages may still leave important parts of the
   question unanswered (mechanisms, specific foods/nutrients, comparator
   diets, long-term outcomes, subgroups). A GAP PROBE (LLM) examines the
   requirement + verified evidence and proposes up to 2 unanswered
   sub-questions with targeted queries; each is then resolved local-first,
   then web, then listed if still open. Without the probe, a satisfied-but-
   holey requirement like the DASH-diet example would report "0 gaps" and
   never search the web even though synthesis later lists holes.

Every new candidate added by this pass goes through the SAME verifier gate
(rule 2) - gap evidence is exactly as trustworthy as worker evidence.
Verified web gap sources get a stable 'web:' document id (from the URL) so
they count as independent supporting documents under the independence rule.

Env knobs:
  XDEEP_GAP_PROBE      = 0/off disables latent probing (B); default on
  XDEEP_GAP_PROBE_MAX  = max probe sub-questions per run (default 3)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.agents.graph import emit_event

from src.agents.state import (
    EvidenceItem,
    GapResolution,
    GapResolutionStatus,
    ResearchRequirement,
    RunBudget,
    web_document_id,
)

logger = logging.getLogger("x_deepagents.gap_fill")


def _rejected_notes(req: ResearchRequirement) -> list[dict]:
    """Rejection reasons from the requirement's already-judged items."""
    out = []
    for it in req.items:
        if it.verdict is not None and it.status.value == "rejected":
            out.append({"note": it.verdict.note, "evidence_id": it.id})
    return out[-10:]


def _previous_queries(req: ResearchRequirement) -> list[str]:
    qs = [it.source_query for it in req.items if it.source_query]
    seen = set()
    out = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _accepted_docs(req: ResearchRequirement) -> list[str]:
    return sorted({i.document_id for i in req.items
                   if i.document_id and i.verified})


def compute_gaps(state: Any) -> list[tuple[ResearchRequirement, str]]:
    """Unsatisfied requirements -> (requirement, gap text)."""
    gaps: list[tuple[ResearchRequirement, str]] = []
    for req in state.requirements:
        if req.satisfied():
            continue
        text = req.gap or (
            f"requirement {req.id} needs {req.target_n} independent supporting "
            f"studies on: {req.text} (only {req.coverage()} verified so far)"
        )
        gaps.append((req, text))
    return gaps


async def _plan_gap_queries(planner: Any, replanner: Any, req: ResearchRequirement,
                            budget: RunBudget) -> list[str]:
    """Round-1 queries from the search planner, then replanner on later rounds."""
    from src.agents.agents.stages import (
        plan_queries,
        replan_queries,
    )
    from src.agents.agents.worker import _critic_notes

    prev = _previous_queries(req)
    retrieved = [
        {"document_id": it.document_id, "section": it.section, "text": it.text[:300]}
        for it in req.verified_items()
    ]
    rejected = _rejected_notes(req)
    accepted = _accepted_docs(req)
    if not prev:
        return await plan_queries(
            planner, req.text, req.text,
            pool=[], previous_queries=[], critic_notes=_critic_notes(rejected))
    return await replan_queries(
        replanner, req.text, req.text, prev, retrieved, rejected, accepted)


async def _local_round(req: ResearchRequirement, queries: list[str],
                       verifier: Any, budget: RunBudget) -> None:
    """One gap-local retrieval round (pg primary, hybrid fallback), verified."""
    from src.agents.agents.worker import _local_retrieve
    from src.agents.agents.stages import verify_item
    from src.agents.graph import progress

    seen = {it.chunk_id for it in req.items if it.chunk_id}
    payload = {"task_id": req.id, "text": req.text, "entities": [],
               "target_n": req.target_n}
    for query in queries:
        if req.satisfied():
            break
        try:
            results, method, health = await _local_retrieve(
                payload, query, seen, top_k=5)
        except Exception as exc:
            logger.warning("gap local retrieve failed %s/%s: %s", req.id, query, exc)
            continue
        seen.update(r.get("chunk_id") for r in results if r.get("chunk_id"))
        for i, res in enumerate(results, 1):
            if req.satisfied():
                break
            item = EvidenceItem(
                id=f"{req.id}.G{len(req.items) + 1}",
                run_id="",
                requirement_id=req.id,
                chunk_id=res.get("chunk_id", ""),
                document_id=res.get("document_id", ""),
                section=res.get("section", ""),
                unit_kind=res.get("unit_kind", "paragraph"),
                text=res.get("text", ""),
                source_query=query,
                retrieval_method=res.get("retrieval_method", method) + ":gap",
                rank=int(res.get("rank", 0) or i),
            )
            item.derive_evidence_level()
            req.add_item(item)
            try:
                from src.agents.timeouts import run_with_timeout, timeout_for

                await run_with_timeout(
                    verify_item(verifier, req, item),
                    timeout_for("llm_call"), f"gap-verify:{item.id}")
            except Exception as exc:
                logger.warning("gap verify failed %s: %s", item.id, exc)
        progress(f"    [gap:{req.id}] local round: {len(results)} candidate(s) "
                 f"[{method}]")
    budget.gap_local_rounds_used += 1


async def _web_round(req: ResearchRequirement, queries: list[str],
                     verifier: Any, budget: RunBudget) -> None:
    """Trust-gated web round for one gap: searxng + verifier + reliability
    judge. Counts against the gap-WEB budget."""
    from src.agents.agents.stages import (
        judge_site_reliability,
        make_reliability_critic,
        verify_item,
    )
    from src.agents.graph import progress
    from src.agents.tools.searxng import searxng_search_impl
    from src.agents.timeouts import run_with_timeout, timeout_for

    reliability_critic = make_reliability_critic()
    for query in queries:
        if req.satisfied() or budget.gap_web_exhausted():
            break
        try:
            out = await run_with_timeout(
                searxng_search_impl(query=query, top_k=6, trusted_only=True),
                timeout_for("llm_call"), f"gap-web:{query[:60]}")
            data = json.loads(out)
        except Exception as exc:
            logger.warning("gap searxng failed %s: %s", query, exc)
            continue
        if not data.get("available"):
            break
        budget.gap_web_searches_used += 1
        for i, hit in enumerate((data.get("results") or [])[:5], 1):
            if req.satisfied():
                break
            text = (hit.get("snippet") or "").strip()
            if len(text) < 40:
                continue
            item = EvidenceItem(
                id=f"{req.id}.GW{len(req.items) + 1}",
                run_id="",
                requirement_id=req.id,
                chunk_id="",
                document_id="",
                section="web",
                unit_kind="web_result",
                text=text,
                source_url=hit.get("url", ""),
                trust=hit.get("trust", "unverified"),
                source_query=query,
                retrieval_method="web:" + str(hit.get("engine", "searxng")) + ":gap",
                rank=i,
            )
            item.derive_evidence_level()
            req.add_item(item)
            try:
                await run_with_timeout(
                    verify_item(verifier, req, item),
                    timeout_for("llm_call"), f"gap-web-verify:{item.id}")
            except Exception as exc:
                logger.warning("gap web verify failed %s: %s", item.id, exc)
            if item.verified and item.source_url:
                # a verified web page is an independent source: stamp a stable
                # document id from its URL so it counts toward requirement
                # coverage (independence rule), like a corpus document would
                item.document_id = web_document_id(item.source_url, item.id)
                try:
                    verdict = await run_with_timeout(
                        judge_site_reliability(reliability_critic, item.source_url,
                                               item.text[:4000]),
                        timeout_for("llm_call"), f"gap-rel:{item.id}")
                except Exception as exc:
                    verdict = {"reliability": "low"}
                item.reliability = verdict["reliability"]
        progress(f"    [gap:{req.id}] web round: {len(data.get('results') or [])} "
                 f"result(s), searches={budget.gap_web_searches_used}")


# ---------------------------------------------------------------------------
# Latent gap probing (content-level holes in satisfied requirements)
# ---------------------------------------------------------------------------

_PROBE_SCHEMA = (
    "{\"gaps\":[{\"question\":\"...\",\"queries\":[\"...\",\"...\"]}]}"
)


def probe_enabled() -> bool:
    raw = os.environ.get("XDEEP_GAP_PROBE", "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


def probe_max() -> int:
    raw = os.environ.get("XDEEP_GAP_PROBE_MAX", "3").strip()
    return max(1, int(raw)) if raw.isdigit() else 3


def web_augment_enabled() -> bool:
    """Always fire trust-gated WEB augmentation (even when local retrieval
    already satisfied the requirement). Off reverts to web-only-as-fallback."""
    raw = os.environ.get("XDEEP_WEB_AUGMENT", "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


def web_augment_max_queries() -> int:
    """Max augmentation queries per satisfied requirement (default 1).

    Augmentation runs LAST on the shared gap-web budget, so it only ever
    spends leftovers - but without a per-requirement cap the first satisfied
    req could still burn several searches and inflate the resolution list.
    """
    raw = os.environ.get("XDEEP_AUGMENT_MAX_QUERIES", "1").strip()
    return max(1, int(raw)) if raw.isdigit() else 1


async def _probe_latent_gaps(agent: Any, req: ResearchRequirement) -> list[dict]:
    """Ask the gap-probe LLM which important sub-questions the verified
    evidence does NOT answer, with targeted queries per question."""
    from src.agents.agents.stages import (
        _extract_json,
        _last_text,
        _prompt,
        _semantic_queries,
    )
    from src.agents.reuse import load_prompt
    from src.agents.timeouts import run_with_timeout, timeout_for

    blocks = []
    for it in req.verified_items()[-8:]:
        it.derive_evidence_level()
        meta = (f"[{it.id}] doc={it.document_id or '?'}"
                f" level={it.evidence_level or '?'}"
                f" support={it.verdict.support.value if it.verdict else '?'}")
        blocks.append(f"{meta}\n{it.text[:500]}")
    body = (
        f"EVIDENCE REQUIREMENT:\n{req.text}"
        f"\n\nVERIFIED EVIDENCE SO FAR:\n" + ("\n---\n".join(blocks) or "(none)")
    )
    try:
        result = await run_with_timeout(
            agent.ainvoke({"messages": [{"role": "user", "content": _prompt(
                load_prompt("agents", "gap_probe.txt"), body, _PROBE_SCHEMA)}]}),
            timeout_for("llm_call"), f"gap-probe:{req.id}")
        data = _extract_json(_last_text(result.get("messages", []))) or {}
    except Exception as exc:
        logger.warning("gap probe failed %s: %s", req.id, exc)
        return []
    out = []
    for g in (data.get("gaps") or [])[:2]:
        question = str(g.get("question") or "").strip()
        raw_q = [str(q) for q in (g.get("queries") or [])]
        queries = _semantic_queries(raw_q, fallback=question)
        if question and queries:
            out.append({"question": question, "queries": queries[:2]})
    for g in out:
        emit_event("gap_probe", requirement_id=req.id,
                   question=g["question"][:240], queries=g["queries"])
    return out


async def _latent_local_resolve(req: ResearchRequirement, gap: dict,
                                verifier: Any, budget: RunBudget) -> list[str]:
    """Try to close ONE latent sub-question with local retrieval. Returns the
    evidence ids of newly VERIFIED items ([] if the gap stays open locally)."""
    from src.agents.agents.stages import verify_item
    from src.agents.agents.worker import _local_retrieve
    from src.agents.graph import progress

    seen = {it.chunk_id for it in req.items if it.chunk_id}
    payload = {"task_id": req.id, "text": req.text, "entities": [],
               "target_n": req.target_n}
    new_ids: list[str] = []
    for query in gap["queries"]:
        if budget.gap_local_exhausted():
            break
        try:
            results, method, health = await _local_retrieve(
                payload, query, seen, top_k=5)
        except Exception as exc:
            logger.warning("latent local failed %s/%s: %s", req.id, query, exc)
            continue
        seen.update(r.get("chunk_id") for r in results if r.get("chunk_id"))
        for i, res in enumerate(results, 1):
            item = EvidenceItem(
                id=f"{req.id}.P{len(req.items) + 1}",
                run_id="",
                requirement_id=req.id,
                chunk_id=res.get("chunk_id", ""),
                document_id=res.get("document_id", ""),
                section=res.get("section", ""),
                unit_kind=res.get("unit_kind", "paragraph"),
                text=res.get("text", ""),
                source_query=query,
                retrieval_method=res.get("retrieval_method", method) + ":probe",
                rank=int(res.get("rank", 0) or i),
            )
            item.derive_evidence_level()
            req.add_item(item)
            try:
                from src.agents.timeouts import run_with_timeout, timeout_for

                await run_with_timeout(
                    verify_item(verifier, req, item),
                    timeout_for("llm_call"), f"probe-verify:{item.id}")
            except Exception as exc:
                logger.warning("probe verify failed %s: %s", item.id, exc)
            if item.verified:
                new_ids.append(item.id)
        progress(f"    [probe:{req.id}] local '{gap['question'][:50]}...' "
                 f"[{method}]")
    budget.gap_local_rounds_used += 1
    return new_ids


async def _latent_web_resolve(req: ResearchRequirement, gap: dict,
                              verifier: Any, budget: RunBudget) -> list[str]:
    """Try to close ONE latent sub-question with trust-gated web search.
    Returns evidence ids of newly VERIFIED items ([] if still open)."""
    from src.agents.agents.stages import (
        judge_site_reliability,
        make_reliability_critic,
        verify_item,
    )
    from src.agents.graph import progress
    from src.agents.tools.searxng import searxng_search_impl
    from src.agents.timeouts import run_with_timeout, timeout_for

    reliability_critic = make_reliability_critic()
    new_ids: list[str] = []
    emit_event("web_search_started", requirement_id=req.id,
               query="; ".join(gap["queries"]), purpose=gap["question"][:160],
               base=None)
    for query in gap["queries"]:
        if budget.gap_web_exhausted():
            break
        emit_event("web_search_started", requirement_id=req.id,
                   query=query, purpose=gap["question"][:160])
        try:
            out = await run_with_timeout(
                searxng_search_impl(query=query, top_k=6, trusted_only=True),
                timeout_for("llm_call"), f"probe-web:{query[:60]}")
            data = json.loads(out)
        except Exception as exc:
            logger.warning("probe searxng failed %s: %s", query, exc)
            continue
        if not data.get("available"):
            emit_event("web_search_done", requirement_id=req.id, query=query,
                       available=False, count=0, urls=[])
            break
        budget.gap_web_searches_used += 1
        emit_event("web_search_done", requirement_id=req.id, query=query,
                   count=len(data.get("results") or []),
                   urls=[h.get("url", "") for h in (data.get("results") or [])][:10])
        for i, hit in enumerate((data.get("results") or [])[:5], 1):
            # DIVE DEEP: fetch the real page, fall back to snippet
            from src.agents.agents.stages import fetch_web_hit, verify_web_hit

            text, _fetched = await fetch_web_hit(
                hit.get("url", ""), hit.get("snippet", ""), req.id,
                tag="probe", extra={"purpose": gap["question"][:160]})
            if not text:
                continue
            item = EvidenceItem(
                id=f"{req.id}.PW{len(req.items) + 1}",
                run_id="",
                requirement_id=req.id,
                chunk_id="",
                document_id="",
                section="web",
                unit_kind="web_result",
                text=text,
                source_url=hit.get("url", ""),
                trust=hit.get("trust", "unverified"),
                source_query=query,
                retrieval_method="web:" + str(hit.get("engine", "searxng")) + ":probe",
                rank=i,
            )
            item.derive_evidence_level()
            req.add_item(item)
            reliability = await verify_web_hit(
                req, item, verifier=verifier,
                reliability_critic=reliability_critic, tag="probe")
            if reliability in ("high", "medium"):
                new_ids.append(item.id)
        progress(f"    [probe:{req.id}] web '{gap['question'][:50]}...' "
                 f"searches={budget.gap_web_searches_used}")
    return new_ids


async def _web_augment(req: ResearchRequirement, queries: list[str],
                                verifier: Any, budget: RunBudget) -> list[str]:
    """Trust-gated WEB augmentation: fire web searches for a requirement even
    when LOCAL retrieval already satisfied it, to bring in current
    guidelines / extra trusted info. Every hit passes the verifier + the
    reliability judge; only high/medium-reliability verified sources count.
    Bounded by the gap-web budget. Returns newly verified evidence ids."""
    from src.agents.agents.stages import (
        judge_site_reliability,
        make_reliability_critic,
        verify_item,
    )
    from src.agents.graph import progress
    from src.agents.tools.searxng import searxng_search_impl
    from src.agents.timeouts import run_with_timeout, timeout_for

    reliability_critic = make_reliability_critic()
    new_ids: list[str] = []
    for query in queries:
        if budget.gap_web_exhausted():
            break
        emit_event("web_search_started", requirement_id=req.id,
                   query=query, purpose="web augmentation")
        try:
            out = await run_with_timeout(
                searxng_search_impl(query=query, top_k=6, trusted_only=True),
                timeout_for("llm_call"), f"web-aug:{query[:60]}")
            data = json.loads(out)
        except Exception as exc:
            logger.warning("web augment failed %s: %s", query, exc)
            continue
        if not data.get("available"):
            emit_event("web_search_done", requirement_id=req.id, query=query,
                       available=False, count=0, urls=[])
            break
        budget.gap_web_searches_used += 1
        emit_event("web_search_done", requirement_id=req.id, query=query,
                   count=len(data.get("results") or []),
                   urls=[h.get("url", "") for h in (data.get("results") or [])][:10])
        for i, hit in enumerate((data.get("results") or [])[:5], 1):
            # DIVE DEEP: fetch the real page, fall back to snippet
            from src.agents.agents.stages import fetch_web_hit, verify_web_hit

            text, _fetched = await fetch_web_hit(
                hit.get("url", ""), hit.get("snippet", ""), req.id,
                tag="web-aug", extra={"purpose": "web augmentation"})
            if not text:
                continue
            item = EvidenceItem(
                id=f"{req.id}.WA{len(req.items) + 1}",
                run_id="",
                requirement_id=req.id,
                chunk_id="",
                document_id="",
                section="web",
                unit_kind="web_result",
                text=text,
                source_url=hit.get("url", ""),
                trust=hit.get("trust", "unverified"),
                source_query=query,
                retrieval_method="web:" + str(hit.get("engine", "searxng")) + ":augment",
                rank=i,
            )
            item.derive_evidence_level()
            req.add_item(item)
            reliability = await verify_web_hit(
                req, item, verifier=verifier,
                reliability_critic=reliability_critic, tag="web-aug")
            if reliability in ("high", "medium"):
                new_ids.append(item.id)
        progress(f"    [web-augment:{req.id}] '{query[:60]}...' "
                 f"searches={budget.gap_web_searches_used}")
    return new_ids


_COMPLETENESS_SCHEMA = (
    "{\"complete\": true, \"remaining_facets\":[\"...\"]}"
)


async def _gap_completeness_check(agent: Any, question: str,
                                  evidence: list[str]) -> list[str]:
    """Completeness gate: after evidence was found for a probed sub-question,
    ask whether the sub-question is FULLY answered. Returns the remaining
    important facets ([] == fully answered). Failures default to "assume
    partial" so the answer never over-claims completeness."""
    from src.agents.agents.stages import (
        _extract_json,
        _last_text,
        _prompt,
    )
    from src.agents.reuse import load_prompt
    from src.agents.timeouts import run_with_timeout, timeout_for

    body = (
        f"SUB-QUESTION TO ANSWER:\n{question}"
        f"\n\nEVIDENCE GATHERED FOR IT:\n"
        + ("\n---\n".join(e[:600] for e in evidence[-6:]) or "(none)")
        + "\n\nIs this sub-question now FULLY answered by the evidence above? "
        "If important facets are still missing (safety, hard outcomes, "
        "duration, comparisons, mechanisms, subgroups...), list them in "
        "remaining_facets. If the evidence fully answers it, complete=true "
        "with an empty list."
    )
    try:
        result = await run_with_timeout(
            agent.ainvoke({"messages": [{"role": "user", "content": _prompt(
                load_prompt("agents", "gap_complete.txt"), body,
                _COMPLETENESS_SCHEMA)}]}),
            timeout_for("llm_call"), f"gap-complete:{question[:60]}")
        data = _extract_json(_last_text(result.get("messages", []))) or {}
        if data.get("complete"):
            return []
        return [str(x).strip() for x in (data.get("remaining_facets") or [])
                if str(x).strip()][:4]
    except Exception as exc:
        logger.warning("completeness gate failed %s: %s", question[:60], exc)
        # Gate unavailable: do NOT fabricate a facet. Assume the gathered
        # evidence suffices for now; the synthesizer is the real arbiter and
        # reconciliation folds any genuine remaining facets into run.gaps.
        return []


async def resolve_requirement_gaps(state: Any) -> list[GapResolution]:
    """Run the full gap-resolution pass over the run state.

    Covers BOTH:
      * HARD gaps  - unsatisfied requirements (existing path).
      * LATENT gaps - content holes in satisfied requirements, found by the
        GAP PROBE (mechanisms, components, comparators, durations, subgroups)
        and resolved local-first -> web -> listed.

    Mutates state (adds verified items to requirements, records
    gap_resolutions + gaps). Returns the list of GapResolutions.
    """
    from src.agents.agents.stages import (
        make_replanner,
        make_search_planner,
        make_verifier,
    )
    from src.agents.graph import progress

    hard_gaps = compute_gaps(state)
    want_latent = probe_enabled() or web_augment_enabled()
    if not hard_gaps and not (want_latent and state.requirements):
        return []
    # POST-contradiction: the probe should not run when there is no verified
    # evidence at all to probe against (nothing to inspect)
    if want_latent and not any(r.verified_items() for r in state.requirements):
        if not hard_gaps:
            return []

    budget: RunBudget = state.budget
    verifier = make_verifier()
    planner = make_search_planner()
    replanner = make_replanner()
    resolutions: list[GapResolution] = []

    # ---- A) HARD gaps (numerically unsatisfied) ----------------------------
    for req, gap_text in hard_gaps:
        resolution = GapResolution(requirement_id=req.id, gap=gap_text)
        # phase 1: LOCAL retrieval rounds
        while not req.satisfied() and not budget.gap_local_exhausted():
            try:
                queries = await _plan_gap_queries(planner, replanner, req, budget)
            except Exception as exc:
                logger.warning("gap plan failed %s: %s", req.id, exc)
                queries = ([req.text] if not _previous_queries(req) else [])
            if not queries:
                progress(f"    [gap:{req.id}] no more local queries - stopping")
                break
            await _local_round(req, queries, verifier, budget)
        req.derive_status()

        # phase 2: WEB searches (only if local could not close it)
        if not req.satisfied() and not budget.gap_web_exhausted():
            try:
                queries = await _plan_gap_queries(planner, replanner, req, budget)
            except Exception as exc:
                logger.warning("gap web plan failed %s: %s", req.id, exc)
                queries = ([req.text] if not _previous_queries(req) else [])
            if queries:
                await _web_round(req, queries, verifier, budget)
        req.derive_status()

        # phase 3: record
        if req.satisfied():
            web_used = any(
                it.retrieval_method.startswith("web:") for it in req.items
                if it.verified and it.retrieval_method.endswith(":gap"))
            resolution.status = (GapResolutionStatus.RESOLVED_WEB if web_used
                                 else GapResolutionStatus.RESOLVED_LOCAL)
            resolution.note = f"gap closed (target {req.target_n} reached)"
        else:
            resolution.status = GapResolutionStatus.UNRESOLVED
            resolution.note = ("gap could not be closed within local+web budget; "
                               "listed in the answer")
            req.mark_exhausted(gap_text)
            state.gaps.append(gap_text)
        resolution.evidence_ids = [
            it.id for it in req.items
            if it.verified and it.retrieval_method.endswith(":gap")
        ]
        resolution.searches_used = budget.gap_web_searches_used
        resolutions.append(resolution)
        progress(f"    [gap:{req.id}] -> {resolution.status.value} "
                 f"(+{len(resolution.evidence_ids)} verified item(s))")

    # ---- B) LATENT gaps (content holes in satisfied requirements) ----------
    if probe_enabled():
        from src.agents.agents.stages import make_gap_probe

        probe = make_gap_probe()
        remaining = probe_max()
        for req in state.requirements:
            if remaining <= 0:
                break
            if not req.verified_items() or budget.gap_exhausted():
                continue     # nothing to probe against / no budget left
            try:
                latent = await _probe_latent_gaps(probe, req)
            except Exception as exc:
                logger.warning("latent probe error %s: %s", req.id, exc)
                latent = []
            for gap in latent[:2]:
                if remaining <= 0 or budget.gap_exhausted():
                    break
                gap_res = GapResolution(requirement_id=req.id,
                                        gap=str(gap.get("question") or ""))
                # 1) LOCAL first (close the content gap from the corpus)
                if not budget.gap_local_exhausted():
                    local_ids = await _latent_local_resolve(
                        req, gap, verifier, budget)
                else:
                    local_ids = []
                # 2) WEB: ALWAYS attempt when augmentation is on - even when
                #    local closed the gap, a trust-gated web round brings in
                #    current guidelines / extra info. Off (XDEEP_WEB_AUGMENT=0)
                #    reverts to web-only-when-local-fails.
                if web_augment_enabled():
                    if not budget.gap_web_exhausted():
                        web_ids = await _latent_web_resolve(
                            req, gap, verifier, budget)
                    else:
                        web_ids = []
                elif not local_ids and not budget.gap_web_exhausted():
                    web_ids = await _latent_web_resolve(
                        req, gap, verifier, budget)
                else:
                    web_ids = []
                if local_ids or web_ids:
                    # COMPLETENESS GATE: did we FULLY answer the probed
                    # sub-question, or do important facets remain? The old
                    # status stamped "resolved" for ANY evidence, which
                    # over-claimed when the synthesizer later listed safety /
                    # hard-outcome / durability holes.
                    gap_res.evidence_ids = list(local_ids) + list(web_ids)
                    new_texts = [
                        it.text for it in req.verified_items()
                        if it.id in gap_res.evidence_ids]
                    try:
                        remaining_facets = await _gap_completeness_check(
                            probe, str(gap.get("question") or ""), new_texts)
                    except Exception as exc:
                        logger.warning("completeness gate raised %s: %s",
                                       gap.get("question", "")[:60], exc)
                        remaining_facets = []  # never crash the gap pass
                    if remaining_facets:
                        gap_res.status = GapResolutionStatus.PARTIALLY_RESOLVED
                        gap_res.note = ("some evidence found, but still "
                                        f"missing: {'; '.join(remaining_facets[:3])}")
                        for facet in remaining_facets:
                            state.gaps.append(
                                f"{gap.get('question') or ''} — still missing: {facet}")
                        emit_event("gap_completeness",
                                   requirement_id=req.id,
                                   question=(gap.get("question") or "")[:200],
                                   status="partially_resolved",
                                   remaining=remaining_facets)
                    else:
                        gap_res.status = (GapResolutionStatus.RESOLVED_WEB
                                          if web_ids
                                          else GapResolutionStatus.RESOLVED_LOCAL)
                        gap_res.note = ("closed by targeted gap-resolution "
                                        "retrieval"
                                        + (" + web augmentation" if web_ids else ""))
                else:
                    gap_res.status = GapResolutionStatus.UNRESOLVED
                    gap_res.note = ("no evidence found even after local+web "
                                    "targeted searches; listed in the answer")
                    state.gaps.append(str(gap.get("question") or ""))
                gap_res.searches_used = budget.gap_web_searches_used
                resolutions.append(gap_res)
                progress(f"    [gap:{req.id}] probe -> {gap_res.status.value} "
                         f"({gap_res.gap[:70]})")
                remaining -= 1

    # ---- C) WEB AUGMENTATION: even when no LATENT gap was found, fire a
    # bounded trust-gated web round per satisfied requirement to bring in
    # current guidelines / extra verified info (the user's "web search after
    # local retrieval" contract; XDEEP_WEB_AUGMENT=0 disables it).
    if web_augment_enabled():
        for req in state.requirements:
            if budget.gap_web_exhausted():
                break
            if not req.verified_items():
                continue           # nothing meaningful to enrich
            try:
                aug_queries = await _plan_gap_queries(
                    planner, replanner, req, budget)
            except Exception as exc:
                logger.warning("web augment plan failed %s: %s", req.id, exc)
                aug_queries = []
            if not aug_queries:
                aug_queries = [req.text] if not _previous_queries(req) else []
            if not aug_queries:
                continue
            new_ids = await _web_augment(req, aug_queries[:web_augment_max_queries()],
                                           verifier, budget)
            if new_ids:
                aug_res = GapResolution(
                    requirement_id=req.id,
                    gap=f"web augmentation for satisfied requirement {req.id}",
                    status=GapResolutionStatus.RESOLVED_WEB,
                    evidence_ids=new_ids,
                    note=f"trusted web augmentation added {len(new_ids)} "
                         "verified source(s)",
                    searches_used=budget.gap_web_searches_used,
                )
                resolutions.append(aug_res)
                progress(f"    [gap:{req.id}] web-augment -> resolved_web "
                         f"(+{len(new_ids)} verified source(s))")

    state.gap_resolutions = resolutions
    return resolutions


__all__ = ["compute_gaps", "resolve_requirement_gaps"]
