"""Stage agents for the research graph (deepagents runtime).

MIRRORS the existing agentic-v3 architecture and its EXACT prompts, but under
an autonomy-oriented deepagents runtime:

  ORCHESTRATOR (master.txt, gemma)
    -> per-task WORKER loop (gemma worker with tools):
         UMLS enrichment (umls_lookup tool)
         SEARCH PLANNER (search_planner.txt) round 1 -> queries [autonomy]
         retriever / searxng tool loop (worker chooses)      [autonomy]
         VERIFIER critic (critic.txt, mistral paced) - HARD gate
         REPLANNER (replanner.txt) rounds 2+ -> new queries
         DEEP INSPECTOR (deep_inspector.txt) on promising docs
       until satisfied or budget exhausted
  CONTRADICTION (contradiction.txt, mistral)
  RESOLUTION (resolution.txt, mistral + search tools)
  SYNTHESIS (synthesize.txt, gemma)

The workflow ORDER is fixed by the graph; the deep agents decide HOW within
each stage (which tools, which queries, whether to inspect deeper).
"""

from __future__ import annotations

import json
import re
from typing import Any

from deepagents import create_deep_agent

from src.agents.config import build_model_for_role
from src.agents.reuse import load_prompt


def _strip_thought_blocks(text: str) -> str:
    """Remove model thinking tags (gemma) and leading prose before JSON."""
    import re

    text = re.sub(r"<\s*/?\s*(thinking|thought|scratchpad|reasoning)\s*>", "",
                  text, flags=re.IGNORECASE)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1] or [-1])
    if start > 0 and text[:start].count("{") == 0 and text[:start].count("[") == 0:
        text = text[start:]
    return text.strip()


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction (handles fences and thought blocks)."""
    if not text:
        raise ValueError("empty model output")
    t = _strip_thought_blocks(text)
    fence = chr(96) * 3
    lines = [ln for ln in t.splitlines() if not ln.strip().startswith(fence)]
    t = "\n".join(lines).strip()
    start = min([i for i in (t.find("{"), t.find("[")) if i != -1] or [-1])
    if start == -1:
        return json.loads(t)
    depth = 0
    for i in range(start, len(t)):
        if t[i] in "[{":
            depth += 1
        elif t[i] in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError(f"unbalanced JSON in output: {t[:200]!r}")


def _last_text(messages: list) -> str:
    for m in reversed(messages):
        content = getattr(m, "content", m)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _prompt(base: str, body: str, schema: str = "") -> str:
    out = base + "\n\n" + body
    if schema:
        out += "\n\nReturn EXACTLY ONE JSON object:\n" + schema
        out += "\nEmit ONLY the JSON. No markdown fences, no thinking/thought blocks, no prose."
    return out


# ---------------------------------------------------------------------------
# ORCHESTRATOR (master.txt)
# ---------------------------------------------------------------------------

class DecomposedReq:
    """A single evidence obligation from the decompose stage."""
    __slots__ = ("id", "text", "entities", "target_n")

    def __init__(self, id: str, text: str, entities=None, target_n: int = 3):
        self.id = id
        self.text = text
        self.entities = entities or []
        self.target_n = target_n


def make_orchestrator(model: Any = None):
    """Orchestrator: master.txt on Gemma (thinking, decomposition)."""
    return create_deep_agent(
        model=model or build_model_for_role("decompose"),
        system_prompt=load_prompt("agents", "master.txt"),
        tools=[],
    )


_MASTER_SCHEMA = (
    "{\"requirements\":[{\"id\":\"R1\",\"text\":\"<one evidence obligation>\","
    "\"entities\":[\"<entity1>\"],\"target_n\":3}]}"
)


async def _decompose_call(agent: Any, question: str, strict: bool = False,
                      memory_context: str = "") -> list[DecomposedReq]:
    """One decompose LLM call. strict=True appends a hard JSON-only nudge used
    on the retry path (Gap F - never silently collapse a broken parse)."""
    body = "USER QUESTION:\n" + question
    if memory_context.strip():
        # advisory-only prior-research context (never evidence): mirrors the
        # v3 master prompt contract - plan ONLY the new question.
        body += (
            "\n\nPRIOR RESEARCH MEMORY (advisory only - do not cite as "
            "evidence, plan only the new question above):\n" + memory_context.strip()
        )
    if strict:
        body += (
            "\n\nYour previous attempt did not produce valid JSON. This is "
            "your FINAL attempt: emit EXACTLY ONE JSON object matching the "
            "schema, no markdown fences, no thinking blocks, no prose, no "
            "comments. If the question genuinely needs only one task, return "
            "one task in that exact schema."
        )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": _prompt(
            load_prompt("agents", "master.txt"), body, _MASTER_SCHEMA)}]}
    )
    data = _extract_json(_last_text(result.get("messages", [])))
    if isinstance(data, dict) and "requirements" in data:
        data = data["requirements"]
    if not isinstance(data, list):
        raise ValueError("decompose returned no requirements list")
    out: list[DecomposedReq] = []
    for i, r in enumerate(data, 1):
        out.append(DecomposedReq(
            id=str(r.get("id") or f"R{i}"),
            text=str(r.get("text") or "").strip(),
            entities=[str(e) for e in (r.get("entities") or [])],
            target_n=int(r.get("target_n") or 3),
        ))
    return [r for r in out if r.text]


async def decompose_requirements(agent: Any, question: str,
                               memory_context: str = "") -> list[DecomposedReq]:
    """Decompose the question into requirements (Gap F: retry once, fail loud)."""
    try:
        return await _decompose_call(agent, question, strict=False,
                                     memory_context=memory_context)
    except Exception:
        # one strict retry - a JSON parse failure must not silently degrade
        # a multi-part question into a single task
        return await _decompose_call(agent, question, strict=True,
                                     memory_context=memory_context)



# ---------------------------------------------------------------------------
# SEARCH PLANNER (search_planner.txt) - round 1
# ---------------------------------------------------------------------------

def make_search_planner(model: Any = None):
    """Search-term planner: search_planner.txt on GEMMA, WITH umls_lookup so
    it can broaden terminology autonomously before proposing queries."""
    from src.agents.tools.umls import umls_lookup

    return create_deep_agent(
        model=model or build_model_for_role("research"),
        system_prompt=load_prompt("agents", "search_planner.txt"),
        tools=[umls_lookup],
    )


_SEARCH_SCHEMA = (
    "{\"rationale\":\"...\",\"requirements\":["
    "{\"requirement_id\":\"R1\",\"rationale\":\"...\","
    "\"queries\":[\"q1\",\"q2\",\"q3\"]}]}"
)



_SEMANTIC_NOTICE = (
    "\n\nRETRIEVAL ENGINE IS SEMANTIC (dense vector + BM25 hybrid, MedCPT)."
    " Write each query as a natural-language phrase/sentence a clinician would"
    " type - e.g. \"does vitamin D supplementation lower systolic blood pressure"
    " in older adults?\". Do NOT write keyword lists, and do NOT use boolean"
    " operators (AND / OR / NOT) or quoted-term conjunctions. Each query must"
    " read fluently and capture the whole relationship (population + exposure +"
    " outcome) in prose. Keep it under ~15 words so dense encoding stays sharp."
)


def _sanitize_semantic_query(query: str) -> str:
    """Turn boolean/keyword queries into semantic-friendly natural phrasing.

    The retriever is SEMANTIC (MedCPT dense + BM25 hybrid): "Cardiac arrest
    AND hypertension" tokenizes into a keyword bag and ruins dense similarity.
    Deterministically: drop boolean operators (AND/OR/NOT) as standalone words,
    unwrap quoted terms, collapse whitespace, strip dangling operators.
    """
    q = " ".join((query or "").split()).strip()
    if not q:
        return ""
    q = q.replace('"', " ").replace("\u201c", " ").replace("\u201d", " ")
    q = re.sub(r"\b(AND|OR|NOT)\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"^(and|or|not)\s+", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+(and|or|not)$", "", q, flags=re.IGNORECASE)
    return q[:240]


def _semantic_queries(raw: list, fallback: str = "") -> list:
    """Sanitize every raw query to semantic phrasing; fall back if empty."""
    out: list = []
    for q in raw:
        s = _sanitize_semantic_query(q)
        if len(s) >= 7 and s not in out:
            out.append(s)
    if not out and fallback:
        out = [_sanitize_semantic_query(fallback)]
    return out[:4]


def search_plan_prompt(task: str, requirement: str, pool: list[str],
                       previous_queries=None, critic_notes: str = "") -> str:
    body = f"TASK OBJECTIVE:\n{task}\n\nEVIDENCE REQUIREMENT:\n{requirement}"
    if pool:
        body += "\n\nUMLS TERMINOLOGY POOL:\n" + "\n".join("- " + t for t in pool[:20])
    if previous_queries:
        body += "\n\nPREVIOUS QUERIES (must not repeat):\n" + "\n".join(
            "- " + q for q in previous_queries)
    if critic_notes:
        body += "\n\nCRITIC REJECTION NOTES:\n" + critic_notes
    body += _SEMANTIC_NOTICE
    return _prompt(load_prompt("agents", "search_planner.txt"), body, _SEARCH_SCHEMA)


async def plan_queries(agent: Any, task: str, requirement_text: str,
                       pool=None, previous_queries=None,
                       critic_notes: str = "") -> list[str]:
    """Return 2-3 targeted queries for ONE requirement (round 1 planner)."""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": search_plan_prompt(
            task, requirement_text, pool or [], previous_queries, critic_notes)}]}
    )
    data = _extract_json(_last_text(result.get("messages", []))) or {}
    reqs = data.get("requirements") or []
    queries: list[str] = []
    for r in reqs:
        queries.extend(str(q) for q in (r.get("queries") or []))
    if not queries:
        queries = [requirement_text]
    return _semantic_queries(queries, fallback=requirement_text)


# ---------------------------------------------------------------------------
# GAP PROBE (gap_probe.txt) - content-level holes in satisfied requirements
# ---------------------------------------------------------------------------

def make_gap_probe(model: Any = None):
    """Latent-gap probe: examines a requirement + its verified evidence and
    proposes up to 2 unanswered sub-questions with targeted queries, so the
    gap pass can search for them even when the requirement is numerically
    satisfied."""
    return create_deep_agent(
        model=model or build_model_for_role("research"),
        system_prompt=load_prompt("agents", "gap_probe.txt"),
        tools=[],
    )


# ---------------------------------------------------------------------------
# REPLANNER (replanner.txt) - rounds 2+
# ---------------------------------------------------------------------------

def make_replanner(model: Any = None):
    """Failure-analysis replanner: replanner.txt on GEMMA."""
    return create_deep_agent(
        model=model or build_model_for_role("research"),
        system_prompt=load_prompt("agents", "replanner.txt"),
        tools=[],
    )


_REPLAN_SCHEMA = (
    "{\"diagnosis\":\"...\",\"missing_evidence\":\"...\","
    "\"strategy\":\"...\",\"queries\":[\"...\"]}"
)




_STOPWORDS = frozenset({
    "the", "and", "for", "with", "without", "from", "into", "onto", "over",
    "under", "versus", "vs", "in", "on", "at", "of", "to", "a", "an", "or",
    "but", "is", "are", "was", "were", "does", "do", "did", "effect",
    "effects", "lower", "reduction", "reduce", "reduces", "reducing",
    "blood", "pressure", "supplementation", "supplement", "associated",
    "association", "meta", "analysis", "randomized", "controlled", "trial",
    "vitamin", "d", "study", "studies", "evidence",
})


def _content_tokens(query: str) -> set:
    """Distinct content tokens of a query (lowercased, stopwords removed)."""
    import re

    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", (query or "").lower())
            if t not in _STOPWORDS and len(t) > 2}


def meaningfully_different(queries: list, tried: list) -> list:
    """Keep only queries with at least one token not covered by any tried query.

    Mirrors the legacy replanner deterministic filter: a new query must contain
    >=1 content token absent from EVERY tried query, and must not be an exact
    casefold duplicate. Fixed in code - never trusted to the LLM.
    """
    tried_tokens = [_content_tokens(q) for q in tried if q]
    tried_exact = {q.casefold() for q in tried if q}
    out: list = []
    seen: set = set()
    for raw in queries:
        q = " ".join((raw or "").split()).strip()
        if not q or len(q) < 7 or q.casefold() in tried_exact:
            continue
        toks = _content_tokens(q)
        if not toks or any(toks <= covered for covered in tried_tokens):
            continue
        key = tuple(sorted(toks))
        if key in seen:
            continue
        seen.add(key)
        out.append(q[:240])
        if len(out) >= 3:
            break
    return out

def replan_prompt(task: str, requirement: str, previous_queries: list[str],
                  retrieved: list[dict], rejected: list[dict],
                  accepted: list[str]) -> str:
    body = (
        f"TASK OBJECTIVE:\n{task}\n\nEVIDENCE REQUIREMENT:\n{requirement}"
        f"\n\nPREVIOUS QUERIES TRIED:\n" + "\n".join("- " + q for q in previous_queries)
    )
    if retrieved:
        body += "\n\nRETRIEVED DOCUMENTS (id + section + excerpt):\n" + "\n".join(
            f"[{d.get('document_id', '?')} | {d.get('section', '')}]: "
            f"{str(d.get('excerpt') or d.get('text') or '')[:200]}" for d in retrieved[:12])
    if rejected:
        body += "\n\nREJECTED BY CRITIC (why each did NOT answer):\n" + "\n".join(
            f"- {r.get('note', '')}" for r in rejected[:12])
    if accepted:
        body += "\n\nACCEPTED PAPERS SO FAR:\n" + "\n".join("- " + str(a) for a in accepted)
    body += _SEMANTIC_NOTICE
    return _prompt(load_prompt("agents", "replanner.txt"), body, _REPLAN_SCHEMA)


async def replan_queries(agent: Any, task: str, requirement_text: str,
                         previous_queries: list[str], retrieved: list[dict],
                         rejected: list[dict], accepted: list[str]) -> list[str]:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": replan_prompt(
            task, requirement_text, previous_queries, retrieved, rejected, accepted)}]}
    )
    data = _extract_json(_last_text(result.get("messages", []))) or {}
    queries = [str(q) for q in (data.get("queries") or [])]
    # semantic sanitization FIRST (strip boolean operators / keyword lists),
    # then the deterministic not-repeated guard (the LLM is not trusted here)
    queries = _semantic_queries(queries, fallback="")
    return meaningfully_different(queries, previous_queries)


# ---------------------------------------------------------------------------
# VERIFIER (critic.txt)
# ---------------------------------------------------------------------------

def make_verifier(model: Any = None):
    """Verifier critic: critic.txt on MISTRAL (paced 1 req / 1.5s)."""
    return create_deep_agent(
        model=model or build_model_for_role("verify"),
        system_prompt=load_prompt("agents", "critic.txt"),
        tools=[],
    )


def _verifier_prompt(req_text, requirement_id, evidence_id, passage) -> str:
    return _prompt(
        load_prompt("agents", "critic.txt"),
        f"EVIDENCE REQUIREMENT ({requirement_id}):\n{req_text}"
        f"\n\nPASSAGE TO JUDGE ({evidence_id}):\n{passage}",
    )


async def verify_item(agent: Any, requirement: Any, item: Any) -> Any:
    """Run the critic on ONE item vs ONE requirement -> set the verdict."""
    from src.agents.logging import xdeep_log
    from src.agents.state import VerifierVerdict

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": _verifier_prompt(
            requirement.text, item.requirement_id, item.id, item.text)}]}
    )
    data = _extract_json(_last_text(result.get("messages", []))) or {}
    verdict = VerifierVerdict(
        evidence_id=item.id,
        requirement_id=item.requirement_id,
        relevance=str(data.get("relevance", "not_relevant")),
        answers_task=str(data.get("answers_task", "no")),
        support=str(data.get("support", "neutral")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        note=str(data.get("note", "")),
    )
    item.submit_to_verifier()
    item.set_verdict(verdict)
    if item.verdict:
        xdeep_log("verify_done", evidence_id=item.id, status=item.status.value,
                  support=item.verdict.support.value,
                  confidence=round(item.verdict.confidence, 3),
                  note=item.verdict.note[:200])
    return item




# ---------------------------------------------------------------------------
# RELIABILITY CRITIC (reliability.txt) - the "another agent" for web sites
# ---------------------------------------------------------------------------

def make_reliability_critic(model: Any = None):
    """Reliability critic: reliability.txt on MISTRAL (paced, short decision).

    This is the SEPARATE agent the user asked for: it judges the RELIABILITY
    of a fetched web source (authority, evidence presence, recency, scope,
    conflict-of-interest) - orthogonal to the verifier, which judges RELEVANCE
    to the requirement. Its verdict lands on EvidenceItem.reliability.
    """
    return create_deep_agent(
        model=model or build_model_for_role("verify"),
        system_prompt=load_prompt("agents", "reliability.txt"),
        tools=[],
    )


_RELIABILITY_SCHEMA = (
    "{\"reliability\":\"high|medium|low|unusable\","
    "\"authority\":\"...\",\"evidence\":\"...\","
    "\"recency\":\"...\",\"conflicts\":\"...\",\"note\":\"...\"}"
)


def _reliability_prompt(url: str, text: str) -> str:
    body = f"SOURCE URL:\n{url}\n\n" if url else "SOURCE URL: (none)\n\n"
    body += f"PAGE TEXT (fetched content, may be truncated):\n{text[:6000]}"
    return _prompt(load_prompt("agents", "reliability.txt"), body,
                   _RELIABILITY_SCHEMA)


async def judge_site_reliability(agent: Any, url: str, text: str) -> dict:
    """Judge ONE fetched web source; returns a reliability dict. Never raises
    - any failure yields a conservative 'low' verdict so an unreliable page
    can never be smuggled in as high-reliability evidence."""
    from src.agents.logging import xdeep_log

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": _reliability_prompt(url, text)}]}
        )
        data = _extract_json(_last_text(result.get("messages", []))) or {}
        reliability = str(data.get("reliability", "low")).lower()
        if reliability not in ("high", "medium", "low", "unusable"):
            reliability = "low"
    except Exception as exc:
        xdeep_log("reliability_critic_failed", url=url, error=str(exc)[:200])
        reliability = "low"
        data = {}
    out = {
        "reliability": reliability,
        "authority": str(data.get("authority", "")),
        "evidence": str(data.get("evidence", "")),
        "recency": str(data.get("recency", "")),
        "conflicts": str(data.get("conflicts", "")),
        "note": str(data.get("note", "")),
    }
    xdeep_log("reliability_verdict", url=url, reliability=reliability,
              note=out["note"][:200])
    return out


async def fetch_web_hit(url: str, snippet: str, requirement_id: str,
                        *, tag: str, extra: dict | None = None) -> tuple[str, bool]:
    """Fetch one web hit's real page text (snippet fallback).

    Emits web_fetch + progress. Returns (text, fetched_ok); ("", False)
    when the text is too short to judge. Never raises.
    """
    from src.agents.graph import emit_event
    from src.agents.graph import progress as _p
    from src.agents.timeouts import run_with_timeout, timeout_for
    from src.agents.tools.fetch import fetch_page_text

    snippet = (snippet or "").strip()
    try:
        page_text = await run_with_timeout(
            fetch_page_text(url or "", max_chars=8000),
            timeout_for("llm_call"), f"{tag}-fetch:{str(url or '')[:60]}")
    except Exception:
        page_text = ""
    text = (page_text or snippet).strip()
    if len(text) < 40:
        return "", False
    emit_event("web_fetch", requirement_id=requirement_id, url=url or "",
               chars=len(page_text) if page_text else 0,
               ok=bool(page_text), snippet_used=not page_text,
               **(extra or {}))
    if page_text:
        _p(f"        [web-fetch:{requirement_id}] {str(url or '')[:80]} "
           f"({len(page_text)} chars)")
    return text, bool(page_text)


async def verify_web_hit(req: Any, item: Any, *, verifier: Any,
                         reliability_critic: Any, tag: str) -> str:
    """Verifier gate + reliability judge for one fetched web item.

    Stamps a stable web document id when verified (so it counts toward
    requirement coverage) and emits reliability_verdict. Returns the
    item's reliability (""/low/medium/high). Never raises.
    """
    from src.agents.state import web_document_id
    from src.agents.timeouts import run_with_timeout, timeout_for

    try:
        await run_with_timeout(
            verify_item(verifier, req, item),
            timeout_for("llm_call"), f"{tag}-verify:{item.id}")
    except Exception as exc:
        _log_warning("web verify failed for %s: %s", item.id, exc)
    if item.verified and item.source_url:
        item.document_id = web_document_id(item.source_url, item.id)
        try:
            verdict = await run_with_timeout(
                judge_site_reliability(
                    reliability_critic, item.source_url, item.text[:4000]),
                timeout_for("llm_call"), f"{tag}-rel:{item.id}")
        except Exception:
            verdict = {"reliability": "low"}
        item.reliability = verdict["reliability"]
        from src.agents.graph import emit_event

        emit_event("reliability_verdict", requirement_id=req.id,
                   evidence_id=item.id, url=item.source_url[:160],
                   reliability=item.reliability,
                   note=verdict.get("note", "")[:200])
    return item.reliability or ""


# ---------------------------------------------------------------------------
# DEEP INSPECTOR (deep_inspector.txt)
# ---------------------------------------------------------------------------

def make_deep_inspector(model: Any = None):
    """Deep paper inspector: deep_inspector.txt on GEMMA (reads full text)."""
    return create_deep_agent(
        model=model or build_model_for_role("research"),
        system_prompt=load_prompt("agents", "deep_inspector.txt"),
        tools=[],
    )


_DEEP_SCHEMA = (
    "{\"findings\":["
    "{\"section\":\"...\",\"quote\":\"...\",\"claim\":\"...\","
    "\"support\":\"supports|contradicts\"}]}"
)


def deep_inspect_prompt(requirement: str, full_text: str) -> str:
    return _prompt(
        load_prompt("agents", "deep_inspector.txt"),
        f"EVIDENCE REQUIREMENT:\n{requirement}\n\nFULL PAPER TEXT:\n{full_text}",
        _DEEP_SCHEMA,
    )


async def deep_inspect(agent: Any, requirement: str, full_text: str) -> list[dict]:
    """Return deep-inspection findings [{section, quote, claim, support}]."""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": deep_inspect_prompt(
            requirement, full_text)}]}
    )
    data = _extract_json(_last_text(result.get("messages", []))) or {}
    return list(data.get("findings") or [])


# ---------------------------------------------------------------------------
# CONTRADICTION (contradiction.txt)
# ---------------------------------------------------------------------------

def make_contradiction_agent(model: Any = None):
    """Contradiction/anomaly agent: contradiction.txt on MISTRAL (paced)."""
    return create_deep_agent(
        model=model or build_model_for_role("contradiction"),
        system_prompt=load_prompt("agents", "contradiction.txt"),
        tools=[],
    )


_CONTRADICTION_SCHEMA = (
    "{\"contradictions\":["
    "{\"claim\":\"...\",\"requirement_id\":\"...\","
    "\"evidence_a\":[\"E-...\"],\"evidence_b\":[\"E-...\"],"
    "\"kind\":\"direct_conflict|context_dependent|anomaly\","
    "\"description\":\"...\"}]}"
)


def contradiction_prompt(items: list[Any]) -> str:
    blocks = []
    for it in items:
        it.derive_evidence_level()
        meta = (f"[{it.id}] (req {it.requirement_id}; doc {it.document_id}; "
                f"support={it.verdict.support.value if it.verdict else '?'}")
        if it.study_type and it.study_type != "unknown":
            meta += f"; study={it.study_type}"
        if it.evidence_level:
            meta += f"; level={it.evidence_level}"
        if it.reliability:
            meta += f"; reliability={it.reliability}"
        if it.publication_date:
            meta += f"; date={it.publication_date}"
        blocks.append(meta + ")\n" + it.text)
    blocks = "\n\n".join(blocks)
    return _prompt(
        load_prompt("agents", "contradiction.txt"),
        "VERIFIED EVIDENCE:\n" + blocks,
        _CONTRADICTION_SCHEMA,
    )


async def detect_contradictions(agent: Any, items: list[Any]) -> list:
    from src.agents.logging import xdeep_log
    from src.agents.state import Contradiction, ContradictionKind

    if len(items) < 2:
        return []
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": contradiction_prompt(items)}]}
    )
    data = _extract_json(_last_text(result.get("messages", []))) or {}
    found = []
    by_id = {it.id: it for it in items}
    for i, c in enumerate(data.get("contradictions") or [], 1):
        ev_a = [str(x) for x in (c.get("evidence_a") or [])]
        ev_b = [str(x) for x in (c.get("evidence_b") or [])]
        found.append(Contradiction(
            id=f"C{i}",
            claim=str(c.get("claim") or ""),
            requirement_id=str(c.get("requirement_id") or ""),
            evidence_a=ev_a,
            evidence_b=ev_b,
            evidence_a_text=(by_id.get(ev_a[0]).text if ev_a and ev_a[0] in by_id else ""),
            evidence_b_text=(by_id.get(ev_b[0]).text if ev_b and ev_b[0] in by_id else ""),
            kind=str(c.get("kind") or ContradictionKind.DIRECT_CONFLICT.value),
            description=str(c.get("description") or ""),
        ))
    xdeep_log("contradictions_detected", n=len(found))
    return found


# ---------------------------------------------------------------------------
# RESOLUTION (resolution.txt)
# ---------------------------------------------------------------------------

def make_resolution_agent(model: Any = None):
    """Contradiction resolver: resolution.txt on MISTRAL (paced) WITH search
    tools so it can CHARACTERISE, then DESIGN queries, then JUDGE - the deep
    agent decides what sources to consult (corpus/web/postgres)."""
    from src.agents.tools import tools_for_agent

    return create_deep_agent(
        model=model or build_model_for_role("resolution"),
        system_prompt=load_prompt("agents", "resolution.txt"),
        tools=tools_for_agent(),
    )


_RESOLUTION_PLAN_SCHEMA = (
    "{\"characterization\":\"...\",\"queries\":[\"query 1\",\"query 2\",\"query 3\"]}"
)
_RESOLUTION_SCHEMA = (
    "{\"status\":\"resolved|partially_resolved|unresolved\","
    "\"explanation\":\"...\",\"characterization\":\"...\"}"
)


def _contradiction_body(contradiction: Any) -> str:
    return (
        f"CONTRADICTION CLAIM:\n{contradiction.claim}"
        f"\n\nEVIDENCE SIDE A ({','.join(contradiction.evidence_a)}):\n"
        f"{contradiction.evidence_a_text}"
        f"\n\nEVIDENCE SIDE B ({','.join(contradiction.evidence_b)}):\n"
        f"{contradiction.evidence_b_text}"
    )


async def _resolution_plan(agent: Any, contradiction: Any) -> dict:
    """Stage 1+2: CHARACTERISE both sides then DESIGN 1-3 targeted queries
    that could explain the conflict. Pure planning - no tool calls yet."""
    body = _contradiction_body(contradiction)
    body += (
        "\n\nCURRENT STAGE - STAGE 1 (CHARACTERISE): compare population, "
        "intervention/exposure, outcome, study design, baseline characteristics, "
        "dose, duration, measurement differences, publication context. Decide "
        "whether the conflict is genuine or a difference in scope."
        "\n\nCURRENT STAGE - STAGE 2 (DESIGN): list 1-3 targeted searches that "
        "could explain the conflict (larger studies, meta-analyses, specific "
        "subpopulations, dose/design variants). Return an EMPTY queries list if "
        "no further search could plausibly explain it - that is honest."
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": _prompt(
            load_prompt("agents", "resolution.txt"), body, _RESOLUTION_PLAN_SCHEMA)}]}
    )
    return _extract_json(_last_text(result.get("messages", []))) or {}


async def _resolution_candidates(query: str, top_k: int = 3) -> list[dict]:
    """Local-first candidate lookup for a resolution query: postgres (primary),
    parquet hybrid (fallback), then trust-gated web. Returns candidate dicts
    with text / retrieval_method / source_url / trust - every one of them must
    still pass the VERIFIER before it may influence the judgement."""
    from src.agents.tools.postgres_search import postgres_search_impl

    candidates: list[dict] = []
    try:
        data = json.loads(await postgres_search_impl(
            requirement_id="RES", query=query, top_k=top_k))
        if data.get("available") and not data.get("empty_corpus"):
            for r in (data.get("results") or [])[:top_k]:
                candidates.append({
                    "text": r.get("text", ""),
                    "retrieval_method": "+".join(r.get("methods") or ["pgfts"]),
                    "source_url": "",
                    "trust": "local",
                    "document_id": r.get("document_id", ""),
                    "section": r.get("section", ""),
                })
    except Exception as exc:
        _log_warning("resolution pg search failed: %s", exc)

    if not candidates:
        try:
            from src.agents.reuse import get_hybrid_retriever
            from src.retrieval.plans import SubQueryPlan

            sub = SubQueryPlan(id="RES", target=query, intent="evidence",
                               query=query, focus="evidence",
                               evidence_required=[query])
            for r in await get_hybrid_retriever().search(
                    sub, top_k=top_k, restore_paragraphs=True):
                candidates.append({
                    "text": r.paragraph_text,
                    "retrieval_method": "+".join(getattr(r, "methods", None) or ["hybrid"]),
                    "source_url": "",
                    "trust": "local",
                    "document_id": r.document_id,
                    "section": r.section,
                })
        except Exception as exc:
            _log_warning("resolution hybrid search failed: %s", exc)

    if not candidates:
        try:
            from src.agents.tools.searxng import searxng_search_impl

            data = json.loads(await searxng_search_impl(
                query=query, top_k=6, trusted_only=True))
            for r in (data.get("results") or [])[:top_k]:
                text = (r.get("snippet") or "").strip()
                if len(text) < 40:
                    continue
                candidates.append({
                    "text": text,
                    "retrieval_method": "web:" + str(r.get("engine", "searxng")),
                    "source_url": r.get("url", ""),
                    "trust": r.get("trust", "unverified"),
                    "document_id": "",
                    "section": "web",
                })
        except Exception as exc:
            _log_warning("resolution web search failed: %s", exc)
    return candidates


def _log_warning(msg: str, *args) -> None:
    import logging

    logging.getLogger("agents.resolution").warning(msg, *args)


async def resolve_contradiction(agent: Any, verifier: Any, contradiction: Any,
                                owner: Any = None) -> Any:
    """Run the FULL resolution loop (Gap D) - the agent's judgement is only as
    good as the literature it actually consulted:

      1. PLAN  - characterize + design targeted queries (no tools yet).
      2. SEARCH - run the designed queries through the LOCAL corpus first
                  (postgres -> hybrid), then web; bounded to the top few hits.
      3. VERIFY - every new passage the resolution would lean on MUST pass the
                  verifier critic (rule 2) - unverified web/pg text can never
                  become a resolution reason.
      4. JUDGE  - the final resolved/partially_resolved/unresolved call, with
                  the VERIFIED new evidence included in the prompt.
    """
    from src.agents.logging import xdeep_log
    from src.agents.state import (
        EvidenceItem,
        ResearchRequirement,
        ResolutionOutcome,
        web_document_id,
    )

    plan = await _resolution_plan(agent, contradiction)
    characterization = str(plan.get("characterization") or "")
    queries = [str(q) for q in (plan.get("queries") or []) if str(q).strip()][:3]

    new_evidence: list[EvidenceItem] = []
    rounds = 0
    req = ResearchRequirement(id=contradiction.requirement_id or "RES",
                              text=contradiction.claim, target_n=1)
    for qi, query in enumerate(queries):
        try:
            candidates = await _resolution_candidates(query, top_k=3)
        except Exception as exc:
            _log_warning("resolution search failed for %r: %s", query, exc)
            candidates = []
        if not candidates:
            continue
        rounds += 1
        for ci, cand in enumerate(candidates):
            if len(new_evidence) >= 4:      # pace + verdict cost bound
                break
            text = (cand.get("text") or "").strip()
            if len(text) < 60:
                continue
            item = EvidenceItem(
                id=f"{contradiction.id}.N{qi + 1}-{ci + 1}",
                run_id="",
                requirement_id=req.id,
                chunk_id="",
                document_id=cand.get("document_id", ""),
                section=cand.get("section", ""),
                unit_kind="resolution_search",
                text=text,
                source_url=cand.get("source_url", ""),
                trust=cand.get("trust", ""),
                source_query=query,
                retrieval_method=cand.get("retrieval_method", "resolution"),
                rank=ci + 1,
            )
            req.add_item(item)
            try:
                await verify_item(verifier, req, item)
            except Exception as exc:
                _log_warning("resolution verify failed for %s: %s", item.id, exc)
                item.verdict = None
            if item.verified:
                new_evidence.append(item)
    xdeep_log("resolution_searched", id=contradiction.id, rounds=rounds,
              verified_new=len(new_evidence))

    if owner is not None:
        # fold verified resolution findings into the run's requirement so
        # they reach contradiction re-checks and synthesis (rule: verified
        # text is evidence, wherever it was found).
        for it in new_evidence:
            it.requirement_id = owner.id
            if not it.document_id and it.source_url:
                it.document_id = web_document_id(it.source_url, it.id)
            owner.add_item(it)
        owner.derive_status()

    # ---- stage 3: the judgement, with VERIFIED new evidence only ----------
    body = _contradiction_body(contradiction)
    if characterization:
        body += f"\n\nCHARACTERIZATION (from planning):\n{characterization}"
    if queries:
        body += "\n\nSEARCHES RUN (targeted at the conflict):\n" + "\n".join(
            f"- {q}" for q in queries)
    if new_evidence:
        blocks = "\n\n".join(
            f"[{it.id}] (verified; {it.retrieval_method}; trust={it.trust or 'local'})\n{it.text}"
            for it in new_evidence)
        body += "\n\nNEW VERIFIED EVIDENCE (already passed the CRITIC):\n" + blocks
    body += (
        "\n\nCURRENT STAGE - STAGE 3 (JUDGE): with the characterized conflict "
        "and the additional verified literature, make the final judgement. "
        "resolved = the conflict is explained without declaring a paper wrong; "
        "partially_resolved = narrowed but not fully explained; unresolved = "
        "the available evidence does not explain the discrepancy (state that "
        "honestly - never force a resolution)."
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": _prompt(
            load_prompt("agents", "resolution.txt"), body, _RESOLUTION_SCHEMA)}]}
    )
    data = _extract_json(_last_text(result.get("messages", []))) or {}
    contradiction.resolution = ResolutionOutcome(
        status=str(data.get("status", "unresolved")),
        explanation=str(data.get("explanation", "")),
        additional_queries=queries,
        characterization=characterization or str(data.get("characterization", "")),
        new_evidence_ids=[it.id for it in new_evidence],
        search_rounds=rounds,
    )
    xdeep_log("contradiction_resolved", id=contradiction.id,
              status=contradiction.resolution.status.value,
              characterization=contradiction.resolution.characterization,
              new_evidence=len(new_evidence), search_rounds=rounds)
    return contradiction


# SYNTHESIS (synthesize.txt)
# ---------------------------------------------------------------------------

def make_synthesizer(model: Any = None):
    """Synthesizer: synthesize.txt on GEMMA (long-running actual synthesis)."""
    return create_deep_agent(
        model=model or build_model_for_role("synthesis"),
        system_prompt=load_prompt("agents", "synthesize.txt"),
        tools=[],
    )


def _synthesize_prompt(state) -> str:
    from src.agents.rules import verified_evidence_items

    base = load_prompt("agents", "synthesize.txt")
    evidence = verified_evidence_items(state)
    blocks = []
    for it in evidence:
        support = it.verdict.support.value if it.verdict else "?"
        it.derive_evidence_level()
        meta = (f"[{it.id}] ({it.requirement_id}; doc {it.document_id}; "
                f"support={support}")
        if it.study_type and it.study_type != "unknown":
            meta += f"; study={it.study_type}"
        if it.evidence_level:
            meta += f"; level={it.evidence_level}"
        if it.reliability:
            meta += f"; reliability={it.reliability}"
        if it.publication_date:
            meta += f"; date={it.publication_date}"
        blocks.append(meta + ")\n" + it.text)
    body = "USER QUESTION:\n" + state.question
    body += "\n\nVERIFIED EVIDENCE (the COMPLETE universe):\n" + "\n---\n".join(blocks)

    # contradictions: RESOLVED ones must be woven INLINE into the body of
    # the section that discusses the claim (NOT a separate list) - the reader
    # should meet the resolution in the relevant paragraph. UNRESOLVED ones
    # get an explicit block because they must be surfaced separately.
    resolved = [c for c in getattr(state, "contradictions", [])
                if c.resolution is not None
                and c.resolution.status.value == "resolved"]
    unresolved = [c for c in getattr(state, "contradictions", [])
                  if c.resolution is None
                  or c.resolution.status.value != "resolved"]
    if resolved:
        body += "\n\nRESOLVED CONTRADICTIONS (weave EACH of these INLINE into "                 "the body of the section discussing that claim - explain the "                 "evidence-supported distinction in the paragraph itself, with "                 "citations on the relevant sentences. Do NOT put them in a "                 "separate resolved_contradictions list):\n"
        body += "\n".join(
            f"- {c.claim}: {c.resolution.explanation[:300]}"
            for c in resolved
        )
    if unresolved:
        body += "\n\nUNRESOLVED CONTRADICTIONS (state these explicitly, do not "
        body += "pick a winner):\n"
        for c in unresolved:
            explanation = ""
            if c.resolution is not None:
                explanation = " | " + c.resolution.explanation[:300]
            body += f"- {c.claim} (evidence: {', '.join(c.evidence_a)} vs "
            body += f"{', '.join(c.evidence_b)}){explanation}\n"

    # ---- gap resolutions + derived limitations (post-contradiction pass) --
    gap_resolutions = getattr(state, "gap_resolutions", [])
    resolved_gaps = [g for g in gap_resolutions if g.resolved]
    listed_gaps = [g for g in gap_resolutions if not g.resolved]
    if resolved_gaps:
        body += "\n\nRESOLVED GAPS (closed by additional retrieval after " \
                "contradiction analysis - their evidence is in the VERIFIED " \
                "EVIDENCE above, use it):\n"
        body += "\n".join(
            f"- {g.gap} -> closed via {'local retrieval' if g.status.value == 'resolved_local' else 'trusted web search'}"
            for g in resolved_gaps
        )
    if listed_gaps:
        body += "\n\nUNRESOLVED GAPS (the system could NOT close these even " \
                "after extra local retrievals and web search - state them as " \
                "limitations / unresolved_gaps in your answer, do NOT fill them " \
                "from your own knowledge):\n"
        body += "\n".join(f"- {g.gap}" for g in listed_gaps)
    # derived limitations: any requirement with fewer independent studies than
    # its target is a reported limitation of the evidence base
    derived_lims = []
    for r in getattr(state, "requirements", []):
        if not r.satisfied() and r.coverage() >= 1:
            derived_lims.append(
                f"requirement {r.id}: only {r.coverage()} of the target "
                f"{r.target_n} independent supporting studies were found"
            )
    if derived_lims:
        body += "\n\nDERIVED LIMITATIONS (from verified evidence counts - " \
                "report these under limitations):\n"
        body += "\n".join(f"- {d}" for d in derived_lims)

    body += "\n\nProduce the FINAL ANSWER following the JSON schema in your prompt."
    return base + "\n\n" + body


# Machine citation marker carried in the prose answer: 〔cite:PMC123〕.
# The web bridge maps these to [n] source numbers against the VERIFIED set
# (dropping any id that is not verified - rules.repair_citations semantics).
_CITE_OPEN = "\u3014cite:"
_CITE_CLOSE = "\u3015"


def _section_cite_marker(sec: dict) -> str:
    """Deduped, order-preserving document ids cited by one synthesis section."""
    ids: list[str] = []
    for c in (sec.get("citations") or []):
        doc = str(c.get("document_id") or "").strip()
        if doc and doc not in ids:
            ids.append(doc)
    # also accept a flat "citations" list of {document_id,...} at section level
    if not ids:
        for c in (sec.get("citations") or []):
            if isinstance(c, dict):
                doc = str(c.get("document_id") or "").strip()
                if doc and doc not in ids:
                    ids.append(doc)
            else:
                doc = str(c).strip()
                if doc and doc not in ids:
                    ids.append(doc)
    return "".join(f"{_CITE_OPEN}{d}{_CITE_CLOSE}" for d in ids)


def _answer_to_prose(data: Any) -> str:
    """Convert the synthesize JSON to a plain-text answer with citation markers.

    Each section's citations (document_ids as provided by the synthesizer) are
    embedded as 〔cite:PMCxxx〕 markers the bridge resolves against verified
    sources. Any id that is not in the verified set is dropped at bridge time.
    """
    if not isinstance(data, dict):
        return str(data) if data else ""
    parts: list[str] = []
    if data.get("summary"):
        parts.append(str(data["summary"]))
    for sec in data.get("sections") or []:
        heading = sec.get("heading") or ""
        body = sec.get("body") or ""
        if body:
            marker = _section_cite_marker(sec)
            parts.append((f"## {heading}\n" if heading else "") + str(body) + marker)
    # gaps / limitations / contradictions as proper '##' headers (the UI
    # renders markdown headings, not plain-text labels)
    # resolved contradictions are INLINE in the section bodies (never a
    # separate header); only unresolved contradictions are surfaced as a block
    for header, key in (("Unresolved Gaps", "unresolved_gaps"),
                        ("Limitations", "limitations"),
                        ("Unresolved Contradictions", "unresolved_contradictions")):
        items = data.get(key) or []
        if items:
            block = ["## " + header]
            for it in items:
                if isinstance(it, dict):
                    block.append("- " + (str(it.get("description") or it.get("text")
                                          or it.get("claim") or "")).strip())
                else:
                    block.append("- " + str(it).strip())
            parts.append("\n".join(block))
    return "\n\n".join(p for p in parts if p) or str(data)


_CITE_OPEN_FB = "\u3014cite:"
_CITE_CLOSE_FB = "\u3015"


def fallback_synthesize(state: Any) -> str:
    """DETERMINISTIC evidence-grounded answer used when the LLM synthesizer
    fails (provider error, timeout, malformed output). No LLM - builds a
    faithful answer from the VERIFIED items only, so the user never gets a
    dead "(no answer)" placeholder when evidence exists. Produces the same
    〔cite:doc〕 markers the bridge resolves into [n] source numbers."""
    from src.agents.rules import verified_evidence_items

    evidence = verified_evidence_items(state)
    q = getattr(state, "question", "") or ""
    parts: list[str] = []
    if q:
        parts.append(f"## Summary")
    if evidence:
        parts.append(
            f"This answer was generated from the verified evidence below "
            f"because the language-model synthesis step was unavailable. "
            f"**It is a faithful extraction, not an interpreted summary.**")
        parts.append("## Verified evidence")
        for it in evidence:
            doc = it.document_id or it.id or ""
            support = it.verdict.support.value if it.verdict else "?"
            text = " ".join((it.text or "").split())
            if len(text) > 600:
                text = text[:600].rstrip() + " …"
            marker = f"{_CITE_OPEN_FB}{doc}{_CITE_CLOSE_FB}" if doc else ""
            parts.append(
                f"[{it.id}] doc={doc or '?'} ({it.requirement_id}; "
                f"support={support}){marker}\n{text}")
        parts.append("## Limitations")
        parts.append("- Synthesis ran in fallback mode because the LLM step "
                     "failed; citation markers are present and the verifier "
                     "state is unchanged.")
    else:
        parts.append("No verified evidence was available.")
    return "\n\n".join(parts)


async def synthesize_answer_with_data(agent: Any, state: Any) -> tuple[str, dict | None]:
    """Synthesize; returns (prose, parsed JSON payload or None).

    The parsed payload exposes the synthesizer's OWN unresolved_gaps /
    limitations so the graph can reconcile run.gaps - the gap-resolution
    pass marks a probed sub-question PARTIALLY_RESOLVED when facets remain,
    and any facets the synthesizer still lists must appear in the final
    state (their absence previously made the UI claim "0 listed" while the
    answer actually listed holes).

    Failure-safe: if the LLM call raises (provider timeout / 429 / malformed
    reply), the deterministic fallback answer is returned instead of letting
    the error bubble up into a dead "(no answer - synthesis failed)" - the
    user always receives an evidence-grounded answer with citation markers.
    """
    from src.agents.agents.stages import fallback_synthesize as _fb
    from src.agents.logging import xdeep_log

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": _synthesize_prompt(state)}]}
        )
        text = _last_text(result.get("messages", []))
    except Exception as exc:
        xdeep_log("synthesis_failed_fallback", reason=str(exc)[:300],
                  verified=len(getattr(state, "verified_items", lambda: [])()))
        return _fb(state), None
    try:
        data = _extract_json(text)
        if not isinstance(data, dict):
            return (text.strip() or _fb(state)), None
    except Exception:
        return (text.strip() or _fb(state)), None
    try:
        prose = _answer_to_prose(data)
    except Exception as exc:
        xdeep_log("synthesis_prose_failed_fallback", reason=str(exc)[:300])
        return _fb(state), data
    return prose, data


__all__ = [
    "DecomposedReq", "decompose_requirements", "plan_queries", "replan_queries",
    "verify_item", "deep_inspect", "detect_contradictions", "resolve_contradiction",
    "synthesize_answer", "synthesize_answer_with_data", "judge_site_reliability",
    "make_orchestrator", "make_search_planner", "make_replanner",
    "make_verifier", "make_reliability_critic", "make_deep_inspector",
    "make_contradiction_agent", "make_resolution_agent", "make_synthesizer",
    "make_gap_probe",
]

