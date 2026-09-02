"""searxng web-search tool adapter (trust-gated).

The research agent may CHOOSE between this (web search) and the hybrid
retriever. Evidence is meant to be sourced HEAVILY from the local corpus;
web search is a HELP when more info is needed. Every web result passes the
site-legitimacy gate (site_reputation): social media / forums are NEVER
returned, and by default only TRUSTED medical journals / medical sites are
returned. The agent may opt into "unverified" results (trusted_only=False),
which are clearly labeled so it never mistakes them for trusted evidence.

Searxng URL: XDEEP_SEARXNG_URL or SEARXNG_URL (default http://127.0.0.1:8888).
Degrades cleanly when unreachable so the agent falls back to the retriever.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from src.x_deepagents.config import searxng_timeout, searxng_top_k, searxng_url
from src.x_deepagents.tools.site_reputation import filter_results


def searxng_base_url() -> str:
    """Searxng instance base URL (env-configurable)."""
    return searxng_url()


async def searxng_search_impl(query: str, top_k: int = None, trusted_only: bool = True) -> str:
    """Core implementation: SearXNG web search + site-legitimacy gate."""
    import httpx

    from src.x_deepagents.logging import xdeep_log

    base = searxng_base_url()
    xdeep_log("web_search_started", query=query, base=base,
              trusted_only=bool(trusted_only))
    try:
        async with httpx.AsyncClient(timeout=searxng_timeout()) as client:
            resp = await client.get(
                base + "/search",
                # a browser-ish UA: several SearXNG engines / the limiter drop
                # requests with the default python-httpx UA
                headers={"User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")},
                params={"q": query, "format": "json", "safesearch": 1},
            )
            if resp.status_code == 403:
                # SearXNG only serves JSON when search.formats whitelists it
                # (modern images default to html only). This is a CONFIG error
                # on the instance, not a query problem - say exactly that so
                # the agent/user knows the fix, instead of a generic failure.
                xdeep_log("web_search_config_error", query=query, base=base,
                          status=403)
                return json.dumps({
                    "query": query,
                    "available": False,
                    "config_error": "json_format_disabled",
                    "results": [],
                    "note": (f"SearXNG at {base} refused format=json (HTTP 403). "
                             "Enable the JSON API in searxng/settings.yml: "
                             "search.formats: [html, json], then restart the "
                             "container. That, or fall back to the local corpus."),
                }, ensure_ascii=False)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        xdeep_log("web_search_failed", query=query, error=str(exc)[:300],
                  base=base)
        return json.dumps({
            "query": query,
            "available": False,
            "results": [],
            "note": f"searxng unreachable at {base}: {str(exc)[:200]}",
        }, ensure_ascii=False)

    limit = top_k if top_k is not None else searxng_top_k()
    raw = []
    for r in (data.get("results") or [])[: max(1, limit)]:
        raw.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or r.get("snippet") or "")[:500],
            "engine": ",".join(r.get("engines") or r.get("engine") or []),
        })

    gated = filter_results(raw, trusted_only=trusted_only)
    xdeep_log("web_search_done", query=query, count=len(gated["results"]),
              dropped_blocked=gated["dropped_blocked"],
              dropped_unverified=gated["dropped_unverified"],
              urls=[r.get("url", "") for r in gated["results"]])
    return json.dumps({
        "query": query,
        "available": True,
        "trusted_only": bool(trusted_only),
        "count": len(gated["results"]),
        "dropped_blocked": gated["dropped_blocked"],
        "dropped_unverified": gated["dropped_unverified"],
        "results": gated["results"],
    }, ensure_ascii=False)


@tool
async def searxng_search(query: str, top_k: int = None, trusted_only: bool = True) -> str:
    """Web search via a local SearXNG instance, trust-gated.

    Returns top results as JSON, each annotated with a "trust" tier:
      * trusted    - reputable medical journal / official medical site
                     (NEJM, Lancet, BMJ, JAMA, PubMed/PMC, WHO, CDC, NIH,
                     Mayo Clinic, Medscape, UpToDate, ...)
      * unverified - other sites (only present when trusted_only=False)
    Social media / forums (Reddit, X/Twitter, Facebook, YouTube, ...) are
    ALWAYS dropped - never trusted.

    Use this ONLY when the local corpus (retrieve / postgres_search) has been
    exhausted and still does not cover what you need. Local evidence first;
    web is for gaps, current guidelines, or later information. If the
    instance is unreachable, returns {"available": false, ...} so you should
    fall back to the local corpus.
    """
    return await searxng_search_impl(
        query=query,
        top_k=top_k if top_k is not None else None,
        trusted_only=bool(trusted_only),
    )


__all__ = ["searxng_search", "searxng_search_impl", "searxng_base_url"]
