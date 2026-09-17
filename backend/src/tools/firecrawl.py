"""
Firecrawl web-search + scrape tools for deep agents.

Self‑hosted on localhost:3002 by default. Override with FIRECRAWL_URL.
No heuristic filtering – relies on Firecrawl's native extraction.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

import httpx
from pydantic_ai import RunContext

from src.middleware.mybib import verify_urls_mybib
from src.tools.umls import DeepDeps

# =============================================================================
# Configuration – self‑hosted default
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:3002"          # <-- self‑hosted default
MAX_SEARCH_RESULTS = 10
MAX_SCRAPE_URLS = 10
SEARCH_TIMEOUT_S = 30.0
SCRAPE_TIMEOUT_S = 45.0
SCRAPE_WAIT_MS = 5000
# Batch job polling: interval between status checks + overall deadline.
BATCH_POLL_S = 2.0
BATCH_TIMEOUT_S = 600.0


def _base_url() -> str:
    return os.environ.get("FIRECRAWL_URL", DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> Optional[str]:
    key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    return key if key else None


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


# =============================================================================
# Search – with built‑in scraping
# =============================================================================

async def search(
    query: str,
    limit: int = MAX_SEARCH_RESULTS,
    scrape_markdown: bool = True,
    **scrape_options,
) -> dict[str, Any]:
    """
    Search the web and optionally scrape results in one call.

    See: https://docs.firecrawl.dev/api-reference/v1-endpoint/search
    """
    cleaned = " ".join((query or "").split())
    if not cleaned:
        return {"success": False, "data": [], "error": "Empty query"}

    payload: dict[str, Any] = {
        "query": cleaned,
        "limit": min(max(1, limit), 20),
    }

    if scrape_markdown:
        formats = scrape_options.pop("formats", ["markdown"])
        payload["scrapeOptions"] = {
            "formats": formats,
            **scrape_options,
        }

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT_S) as client:
        resp = await client.post(
            f"{_base_url()}/v2/search",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# =============================================================================
# Scrape – full control
# =============================================================================

async def scrape(
    url: str,
    formats: Optional[list[str]] = None,
    only_main_content: bool = True,
    wait_for: int = SCRAPE_WAIT_MS,
    timeout: int = 30000,
    mobile: bool = False,
    block_ads: bool = True,
    remove_base64_images: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Scrape a single URL with full Firecrawl options.

    See: https://docs.firecrawl.dev/api-reference/v1-endpoint/scrape
    """
    if not url:
        return {"success": False, "error": "Empty URL"}

    payload = {
        "url": url,
        "onlyMainContent": only_main_content,
        "waitFor": wait_for,
        "timeout": timeout,
        "mobile": mobile,
        "blockAds": block_ads,
        "removeBase64Images": remove_base64_images,
        "formats": formats or ["markdown"],
        **kwargs,
    }

    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_S) as client:
        resp = await client.post(
            f"{_base_url()}/v1/scrape",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def _nested_data(item: Any) -> dict[str, Any]:
    """The nested scrape envelope, when the API wraps each batch result
    as ``{"success": ..., "data": {...}}`` (flat items yield {})."""
    if not isinstance(item, dict):
        return {}
    data = item.get("data")
    return data if isinstance(data, dict) else {}


def _batch_item_url(item: Any, fallback: str) -> str:
    """URL for a batch result item (metadata.sourceURL, url, or order).

    Checks the nested ``data`` envelope too — some API versions wrap each
    scrape result, so the URL only lives at ``data.url`` /
    ``data.metadata.sourceURL``. Unmatched items would otherwise all key
    on "" and every page would report "missing from batch response".
    """
    if not isinstance(item, dict):
        return fallback
    nested = _nested_data(item)
    candidates = [
        item.get("url"),
        nested.get("url"),
        item.get("sourceURL"),
        nested.get("sourceURL"),
    ]
    for container in (item, nested):
        meta = container.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        candidates.append(meta.get("sourceURL"))
    for candidate in candidates:
        if candidate:
            return candidate
    return fallback


def _normalize_scrape_item(item: Any, fallback_url: str) -> dict[str, Any]:
    """Flatten one batch/scrape result into judge-ready shape.

    Firecrawl carries page content under ``markdown`` (top-level, or nested
    under ``data``), but every downstream consumer — the evidence judge
    (``text``/``snippet``/``content``), the ledger, the gap checker — reads
    ``text``. Without this mapping a successfully scraped page arrives as an
    empty string, is rejected as boilerplate, and no website ever reaches
    synthesis. Raw keys are never dropped; ``text`` is only filled when the
    item has no quotable text of its own, and never for explicit failures.
    """
    if not isinstance(item, dict):
        return {"url": fallback_url, "success": False,
                "error": "non-object batch item"}
    nested = _nested_data(item)
    out = dict(item)
    out["url"] = _batch_item_url(item, fallback_url) or fallback_url
    if "success" not in out and isinstance(nested.get("success"), bool):
        out["success"] = nested["success"]
    if not out.get("error") and isinstance(nested.get("error"), str) \
            and nested["error"].strip():
        out["error"] = nested["error"]
    if out.get("success") is not False \
            and not out.get("text") and not out.get("snippet") \
            and not out.get("content"):
        markdown = out.get("markdown") or nested.get("markdown") or ""
        if isinstance(markdown, str) and markdown.strip():
            out["text"] = markdown
    return out


def _search_result_items(search_result):
    data = search_result.get("data") if isinstance(search_result, dict) else None
    if isinstance(data, dict):
        buckets = []
        keys = ["web"] + [k for k in sorted(data) if k != "web"]
        for key in keys:
            bucket = data.get(key)
            if isinstance(bucket, list):
                buckets.append(bucket)
        items = [r for bucket in buckets for r in bucket]
    elif isinstance(data, list):
        items = data
    else:
        return []
    return [r for r in items if isinstance(r, dict)]


async def search_listings(query: str, limit: int = 6) -> dict[str, Any]:
    """Normalized search shape for legacy callers (gap-fill/worker/stages).

    Wraps :func:`search` (no inline scrape) into
    ``{query, available, results:[{title, url, snippet}]}``. Transport and
    API failures report ``available: False`` instead of raising.
    """
    cleaned = " ".join((query or "").split())
    try:
        result = await search(cleaned, limit=limit, scrape_markdown=False)
    except Exception:
        return {"query": cleaned, "available": False, "results": []}
    if not isinstance(result, dict) or not result.get("success"):
        error = result.get("error", "search failed") if isinstance(result, dict) else "search failed"
        return {"query": cleaned, "available": False, "results": [],
                "note": str(error)}
    items = []
    for r in _search_result_items(result):
        items.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": (r.get("description") or r.get("markdown") or "")[:500],
        })
    return {"query": cleaned, "available": True, "results": items}


async def _scrape_individually(
    urls: list[str],
    options: dict[str, Any],
    batch_error: str = "",
) -> list[dict[str, Any]]:
    """Per-URL /v1/scrape fallback when the batch endpoint is unavailable.

    Same normalized shape as scrape_many: one row per requested URL, in
    input order, never raising. ``batch_error`` is attached to any row that
    also fails so the caller can see why the primary path was skipped.
    """
    out: list[dict[str, Any]] = []
    for url in urls:
        try:
            # Named params only: the v1 endpoint 400s on v2-only keys
            # (maxAge, ignoreInvalidURLs, storeInCache), so the batch
            # options never pass through verbatim.
            row = await scrape(
                url,
                formats=options.get("formats", ["markdown"]),
                only_main_content=options.get("onlyMainContent", True),
                wait_for=options.get("waitFor", SCRAPE_WAIT_MS),
                timeout=options.get("timeout", 30000),
                mobile=options.get("mobile", False),
                block_ads=options.get("blockAds", True),
                remove_base64_images=options.get("removeBase64Images", True),
            )
        except Exception as exc:  # noqa: BLE001 — one dead URL never kills the set
            out.append({"url": url, "success": False,
                        "error": f"{exc}" + (f" (batch: {batch_error})"
                                             if batch_error else "")})
            continue
        normalized = _normalize_scrape_item(row.get("data", row), url)
        if isinstance(row, dict) and row.get("success") is False \
                and not normalized.get("error"):
            normalized["success"] = False
            normalized["error"] = str(row.get("error", "scrape failed"))
        out.append(normalized)
    return out


# Allowed v2 batch payload keys (snake_case translated); unknown keys
# are dropped because v2 rejects them with 400.
_V2_KEYMAP = {
    "formats": "formats",
    "onlyMainContent": "onlyMainContent",
    "only_main_content": "onlyMainContent",
    "waitFor": "waitFor",
    "wait_for": "waitFor",
    "timeout": "timeout",
    "mobile": "mobile",
    "blockAds": "blockAds",
    "block_ads": "blockAds",
    "removeBase64Images": "removeBase64Images",
    "remove_base64_images": "removeBase64Images",
    "maxAge": "maxAge",
    "max_age": "maxAge",
    "ignoreInvalidURLs": "ignoreInvalidURLs",
    "storeInCache": "storeInCache",
    "maxConcurrency": "maxConcurrency",
    "proxy": "proxy",
}


async def scrape_many(
    urls: list[str],
    max_urls: int = MAX_SCRAPE_URLS,
    **scrape_kwargs,
) -> list[dict[str, Any]]:
    """Batch-scrape URLs in ONE server-side job (POST /v2/batch/scrape).

    The server scrapes all pages in parallel and this polls the job to
    completion — one submit plus cheap status checks instead of N full
    scrape round-trips. maxAge pins cache reuse to 48h so repeat evidence
    runs cost no fresh scrapes. Output order matches the (deduped, capped)
    input; a failed job reports per-URL errors, and pages missing from
    the response are reported, never silently dropped.
    """
    if not urls:
        return []

    seen = set()
    unique_urls = []
    for u in urls:
        u = str(u or "").strip()
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)
            if len(unique_urls) >= max_urls:
                break

    options = {
        "onlyMainContent": True,
        "waitFor": SCRAPE_WAIT_MS,
        "timeout": 30000,
        "mobile": False,
        "blockAds": True,
        "removeBase64Images": True,
        "formats": ["markdown"],
        "ignoreInvalidURLs": True,
        "storeInCache": True,
        "maxAge": 172800000,
    }
    # v2 rejects unknown payload keys with 400, so caller overrides pass
    # through a strict allow-list (snake_case translated); anything else
    # is dropped instead of failing the whole batch submit.
    for key, value in scrape_kwargs.items():
        target = _V2_KEYMAP.get(key)
        if target is not None:
            options[target] = value
    base = _base_url()
    headers = _headers()
    invalid_urls: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_S) as client:
            resp = await client.post(
                base + "/v2/batch/scrape", headers=headers,
                json={"urls": unique_urls, **options},
            )
            resp.raise_for_status()
            submitted = resp.json()
            if isinstance(submitted, dict) and submitted.get("success") is False:
                err = submitted.get("error", submitted)
                raise ValueError(f"batch submit rejected: {err!r}")
            job_id = submitted.get("id") if isinstance(submitted, dict) else None
            raw_invalid = submitted.get("invalidURLs", []) if isinstance(submitted, dict) else []
            if isinstance(raw_invalid, list):
                for entry in raw_invalid:
                    if isinstance(entry, str) and entry.strip():
                        invalid_urls.add(entry.strip())
                    elif isinstance(entry, dict):
                        for key in ("url", "sourceURL"):
                            value = entry.get(key)
                            if isinstance(value, str) and value.strip():
                                invalid_urls.add(value.strip())
                                break
            if not job_id:
                raise ValueError(f"batch submit returned no job id: {submitted!r}")
            deadline = time.monotonic() + BATCH_TIMEOUT_S
            payload: Any = None
            while True:
                status_resp = await client.get(
                    f"{base}/v2/batch/scrape/{job_id}", headers=headers)
                status_resp.raise_for_status()
                payload = status_resp.json()
                if (not isinstance(payload, dict)
                        or payload.get("status") in ("completed", "failed")):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"batch scrape job {job_id} still running "
                        f"after {BATCH_TIMEOUT_S:.0f}s")
                await asyncio.sleep(BATCH_POLL_S)
    except Exception as exc:
        # Batch scraping is unavailable on some plans (the hosted free tier
        # rejects /v1/batch/scrape with 401 while plain /v1/scrape works).
        # Fall back to per-URL scrapes so web evidence still reaches the
        # ledger instead of the whole web leg coming back empty.
        return await _scrape_individually(unique_urls, options, str(exc))
    data = payload.get("data") if isinstance(payload, dict) else None
    items = data if isinstance(data, list) else []
    by_url: dict[str, dict] = {}
    for item in items:
        by_url.setdefault(_batch_item_url(item, ""), item)
    out = []
    for url in unique_urls:
        match = by_url.pop(url, None)
        if isinstance(match, dict):
            out.append(_normalize_scrape_item(match, url))
        elif url in invalid_urls:
            out.append({"url": url, "success": False,
                        "error": "rejected as invalid URL by batch submit"})
        else:
            out.append({"url": url, "success": False,
                        "error": "missing from batch response"})
    return out


# =============================================================================
# Actions – click, wait, type, etc.
# =============================================================================

async def scrape_with_actions(
    url: str,
    actions: list[dict[str, Any]],
    formats: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Scrape after performing browser actions (click, wait, type, etc.).
    """
    return await scrape(url, formats=formats, actions=actions, **kwargs)


# =============================================================================
# Agent‑facing tools (your existing interface)
# =============================================================================

async def web_search(
    ctx: RunContext[DeepDeps],
    query: str,
    limit: int = MAX_SEARCH_RESULTS,
) -> str:
    """
    Search the web, credibility-gate, and scrape each kept URL.

    Returns JSON with 'results', 'pages', 'trust', 'dropped'.
    """
    # 1. Search with markdown
    try:
        search_result = await search(query, limit=limit, scrape_markdown=True)
    except Exception as e:
        return json.dumps({
            "query": query,
            "available": False,
            "error": str(e),
        }, ensure_ascii=False)

    if not search_result.get("success"):
        return json.dumps({
            "query": query,
            "available": False,
            "error": search_result.get("error", "Search failed"),
        }, ensure_ascii=False)

    data = _search_result_items(search_result)
    if not data:
        return json.dumps({
            "query": query,
            "available": True,
            "results": [],
            "pages": [],
            "dropped": 0,
        }, ensure_ascii=False)

    # 2. MyBib credibility gate (score >= 4 kept, fail-open per URL)
    try:
        kept, report = await verify_urls_mybib(
            [{"url": r.get("url"), "title": r.get("title")} for r in data],
            label="mybib:web_search",
        )
    except Exception as e:
        kept, report = data, None

    kept_urls = []
    for item in kept:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if url:
            kept_urls.append(url)

    # 3. Scrape each kept URL
    pages = await scrape_many(
        kept_urls,
        max_urls=MAX_SCRAPE_URLS,
        only_main_content=True,
        wait_for=SCRAPE_WAIT_MS,
        block_ads=True,
        remove_base64_images=True,
    )

    return json.dumps({
        "query": query,
        "available": True,
        "results": data,
        "pages": pages,
        "trust": [v.model_dump() for v in (report.results if report else [])],
        "dropped": len(data) - len(kept),
    }, ensure_ascii=False)


async def firecrawl_fetch_urls(
    ctx: RunContext[DeepDeps],
    urls: list[str] | str,
    query: str = "",
    **scrape_kwargs,
) -> str:
    """Fetch and scrape a list of URLs, with optional override parameters."""
    if isinstance(urls, str):
        try:
            parsed = json.loads(urls)
            urls = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            urls = []

    results = await scrape_many(urls, **scrape_kwargs)
    return json.dumps({
        "urls": [r.get("url") for r in results],
        "results": results,
    }, ensure_ascii=False)

