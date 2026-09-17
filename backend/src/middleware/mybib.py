"""MyBib credibility gate for web-search URLs.

This replaced the LLM trust gate on the web path: for each candidate URL
the gate asks MyBib\u2019s autocite endpoint for a credibility score and
keeps only URLs scoring at least ``MYBIB_MIN_SCORE`` (default 4). MyBib
resolves the citation from the page itself \u2014 a PMC link comes back
with authors, journal and DOI at 5/5, a social-media page at 3/5 with an
explicit comment \u2014 so the score reflects the actual source rather
than a domain allow-list plus one LLM judgment call.

Contract: ``(kept_items, TrustReport)`` with ``UrlVerdict`` rows,
which keeps the tool output shape (``trust: [...]``) stable for the judge
and the UI. Fail-open per URL: an unreachable MyBib keeps the URL with a
marked reason, so a dead gate never drops recall. A reachable gate that
returns no score for a URL drops it \u2014 unknown is untrusted.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from src.lib import narrate
from src.lib.trace import get_trace
from src.middleware.trust import TrustReport, UrlVerdict

logger = logging.getLogger(__name__)

MYBIB_SEARCH_URL = "https://www.mybib.com/api/autocite/search"
MYBIB_TIMEOUT_S = 15.0
MYBIB_MAX_CONCURRENCY = 5
DEFAULT_MIN_SCORE = 4

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
)


def min_score() -> int:
    """Credibility floor; ``MYBIB_MIN_SCORE`` overrides the default 4."""
    try:
        return max(0, int(os.environ.get("MYBIB_MIN_SCORE", DEFAULT_MIN_SCORE)))
    except (TypeError, ValueError):
        return DEFAULT_MIN_SCORE


def _headers() -> dict[str, str]:
    return {
        "accept": "application/json, text/plain, */*",
        "referer": "https://www.mybib.com/",
        "user-agent": _USER_AGENT,
    }


def _parse_credibility(payload: object) -> tuple[float | None, str]:
    """Pull (score, comment) out of an autocite reply; (None, reason) when
    the reply carries no usable credibility verdict."""
    if not isinstance(payload, dict):
        return None, "non-object autocite reply"
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None, "no autocite results"
    first = results[0] if isinstance(results[0], dict) else {}
    cred = first.get("credibility") if isinstance(first, dict) else None
    if not isinstance(cred, dict):
        return None, "no credibility verdict"
    try:
        score = float(cred.get("score"))
    except (TypeError, ValueError):
        return None, "unparsable credibility score"
    comment = cred.get("comment")
    return score, str(comment or "").strip()


async def _score_one(client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> tuple[str, float | None, str]:
    """Score one URL; never raises — transport/parse failure is (None, reason)."""
    async with semaphore:
        try:
            resp = await client.get(
                MYBIB_SEARCH_URL,
                headers=_headers(),
                params={"q": url, "sourceId": "webpage"},
            )
            resp.raise_for_status()
            score, comment = _parse_credibility(resp.json())
        except Exception as exc:  # noqa: BLE001 \u2014 per-URL fail-open
            return url, None, f"mybib unreachable ({exc})"
    if score is None:
        return url, None, f"mybib: {comment}"
    return url, score, comment


async def verify_urls_mybib(
    items: list[dict],
    label: str = "mybib",
    floor: int | None = None,
) -> tuple[list[dict], TrustReport]:
    """Keep URLs whose MyBib credibility score meets the floor.

    Returns ``(trustworthy items, report)`` like the old trust gate. One
    autocite request per URL, run concurrently (bounded); the input order
    is preserved. Unreachable-MyBib URLs are kept and marked (fail-open);
    scored-below-floor and score-less URLs are dropped with reasons.
    """
    clean = [dict(i) for i in items if isinstance(i, dict) and i.get("url")]
    if not clean:
        return [], TrustReport(results=[])
    threshold = floor if floor is not None else min_score()
    narrate.say(
        f"[{label}] credibility-checking {len(clean)} url(s)"
        f" (MyBib floor {threshold})\u2026"
    )
    semaphore = asyncio.Semaphore(MYBIB_MAX_CONCURRENCY)
    async with httpx.AsyncClient(timeout=MYBIB_TIMEOUT_S) as client:
        scored = await asyncio.gather(*(
            _score_one(client, str(item["url"]), semaphore) for item in clean
        ))
    by_url = {url: (score, comment) for url, score, comment in scored}
    kept: list[dict] = []
    verdicts: list[UrlVerdict] = []
    for item in clean:
        url = str(item["url"])
        score, comment = by_url.get(url, (None, "not scored"))
        if score is None and comment.startswith("mybib unreachable"):
            kept.append(item)
            verdicts.append(UrlVerdict(
                url=url, trustworthy=True, category="mybib:unavailable",
                reason=comment + " \u2014 kept (fail-open)"))
        elif score is not None and score >= threshold:
            kept.append(item)
            shown = int(score) if float(score).is_integer() else score
            verdicts.append(UrlVerdict(
                url=url, trustworthy=True, category=f"mybib:{shown}",
                reason=f"MyBib credibility {shown}/{threshold} floor"
                + (f" \u2014 {comment}" if comment else "") + " \u2014 kept"))
        else:
            why = comment if score is None else f"credibility {score} < {threshold}"
            verdicts.append(UrlVerdict(
                url=url, trustworthy=False, category="mybib:below-floor",
                reason=f"MyBib {why}" + (f" \u2014 {comment}" if score is not None and comment else "") + " \u2014 rejected"))
    get_trace().log("mybib_gate", label=label, floor=threshold,
                    kept=[v.url for v in verdicts if v.trustworthy],
                    dropped=[v.url for v in verdicts if not v.trustworthy])
    narrate.say(f"    [{label}] credibility judgment: {len(kept)} kept, "
                f"{len(clean) - len(kept)} rejected of {len(clean)} url(s)")
    return kept, TrustReport(results=verdicts)
