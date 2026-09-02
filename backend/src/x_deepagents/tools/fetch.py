"""Fetch a web page and extract its readable text for deep web evidence.

The search tool returns SNIPPETS (<= ~500 chars). That is not enough to
verify a claim or judge reliability properly - so the gap-fill / web-augment
passes fetch the ACTUAL page and use the extracted text as the evidence body:

  snippet -> page fetch (full text) -> VERIFIER judges the real page
                                   -> RELIABILITY critic reads the real page

Failure is safe: any network / parse / robots issue returns "" and the caller
falls back to the snippet (the searxng snippet is always kept as the floor).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("x_deepagents.fetch")

_PAGE_MAX_BYTES = 1_500_000          # never buffer a huge download
# Browser-like headers: several medical sites (PMC, Mayo, AHA) serve bot-
# challenge pages to bare clients; a fuller header set avoids many of them.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Pages that are actually anti-bot interstitials (reCAPTCHA / JS challenge)
# must be treated as fetch FAILURE - the caller falls back to the snippet.
_CHALLENGE_MARKERS = (
    "checking your browser", "enable javascript and cookies",
    "verify you are human", "recaptcha", "captcha",
    "attention required!",
)


def extract_html_text(html: str, max_chars: int = 10000) -> str:
    """Best-effort HTML -> plain text (strip scripts/styles/tags, collapse
    whitespace). Deterministic regex pipeline; no heavy parser dependency."""
    if not html:
        return ""
    text = re.sub(r"(?is)<(script|style|noscript|svg|template).*?</\1>", " ", html)
    # drop comments + remaining tags
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    # decode common entities
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
        "&#39;": "'", "&nbsp;": " ", "&apos;": "'",
    }
    for k, v in entities.items():
        text = text.replace(k, v)
    # collapse whitespace, then cut at a sentence boundary near max_chars
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"[ \n]{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        cut = max_chars
        # back off to a sentence/space boundary so we never split mid-claim
        while cut > max_chars - 200 and not text[cut:cut + 1].isspace():
            cut -= 1
        if cut <= max_chars - 200:
            cut = max_chars
        text = text[:cut].rstrip() + " ..."
    return text


async def fetch_page_text(url: str, max_chars: int = 10000,
                          timeout_s: float = 12.0) -> str:
    """Fetch one web page and return its extracted text ("" on any failure).

    Returns "" when the page is unreachable OR is an anti-bot challenge
    interstitial - the caller MUST then fall back to the searxng snippet.
    """
    import httpx

    if not url or not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        async with httpx.AsyncClient(
            timeout=timeout_s, follow_redirects=True,
            headers=HEADERS, max_redirects=4,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            if len(resp.content) > _PAGE_MAX_BYTES:
                logger.warning("page too large to fetch fully: %s", url)
            snippet_html = resp.text[: _PAGE_MAX_BYTES]
    except Exception as exc:
        logger.info("page fetch failed %s: %s", url, str(exc)[:160])
        return ""
    text = extract_html_text(snippet_html, max_chars=max_chars)
    low = text.lower()
    if any(m in low for m in _CHALLENGE_MARKERS) and len(text) < 600:
        logger.info("page served an anti-bot challenge (falling back to snippet): %s", url)
        return ""
    return text


__all__ = ["fetch_page_text", "extract_html_text"]
