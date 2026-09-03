"""Site-legitimacy gate for web-evidence sourcing.

The user's contract: evidence is sourced HEAVILY from the local corpus; web
search is used only when more info/help is needed, and web results must come
from TRUSTED medical journals / medical websites. Reddit/Twitter/social and
forums are NEVER trusted.

This is a DETERMINISTIC gate (a rule, not an LLM judgment): classify_url /
filter_results commit the tool's contract so an agent output can never
smuggle an untrusted site into evidence.

Tiers:
  * trusted   - reputable medical journals, national bodies, official
                medical/health sites (NEJM, Lancet, BMJ, JAMA, PubMed/PMC,
                WHO, CDC, NIH, Mayo, Cleveland Clinic, Medscape, UpToDate...)
  * consumer  - authoritative-but-consumer health portals (WebMD, Drugs.com,
                RxList, Everyday Health). Retrievable, but explicitly NOT
                evidence-grade: the reliability critic agent must judge any
                claim sourced from them (they are never trusted as primary
                evidence).
  * blocked   - social media, forums, aggregators - NEVER returned
  * unknown   - anything else; excluded under trusted_only (default), or
                returned labeled "unverified" when the agent opts in
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Domain lists (match host or any subdomain)
# ---------------------------------------------------------------------------

TRUSTED_DOMAINS: frozenset = frozenset({
    # Journals / literature repositories
    "nejm.org", "thelancet.com", "bmj.com", "jamanetwork.com", "annals.org",
    "academic.oup.com", "nature.com", "science.org", "cell.com",
    "sciencedirect.com", "elsevier.com", "springer.com", "wiley.com",
    "taylorandfrancis.com", "cochranelibrary.com", "europepmc.org",
    "ahajournals.org", "jacc.org", "jco.org", "asco.org", "escardio.org",
    "acc.org", "heart.org", "diabetesjournals.org", "endojournals.org",
    # National / institutional bodies
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "nih.gov", "nlm.nih.gov", "medlineplus.gov", "clinicaltrials.gov",
    "cdc.gov", "fda.gov", "who.int", "europa.eu", "nice.org.uk",
    # Major medical centers / knowledge bases
    "mayoclinic.org", "clevelandclinic.org", "hopkinsmedicine.org",
    "mgh.harvard.edu", "med.upenn.edu", "stanfordhealthcare.org",
    "medscape.com", "uptodate.com",
})

# Consumer health portals: authoritative for patient education, NOT
# evidence-grade primary sources. Always labeled "consumer" - the reliability
# critic agent judges claims sourced from them, and they never count as
# trusted evidence on their own.
CONSUMER_DOMAINS: frozenset = frozenset({
    "webmd.com", "drugs.com", "rxlist.com", "everydayhealth.com",
    "healthline.com", "verywellhealth.com", "medicalnewstoday.com",
})

BLOCKED_DOMAINS: frozenset = frozenset({
    # Social media / forums / content aggregators - explicitly untrusted
    "reddit.com", "old.reddit.com", "twitter.com", "x.com", "t.co",
    "facebook.com", "instagram.com", "tiktok.com", "youtube.com",
    "pinterest.com", "snapchat.com", "quora.com", "medium.com",
    "tumblr.com", "9gag.com", "imgur.com", "4chan.org", "discord.com",
    "telegram.org", "fandom.com", "wikihow.com", "answers.com",
    "stackexchange.com", "yahooanswers.com", "ask.com",
})


def _host(url: str) -> str:
    """Lower-cased host of a url, without www."""
    try:
        parsed = urlparse(url if "://" in (url or "") else "http://" + url)
        host = (parsed.hostname or "").lower()
    except Exception:
        host = (url or "").lower()
    return host.removeprefix("www.")


def classify_url(url: str) -> str:
    """Return 'trusted' | 'consumer' | 'blocked' | 'unknown' for a url's site."""
    host = _host(url)
    if not host:
        return "unknown"
    if host in BLOCKED_DOMAINS or host.endswith(tuple("." + d for d in BLOCKED_DOMAINS)):
        return "blocked"
    if host in TRUSTED_DOMAINS or host.endswith(tuple("." + d for d in TRUSTED_DOMAINS)):
        return "trusted"
    if host in CONSUMER_DOMAINS or host.endswith(tuple("." + d for d in CONSUMER_DOMAINS)):
        return "consumer"
    return "unknown"


def is_trusted_url(url: str) -> bool:
    return classify_url(url) == "trusted"


# ---------------------------------------------------------------------------
# Result filtering
# ---------------------------------------------------------------------------

def filter_results(entries: list[dict], trusted_only: bool = True) -> dict:
    """Run every entry through the trust gate.

    Returns {"results": [...], "dropped_blocked": n, "dropped_unverified": n}.
    Blocked is ALWAYS dropped; unknown is dropped unless trusted_only=False
    (then it is returned labeled "unverified").
    """
    kept: list[dict] = []
    blocked = unverified = 0
    for e in entries:
        tier = classify_url(e.get("url", ""))
        if tier == "blocked":
            blocked += 1
            continue
        if tier == "unknown" and trusted_only:
            unverified += 1
            continue
        out = dict(e)
        # normalize: trusted stays trusted; consumer is a distinct (lower)
        # tier; anything else is 'unverified'
        if tier == "trusted":
            out["trust"] = "trusted"
        elif tier == "consumer":
            out["trust"] = "consumer"
        else:
            out["trust"] = "unverified"
        kept.append(out)
    return {"results": kept, "dropped_blocked": blocked,
            "dropped_unverified": unverified}


def trust_summary(entries: list[dict], trusted_only: bool = True) -> str:
    """JSON one-liner describing what survived the gate (for the tool)."""
    return json.dumps(filter_results(entries, trusted_only=trusted_only),
                      ensure_ascii=False)


__all__ = [
    "TRUSTED_DOMAINS", "CONSUMER_DOMAINS", "BLOCKED_DOMAINS",
    "classify_url", "is_trusted_url", "filter_results", "trust_summary",
]
