"""Site-legitimacy gate tests (offline).

The contract: web evidence must come from TRUSTED medical journals / medical
sites; social media and forums are NEVER trusted; unknown sites are excluded
by default (or clearly labeled 'unverified' when the agent opts in).
"""

from __future__ import annotations

from src.agents.tools.site_reputation import (
    classify_url,
    filter_results,
    is_trusted_url,
)


# --- classification ---------------------------------------------------------

def test_trusted_medical_journals():
    for url in (
        "https://www.nejm.org/doi/full/10.1056/NEJMoa123",
        "https://www.thelancet.com/journals/lancet/article/PIIS0140",
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/",
        "https://www.bmj.com/content/370/m3571",
        "https://jamanetwork.com/journals/jama/fullarticle/2800000",
        "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "https://www.cdc.gov/bloodpressure/index.htm",
        "https://www.mayoclinic.org/diseases-conditions/hypertension",
        "https://www.medscape.com/viewarticle/990000",
    ):
        assert classify_url(url) == "trusted", url
        assert is_trusted_url(url) is True


def test_blocked_social_and_forums():
    for url in (
        "https://www.reddit.com/r/science/comments/xyz/",
        "https://old.reddit.com/r/medicine/",
        "https://twitter.com/someuser/status/123",
        "https://x.com/someuser/status/123",
        "https://www.facebook.com/groups/123",
        "https://www.youtube.com/watch?v=abc",
        "https://www.tiktok.com/@user/video/123",
        "https://www.quora.com/Is-vitamin-D-good",
        "https://medium.com/@somebody/vitamin-d-bp",
    ):
        assert classify_url(url) == "blocked", url


def test_unknown_sites():
    assert classify_url("https://example-random-site.org/article") == "unknown"
    assert classify_url("https://www.some-blog.net/vitamin-d") == "unknown"


def test_subdomain_matching():
    # anything under a trusted/blocked domain inherits the tier
    assert classify_url("https://www.nejm.org/page") == "trusted"
    assert classify_url("https://sub.reddit.com/x") == "blocked"


# --- filtering --------------------------------------------------------------

def test_filter_drops_blocked_and_unknown_by_default():
    entries = [
        {"url": "https://www.nejm.org/x", "title": "journal"},
        {"url": "https://www.reddit.com/r/x", "title": "reddit post"},
        {"url": "https://random-blog.net/x", "title": "blog"},
        {"url": "https://www.who.int/x", "title": "who"},
    ]
    out = filter_results(entries, trusted_only=True)
    urls = [r["url"] for r in out["results"]]
    assert urls == ["https://www.nejm.org/x", "https://www.who.int/x"]
    assert out["dropped_blocked"] == 1
    assert out["dropped_unverified"] == 1
    # results carry a trust tier
    assert all(r["trust"] == "trusted" for r in out["results"])


def test_unverified_only_when_opted_in():
    entries = [
        {"url": "https://www.nejm.org/x", "title": "journal"},
        {"url": "https://random-blog.net/x", "title": "blog"},
    ]
    out = filter_results(entries, trusted_only=False)
    urls = [r["url"] for r in out["results"]]
    # both survive, but the blog is explicitly labeled unverified
    assert len(urls) == 2
    by_url = {r["url"]: r["trust"] for r in out["results"]}
    assert by_url["https://www.nejm.org/x"] == "trusted"
    assert by_url["https://random-blog.net/x"] == "unverified"


def test_blocked_never_survives_even_when_opted_in():
    entries = [
        {"url": "https://twitter.com/u/1", "title": "tweet"},
        {"url": "https://www.reddit.com/r/x", "title": "post"},
    ]
    out = filter_results(entries, trusted_only=False)
    assert out["results"] == []
    assert out["dropped_blocked"] == 2


# --- searxng tool contract --------------------------------------------------

def test_searxng_tool_mentions_trust_in_contract():
    """The tool's description must encode the trust contract for the LLM."""
    from src.agents.tools.searxng import searxng_search

    desc = searxng_search.description
    assert "trusted" in desc
    assert "retrieve" in desc          # documents-first guidance
    assert "Reddit" in desc or "social" in desc


# ---------------------------------------------------------------------------
# SearXNG 403 (json_format_disabled) diagnosis
# ---------------------------------------------------------------------------

def test_searxng_403_is_reported_as_config_error(monkeypatch):
    """A live SearXNG that refuses format=json must be diagnosed as a
    CONFIG error (JSON not whitelisted), not a generic unreachable failure -
    so the operator knows the settings.yml fix instead of hunting a network
    issue."""
    import asyncio
    import json

    import httpx

    from src.agents.tools.searxng import searxng_search_impl

    class _Fake403Resp:
        status_code = 403

        def raise_for_status(self):
            pass  # we handle 403 before raise_for_status

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, *a, **k):
            return _Fake403Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    out = json.loads(asyncio.run(searxng_search_impl("stroke prevention")))
    assert out["available"] is False
    assert out["config_error"] == "json_format_disabled"
    assert "search.formats" in out["note"]   # actionable fix mentioned
