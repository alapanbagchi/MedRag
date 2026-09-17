"""MyBib credibility-gate tests (stubbed HTTP, no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from src.middleware import mybib as mybib_mod
from src.middleware.mybib import _parse_credibility, min_score, verify_urls_mybib


class FakeMyBibResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)

    def json(self):
        return self._payload


class FakeMyBibClient:
    """Replays per-URL credibility verdicts; records requested URLs."""

    def __init__(self, scores=None, error=None):
        self._scores = scores or {}
        self._error = error
        self.seen = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, **kwargs):
        q = (kwargs.get("params") or {}).get("q", "")
        self.seen.append(q)
        if self._error is not None:
            raise self._error
        entry = self._scores.get(q, "missing")
        if entry == "unreachable":
            raise httpx.ConnectError("refused")
        if entry == "missing":
            return FakeMyBibResponse({"results": []})
        score, comment = entry
        return FakeMyBibResponse({"results": [{
            "sourceId": "webpage",
            "metadata": {"title": "T", "url": q},
            "credibility": {"score": score, "comment": comment},
        }]})


def _client(monkeypatch, **kwargs):
    client = FakeMyBibClient(**kwargs)
    monkeypatch.setattr(mybib_mod.httpx, "AsyncClient", lambda **kw: client)
    return client


def _items(*urls):
    return [{"url": u, "title": "T"} for u in urls]


async def test_verify_keeps_above_floor_drops_below(monkeypatch):
    client = _client(monkeypatch, scores={
        "https://journal.example/a": (5, "peer-reviewed article"),
        "https://social.example/b": (3, "social media source"),
    })
    kept, report = await verify_urls_mybib(_items(
        "https://journal.example/a", "https://social.example/b"))
    assert [i["url"] for i in kept] == ["https://journal.example/a"]
    by_url = {v.url: v for v in report.results}
    assert by_url["https://journal.example/a"].trustworthy is True
    assert by_url["https://social.example/b"].trustworthy is False
    assert by_url["https://journal.example/a"].category == "mybib:5"
    assert len(client.seen) == 2


async def test_verify_fail_open_when_mybib_down(monkeypatch):
    _client(monkeypatch, error=httpx.ConnectError("refused"))
    kept, report = await verify_urls_mybib(_items(
        "https://a.example/", "https://b.example/"))
    assert [i["url"] for i in kept] == ["https://a.example/", "https://b.example/"]
    assert all(v.trustworthy for v in report.results)
    assert all(v.category == "mybib:unavailable" for v in report.results)


async def test_verify_drops_scoreless_but_service_up(monkeypatch):
    _client(monkeypatch, scores={})
    kept, report = await verify_urls_mybib(_items("https://x.example/"))
    assert kept == []
    assert report.results[0].trustworthy is False


async def test_min_score_env_override(monkeypatch):
    monkeypatch.setenv("MYBIB_MIN_SCORE", "5")
    assert min_score() == 5
    _client(monkeypatch, scores={"https://a.example/": (4, "solid")})
    kept, _ = await verify_urls_mybib(_items("https://a.example/"))
    assert kept == []


def test_min_score_defaults_and_bad_env(monkeypatch):
    monkeypatch.delenv("MYBIB_MIN_SCORE", raising=False)
    assert min_score() == 4
    monkeypatch.setenv("MYBIB_MIN_SCORE", "junk")
    assert min_score() == 4


def test_parse_credibility_shapes():
    assert _parse_credibility(
        {"results": [{"credibility": {"score": 5, "comment": "c"}}]}) == (5.0, "c")
    assert _parse_credibility({"results": []})[0] is None
    assert _parse_credibility({"results": [{"nope": 1}]})[0] is None
    assert _parse_credibility(
        {"results": [{"credibility": {"score": "high"}}]})[0] is None
    assert _parse_credibility("junk")[0] is None


async def test_web_search_wires_mybib_gate(monkeypatch):
    # Locks the swap: the agent web tool calls the MyBib gate, not the
    # old LLM trust checker, with dict-shaped cloud results flowing through.
    from src.tools import firecrawl as firecrawl_mod

    web = [
        {"title": "T0", "url": "https://journal.example/a", "description": "s0"},
        {"title": "T1", "url": "https://social.example/b", "description": "s1"},
    ]

    async def fake_search(query, limit=10, **kwargs):
        return {"success": True, "data": {"web": web}}

    async def fake_scrape_many(urls, **kwargs):
        return [{"url": u, "success": True, "text": "Body"} for u in urls]

    seen = {}

    async def fake_mybib(items, **kwargs):
        seen["label"] = kwargs.get("label")
        kept = [i for i in items if "journal" in i["url"]]
        from src.middleware.trust import TrustReport, UrlVerdict
        return kept, TrustReport(results=[
            UrlVerdict(url=i["url"], trustworthy="journal" in i["url"],
                       category="test", reason="stubbed") for i in items
        ])

    monkeypatch.setattr(firecrawl_mod, "search", fake_search)
    monkeypatch.setattr(firecrawl_mod, "scrape_many", fake_scrape_many)
    monkeypatch.setattr(firecrawl_mod, "verify_urls_mybib", fake_mybib)
    data = json.loads(await firecrawl_mod.web_search(
        SimpleNamespace(), "q"))
    assert data["available"] is True
    assert len(data["results"]) == 2
    assert data["dropped"] == 1
    assert [p["url"] for p in data["pages"]] == ["https://journal.example/a"]
    assert seen["label"] == "mybib:web_search"
