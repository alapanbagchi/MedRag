"""Firecrawl tool tests (stubbed HTTP, no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from src.agents.deep_agent import build_deep_agent
from src.tools import firecrawl as firecrawl_mod
from src.tools.firecrawl import scrape, scrape_many, search, search_listings


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)

    def json(self):
        return self._payload


class FakeClient:
    """Stand-in for httpx.AsyncClient; records calls, replays responses."""

    def __init__(self, post=None, get=None, error=None):
        self._post = post
        self._get = get
        self._error = error
        self.seen: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.seen.append(("POST", url, kwargs.get("json")))
        if self._error is not None:
            raise self._error
        return self._post

    async def get(self, url, **kwargs):
        self.seen.append(("GET", url, None))
        if self._error is not None:
            raise self._error
        if callable(self._get):
            return self._get()
        return self._get


def _search_payload(n=3):
    return {"success": True, "data": [
        {"title": f"T{i}", "url": f"https://example.com/{i}",
         "description": f"snippet {i}"}
        for i in range(n)
    ]}


async def test_search_empty_query_short_circuits(monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("no HTTP for blank queries")

    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", _boom)
    out = await search("   ")
    assert out["success"] is False


async def test_search_posts_query_with_scrape_options(monkeypatch):
    client = FakeClient(post=FakeResponse(_search_payload(2)))
    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setenv("FIRECRAWL_URL", "http://localhost:3002")
    out = await search("diabetes guidelines", limit=5)
    assert out["success"] is True
    method, url, body = client.seen[0]
    assert url.endswith("/v2/search")
    assert body["query"] == "diabetes guidelines"
    assert body["limit"] == 5
    assert body["scrapeOptions"]["formats"] == ["markdown"]


async def test_search_raises_on_transport_error(monkeypatch):
    import pytest

    client = FakeClient(error=httpx.ConnectError("refused"))
    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", lambda **kwargs: client)
    with pytest.raises(httpx.ConnectError):
        await search("q")


async def test_scrape_empty_url_short_circuits():
    out = await scrape("")
    assert out == {"success": False, "error": "Empty URL"}


async def test_scrape_posts_full_options(monkeypatch):
    payload = {"success": True, "data": {"markdown": "# Hi\n\nBody text here.",
                                         "metadata": {"title": "Hi"}}}
    client = FakeClient(post=FakeResponse(payload))
    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", lambda **kwargs: client)
    out = await scrape("https://example.com/x", wait_for=8000)
    assert out["success"] is True
    method, url, body = client.seen[0]
    assert url.endswith("/v1/scrape")
    assert body["url"] == "https://example.com/x"
    assert body["waitFor"] == 8000
    assert body["formats"] == ["markdown"]
    assert body["onlyMainContent"] is True


def _batch_client(monkeypatch, items, statuses=("scraping", "completed")):
    """Fake batch endpoint: submit -> {id}, then status sequence, then data."""
    calls = {"n": 0}

    def _status():
        calls["n"] += 1
        status = statuses[min(calls["n"] - 1, len(statuses) - 1)]
        if status in ("completed", "failed"):
            return FakeResponse({"success": True, "status": status,
                                 "data": items})
        return FakeResponse({"success": True, "status": status, "data": []})

    client = FakeClient(post=FakeResponse({"success": True, "id": "job-1"}),
                        get=_status)
    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", lambda **kwargs: client)
    return client


def _batch_item(i):
    return {"markdown": f"# T{i}\n\nBody {i}.",
            "metadata": {"title": f"T{i}",
                         "sourceURL": f"https://example.com/{i}"}}


async def test_scrape_many_batches_one_job(monkeypatch):
    client = _batch_client(monkeypatch, [_batch_item(0), _batch_item(1)])
    out = await scrape_many(["https://example.com/0", "https://example.com/1"])
    assert [r["url"] for r in out] == ["https://example.com/0",
                                       "https://example.com/1"]
    assert "Body 0." in out[0]["markdown"]
    method, url, body = client.seen[0]
    assert url.endswith("/v2/batch/scrape")
    assert body["urls"] == ["https://example.com/0", "https://example.com/1"]
    assert body["formats"] == ["markdown"]
    assert body["maxAge"] == 172800000
    assert any(m == "GET" for m, _, _ in client.seen)


async def test_scrape_many_dedupes_and_caps(monkeypatch):
    from src.tools.firecrawl import MAX_SCRAPE_URLS
    items = [_batch_item(i) for i in range(5)]
    _batch_client(monkeypatch, items)
    urls = ["https://example.com/0", "https://example.com/0"]
    urls += [f"https://example.com/{i}" for i in range(MAX_SCRAPE_URLS + 3)]
    out = await scrape_many(urls)
    assert len(out) == MAX_SCRAPE_URLS
    assert len({r["url"] for r in out}) == MAX_SCRAPE_URLS


async def test_scrape_many_job_failure_reports_per_url(monkeypatch):
    client = FakeClient(error=httpx.ConnectError("refused"))
    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", lambda **kwargs: client)
    out = await scrape_many(["https://example.com/a", "https://example.com/b"])
    assert [r["url"] for r in out] == ["https://example.com/a",
                                       "https://example.com/b"]
    assert all(r["success"] is False and r["error"] for r in out)


async def test_scrape_many_missing_items_reported(monkeypatch):
    _batch_client(monkeypatch, [_batch_item(0)])
    out = await scrape_many(["https://example.com/0", "https://example.com/9"])
    assert out[0]["markdown"].startswith("# T0")
    assert out[1] == {"url": "https://example.com/9", "success": False,
                      "error": "missing from batch response"}


async def test_scrape_many_empty():
    assert await scrape_many([]) == []


async def test_scrape_many_maps_markdown_to_text(monkeypatch):
    """Scraped markdown must reach the judge as text (it only reads
    text/snippet/content)."""
    _batch_client(monkeypatch, [_batch_item(0)])
    out = await scrape_many(["https://example.com/0"])
    assert "Body 0." in out[0]["markdown"]
    assert "Body 0." in out[0]["text"]
    assert out[0]["url"] == "https://example.com/0"


async def test_scrape_many_lifts_nested_data_envelope(monkeypatch):
    """API versions wrapping results as {success, data: {...}} must match
    by the nested URL and lift the nested markdown."""
    nested = {"success": True,
              "data": {"markdown": "# N\n\nNested body.",
                       "metadata": {"sourceURL": "https://example.com/n"}}}
    _batch_client(monkeypatch, [nested])
    out = await scrape_many(["https://example.com/n"])
    assert out[0]["url"] == "https://example.com/n"
    assert "Nested body." in out[0]["text"]


async def test_scrape_many_nested_failure_stays_failure(monkeypatch):
    """An explicit page failure must not gain invented text."""
    nested = {"success": False,
              "data": {"markdown": "# N\n\nNested body.",
                       "metadata": {"sourceURL": "https://example.com/f"}}}
    _batch_client(monkeypatch, [nested])
    out = await scrape_many(["https://example.com/f"])
    assert out[0]["url"] == "https://example.com/f"
    assert out[0]["success"] is False
    assert "text" not in out[0]


async def test_search_listings_normalizes(monkeypatch):
    async def fake_search(query, limit=6, **kwargs):
        assert query == "salt guidelines"
        return _search_payload(2)

    monkeypatch.setattr(firecrawl_mod, "search", fake_search)
    data = await search_listings("salt guidelines")
    assert data["available"] is True
    assert [r["title"] for r in data["results"]] == ["T0", "T1"]
    assert data["results"][0]["snippet"] == "snippet 0"


async def test_search_listings_failure_reports_unavailable(monkeypatch):
    async def fake_search(query, limit=6, **kwargs):
        return {"success": False, "error": "no provider"}

    monkeypatch.setattr(firecrawl_mod, "search", fake_search)
    data = await search_listings("q")
    assert data["available"] is False
    assert data["results"] == []

    async def boom(query, limit=6, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(firecrawl_mod, "search", boom)
    data = await search_listings("q")
    assert data["available"] is False
    assert data["results"] == []


async def test_web_search_mybib_gates_then_scrapes(monkeypatch):
    from src.tools.firecrawl import web_search

    async def fake_search(query, limit=10, **kwargs):
        return _search_payload(3)

    async def fake_verify(items, **kwargs):
        kept = [i for i in items if "example.com/2" not in i["url"]]
        return kept, SimpleNamespace(results=[
            SimpleNamespace(url=i["url"],
                            trustworthy="example.com/2" not in i["url"],
                            reason="stubbed",
                            model_dump=lambda i=i: {
                                "url": i["url"],
                                "trustworthy": "example.com/2" not in i["url"],
                                "reason": "stubbed"})
            for i in items
        ])

    async def fake_scrape_many(urls, **kwargs):
        return [{"url": u, "success": True,
                 "data": {"markdown": f"# {u}\n\nBody.", "metadata": {}}}
                for u in urls]

    monkeypatch.setattr(firecrawl_mod, "search", fake_search)
    monkeypatch.setattr(firecrawl_mod, "verify_urls_mybib", fake_verify)
    monkeypatch.setattr(firecrawl_mod, "scrape_many", fake_scrape_many)
    data = json.loads(await web_search(SimpleNamespace(), "DASH diet"))
    assert data["available"] is True
    assert len(data["results"]) == 3
    assert data["dropped"] == 1
    assert [p["url"] for p in data["pages"]] == [
        "https://example.com/0", "https://example.com/1"]
    assert [v["url"] for v in data["trust"] if v["trustworthy"]] == [
        "https://example.com/0", "https://example.com/1"]


async def test_web_search_failure_reports_unavailable(monkeypatch):
    from src.tools.firecrawl import web_search

    async def boom(query, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(firecrawl_mod, "search", boom)
    data = json.loads(await web_search(SimpleNamespace(), "q"))
    assert data["available"] is False


async def test_web_search_handles_cloud_dict_shaped_data(monkeypatch):
    # Regression: hosted /v2/search returns data as {"web": [...]}.
    # Iterating the dict yielded string keys and crashed the trust gate
    # with str object has no attribute get, failing the whole web leg.
    from src.tools.firecrawl import web_search

    web = [
        {"title": f"T{i}", "url": f"https://example.com/{i}",
         "description": f"snippet {i}", "markdown": f"# T{i} Body."}
        for i in range(3)
    ]

    async def fake_search(query, limit=10, **kwargs):
        return {"success": True, "data": {"web": web}}

    async def fake_verify(items, **kwargs):
        assert all(isinstance(i, dict) for i in items)
        assert [i["url"] for i in items] == [
            f"https://example.com/{i}" for i in range(3)]
        return items, SimpleNamespace(results=[])

    async def fake_scrape_many(urls, **kwargs):
        return [{"url": u, "success": True, "text": f"Body of {u}"}
                for u in urls]

    monkeypatch.setattr(firecrawl_mod, "search", fake_search)
    monkeypatch.setattr(firecrawl_mod, "verify_urls_mybib", fake_verify)
    monkeypatch.setattr(firecrawl_mod, "scrape_many", fake_scrape_many)
    data = json.loads(await web_search(
        SimpleNamespace(), "sodium potassium hypertension"))
    assert data["available"] is True
    assert len(data["results"]) == 3
    assert len(data["pages"]) == 3
    assert data["dropped"] == 0


async def test_search_listings_handles_cloud_dict_shaped_data(monkeypatch):
    async def fake_search(query, limit=6, **kwargs):
        return {"success": True, "data": {"web": [
            {"title": "T0", "url": "https://example.com/0",
             "description": "snippet 0"},
        ], "news": "not-a-list"}}

    monkeypatch.setattr(firecrawl_mod, "search", fake_search)
    data = await search_listings("salt guidelines")
    assert data["available"] is True
    assert [r["url"] for r in data["results"]] == ["https://example.com/0"]


def test_search_result_items_prefers_web_and_skips_junk():
    from src.tools.firecrawl import _search_result_items

    assert _search_result_items({"success": True, "data": [
        {"url": "https://a.example/"}, "junk-string", None, 42,
    ]}) == [{"url": "https://a.example/"}]
    out = _search_result_items({"success": True, "data": {
        "news": [{"url": "https://n.example/"}],
        "web": [{"url": "https://w.example/"}],
        "count": 2,
    }})
    assert [r["url"] for r in out] == [
        "https://w.example/", "https://n.example/"]
    assert _search_result_items({"success": True, "data": None}) == []
    assert _search_result_items({"success": False, "error": "x"}) == []


async def test_fetch_urls_accepts_json_string(monkeypatch):
    from src.tools.firecrawl import firecrawl_fetch_urls

    async def fake_scrape_many(urls, **kwargs):
        return [{"url": u, "success": True} for u in urls]

    monkeypatch.setattr(firecrawl_mod, "scrape_many", fake_scrape_many)
    data = json.loads(await firecrawl_fetch_urls(
        SimpleNamespace(), json.dumps(["https://example.com/a"])))
    assert data["urls"] == ["https://example.com/a"]
    assert data["results"][0]["success"] is True


def test_build_deep_agent_registers_web_tools(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-qwen")
    agent = build_deep_agent()
    names = [name for toolset in agent.toolsets for name in toolset.tools]
    assert "web_search" in names
    assert "local_search" in names


def test_batch_item_url_reads_v2_flat_metadata():
    from src.tools.firecrawl import _batch_item_url

    item = {"markdown": "# Hi", "metadata": {
        "sourceURL": "https://example.com/a",
        "url": "https://example.com/a/", "title": "Hi"}}
    assert _batch_item_url(item, "fallback") == "https://example.com/a"
    assert _batch_item_url({"markdown": "x"}, "fallback") == "fallback"


async def test_scrape_many_v2_rejection_uses_individual_fallback(monkeypatch):
    client = FakeClient(post=FakeResponse(
        {"success": False, "error": "plan limit"}))
    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", lambda **kwargs: client)

    async def fake_individual(urls, options, batch_error=""):
        assert "plan limit" in batch_error
        return [{"url": u, "success": True, "text": "Body"} for u in urls]

    monkeypatch.setattr(firecrawl_mod, "_scrape_individually", fake_individual)
    out = await scrape_many(["https://example.com/a"])
    assert out[0]["text"] == "Body"


async def test_scrape_many_reports_invalid_urls(monkeypatch):
    client = FakeClient(
        post=FakeResponse({"success": True, "id": "job-9",
                           "invalidURLs": ["https://bad.example/x"]}),
        get=FakeResponse({"success": True, "status": "completed", "data": []}),
    )
    monkeypatch.setattr(firecrawl_mod.httpx, "AsyncClient", lambda **kwargs: client)
    out = await scrape_many(["https://bad.example/x"])
    assert out[0]["success"] is False
    assert "invalid" in out[0]["error"]


async def test_scrape_many_translates_snake_kwargs(monkeypatch):
    # v2 rejects unknown payload keys with 400, so caller overrides pass
    # through a strict allow-list (snake_case translated, junk dropped).
    client = _batch_client(monkeypatch, [_batch_item(0)])
    out = await scrape_many(["https://example.com/0"],
                            only_main_content=False, wait_for=9000,
                            bogus_key=1)
    assert out[0]["url"] == "https://example.com/0"
    method, url, body = client.seen[0]
    assert url.endswith("/v2/batch/scrape")
    assert body["onlyMainContent"] is False
    assert body["waitFor"] == 9000
    assert body["maxAge"] == 172800000
    assert "only_main_content" not in body
    assert "bogus_key" not in body
