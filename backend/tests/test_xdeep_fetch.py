"""Page-fetch tests (deep web evidence): snippets alone are not enough -
the pipeline now fetches the ACTUAL page and uses its text for verification
and reliability. These tests are fully offline (httpx faked)."""

from __future__ import annotations

import asyncio

import pytest

from src.x_deepagents.tools.fetch import extract_html_text


class _FakeResp:
    status_code = 200

    def __init__(self, body: str):
        self._body = body

    @property
    def content(self):
        return self._body.encode("utf-8")

    @property
    def text(self):
        return self._body

    def raise_for_status(self):
        pass




def _with_real_fetch(monkeypatch):
    """Undo the conftest offline pin so the real fetch_page_text runs, while
    httpx is faked in the test body. The real implementation was stashed by
    the conftest fixture as _REAL_FETCH BEFORE any pinning."""
    import src.x_deepagents.tools.fetch as fetch_mod

    real = fetch_mod._REAL_FETCH
    monkeypatch.setattr(fetch_mod, "fetch_page_text", real)
    return real

class _FakeClient:
    def __init__(self, body: str):
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def get(self, url, **kw):
        return _FakeResp(self._body)


def test_extract_html_text_strips_tags_and_scripts():
    html = ("<html><style>.x{}</style><body><h1>T</h1>"
            "<p>Vitamin D lowers BP <b>by 5 mmHg</b>.</p>"
            "<script>malicious()</script></body></html>")
    t = extract_html_text(html)
    assert "Vitamin D lowers BP by 5 mmHg" in t
    assert "malicious" not in t
    assert "<script" not in t


def test_extract_html_text_caps_length():
    html = "<p>" + "word " * 500 + "</p>"
    t = extract_html_text(html, max_chars=300)
    assert len(t) <= 310   # small tolerance at the sentence boundary


def test_fetch_returns_page_text(monkeypatch):
    """The deep fetch returns the actual page text (not the snippet)."""
    import httpx

    body = ("<html><body><article>WHO guideline: the DASH diet reduces "
            "systolic blood pressure by 11 mmHg in hypertensive adults "
            "compared with a control diet.</article></body></html>")
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(body))

    fetch_page_text = _with_real_fetch(monkeypatch)

    text = asyncio.run(fetch_page_text("https://www.who.int/dash"))
    assert "DASH diet reduces systolic blood pressure" in text
    assert text.strip()


def test_fetch_rejects_challenge_pages(monkeypatch):
    """An anti-bot interstitial (reCAPTCHA) is treated as a fetch FAILURE so
    the caller falls back to the snippet instead of using junk as evidence."""
    import httpx

    body = ("<html><body>Checking your browser - reCAPTCHA<br>"
            "Click here if you are not redirected after 5 seconds.</body></html>")
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(body))

    fetch_page_text = _with_real_fetch(monkeypatch)

    text = asyncio.run(fetch_page_text("https://pmc.ncbi.nlm.nih.gov/x"))
    assert text == ""


def test_fetch_empty_on_http_error(monkeypatch):
    """4xx/5xx and network errors degrade to '' (snippet fallback)."""
    import httpx

    class _ErrClient(_FakeClient):
        async def get(self, url, **kw):
            class _R:
                status_code = 403

                def raise_for_status(self):
                    raise RuntimeError("403 Forbidden")
            return _R()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _ErrClient(""))

    fetch_page_text = _with_real_fetch(monkeypatch)

    assert asyncio.run(fetch_page_text("https://x.example.com/blocked")) == ""
