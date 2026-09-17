"""Unit tests for src.embedding — the OpenAI-compatible client and CLI.

No DB, no network: httpx.MockTransport fakes the /v1/embeddings endpoint so
we exercise response parsing, input ordering, L2 normalization, dimension
guards, auth headers and retry behavior against a synthetic server.
"""

from __future__ import annotations

import json
import os

import httpx
import numpy as np
import pytest

from src.embedding import cli as embed_cli
from src.embedding import store as embed_store
from src.embedding.client import EmbedClient

DIM = 768


def _vec(seed: float) -> list:
    """A 768-dim vector with a distinct direction per seed (not parallel)."""
    out = [0.0] * DIM
    out[0] = seed
    out[1] = seed + 1.0
    return out


def _norm(seed: float) -> float:
    return float((seed ** 2 + (seed + 1.0) ** 2) ** 0.5)


def _openai_response(texts):
    """Server returns items in REVERSE input order on purpose."""
    n = len(texts)
    data = [
        {"object": "embedding", "index": i,
         "embedding": _vec(float(i + 1))}
        for i in reversed(range(n))
    ]
    return {"object": "list", "data": data, "model": "ncbi/MedCPT-Article-Encoder",
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


def _make_client(handler, **kw) -> EmbedClient:
    transport = httpx.MockTransport(handler)
    cli = EmbedClient(
        base_url="https://fake-embed/v1",
        model="ncbi/MedCPT-Article-Encoder",
        retries=kw.pop("retries", 0),
        normalize=kw.pop("normalize", True),
        **kw,
    )
    cli._client = httpx.Client(
        base_url="https://fake-embed/v1",
        timeout=httpx.Timeout(30.0),
        transport=transport,
        headers=cli._client.headers,
    )
    return cli


def test_embed_returns_input_order_and_normalizes():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        body = json.loads(request.content)
        return httpx.Response(200, json=_openai_response(body["input"]))

    cli = _make_client(handler)
    vecs = cli.embed(["alpha", "beta", "gamma"])
    cli.close()

    assert vecs.shape == (3, DIM)
    assert vecs.dtype == np.float32
    # input order preserved despite reversed server response
    for i in range(3):
        seed = float(i + 1)
        assert vecs[i][0] == pytest.approx(seed / _norm(seed), abs=1e-6)
        assert vecs[i][1] == pytest.approx((seed + 1.0) / _norm(seed), abs=1e-6)
    # L2-normalized
    norms = np.linalg.norm(vecs, axis=1)
    assert norms == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    # payload shape
    assert calls[0]["model"] == "ncbi/MedCPT-Article-Encoder"
    assert calls[0]["input"] == ["alpha", "beta", "gamma"]


def test_no_normalize_keeps_raw_vectors():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=_openai_response(body["input"]))

    cli = _make_client(handler, normalize=False)
    vecs = cli.embed(["a"])
    cli.close()
    # raw [1.0, 2.0, ...] untouched
    assert vecs[0][0] == pytest.approx(1.0)
    assert vecs[0][1] == pytest.approx(2.0)
    assert np.linalg.norm(vecs[0]) != pytest.approx(1.0)


def test_dimension_mismatch_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "object": "list",
            "data": [{"object": "embedding", "index": 0,
                      "embedding": [0.1, 0.2, 0.3, 0.4]}],
        })

    cli = _make_client(handler)
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        cli.embed(["boom"])
    cli.close()


def test_bad_response_length_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # only one item for a two-text request
        return httpx.Response(200, json={"object": "list", "data": [
            {"object": "embedding", "index": 0, "embedding": _vec(1.0)}]})

    cli = _make_client(handler)
    with pytest.raises(RuntimeError, match="expected 2 items"):
        cli.embed(["a", "b"])
    cli.close()


def test_retry_on_503_then_success():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, text="service unavailable")
        body = json.loads(request.content)
        return httpx.Response(200, json=_openai_response(body["input"]))

    cli = _make_client(handler, retries=2)
    vecs = cli.embed(["after-retry"])
    cli.close()
    assert attempts["n"] == 2
    assert vecs.shape == (1, DIM)


def test_exhausted_retries_raise():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, text="still down")

    cli = _make_client(handler, retries=1)
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        cli.embed(["x"])
    cli.close()
    assert attempts["n"] == 2


def test_api_key_sets_bearer_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_openai_response(["t"]))

    cli = _make_client(handler, api_key="sekrit")
    cli.embed(["t"])
    cli.close()
    assert seen["auth"] == "Bearer sekrit"


def test_health_probe_strips_v1():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "status": "ok", "model": "ncbi/MedCPT-Article-Encoder",
                "gpus": 2, "embedding_dimension": 768, "batch_size": 32})
        return httpx.Response(200, json=_openai_response(["t"]))

    cli = _make_client(handler)
    health = cli.check_health()
    cli.close()
    assert seen["url"].endswith("/health")
    assert health["embedding_dimension"] == 768


def test_health_probe_failure_is_advisory(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    cli = _make_client(handler)
    assert cli.check_health() is None
    cli.close()
    assert "health probe failed" in capsys.readouterr().out


def test_upsert_stores_full_precision_vector():
    assert "::vector" in embed_store.UPSERT_EMBEDDING
    assert "halfvec" not in embed_store.UPSERT_EMBEDDING


def test_hash_and_format_vec():
    h = embed_store.hash_text("hello world")
    assert len(h) == 64
    vec = np.array([1.0, 0.5], dtype=np.float32)
    lit = embed_store.format_vector(vec)
    assert lit.startswith("[") and lit.endswith("]")
    assert len(lit.split(",")) == 2
    assert "0.5" in lit


def test_empty_text_swapped_for_space():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["input"] = body["input"]
        return httpx.Response(200, json=_openai_response(body["input"]))

    cli = _make_client(handler)
    cli.embed(["", "real text"])
    cli.close()
    assert seen["input"] == [" ", "real text"]


# -- .env loading (EMBEDDING_BASE_URL is not picked up by make) ----------

def test_env_file_supplies_embedding_config(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "EMBEDDING_BASE_URL=https://from-file/v1\n"
        "MEDPAT_EMBED_MODEL=from-file-model\n"
        "EMBEDDING_API_KEY=from-file-key\n"
    )
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("MEDPAT_EMBED_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    embed_cli._load_env_file(env)
    assert os.environ["EMBEDDING_BASE_URL"] == "https://from-file/v1"
    assert os.environ["MEDPAT_EMBED_MODEL"] == "from-file-model"
    assert os.environ["EMBEDDING_API_KEY"] == "from-file-key"


def test_env_file_does_not_override_exported_value(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("EMBEDDING_BASE_URL=https://from-file/v1\n")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://from-env/v1")
    embed_cli._load_env_file(env)
    assert os.environ["EMBEDDING_BASE_URL"] == "https://from-env/v1"


def test_empty_exported_value_is_filled_from_env_file(tmp_path, monkeypatch):
    # A stale `export EMBEDDING_BASE_URL=` in the shell must not hide .env.
    env = tmp_path / ".env"
    env.write_text("EMBEDDING_BASE_URL=https://from-file/v1\n")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    embed_cli._load_env_file(env)
    assert os.environ["EMBEDDING_BASE_URL"] == "https://from-file/v1"


def test_missing_env_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    embed_cli._load_env_file(tmp_path / "does-not-exist.env")
    assert "EMBEDDING_BASE_URL" not in os.environ


# -- empty flag values fall back to .env / env ---------------------------

def test_empty_base_url_flag_falls_back_to_env(monkeypatch):
    # The user's case: --base-url "$EMBEDDING_BASE_URL" with the shell var
    # unset must NOT clobber the .env value.
    from types import SimpleNamespace
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://from-env/v1")
    args = SimpleNamespace(base_url="", model="", api_key=None, dsn="")
    embed_cli._normalize_args(args)
    assert args.base_url == "https://from-env/v1"
    assert args.model == "ncbi/MedCPT-Article-Encoder"
    assert args.api_key == ""


def test_whitespace_base_url_is_treated_as_unset(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://from-env/v1")
    args = SimpleNamespace(base_url="   ", model="m", api_key="", dsn="d")
    embed_cli._normalize_args(args)
    assert args.base_url == "https://from-env/v1"


def test_nonempty_flags_win_over_env(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://from-env/v1")
    monkeypatch.setenv("MEDPAT_EMBED_MODEL", "env-model")
    args = SimpleNamespace(base_url="https://from-flag/v1",
                           model="flag-model", api_key="secret",
                           dsn="dsn-from-flag")
    embed_cli._normalize_args(args)
    assert args.base_url == "https://from-flag/v1"
    assert args.model == "flag-model"
    assert args.api_key == "secret"
    assert args.dsn == "dsn-from-flag"


# -- streaming store reads -----------------------------------------------

class _FakeCursor:
    """Minimal DB-API cursor: scripted pages, respects the LIMIT param."""

    def __init__(self, pages):
        self._pages = list(pages)
        self._current = []
        self.params = []

    def execute(self, sql, params=None):
        self.params.append(params)
        page = self._pages.pop(0) if self._pages else []
        n = params[-1] if params else None
        self._current = page[:n] if isinstance(n, int) else page

    def fetchall(self):
        return self._current

    def fetchone(self):
        return self._current[0] if self._current else None

    def close(self):
        pass


class _FakeConn:
    def __init__(self, pages):
        self.cursor_obj = _FakeCursor(pages)

    def cursor(self):
        return self.cursor_obj


def test_iter_pending_chunks_paginates_with_keyset():
    pages = [
        [("a", "t1", "", 0), ("b", "t2", "", 1)],
        [("c", "t3", "", 1)],
    ]
    conn = _FakeConn(pages)
    batches = list(embed_store.iter_pending_chunks(conn, batch_size=2))
    assert batches == [[("a", "t1", ""), ("b", "t2", "")], [("c", "t3", "")]]
    # page 1 starts at the beginning; page 2 advances past page 1's last row
    assert conn.cursor_obj.params[0] == (-1, "", 2)
    assert conn.cursor_obj.params[1] == (1, "b", 2)


def test_iter_pending_chunks_respects_limit():
    pages = [
        [("a", "t1", "", 0), ("b", "t2", "", 1)],
        [("c", "t3", "", 1), ("d", "t4", "", 2)],
    ]
    conn = _FakeConn(pages)
    batches = list(embed_store.iter_pending_chunks(conn, batch_size=2, limit=3))
    assert [r[0] for b in batches for r in b] == ["a", "b", "c"]
    assert conn.cursor_obj.params[-1] == (1, "b", 1)   # only 1 more requested


def test_count_pending_chunks():
    conn = _FakeConn([[(5,)]])
    assert embed_store.count_pending_chunks(conn) == 5


def test_embed_reports_stages():
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=_openai_response(body["input"]))

    cli = _make_client(handler)
    cli.embed(["a", "b"], on_stage=lambda ev, n: events.append((ev, n)))
    cli.close()
    assert events == [("sent", 2), ("recv", 2), ("norm", 2)]
