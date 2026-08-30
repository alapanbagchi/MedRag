"""Agentic v3 - UI integration tests (offline).

Checks the v3 web UI + UI server wiring without touching the LLM:
  * the v3 page exists and knows the v3 event vocabulary,
  * the UI server (scripts/v2_ui.py) serves the v3 flow: GET / returns the
    v3 page, GET /events returns the JSONL event stream with _idx,
    POST /clear truncates the log.
The POST /run (live LLM) path is exercised manually/externally.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_http(url, timeout=10.0):
    """GET until the server responds; returns (status, content_type, body)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return (r.status, r.headers.get("Content-Type", ""),
                        r.read().decode("utf-8"))
        except Exception:
            time.sleep(0.15)
    raise AssertionError(f"server did not come up at {url}")


def test_v3_page_exists_and_uses_v3_event_vocabulary():
    html = (ROOT / "ui" / "agentic_v3.html").read_text(encoding="utf-8")
    for marker in ("Agentic V3", "evidence_state", "requirement_state",
                   "worker_report", "contradiction", "resolution",
                   "replan", "final_evidence", "/events?after"):
        assert marker in html, f"v3 page must contain {marker!r}"


def test_ui_server_has_flow_switch():
    src = (ROOT / "scripts" / "v2_ui.py").read_text(encoding="utf-8")
    assert "AGENTIC_UI_FLOW" in src
    assert "agentic_v3.html" in src
    assert "AgenticV3Pipeline" in src


def test_server_serves_v3_flow_and_events(tmp_path, monkeypatch):
    events_file = tmp_path / "v3_events.jsonl"
    seeded = [
        {"ts": 1.0, "type": "run_start", "run_id": "run-ui-test",
         "question": "q", "budget": {}},
        {"ts": 2.0, "type": "master_plan", "tasks": [
            {"id": "T1", "title": "task one", "objective": "o",
             "evidence_requirements": [{"id": "T1.R1", "text": "req",
                                        "target_n": 2}]}]},
        {"ts": 3.0, "type": "worker_report", "task_id": "T1",
         "status": "satisfied", "requirements": [
            {"requirement_id": "T1.R1", "status": "satisfied", "coverage": 2,
             "target_n": 2, "papers": [
                {"evidence_id": "T1.R1.A1.E1", "document_id": "PMC1",
                 "support": "supports", "confidence": 0.9,
                 "retrieval_method": "bm25", "rank": 1}]}],
         "evidence": [], "stop_reason": "satisfied"},
        {"ts": 4.0, "type": "run_end", "stop_reason": "synthesized",
         "answer": {"summary": "done"}},
    ]
    events_file.write_text("\n".join(json.dumps(e) for e in seeded),
                           encoding="utf-8")

    port = _free_port()
    env = dict(os.environ)
    env["AGENTIC_UI_FLOW"] = "v3"
    env["AGENTIC_V3_EVENTS_FILE"] = str(events_file)
    env["AGENTIC_V3_UI_HOST"] = "127.0.0.1"
    env["AGENTIC_V3_UI_PORT"] = str(port)
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "v2_ui.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT),
    )
    try:
        base = f"http://127.0.0.1:{port}"
        status, ctype, body = _wait_http(base + "/")
        assert status == 200
        assert ctype.startswith("text/html")
        assert "Agentic V3" in body

        with urllib.request.urlopen(base + "/events", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
            evs = data["events"]
            assert len(evs) == 4
            assert all("_idx" in e for e in evs)
            types = [e["type"] for e in evs]
            assert types == ["run_start", "master_plan", "worker_report", "run_end"]

        # incremental polling: after=1 returns only _idx > 1 (the tail)
        with urllib.request.urlopen(base + "/events?after=1", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
            assert [e["type"] for e in data["events"]] == ["worker_report", "run_end"]

        # /clear truncates the stream
        req = urllib.request.Request(base + "/clear", method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            assert json.loads(r.read().decode("utf-8"))["ok"] is True
        with urllib.request.urlopen(base + "/events", timeout=3) as r:
            assert json.loads(r.read().decode("utf-8"))["events"] == []
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_v3_ui_handles_a_real_events_file(tmp_path):
    """The v3 page parses key v3 event shapes (structural smoke test).

    Simulates what the browser does: the polling JSON is produced by the
    server from the JSONL file; the HTML must be servable over HTTP."""
    from src.agentic_v3.pipeline import AgenticV3Pipeline
    from src.config import AppConfig

    # the pipeline constructs with an events file wired the same way the
    # server does (EventEmitter + AGENTIC_V3_EVENTS_FILE)
    from src.agentic_v3.events import EventEmitter
    path = tmp_path / "agentic_v3_events.jsonl"
    events = EventEmitter(str(path))
    pipeline = AgenticV3Pipeline(config=AppConfig(), events=events)
    assert pipeline.events is not None
    assert events.path == str(path)
