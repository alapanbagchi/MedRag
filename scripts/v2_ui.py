#!/usr/bin/env python
"""Agentic UI server (stdlib only) - serves the v2 OR the v3 trajectory.

Serves the flow's HTML page and exposes three endpoints:

  GET  /events   -> the structured event stream (JSONL) as JSON, with _idx
                    line indexes so the browser can poll incrementally.
  POST /run      -> run the pipeline for a question in a background thread
                    (writing events to the same file the UI reads).
  POST /clear    -> truncate the event log.

Flow selection (env AGENTIC_UI_FLOW, default "v2" so existing usage is
unchanged):

  v2: serves ui/agentic_v2.html, runs src.agentic_v2
      (AGENTIC_V2_EVENTS_FILE, AGENTIC_V2_UI_HOST, AGENTIC_V2_UI_PORT)
  v3: serves ui/agentic_v3.html, runs src.agents
      (AGENTIC_V3_EVENTS_FILE, AGENTIC_V3_UI_HOST, AGENTIC_V3_UI_PORT)

Usage:
  python scripts/v2_ui.py                         # v2  -> http://127.0.0.1:8090
  AGENTIC_UI_FLOW=v3 python scripts/v2_ui.py      # v3  -> http://127.0.0.1:8091

The CLI can also feed the same UI directly - run the pipeline with
AGENTIC_*_EVENTS_FILE=...jsonl and leave this server open.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # make src importable from the scripts dir

from src.prompts.load import load_prompt  # noqa: E402  (needs ROOT on sys.path)

FLOW = "v3"  # v3-only; the v2 agentic flow was removed

if FLOW == "v3":
    UI_FILE = ROOT / "ui" / "agentic_v3.html"
    EVENTS_FILE = os.environ.get("AGENTIC_V3_EVENTS_FILE",
                                 str(ROOT / "agentic_v3_events.jsonl"))
    HOST = os.environ.get("AGENTIC_V3_UI_HOST", "127.0.0.1")
    PORT = int(os.environ.get("AGENTIC_V3_UI_PORT", "8091"))
else:  # v2 (default; existing behaviour unchanged)
    UI_FILE = ROOT / "ui" / "agentic_v2.html"
    EVENTS_FILE = os.environ.get("AGENTIC_V2_EVENTS_FILE",
                                 str(ROOT / "agentic_v2_events.jsonl"))
    HOST = os.environ.get("AGENTIC_V2_UI_HOST", "127.0.0.1")
    PORT = int(os.environ.get("AGENTIC_V2_UI_PORT", "8090"))

# Agent (llm-observer agent name) -> prompt file, for the UI's per-agent
# "System prompt" box. Covers every flow; the UI shows whichever it needs.
SYSTEM_PROMPT_AGENTS = {
    "master_orchestrator": ("agents", "master.txt"),
    "search_term_planner": ("agents", "search_planner.txt"),
    "critic": ("agents", "critic.txt"),
    "replanner": ("agents", "replanner.txt"),
    "paper_inspector": ("agents", "deep_inspector.txt"),
    "contradiction_agent": ("agents", "contradiction.txt"),
    "resolution_agent": ("agents", "resolution.txt"),
    "synthesizer_v3": ("agents", "synthesize.txt"),
    "planner": ("agentic_v1", "planner.txt"),
    "orchestrator": ("agentic_v2", "orchestrator.txt"),
    "verifier": ("agentic_v2", "verifier.txt"),
    "synthesizer": ("agentic_v2", "synthesize.txt"),
}


def _system_prompts() -> dict:
    """{agent: {file, prompt}} read live from the prompt tree (PROMPT_DIR aware)."""
    out = {}
    for agent, (subdir, fname) in SYSTEM_PROMPT_AGENTS.items():
        try:
            out[agent] = {"file": f"{subdir}/{fname}",
                          "prompt": load_prompt(subdir, fname)}
        except Exception:
            continue
    return out


def _read_events() -> list:
    """Parse the JSONL event file; attach a stable line index to each record."""
    events = []
    path = Path(EVENTS_FILE)
    if not path.is_file():
        return events
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        record["_idx"] = idx
        events.append(record)
    return events


def _run_pipeline(question: str) -> None:
    """Run the flow's pipeline in this thread, writing events to the shared file."""
    try:
        sys.path.insert(0, str(ROOT))
        # The server is long-lived; Python caches imported modules in
        # sys.modules, so a module loaded before a code change stays stale for
        # every later run. Re-import the flow package fresh on each run so the
        # UI always executes the current code on disk.
        prefixes = ("src.llm.", "src.llm", "src.ingestion.collector", "src.config",
                    "src.lib.trace")
        if FLOW == "v3":
            prefixes += ("src.agents.", "src.agents")
        else:
            prefixes += ("src.agentic_v2.", "src.agentic_v2")
        for mod_name in list(sys.modules):
            if mod_name.startswith(prefixes):
                del sys.modules[mod_name]

        from src.config import AppConfig

        cfg = AppConfig()
        from src.agents.events import EventEmitter
        from src.agents.pipeline import AgenticV3Pipeline

        cfg.agentic_v3_events_file = EVENTS_FILE
        emitter = EventEmitter(EVENTS_FILE)
        emitter.truncate()  # fresh log for this run
        pipeline = AgenticV3Pipeline(config=cfg, events=emitter)
        import asyncio
        asyncio.run(pipeline.answer(question))
    except Exception as exc:  # noqa: BLE001 - surface any startup error to the log
        try:
            emitter.emit("failure", action="RUN", error=str(exc)[:300])
            emitter.emit("run_end", stop_reason=f"run error: {str(exc)[:200]}",
                         iterations=0, terminal=True, confidence=0.0, answer="")
        except Exception:
            pass
        print(f"[{FLOW}_ui] run failed: {exc}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default request logging
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not UI_FILE.is_file():
                self._send_json({"error": f"{UI_FILE.name} missing"}, 500)
                return
            body = UI_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/system_prompts":
            self._send_json(_system_prompts())
            return
        if path == "/events":
            events = _read_events()
            query = urlparse(self.path).query
            after = -1
            for part in query.split("&"):
                if part.startswith("after="):
                    try:
                        after = int(part[len("after="):])
                    except ValueError:
                        after = -1
            if after >= 0:
                events = [e for e in events if e["_idx"] > after]
            self._send_json({"events": events})
            return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/clear":
            Path(EVENTS_FILE).write_text("", encoding="utf-8")
            self._send_json({"ok": True})
            return
        if path == "/run":
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                data = {}
            question = " ".join((data.get("question") or "").split())
            if not question:
                self._send_json({"error": "empty question"}, 400)
                return
            threading.Thread(target=_run_pipeline, args=(question,), daemon=True).start()
            self._send_json({"ok": True, "question": question})
            return
        self.send_error(404)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"agentic {FLOW} UI  ->  http://{HOST}:{PORT}", flush=True)
    print(f"events file    ->  {EVENTS_FILE}", flush=True)
    if FLOW == "v3":
        print("(also usable with: AGENTIC_V3_EVENTS_FILE=agentic_v3_events.jsonl python -m src --agentic-v3 \"<q>\")", flush=True)
    else:
        print("(also usable with: AGENTIC_V2_EVENTS_FILE=agentic_v2_events.jsonl python -m src --agentic-v2 \"<q>\")", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
