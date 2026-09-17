"""MedRAG API: one route that streams a deep-research run as NDJSON.

Run without autoreload (default; keeps the retrieval models warm):
    python api.py
    # or:  python -m uvicorn api:app --port 8000
Run with autoreload (dev - each .py edit restarts the worker and reloads the
retrieval models, so use it only while editing):
    python api.py --reload
    # or:  python -m uvicorn api:app --port 8000 --reload
The frontend dev server proxies /v1 here (see frontend/vite.config.ts).
"""

from __future__ import annotations

import json
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv

from src.lib.trace import get_trace
from src.agents.stream_adapter import stream_deep_agent
from src.agents.ag_ui_endpoint import handle_ag_ui
from src.agents.clarify import submit_answer
from src.runstate.store import get_store

load_dotenv()

app = FastAPI(title="MedRAG API")
# No cookies are used, so credentialed cross-origin requests are never
# legitimate: "*" + credentials is the unsafe CORS combination.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Optional bearer gate: inert unless MEDRAG_API_TOKEN is set, so existing
# clients keep working while a deployment can require a token.
@app.middleware("http")
async def _require_token(request: Request, call_next):
    token = os.environ.get("MEDRAG_API_TOKEN", "").strip()
    if token and request.headers.get("authorization", "") != f"Bearer {token}":
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.on_event("startup")
async def _startup() -> None:
    import os as _os

    print(f"[api] pid {_os.getpid()} starting (reload restarts change this pid)",
          flush=True)
    get_trace().open_stream("logs.txt", query="api-server")
    _preload_retriever()


def _preload_retriever() -> None:
    """Load the retrieval models (query encoder + cross-encoder) once at boot.

    get_retriever() is a process singleton, so the weights are loaded a single
    time and then reused by every request; doing it here instead of on the
    first retrieval keeps that first request fast. The load runs in a daemon
    thread so the server accepts connections immediately - a request that
    arrives mid-load simply waits on the retriever's build lock.

    Set RETRIEVER_PRELOAD=0 to skip (tests, or a process that never retrieves).
    """
    import os
    import threading

    if os.environ.get("RETRIEVER_PRELOAD", "1").strip().lower() in ("0", "false", "no"):
        return

    def _load() -> None:
        try:
            from src.tools.retrieval import get_retriever

            ready = get_retriever().warmup(
                lambda stage: print(f"[retriever] {stage}", flush=True))
            print(f"[retriever] preload {'ready' if ready else 'PARTIAL'}", flush=True)
        except Exception as exc:  # noqa: BLE001 - preload must never crash boot
            print(f"[retriever] preload failed: {exc}", flush=True)

    threading.Thread(target=_load, name="retriever-preload", daemon=True).start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    get_trace().close_stream()


class ChatRequest(BaseModel):
    question: str
    conversation_id: str = ""
    history: list = []
    engine: str = ""


class QuestionAnswerRequest(BaseModel):
    """Answer payload for one human-in-the-loop clarification question."""

    selections: list[str] = []
    other: str = ""
    find_all: bool = False


@app.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest):
    """Stream one research run as newline-delimited JSON (see stream_adapter)."""
    t0 = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    get_trace().log(
        "chat_stream_start", run_id=run_id,
        question=body.question[:160], engine=body.engine or "deep",
    )

    async def _gen():
        # The chat owns one state JSON; every request appends a turn to
        # it (GET /v1/chats/{chat_id}). Without a conversation id each
        # request is its own single-turn chat.
        chat_id = (body.conversation_id or "").strip() or run_id
        async for event in stream_deep_agent(body.question, run_id=run_id,
                                             conversation_id=chat_id):
            yield json.dumps(event, ensure_ascii=False) + "\n"
        get_trace().log(
            "chat_stream_end", run_id=run_id,
            elapsed_s=round(time.perf_counter() - t0, 2),
        )

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


@app.post("/v1/ag-ui")
async def ag_ui(request: Request):
    """AG-UI protocol endpoint (SSE). See src/agents/ag_ui_endpoint.py."""
    return await handle_ag_ui(request)


@app.get("/v1/health")
async def health():
    """Process id + retriever residency.

    Curl this before and after a run: if pid is unchanged and the retriever
    flags are true, the weights are loaded once and reused. A new pid on every
    request means the process is being restarted (e.g. a reloader), and a flag
    flipping false->true on a run means that run paid the load.
    """
    import os as _os

    out: dict = {"pid": _os.getpid()}
    try:
        from src.tools.retrieval import get_retriever

        out["retriever"] = get_retriever().status()
    except Exception as exc:  # noqa: BLE001 - health must never 500
        out["retriever"] = {"error": str(exc)}
    return out


@app.get("/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Return the per-chat state JSON (every turn in the conversation)."""
    chat = get_store().get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"unknown chat {chat_id}")
    return json.loads(chat.to_json())


@app.post("/v1/chats/{chat_id}/questions/{question_id}/answer")
async def answer_question(chat_id: str, question_id: str,
                          body: QuestionAnswerRequest):
    """Deliver the user's answer to a parked ask_user call (HITL).

    Resolves the waiting tool inside the paused deep-agent run, which
    then continues streaming. Returns 404 when no question is waiting
    (e.g. the run already timed out or moved on)."""
    resolved = submit_answer(
        chat_id, question_id,
        {"selections": body.selections, "other": body.other,
         "find_all": body.find_all},
    )
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"no pending question {question_id} in chat {chat_id}",
        )
    return {"ok": True, "question_id": question_id, "chat_id": chat_id}


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Run the MedRAG API (no autoreload by default)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Autoreload on code changes (dev). Off by default: a reload "
             "restarts the worker and reloads the retrieval models.",
    )
    args = parser.parse_args()

    # NB: reload requires the app as an import string, not the object.
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)
