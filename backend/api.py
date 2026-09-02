"""Lightweight MedPat API: streams the AgenticV3 pipeline to the web UI.

Every request and every internal pipeline action (stages, agent calls,
retrievals, verdicts, contradictions, resolutions, answers) is mirrored to
logs.txt (LOG_FILE to override) via the shared streaming trace — the same
feed the CLI writes — so a web run is fully inspectable after the fact.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.lib.trace import get_trace

_DEFAULT_LOG = Path(__file__).resolve().parent / "logs.txt"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    trace = get_trace()
    log_path = os.environ.get("LOG_FILE") or _DEFAULT_LOG
    # append: keep the file across requests + server restarts
    trace.open_stream(log_path, query="(MedPat API server)", append=True)
    try:
        yield
    finally:
        trace.close_stream()


app = FastAPI(title="MedPat API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# pipeline event -> frontend top-level stage (frontend/lib/types.ts)
STAGE = {
    "run_start": "understanding",
    "master_plan": "decomposing",
    "retrieved": "retrieving",
    "verdict": "retrieving",
    "evidence_added": "reranking",
    "answer": "synthesizing",
}

# streamed under their own top-level type; everything else is mirrored
# verbatim as {"type":"pipeline","event":<name>, ...fields} so the UI's
# thinking log shows EVERY pipeline / LLM event (like Logfire does).
_RESERVED = {"status", "sources", "token", "done", "error",
             "memory_prepare", "memory_commit"}

# ---------------------------------------------------------------------------
# Memory + Context layer (src/memory): one MemoryAPI per server process.
# auto backend = Postgres (medrag_memory schema) when reachable, in-memory
# otherwise. MEMORY_ENABLED=0/off disables it entirely.
# ---------------------------------------------------------------------------

_memory_api = None


def get_memory_api():
    """Lazy per-process MemoryAPI singleton (safe to call from any request)."""
    global _memory_api
    if _memory_api is not None:
        return _memory_api
    switch = os.environ.get("MEMORY_ENABLED", "").strip().lower()
    if switch in ("0", "false", "off", "no", "disabled"):
        return None
    try:
        from src.config import AppConfig
        from src.memory.api import MemoryAPI
        from src.memory.config import MemoryConfig
        _memory_api = MemoryAPI.build(MemoryConfig.from_appconfig(AppConfig()))
        print(f"[memory] API attached: backend={_memory_api.config.backend} "
              f"embedder={_memory_api.config.embedder}")
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] attach failed ({exc}); API runs without it")
        _memory_api = None
    return _memory_api


class ChatBody(BaseModel):
    question: str
    conversation_id: str = ""
    history: list = []
    # engine: "" (default) | "v3" | "xdeep" - select the research pipeline.
    # XDEEP_ENGINE=1 env also forces xdeep when the body does not ask.
    engine: str = ""


class _Emitter:
    """Pipeline capture object: V3Events calls .emit(type_, **fields)."""

    def __init__(self, q: asyncio.Queue):
        self.q = q

    def emit(self, type_: str, **fields) -> None:
        self.q.put_nowait((type_, dict(fields)))


def _line(o: dict) -> str:
    return json.dumps(o, default=str) + "\n"


# ---- grounding: every sentence of the answer gets its source ---------
# ponytail: citation choice is nearest-excerpt matching (token Jaccard)
# on the pipeline's OWN verified evidence; upgrade to embeddings when the
# dense checkpoint is already loaded in-process.

def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))


def _best_source(sentence_tokens: set, excerpts: list) -> int:
    best, best_score = 0, -1.0
    for i, ex in enumerate(excerpts):
        st = _tokens(ex)
        if not st:
            continue
        score = len(sentence_tokens & st) / max(1, len(sentence_tokens | st))
        if score > best_score:
            best_score, best = score, i
    return best + 1  # 1-based citation number


def _evidence_sources(evidence: list) -> tuple:
    """Dedupe evidence by document; returns (frontend sources, excerpts)."""
    seen = {}
    for e in evidence:
        doc = e.get("document_id") or e.get("id") or ""
        if doc and doc not in seen:
            seen[doc] = e
    items = list(seen.values())[:12]
    sources = [_src(e) for e in items]
    excerpts = [((e.get("excerpt") or e.get("snippet") or "")) for e in items]
    return sources, excerpts


def _cite_lines(lines, excerpts, source_docs, section_citations) -> list:
    """Append [n] to every prose sentence; keep markdown structure intact."""
    out = []
    for line in lines:
        s = line.strip()
        # keep structural markdown (headings, tables) bare; cite everything else
        if not s or s.startswith("#") or "|" in s:
            out.append(line)
            continue
        cites = section_citations or []
        # bullets/blockquotes keep their marker, the sentence inside gets cited
        marker = ""
        m = re.match(r"^(\s*[-*>]\s+|\d+\.\s+)", line)
        if m:
            marker = m.group(1)
            s = line[len(marker):].strip()
        if not s or s.startswith("#") or "|" in s:
            out.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", s)
        cited = []
        for part in parts:
            toks = _tokens(part)
            if not toks:
                cited.append(part)
                continue
            if not excerpts:
                cited.append(part)  # nothing verified -> no dangling [n]
                continue
            if cites:
                # pipeline LLM citation (repair-checked) bound to a verified
                # document; resolve its number, else nearest-excerpt match
                n = None
                for c in cites:
                    doc = c.get("document_id") or ""
                    if doc in source_docs:
                        n = source_docs.index(doc) + 1
                        if len(cites) == 1:
                            break
                n = n or _best_source(toks, excerpts)
            else:
                n = _best_source(toks, excerpts)
            cited.append(f"{part} [{n}]")
        out.append(marker + " ".join(cited))
    return out


def _compose_cited(answer: dict, sources: list, excerpts: list) -> str:
    """Answer markdown with a [n] citation on every sentence."""
    source_docs = [s.get("id") or "" for s in sources]
    parts = []
    if answer.get("summary"):
        parts.append("## Summary")
        parts += _cite_lines(answer["summary"].split("\n"), excerpts, source_docs, None)
    for sec in answer.get("sections", []):
        parts.append(f"## {sec.get('heading', '')}")
        parts += _cite_lines((sec.get("body") or "").split("\n"), excerpts,
                             source_docs, sec.get("citations") or [])
    if answer.get("limitations"):
        parts.append("## Limitations")
        parts += _cite_lines([f"- {l}" for l in answer["limitations"]], excerpts,
                             source_docs, None)
    return "\n".join(parts)


# ---- PMC article proxy -------------------------------------------------
# The sidebar article view: fetch the article server-side (no CORS), parse it
# into sections + paragraphs, and locate the passage that supports the cited
# excerpt (token overlap + longest contiguous match) so the UI can highlight
# exactly the portion "that says it".
# ------------------------------------------------------------------------

_ARTICLE_CACHE: dict[str, dict] = {}
_SECTION_LABEL = {
    "ABSTRACT": "Abstract",
    "INTRO": "Introduction",
    "METHODS": "Materials & Methods",
    "RESULTS": "Results",
    "DISCUSS": "Discussion",
    "CONCL": "Conclusion",
    "CONCLUSION": "Conclusion",
}
# BioC section types that carry no citable body text
_SKIP_SECTION = {"TITLE", "FIG", "TABLE", "REF", "ABBR", "ACK",
                 "CONFLICT", "SUPPL", "SUPPLEMENT", "AUTHINFO",
                 "AUTH_CONT", "COMP_INT"}
_SKIP_SECTION_RE = re.compile(r"references|bibliography|footnotes|acknowledg", re.I)


def _norm(s) -> str:
    return " ".join((s or "").split())


def _tok(s: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))


async def _fetch_article(pmcid: str) -> dict | None:
    """Fetch + parse one PMC article -> {title, url, sections, flat}.

    Primary: the OA BioC API (clean text, no bot-check dance). Fallback: the
    article HTML page via ``requests`` (httpx is TLS-fingerprinted by NCBI;
    requests is not) for non-OA articles.
    """
    article = await _fetch_bioc(pmcid)
    return article if article else await _fetch_html(pmcid)


async def _fetch_bioc(pmcid: str) -> dict | None:
    """Clean-text extraction via the PMC Open-Access BioC XML API."""
    from lxml import etree

    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    api = ("https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
           f"pmcoa.cgi/BioC_xml/{pmcid}/unicode")
    import requests
    resp = await asyncio.to_thread(requests.get, api, timeout=25.0)
    if resp.status_code != 200 or not resp.text.strip():
        return None
    try:
        root = etree.fromstring(resp.text.encode("utf-8"))
    except Exception:
        return None

    title = pmcid
    sections: list[dict] = []
    flat: list[str] = []
    current: dict | None = None
    for passage in root.xpath("//passage"):
        infons = {i.get("key"): (i.text or "") for i in passage.xpath("infon")}
        stype = (infons.get("section_type") or "").strip().upper()
        sub = (infons.get("section_title") or "").strip()
        text = _norm(passage.findtext("text"))
        if stype == "TITLE" and text and title == pmcid:
            title = text
            continue
        if stype in _SKIP_SECTION or not text or len(text) < 40:
            continue
        heading = sub or _SECTION_LABEL.get(stype, stype.capitalize())
        if current is None or heading != current["heading"]:
            if current and current["paragraphs"]:
                sections.append(current)
            current = {"heading": heading, "paragraphs": []}
        current["paragraphs"].append(text)
        flat.append(text)
    if current and current["paragraphs"]:
        sections.append(current)
    if not flat:
        return None
    return {"pmcid": pmcid, "title": title, "url": url,
            "sections": sections, "flat": flat}


async def _fetch_html(pmcid: str) -> dict | None:
    """HTML-page fallback for non-OA articles (requests bypasses NCBI's
    TLS fingerprint check that blocks httpx)."""
    from lxml import html as lxml_html

    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    import requests
    resp = await asyncio.to_thread(requests.get, url, timeout=25.0, headers={
        "user-agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"),
        "accept": "text/html,application/xhtml+xml",
    })
    if resp.status_code != 200 or not resp.text:
        return None
    tree = lxml_html.fromstring(resp.text)
    bodies = tree.xpath("//div[contains(@class,'body') and contains(@class,'main-article-body')]")
    root = bodies[0] if bodies else (tree.body if tree.body is not None else tree)

    titles = tree.xpath("//h1")
    title = _norm(titles[0].text_content()) if titles else pmcid

    sections: list[dict] = []
    flat: list[str] = []
    current: dict | None = None
    for el in root.iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in ("h2", "h3", "h4") and "pmc_sec_title" in (el.get("class") or ""):
            heading = _norm(el.text_content())
            if _SKIP_SECTION_RE.search(heading):
                continue
            if current and current["paragraphs"]:
                sections.append(current)
            current = {"heading": heading, "paragraphs": []}
            continue
        if tag != "p":
            continue
        anc = set(a.tag for a in el.iterancestors())
        if anc & {"table", "figure", "figcaption", "aside", "ol", "ul", "blockquote", "foot"}:
            continue
        text = _norm(el.text_content())
        if len(text) < 40:
            continue
        if current is None:
            current = {"heading": "", "paragraphs": []}
        current["paragraphs"].append(text)
        flat.append(text)
    if current and current["paragraphs"]:
        sections.append(current)
    if not flat:
        return None
    return {"pmcid": pmcid, "title": title, "url": url,
            "sections": sections, "flat": flat}


def _match_anchor(article: dict, anchor: str) -> dict | None:
    """Best paragraph + contiguous span for the cited excerpt (or None)."""
    a = _norm(anchor)[:240]
    if not a or not article["flat"]:
        return None
    a_tok = _tok(a)
    scored = []
    for i, p in enumerate(article["flat"]):
        p_tok = _tok(p)
        if p_tok:
            scored.append((len(a_tok & p_tok) / max(1, len(a_tok | p_tok)), i, p))
    if not scored:
        return None
    score, idx, text = max(scored, key=lambda x: x[0])
    if score <= 0:
        return None
    # longest contiguous fragment of the anchor inside the paragraph
    for cut in range(len(a), 79, -1):
        m = text.find(a[:cut])
        if m >= 0:
            return {"paragraph": idx, "start": m, "end": m + cut, "text": a[:cut]}
    m = text.find(a)
    if m >= 0:
        return {"paragraph": idx, "start": m, "end": m + len(a), "text": a}
    # no exact stretch: highlight the best whole paragraph
    return {"paragraph": idx, "start": 0, "end": len(text), "text": text}


@app.get("/v1/articles/{pmcid}")
async def article_view(pmcid: str, anchor: str = ""):
    """Article text for the sidebar with the cited passage located."""
    pmcid = pmcid.strip().upper()
    if not re.fullmatch(r"PMC\d+", pmcid):
        raise HTTPException(status_code=404, detail=f"invalid PMCID {pmcid!r}")
    if pmcid not in _ARTICLE_CACHE:
        article = await _fetch_article(pmcid)
        if article is None:
            raise HTTPException(status_code=502,
                                detail=f"could not fetch {pmcid} from PMC")
        _ARTICLE_CACHE[pmcid] = article
        if len(_ARTICLE_CACHE) > 96:  # ponytail: simple cap, no eviction logic
            _ARTICLE_CACHE.pop(next(iter(_ARTICLE_CACHE)))
    article = _ARTICLE_CACHE[pmcid]
    match = _match_anchor(article, anchor) if anchor else None
    return {
        "pmcid": article["pmcid"],
        "title": article["title"],
        "url": article["url"],
        "total_paragraphs": len(article["flat"]),
        "anchor": match,
        "sections": article["sections"],
    }


def _src(e: dict) -> dict:
    doc = e.get("document_id") or e.get("id") or ""
    pmcid = e.get("pmcid")
    return {
        "id": doc,
        "pmcid": pmcid or (doc if str(doc).startswith("PMC") else None),
        "pmid": e.get("pmid"),
        "title": e.get("title") or e.get("document_title") or (e.get("section") or "Untitled"),
        "authors": e.get("authors") or [],
        "journal": e.get("journal") or "",
        "year": e.get("year") or 0,
        "score": (float(e.get("confidence") or 0) or None),
        "snippet": " ".join((e.get("excerpt") or e.get("snippet") or "").split())[:400],
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None,
    }


def _chunks(text: str, size: int = 28):
    """Word-aligned chunks, each word followed by its space, so the streamed
    text reassembles to the original string byte-for-byte."""
    out = []
    buf = ""
    for w in text.split(" "):
        if not w:
            continue
        if buf and len(buf.rstrip()) + 1 + len(w) > size:
            yield buf
            buf = ""
        buf += w + " "
    if buf:
        yield buf


def _engine_selected(body: ChatBody) -> str:
    """Which pipeline to run: body.engine, else XDEEP_ENGINE env."""
    if (body.engine or "").strip().lower() in ("xdeep", "v3"):
        return (body.engine or "").strip().lower()
    switch = os.environ.get("XDEEP_ENGINE", "").strip().lower()
    if switch in ("1", "true", "yes", "on", "xdeep"):
        return "xdeep"
    return "v3"


@app.post("/v1/chat/stream")
async def chat_stream(body: ChatBody, request: Request):
    if _engine_selected(body) == "xdeep":
        # x_deepagents research graph streamed under the same UI contract
        from src.x_deepagents.bridge import stream_xdeep

        trace = get_trace()
        trace.stage(f"API REQUEST (HTTP POST /v1/chat/stream, engine=xdeep) "
                    f"conv={body.conversation_id or '-'}")
        trace.bullet(f"question: {body.question[:300]}")
        return StreamingResponse(
            stream_xdeep(body.question), media_type="application/x-ndjson")

    from src.config import AppConfig
    from src.agentic.pipeline import AgenticV3Pipeline
    from src.llm.client import set_llm_event_sink

    async def gen():
        started = asyncio.get_event_loop().time()
        q: asyncio.Queue = asyncio.Queue()
        emitter = _Emitter(q)
        # LLM-call observations (attempts, rate-limit failures + retries)
        # flow into the same stream; contextvars scope them to this request.
        set_llm_event_sink(emitter.emit)

        trace = get_trace()
        trace.stage(f"API REQUEST (HTTP POST /v1/chat/stream) conv={body.conversation_id or '-'}")
        trace.bullet(f"question: {body.question[:300]}")

        memory_api = get_memory_api()
        task = asyncio.create_task(
            AgenticV3Pipeline(config=AppConfig(), events=emitter,
                              memory=memory_api).answer(
                body.question, conversation_id=body.conversation_id)
        )
        result: dict | None = None
        while True:
            try:
                t, f = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if task.done():
                    break
                continue
            stage = STAGE.get(t)
            if stage:
                yield _line({"type": "status", "stage": stage})
            if t == "retrieved" and f.get("papers"):
                papers = [_src(p) for p in f["papers"][:10]]
                yield _line({"type": "sources", "sources": papers})
            if t == "memory_prepare":
                yield _line({
                    "type": "memory", "kind": "prepare",
                    "session_id": f.get("session_id", ""),
                    "session_title": f.get("session_title", ""),
                    "prior_claims": f.get("prior_claims", 0),
                    "prior_contradictions": f.get("prior_contradictions", 0),
                    "prior_gaps": f.get("prior_gaps", 0),
                })
            elif t == "memory_commit":
                yield _line({
                    "type": "memory", "kind": "commit",
                    "session_id": f.get("session_id", ""),
                    "stats": f.get("stats", {}),
                })
            if t not in _RESERVED:
                # verbose mirror of EVERY pipeline / LLM event
                yield _line({"type": "pipeline", "event": t, "fields": f})

        if result is None and not task.cancelled():
            try:
                result = task.result()
            except Exception as exc:
                yield _line({"type": "error", "message": str(exc)[:500]})
        if result:
            sources, excerpts = _evidence_sources(result.get("evidence") or [])
            if sources:
                yield _line({"type": "sources", "sources": sources})
            answer = result.get("answer") or {}
            if answer:
                body_text = _compose_cited(answer, sources, excerpts)
                for part in _chunks(body_text):
                    if await request.is_disconnected():
                        break
                    yield _line({"type": "token", "content": part})
        yield _line({"type": "done", "timingMs": int((asyncio.get_event_loop().time() - started) * 1000)})
        trace.bullet(
            f"API RESPONSE complete in {int((asyncio.get_event_loop().time() - started) * 1000)}ms "
            f"(conv={body.conversation_id or '-'}) — streamed to client")

    return StreamingResponse(gen(), media_type="application/x-ndjson")