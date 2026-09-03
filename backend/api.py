"""MedRag API: streams the singular deepagents research flow to the web UI.

Every request and every internal pipeline action is mirrored to logs.txt
(LOG_FILE to override) via the shared streaming trace, so a web run is
fully inspectable after the fact.
"""
from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
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


app = FastAPI(title="MedRag API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChatBody(BaseModel):
    question: str
    conversation_id: str = ""
    history: list = []
    # accepted for wire compatibility; ignored (singular deepagents flow).
    engine: str = ""


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


@app.post("/v1/chat/stream")
async def chat_stream(body: ChatBody):
    # Singular deepagents flow: every request runs the agents research
    # graph streamed under the UI wire contract (status / sources /
    # pipeline / memory / token / done).
    from src.agents.bridge import stream_xdeep

    trace = get_trace()
    trace.stage(f"API REQUEST (HTTP POST /v1/chat/stream) "
                f"conv={body.conversation_id or '-'}")
    trace.bullet(f"question: {body.question[:300]}")
    return StreamingResponse(
        stream_xdeep(body.question,
                     conversation_id=body.conversation_id),
        media_type="application/x-ndjson")