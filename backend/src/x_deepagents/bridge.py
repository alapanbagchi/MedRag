"""Bridge: run the x_deepagents research graph and stream it as the
MedPat UI wire contract. The UI speaks newline-delimited JSON:

  {"type":"status","stage":<ResearchStatus>,"message":...,"count":...}
  {"type":"sources","sources":[...]}
  {"type":"pipeline","event":...,"fields":...}     (verbose trace)
  {"type":"token","content":...}
  {"type":"done","timingMs":...}

Milestones from the graph progress() feed are mapped to UI stages; the
final XDeepRunState is converted to sources + a cited answer (nearest
excerpt matching, same as the v3 API) + done. Memory events are emitted
as no-ops here (the memory layer is a v3 feature) unless later wired.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Optional
from urllib.parse import quote

from src.x_deepagents.graph import run_research, set_event_sink, set_progress_sink


# progress-marker -> UI stage (must stay within ResearchStatus values)
STAGE_MARKERS: list[tuple[tuple[str, ...], str]] = [
    (("[decompose]", "[join]"), "decomposing"),
    (("[research:", "SEARCH PLANNER", "REPLANNER", "umls:"), "understanding"),
    (("retrieve:", "retrieved"), "retrieving"),
    (("[verify]", "verify:"), "reranking"),
    (("[conflict]", "[resolution]", "contradiction"), "verifying"),
    (("[gap-resolution]", "[gap-resolution]"), "verifying"),
    (("[synthesis]", "[done]"), "synthesizing"),
]


def stage_for_progress(message: str) -> str:
    """Map one graph progress line to a UI stage (or None)."""
    for markers, stage in STAGE_MARKERS:
        if any(m in message for m in markers):
            return stage
    return ""


def _line(o: dict) -> str:
    return json.dumps(o, default=str) + "\n"


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))


def _best_source(sentence_tokens: set, excerpts: list) -> int:
    """Nearest-excerpt citation number (1-based) - mirrors api.py."""
    best, best_score = 0, -1.0
    for i, ex in enumerate(excerpts):
        st = _tokens(ex)
        if not st:
            continue
        score = len(sentence_tokens & st) / max(1, len(sentence_tokens | st))
        if score > best_score:
            best_score, best = score, i
    return best + 1


def _cited_text(answer: str, excerpts: list, source_docs: list,
                doc_to_index: Optional[dict] = None) -> str:
    """Resolve 〔cite:PMCxxx〕 markers to [n] against VERIFIED sources.

    Every marker whose document_id appears in the verified source list becomes
    [n]; markers for unverified/hallucinated ids are DROPPED (repair_citations
    semantics). When the model emitted no markers at all (e.g. plain-text
    fallback), falls back to nearest-excerpt matching so the UI never shows
    uncited prose.

    IMPORTANT: headers and gap/limitation bullet lines never carry markers
    AND must not receive legacy-overlap citations. When ANY marker exists in
    the answer, non-marker lines (## headers, - bullets, plain prose) are
    passed through untouched - the citation system is the markers, not
    sentence overlap. Only a fully-marker-less answer uses the overlap path.
    """
    raw = str(answer or "")
    has_markers = _CITE_OPEN in raw
    out: list[str] = []
    for line in raw.split("\n"):
        s = line
        if _CITE_OPEN in s:
            # marker-aware path (exact, deterministic)
            def _sub(m):
                doc = m.group(1)
                n = (doc_to_index or {}).get(doc)
                return f" [{n}]" if n else ""   # drop unverified markers
            resolved = _CITE_RE.sub(_sub, s)
            out.append(resolved)
            continue
        if has_markers:
            # markers govern citations: nothing else gets [n]
            out.append(s)
            continue
        # legacy overlap path (no markers anywhere in the answer)
        line = s.strip()
        if not line or line.startswith("#") or "|" in line or not excerpts:
            out.append(s)
            continue
        marker = ""
        m = re.match(r"^(\s*[-*>]\s+|\d+\.\s+)", line)
        if m:
            marker = m.group(1)
            line = line[len(marker):].strip()
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", line)
        cited = []
        for part in parts:
            toks = _tokens(part)
            if not toks or not excerpts:
                cited.append(part)
                continue
            n = _best_source(toks, excerpts)
            cited.append(f"{part} [{n}]")
        out.append(marker + " ".join(cited))
    return "\n".join(out)


def _text_fragment_url(base_url: str, passage: str) -> tuple[str, str]:
    """Build a browser-native Text-Fragment URL that opens the page AND
    scrolls/highlights the cited passage:

        https://site/page#:~:text=<urlencoded passage>

    Returns (url_with_fragment, fragment_text). The passage is truncated to a
    distinctive ~120-char span; if nothing usable, the plain URL is returned.
    """
    frag = " ".join((passage or "").split()) if passage else ""
    if len(frag) < 25:
        # too short / nothing to highlight
        return base_url, ""
    frag = frag[:140].rstrip(" .,;:")
    return base_url + "#:~:text=" + quote(frag, safe=""), frag


def _src(item: Any) -> dict:
    """One EvidenceItem -> frontend Source dict.

    WEB sources (source_url set): the URL becomes the actual website with a
    Text-Fragment hash that scrolls to and highlights the cited passage;
    'highlight' carries the quoted span and 'isWeb' tells the UI to label the
    button as the site (not PMC). PMC sources keep the article URL.
    """
    pmcid = item.document_id or ""
    source_url = str(getattr(item, "source_url", "") or "")
    if source_url:
        url, frag = _text_fragment_url(source_url, item.text)
        return {
            "id": pmcid or url,
            "pmcid": "",
            "title": (getattr(item, "journal", "") or "Web source") or "Web source",
            "authors": [],
            "journal": "Web (" + (getattr(item, "trust", "") or "web") + ")",
            "year": 0,
            "score": round(float(getattr(item, "confidence_float", 0.0) or 0.0), 4)
            if hasattr(item, "confidence_float") else None,
            "snippet": " ".join((item.text or "").split())[:400],
            "url": url,
            "highlight": frag or (str(getattr(item, "text", "")) or "")[:140],
            "isWeb": True,
        }
    return {
        "id": pmcid,
        "pmcid": pmcid,
        "title": pmcid or "PMC article",
        "authors": [],
        "journal": "PMC (xdeep)",
        "year": 0,
        "score": round(float(getattr(item, "confidence_float", 0.0) or 0.0), 4)
        if hasattr(item, "confidence_float") else None,
        "snippet": " ".join((item.text or "").split())[:400],
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None,
        "isWeb": False,
    }


_CITE_OPEN = "\u3014cite:"
_CITE_RE = re.compile("\u3014cite:([^\u3015]+)\u3015")


def _doc_to_index(sources: list) -> dict:
    """document_id -> 1-based source number (from the deduped verified list)."""
    out: dict = {}
    for i, src in enumerate(sources, 1):
        doc = str(src.get("id") or src.get("pmcid") or "")
        if doc:
            out[doc] = i
    return out


def _verified_sources(run) -> tuple:
    """(sources, excerpts) from the run state verified items."""
    seen: dict = {}
    for it in run.verified_items():
        doc = it.document_id or it.id or ""
        if doc and doc not in seen:
            seen[doc] = it
    items = list(seen.values())[:12]
    sources = [_src(i) for i in items]
    excerpts = [(i.text or "") for i in items]
    return sources, excerpts


async def stream_xdeep(question: str) -> AsyncIterator[str]:
    """Run the xdeep research graph, yielding UI wire lines as it goes."""
    started = time.monotonic()
    q: asyncio.Queue = asyncio.Queue()

    def on_progress(message: str) -> None:
        """Sink: forward every graph progress line to the stream queue."""
        q.put_nowait(("progress", None, message))
        stage = stage_for_progress(message)
        if stage:
            q.put_nowait(("status", stage, message))

    def on_event(event: str, fields: dict) -> None:
        """Structured trace sink: rich events (web searches, site fetches,
        verdicts, gaps, contradictions...) flow to the UI think-log."""
        q.put_nowait(("event", event, fields))

    set_progress_sink(on_progress)
    set_event_sink(on_event)
    try:
        task = asyncio.create_task(run_research(question))
        run = None
        while True:
            try:
                kind, stage, message = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if task.done():
                    break
                continue
            if kind == "progress":
                yield _line({"type": "pipeline", "event": "progress", "fields": {"msg": message}})
            elif kind == "event":
                yield _line({"type": "pipeline", "event": stage, "fields": message})
            else:
                # status event: stage + the progress line as the message detail
                yield _line({"type": "status", "stage": stage, "message": message})
        run = task.result()
    except Exception as exc:
        yield _line({"type": "error", "message": str(exc)[:500]})
        run = None
    finally:
        set_progress_sink(None)
        set_event_sink(None)

    if run is not None:
        sources, excerpts = _verified_sources(run)
        if sources:
            yield _line({"type": "sources", "sources": sources})
        if run.answer and excerpts:
            body = _cited_text(run.answer, excerpts,
                               [s.get("id") for s in sources],
                               doc_to_index=_doc_to_index(sources))
            for part in _chunks(body):
                yield _line({"type": "token", "content": part})
        elif run.answer:
            for part in _chunks(run.answer):
                yield _line({"type": "token", "content": part})

    yield _line({"type": "done", "timingMs": int((time.monotonic() - started) * 1000)})


def _chunks(text: str, size: int = 28):
    """Word-aligned chunks preserving the source text byte-for-byte."""
    buf = ""
    for w in str(text).split(" "):
        if not w:
            continue
        if buf and len(buf.rstrip()) + 1 + len(w) > size:
            yield buf
            buf = ""
        buf += w + " "
    if buf:
        yield buf


__all__ = ["stream_xdeep", "stage_for_progress"]