"""Chunker v2 — structure-first chunking of Markdown articles (no LLM).

Consumes the ``.md`` files produced by ``src.processing.jats_to_md`` (YAML front
matter + headings + paragraphs + lists + tables + figures + equations +
references) and emits:

  1. fine-grained retrieval chunks — the SAME ``Chunk`` schema / chunk types
     as the XML pipeline (``src/chunker.py``), so the embedding and retrieval
     code is untouched.  Prose is PARAGRAPH-FIRST: one paragraph is one
     retrieval chunk.  Adjacent paragraphs merge only when one is below
     ``min_paragraph_tokens`` and the pair fits ``max_tokens``; a paragraph
     above the budget is sentence-split.  ``--prose-strategy window``
     restores the legacy pack-to-budget behavior for eval A/B.
  2. a parallel **units** artifact (P1 parent/child): one unit per section /
     table with the full text plus the ids of its granular child chunks
     (small-to-big context expansion lives HERE, not in duplicated text).
  3. deterministic **entity tags** on chunks from an optional local lexicon
     (P2, no LLM / no network);
  4. document metadata from the YAML front matter (P2), exact-duplicate
     suppression (per-doc in-process; ``--global-dedup`` as a deterministic
     sorted post-pass over the written parquets), best-effort **bracketed
     citation extraction** (``citation_refs`` -> ``ref_{n}`` ids, resolved
     against the parsed reference list), and a coverage/quality report
     (P2 eval-gate seed) that records the full chunker config.

Tables: each table becomes a ``table_summary`` chunk (label + caption +
columns), one self-contained ``table_row`` chunk per data row (table identity +
row label + ``column: value`` pairs; missing/empty cells render as ``—``), and optional
``table_footnotes``.  Rows point at the summary chunk via ``parent_id``,
carry the table unit via ``metadata.table_unit_id`` (direct route to
full-table text), and inherit the most recent spanning-subheader row via
``group_path`` (colspan subheaders flattened by the MD generator are group
context, not retrieval chunks).  A first-class TABLE unit holds the full
table text and all child chunk ids.

THIS FILE IS CHUNKING ONLY — encoding is a separate downstream step that
consumes the parquets written here.  The chunker is encoder-AGNOSTIC: no
transformers import, no model download, no network.  What it does own:

  * ``embedding_text`` = ``prefix | text`` — the object label (Table N /
    Figure N) followed by the complete chunk evidence. No document title and
    no section breadcrumb are embedded (matching src/chunker.py's original
    encoding minus breadcrumb): title/section context is resolved at
    retrieval time via metadata. Journal and publication date are also not
    embedded — they are filter/rank metadata, not semantic content.
  * Paragraph chunks never duplicate neighboring evidence. ``overlap`` is kept
    only for backwards compatibility and defaults to 0. Neighbor relationships
    are stored in metadata and the units artifact, so context expansion happens
    after retrieval rather than contaminating vectors.
  * Token budgets are enforced with a LOCAL tokenizer (cl100k when tiktoken is
    installed, chars/4 otherwise). The downstream MedCPT encoder should still
    use ``truncation=True`` as a final guard.
  * ``embedding_max_tokens`` defaults to 448 as a conservative budget inside
    MedCPT's documented 512-token encoder window.
  * ``embedding_token_count`` is populated at chunk time as an overflow audit
    before the embedding step runs.

Citations: bracketed numeric citations (``[1]``, ``[5-7]``, ``[2,4,9]``)
in paragraph/list text are harvested and resolved to ``ref_{n-1}``
reference ids once the whole document — including its reference list —
has been chunked; numbers outside the parsed reference range are dropped.
This assumes citation-order reference numbering (the norm for bracketed
style).  Superscript citations glued to words by ``jats_to_md.py``
(``"youth1,2,3"``) are NOT extractable here — that remains an upstream
conversion fix.

The XML pipeline (``src/parser.py`` + ``src/chunker.py``) is left completely
untouched.  Skipped on purpose (need an LLM or new embedding code):
LLM-generated table summaries / question generation, and late-chunking
multi-vector indexing.

CLI (defaults ARE the recommended config for a downstream MedCPT encoder —
no flags needed):
    python -m src.chunking.md_chunker --input data/md --chunks-out chunks_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from markdown_it import MarkdownIt

from src.chunking.classification import classify_section_title
from src.lib.models import CHUNK_VERSION, Chunk

_MD = MarkdownIt("commonmark").enable("table")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")

# Compatibility constant retained for older imports. Production paragraph
# embeddings never include neighboring evidence.
_CONTEXT_PREFIX = "Previous context: "

# Bracketed numeric citations: [1], [5-7] (hyphen/en-dash/em-dash), [2,4,9],
# and combinations.  1-3 digits per number so 4-digit years in brackets are
# never matched.
_CITE_RE = re.compile(
    r"\[(\d{1,3}(?:\s*[–—-]\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*[–—-]\s*\d{1,3})?)*)\]"
)

# ---------------------------------------------------------------------------
# Token estimation (LOCAL ONLY: cl100k when tiktoken is available, chars/4
# fallback). No transformers, no network, no encoder dependency — token
# counting here is a planning tool for shaping chunks, never encoding.
#
# Budgets are SIZED for the planned downstream encoder (MedCPT
# ArticleEncoder, 512-token window). cl100k undercounts BERT-style
# WordPiece by ~0-14% on biomedical prose, so
# DEFAULT_EMBEDDING_MAX_TOKENS = (512 - 2 special tokens) * ~0.88 = 448:
# the ~12% margin absorbs the proxy's undercount.
# ---------------------------------------------------------------------------

TARGET_ENCODER_WINDOW = 512          # planned downstream encoder (MedCPT)
DEFAULT_EMBEDDING_MAX_TOKENS = 448   # conservative pre-encoding budget

_TOKENIZER: Any = None
_TOKENIZER_KIND = "chars4"

try:  # pragma: no cover - env dependent
    from tiktoken import get_encoding

    _TOKENIZER = get_encoding("cl100k_base")
    _TOKENIZER_KIND = "cl100k"
except Exception:  # pragma: no cover
    _TOKENIZER = None
    _TOKENIZER_KIND = "chars4"


def _estimate_tokens(text: str) -> int:
    if _TOKENIZER is not None:
        try:
            return max(1, len(_TOKENIZER.encode(text or "")))
        except Exception:  # noqa: BLE001
            pass
    return max(1, (len(text or "") + 3) // 4)


def _truncate_tokens(text: str, limit: int) -> str:
    """Truncate ``text`` to at most ``limit`` TOKENS (cl100k when available,
    chars/4 fallback). Never raises.

    Only ever applied to the EMBEDDED COPY in pathological overflow cases —
    stored ``text`` is never routed through here.
    """
    if limit <= 0:
        return ""
    if _estimate_tokens(text) <= limit:
        return text
    if _TOKENIZER is not None:
        try:
            return _TOKENIZER.decode(_TOKENIZER.encode(text)[:limit]).strip()
        except Exception:  # noqa: BLE001
            pass
    return text[: limit * 4].strip()


# ---------------------------------------------------------------------------
# MD parsing: front matter, fences, blocks, sections
# ---------------------------------------------------------------------------


def _parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split ``--- yaml ---`` front matter from the body. Never raises."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    idx = 1
    while idx < len(lines) and lines[idx].strip() != "---":
        idx += 1
    if idx >= len(lines):
        return {}, text
    try:
        import yaml

        meta = yaml.safe_load("\n".join(lines[1:idx])) or {}
    except Exception:  # noqa: BLE001 - malformed front matter must not kill the run
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, "\n".join(lines[idx + 1:])


def _extract_fences(text: str) -> Tuple[str, List[Tuple[str, str, int]]]:
    """Pull ``$$...$$`` equation and ``` code fences out of the body.

    markdown-it has no built-in math rule, so equations are extracted first
    and re-injected as structured blocks. Fence lines are BLANKED (line
    numbers preserved) so token ``.map`` positions still line up.

    Unclosed fences (``$$`` or ``` ```) are treated as ordinary text — a
    stray delimiter must never swallow the rest of the document.
    """
    out: List[str] = []
    blocks: List[Tuple[str, str, int]] = []
    i = 0
    lines = text.splitlines()
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("$$"):
            start = i
            buf = []
            if s == "$$":
                i += 1
                while i < len(lines) and lines[i].strip() != "$$":
                    buf.append(lines[i])
                    i += 1
                if i >= len(lines):
                    # UNCLOSED fence: never swallow the rest of the document;
                    # treat the line as ordinary text instead
                    out.append(lines[start])
                    i = start + 1
                    continue
                i += 1  # closing $$
            else:
                body = s[2:]
                if body.endswith("$$"):
                    body = body[:-2]
                buf.append(body.strip())
                i += 1  # single-line equation: advance past it
            blocks.append(("equation", "\n".join(buf).strip(), start))
            for j in range(start, i):
                out.append("")
            continue
        if s.startswith("```"):
            start = i
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            if i >= len(lines):
                # UNCLOSED code fence: same guard as $$ — don't swallow the doc
                out.append(lines[start])
                i = start + 1
                continue
            i += 1  # closing fence
            blocks.append(("code", "\n".join(buf).strip(), start))
            for j in range(start, i):
                out.append("")
            continue
        out.append(line)
        i += 1
    return "\n".join(out), blocks


def _inline_to_plain(text: str) -> str:
    """Strip light inline Markdown (bold/italic/code/links) to plain text."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)       # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)    # links -> label
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)          # **bold**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)              # *italic*
    text = re.sub(r"`([^`]+)`", r"\1", text)                # `code`
    text = re.sub(r"~~([^~]+)~~", r"\1", text)              # ~~strike~~
    return " ".join(text.split())


def _citation_numbers(text: str) -> List[int]:
    """Extract 1-based reference numbers from bracketed citations.

    Handles single refs (``[7]``), ranges (``[1-11]``, en/em-dash or
    hyphen), and lists (``[2,4,9]``).  Returns sorted unique ints; the
    mapping to ``ref_{n-1}`` ids happens in ``chunk_md`` once the parsed
    reference count is known (out-of-range numbers are dropped there).
    """
    nums: List[int] = []
    for group in _CITE_RE.findall(text or ""):
        for part in group.split(","):
            part = part.strip()
            if not part:
                continue
            pieces = re.split(r"\s*[–—-]\s*", part)
            if len(pieces) == 2 and pieces[0].isdigit() and pieces[1].isdigit():
                a, b = int(pieces[0]), int(pieces[1])
                if a <= b and b - a <= 200:  # guard against absurd ranges
                    nums.extend(range(a, b + 1))
            elif part.isdigit():
                nums.append(int(part))
    return sorted(set(nums))


@dataclass
class RawBlock:
    """A parsed Markdown block with its raw layout facts preserved."""
    kind: str                                  # paragraph|list|table|figure|equation|code
    text: str = ""                             # plain text (paragraph/equation/code)
    label: str = ""                            # figure/table label
    caption: str = ""                          # figure/table caption
    image_ref: str = ""                        # figure url
    italic: bool = False                       # raw line was *...* (table caption/footnote)
    bold: bool = False                         # raw line was **...** (figure caption)
    items: List[str] = field(default_factory=list)          # list items (plain)
    raw_items: List[str] = field(default_factory=list)      # list items (raw md)
    header: List[str] = field(default_factory=list)         # table header
    rows: List[List[str]] = field(default_factory=list)     # table body rows
    footnotes: List[Tuple[str, str]] = field(default_factory=list)  # (marker, text)


def _consume_raw_block(tokens: List[Any], i: int) -> Tuple[Optional[RawBlock], int]:
    """Build one RawBlock from the token stream at index ``i``.

    Returns ``(block, next_index)``. Handles paragraphs (and bare-image
    paragraphs -> figures), lists, tables, and code fences.
    """
    tok = tokens[i]
    t = tok.type
    n = len(tokens)

    def inline_at(j: int) -> str:
        return _inline_to_plain(tokens[j].content) if j < n else ""

    if t == "inline":
        imgs = _IMAGE_RE.findall(tok.content)
        if imgs:
            # robust to figures whose caption shares the same paragraph
            # (image + caption on adjacent lines without a blank line);
            # the caption is everything AFTER the last image (spans are
            # position-based, so captions containing ']' are safe)
            caption = ""
            last_end = None
            for m in _IMAGE_RE.finditer(tok.content):
                last_end = m.end()
            if last_end is not None:
                caption = _inline_to_plain(tok.content[last_end:])
            return RawBlock(kind="figure",
                            label=imgs[0][0].strip() or "Figure",
                            caption=caption,
                            image_ref=imgs[0][1].strip()), i + 1
        text = inline_at(i)
        if not text:
            return None, i + 1
        children = getattr(tok, "children", None) or []
        kinds = {getattr(ch, "type", "") for ch in children}
        return RawBlock(kind="paragraph", text=text,
                        italic="em_open" in kinds,
                        bold="strong_open" in kinds), i + 1

    if t in ("bullet_list_open", "ordered_list_open"):
        items: List[str] = []
        raw_items: List[str] = []
        j = i
        while j < n and tokens[j].type not in ("bullet_list_close", "ordered_list_close"):
            if tokens[j].type == "list_item_open":
                k = j + 1
                parts: List[str] = []
                raw_parts: List[str] = []
                while k < n and tokens[k].type != "list_item_close":
                    if tokens[k].type == "inline":
                        raw_parts.append(tokens[k].content)
                        parts.append(_inline_to_plain(tokens[k].content))
                    k += 1
                item = " ".join(p for p in parts if p)
                if item:
                    items.append(item)
                    raw_items.append(" ".join(raw_parts))
                j = k
            else:
                j += 1
        return RawBlock(kind="list", items=items, raw_items=raw_items), j + 1

    if t == "table_open":
        header: List[str] = []
        rows: List[List[str]] = []
        cur: List[str] = []
        in_header = False
        j = i + 1
        while j < n and tokens[j].type != "table_close":
            tt = tokens[j].type
            if tt == "thead_open":
                in_header = True
            elif tt == "tbody_open":
                in_header = False
            elif tt == "tr_open":
                cur = []
            elif tt == "inline":
                cell = inline_at(j)
                if in_header:
                    header.append(cell)
                else:
                    cur.append(cell)
            elif tt == "tr_close":
                if cur:
                    if in_header:
                        header = cur
                    else:
                        rows.append(cur)
                cur = []
            j += 1
        return RawBlock(kind="table", header=header, rows=rows), j + 1

    if t == "fence":
        return RawBlock(kind="code", text=tok.content.strip()), i + 1

    return None, i + 1


def _attach_table_context(blocks: List[RawBlock]) -> None:
    """Attach caption (italic line before a table) and footnotes (italic lines
    after a table) produced by src.processing.jats_to_md."""
    for idx, b in enumerate(blocks):
        if b.kind != "table":
            continue
        if idx > 0 and blocks[idx - 1].kind == "paragraph" and blocks[idx - 1].italic:
            head = blocks[idx - 1].text
            if ":" in head:
                label, _, caption = head.partition(":")
                b.label = label.strip()
                b.caption = caption.strip()
            else:
                b.label = head
            blocks[idx - 1].text = ""
        fn: List[Tuple[str, str]] = []
        j = idx + 1
        while j < len(blocks) and blocks[j].kind == "paragraph" and blocks[j].italic:
            fn_text = blocks[j].text.strip()
            if fn_text:
                marker = ""
                if "]" in fn_text:
                    marker = fn_text.split("]", 1)[0].lstrip("[")
                    fn_text = fn_text.split("]", 1)[1].strip()
                fn.append((marker, fn_text))
                blocks[j].text = ""
            j += 1
        if fn:
            b.footnotes = fn
        blocks[idx].text = ""


def _attach_figure_caption(blocks: List[RawBlock]) -> None:
    """Attach the bold "*label caption*" line emitted after a bare-image
    paragraph as the figure caption."""
    for idx, b in enumerate(blocks):
        if b.kind != "figure":
            continue
        if idx + 1 < len(blocks) and blocks[idx + 1].kind == "paragraph" and blocks[idx + 1].bold:
            b.caption = blocks[idx + 1].text
            blocks[idx + 1].text = ""


@dataclass
class MDNode:
    """One Markdown heading (section) with its blocks and child sections."""
    level: int
    title: str
    blocks: List[RawBlock] = field(default_factory=list)
    children: List["MDNode"] = field(default_factory=list)


def build_section_tree(text: str, fences: List[Tuple[str, str, int]]) -> List[MDNode]:
    """Turn a Markdown body into a heading tree of ``MDNode``.

    One token walk: heading_open creates nodes on a stack (like the JATS
    parser's nested sections); every other block attaches to the current
    node. Equation/code fences are injected by blanked line position.
    """
    tokens = _MD.parse(text)
    roots: List[MDNode] = []
    stack: List[MDNode] = []
    fences = list(fences)  # [(kind, content, line)]

    def flush_fences_before(line: Optional[int], target: List[RawBlock]) -> None:
        while fences:
            kind, content, fl = fences[0]
            if line is not None and fl >= line:
                break
            target.append(RawBlock(kind="equation" if kind == "equation" else "code",
                                   text=content))
            fences.pop(0)

    i = 0
    n = len(tokens)
    pending: List[RawBlock] = []     # content before the first heading
    while i < n:
        tok = tokens[i]
        t = tok.type
        if t == "heading_open":
            # flush any fences that ended before this heading into the
            # current holder, so equations are never attributed to the
            # section that FOLLOWS them
            line = tok.map[0] if tok.map else None
            flush_fences_before(line, stack[-1].blocks if stack else pending)
            title = ""
            if i + 1 < n and tokens[i + 1].type == "inline":
                title = _inline_to_plain(tokens[i + 1].content)
            level = int(tok.tag[1]) if tok.tag and tok.tag[1:].isdigit() else 2
            while stack and stack[-1].level >= level:
                stack.pop()
            node = MDNode(level=level, title=title)
            if stack:
                stack[-1].children.append(node)
            else:
                if pending:
                    node.blocks.extend(pending)
                    pending.clear()
                roots.append(node)
            stack.append(node)
            i += 2
            continue
        line = tok.map[0] if tok.map else None
        raw, next_i = _consume_raw_block(tokens, i)
        i = next_i
        if raw is None:
            continue
        holder = stack[-1].blocks if stack else pending
        flush_fences_before(line, holder)
        holder.append(raw)
    flush_fences_before(None, stack[-1].blocks if stack else pending)
    if pending:
        if roots:
            roots[0].blocks = pending + roots[0].blocks
        else:
            # a heading-less document still chunks: everything becomes the
            # blocks of a single anonymous section
            roots = [MDNode(level=2, title="", blocks=pending)]
    return roots


# ---------------------------------------------------------------------------
# Document metadata from YAML front matter (P2)
# ---------------------------------------------------------------------------

def _csv_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value)]


def _document_metadata(front: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """Normalize YAML front matter into the same metadata shape v1 uses.

    Journal / publication_date live here (chunk metadata, filterable and
    rankable) but are deliberately NOT part of embedding_text.
    """
    authors = _csv_list(front.get("authors"))
    keywords = _csv_list(front.get("keywords"))
    categories = _csv_list(front.get("categories"))
    return {
        "pmcid": str(front.get("pmcid") or doc_id),
        "title": str(front.get("title") or ""),
        "journal": str(front.get("journal") or ""),
        "doi": str(front.get("doi") or ""),
        "publication_date": str(front.get("published") or ""),
        "keywords": keywords,
        "categories": categories,
        "authors": authors,
        "funding_sources": _csv_list(front.get("funding")),
        "volume": str(front.get("volume") or ""),
        "issue": str(front.get("issue") or ""),
        "pages": str(front.get("pages") or ""),
        "pmid": str(front.get("pmid") or ""),
    }


# ---------------------------------------------------------------------------
# Entity tagging (P2, deterministic lexicon, no LLM)
# ---------------------------------------------------------------------------

class LexiconTagger:
    """Word-boundary, case-insensitive concept tagging from a local lexicon."""

    def __init__(self, lexicon: Optional[Dict[str, List[str]]] = None):
        self._patterns: List[Tuple[str, "re.Pattern"]] = []
        for concept_id, forms in (lexicon or {}).items():
            for form in forms or []:
                form = " ".join(str(form).split())
                if not form:
                    continue
                self._patterns.append((str(concept_id),
                                       re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)",
                                                  re.IGNORECASE)))

    def tag(self, text: str) -> List[str]:
        """Return the concept ids whose surface forms occur in ``text``."""
        hits: Dict[str, int] = {}
        low = text or ""
        for concept_id, pattern in self._patterns:
            if pattern.search(low):
                hits[concept_id] = hits.get(concept_id, 0) + 1
        return sorted(hits.keys())


def _load_lexicon(path: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): [str(v) for v in (val if isinstance(val, list) else [val])]
                for k, val in raw.items()}
    out: Dict[str, List[str]] = {}
    for item in raw:
        if isinstance(item, dict) and item.get("concept_id"):
            terms = item.get("terms") or item.get("synonyms") or []
            out[str(item["concept_id"])] = [str(t) for t in terms]
    return out or None


# ---------------------------------------------------------------------------
# Units artifact (P1 parent/child)
# ---------------------------------------------------------------------------

@dataclass
class UnitRecord:
    """One retrievable unit (section or table) with its child chunk ids."""
    unit_id: str
    document_id: str
    kind: str                      # "section" | "table"
    title: str                     # unit title (heading or table label)
    breadcrumb: List[str]
    text: str                      # full unit text (parent context)
    chunk_ids: List[str]           # granular child chunks that live under it


# ---------------------------------------------------------------------------
# Chunker v2
# ---------------------------------------------------------------------------

class MDChunker:
    """Structure-first, paragraph-first chunker over Markdown articles (no LLM).

    Encoder-AGNOSTIC: budgets are SIZED for a downstream MedCPT encoder
    (512-token window) but nothing here loads or depends on any encoder —
    encoding is a separate step consuming the parquets this class writes.

      * ``max_tokens``=320 keeps the source evidence compact enough that the
        title/structure head normally fits inside the 448-token embedding
        budget without truncation;
      * ``hard_max_tokens``=640 is the pathological-sentence ceiling;
      * ``min_paragraph_tokens``=0 by default, preserving every source
        paragraph as an independent evidence unit; opt-in merging is available
        for corpora with many genuinely tiny fragments;
      * ``overlap``=0 (recommended; retained only for compatibility);
      * ``embedding_max_tokens``=448 is a conservative pre-encoding budget
        inside MedCPT's documented 512-token window; the downstream encoder
        remains the final authority on truncation.

    ``parent_id`` semantics: the immediately-enclosing retrievable parent —
    the section/unit id for prose, list, figure, equation, reference and
    administrative chunks; the table summary CHUNK id for table rows and
    table footnotes.  Table chunks additionally carry
    ``metadata.table_unit_id`` — the direct route from any row to the table
    unit (full-table text) without reverse-searching the units artifact.

    ``group_path`` on table rows holds the most recent spanning-subheader
    label (e.g. "Net results (mmHg) ..."), so a row like "Change 24h-SBP"
    can be read in its group context; the group label is also folded into
    the row's embedding prefix.

    ``citation_refs`` on paragraph/list chunks holds ``ref_{n-1}``
    reference ids harvested from bracketed citations and validated against
    the parsed reference list (see ``_citation_numbers``).
    """

    def __init__(self, max_tokens: int = 320, hard_max_tokens: int = 640,
                 overlap: float = 0.0,
                 min_paragraph_tokens: int = 0,
                 embedding_max_tokens: int = DEFAULT_EMBEDDING_MAX_TOKENS,
                 prose_strategy: str = "paragraph",
                 tagger: Optional[LexiconTagger] = None,
                 dedup_cache: Optional[Dict[str, str]] = None):
        if max_tokens <= 0 or hard_max_tokens < max_tokens:
            raise ValueError("need 0 < max_tokens <= hard_max_tokens")
        if prose_strategy not in ("paragraph", "window"):
            raise ValueError("prose_strategy must be 'paragraph' or 'window'")
        self.max_tokens = int(max_tokens)
        self.hard_max_tokens = int(hard_max_tokens)
        self.overlap = max(0.0, min(0.5, float(overlap)))
        self._overlap_tokens = int(self.max_tokens * self.overlap)
        # paragraph strategy: pieces below this floor merge into a neighbor
        # when the pair fits max_tokens; 0 disables merging
        self.min_paragraph_tokens = max(0, int(min_paragraph_tokens))
        # 0 = no embedding-side trimming; otherwise trim embedding_text to
        # this many tokens (context tail first, then the embedded body copy).
        # Default 448 is sized for the target encoder's 512-token window
        # (see module docstring for the margin math).
        self.embedding_max_tokens = max(0, int(embedding_max_tokens))
        self.prose_strategy = prose_strategy
        self.tagger = tagger
        # id/position state (reset per document)
        self._used_ids: set = set()
        self._prose_counter = 0
        self._table_counter = 0
        self._equation_counter = 0
        self._reference_counter = 0
        self._position_counter = 0
        self._unit_counter = 0
        self._doc_id = ""
        self._meta: Dict[str, Any] = {}
        self._dedup_cache = dedup_cache or {}   # fingerprint -> first chunk id
        self._dedup_hits: List[str] = []

    # -- ids ----------------------------------------------------------

    def _unique(self, candidate: str) -> str:
        if candidate not in self._used_ids:
            self._used_ids.add(candidate)
            return candidate
        suffix = 1
        while f"{candidate}_{suffix}" in self._used_ids:
            suffix += 1
        out = f"{candidate}_{suffix}"
        self._used_ids.add(out)
        return out

    def _next_pos(self) -> int:
        pos = self._position_counter
        self._position_counter += 1
        return pos

    @staticmethod
    def _slug(text: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
        return s or "x"

    # -- chunk factory ------------------------------------------------

    def _make_chunk(self, chunk_id: str, text: str, chunk_type: str,
                    breadcrumb: List[str], *,
                    embedding_text: Optional[str] = None,
                    embedding_context: Optional[str] = None,
                    position: int,
                    retrieval_eligible: bool = True,
                    parent_id: Optional[str] = None,
                    object_id: Optional[str] = None,
                    section: Optional[str] = None,
                    subsection: Optional[str] = None,
                    table_id: Optional[str] = None,
                    figure_id: Optional[str] = None,
                    equation_id: Optional[str] = None,
                    reference_id: Optional[str] = None,
                    row_label: Optional[str] = None,
                    group_path: Optional[List[str]] = None,
                    citation_refs: Optional[List[str]] = None,
                    footnote_refs: Optional[List[str]] = None,
                    extra_metadata: Optional[Dict[str, Any]] = None) -> Chunk:
        text = " ".join((text or "").split()) if text else ""
        if not text:
            text = chunk_id  # never emit an empty chunk
        text = text.strip()
        metadata = dict(self._meta)
        metadata["full_breadcrumb"] = " > ".join(breadcrumb)
        metadata["unit_id"] = extra_metadata.pop("unit_id", "") if extra_metadata else ""
        metadata["source_format"] = "md"
        metadata["prose_strategy"] = self.prose_strategy
        metadata["embedding_model_family"] = "MedCPT"
        metadata["embedding_representation"] = "object_label_plus_chunk"
        if extra_metadata:
            metadata.update(extra_metadata)
        if embedding_text is None:
            embedding_text = self._embedding_text(text, breadcrumb,
                                                  context_tail=embedding_context)
        else:
            embedding_text = embedding_text.strip()
        chunk = Chunk(
            id=chunk_id,
            document_id=self._doc_id,
            text=text,
            embedding_text=embedding_text,
            chunk_type=chunk_type,  # type: ignore[arg-type]
            section=section if section is not None else (breadcrumb[0] if breadcrumb else None),
            subsection=subsection if subsection is not None else (
                breadcrumb[-1] if len(breadcrumb) > 1 else None),
            breadcrumb=list(breadcrumb),
            parent_id=parent_id,
            object_id=object_id,
            source_block_ids=[],
            table_id=table_id,
            figure_id=figure_id,
            equation_id=equation_id,
            reference_id=reference_id,
            row_label=row_label,
            group_path=list(group_path or []),
            citation_refs=citation_refs or [],
            footnote_refs=footnote_refs or [],
            metadata=metadata,
            document_position=position,
            retrieval_eligible=retrieval_eligible,
            chunk_version=CHUNK_VERSION,
        )
        # populated at chunk time: doubles as an overflow audit against
        # embedding_max_tokens before encoding ever runs
        chunk.embedding_token_count = (
            _estimate_tokens(embedding_text) if embedding_text else 0
        )
        chunk.metadata["embedding_budget"] = self.embedding_max_tokens
        chunk.metadata["embedding_tokens_within_budget"] = (
            not self.embedding_max_tokens
            or chunk.embedding_token_count <= self.embedding_max_tokens
        )
        if self.tagger is not None:
            chunk.concept_ids = self.tagger.tag(
                f"{chunk.embedding_text}\n{chunk.text}"
            )
        if chunk_type in ("paragraph", "list"):
            # bracketed citation numbers, raw; resolved to ref ids in a
            # post-pass once the reference list has been chunked
            nums = _citation_numbers(text)
            if nums:
                chunk.metadata["citation_numbers"] = nums
        if chunk_type in ("paragraph", "list") and chunk.retrieval_eligible:
            fp = self._fingerprint(chunk.text)
            first = self._dedup_cache.get(fp)
            if first is not None:
                chunk.retrieval_eligible = False
                chunk.metadata["dedup_of"] = first
                self._dedup_hits.append(chunk_id)
            else:
                self._dedup_cache[fp] = chunk_id
        return chunk

    @staticmethod
    def _fingerprint(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).casefold()

    def _embedding_text(
        self,
        text: str,
        breadcrumb: List[str],
        prefix: Optional[str] = None,
        context_tail: Optional[str] = None,
    ) -> str:
        """Build a self-contained MedCPT-oriented chunk representation.

        Original src/chunker.py encoding, minus breadcrumb: only the object
        label (``prefix``, e.g. "Table 1") followed by the chunk evidence.
        No document title and no section breadcrumb are embedded — that
        context is resolved from metadata at retrieval time, not duplicated
        into every vector.

        ``context_tail`` is accepted for API compatibility but is deliberately
        ignored. Neighbor evidence is represented by metadata and the units
        artifact rather than duplicated inside vectors.

        When the representation exceeds ``embedding_max_tokens``, structural
        context is reduced before the body is truncated. Stored ``text`` is
        never modified by this budget operation.
        """
        structural_parts: List[str] = []
        if prefix:
            structural_parts.append(str(prefix).strip())

        body = " ".join((text or "").split())
        head = "\n\n".join(p for p in structural_parts if p)

        budget = self.embedding_max_tokens
        if budget > 0:
            body_t = _estimate_tokens(body)
            head_t = _estimate_tokens(head) if head else 0

            if head_t + body_t > budget:
                # Drop the structural prefix before shortening the body.
                reduced = list(structural_parts)
                while len(reduced) > 1:
                    candidate = "\n\n".join(reduced)
                    if _estimate_tokens(candidate) + body_t <= budget:
                        break
                    reduced.pop()  # deepest breadcrumb/prefix first
                head = "\n\n".join(reduced)
                head_t = _estimate_tokens(head) if head else 0

            if head_t + body_t > budget:
                marker = " [...]"
                room = max(1, budget - head_t - _estimate_tokens(marker))
                body = (_truncate_tokens(body, room) + marker).strip()

        return "\n\n".join(part for part in (head, body) if part)

    # -- main entry ---------------------------------------------------

    def chunk_md(self, text: str, doc_id: Optional[str] = None) -> Tuple[List[Chunk], List[UnitRecord], Dict[str, Any]]:
        """Chunk one Markdown article into ``(chunks, units, report)``."""
        self._reset()
        self._doc_id = doc_id or self._digest_id(text)
        front, body = _parse_front_matter(text)
        self._meta = _document_metadata(front, self._doc_id)

        body_clean, fences = _extract_fences(body)
        roots = self._drop_doc_title_root(build_section_tree(body_clean, fences))

        chunks: List[Chunk] = []
        units: List[UnitRecord] = []
        seen_blocks = 0

        def process(node: MDNode, breadcrumb: List[str]) -> None:
            nonlocal seen_blocks
            # breadcrumb = the path INCLUDING this heading (v1 semantics:
            # section() <- breadcrumb[0], subsection() <- breadcrumb[-1])
            path = breadcrumb + [node.title]
            _attach_table_context(node.blocks)
            _attach_figure_caption(node.blocks)
            seen_blocks += len(node.blocks)
            unit = self._section_unit(node, path, chunks, units)
            if unit:
                units.append(unit)
            for child in node.children:
                process(child, path)

        for root in roots:
            process(root, [])

        # Resolve bracketed citation numbers -> reference ids now that the
        # whole document (including its reference list) has been chunked.
        # Numbers outside the parsed reference range are dropped: they are
        # almost always non-citations (years, list numbering, ...). Mapping
        # assumes the reference list is numbered in citation order, which
        # bracketed style implies.
        ref_count = self._reference_counter
        for c in chunks:
            nums = c.metadata.get("citation_numbers")
            if not nums:
                continue
            valid = [n for n in nums if 1 <= n <= ref_count] if ref_count else []
            if valid:
                c.citation_refs = [f"ref_{n - 1}" for n in valid]
                c.metadata["citation_numbers"] = valid
            else:
                del c.metadata["citation_numbers"]

        return chunks, units, self._report(chunks, units, seen_blocks)

    @staticmethod
    def _drop_doc_title_root(roots: List[MDNode]) -> List[MDNode]:
        """The generator emits ``# {title}`` as the only h1. That is the
        document title, not a section: drop it, promote its children, and
        preserve any direct blocks (an unheaded abstract / citation line)
        by prepending them to the first child."""
        if len(roots) == 1 and roots[0].level == 1:
            root = roots[0]
            if not root.children:
                return [root]
            if root.blocks:
                root.children[0].blocks = root.blocks + root.children[0].blocks
            return root.children
        return roots

    def _reset(self) -> None:
        self._used_ids = set()
        self._prose_counter = 0
        self._table_counter = 0
        self._equation_counter = 0
        self._reference_counter = 0
        self._position_counter = 0
        self._unit_counter = 0
        self._dedup_hits = []
        self._meta = {}

    @staticmethod
    def _digest_id(text: str) -> str:
        return "MD" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

    # -- node processing ----------------------------------------------

    def _section_unit(self, node: MDNode, path: List[str],
                      chunks: List[Chunk], units: List[UnitRecord]) -> Optional[UnitRecord]:
        """Chunk one node's direct blocks; return its section unit."""
        unit_id = self._unique(f"{self._doc_id}_unit_{self._unit_counter}")
        self._unit_counter += 1
        unit_children: List[str] = []
        kind = classify_section_title(node.title)

        if kind == "references":
            for b in node.blocks:
                if b.kind == "list":
                    for item, raw in zip(b.items, b.raw_items):
                        c = self._reference_chunk(item, path, raw=raw, unit_id=unit_id)
                        chunks.append(c)
                        unit_children.append(c.id)
                elif b.text.strip():
                    c = self._reference_chunk(b.text, path, raw=b.text, unit_id=unit_id)
                    chunks.append(c)
                    unit_children.append(c.id)
            unit_text = "\n".join(node_plain(node.blocks))
            if not unit_children:
                return None
            return UnitRecord(unit_id, self._doc_id, "section", node.title,
                              path, unit_text or "References", unit_children)

        if kind == "administrative":
            for b in node.blocks:
                if b.kind == "table":
                    sub = self._table_chunks(b, path, unit_id, units)
                    chunks.extend(sub)
                    unit_children.extend(c.id for c in sub)
                    continue
                if b.kind == "figure":
                    c = self._figure_chunk(b, path, unit_id)
                    chunks.append(c)
                    unit_children.append(c.id)
                    continue
                if b.kind == "equation" and b.text.strip():
                    c = self._equation_chunk(b, path, unit_id)
                    chunks.append(c)
                    unit_children.append(c.id)
                    continue
                text = b.text if b.kind in ("paragraph", "code") else None
                if text and text.strip():
                    c = self._administrative_chunk(text, path, unit_id)
                    chunks.append(c)
                    unit_children.append(c.id)
                elif b.kind == "list":
                    text = "\n".join(f"- {it}" for it in b.items)
                    if text:
                        c = self._administrative_chunk(text, path, unit_id)
                        chunks.append(c)
                        unit_children.append(c.id)
            if not unit_children:
                return None
            return UnitRecord(unit_id, self._doc_id, "section", node.title,
                              path, "\n".join(node_plain(node.blocks)), unit_children)

        # content
        prose: List[RawBlock] = []
        for b in node.blocks:
            if b.kind == "paragraph":
                if b.text.strip():
                    prose.append(b)
                continue
            if b.kind == "list":
                c = self._list_chunk(b, path, unit_id)
                if c:
                    chunks.append(c)
                    unit_children.append(c.id)
                continue
            if b.kind == "table":
                sub = self._table_chunks(b, path, unit_id, units)
                chunks.extend(sub)
                unit_children.extend(c.id for c in sub)
                continue
            if b.kind == "figure":
                c = self._figure_chunk(b, path, unit_id)
                chunks.append(c)
                unit_children.append(c.id)
                continue
            if b.kind == "equation":
                if b.text.strip():
                    c = self._equation_chunk(b, path, unit_id)
                    chunks.append(c)
                    unit_children.append(c.id)
                continue
            if b.kind == "code":
                if b.text.strip():
                    c = self._paragraph_chunk(b.text, path, unit_id,
                                              extra_metadata={"original_block_type": "code"})
                    chunks.append(c)
                    unit_children.append(c.id)
                continue
        for c in self._prose_chunks(prose, path, unit_id):
            chunks.append(c)
            unit_children.append(c.id)
        unit_text = "\n".join(node_plain(node.blocks))
        return UnitRecord(unit_id, self._doc_id, "section", node.title, path,
                          unit_text, unit_children)

    # -- prose: paragraph-first (window = legacy A/B baseline) ---------

    def _prose_chunks(self, paragraphs: List[RawBlock], path: List[str],
                      unit_id: str) -> List[Chunk]:
        """Prose strategy dispatch.

        Both strategies start the same way: a paragraph that alone exceeds
        the soft budget is sentence-split (and hard-capped). Then:

        * ``paragraph`` (default): one piece = one retrieval chunk; a
          below-floor piece may merge into ONE neighbor if the pair fits.
        * ``window`` (legacy, for eval A/B): pieces are packed into
          ~max_tokens windows with the previous window's tail prepended to
          the stored text.
        """
        if not paragraphs:
            return []
        pieces: List[RawBlock] = []
        for p in paragraphs:
            if _estimate_tokens(p.text) > self.max_tokens:
                pieces.extend(self._split_long_paragraph(p))
            else:
                pieces.append(p)
        if self.prose_strategy == "window":
            return self._window_chunks(pieces, path, unit_id)
        pieces = self._merge_tiny(pieces)
        return self._paragraph_first_chunks(pieces, path, unit_id)

    def _paragraph_first_chunks(self, pieces: List[RawBlock], path: List[str],
                                unit_id: str) -> List[Chunk]:
        """One paragraph (or merged pair) = one independent retrieval chunk.

        Neighboring evidence is linked after creation rather than copied into
        the embedding representation. This keeps each vector attributable to
        its own source paragraph while still enabling deterministic expansion.
        """
        chunks: List[Chunk] = [
            self._paragraph_chunk(p.text, path, unit_id)
            for p in pieces
        ]

        for i, chunk in enumerate(chunks):
            prev_id = chunks[i - 1].id if i > 0 else ""
            next_id = chunks[i + 1].id if i + 1 < len(chunks) else ""
            chunk.metadata["prev_chunk_id"] = prev_id
            chunk.metadata["next_chunk_id"] = next_id
            chunk.metadata["context_expansion"] = {
                "unit_id": unit_id,
                "prev": prev_id,
                "self": chunk.id,
                "next": next_id,
            }
        return chunks

    def _window_chunks(self, pieces: List[RawBlock], path: List[str],
                       unit_id: str) -> List[Chunk]:
        """LEGACY strategy, kept verbatim for the eval-gate A/B baseline:
        pack adjacent pieces into ~max_tokens windows; prepend the previous
        window's tail to the STORED text (old overlap semantics) when the
        result stays under the hard ceiling."""
        windows: List[List[RawBlock]] = []
        cur: List[RawBlock] = []
        cur_tokens = 0
        for p in pieces:
            tokens = _estimate_tokens(p.text)
            sep = 2 if cur else 0
            if cur and cur_tokens + sep + tokens > self.max_tokens:
                windows.append(cur)
                cur = []
                cur_tokens = 0
            cur.append(p)
            cur_tokens += tokens + (2 if len(cur) > 1 else 0)
            if cur_tokens > self.max_tokens and len(cur) > 1:
                windows.append(cur)
                cur = []
                cur_tokens = 0
        if cur:
            windows.append(cur)

        chunks: List[Chunk] = []
        prev_tail = ""
        for w in windows:
            window_text = "\n\n".join(p.text for p in w)
            text = window_text
            if prev_tail and self.overlap > 0:
                candidate = prev_tail + "\n\n" + window_text
                # never let the prepended overlap push a chunk past the
                # hard ceiling: drop the tail rather than emit oversized
                if _estimate_tokens(candidate) <= self.hard_max_tokens:
                    text = candidate
            chunks.append(self._paragraph_chunk(text, path, unit_id))
            prev_tail = self._tail_tokens(window_text, self._overlap_tokens) \
                if self.overlap > 0 else ""
        return chunks

    def _merge_tiny(self, pieces: List[RawBlock]) -> List[RawBlock]:
        """Merge a below-floor piece into a neighbor when the pair fits the
        soft budget. Backward pass first (tiny joins the previous piece);
        a forward pass then rescues tiny pieces whose predecessor was full
        by joining them to the NEXT piece. This is the ONLY place two
        paragraphs are ever combined in the paragraph strategy."""
        if self.min_paragraph_tokens <= 0 or not pieces:
            return pieces
        out: List[RawBlock] = []
        for p in pieces:
            if (out
                    and _estimate_tokens(p.text) < self.min_paragraph_tokens
                    and _estimate_tokens(out[-1].text) + 2
                        + _estimate_tokens(p.text) <= self.max_tokens):
                out[-1] = RawBlock(kind="paragraph",
                                   text=out[-1].text + "\n\n" + p.text)
            else:
                out.append(p)
        merged: List[RawBlock] = []
        i = 0
        while i < len(out):
            p = out[i]
            if (i + 1 < len(out)
                    and _estimate_tokens(p.text) < self.min_paragraph_tokens
                    and _estimate_tokens(p.text) + 2
                        + _estimate_tokens(out[i + 1].text) <= self.max_tokens):
                merged.append(RawBlock(kind="paragraph",
                                       text=p.text + "\n\n" + out[i + 1].text))
                i += 2
            else:
                merged.append(p)
                i += 1
        return merged

    def _split_long_paragraph(self, p: RawBlock) -> List[RawBlock]:
        """Split one paragraph into <= max_tokens pieces at sentence
        boundaries (deterministic; no LLM). A piece that still exceeds
        hard_max_tokens (one giant sentence) is hard-capped; the truncation
        marker's own tokens are accounted for so no piece ever exceeds the
        declared ceiling."""
        sentences = re.split(r"(?<=[.!?])\s+", p.text.strip())
        pieces: List[RawBlock] = []
        cur_text: List[str] = []
        cur_tokens = 0
        for sent in sentences:
            t = _estimate_tokens(sent)
            if cur_tokens + t > self.max_tokens and cur_text:
                pieces.append(RawBlock(kind="paragraph", text=" ".join(cur_text)))
                cur_text = []
                cur_tokens = 0
            cur_text.append(sent)
            cur_tokens += t
        if cur_text:
            pieces.append(RawBlock(kind="paragraph", text=" ".join(cur_text)))
        return [self._hard_cap_piece(piece) for piece in pieces] or [p]

    def _hard_cap_piece(self, piece: RawBlock) -> RawBlock:
        """Clamp a pathological piece to hard_max_tokens TOKENS — not words
        (words under-count ~1.3-1.6x and would let oversized chunks
        through) — with the ``[...]`` marker's tokens budgeted up front."""
        if _estimate_tokens(piece.text) <= self.hard_max_tokens:
            return piece
        marker = " [...]"
        limit = max(1, self.hard_max_tokens - _estimate_tokens(marker))
        kept = _truncate_tokens(piece.text, limit)
        return RawBlock(kind=piece.kind, text=(kept + marker).strip())

    @staticmethod
    def _tail_tokens(text: str, n: int, max_fraction: float = 0.5) -> str:
        """Last ~``n`` TOKENS of ``text``.

        Token-approximate (per-word token counts summed backwards). For
        overlap tails ``max_fraction`` defaults to 0.5 so a short window can
        never near-duplicate the whole next chunk; pass 1.0 when trimming an
        arbitrary span (e.g. fitting an embedding budget).
        """
        words = re.findall(r"\S+", text or "")
        if not words or n <= 0:
            return ""
        word_cap = max(1, int(len(words) * max_fraction))
        out: List[str] = []
        count = 0
        for pos in range(len(words) - 1, -1, -1):
            if len(out) >= word_cap:
                break
            w = words[pos]
            out.append(w)
            count += max(1, _estimate_tokens(w))
            if count >= n:
                break
        return " ".join(reversed(out))

    # -- chunk builders -----------------------------------------------

    def _paragraph_chunk(self, text: str, path: List[str], unit_id: str,
                         extra_metadata: Optional[Dict[str, Any]] = None,
                         embedding_context: Optional[str] = None) -> Chunk:
        cid = self._unique(f"{self._doc_id}_{self._prose_counter}")
        self._prose_counter += 1
        return self._make_chunk(cid, text, "paragraph", path,
                                position=self._next_pos(),
                                parent_id=unit_id,
                                embedding_context=embedding_context,
                                extra_metadata={**(extra_metadata or {}), "unit_id": unit_id})

    def _list_chunk(self, b: RawBlock, path: List[str], unit_id: str) -> Optional[Chunk]:
        text = "\n".join(f"- {it}" for it in b.items)
        if not text.strip():
            return None
        cid = self._unique(f"{self._doc_id}_{self._prose_counter}")
        self._prose_counter += 1
        return self._make_chunk(cid, text, "list", path,
                                position=self._next_pos(),
                                parent_id=unit_id,
                                extra_metadata={"unit_id": unit_id})

    # -- tables -------------------------------------------------------

    def _table_id(self, b: RawBlock) -> Tuple[str, str]:
        """Return ``(doc_local_table_id, display_label)``."""
        if b.label:
            return self._unique(self._slug(b.label)), b.label
        n = self._table_counter
        self._table_counter += 1
        return self._unique(f"table_{n}"), f"Table {n}"

    def _table_chunks(self, b: RawBlock, path: List[str], unit_id: str,
                      units: List[UnitRecord]) -> List[Chunk]:
        table_id, display = self._table_id(b)
        # allocate the TABLE unit id up front so every chunk it contains can
        # point at it (metadata.table_unit_id): a row's parent_id is the
        # summary CHUNK; table_unit_id is the direct route to full-table text
        table_unit_id = self._unique(f"{self._doc_id}_unit_{self._unit_counter}")
        self._unit_counter += 1

        columns = list(b.header)
        if not columns and b.rows:
            columns = [f"column_{i}" for i in range(len(b.rows[0]))]

        summary_parts = []
        if b.label or b.caption:
            summary_parts.append(f"{b.label + ': ' if b.label else ''}{b.caption}".strip())
        if columns:
            summary_parts.append("Columns:\n" + "\n".join(f"- {c}" for c in columns))
        summary_text = "\n\n".join(p for p in summary_parts if p) or display
        table_context = display
        if b.caption:
            table_context = f"{display}: {b.caption}"

        summary_id = self._unique(f"{self._doc_id}_{table_id}_summary")
        s_pos = self._next_pos()
        summary = self._make_chunk(
            summary_id, summary_text, "table_summary", path,
            embedding_text=self._embedding_text(
                summary_text,
                path,
                prefix=f"Table: {table_context}",
            ),
            position=s_pos, object_id=table_id, table_id=table_id,
            extra_metadata={"unit_id": unit_id, "table_id": table_id,
                            "columns": columns, "table_unit_id": table_unit_id},
        )
        out = [summary]

        # current spanning-subheader group (see row loop): carries over to
        # every following row's group_path + embedding prefix
        current_group: List[str] = []
        for i, row in enumerate(b.rows):
            cells = [self._clean_cell(c) for c in row]
            if not any(cells):
                continue  # fully empty row -> no chunk
            non_empty = [c for c in cells if c]
            if (len(cells) > 1 and len(non_empty) > 1
                    and len(set(non_empty)) == 1
                    and " " in non_empty[0]):
                # Spanning subheader row: the MD generator flattens a JATS
                # colspan header by repeating its label into every cell.
                # That is group context, not evidence — emitting it as a
                # chunk would put the same sentence in the index N times.
                # The " " guard keeps all-identical NUMERIC data rows
                # (e.g. "100 100 100") from being misread as subheaders.
                label = non_empty[0].rstrip(":").strip()
                current_group = [label] if label else []
                continue
            if len(non_empty) == 1 and non_empty[0].strip() and len(columns) > 1:
                # One populated cell (whatever the padding): a JATS colspan
                # subheader ("Sex", "Median (IQR)", panel labels "a.
                # Stepwise ..."). Group context — it must qualify the
                # FOLLOWING rows via group_path, never become a data row
                # full of "—".
                current_group = [non_empty[0].rstrip(":").strip()]
                continue
            if columns and all(
                    (a or "").strip() == (b or "").strip()
                    for a, b in zip(cells, columns)):
                # Repeated header row inside a multi-panel table: identical
                # to the column headings -> context, not evidence.
                current_group = [(cells[0] or "").rstrip(":").strip()]
                continue
            if not cells[0].strip() and len(non_empty) > 0:
                # Column-context row: first cell empty, values elsewhere
                # (denominator rows "n1 = 1971 | n2 = 1900", format headers
                # "No. (%)"). Consume as context, not as a "Row: —" chunk.
                ctx = " ".join(non_empty)
                if ctx:
                    current_group = [ctx]
                continue
            row_label = cells[0] if cells else ""
            text_bits: List[str] = []
            if columns:
                first_header = columns[0].strip() or "Row"
                text_bits.append(
                    f"{first_header}: {row_label if row_label.strip() else '—'}"
                )
            elif row_label:
                text_bits.append(f"Row: {row_label}")
            for j in range(1, max(len(columns), len(cells))):
                header = (
                    columns[j].strip()
                    if j < len(columns) and columns[j].strip()
                    else f"column_{j}"
                )
                val = cells[j] if j < len(cells) else ""
                # Empty cells are explicit so "not reported" is distinguishable
                # from a missing structural cell.
                text_bits.append(f"{header}: {val if val.strip() else '—'}")
            row_text = "\n".join(text_bits) or (row_label or "table row")
            rid = self._unique(f"{self._doc_id}_{table_id}_row_{i}")
            prefix = f"Table: {table_context}"
            if current_group:
                prefix += f" - Group: {current_group[0]}"
            out.append(self._make_chunk(
                rid, row_text, "table_row", path,
                embedding_text=self._embedding_text(row_text, path, prefix=prefix),
                position=self._next_pos(), parent_id=summary_id,
                object_id=table_id, table_id=table_id,
                row_label=row_label or None,
                group_path=list(current_group),
                extra_metadata={"unit_id": unit_id, "table_id": table_id,
                                "row_index": i,
                                "table_unit_id": table_unit_id},
            ))

        if b.footnotes:
            fn_text = "\n".join(
                f"[{m}] {t}" if m else t for m, t in b.footnotes if t
            )
            if fn_text:
                fid = self._unique(f"{self._doc_id}_{table_id}_footnotes")
                out.append(self._make_chunk(
                    fid, fn_text, "table_footnotes", path,
                    embedding_text=self._embedding_text(fn_text, path,
                                                        prefix=f"{display} footnotes"),
                    position=self._next_pos(), parent_id=summary_id,
                    object_id=table_id, table_id=table_id,
                    extra_metadata={"unit_id": unit_id, "table_id": table_id,
                                    "table_unit_id": table_unit_id},
                ))

        # P1: a first-class TABLE unit (parent context for its row children)
        unit_lines: List[str] = []
        head = f"{b.label + ': ' if b.label else ''}{b.caption}".strip()
        if head:
            unit_lines.append(head)
        if b.header:
            unit_lines.append(" | ".join(b.header))
        unit_lines.extend(" | ".join(row) for row in b.rows)
        unit_lines.extend(f"[{m}] {t}" if m else t for m, t in b.footnotes)
        units.append(UnitRecord(
            unit_id=table_unit_id,
            document_id=self._doc_id, kind="table", title=display,
            breadcrumb=path,
            text="\n".join(unit_lines),
            chunk_ids=[c.id for c in out],
        ))

        # Direct post-retrieval table expansion routes. Keep the list small so
        # metadata remains inexpensive while the full TABLE unit remains the
        # authoritative source for complete expansion.
        for i, chunk in enumerate(out):
            neighbors = [
                c.id for j, c in enumerate(out)
                if j != i
            ][:8]
            chunk.metadata["table_context"] = {
                "table_unit_id": table_unit_id,
                "summary_chunk_id": summary_id,
                "neighbor_chunk_ids": neighbors,
            }
        return out

    @staticmethod
    def _clean_cell(text: str) -> str:
        return " ".join((text or "").replace("\n", " ").split())

    # -- figures / equations ------------------------------------------

    def _figure_chunk(self, b: RawBlock, path: List[str], unit_id: str) -> Chunk:
        figure_id = self._unique(self._slug(b.label or "figure"))
        if b.label and b.caption and b.caption.lower().startswith(b.label.lower()):
            text = b.caption
        else:
            text = " ".join([b.label or "", b.caption or ""]).strip()
        if not text:
            text = f"Figure {figure_id}"
        return self._make_chunk(
            self._unique(f"{self._doc_id}_{figure_id}"), text, "figure", path,
            embedding_text=self._embedding_text(text, path, prefix=f"Figure {figure_id}"),
            position=self._next_pos(), parent_id=unit_id,
            object_id=figure_id, figure_id=figure_id,
            extra_metadata={"unit_id": unit_id, "figure_id": figure_id,
                            "image_ref": b.image_ref},
        )

    def _equation_chunk(self, b: RawBlock, path: List[str], unit_id: str) -> Chunk:
        eid = self._unique(f"equation_{self._equation_counter}")
        self._equation_counter += 1
        return self._make_chunk(
            self._unique(f"{self._doc_id}_{eid}"), b.text, "equation", path,
            embedding_text=self._embedding_text(b.text, path, prefix=f"Equation {eid}"),
            position=self._next_pos(), parent_id=unit_id,
            object_id=eid, equation_id=eid,
            extra_metadata={"unit_id": unit_id, "equation_id": eid, "latex": b.text},
        )

    # -- references / administrative ----------------------------------

    def _reference_chunk(self, text: str, path: List[str], raw: str = "",
                         unit_id: str = "") -> Chunk:
        ref_id = f"ref_{self._reference_counter}"
        self._reference_counter += 1
        doi = re.search(r"https://doi\.org/([^)\s]+)", raw or text)
        pmid = re.search(r"https://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", raw or text)
        pmcid = re.search(r"https://www\.ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)", raw or text)
        return self._make_chunk(
            self._unique(f"{self._doc_id}_{ref_id}"), text, "reference", path,
            embedding_text="",
            position=self._next_pos(), parent_id=unit_id or None,
            object_id=ref_id, reference_id=ref_id,
            retrieval_eligible=False,
            extra_metadata={"unit_id": unit_id,
                            "reference_id": ref_id,
                            "doi": doi.group(1) if doi else "",
                            "pmid": pmid.group(1) if pmid else "",
                            "pmcid": pmcid.group(1) if pmcid else ""},
        )

    def _administrative_chunk(self, text: str, path: List[str], unit_id: str) -> Chunk:
        cid = self._unique(f"{self._doc_id}_{self._prose_counter}")
        self._prose_counter += 1
        return self._make_chunk(cid, text, "administrative", path,
                                position=self._next_pos(), parent_id=unit_id,
                                retrieval_eligible=False,
                                extra_metadata={"unit_id": unit_id})

    # -- report -------------------------------------------------------

    def _report(self, chunks: List[Chunk], units: List[UnitRecord],
                source_blocks: int) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        eligible = 0
        entities = 0
        cited = 0
        for c in chunks:
            by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
            if c.retrieval_eligible:
                eligible += 1
            if c.concept_ids:
                entities += 1
            if c.citation_refs:
                cited += 1
        return {
            "document_id": self._doc_id,
            "source_blocks": source_blocks,
            "chunks": len(chunks),
            "retrieval_eligible": eligible,
            "chunks_by_type": dict(sorted(by_type.items())),
            "units": len(units),
            "entities_tagged_chunks": entities,
            "chunks_with_citations": cited,
            "dedup_suppressed": len(self._dedup_hits),
            "dedup_of": list(self._dedup_hits),
            "chunker_config": {
                "prose_strategy": self.prose_strategy,
                "max_tokens": self.max_tokens,
                "hard_max_tokens": self.hard_max_tokens,
                "overlap": self.overlap,
                "neighbor_context_in_embedding": False,
                "context_expansion_mode": "metadata_and_units",
                "min_paragraph_tokens": self.min_paragraph_tokens,
                "embedding_max_tokens": self.embedding_max_tokens,
                # design target recorded for the downstream ENCODING step to
                # validate against (this file never loads an encoder)
                "target_encoder_window": TARGET_ENCODER_WINDOW,
                "tokenizer": _TOKENIZER_KIND,
                "embedding_representation": "structure_plus_chunk",
                "medcpt_document_input_note": (
                    "Represent each retrieval unit as object label plus "
                    "self-contained chunk text (original src/chunker.py "
                    "encoding, no document title, no breadcrumb); use units/"
                    "neighbor ids and metadata for post-retrieval context."
                ),
            },
            "meta": dict(self._meta),
        }

    @property
    def dedup_hits(self) -> List[str]:
        return list(self._dedup_hits)


def node_plain(blocks: List[RawBlock]) -> List[str]:
    """Rendered plain text of a node's blocks (used for unit text + report).

    Table rows are rendered from the RAW rows (including subheader rows) so
    unit text keeps full fidelity even though subheaders are not chunks."""
    out: List[str] = []
    for b in blocks:
        if b.kind == "paragraph" and b.text.strip():
            out.append(b.text.strip())
        elif b.kind == "list":
            out.extend(f"- {it}" for it in b.items)
        elif b.kind == "table":
            head = f"{b.label + ': ' if b.label else ''}{b.caption}".strip()
            if head:
                out.append(head)
            if b.header:
                out.append(" | ".join(b.header))
            for row in b.rows:
                out.append(" | ".join(row))
            for m, t in b.footnotes:
                out.append(f"[{m}] {t}" if m else t)
        elif b.kind == "figure":
            out.append(f"{b.label + ' ' if b.label else ''}{b.caption}".strip())
        elif b.kind in ("equation", "code") and b.text.strip():
            out.append(b.text.strip())
    return [o for o in out if o]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _inspect_single(args) -> int:
    """Test mode: chunk ONE document (md/xml path or bare PMC id) and print
    every chunk so single values can be eyeballed. XML inputs are converted
    to Markdown first; the parquet land in .single_test/chunks/."""
    from src.processing.jats_to_md import convert_one

    name = args.file.strip()
    p = Path(name)
    if not p.exists():
        for cand in (Path("data/chunked") / name,
                     Path("data/chunked") / (name + ".xml"),
                     Path("data/chunked") / (name + ".md"),
                     Path(".") / (name + ".xml"), Path(".") / (name + ".md")):
            if cand.exists():
                p = cand
                break
    if not p.exists():
        print(f"file not found: {args.file} (tried data/chunked/ and CWD)", file=sys.stderr)
        return 1

    if p.suffix.lower() == ".xml":
        out_md = Path(".single_test/md") / (p.stem + ".md")
        res = convert_one(p, out_md, overwrite=True, verbose=False)
        if res["status"] != "converted":
            print(f"conversion failed: {p}", file=sys.stderr)
            return 2
        md_text = out_md.read_text(encoding="utf-8")
        doc_id = p.stem
    else:
        md_text = p.read_text(encoding="utf-8")
        doc_id = p.stem

    chunker = MDChunker(
        max_tokens=args.max_tokens,
        hard_max_tokens=args.hard_max_tokens or 2 * args.max_tokens,
        min_paragraph_tokens=args.min_paragraph_tokens,
        overlap=args.overlap,
        embedding_max_tokens=args.embedding_max_tokens,
        prose_strategy=args.prose_strategy,
        tagger=LexiconTagger(_load_lexicon(Path(args.lexicon))) if args.lexicon else None,
    )
    chunks, units, report = chunker.chunk_md(md_text, doc_id=doc_id)

    # also land the parquet for pandas inspection
    out_dir = Path(".single_test/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    _chunks_to_df(chunks).to_parquet(out_dir / f"{doc_id}.parquet", index=False)

    print(f"=== {doc_id} | {len(chunks)} chunks | {len(units)} units "
          f"| eligible {report['retrieval_eligible']} ===")
    by_type: Dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    print("types:", dict(sorted(by_type.items())))

    # tables in detail (grouped by table)
    tables = [c for c in chunks if c.chunk_type in
              ("table_summary", "table_row", "table_footnotes")]
    from collections import defaultdict
    grouped = defaultdict(list)
    for c in tables:
        grouped[c.table_id].append(c)
    for tid, tchunks in sorted(grouped.items()):
        print(f"\n-- TABLE {tid}")
        for c in tchunks:
            if c.chunk_type == "table_summary":
                print("  SUMMARY:", c.text.replace("\n", " | ")[:200])
            elif c.chunk_type == "table_row":
                print(f"  ROW [{c.id.split('_')[-1]}] label={c.row_label!r} "
                      f"group={c.group_path!r} parent={c.parent_id.split('_')[-1]}")
                print("      ", c.text.replace("\n", " | ")[:220])
            else:
                print("  FOOTNOTES:", c.text.replace("\n", " | ")[:160])

    print("\n-- OTHER CHUNKS (first 3 of each type, text truncated)")
    shown: Dict[str, int] = {}
    for c in chunks:
        if c.chunk_type in ("table_summary", "table_row", "table_footnotes"):
            continue
        if shown.get(c.chunk_type, 0) >= 3:
            continue
        shown[c.chunk_type] = shown.get(c.chunk_type, 0) + 1
        print(f"  [{c.chunk_type}] {c.id}: {c.text[:110]!r}")
    return 0


def _discover_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".md" else []
    if input_path.is_dir():
        return sorted(input_path.rglob("*.md"))
    return []


def _chunks_to_df(chunks: List[Chunk]) -> "Any":
    import pandas as pd

    rows = []
    for c in chunks:
        d = asdict(c)
        # json-encode columns that pyarrow cannot serialize as objects
        # (ensure_ascii=False keeps metadata human-readable: real en-dashes,
        # U+2212 minus signs, etc. — no \uXXXX escapes)
        d["metadata"] = json.dumps(d["metadata"], default=str, ensure_ascii=False)
        for col in ("group_path", "citation_refs", "footnote_refs", "concept_ids",
                    "source_block_ids"):
            d[col] = json.dumps(d[col], default=str, ensure_ascii=False)
        rows.append(d)
    df = pd.DataFrame(rows)
    # keep breadcrumb as a real list for the corpus schema
    return df


def _units_to_df(units: List[UnitRecord]) -> "Any":
    import pandas as pd

    rows = []
    for u in units:
        d = asdict(u)
        # both structural columns kept as real lists (consistent with
        # CORPUS_COLUMNS breadcrumb; nothing json-encodes units)
        d["chunk_ids"] = list(d["chunk_ids"])
        rows.append(d)
    return pd.DataFrame(rows)


def _global_dedup_pass(chunks_out: Path) -> Dict[str, int]:
    """Deterministic exact-duplicate suppression across documents.

    A sorted post-pass over the WRITTEN parquets (never inside the worker
    pool), in ``(document_id, document_position)`` order, so which duplicate
    wins never depends on thread scheduling.

    Idempotent: suppression marks from a previous pass are reset before the
    pass re-runs, so re-running converges to the same output. ``dedup_of``
    always points at an *unsuppressed* chunk (first occurrence), so no
    dedup chains can form.

    NOTE: under the paragraph strategy fingerprints are single paragraphs,
    so repeated boilerplate (funding statements, consent language) now
    deduplicates much more aggressively than under window packing. That is
    by design; the report counts make it visible.

    Rewrites the parquet AND refreshes the report in the sidecar meta.json.
    Suppressed chunks stay in their parent units (retrieval eligibility is
    the only thing that changes).

    Returns ``{"documents": n_scanned, "suppressed": n_marked}``.
    """
    import pandas as pd

    parquets = sorted(chunks_out.glob("*.parquet"))
    seen: Dict[str, str] = {}          # fingerprint -> first (winning) chunk id
    suppressed_total = 0
    scanned = 0
    for pq in parquets:
        df = pd.read_parquet(pq)
        if df is None or df.empty:
            continue
        scanned += 1
        # rows are already written in document order; stable sort is a
        # belt-and-braces guarantee of determinism
        df = df.sort_values("document_position", kind="mergesort").reset_index(drop=True)
        changed = False
        hit_ids: List[str] = []
        for idx in range(len(df)):
            if df.at[idx, "chunk_type"] not in ("paragraph", "list"):
                continue
            meta = json.loads(df.at[idx, "metadata"])
            was_marked = "dedup_of" in meta
            # a paragraph/list chunk is dedup-eligible if it is currently
            # eligible OR was suppressed by a previous dedup run (idempotency)
            eligible = bool(df.at[idx, "retrieval_eligible"]) or was_marked
            meta.pop("dedup_of", None)
            if not eligible:
                continue
            fp = MDChunker._fingerprint(str(df.at[idx, "text"]))
            first = seen.get(fp)
            if first is not None:
                meta["dedup_of"] = first
                df.at[idx, "retrieval_eligible"] = False
                df.at[idx, "metadata"] = json.dumps(meta, default=str,
                                                    ensure_ascii=False)
                changed = True
                hit_ids.append(str(df.at[idx, "id"]))
            else:
                seen[fp] = str(df.at[idx, "id"])
                if was_marked:
                    # earlier winner changed / disappeared: un-suppress
                    df.at[idx, "retrieval_eligible"] = True
                    df.at[idx, "metadata"] = json.dumps(meta, default=str,
                                                        ensure_ascii=False)
                    changed = True
        if not changed:
            continue
        suppressed_total += len(hit_ids)
        df.to_parquet(pq, index=False)
        meta_path = pq.parent / (pq.stem + ".meta.json")
        if meta_path.exists():
            try:
                outer = json.loads(meta_path.read_text(encoding="utf-8"))
                report = outer.get("report") or {}
                report["retrieval_eligible"] = int(df["retrieval_eligible"].sum())
                report["dedup_suppressed"] = len(hit_ids)
                report["dedup_of"] = hit_ids
                outer["report"] = report
                meta_path.write_text(json.dumps(outer, ensure_ascii=False),
                                     encoding="utf-8")
            except Exception:  # noqa: BLE001 - report refresh is best-effort
                pass
    return {"documents": scanned, "suppressed": suppressed_total}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chunker v2: structure-first, paragraph-first chunking of "
                    "Markdown articles (no LLM). Chunking only — encoding is a "
                    "separate downstream step. Defaults are sized for a MedCPT "
                    "encoder (512-token window); no flags needed."
    )
    parser.add_argument("--input", default="",
                        help="MD file/dir. Not required when --file is given.")
    parser.add_argument("--chunks-out", default="chunks_v2",
                        help="Directory for per-document chunk parquets.")
    parser.add_argument("--units-out", default="units_v2",
                        help="Directory for per-document unit parquets.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prose-strategy", choices=("paragraph", "window"),
                        default="paragraph",
                        help="paragraph: one paragraph = one retrieval chunk "
                             "(default, recommended). window: legacy "
                             "pack-to-budget chunks (kept as the eval A/B baseline).")
    parser.add_argument("--max-tokens", type=int, default=320,
                        help="Soft token budget per prose chunk (default 320: "
                             "head + context + body fits the 448-token embedding "
                             "budget without body truncation).")
    parser.add_argument("--hard-max-tokens", type=int, default=0,
                        help="Absolute per-chunk token ceiling (default: 2x --max-tokens).")
    parser.add_argument("--min-paragraph-tokens", type=int, default=0,
                        help="Paragraph strategy: pieces below this floor merge "
                             "into one neighbor when the pair fits --max-tokens. "
                             "0 disables merging.")
    parser.add_argument("--overlap", type=float, default=0.0,
                         help="Compatibility field. Recommended paragraph mode "
                         "does not copy neighboring evidence into embeddings; "
                         "adjacency is stored in metadata/units. Legacy window "
                         "mode may still use this value.")

    parser.add_argument("--embedding-max-tokens", type=int,
                        default=DEFAULT_EMBEDDING_MAX_TOKENS,
                        help=f"Trim embedding_text to at most N tokens "
                             f"(default {DEFAULT_EMBEDDING_MAX_TOKENS}, sized for "
                             f"the target encoder's {TARGET_ENCODER_WINDOW}-token "
                             f"window: (512 - 2 special tokens) * ~0.88, the "
                             f"margin absorbing the cl100k proxy's undercount of "
                             f"WordPiece). Context tail is dropped first, then "
                             f"the embedded body copy; stored text is never "
                             f"modified.")
    parser.add_argument("--lexicon", default="",
                        help="Optional JSON lexicon for entity tagging "
                             "({concept_id: [surface forms]}).")
    parser.add_argument("--global-dedup", action="store_true",
                        help="Deduplicate exact-duplicate chunks across files "
                             "(deterministic sorted post-pass over the written parquets).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-chunk files whose outputs already exist.")
    parser.add_argument("--file", default="",
                        help="TEST MODE: inspect a single document instead of a "
                             "directory. Accepts a .md/.xml path or a bare PMC id "
                             "(e.g. PMC10327125.4); XML is converted first. Prints "
                             "every chunk (tables in detail) so you can eyeball "
                             "single values; parquet written to .single_test/chunks/.")
    args = parser.parse_args(argv)

    if args.file:
        return _inspect_single(args)
    if not args.input.strip():
        parser.error("provide --input <md dir/file> or --file <pmc id/path>")

    input_path = Path(args.input)
    files = _discover_files(input_path)
    if not files:
        print(f"No Markdown files found under {input_path}", file=sys.stderr)
        return 1

    chunks_out = Path(args.chunks_out)
    units_out = Path(args.units_out)
    tagger = LexiconTagger(_load_lexicon(Path(args.lexicon))) if args.lexicon else None
    hard_max = args.hard_max_tokens or 2 * args.max_tokens

    if 0 < args.embedding_max_tokens < args.max_tokens:
        print("warning: --embedding-max-tokens < --max-tokens; some prose "
              "embeddings may be truncated — consider lowering --max-tokens",
              file=sys.stderr)
    if args.embedding_max_tokens > TARGET_ENCODER_WINDOW:
        print(
            f"warning: embedding budget {args.embedding_max_tokens} exceeds "
            f"the documented MedCPT {TARGET_ENCODER_WINDOW}-token window; "
            f"the downstream encoder must still use truncation=True.",
            file=sys.stderr,
        )

    print(f"config: prose_strategy={args.prose_strategy} max_tokens={args.max_tokens} "
          f"hard_max_tokens={hard_max} overlap={args.overlap} "
          f"min_paragraph_tokens={args.min_paragraph_tokens} "
          f"embedding_max_tokens={args.embedding_max_tokens} "
          f"tokenizer={_TOKENIZER_KIND} "
          f"(sized for a {TARGET_ENCODER_WINDOW}-token encoder; "
          f"encoding is a separate step)")

    converted = skipped = failed = 0
    from tqdm import tqdm

    def run_one(md_path: Path) -> Tuple[str, str]:
        """Returns (status, message); status in ok|skipped|failed."""
        try:
            text = md_path.read_text(encoding="utf-8")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            stem = md_path.stem
            chunk_path = chunks_out / f"{stem}.parquet"
            meta_path = chunks_out / f"{stem}.meta.json"
            units_path = units_out / f"{stem}.parquet"
            if chunk_path.exists() and meta_path.exists() and not args.overwrite:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                if old.get("md_sha256") == digest:
                    # skip only if the units artifact is consistent too
                    # (guards against an interrupted previous run that wrote
                    # chunks but died before writing units)
                    if int(old.get("units", 0)) == 0 or units_path.exists():
                        return "skipped", ""
            # fresh chunker per file: the class is intentionally
            # single-document stateful, so workers never share counters.
            # dedup_cache is left per-document here; cross-file dedup is the
            # deterministic post-pass below (--global-dedup), never the pool.
            chunker = MDChunker(max_tokens=args.max_tokens,
                                hard_max_tokens=hard_max,
                                overlap=args.overlap,
                                min_paragraph_tokens=args.min_paragraph_tokens,
                                embedding_max_tokens=args.embedding_max_tokens,
                                prose_strategy=args.prose_strategy,
                                tagger=tagger)
            chunks, units, report = chunker.chunk_md(text, doc_id=stem)
            if not chunks:
                return "failed", f"no chunks produced: {md_path}"
            _chunks_to_df(chunks).to_parquet(chunk_path, index=False)
            if units:
                _units_to_df(units).to_parquet(units_path, index=False)
            elif units_path.exists():
                units_path.unlink()  # stale units from a previous run
            meta_path.write_text(json.dumps({
                "md_sha256": digest,
                "chunks": len(chunks),
                "units": len(units),
                "report": report,
            }, ensure_ascii=False), encoding="utf-8")
            return "ok", f"{md_path.name}: {len(chunks)} chunks, {len(units)} units"
        except Exception as exc:  # noqa: BLE001 - per-file failures logged
            return "failed", f"FAILED {md_path}: {exc}"

    chunks_out.mkdir(parents=True, exist_ok=True)
    units_out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, p): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(files), desc="Chunking MD",
                        unit="article"):
            status, msg = fut.result()
            if status == "ok":
                converted += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(msg, file=sys.stderr)

    if args.global_dedup:
        stats = _global_dedup_pass(chunks_out)
        print(f"global dedup: suppressed {stats['suppressed']} duplicate chunk(s) "
              f"across {stats['documents']} document(s)")

    print(f"chunked={converted} skipped={skipped} failed={failed} "
          f"(prose_strategy={args.prose_strategy})")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
