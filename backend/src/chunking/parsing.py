"""Markdown parsing: front matter, fences, blocks, section tree (no LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from markdown_it import MarkdownIt


_MD = MarkdownIt("commonmark").enable("table")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")

_CITE_RE = re.compile(
    r"\[(\d{1,3}(?:\s*[–—-]\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*[–—-]\s*\d{1,3})?)*)\]"
)

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

def _attach_figure_context(blocks: List[RawBlock]) -> None:
    """Attach a figure's caption and footnotes from the lines after the image.

    jats_to_md emits the image, then ONE bold caption line, then (when the
    JATS figure carries fn footnotes) one or more ITALIC continuation lines
    (e.g. "*BMPR2*, *ARRB2* or both genes were reduced..." or "*p* < 0.01;
    ^✭✭✭^"). The caption lands on RawBlock.caption; italic lines become
    RawBlock.footnotes so the figure chunk can carry them (with its link).
    """
    for idx, b in enumerate(blocks):
        if b.kind != "figure":
            continue
        if (idx + 1 < len(blocks) and blocks[idx + 1].kind == "paragraph"
                and blocks[idx + 1].bold):
            b.caption = blocks[idx + 1].text
            blocks[idx + 1].text = ""
        fn: List[Tuple[str, str]] = []
        # step past the caption line (bold, not italic), then collect every
        # following ITALIC paragraph as a figure footnote
        j = idx + 1
        if (j < len(blocks) and blocks[j].kind == "paragraph"
                and blocks[j].bold and not blocks[j].italic):
            j += 1
        while (j < len(blocks) and blocks[j].kind == "paragraph"
               and blocks[j].italic):
            t = blocks[j].text.strip()
            if t:
                fn.append(("", t))  # (marker, text) - same shape as table footnotes
                blocks[j].text = ""
            j += 1
        if fn:
            b.footnotes = fn

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
    rankable) but are never part of the chunk text itself.
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

