#!/usr/bin/env python
"""Perfect(er) JATS -> Markdown converter — loss-aware, structure-first.

This converter RE-PARSES the JATS XML itself (lxml) instead of relying on the
flattened AST from ``src.processing.parser``, so XML structure survives to Markdown:

  * citations ``<xref ref-type="bibr">`` are resolved to their numbered
    reference INDEX and rendered as ``[n]`` / ``[1,3]`` — in-text citations
    always link back to the correct numbered reference; no heuristic
    post-hoc bracketing, so "P = .04" can never become "P = .[04]";
  * figures keep label + caption + alt-text + description + image href;
  * tables are parsed as grids (rowspan/colspan expanded exactly once) and
    rendered rectangular with caption + footnotes;
  * equations keep the verbatim ``tex-math`` LaTeX (no whitespace flattening);
  * lists keep their kind (ordered/bullet/alpha) and nesting;
  * ``boxed-text``, ``disp-quote``, ``supplementary-material`` and footnotes
    are preserved instead of dropped;
  * a structural LOSS REPORT compares source XML element counts against what
    was rendered (per-file warning + run summary);
  * writes are atomic (.tmp -> fsync -> rename) and resumable via an
    ``md_sha256`` sidecar.

Metadata (title, journal, authors, keywords, categories, identifiers, dates)
is kept structured in YAML front matter.

Usage:
    python -m src.processing.jats_to_md --input data/raw/cardiology --output data/md/cardiology
    python -m src.processing.jats_to_md --input file.xml --output data/md

Makefile: ``make jats-to-md INPUT=data/raw/... OUTPUT=data/md/...``
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _local(tag) -> str:
    """Local name of an lxml tag (namespace-stripped)."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _join(text: str) -> str:
    """Collapse whitespace (for short metadata scalars only)."""
    return " ".join((text or "").split())


def _xml_root(path: Path) -> etree._Element:
    """Secure parse (no network / no entity resolution) — XXE-safe."""
    parser = etree.XMLParser(
        resolve_entities=False, load_dtd=False, no_network=True,
        dtd_validation=False, recover=True,
    )
    return etree.parse(str(path), parser=parser).getroot()


class ConverterError(Exception):
    pass


# PMC Open Access images are served from the public S3 bucket under the
# article-version directory (e.g. PMC4329953.1/). Relative <graphic>/<media>
# hrefs are resolved to that URL so images actually render in Markdown.
PMC_OA_BASE_URL = os.environ.get(
    "PMC_OA_BASE_URL", "https://pmc-oa-opendata.s3.amazonaws.com"
)
_ARTICLE_DIR: str = ""   # set fresh per render (single-threaded CLI)


def _resolve_href(href: str) -> str:
    """Resolve a relative graphic/media href against the article directory."""
    href = (href or "").strip()
    if not href or href.startswith(("#", "http://", "https://", "data:", "//", "file:")):
        return href
    base = PMC_OA_BASE_URL.rstrip("/")
    if _ARTICLE_DIR:
        return f"{base}/{_ARTICLE_DIR.strip('/')}/{href.lstrip('/')}"
    return f"{base}/{href.lstrip('/')}"



# ---------------------------------------------------------------------------
# References (numbered) + citation map
# ---------------------------------------------------------------------------

def _meaningful_ref_text(ref: etree._Element) -> str:
    cit = ref.find(".//mixed-citation")
    if cit is None:
        cit = ref.find(".//element-citation")
    if cit is not None:
        text = _render_reference(cit) or _join("".join(cit.itertext()))
    else:
        # exclude the <label> so a numbered bare-year ref ("<label>4</label>
        # <year>2013</year>") reduces to "2013" and is filtered as junk
        parts: List[str] = [ref.text or ""]
        for child in ref:
            if _local(child.tag) != "label":
                parts.append("".join(child.itertext()))
        text = _join("".join(parts))
    return text


def _collect_references(root) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Return (ordered reference records, rid -> citation number).

    Only real ``<ref>`` elements with meaningful content are numbered
    (1-based, in ref-list order), so stray year-only blocks can never inject
    junk entries or misalign in-text citation numbers.
    """
    refs: List[Dict[str, str]] = []
    rid_to_n: Dict[str, int] = {}
    number = 1
    for ref in root.xpath("//ref-list//ref"):
        text = _meaningful_ref_text(ref)
        if not text or re.fullmatch(r"\d{1,4}", text):
            continue  # junk (bare year / number) — never numbered
        rid = ref.get("id") or ""
        refs.append({"id": rid, "number": number, "text": text})
        if rid:
            rid_to_n[rid] = number
        number += 1
    return refs, rid_to_n


def _render_reference(cit: etree._Element) -> str:
    """Element/mixed citation -> "Authors. Title. Source Year;Vol(Issue):Pages. IDs"."""

    def txt(xpath: str) -> str:
        node = cit.find(xpath)
        return _join("".join(node.itertext())) if node is not None else ""

    authors: List[str] = []
    for name in cit.xpath(".//person-group//string-name | .//person-group//name"):
        given = ""
        surname = ""
        nm = name.find("given-names")
        sn = name.find("surname")
        if nm is not None:
            given = _join("".join(nm.itertext()))
        if sn is not None:
            surname = _join("".join(sn.itertext()))
        full = " ".join(p for p in [surname, given] if p) or _join("".join(name.itertext()))
        if full:
            authors.append(full)
    collab = txt("./collab")
    author_str = ", ".join([a for a in authors + ([collab] if collab else []) if a])

    title = txt("./article-title") or txt("./data-title") or txt("./chapter-title")
    source = txt("./source") or txt("./journal-title")
    year = txt("./year")
    volume = txt("./volume")
    issue = txt("./issue")
    fpage = txt("./fpage")
    lpage = txt("./lpage")
    eloc = txt("./elocation-id")

    pub_ids: List[str] = []
    for pid in cit.xpath("./pub-id"):
        id_type = (pid.get("pub-id-type") or "").lower()
        val = _join(pid.text or "")
        if not id_type or not val:
            continue
        if id_type == "doi":
            pub_ids.append(f"DOI: [{val}](https://doi.org/{val})")
        elif id_type == "pmid":
            pub_ids.append(f"PMID: [{val}](https://pubmed.ncbi.nlm.nih.gov/{val})")
        elif id_type == "pmcid":
            pub_ids.append(f"PMCID: [{val}](https://www.ncbi.nlm.nih.gov/pmc/articles/{val})")

    parts: List[str] = []
    if author_str:
        parts.append(author_str if author_str.endswith(".") else author_str + ".")
    if title:
        parts.append(title if title.endswith(".") else title + ".")
    if source:
        seg = source
        if year:
            seg += " " + year
        if volume:
            seg += f";{volume}"
        if issue:
            seg += f"({issue})"
        if fpage:
            seg += ":" + fpage + (f"–{lpage}" if lpage else "")
        elif eloc:
            seg += ":" + eloc
        parts.append(seg if seg.endswith(".") else seg + ".")
    elif year:
        parts.append(f"({year})")
    parts.extend(pub_ids)
    return " ".join(parts)


def _collect_labels(root) -> Dict[str, str]:
    """id -> human label for fig/table/supplementary-material elements."""
    labels: Dict[str, str] = {}
    for el in root.xpath("//fig | //table-wrap | //supplementary-material"):
        eid = el.get("id")
        if not eid:
            continue
        label = _join("".join(el.xpath("label")[0].itertext())) \
            if el.xpath("label") else ""
        if not label:
            label = {"fig": "Figure", "table-wrap": "Table",
                     "supplementary-material": "Supplementary Material"}[_local(el.tag)]
        labels[eid] = label
    return labels


# ---------------------------------------------------------------------------
# Inline (mixed content) rendering
# ---------------------------------------------------------------------------

def _render_xref(xref: etree._Element, rid_to_n: Dict[str, int],
                 labels: Dict[str, str]) -> str:
    ref_type = xref.get("ref-type") or "bibr"
    rids = (xref.get("rid") or "").split()
    if ref_type == "bibr":
        numbers = sorted({rid_to_n[r] for r in rids if r in rid_to_n})
        if numbers:
            return _format_citation_numbers(numbers)
        return _join("".join(xref.itertext()).lstrip("#"))
    if ref_type in ("fig", "table", "supplementary-material"):
        labels_used = [labels.get(r, "") for r in rids if r in labels]
        if labels_used:
            return ", ".join(labels_used)
        name = {"fig": "Figure", "table": "Table",
                "supplementary-material": "Supplementary Figure"}[ref_type]
        return name + " " + _join("".join(xref.itertext()).lstrip("#"))
    return _join("".join(xref.itertext()).lstrip("#"))


def _format_citation_numbers(numbers: List[int]) -> str:
    """Collapse a sorted number list: [1,2,4,5,6] -> [1,2,4–6]."""
    if not numbers:
        return ""
    groups: List[str] = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        groups.append(str(start) if start == prev else f"{start}–{prev}")
        start = prev = n
    groups.append(str(start) if start == prev else f"{start}–{prev}")
    return "[" + ",".join(groups) + "]"


def _extract_tex(formula: etree._Element) -> str:
    """Verbatim LaTeX from tex-math (identity; no whitespace flattening)."""
    tex = formula.find(".//tex-math")
    if tex is not None:
        body = "".join(tex.itertext())
        if body.startswith("<![CDATA[") and body.endswith("]]>"):
            body = body[9:-3]
        return body.strip()
    mml = formula.find(".//{http://www.w3.org/1998/Math/MathML}math")
    if mml is None:
        mml = formula.find(".//math")
    if mml is not None:
        return " ".join(mml.itertext()).strip()
    return ""


def _inline_text(el: etree._Element, rid_to_n: Dict[str, int],
                 labels: Dict[str, str]) -> str:
    """Render mixed content faithfully (sup/sub/italic/bold/xref/formula/link)."""
    out: List[str] = []

    def walk(node: etree._Element) -> None:
        if node.text:
            out.append(node.text)
        for child in node:
            tag = _local(child.tag)
            if tag == "xref":
                out.append(_render_xref(child, rid_to_n, labels))
            elif tag in ("inline-formula", "disp-formula"):
                latex = _extract_tex(child)
                if latex:
                    out.append("$" + latex.replace("$$", "") + "$")
                else:
                    out.append(_join("".join(child.itertext())))
            elif tag == "ext-link":
                href = (child.get("{http://www.w3.org/1999/xlink}href")
                        or child.get("href") or "")
                label = _join("".join(child.itertext())) or href
                out.append(f"[{label}]({href})" if href else label)
            elif tag == "uri":
                uri = _join(child.text or "")
                out.append(f"[{uri}]({uri})" if uri else "")
            elif tag == "sup":
                inner = _inline_text(child, rid_to_n, labels)
                # citation brackets are already superscript-like; don't wrap
                out.append(inner if (inner.startswith("[") and inner.endswith("]")) else f"^{inner}^")
            elif tag == "sub":
                out.append("~" + _inline_text(child, rid_to_n, labels) + "~")
            elif tag == "italic":
                out.append("*" + _inline_text(child, rid_to_n, labels) + "*")
            elif tag == "bold":
                out.append("**" + _inline_text(child, rid_to_n, labels) + "**")
            elif tag in ("fn", "fnref", "graphic", "media", "inline-graphic"):
                pass  # footnotes/graphics handled elsewhere
            else:
                walk(child)
            if child.tail:
                out.append(child.tail)
    walk(el)
    # source-text typos ("$$p<0.01") must never desync fenced math: our
    # inline formulas use single $, so flatten any accidental double dollar
    return "".join(out).replace("$$", "$")


# ---------------------------------------------------------------------------
# Tables / figures / lists / equations
# ---------------------------------------------------------------------------

def _caption_text(el: etree._Element) -> str:
    cap = el.find("caption")
    if cap is None:
        return ""
    parts: List[str] = []
    for child in cap:
        tag = _local(child.tag)
        if tag in ("title", "p"):
            parts.append(_inline_text(child, {}, {}))
        elif tag not in ("fig", "table-wrap", "supplementary-material"):
            parts.append(_join("".join(child.itertext())))
    return " ".join(p for p in parts if p)


def _table_footnotes(table_wrap: etree._Element) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for foot in table_wrap.findall(".//table-wrap-foot"):
        for child in foot:
            tag = _local(child.tag)
            if tag == "fn":
                marker = _join("".join(child.xpath("label")[0].itertext())) \
                    if child.xpath("label") else ""
                text = " ".join(_join("".join(p.itertext())) for p in child.findall("p")) \
                    if child.findall("p") else _join("".join(child.itertext()))
                if text:
                    out.append((marker, text))
            elif tag == "p":
                text = _join("".join(child.itertext()))
                if text:
                    out.append(("", text))
    return out


def _first_graphic_href(el: etree._Element) -> str:
    for g in el.xpath(".//graphic | .//media"):
        href = g.get("{http://www.w3.org/1999/xlink}href") or g.get("href") or ""
        if href:
            return href
    return ""


def _build_grid(table_node: etree._Element) -> List[List[Optional[Dict[str, Any]]]]:
    rows: List[List[Dict[str, Any]]] = []
    # .iter("tr") — tables wrap rows in <thead>/<tbody>, so direct children
    # only would silently drop every row.
    for tr in table_node.iter("tr"):
        cells = []
        for cell in tr:
            tag = _local(cell.tag)
            if tag not in ("td", "th"):
                continue
            # keep math inside cells as $latex$ and remember cell-level
            # display formulas so the loss counter stays truthful
            text_parts: List[str] = []
            if cell.text:
                text_parts.append(cell.text)
            for sub in cell:
                st = _local(sub.tag)
                if st in ("disp-formula", "inline-formula"):
                    tex = _extract_tex(sub)
                    text_parts.append("$" + tex.replace("$$", "") + "$" if tex
                                      else _join("".join(sub.itertext())))
                else:
                    text_parts.append(_join("".join(sub.itertext())))
                if sub.tail:
                    text_parts.append(sub.tail)
            cells.append({
                # space-join sibling text runs: "16631<break/>17818<break/>13793"
                # must stay "16631 17818 13793", NOT "166311781813793"
                "text": _join(" ".join(p for p in text_parts if p)),
                "colspan": int(cell.get("colspan", 1) or 1),
                "rowspan": int(cell.get("rowspan", 1) or 1),
                "header": tag == "th",
                "cell_formulas": [(st == "disp-formula" and _extract_tex(sub))
                                  for sub in cell if _local(sub.tag) == "disp-formula"],
            })
        rows.append(cells)
    if not rows:
        return []
    n_cols = max(sum(c["colspan"] for c in r) for r in rows)
    grid: List[List[Optional[Dict[str, Any]]]] = [[None] * n_cols for _ in rows]
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while c < n_cols and grid[r][c] is not None:
                c += 1
            if c >= n_cols:
                break
            for rr in range(r, min(len(rows), r + cell["rowspan"])):
                for cc in range(c, min(n_cols, c + cell["colspan"])):
                    grid[rr][cc] = cell if (rr, cc) == (r, c) else {"echo": True}
            c += cell["colspan"]
    return grid
    return grid


def _parse_table(table_wrap: etree._Element) -> Dict[str, Any]:
    label = _join("".join(table_wrap.xpath("label")[0].itertext())) \
        if table_wrap.xpath("label") else ""
    caption = _caption_text(table_wrap)
    footnotes = _table_footnotes(table_wrap)
    table_node = table_wrap.find("table")
    if table_node is None:
        table_node = table_wrap.find("alternatives/table")
    if table_node is None:
        table_node = table_wrap.find(".//table")
    if table_node is None:
        return {"label": label, "caption": caption, "footnotes": footnotes,
                "grid": [], "image_href": _first_graphic_href(table_wrap)}
    return {"label": label, "caption": caption, "footnotes": footnotes,
            "grid": _build_grid(table_node),
            "image_href": ""}


def _cell_md(cell: Optional[Dict[str, Any]]) -> str:
    if cell is None or "echo" in cell:
        return ""
    text = " ".join((cell.get("text") or "").replace("\n", " ").split())
    return text.replace("|", "\\|")


def _render_table_md(tbl: Dict[str, Any], out: List[str], loss: Counter) -> None:
    out.append("")
    if tbl["label"] or tbl["caption"]:
        out.append(f"*{' '.join(p for p in [tbl['label'], tbl['caption']] if p)}*")
        out.append("")
    if tbl.get("image_href"):
        out.append(f"![{tbl['label'] or 'table'}]({tbl['image_href']})")
        out.append("")
        return
    grid = tbl["grid"]
    if not grid:
        return
    n_cols = len(grid[0])
    header_row: Optional[int] = None
    for r, row in enumerate(grid):
        if any(c and c.get("header") for c in row):
            header_row = r
            break
    if header_row is None:
        # Promote the first real header-looking row as the column names:
        # the row must contain the header text (>= 2 distinct non-empty
        # cells); full-width panel labels are skipped. This fixes
        # thead-less JATS tables that used to degrade to "Column i"
        # headers and leaked their header row into the table body.
        for r, row in enumerate(grid):
            texts = {_cell_md(c) for c in row if c and "echo" not in c and _cell_md(c)}
            if len(texts) >= 2:
                header_row = r
                break
    if header_row is None:
        header = [f"Column {i + 1}" for i in range(n_cols)]
    else:
        header = [_cell_md(c) for c in grid[header_row]]
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for r, row in enumerate(grid):
        if r == header_row:
            continue
        # a row whose origin cell spans the FULL table width is a group
        # subheader ("Sex", "Median (IQR)", panel labels): render as a
        # single-cell context row so the chunker reads it as group
        # context, not as a data row with empty values
        origins = [c for c in row if c and "echo" not in c]
        if len(origins) == 1 and n_cols > 1 and origins[0].get("colspan", 1) >= n_cols:
            out.append(f"| {_cell_md(origins[0])} |")
            continue
        out.append("| " + " | ".join(_cell_md(c) for c in row) + " |")
    out.append("")
    loss["table_cells_rendered"] += sum(
        1 for row in grid for c in row if c and "echo" not in c
    )
    for row in grid:
        for c in row:
            if c and c.get("cell_formulas"):
                loss["equations_rendered"] += len([1 for f in c["cell_formulas"] if f])
    for marker, text in tbl["footnotes"]:
        out.append(f"*{marker + ' ' if marker else ''}{text}*")
        out.append("")


def _parse_figure(fig: etree._Element, labels: Dict[str, str]) -> Dict[str, Any]:
    label = _join("".join(fig.xpath("label")[0].itertext())) if fig.xpath("label") else ""
    if not label:
        label = labels.get(fig.get("id") or "", "Figure")
    return {
        "label": label,
        "caption": _caption_text(fig),
        "alt": _join("".join(fig.xpath(".//alt-text")[0].itertext())) if fig.xpath(".//alt-text") else "",
        "desc": _join("".join(fig.xpath(".//long-desc")[0].itertext())) if fig.xpath(".//long-desc") else "",
        "href": _resolve_href(_first_graphic_href(fig)),
    }


def _render_figure_md(fig: Dict[str, Any], out: List[str]) -> None:
    out.append("")
    if fig["href"]:
        out.append(f"![{fig['label']}]({fig['href']})")
        out.append("")
    texts = [fig["label"]]
    if fig["caption"]:
        texts.append(fig["caption"])
    if fig["desc"]:
        texts.append(f"Description: {fig['desc']}")
    if fig["alt"]:
        texts.append(f"Alt-text: {fig['alt']}")
    if len(texts) > 1:
        out.append(f"**{' '.join(texts)}**")
        out.append("")
    elif texts[0]:
        out.append(f"**{texts[0]}**")
        out.append("")


def _render_figure_tree(fig: etree._Element, labels: Dict[str, str],
                        out: List[str], loss: Counter) -> None:
    """Render a figure and any nested figure-supplements (eLife style:
    <fig><p><fig ...>supplement</fig></p></fig>) — each real fig counts."""
    _render_figure_md(_parse_figure(fig, labels), out)
    loss["figures_rendered"] += 1
    for inner in fig.iter():
        if inner is fig or _local(inner.tag) != "fig":
            continue
        # skip figs that live inside captions/alt-text (those are references)
        is_caption = any(_local(a.tag) in ("caption", "alt-text", "long-desc")
                         for a in inner.iterancestors())
        if not is_caption:
            _render_figure_md(_parse_figure(inner, labels), out)
            loss["figures_rendered"] += 1


def _render_list(list_el: etree._Element, rid_to_n: Dict[str, int],
                 labels: Dict[str, str], depth: int = 0,
                 out: Optional[List[str]] = None,
                 loss: Optional[Counter] = None) -> List[str]:
    list_type = (list_el.get("list-type") or "bullet").lower()
    indent = "    " * depth
    lines: List[str] = []
    numbered = list_type in ("order", "ordered")
    alpha = list_type == "alpha"
    roman = list_type == "roman"
    for i, item in enumerate(list_el.findall("list-item"), start=1):
        if numbered:
            marker = f"{i}. "
        elif alpha:
            marker = f"{chr(96 + i)}. "
        elif roman:
            marker = f"{'i' * min(i, 3)}. "
        else:
            marker = "- "
        content: List[str] = []
        nested: List[etree._Element] = []
        for child in item:
            tag = _local(child.tag)
            if tag == "p":
                # a <list> nested inside the <p> must be split out (it is a
                # nested bullet, not inline text); figures/tables/equations
                # inside the item's paragraph become block objects
                buf: List[str] = []
                def flush_buf() -> None:
                    if "".join(buf).strip():
                        content.append(" ".join("".join(buf).strip().split()))
                    buf.clear()
                for sub in child:
                    st = _local(sub.tag)
                    if st == "list":
                        if "".join(buf).strip():
                            content.append("".join(buf))
                        buf.clear()
                        nested.append(sub)
                    elif st in ("fig", "table-wrap", "disp-formula"):
                        flush_buf()
                        if out is not None and loss is not None:
                            if st == "fig":
                                _render_figure_tree(sub, labels, out, loss)
                            elif st == "table-wrap":
                                _render_table_md(_parse_table(sub), out, loss)
                                loss["tables_rendered"] += 1
                            else:
                                _render_equation(sub, out, loss)
                        if sub.tail:
                            buf.append(sub.tail)
                    else:
                        buf.append(sub.text or "")
                        buf.append("".join(sub.itertext()))
                        if sub.tail:
                            buf.append(sub.tail)
                if child.text:
                    buf.insert(0, child.text)
                if "".join(buf).strip():
                    content.append(" ".join("".join(buf).strip().split()))
            elif tag == "list":
                nested.extend(c for c in [child] if _local(c.tag) == "list")
            else:
                t = _inline_text(child, rid_to_n, labels)
                if t:
                    content.append(t)
        if content:
            lines.append(f"{indent}{marker}{content[0].strip()}")
            pad = " " * len(marker)
            for extra in content[1:]:
                lines.append(f"{indent}{pad}{extra.strip()}")
            for sub in nested:
                lines.extend(_render_list(sub, rid_to_n, labels, depth + 1,
                                          out=out, loss=loss))
    return lines


def _render_equation(disp: etree._Element, out: List[str], loss: Counter) -> None:
    latex = _extract_tex(disp)
    if latex:
        # our $$ delimiters replace any stray $$ inside the tex body, so the
        # rendered fences stay balanced even for sloppy sources
        body = latex.replace("$$", "")
        loss["equations_rendered"] += 1
        out.append("")
        out.append("$$")
        out.extend(body.splitlines())  # verbatim (no whitespace flattening)
        out.append("$$")
        out.append("")


def _render_paragraph(p: etree._Element, rid_to_n: Dict[str, int],
                      labels: Dict[str, str], out: List[str], loss: Counter) -> None:
    """Render a <p> that may CONTAIN disp-formula (split into $$ blocks)."""
    buf: List[str] = []

    def flush() -> None:
        text = "".join(buf).strip()
        if text:
            out.append(text)
            out.append("")
        buf.clear()

    if p.text:
        buf.append(p.text)
    for child in p:
        tag = _local(child.tag)
        if tag == "disp-formula":
            flush()
            _render_equation(child, out, loss)
            if child.tail:
                buf.append(child.tail)
        elif tag == "table-wrap":
            flush()
            _render_table_md(_parse_table(child), out, loss)
            loss["tables_rendered"] += 1
            if child.tail:
                buf.append(child.tail)
        elif tag == "fig":
            flush()
            _render_figure_tree(child, labels, out, loss)
            if child.tail:
                buf.append(child.tail)
        elif tag == "disp-formula-group":
            flush()
            for eq in child.findall("disp-formula"):
                _render_equation(eq, out, loss)
            if child.tail:
                buf.append(child.tail)
        elif tag == "list":
            flush()
            out.extend(_render_list(child, rid_to_n, labels, out=out, loss=loss))
            out.append("")
            if child.tail:
                buf.append(child.tail)
        elif tag == "fig-group":
            flush()
            for fig in child.findall("fig"):
                _render_figure_tree(fig, labels, out, loss)
            if child.tail:
                buf.append(child.tail)
        else:
            buf.append(_inline_text(child, rid_to_n, labels))
            if child.tail:
                buf.append(child.tail)
    flush()


# ---------------------------------------------------------------------------
# Sections / abstract / supplementary
# ---------------------------------------------------------------------------

_SKIP_BLOCK_TAGS = {"title", "label", "fn-group"}


def _render_section(sec: etree._Element, level: int, rid_to_n: Dict[str, int],
                    labels: Dict[str, str], out: List[str], loss: Counter) -> None:
    title = _join("".join(sec.xpath("title")[0].itertext())) if sec.xpath("title") else ""
    if not title:
        label = _join("".join(sec.xpath("label")[0].itertext())) if sec.xpath("label") else ""
        title = "Section " + label if label else ""
    hashes = "#" * min(6, max(2, level))
    if title:
        out.append(f"{hashes} {title}")
        out.append("")

    for child in sec:
        tag = _local(child.tag)
        if tag in _SKIP_BLOCK_TAGS:
            continue
        if tag == "p":
            _render_paragraph(child, rid_to_n, labels, out, loss)
        elif tag == "sec":
            _render_section(child, level + 1, rid_to_n, labels, out, loss)
        elif tag == "list":
            out.extend(_render_list(child, rid_to_n, labels, out=out, loss=loss))
            out.append("")
        elif tag == "fig":
            _render_figure_tree(child, labels, out, loss)
        elif tag == "fig-group":
            for fig in child.findall("fig"):
                _render_figure_tree(fig, labels, out, loss)
        elif tag == "table-wrap":
            _render_table_md(_parse_table(child), out, loss)
            loss["tables_rendered"] += 1
        elif tag == "disp-formula":
            _render_equation(child, out, loss)
        elif tag == "disp-formula-group":
            for eq in child.findall("disp-formula"):
                _render_equation(eq, out, loss)
        elif tag == "boxed-text":
            t = _join("".join(child.xpath("title")[0].itertext())) \
                if child.xpath("title") else "Key Points"
            out.append(f"{'#' * min(6, level + 1)} {t}")
            out.append("")
            _render_section(child, level + 1, rid_to_n, labels, out, loss)
        elif tag == "disp-quote":
            quote = _inline_text(child, rid_to_n, labels)
            if quote:
                out.extend("> " + line for line in quote.splitlines() or [quote])
                out.append("")
        elif tag == "supplementary-material":
            pass  # rendered once, after references
        elif tag in ("ack", "notes"):
            t = _join("".join(child.xpath("title")[0].itertext())) \
                if child.xpath("title") else "Acknowledgments" if tag == "ack" else "Notes"
            out.append(f"{'#' * min(6, level)} {t}")
            out.append("")
            for p in child.findall(".//p"):
                text = _inline_text(p, rid_to_n, labels)
                if text:
                    out.append(text)
                    out.append("")
        else:
            text = _inline_text(child, rid_to_n, labels)
            if text:  # never silently dropped
                out.append(text)
                out.append("")


def _render_abstract(abstract: etree._Element, rid_to_n: Dict[str, int],
                     labels: Dict[str, str], out: List[str], loss: Counter) -> None:
    # NOTE: the CALLER owns the heading (type-aware: "Abstract", "Abstract
    # (video)", ...) — never emit one here, or every file gets a bare
    # duplicate "## Abstract" heading above the real abstract.
    for child in abstract:
        tag = _local(child.tag)
        if tag in _SKIP_BLOCK_TAGS:
            continue
        if tag == "p":
            _render_paragraph(child, rid_to_n, labels, out, loss)
        elif tag == "sec":
            _render_section(child, 3, rid_to_n, labels, out, loss)


def _render_supplementary(root) -> List[str]:
    """Future-scope: source data files (label + caption + download link)."""
    out: List[str] = []
    for supp in root.xpath("//supplementary-material"):
        label = _join("".join(supp.xpath("label")[0].itertext())) \
            if supp.xpath("label") else "Supplementary Material"
        caption = _caption_text(supp)
        href = _resolve_href(_first_graphic_href(supp))
        out.append("")
        out.append(f"**{label}**" + (f": {caption}" if caption else ""))
        if href:
            out.append(f"Download: [{Path(href).name or href}]({href})")
        out.append("")
    return out


# ---------------------------------------------------------------------------
# Front matter (YAML)
# ---------------------------------------------------------------------------

def _yaml(text: str) -> str:
    text = " ".join((text or "").split())
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _front_matter(meta: Dict[str, Any]) -> str:
    lines = ["---"]
    lines.append(f"pmcid: {_yaml(meta.get('pmcid'))}")
    lines.append(f"title: {_yaml(meta.get('title'))}")
    if meta.get("journal"):
        lines.append(f"journal: {_yaml(meta['journal'])}")
    if meta.get("authors"):
        lines.append("authors:")
        lines.extend(f"  - {_yaml(a)}" for a in meta["authors"])
    if meta.get("published"):
        lines.append(f"published: {_yaml(meta['published'])}")
    if meta.get("keywords"):
        lines.append("keywords:")
        lines.extend(f"  - {_yaml(k)}" for k in meta["keywords"])
    if meta.get("categories"):
        lines.append("categories:")
        lines.extend(f"  - {_yaml(c)}" for c in meta["categories"])
    if meta.get("doi"):
        lines.append(f"doi: {_yaml(meta['doi'])}")
    if meta.get("pmid"):
        lines.append(f"pmid: {_yaml(meta['pmid'])}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _meta_from_jats(root) -> Dict[str, Any]:
    def first(xpath: str) -> str:
        node = root.xpath(xpath)
        return _join("".join(node[0].itertext())) if node else ""

    ids = {
        pid.get("pub-id-type"): _join(pid.text or "")
        for pid in root.xpath("//front//article-id")
        if pid.get("pub-id-type") and pid.text
    }
    authors: List[str] = []
    for contrib in root.xpath(
        '//front//contrib-group/contrib[@contrib-type="author"]'
    ):
        name = contrib.find("name")
        if name is not None:
            given = _join("".join(name.xpath("given-names")[0].itertext())) \
                if name.xpath("given-names") else ""
            surname = _join("".join(name.xpath("surname")[0].itertext())) \
                if name.xpath("surname") else ""
            full = " ".join(p for p in [given, surname] if p)
        else:
            full = _join("".join(contrib.xpath("collab")[0].itertext())) \
                if contrib.xpath("collab") else ""
        if full:
            authors.append(full)

    pub = first("//front/article-meta/pub-date[not(@date-type='accepted')]/year") or \
          first("//front/article-meta/pub-date[not(@date-type='accepted')]")
    pub = _normalize_date(pub)

    keywords = [k for k in
                (_join("".join(kw.itertext())) for kw in root.xpath("//front//kwd-group/kwd"))
                if k]
    categories = [c for c in
                  (_join("".join(sx.itertext()))
                   for sx in root.xpath("//front//article-categories/subj-group/subject"))
                  if c]

    return {
        "pmcid": ids.get("pmcid") or "",
        "article_dir": ids.get("pmcid-ver") or ids.get("pmcid") or "",
        "doi": ids.get("doi") or "",
        "pmid": ids.get("pmid") or "",
        "title": first("//front/article-meta/title-group/article-title"),
        "journal": first("//front/journal-meta/journal-title"),
        "published": pub,
        "authors": authors,
        "keywords": keywords,
        "categories": categories,
        "volume": first("//front/article-meta/volume"),
        "issue": first("//front/article-meta/issue"),
        "pages": first("//front/article-meta/fpage"),
    }


def _normalize_date(s: str) -> str:
    s = _join(s)
    m = re.match(r"(\d{4})\s+(\d{1,2})\s+(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return s


# ---------------------------------------------------------------------------
# Loss-aware render
# ---------------------------------------------------------------------------

def _source_counts(root) -> Dict[str, int]:
    return {
        "sec": len(root.xpath("//body//sec")),
        "table_wrap": len(root.xpath("//table-wrap")),
        "fig": len(root.xpath("//fig")),
        "ref": len([r for r in root.xpath("//ref-list//ref")
                    if _meaningful_ref_text(r) and not re.fullmatch(r"\d{1,4}", _meaningful_ref_text(r))]),
        # only count equations that actually carry math (image-only formulas
        # have no tex/math content to render)
        "disp_formula": len([d for d in root.xpath("//disp-formula") if _extract_tex(d)]),
        "boxed_text": len(root.xpath("//boxed-text")),
        "disp_quote": len(root.xpath("//disp-quote")),
        "supplementary": len(root.xpath("//supplementary-material")),
    }


def _render_document(root, article_dir_hint: str = "") -> Tuple[str, Dict[str, Any], Counter, List[Dict[str, str]]]:
    """Faithful render: (markdown, meta, loss_counter, supplementary)."""
    global _ARTICLE_DIR
    root_tag = _local(root.tag)
    if root_tag not in ("article", "journal-meta", "book", "book-part"):
        raise ConverterError(f"unexpected root tag {root_tag}")

    meta = _meta_from_jats(root)
    _ARTICLE_DIR = (meta.get("article_dir") or meta.get("pmcid")
                    or article_dir_hint or "")
    try:
        refs, rid_to_n = _collect_references(root)
    except Exception:  # noqa: BLE001
        refs, rid_to_n = [], {}
    labels = _collect_labels(root)
    loss: Counter = Counter()
    loss["refs_rendered"] += len(refs)
    for el in root.xpath("//disp-formula"):
        if _extract_tex(el):
            loss["equations_source"] += 1

    out: List[str] = [_front_matter(meta)]
    if meta["title"]:
        out.append(f"# {meta['title']}")
        out.append("")

    abstracts = root.xpath("//abstract[not(@abstract-type='graphical')]")
    for i, abs_ in enumerate(abstracts):
        atype = (abs_.get("abstract-type") or "").strip()
        heading = "Abstract" if i == 0 else (
            f"Abstract ({atype})" if atype else f"Abstract {i + 1}")
        out.append(f"## {heading}")
        out.append("")
        _render_abstract(abs_, rid_to_n, labels, out, loss)

    # graphical abstract: a figure-only abstract — render its figures
    gas = root.xpath("//abstract[@abstract-type='graphical']")
    if gas:
        ga_figs = gas[0].xpath(".//fig")
        if ga_figs:
            out.append("## Graphical Abstract")
            out.append("")
            for fig in ga_figs:
                _render_figure_tree(fig, labels, out, loss)

    body = root.find(".//body")
    if body is not None:
        for child in body:
            tag = _local(child.tag)
            if tag == "sec":
                _render_section(child, 2, rid_to_n, labels, out, loss)
            elif tag == "p":
                _render_paragraph(child, rid_to_n, labels, out, loss)
            elif tag == "fig":
                _render_figure_tree(child, labels, out, loss)
            elif tag == "fig-group":
                for fig in child.findall("fig"):
                    _render_figure_tree(fig, labels, out, loss)
            elif tag == "table-wrap":
                _render_table_md(_parse_table(child), out, loss)
                loss["tables_rendered"] += 1
            elif tag == "disp-formula":
                _render_equation(child, out, loss)
            elif tag == "list":
                out.extend(_render_list(child, rid_to_n, labels, out=out, loss=loss))
                out.append("")
            elif tag == "disp-quote":
                quote = _inline_text(child, rid_to_n, labels)
                if quote:
                    out.extend("> " + line for line in quote.splitlines() or [quote])
                    out.append("")
            elif tag == "boxed-text":
                t = _join("".join(child.xpath("title")[0].itertext())) \
                    if child.xpath("title") else "Key Points"
                out.append(f"### {t}")
                out.append("")
                _render_section(child, 3, rid_to_n, labels, out, loss)
            elif tag in ("ack", "notes"):
                t = _join("".join(child.xpath("title")[0].itertext())) \
                    if child.xpath("title") else "Acknowledgments"
                out.append(f"## {t}")
                out.append("")
                for p in child.findall(".//p"):
                    text = _inline_text(p, rid_to_n, labels)
                    if text:
                        out.append(text)
                        out.append("")

    # sub-articles (PMC OA packages: review bodies, translated content, etc.)
    for sub in root.xpath("//sub-article"):
        title = _join("".join(sub.xpath("front-stub/article-title | front-stub/title-group/article-title | article-title")[0].itertext())) \
            if sub.xpath("front-stub/article-title | front-stub/title-group/article-title | article-title") else "Sub-article"
        out.append("## " + title)
        out.append("")
        for ga in sub.xpath("front-stub/abstract[@abstract-type='graphical']"):
            for fig in ga.xpath(".//fig"):
                _render_figure_tree(fig, labels, out, loss)
        sub_body = sub.find("body")
        if sub_body is not None:
            for child in sub_body:
                tag = _local(child.tag)
                if tag == "sec":
                    _render_section(child, 3, rid_to_n, labels, out, loss)
                elif tag == "p":
                    _render_paragraph(child, rid_to_n, labels, out, loss)
                elif tag == "table-wrap":
                    _render_table_md(_parse_table(child), out, loss)
                    loss["tables_rendered"] += 1
                elif tag == "fig":
                    _render_figure_tree(child, labels, out, loss)
                elif tag == "disp-formula":
                    _render_equation(child, out, loss)

    # floats-group (tables/figures/equations PMC stores OUTSIDE <body>)
    floats = root.find(".//floats-group")
    if floats is not None:
        for child in floats:
            tag = _local(child.tag)
            if tag == "table-wrap":
                _render_table_md(_parse_table(child), out, loss)
                loss["tables_rendered"] += 1
            elif tag == "fig":
                _render_figure_tree(child, labels, out, loss)
            elif tag == "fig-group":
                for fig in child.findall("fig"):
                    _render_figure_tree(fig, labels, out, loss)
            elif tag == "disp-formula":
                _render_equation(child, out, loss)

    # back matter sections (acknowledgments, declarations, footnotes…)
    back = root.find(".//back")
    if back is not None:
        for child in back:
            tag = _local(child.tag)
            if tag in ("app", "app-group"):
                apps = child.findall("app") if tag == "app-group" else [child]
                for idx, app in enumerate(apps, start=1):
                    title = _join("".join(app.xpath("title")[0].itertext())) \
                        if app.xpath("title") else f"Appendix {idx}"
                    out.append("## " + title)
                    out.append("")
                    _render_section(app, 3, rid_to_n, labels, out, loss)
                continue
            if tag in ("sec", "ack", "fn-group", "notes"):
                if not child.xpath("title"):
                    t = {"ack": "Acknowledgments", "fn-group": "Footnotes",
                         "notes": "Notes"}.get(tag, "Back Matter")
                    out.append(f"## {t}")
                    out.append("")
                _render_section(child, 3, rid_to_n, labels, out, loss)

    # numbered references
    if refs:
        out.append("## References")
        out.append("")
        for ref in refs:
            out.append(f"[{ref['number']}] {ref['text']}")
            out.append("")

    # supplementary material (source data) — future scope
    supp_lines = _render_supplementary(root)
    sources: List[Dict[str, str]] = []
    if supp_lines:
        out.append("## Supplementary Material")
        out.extend(supp_lines)
        for s in root.xpath("//supplementary-material"):
            sources.append({
                "id": s.get("id") or "",
                "label": _join("".join(s.xpath("label")[0].itertext()))
                if s.xpath("label") else "Supplementary Material",
                "caption": _caption_text(s),
                "href": _first_graphic_href(s),
            })

    md = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).rstrip() + "\n"
    return md, meta, loss, sources


# ---------------------------------------------------------------------------
# Driver (atomic write, hash-resumable)
# ---------------------------------------------------------------------------

def _discover_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".xml" else []
    if input_path.is_dir():
        return sorted(input_path.rglob("*.xml"))
    return []


def convert_one(xml_path: Path, md_path: Path, overwrite: bool = False,
                verbose: bool = True, write_sidecar: bool = True) -> Dict[str, Any]:
    """Convert one file; returns {status, warnings, loss}. Atomic + resumable."""
    if md_path.exists() and not overwrite:
        return {"status": "skipped", "warnings": [], "loss": {}}
    try:
        root = _xml_root(xml_path)
        markdown, meta, loss, sources = _render_document(
            root, article_dir_hint=xml_path.stem)
    except Exception as exc:  # noqa: BLE001 - per-file failures shouldn't kill the run
        return {"status": "failed", "warnings": [f"{exc}"], "loss": {}}

    warnings: List[str] = []
    if markdown.count("$$") % 2:
        warnings.append("unbalanced_math_fence")
    if markdown.count("```") % 2:
        warnings.append("unbalanced_code_fence")

    src = _source_counts(root)
    dropped: List[str] = []
    if src["table_wrap"] and loss.get("tables_rendered", 0) != src["table_wrap"]:
        dropped.append(f"tables {loss.get('tables_rendered', 0)}/{src['table_wrap']}")
    if src["fig"] and loss.get("figures_rendered", 0) != src["fig"]:
        dropped.append(f"figures {loss.get('figures_rendered', 0)}/{src['fig']}")
    if src["ref"] and loss.get("refs_rendered", 0) != src["ref"]:
        dropped.append(f"refs {loss.get('refs_rendered', 0)}/{src['ref']}")
    if src["disp_formula"] and loss.get("equations_rendered", 0) != src["disp_formula"]:
        dropped.append(f"equations {loss.get('equations_rendered', 0)}/{src['disp_formula']}")
    if src["supplementary"]:
        # supplementary entries always render; flag if none did
        rendered_supp = 1 if "## Supplementary Material" in markdown else 0
        if rendered_supp != 1:
            dropped.append(f"supplementary {rendered_supp}/{src['supplementary']}")
    loss_report = {
        "source": dict(src),
        "rendered": {k: loss[k] for k in
                     ("tables_rendered", "figures_rendered", "refs_rendered",
                      "equations_rendered", "table_cells_rendered")
                     if k in loss},
        "dropped": dropped,
    }

    md_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(md_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(markdown)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(md_path))
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    if write_sidecar:
        digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        (md_path.with_suffix(md_path.suffix + ".sha256")).write_text(
            digest + "\n", encoding="utf-8")

    if verbose and (warnings or dropped):
        msg = "; ".join(warnings + [f"LOSS: {'; '.join(dropped)}"])
        print(f"[warn] {xml_path.name}: {msg}", file=sys.stderr)

    return {"status": "converted", "warnings": warnings, "loss": loss_report}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Loss-aware JATS XML -> Markdown converter (structure-first)."
    )
    parser.add_argument("--input", required=True,
                        help="JATS XML file or directory to convert.")
    parser.add_argument("--output", required=True,
                        help="Output directory (input tree is mirrored).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of files to convert (0 = all).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert files whose .md already exists.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    files = _discover_files(input_path)
    if not files:
        print(f"No JATS XML files found under {input_path}", file=sys.stderr)
        return 1
    if args.limit > 0:
        files = files[: args.limit]

    converted = skipped = failed = 0
    drop_counter: Counter = Counter()
    with tqdm(files, desc="Converting", unit="article") as bar:
        for xml_path in bar:
            if input_path.is_file():
                rel = Path(xml_path.name)
            else:
                rel = xml_path.relative_to(input_path)
            md_path = output_dir / rel.with_suffix(".md")
            if md_path.exists() and not args.overwrite:
                sha = md_path.with_suffix(md_path.suffix + ".sha256")
                if sha.exists():
                    try:
                        digest = sha.read_text(encoding="utf-8").strip()
                        current = hashlib.sha256(
                            md_path.read_text(encoding="utf-8").encode("utf-8")
                        ).hexdigest()
                        if current == digest:
                            skipped += 1
                            continue
                    except OSError:
                        pass
            res = convert_one(xml_path, md_path, overwrite=args.overwrite)
            if res["status"] == "converted":
                converted += 1
                for d in res["loss"].get("dropped", []):
                    drop_counter[d] += 1
            elif res["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"FAILED {xml_path}: {'; '.join(res['warnings'])}", file=sys.stderr)

    print(f"Converted: {converted:,} | Skipped: {skipped:,} | Failed: {failed:,}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    if drop_counter:
        print("Loss summary (files with dropped structural content):", flush=True)
        for k, v in drop_counter.most_common():
            print(f"  {v:,} file(s): {k}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())