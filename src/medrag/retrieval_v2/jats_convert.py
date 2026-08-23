"""JATS XML -> Markdown conversion command (V2.3).

Uses the INSTALLED jats package (jats 0.2.2): parse_jats_xml + convert_to_markdown
convert the ORIGINAL ARTICLE XML into Markdown that follows the real XML
structure (section titles become headings, tables/figures stay at their source
location, references are preserved).

    ORIGINAL XML  ->  (jats package)  ->  structured Markdown  ->  Folder

The Markdown folder is the input to the official PageIndex library
(pageindex.page_index_md.md_to_tree) for the paper-local navigation tree.

Usage:
    python -m medrag.retrieval_v2.jats_convert --paper PMC11743609
    python -m medrag.retrieval_v2.jats_convert --papers PMC11743609 PMC11092466
    python -m medrag.retrieval_v2.jats_convert --all
    python -m medrag.retrieval_v2.jats_convert --limit 50

Options:
    --xml-dir   input XML folder (default data/chunked)
    --md-dir    output Markdown folder (default index/pageindex_md)
    --raw       disable PageIndex-friendly normalization (keep jats output verbatim)
    --rebuild   overwrite existing .md files
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from medrag.retrieval_v2.xml_tree import locate_xml


# ---------------------------------------------------------------------------
# Floats-group supplement
# ---------------------------------------------------------------------------
# The installed jats converter renders inline tables/figures but DROPS
# floats-group content (JATS verdict: tables/figures detached at the end of the
# XML). To avoid information loss the converter re-adds them:
#   * figures are placed at their first in-text citation (an explicit
#     relationship: [Figure N](img.jpg) links in the body markdown),
#   * tables keep the XML floats-group order under a "Floats-group" section
#     (the JATS element name), i.e. actual XML order with low confidence -
#     never an invented "Figures and Tables" branch.

XLink = "{http://www.w3.org/1999/xlink}"
_FIG_MENTION_RE = re.compile(r"^.*?\[Figure[s]?\s+([0-9]+)")


def _insert_figures_at_mentions(md: str, fig_blocks: List[Tuple[str, str]]) -> str:
    """Insert figure blocks right after their first in-text citation line."""
    lines = md.split("\n")
    inserted: set = set()
    out: List[str] = []
    pending: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        out.append(line)
        m = _FIG_MENTION_RE.match(line)
        if m:
            num = m.group(1)
            for label, block in fig_blocks:
                if label in inserted:
                    continue
                label_num = re.sub(r"[^0-9]", "", label)
                if label_num == num:
                    pending.append((i + len(out), block))
                    inserted.add(label)
    # append pending blocks after their anchor lines
    if pending:
        anchors = sorted(pending, key=lambda x: x[0])
        # rebuild with insertion points
        out_lines = lines
    # simple approach: two-pass over lines inserting after match
    result: List[str] = []
    inserted_ids: set = set()
    for line in lines:
        result.append(line)
        m = _FIG_MENTION_RE.match(line)
        if m:
            num = m.group(1)
            for label, block in fig_blocks:
                if label in inserted_ids:
                    continue
                if re.sub(r"[^0-9]", "", label) == num:
                    result.append(block)
                    result.append("")
                    inserted_ids.add(label)
    for label, block in fig_blocks:
        if label not in inserted_ids:
            result.append(block)
            result.append("")
    return "\n".join(result).rstrip() + "\n"


def _table_wrap_to_html(tw) -> str:
    """Minimal JATS table-wrap -> HTML table (mirrors the jats converter's
    rendering so PageIndex treats the table as one node)."""
    tbl = tw.find(".//{*}table")
    if tbl is None:
        return ""
    rows_html: List[str] = ["<table>"]
    thead = tbl.find("{*}thead")
    if thead is not None:
        for tr in thead.findall("{*}tr"):
            cells = [f"<th>{''.join(c.itertext())}</th>" for c in tr.findall(".//{*}th")]
            if not cells:
                cells = [f"<td>{''.join(c.itertext())}</td>" for c in tr.findall(".//{*}td")]
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
    tbody = tbl.find("{*}tbody")
    if tbody is not None:
        for tr in tbody.findall("{*}tr"):
            cells = [f"<td>{''.join(c.itertext())}</td>" for c in tr.findall(".//{*}td")]
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
    rows_html.append("</table>")
    return "\n".join(rows_html)


def supplement_floats_group(xml_path: Path, md: str) -> str:
    """Re-add floats-group tables/figures that the jats converter dropped."""
    import lxml.etree as ET
    root = ET.parse(str(xml_path)).getroot()
    floats = root.find(".//{*}floats-group")
    if floats is None:
        return md
    fig_blocks: List[Tuple[str, str]] = []
    for fig in floats.findall("{*}fig"):
        label = (fig.findtext(".//{*}label") or "").strip()
        if not label:
            label = "Figure " + (fig.get("id") or "?")
        cap_el = fig.find(".//{*}caption")
        caption = " ".join(("".join((cap_el or fig).itertext())).split())
        img = fig.find(".//{*}graphic")
        img_src = ""
        if img is not None:
            img_src = img.get(XLink + "href") or img.get("href") or ""
        block = f"#### {label}\n\n![{label}]({img_src})"
        if caption:
            block += f"\n\n{label}: {caption}"
        fig_blocks.append((label, block.strip()))
    md = _insert_figures_at_mentions(md, fig_blocks)
    table_blocks: List[str] = []
    for tw_idx, tw in enumerate(floats.findall("{*}table-wrap") + floats.findall("{*}table-wrap-foot"), start=1):
        label = (tw.findtext(".//{*}label") or "").strip()
        if not label:
            label = f"Table {tw_idx}"
        caption = " ".join(("".join((tw.find(".//{*}caption") or tw).itertext())).split())
        html = _table_wrap_to_html(tw)
        block = f"#### {label}"
        if caption:
            block += f"\n\n{caption}"
        if html:
            block += f"\n\n{html}"
        table_blocks.append(block.strip())
    if table_blocks:
        md = md.rstrip() + "\n\n## Floats-group\n\n" + "\n\n".join(table_blocks) + "\n"
    return md


# ---------------------------------------------------------------------------
# PageIndex-friendly normalization
# ---------------------------------------------------------------------------
# The jats converter renders table labels / figure captions as **bold-only**
# lines; PageIndex's md_to_tree would read those as level-1 headings and hoist
# tables/figures out of their sections. Normalization rewrites them into real
# '####' headings so the tree keeps them at their source location.
_BOLD_TABLE_RE = re.compile(r"^\*\*(Table|Figure|Supplementary Table|Supplementary Figure)\b(.*?)\*\*\s*$")
# "**Figure 1:** caption text ..." - only the label is bold
_BOLD_CAPTION_RE = re.compile(
    r"^\*\*(Table|Figure|Supplementary Table|Supplementary Figure)\b\s*([0-9A-Za-z-]+):\s*(.*)$")


def normalize_pageindex_md(md: str) -> str:
    """Rewrite bold table/figure labels and captions into #### headings."""
    out_lines: List[str] = []
    for line in md.split("\n"):
        stripped = line.strip()
        m = _BOLD_CAPTION_RE.match(stripped)
        if m:
            kind = m.group(1).strip()
            label_num = m.group(2).strip()
            caption = m.group(3).strip()
            label = f"{kind} {label_num}".strip()
            out_lines.append(f"#### {label}")
            if caption:
                out_lines.append(caption)
            continue
        m = _BOLD_TABLE_RE.match(stripped)
        if m:
            label = (f"{m.group(1)} {m.group(2)}").strip()
            out_lines.append(f"#### {label}")
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert_paper_xml_to_md(paper_id: str, xml_path: Path, md_dir: Path, *,
                            no_refs: bool = False, normalize: bool = True) -> Dict[str, Any]:
    """Convert ONE paper's original XML to Markdown via the installed jats pkg."""
    from jats import convert_to_markdown, parse_jats_xml

    article = parse_jats_xml(xml_path, no_refs=no_refs)
    md = convert_to_markdown(article)
    md = supplement_floats_group(xml_path, md)   # floats-group tables/figures are dropped by the converter
    if normalize:
        md = normalize_pageindex_md(md)
    out_path = md_dir / f"{paper_id}.md"
    out_path.write_text(md, encoding="utf-8")
    return {
        "paper_id": paper_id,
        "xml": str(xml_path),
        "md": str(out_path),
        "chars": len(md),
        "n_headings": sum(1 for l in md.splitlines() if l.startswith("#")),
        "title": article.title or "",
        "n_tables": len(getattr(article, "tables", []) or []),
        "n_figures": len(getattr(article, "figures", []) or []),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert JATS XML to Markdown (PageIndex input)")
    parser.add_argument("--paper", default=None, help="single paper id")
    parser.add_argument("--papers", nargs="*", default=[], help="multiple paper ids")
    parser.add_argument("--limit", type=int, default=None, help="convert first N papers")
    parser.add_argument("--all", action="store_true", help="convert every paper")
    parser.add_argument("--xml-dir", type=Path, default=Path("data/chunked"))
    parser.add_argument("--md-dir", type=Path, default=Path("index/pageindex_md"))
    parser.add_argument("--no-refs", action="store_true", help="strip URL links from references")
    parser.add_argument("--raw", action="store_true", help="keep the jats output verbatim (no normalization)")
    parser.add_argument("--rebuild", action="store_true", help="overwrite existing .md files")
    args = parser.parse_args(argv)

    xml_dir = Path(args.xml_dir)
    md_dir = Path(args.md_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    if args.paper:
        papers = [args.paper]
    elif args.papers:
        papers = list(args.papers)
    elif args.all or args.limit is not None:
        papers = [p[:p.index(".")] for p in sorted(x.name for x in xml_dir.iterdir()
                                                   if x.name.startswith("PMC") and x.name.endswith(".xml"))]
        if args.limit is not None:
            papers = papers[: args.limit]
    else:
        parser.error("provide --paper, --papers, --limit, or --all")

    ok, failed, skipped = 0, 0, 0
    t0 = time.perf_counter()
    reports: List[Dict[str, Any]] = []
    for pid in papers:
        if pid.startswith("PMC"):
            pass
        xml = locate_xml(pid, xml_dir)
        if xml is None:
            print(f"  NO-XML {pid}")
            failed += 1
            continue
        out = md_dir / f"{pid}.md"
        if out.exists() and not args.rebuild:
            skipped += 1
            print(f"  exists {pid}")
            continue
        try:
            rep = convert_paper_xml_to_md(pid, xml, md_dir,
                                          no_refs=args.no_refs, normalize=not args.raw)
            reports.append(rep)
            ok += 1
            print(f"  ok     {pid}  md={rep['md']}  chars={rep['chars']}  headings={rep['n_headings']}  tables={rep['n_tables']} figs={rep['n_figures']}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAILED {pid}: {exc}")
    dt = time.perf_counter() - t0
    print(f"Done: ok={ok} skipped={skipped} failed={failed} in {dt:.1f}s")
    print(f"Markdown folder: {md_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
