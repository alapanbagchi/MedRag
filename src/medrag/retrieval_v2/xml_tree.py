"""XML-derived PageIndex tree builder (V2.2, task: rebuild from ORIGINAL XML).

Architecture decision (the task spec):
    ORIGINAL XML  ->  XML structural parser  ->  semantic document tree
                    ->  mapping onto existing chunk_ids

The ORIGINAL ARTICLE XML is the authoritative source for document structure;
the existing chunk index is ONLY the authoritative source for retrieval /
evidence identifiers. The previous chunk-derived tree (which invented
paragraph-first-sentence titles and a synthetic "Figures and Tables" branch)
is replaced here for tree generation.

Responsibilities are kept separate (spec part 30):
    XML        answers "what is the structure of the paper?"
    PageIndex  answers "where is the relevant structure?"
    chunks     answer "what exact evidence object corresponds to that location?"

The builder produces the SAME artifact schema as before (tree / node_map /
chunk_to_node) so the PageIndex navigation layer keeps working, extended with
node_type, breadcrumb, parent/child ids, location_confidence, mapping_status,
and lookup registries (section_index, table_index, figure_index,
node_to_parent, node_to_children).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lxml.etree as ET

# ---------------------------------------------------------------------------
# Node taxonomy (part 3) — no arbitrary node types invented from text.
# ---------------------------------------------------------------------------
DOCUMENT = "document"
SECTION = "section"
SUBSECTION = "subsection"
PARAGRAPH = "paragraph"
TABLE = "table"
TABLE_SUMMARY = "table_summary"
TABLE_HEADER = "table_header"
TABLE_ROW = "table_row"
TABLE_FOOTNOTE = "table_footnote"
FIGURE = "figure"
FIGURE_CAPTION = "figure_caption"
LIST = "list"
LIST_ITEM = "list_item"
REFERENCES = "references"
REFERENCE = "reference"

NODE_TYPES = (
    DOCUMENT, SECTION, SUBSECTION, PARAGRAPH, TABLE, TABLE_SUMMARY,
    TABLE_HEADER, TABLE_ROW, TABLE_FOOTNOTE, FIGURE, FIGURE_CAPTION,
    LIST, LIST_ITEM, REFERENCES, REFERENCE,
)

_NSURL = "http://www.w3.org/XML/1998/namespace"


def localname(tag) -> str:
    return ET.QName(tag).localname


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def normalize_text(value: Any) -> str:
    """Mirror the chunker's _clean_text so mapping comparisons are fair."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def element_text(el) -> str:
    """Full text content of an element (includes inline content)."""
    if el is None:
        return ""
    return normalize_text("".join(el.itertext()))


def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# XML discovery (part 1)
# ---------------------------------------------------------------------------
_XML_NAME_RE = re.compile(r"^(PMC\d+)\.\d+\.xml$")


def locate_xml(paper_id: str, xml_dir: Path) -> Optional[Path]:
    """Find the original article XML for a paper id.

    filename convention: {paper_id}.{version}.xml  (e.g. PMC11092466.2.xml)
    """
    xml_dir = Path(xml_dir)
    direct = xml_dir / f"{paper_id}.xml"
    if direct.exists():
        return direct
    # versioned filename (PMC11092466.2.xml)
    pat = re.compile(r"^" + re.escape(paper_id) + r"\.\d+\.xml$")
    for f in xml_dir.iterdir():
        if f.is_file() and pat.match(f.name):
            return f
    return None


def xml_paper_id(root) -> str:
    """paper_id from the XML itself (article-id pub-id-type='pmcid')."""
    for aid in root.iter("{*}article-id"):
        if aid.get("pub-id-type") == "pmcid" and aid.text:
            return aid.text.strip()
    return ""


# ---------------------------------------------------------------------------
# Paper chunk inventory (from the existing corpus parquet)
# ---------------------------------------------------------------------------
def load_paper_chunks(paper_id: str, corpus_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all chunk metadata + text for ONE paper from the corpus parquet.

    The corpus is the authoritative source of EVIDENCE IDs; we never regenerate
    them (part 10 / 23).
    """
    import pandas as pd
    df = pd.read_parquet(
        str(corpus_path),
        columns=["id", "document_id", "chunk_type", "section", "subsection",
                 "table_id", "figure_id", "document_position", "text"],
    )
    sub = df[df["document_id"] == paper_id]
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        s = str(value)
        return "" if s.lower() == "nan" else s

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in sub.iterrows():
        cid = str(row["id"])
        out[cid] = {
            "chunk_type": str(row["chunk_type"] or "paragraph"),
            "section": _clean(row["section"]),
            "subsection": _clean(row["subsection"]),
            "table_id": _clean(row["table_id"]),
            "figure_id": _clean(row["figure_id"]),
            "position": int(row["document_position"]),
            "text": normalize_text(row["text"]),
        }
    return out


# ---------------------------------------------------------------------------
# Tree node construction
# ---------------------------------------------------------------------------
class XmlTreeBuilder:
    """Builds the semantic document tree from the original article XML."""

    def __init__(self, paper_id: str, xml_path: Path, corpus_path: Path,
                 xml_dir: Optional[Path] = None) -> None:
        self.paper_id = paper_id
        self.xml_path = Path(xml_path)
        self.xml_dir = Path(xml_dir) if xml_dir is not None else self.xml_path.parent
        self.corpus_path = Path(corpus_path)
        self.chunks: Dict[str, Dict[str, Any]] = load_paper_chunks(paper_id, corpus_path)
        self.root = ET.parse(str(self.xml_path)).getroot()

        self._counter = 0
        self.nodes: List[Dict[str, Any]] = []
        self.node_map: Dict[str, Dict[str, Any]] = {}
        self.chunk_to_node: Dict[str, str] = {}
        self.node_to_parent: Dict[str, str] = {}
        self.node_to_children: Dict[str, List[str]] = {}
        self.section_index: Dict[str, List[str]] = {}
        self.table_index: Dict[str, str] = {}
        self.figure_index: Dict[str, str] = {}
        self.mapping_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # node factory
    # ------------------------------------------------------------------
    def _new_node(self, node_type: str, title: Optional[str], text: str, level: int,
                  breadcrumb: List[str], *, xml_id: str = "",
                  location_confidence: str = "high", table_id: str = "",
                  figure_id: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        node_id = f"{self._counter:04d}"
        self._counter += 1
        node: Dict[str, Any] = {
            "node_id": node_id,
            "node_type": node_type,
            "title": title,
            "text": text,
            "level": level,
            "section": " > ".join([self.paper_id] + list(breadcrumb)),
            "breadcrumb": list(breadcrumb),
            "chunk_ids": [],
            "page_refs": None,
            "parent_node_id": None,
            "child_node_ids": [],
            "xml_id": xml_id,
            "location_confidence": location_confidence,
            "mapping_status": "unmapped",
            "table_id": table_id or None,
            "figure_id": figure_id or None,
        }
        if extra:
            node.update(extra)
        self.nodes.append(node)
        self.node_map[node_id] = node
        self.node_to_children[node_id] = []
        return node

    def _attach(self, parent: Dict[str, Any], child: Dict[str, Any]) -> None:
        child["parent_node_id"] = parent["node_id"]
        self.node_to_parent[child["node_id"]] = parent["node_id"]
        parent["child_node_ids"].append(child["node_id"])
        self.node_to_children[parent["node_id"]].append(child["node_id"])

    def _index_node(self, node: Dict[str, Any]) -> None:
        """Registry entries (part 22)."""
        nt = node["node_type"]
        if nt in (SECTION, SUBSECTION):
            key = (node["title"] or "").strip().lower()
            if key:
                self.section_index.setdefault(node["title"].strip(), [])
                if node["node_id"] not in self.section_index[node["title"].strip()]:
                    self.section_index[node["title"].strip()].append(node["node_id"])
        elif nt == TABLE:
            key = (node.get("table_id") or node.get("xml_id") or node["title"] or "").strip()
            if key and key not in self.table_index:
                self.table_index[key] = node["node_id"]
        elif nt == FIGURE:
            key = (node.get("figure_id") or node.get("xml_id") or node["title"] or "").strip()
            if key and key not in self.figure_index:
                self.figure_index[key] = node["node_id"]

    def _map_chunks(self, node: Dict[str, Any], chunk_ids: List[str],
                    status: str = "mapped", note: str = "") -> None:
        node["chunk_ids"] = list(dict.fromkeys(c for c in chunk_ids if c))
        node["mapping_status"] = status
        for cid in node["chunk_ids"]:
            if cid not in self.chunk_to_node:
                self.chunk_to_node[cid] = node["node_id"]
        if note:
            self.mapping_log.append({"node_id": node["node_id"], "node_type": node["node_type"],
                                     "chunk_ids": list(node["chunk_ids"]), "status": status, "note": note})

    def _exact_text_fallback(self, node: Dict[str, Any], text: str) -> bool:
        """Deterministic fallback: if the XML text equals an UNMAPPED chunk's
        text exactly, map it here (cross-group placements: ack / fn-group /
        supplementary / merged list labels). Never guesses - exact equality."""
        if not text or node["chunk_ids"]:
            return False
        want = normalize_text(text)
        if not want:
            return False
        for cid in sorted(self.chunks, key=lambda c: self.chunks[c]["position"]):
            if cid in self.chunk_to_node:
                continue
            if self.chunks[cid]["text"] == want:
                self._map_chunks(node, [cid], status="mapped-text",
                                 note="exact text fallback (cross-group)")
                return True
        return False

    # ------------------------------------------------------------------
    # registry plumbing
    # ------------------------------------------------------------------
    def _register(self, node: Dict[str, Any]) -> None:
        self._index_node(node)

    # ------------------------------------------------------------------
    # grouping helpers (mapping)
    # ------------------------------------------------------------------
    def _prose_groups(self) -> Dict[Tuple[str, str], List[str]]:
        """Corpus prose chunk ids (paragraph/list/equation) grouped by
        (section, subsection)."""
        groups: Dict[Tuple[str, str], List[str]] = {}
        for cid, meta in self.chunks.items():
            if meta["chunk_type"] not in ("paragraph", "list", "equation"):
                continue
            key = (meta["section"], meta["subsection"])
            groups.setdefault(key, []).append(cid)
        for key in groups:
            groups[key].sort(key=lambda c: self.chunks[c]["position"])
        return groups

    def _match_prose(self, xml_paras: Sequence[str], corpus_ids: Sequence[str],
                     chunk_type_hint: str = "paragraph") -> Tuple[List[Dict[str, Any]], List[str]]:
        """Greedy deterministic alignment: consume consecutive XML paragraphs
        until the normalized join equals one corpus chunk's normalized text
        (the chunker merges paragraphs / assigns one chunk per list).

        Returns (matches, remaining_corpus_ids) where each match is
        {paragraph_texts: [...], chunk_ids: [...], status}.
        """
        matches: List[Dict[str, Any]] = []
        pool = list(corpus_ids)
        remaining_xml = list(xml_paras)
        for cid in pool:
            ctext = self.chunks[cid]["text"]
            if not ctext:
                matches.append({"paragraph_texts": [], "chunk_ids": [cid], "status": "unmatched-empty"})
                continue
            if not remaining_xml:
                matches.append({"paragraph_texts": [], "chunk_ids": [cid], "status": "excess-chunk"})
                continue
            consumed: List[str] = []
            matched = False
            for i in range(1, min(len(remaining_xml), 40) + 1):
                joined = "\n\n".join(remaining_xml[:i])
                if normalize_text(joined) == ctext:
                    consumed = remaining_xml[:i]
                    remaining_xml = remaining_xml[i:]
                    matches.append({"paragraph_texts": consumed, "chunk_ids": [cid], "status": "exact"})
                    matched = True
                    break
                if i < min(len(remaining_xml), 40):
                    continue
            if not matched:
                # fuzzy: single paragraph with high similarity (marker for review)
                head = remaining_xml[0]
                r = _ratio(normalize_text(head), ctext)
                if r >= 0.75:
                    consumed = remaining_xml[:1]
                    remaining_xml = remaining_xml[1:]
                    matches.append({"paragraph_texts": consumed, "chunk_ids": [cid], "status": "fuzzy", "ratio": round(r, 3)})
                else:
                    matches.append({"paragraph_texts": [], "chunk_ids": [cid], "status": "unmatched",
                                    "reason": "no xml paragraph group matches chunk text"})
        return matches, remaining_xml

    # ------------------------------------------------------------------
    # main build entry
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        root = self.root
        doc_node = self._new_node(DOCUMENT, self.paper_id, "", 0, [])
        self._register(doc_node)

        prose_groups = self._prose_groups()

        # -- abstract --------------------------------------------------
        abs_el = root.find(".//{*}abstract")
        if abs_el is not None:
            self._build_abstract(abs_el, doc_node, prose_groups)

        # -- body ------------------------------------------------------
        body = root.find(".//{*}body")
        if body is not None:
            self._build_body(body, doc_node, prose_groups)

        # -- floats (JATS floats-group: tables/figures detached at end) --
        floats = root.find(".//{*}floats-group")
        if floats is not None:
            self._build_floats(floats, doc_node, prose_groups)

        # -- back (references, ack, fn-group, appendix) ----------------
        back = root.find(".//{*}back")
        if back is not None:
            self._build_back(back, doc_node, prose_groups)

        # -- references appear structurally whenever ref-list exists; the
        #    parser kept ref-list OUT of back loose blocks. If a body/floats
        #    section still lacks references and back was empty, nothing extra.

        # -- unmatched chunk accounting (part 24) ----------------------
        mapped_chunks = set(self.chunk_to_node.keys())
        all_chunks = set(self.chunks.keys())
        unmatched = sorted(all_chunks - mapped_chunks)
        self.unmatched_chunks = [c for c in unmatched]
        for cid in unmatched:
            meta = self.chunks[cid]
            root_node = self.node_map["0000"]
            root_node["chunk_ids"].append(cid)   # keep reachability via document root
            self.mapping_log.append({"node_id": "0000", "node_type": "document",
                                     "chunk_ids": [cid], "status": "unmatched",
                                     "note": f"chunk ({meta['chunk_type']}) has no XML structural counterpart in {meta['section']} / {meta['subsection']}"})

        return self._artifact()

    # ------------------------------------------------------------------
    # abstract (part 7)
    # ------------------------------------------------------------------
    def _build_abstract(self, abs_el, doc_node, prose_groups) -> None:
        abs_node = self._new_node(SECTION, "Abstract", "", 1, ["Abstract"])
        self._attach(doc_node, abs_node)
        self._register(abs_node)
        secs = abs_el.findall("{*}sec")
        if secs:
            for sec in secs:
                self._build_section(sec, abs_node, prose_groups, level=2, section_label="Abstract")
        else:
            paras = [p for p in abs_el.iter("{*}p")]
            if paras:
                self._build_paragraphs(paras, abs_node, prose_groups, ("Abstract", ""), level=2,
                                       breadcrumb=["Abstract"])

    # ------------------------------------------------------------------
    # body (part 5/16)
    # ------------------------------------------------------------------
    def _build_body(self, body, doc_node, prose_groups) -> None:
        loose: List[Any] = []
        for child in body:
            tag = localname(child.tag)
            if tag == "sec":
                if loose:
                    self._build_loose(loose, doc_node, prose_groups, level=1, breadcrumb=[])
                    loose = []
                self._build_section(child, doc_node, prose_groups, level=1, section_label="")
            elif tag in ("boxed-text",):
                if loose:
                    self._build_loose(loose, doc_node, prose_groups, level=1, breadcrumb=[])
                    loose = []
                node = self._build_boxed_text(child, doc_node, prose_groups, level=1, breadcrumb=[])
            elif tag != "sec":
                loose.append(child)
        if loose:
            self._build_loose(loose, doc_node, prose_groups, level=1, breadcrumb=[])

    def _build_loose(self, els, parent, prose_groups, level, breadcrumb) -> None:
        body_node = self._new_node(SECTION, "Body", "", level, list(breadcrumb) + ["Body"],
                                   location_confidence="high")
        self._attach(parent, body_node)
        self._register(body_node)
        ptexts = [element_text(el) for el in els if localname(el.tag) == "p"]
        if ptexts:
            matches, _left = self._match_prose(ptexts, prose_groups.get(("Body", ""), []))
            for m in matches:
                self._build_paragraph_node(m, body_node, level + 1, list(breadcrumb) + ["Body"])

    def _build_section(self, sec_el, parent, prose_groups, level, section_label) -> None:
        title_el = sec_el.find("{*}title")
        title = element_text(title_el) if title_el is not None else ""
        if not title:
            title = "Untitled section"
        node_type = SECTION if parent["node_type"] in (DOCUMENT, SECTION) and level == 1 else SUBSECTION
        # abstract top-level secs are subsections of Abstract; body level-1 secs are sections
        if section_label == "Abstract" or level > 1:
            node_type = SUBSECTION if level > 1 else SECTION
        if section_label == "Abstract" and level == 2:
            node_type = SUBSECTION
        elif section_label == "" and level == 1:
            node_type = SECTION
        breadcrumb = list(parent["breadcrumb"]) + [title]
        sec_node = self._new_node(node_type, title, "", level, breadcrumb,
                                  xml_id=sec_el.get("id", ""))
        self._attach(parent, sec_node)
        self._register(sec_node)

        # gather direct prose paragraphs in THIS section for grouped matching
        paras = [c for c in sec_el if localname(c.tag) == "p"]
        lists = [c for c in sec_el if localname(c.tag) == "list"]
        sub_key = ("", "") if section_label == "" and level == 1 else None
        if section_label == "Abstract":
            group_key = ("Abstract", title)
        else:
            group_key = (title if level == 1 else self._top_section_title(sec_el), self._subsection_title_for(sec_el, level))
        # build the section's own content in XML order
        for child in sec_el:
            tag = localname(child.tag)
            if tag == "sec":
                self._build_section(child, sec_node, prose_groups, level + 1, section_label)
            elif tag == "p":
                # paragraphs handled by grouped matching below
                continue
            elif tag == "list":
                self._build_list(child, sec_node, prose_groups, group_key, level + 1, breadcrumb)
            elif tag == "table-wrap" or tag == "table-wrap-foot":
                self._build_table_wrap(child, sec_node, prose_groups, level + 1, breadcrumb, inline=True)
            elif tag == "fig":
                self._build_figure(child, sec_node, prose_groups, level + 1, breadcrumb, inline=True)
            elif tag in ("supplementary-material", "boxed-text"):
                self._build_special(child, sec_node, prose_groups, level + 1, breadcrumb)
        # paragraph matching for this section
        if paras:
            xml_paras = [element_text(p) for p in sec_el.iter("{*}p")
                         if not p.xpath("ancestor::*[local-name() = 'table-wrap' or local-name() = 'table' or local-name() = 'fig' or local-name() = 'table-wrap-foot' or local-name() = 'list' or local-name() = 'list-item' or local-name() = 'fn' or local-name() = 'supplementary-material']")
                         and self._direct_or_nested(p, sec_el)]
            if xml_paras:
                matches, left_paras = self._match_prose(xml_paras, prose_groups.get(group_key, []))
                for m in matches:
                    self._build_paragraph_node(m, sec_node, level + 1, breadcrumb)
                # no information loss: XML paragraphs without a corpus chunk
                # still become leaf paragraph nodes (explicitly reported).
                for ptext in left_paras:
                    node = self._new_node(PARAGRAPH, None, ptext, level + 1, list(breadcrumb),
                                          location_confidence="high")
                    self._attach(sec_node, node)
                    self._register(node)
                    if not self._exact_text_fallback(node, ptext):
                        self._map_chunks(node, [], status="unmatched",
                                         note="no corpus chunk matches this XML paragraph")

    def _direct_or_nested(self, p, sec_el) -> bool:
        # p belongs to this section if its nearest ancestor sec (skipping
        # list/fig/table-wraps) is sec_el
        anc = p.getparent()
        while anc is not None and localname(anc.tag) not in ("sec", "list", "list-item",
                                                             "table-wrap", "fig", "fn"):
            anc = anc.getparent()
        if anc is not None and localname(anc.tag) != "sec":
            return False
        return anc is sec_el

    def _top_section_title(self, el) -> str:
        """The TOP-LEVEL body section title (the corpus stores section = the
        top-level sec, subsection = the DEEPEST nested title; intermediate
        nesting levels are collapsed by the chunker's metadata model)."""
        cur = el
        top = None
        while cur is not None:
            if localname(cur.tag) == "sec":
                t = cur.find("{*}title")
                if t is not None and (t.text or "").strip():
                    top = (t.text or "").strip()
            if cur.getparent() is None or localname(cur.getparent().tag) == "body":
                break
            cur = cur.getparent()
        return top or ""
    def _subsection_title_for(self, el, level) -> str:
        if level == 1:
            return ""  # top-level section: no subsection
        # deepest own title
        t = el.find("{*}title")
        return (t.text or "").strip() if t is not None else ""
    def _build_paragraph_node(self, match, parent, level, breadcrumb) -> None:
        for ptext in match.get("paragraph_texts", []):
            node = self._new_node(PARAGRAPH, None, ptext, level, list(breadcrumb),
                                  location_confidence="high")
            self._attach(parent, node)
            self._register(node)
            self._map_chunks(node, match.get("chunk_ids", []),
                             status=match.get("status", "mapped"),
                             note=("merged" if len(match.get("paragraph_texts", [])) > 1 else ""))

    def _build_paragraphs(self, paras, parent, prose_groups, group_key, level, breadcrumb) -> None:
        xml_paras = [element_text(p) for p in paras]
        matches, _left = self._match_prose(xml_paras, prose_groups.get(group_key, []))
        # unmatched paragraphs still become nodes (explicitly reported)
        used = 0
        for m in matches:
            self._build_paragraph_node(m, parent, level, breadcrumb)
            used += len(m["paragraph_texts"])
        for ptext in xml_paras[used:]:
            node = self._new_node(PARAGRAPH, None, ptext, level, list(breadcrumb), location_confidence="high")
            self._attach(parent, node)
            self._register(node)
            if not self._exact_text_fallback(node, ptext):
                self._map_chunks(node, [], status="unmatched", note="no corpus chunk matches this XML paragraph")

    # ------------------------------------------------------------------
    # lists (part 14)
    # ------------------------------------------------------------------
    def _build_list(self, list_el, parent, prose_groups, group_key, level, breadcrumb) -> None:
        label_el = list_el.find("{*}title")
        label = element_text(label_el) if label_el is not None else ""
        title = label or "List"
        list_node = self._new_node(LIST, title, "", level, list(breadcrumb) + ([title] if label else []),
                                   location_confidence="high")
        self._attach(parent, list_node)
        self._register(list_node)
        items = list_el.findall("{*}list-item")
        for i, item in enumerate(items):
            item_text = self._list_item_text(item)
            inode = self._new_node(LIST_ITEM, f"List item {i + 1}", item_text, level + 1,
                                   list(list_node["breadcrumb"]), location_confidence="high")
            self._attach(list_node, inode)
            self._register(inode)
        # list chunk mapping: the chunker produced ONE list chunk per list with
        # the shape [optional label p] + "- item" lines (the parser merges a
        # preceding label paragraph into the same chunk).
        label_p_text = ""
        siblings = [c for c in list_el.getparent().iterchildren() if localname(c.tag) in ("p", "list")]
        idx = 0
        for i, c in enumerate(siblings):
            if c is list_el:
                idx = i
                break
        for j in range(idx - 1, -1, -1):
            if localname(siblings[j].tag) == "p":
                label_p_text = element_text(siblings[j])
                break
        item_texts = ["- " + element_text(it) for it in list_el.findall("{*}list-item")]
        candidates = []
        if label_p_text:
            candidates.append(normalize_text("\n".join([label_p_text] + item_texts)))
        candidates.append(normalize_text("\n".join(item_texts)))
        candidates.append(normalize_text("\n".join(element_text(it) for it in list_el.findall("{*}list-item"))))
        cands = prose_groups.get(group_key, [])
        list_chunks = [c for c in cands if self.chunks[c]["chunk_type"] == "list"]
        hit = None
        for c in list_chunks:
            if c in self.chunk_to_node:
                continue
            ctext = self.chunks[c]["text"]
            if ctext in candidates:
                hit = c
                break
        if hit is not None:
            self._map_chunks(list_node, [hit], status="exact", note="list chunk")
        elif not self._exact_text_fallback(list_node, normalize_text("\n".join(item_texts))):
            self._map_chunks(list_node, [], status="unmatched", note="no list chunk matches this XML list")

    def _list_item_text(self, item) -> str:
        return element_text(item)

    # ------------------------------------------------------------------
    # tables (parts 8-10, 13)
    # ------------------------------------------------------------------
    def _build_table_wrap(self, wrap, parent, prose_groups, level, breadcrumb, inline: bool,
                          confidence: str = "high") -> None:
        table_id = wrap.get("id", "")
        label_el = wrap.find("{*}label")
        label = element_text(label_el) if label_el is not None else ""
        title = label or (f"Table {table_id}" if table_id else "Table")
        table_node = self._new_node(TABLE, title, "", level,
                                    list(breadcrumb) + [title],
                                    xml_id=table_id,
                                    table_id=table_id or None,
                                    location_confidence=confidence,
                                    extra={"label": label.strip(" .")})
        self._attach(parent, table_node)
        self._register(table_node)
        prefix = f"{self.paper_id}_{table_id}" if table_id else None

        # summary node (label + caption + columns from chunker conventions)
        (summary_text, summary_chunks) = self._table_summary_mapping(wrap, prefix, table_id)
        summary_node = self._new_node(TABLE_SUMMARY, "Summary", summary_text, level + 1,
                                      list(table_node["breadcrumb"]), xml_id=table_id,
                                      location_confidence="high", table_id=table_id)
        self._attach(table_node, summary_node)
        self._register(summary_node)
        self._map_chunks(summary_node, summary_chunks,
                         status="mapped" if summary_chunks else "unmatched",
                         note="table_summary chunk(s)" if summary_chunks else "no summary chunk")

        # structural children: thead -> header; tbody rows -> rows; fn -> footnotes
        tbl = wrap.find(".//{*}table")
        if tbl is not None:
            thead = tbl.find("{*}thead")
            if thead is not None:
                header_nodes = self._build_table_header(thead, table_node, table_id, prefix, level + 1)
        tbody = tbl.find("{*}tbody") if tbl is not None else None
        rows = wrap.findall(".//{*}tr") if tbl is not None else []
        self._build_table_rows(rows, table_node, table_id, prefix, level + 1)
        # footnotes
        fns = [f for f in wrap.findall("{*}fn")]
        if fns:
            fn_chunks = [f"{prefix}_footnotes"] if prefix and f"{prefix}_footnotes" in self.chunks else []
            for i, fn in enumerate(fns):
                fn_node = self._new_node(TABLE_FOOTNOTE, f"Footnote {i + 1}", element_text(fn),
                                         level + 1, list(table_node["breadcrumb"]),
                                         xml_id=fn.get("id", ""), location_confidence="high",
                                         table_id=table_id)
                self._attach(table_node, fn_node)
                self._register(fn_node)
                self._map_chunks(fn_node, fn_chunks,
                                 status="mapped" if fn_chunks else "unmatched",
                                 note="shared table_footnotes chunk" if fn_chunks else "no footnotes chunk")

    def _table_summary_mapping(self, wrap, prefix, table_id) -> Tuple[str, List[str]]:
        label_el = wrap.find("{*}label")
        label = element_text(label_el) if label_el is not None else ""
        cap = wrap.find("{*}caption")
        cap_text = element_text(cap) if cap is not None else ""
        summary_chunks = [f"{prefix}_summary"] if prefix and f"{prefix}_summary" in self.chunks else []
        summary_text = normalize_text(f"{label} {cap_text}".strip())
        return summary_text, summary_chunks

    def _build_table_header(self, thead, table_node, table_id, prefix, level) -> List[Dict[str, Any]]:
        cells = []
        for tr in thead.findall("{*}tr"):
            cells.append(" | ".join(element_text(c) for c in tr.findall(".//{*}th") + tr.findall(".//{*}td")))
        header_text = normalize_text("\n".join(cells))
        header_node = self._new_node(TABLE_HEADER, "Header", header_text, level,
                                     list(table_node["breadcrumb"]), location_confidence="high",
                                     table_id=table_id)
        self._attach(table_node, header_node)
        self._register(header_node)
        # thead rows may be surfaced as row chunks (row_0...) in some papers
        self._map_chunks(header_node, [], status="structural", note="thead header (no dedicated chunk type)")
        return [header_node]

    def _build_table_rows(self, trs, table_node, table_id, prefix, level) -> None:
        if not trs:
            return
        # order-based deterministic mapping + first-cell verification
        row_chunks_pool = sorted(
            [cid for cid in self.chunks if cid.startswith(f"{prefix}_row_")],
            key=lambda c: int(re.sub(r"^.*_row_", "", c) or 0),
        )
        used = set()
        for i, tr in enumerate(trs):
            cells_td = tr.findall(".//{*}td")
            cells_th = tr.findall(".//{*}th")
            cells = [element_text(c) for c in cells_th + cells_td]
            label = cells[0] if cells else f"row {i}"
            expected = f"{prefix}_row_{i}"
            # find a matching chunk by first-cell text else by order
            chunk = None
            if prefix and expected in self.chunks and expected not in used:
                chunk = expected
            else:
                for cid in row_chunks_pool:
                    if cid in used or not cid.startswith(f"{prefix}_row_"):
                        continue
                    ctext = self.chunks[cid]["text"]
                    if ctext.startswith(normalize_text("Row: " + label)) or label in ctext:
                        chunk = cid
                        break
            row_node = self._new_node(TABLE_ROW, label, "", level,
                                      list(table_node["breadcrumb"]) + ([label] if label else []),
                                      xml_id=str(tr.get("id", "")), location_confidence="high",
                                      table_id=table_id,
                                      extra={"row_index": i, "row_label": label, "columns": cells[1:] or cells})
            self._attach(table_node, row_node)
            self._register(row_node)
            if chunk:
                used.add(chunk)
                self._map_chunks(row_node, [chunk], status="mapped", note="table_row chunk")
            else:
                self._map_chunks(row_node, [], status="unmatched", note="no row chunk for XML row")

    # ------------------------------------------------------------------
    # figures (parts 11-12)
    # ------------------------------------------------------------------
    def _build_figure(self, fig_el, parent, prose_groups, level, breadcrumb, inline: bool,
                       confidence: str = "high") -> None:
        figure_id = fig_el.get("id", "")
        label_el = fig_el.find("{*}label")
        label = element_text(label_el) if label_el is not None else (f"Figure {figure_id}" if figure_id else "Figure")
        title = label
        figure_node = self._new_node(FIGURE, title, "", level,
                                     list(breadcrumb) + [title],
                                     xml_id=figure_id, figure_id=figure_id or None,
                                     location_confidence=confidence,
                                     extra={"label": label.strip(" .")})
        self._attach(parent, figure_node)
        self._register(figure_node)
        cap_el = fig_el.find("{*}caption")
        cap_text = element_text(cap_el) if cap_el is not None else element_text(fig_el)
        cap_node = self._new_node(FIGURE_CAPTION, "Caption", cap_text, level + 1,
                                  list(figure_node["breadcrumb"]), xml_id=figure_id,
                                  location_confidence="high", figure_id=figure_id)
        self._attach(figure_node, cap_node)
        self._register(cap_node)
        chunk_ids = [f"{self.paper_id}_{figure_id}"] if figure_id and f"{self.paper_id}_{figure_id}" in self.chunks else []
        self._map_chunks(cap_node, chunk_ids, status="mapped" if chunk_ids else "unmatched",
                         note="figure chunk carries label+caption" if chunk_ids else "no figure chunk")

    # ------------------------------------------------------------------
    # floats-group placement (parts 11, 17, 18)
    # ------------------------------------------------------------------
    def _build_floats(self, floats, doc_node, prose_groups) -> None:
        # explicit rid-based placement via in-text xrefs (parts 18)
        xref_targets: Dict[str, str] = {}     # rid -> section breadcrumb (first xref position)
        xref_node_path: Dict[str, Dict[str, Any]] = {}
        body = self.root.find(".//{*}body")
        for xref in self.root.iter("{*}xref"):
            rid = xref.get("rid", "")
            rt = xref.get("ref-type", "")
            if not rid or rt not in ("table", "fig"):
                continue
            if rid in xref_targets:
                continue
            sec_path = self._section_path_of(xref)
            xref_targets[rid] = sec_path
            xref_node_path[rid] = self._section_node_for_path(sec_path, doc_node)
        for child in floats:
            tag = localname(child.tag)
            if tag == "table-wrap" or tag == "table-wrap-foot":
                rid = child.get("id", "")
                if rid and rid in xref_node_path:
                    target = xref_node_path[rid]
                    self._build_table_wrap(child, target, prose_groups, target["level"] + 1,
                                           list(target["breadcrumb"]), inline=False)
                    # mark confidence via xml_id-based handling inside _build_table_wrap
                    continue
                # no explicit relationship -> preserve XML floats order at document level
                self._build_table_wrap(child, doc_node, prose_groups, 1, [], inline=False)
            elif tag == "fig":
                rid = child.get("id", "")
                if rid and rid in xref_node_path:
                    target = xref_node_path[rid]
                    self._build_figure(child, target, prose_groups, target["level"] + 1,
                                       list(target["breadcrumb"]), inline=False)
                    continue
                self._build_figure(child, doc_node, prose_groups, 1, [], inline=False)

    def _section_path_of(self, el) -> List[str]:
        names = []
        node = el
        while node is not None:
            if localname(node.tag) == "sec":
                t = node.find("{*}title")
                if t is not None and (t.text or "").strip():
                    names.insert(0, (t.text or "").strip())
            node = node.getparent()
        return names

    def _section_node_for_path(self, path: List[str], doc_node) -> Dict[str, Any]:
        """Find the deepest tree node matching this section title path."""
        if not path:
            return doc_node
        current = doc_node
        for want in path:
            found = None
            for cid in current["child_node_ids"]:
                child = self.node_map[cid]
                if child.get("title") == want and child["node_type"] in (SECTION, SUBSECTION):
                    found = child
                    break
            if found is None:
                return current
            current = found
        return current

    # ------------------------------------------------------------------
    # back (parts 15-16)
    # ------------------------------------------------------------------
    def _build_back(self, back, doc_node, prose_groups) -> None:
        for child in back:
            tag = localname(child.tag)
            if tag == "ref-list":
                refs_node = self._new_node(REFERENCES, "References", "", 1, ["References"],
                                           location_confidence="high")
                self._attach(doc_node, refs_node)
                self._register(refs_node)
                refs = child.findall("{*}ref")
                for i, ref in enumerate(refs):
                    ref_text = element_text(ref)
                    inode = self._new_node(REFERENCE, f"Reference {i + 1}", ref_text, 2,
                                           ["References"], xml_id=ref.get("id", ""))
                    self._attach(refs_node, inode)
                    self._register(inode)
                    # reference chunks (if the corpus emitted them)
                    cands = [c for c in self.chunks if self.chunks[c]["chunk_type"] == "reference"]
                    hit = None
                    for c in cands:
                        if self.chunks[c]["text"] == ref_text:
                            hit = c
                            break
                    self._map_chunks(inode, [hit] if hit else [],
                                     status="mapped" if hit else "no-ref-chunk",
                                     note="reference chunk" if hit else "corpus has no reference chunks for this paper")
            elif tag == "ack":
                ack_node = self._new_node(SECTION, "Acknowledgments", "", 1, ["Acknowledgments"])
                self._attach(doc_node, ack_node)
                self._register(ack_node)
                for p in child.findall("{*}p"):
                    pnode = self._new_node(PARAGRAPH, None, element_text(p), 2, ["Acknowledgments"])
                    self._attach(ack_node, pnode)
                    self._register(pnode)
                    self._map_chunks(pnode, [], status="unmatched", note="ack paragraph (chunk may live elsewhere)")
            elif tag == "fn-group":
                fng = self._new_node(SECTION, "Footnotes", "", 1, ["Footnotes"])
                self._attach(doc_node, fng)
                self._register(fng)
                for fn in child.findall("{*}fn"):
                    fnnode = self._new_node(PARAGRAPH, None, element_text(fn), 2,
                                            ["Footnotes"], xml_id=fn.get("id", ""))
                    self._attach(fng, fnnode)
                    self._register(fnnode)
                    if not self._exact_text_fallback(fnnode, element_text(fn)):
                        self._map_chunks(fnnode, [], status="unmatched",
                                         note="document footnote chunk lookup TBD")
            elif tag == "app" or tag == "app-group":
                for sub in child:
                    if localname(sub.tag) == "sec":
                        self._build_section(sub, doc_node, prose_groups, 1, "")
                    else:
                        self._build_special(sub if localname(sub.tag) in ("p", "table-wrap", "fig", "list") else sub,
                                            doc_node, prose_groups, 1, ["Appendix"])
            elif tag == "sec":
                self._build_section(child, doc_node, prose_groups, 1, "")

    def _build_special(self, el, parent, prose_groups, level, breadcrumb) -> None:
        """supplementary-material / boxed-text / app content."""
        title_el = el.find("{*}title")
        title = element_text(title_el) if title_el is not None else localname(el.tag)
        node = self._new_node(SECTION, title, "", level, list(breadcrumb) + [title])
        self._attach(parent, node)
        self._register(node)
        for p in el.findall(".//{*}p"):
            pnode = self._new_node(PARAGRAPH, None, element_text(p), level + 1, list(node["breadcrumb"]))
            self._attach(node, pnode)
            self._register(pnode)
            self._map_chunks(pnode, [], status="unmatched", note=f"{title} paragraph (chunk may be absent)")

    def _build_boxed_text(self, el, parent, prose_groups, level, breadcrumb) -> Dict[str, Any]:
        title_text = element_text(el.find("{*}title")) if el.find("{*}title") is not None else "Boxed text"
        node = self._new_node(SECTION, title_text, "", level, list(breadcrumb) + [title_text])
        self._attach(parent, node)
        self._register(node)
        for p in el.findall("{*}p"):
            pnode = self._new_node(PARAGRAPH, None, element_text(p), level + 1, list(node["breadcrumb"]))
            self._attach(node, pnode)
            self._register(pnode)
        return node

    # ------------------------------------------------------------------
    # artifact assembly (part 21-22)
    # ------------------------------------------------------------------
    def _artifact(self) -> Dict[str, Any]:
        # tree: nested structure for the PageIndex navigation layer (title,
        # node_id, text, nodes + node_type); keep adapter-compatible.
        def to_tree(node: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "node_id": node["node_id"],
                "node_type": node["node_type"],
                "title": node["title"] if node["title"] is not None else (node["title"] or ""),
                "text": node["text"] or "",
                "nodes": [to_tree(self.node_map[c]) for c in node["child_node_ids"]],
            }
        tree = [to_tree(self.node_map["0000"])]
        # whether the XML itself contains a genuine section titled "Figures and
        # Tables" (only then may such a branch exist - part 17).
        xml_has_ft = False
        for title in self.root.iter("{*}title"):
            if (title.text or "").strip().lower() == "figures and tables":
                xml_has_ft = True
                break
        return {
            "paper_id": self.paper_id,
            "metadata": {
                "source_xml": str(self.xml_path),
                "paper_id": self.paper_id,
                "builder_version": "xml-tree-v2.2",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_existing_chunks": len(self.chunks),
                "mapped_chunks": len(self.chunk_to_node),
                "unmapped_chunks": len(self.unmatched_chunks),
                "source_xml_figures_tables_section": xml_has_ft,
                "format": "xml-derived semantic document tree",
            },
            "tree": tree,
            "node_map": dict(self.node_map),
            "chunk_to_node": dict(self.chunk_to_node),
            "section_index": dict(self.section_index),
            "table_index": dict(self.table_index),
            "figure_index": dict(self.figure_index),
            "node_to_parent": dict(self.node_to_parent),
            "node_to_children": dict(self.node_to_children),
            "mapping_log": list(self.mapping_log),
        }
# ---------------------------------------------------------------------------
# Validation (part 25)
# ---------------------------------------------------------------------------

_EVIDENCE_TYPES = (PARAGRAPH, TABLE_SUMMARY, TABLE_HEADER, TABLE_ROW,
                   TABLE_FOOTNOTE, FIGURE_CAPTION, LIST_ITEM, REFERENCE)


def validate_tree(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Run the 14 structural checks from the spec (part 25)."""
    nm = artifact.get("node_map", {}) or {}
    n2c = artifact.get("node_to_children", {}) or {}
    results: List[Dict[str, Any]] = []

    def check(num: int, name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": num, "name": name, "ok": bool(ok), "detail": detail})

    # 1. exactly one root node, node_type document
    roots = [n for n in nm.values() if not n.get("parent_node_id")]
    check(1, "one document root", len(roots) == 1 and roots and roots[0]["node_type"] == "document",
          f"roots={len(roots)}" + (f" type={roots[0].get('node_type')}" if roots else ""))

    # 2. every node has at most one parent (a valid one)
    bad_parents = [nid for nid, n in nm.items() if n.get("parent_node_id") and n["parent_node_id"] not in nm]
    check(2, "single valid parent per node", not bad_parents,
          f"bad_parent_refs={bad_parents[:4]}")

    # 3. every child belongs to the parent's child list (bidirectional)
    inconsistent = [nid for nid, n in nm.items()
                    if n.get("parent_node_id") and nid not in n2c.get(n["parent_node_id"], [])]
    check(3, "child list consistency", not inconsistent, f"inconsistent={inconsistent[:4]}")

    # 4. no cycles (DFS from the root)
    cycle = False
    if roots:
        seen = set()
        def dfs(rid: str) -> None:
            nonlocal cycle
            if rid in seen:
                cycle = True
                return
            seen.add(rid)
            for c in n2c.get(rid, []):
                dfs(c)
        dfs(roots[0]["node_id"])
        check(4, "no cycles", not cycle and len(seen) == len(nm),
              f"visited={len(seen)}/total={len(nm)}")
    else:
        check(4, "no cycles", False, "no root")

    # 5. node types are from the taxonomy
    bad_types = [nid for nid, n in nm.items() if n.get("node_type") not in NODE_TYPES]
    check(5, "node types from taxonomy", not bad_types, f"bad_types={bad_types[:5]}")

    # 6. every evidence object has a breadcrumb
    no_crumb = [nid for nid, n in nm.items()
                if n.get("node_type") in _EVIDENCE_TYPES and not n.get("breadcrumb")]
    check(6, "evidence objects carry breadcrumbs", not no_crumb, f"missing={no_crumb[:5]}")

    # 7. every table node keeps its table id where available
    tables_missing_id = [nid for nid, n in nm.items()
                         if n.get("node_type") == TABLE and not n.get("table_id") and not n.get("xml_id")]
    check(7, "tables carry table id", not tables_missing_id, f"tables_no_id={tables_missing_id[:5]}")

    # 8. every table row maps to its parent table
    row_no_table = [nid for nid, n in nm.items()
                    if n.get("node_type") == TABLE_ROW and n.get("table_id") is None]
    check(8, "rows know their parent table", not row_no_table, f"rows_no_table={row_no_table[:5]}")

    # 9. every table footnote maps to its parent table
    fn_no_table = [nid for nid, n in nm.items()
                   if n.get("node_type") == TABLE_FOOTNOTE and n.get("table_id") is None]
    check(9, "footnotes know their parent table", not fn_no_table, f"fn_no_table={fn_no_table[:5]}")

    # 10. every figure caption maps to its parent figure
    cap_no_fig = [nid for nid, n in nm.items()
                  if n.get("node_type") == FIGURE_CAPTION and n.get("figure_id") is None
                  and not n.get("xml_id")]
    check(10, "captions know their parent figure", not cap_no_fig, f"cap_no_fig={cap_no_fig[:5]}")

    # 11. every existing chunk is mapped or explicitly reported
    total = artifact.get("metadata", {}).get("total_existing_chunks", 0)
    mapped = artifact.get("metadata", {}).get("mapped_chunks", 0)
    unmatched = artifact.get("metadata", {}).get("unmapped_chunks", 0)
    accounted = mapped + unmatched
    check(11, "all chunks mapped or reported", total == accounted,
          f"total={total} mapped={mapped} unmatched={unmatched} accounted={accounted}")

    # 12. no invented section titles: paragraphs carry NO title
    para_titled = [nid for nid, n in nm.items()
                   if n.get("node_type") == PARAGRAPH and n.get("title")]
    check(12, "paragraphs are not section titles", not para_titled,
          f"paragraph_titles={para_titled[:5]}")

    # 13. sections carry XML-derived titles
    sec_no_title = [nid for nid, n in nm.items()
                    if n.get("node_type") in (SECTION, SUBSECTION) and not n.get("title")]
    check(13, "sections carry XML-derived titles", not sec_no_title,
          f"section_no_title={sec_no_title[:5]}")

    # 14. no manufactured global Figures-and-Tables branch
    gft = [nid for nid, n in nm.items()
           if n.get("node_type") in (SECTION, SUBSECTION)
           and (n.get("title") or "").strip().lower() == "figures and tables"
           and artifact.get("metadata", {}).get("source_xml_figures_tables_section") is False]
    check(14, "no manufactured Figures-and-Tables branch", not gft, f"manufactured={gft[:5]}")

    ok = all(r["ok"] for r in results)
    return {"passed": bool(ok), "checks": results, "n_checks": len(results),
            "passed_checks": sum(1 for r in results if r["ok"])}


# ---------------------------------------------------------------------------
# Rendering (part 28) + stats
# ---------------------------------------------------------------------------
def tree_stats(artifact: Dict[str, Any]) -> Dict[str, int]:
    from collections import Counter
    nm = artifact.get("node_map", {}) or {}
    return dict(Counter(n.get("node_type", "?") for n in nm.values()))


def render_tree(artifact: Dict[str, Any], max_depth: int = 8, max_children: int = 64) -> List[str]:
    """Human-readable hierarchy (part 28)."""
    tree = artifact.get("tree") or []
    lines: List[str] = []
    if not tree:
        return lines

    def walk(nodes, prefix: str, depth: int) -> None:
        for i, node in enumerate(nodes):
            last = i == len(nodes) - 1
            branch = "└── " if last else "├── "
            title = node.get("title") or ""
            ntype = node.get("node_type", "")
            label = title or node.get("node_id")
            suffix = f"  [{ntype}]"
            if node.get("chunk_ids"):
                suffix += "  ↦ " + ",".join(node["chunk_ids"][:2]) + ("…" if len(node["chunk_ids"]) > 2 else "")
            lines.append(("" if ntype == "document" else prefix + branch) + str(label) + suffix)
            children = node.get("nodes") or []
            ext = "    " if last else "│   "
            if children and depth < max_depth:
                walk(children[:max_children], prefix + ext, depth + 1)
            elif children:
                lines.append(prefix + ext + "…")

    walk(tree, "", 0)
    return lines


def _xml_schema_report(root) -> Dict[str, Any]:
    """Summary of the discovered XML schema (part 1)."""
    q = lambda n: ET.QName(n).localname
    body = root.find(".//{*}body")
    secs = body.findall(".//{*}sec") if body is not None else []
    paras = sum(1 for p in root.iter("{*}p")
                if not p.xpath("ancestor::*[local-name() = 'table-wrap' or local-name() = 'table' or local-name() = 'fig']"))
    tables = list(root.iter("{*}table-wrap"))
    figs = list(root.iter("{*}fig"))
    fns = list(root.iter("{*}fn"))
    refs = list(root.iter("{*}ref"))
    lists = list(root.iter("{*}list"))
    floats = root.find(".//{*}floats-group")
    return {
        "root_tag": q(root),
        "article_ids": sorted({(a.get("pub-id-type"), (a.text or "").strip()) for a in root.iter("{*}article-id")}),
        "has_body": body is not None,
        "has_back": root.find(".//{*}back") is not None,
        "has_floats_group": floats is not None,
        "n_secs": len(secs),
        "n_paragraphs": paras,
        "n_lists": len(lists),
        "n_tables": len(tables),
        "n_figures": len(figs),
        "n_table_fns": len(fns),
        "n_refs": len(refs),
        "table_ids": [t.get("id") for t in tables[:10]],
        "figure_ids": [f.get("id") for f in figs[:10]],
    }


# ---------------------------------------------------------------------------
# Old chunk-derived tree (comparison only, part 26) - reuse existing builder
# ---------------------------------------------------------------------------
def build_chunk_derived_artifact(paper_id: str, index_dir: Path, pageindex_dir: Path):
    from medrag.retrieval_v2.document_index import LogicalDocumentIndex
    from medrag.retrieval.corpus import CORPUS_FILENAME
    from medrag.retrieval_v2.pageindex_build import build_pageindex_artifact

    doc_index = LogicalDocumentIndex(Path(index_dir) / CORPUS_FILENAME)
    return build_pageindex_artifact(doc_index, paper_id, Path(pageindex_dir))


# ---------------------------------------------------------------------------
# Orchestrated report (part 28)
# ---------------------------------------------------------------------------
def report_paper(paper_id: str, xml_dir: Path, corpus_path: Path) -> Dict[str, Any]:
    """Part-28 debug output for one paper: XML schema, counts, mapping,
    hierarchy render, validation."""
    xml_path = locate_xml(paper_id, xml_dir)
    report: Dict[str, Any] = {"paper_id": paper_id}
    if xml_path is None:
        report["error"] = "no XML found"
        print(f"{paper_id}: NO XML FOUND in {xml_dir}")
        return report
    report["xml"] = {"path": str(xml_path)}
    print(f"paper_id: {paper_id}")
    print(f"XML:      {xml_path}")

    builder = XmlTreeBuilder(paper_id, xml_path, corpus_path, xml_dir=xml_dir)
    root = builder.root
    schema = _xml_schema_report(root)
    report["xml_schema"] = schema
    print("XML schema:")
    for k, v in schema.items():
        print(f"  {k}: {v}")

    artifact = builder.build()
    stats = tree_stats(artifact)
    report["tree_stats"] = stats
    report["mapping"] = {
        "existing_chunks": artifact["metadata"]["total_existing_chunks"],
        "mapped_chunks": artifact["metadata"]["mapped_chunks"],
        "unmapped_chunks": artifact["metadata"]["unmapped_chunks"],
    }
    print("XML counts:")
    for k in ("n_secs", "n_paragraphs", "n_tables", "n_figures", "n_table_fns", "n_lists", "n_refs"):
        print(f"  XML {k}: {schema.get(k)}")
    print("tree counts:")
    for k, v in sorted(stats.items()):
        print(f"  tree {k}: {v}")
    print("mapping:")
    print(f"  existing chunks: {artifact['metadata']['total_existing_chunks']}")
    print(f"  mapped chunks:   {artifact['metadata']['mapped_chunks']}")
    print(f"  unmatched chunks:{artifact['metadata']['unmapped_chunks']}")
    print()
    print("hierarchy:")
    for line in render_tree(artifact):
        print("  " + line)
    print()
    val = validate_tree(artifact)
    report["validation"] = {k: v for k, v in val.items() if k != "checks"}
    report["validation_checks"] = val["checks"]
    print("validation: " + ("PASS" if val["passed"] else "FAIL"))
    for r_ in val["checks"]:
        print(f"  [{r_['check']:>2}] {'OK ' if r_['ok'] else 'FAIL'} {r_['name']}" + (f"  ({r_['detail']})" if r_['detail'] else ""))
    print()
    report["artifact"] = artifact
    return report
