"""JATS XML -> Document AST parser.

Merged flat-module version of the old ``plugins.parsers.jats`` and
``plugins.parsers.pmc_ast_parser``. Parses PMC/JATS XML directly into a
structure-aware ``Document`` AST (never via Markdown).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree

from src.chunking.classification import classify_section_title
from src.lib.models import Block, Document, Section


class JATSBaseParser:
    """Extracts article metadata and provides XML helpers for JATS documents."""

    PMC_OA_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com"

    IMAGE_EXTENSIONS = frozenset({
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".tif", ".tiff", ".bmp", ".eps",
    })

    _NON_IMAGE_PATTERNS = [
        re.compile(r"^G[SEPL]\d+$"),
        re.compile(r"^[NX][MR]_\d+(\.\d+)?$"),
        re.compile(r"^[A-Z]{2}\d{6,}(\.\d+)?$"),
        re.compile(r"^10\.\d{4,}/"),
        re.compile(r"^[A-Z]\d{5}$"),
        re.compile(r"^[A-Z]{2,}\d{4,}$"),
        re.compile(r"^www\."),
        re.compile(r"biorender", re.IGNORECASE),
        re.compile(r"^Movies?\s", re.IGNORECASE),
    ]

    def __init__(self, image_base_url: str = PMC_OA_BASE_URL) -> None:
        self.image_base_url = image_base_url.rstrip("/") if image_base_url else ""
        self._article_dir = ""

    # ==================================================================
    # DOCUMENT-LEVEL HELPERS
    # ==================================================================

    def _resolve_article_dir(self, root) -> None:
        """Determine the article directory from its PMC identifiers."""
        ids = {
            aid.get("pub-id-type"): "".join(aid.itertext()).strip()
            for aid in root.xpath("//front/article-meta/article-id")
        }
        self._article_dir = ids.get("pmcid-ver") or ids.get("pmcid") or ""

    # ==================================================================
    # METADATA EXTRACTION
    # ==================================================================

    def _parse_front_matter(self, root) -> Dict[str, Any]:
        meta = {
            "journal_meta": self._parse_journal_meta(root),
            "article_meta": self._parse_article_meta(root),
            "supplementary_assets": self._parse_supplementary_assets(root),
        }
        return self._clean_metadata(meta)

    def _parse_journal_meta(self, root) -> Dict[str, Any]:
        nodes = root.xpath("//front/journal-meta")
        if not nodes:
            return {}
        j_meta = nodes[0]
        return {
            "journal_ids": {
                jid.get("journal-id-type"): "".join(jid.itertext()).strip()
                for jid in j_meta.xpath("./journal-id")
                if jid.get("journal-id-type")
            },
            "title": self._get_text(j_meta, ".//journal-title"),
            "issns": {
                issn.get("pub-type"): "".join(issn.itertext()).strip()
                for issn in j_meta.xpath("./issn")
                if issn.get("pub-type")
            },
            "publisher": {
                "name": self._get_text(j_meta, ".//publisher-name"),
                "loc": self._get_text(j_meta, ".//publisher-loc"),
            },
        }

    def _parse_article_meta(self, root) -> Dict[str, Any]:
        nodes = root.xpath("//front/article-meta")
        if not nodes:
            return {}
        a_meta = nodes[0]
        return {
            "article_ids": {
                aid.get("pub-id-type"): "".join(aid.itertext()).strip()
                for aid in a_meta.xpath("./article-id")
                if aid.get("pub-id-type")
            },
            "categories": [
                subj.text.strip()
                for sg in a_meta.xpath("./article-categories/subj-group")
                for subj in sg.xpath("./subject")
                if subj.text
            ],
            "title": self._get_text(a_meta, ".//article-title"),
            "subtitle": self._get_text(a_meta, ".//subtitle"),
            "authors": self._parse_contributors(a_meta),
            "author_notes": self._parse_author_notes(a_meta),
            "publication_dates": self._parse_pub_dates(a_meta),
            "history": self._parse_history(a_meta),
            "pub_history": self._parse_pub_history(a_meta),
            "issue_info": self._parse_issue_info(a_meta),
            "permissions": self._parse_permissions(a_meta),
            "keywords": [
                kw.text.strip()
                for kw in a_meta.xpath(".//kwd-group/kwd")
                if kw.text
            ],
            "funding": self._parse_funding(a_meta),
            "counts": {cnt.tag: cnt.get("count") for cnt in a_meta.xpath("./counts/*")},
            "custom_meta": self._parse_custom_meta(a_meta),
        }

    def _parse_custom_meta(self, node) -> Dict[str, str]:
        return {
            m.findtext("meta-name"): m.findtext("meta-value")
            for m in node.xpath(".//custom-meta-group/custom-meta")
            if m.findtext("meta-name") and m.findtext("meta-value")
        }

    def _parse_supplementary_assets(self, root) -> List[Dict[str, str]]:
        assets = []
        for supp in root.xpath("//supplementary-material"):
            asset: Dict[str, str] = {}

            label_node = supp.find("label")
            if label_node is None:
                label_node = supp.find("title")
            label = " ".join(self._extract_text(label_node).split()) if label_node is not None else ""

            caption_node = supp.find("caption")
            caption_text = " ".join(self._extract_text(caption_node).split()) if caption_node is not None else ""

            if not label and caption_text:
                label = caption_text.split(".")[0] if "." in caption_text else caption_text[:60]
            if label:
                asset["label"] = label

            media = supp.find(".//media")
            if media is None:
                media = supp.find(".//graphic")
            if media is None:
                media = supp.find(".//inline-supplementary-material")
            if media is not None:
                href = self._get_href(media)
                if href:
                    asset["filename"] = href
                    asset["href"] = self._resolve_href(href)

            if not asset:
                asset["label"] = label or supp.get("id") or "Supplementary Material"
            assets.append(asset)
        return assets

    def _parse_contributors(self, a_meta) -> List[Dict[str, Any]]:
        return [self._parse_single_contrib(c) for c in a_meta.xpath("./contrib-group/contrib")]

    def _parse_single_contrib(self, contrib) -> Dict[str, Any]:
        author: Dict[str, Any] = {
            cid.get("contrib-id-type", "unknown"): "".join(cid.itertext()).strip()
            for cid in contrib.xpath("./contrib-id")
        }

        name_node = contrib.find("./name")
        if name_node is not None:
            author["given_names"] = self._get_text(name_node, "given-names")
            author["surname"] = self._get_text(name_node, "surname")
            prefix = self._get_text(name_node, "prefix")
            suffix = self._get_text(name_node, "suffix")
            author["name"] = (
                f"{prefix} {author.get('given_names', '')} {author.get('surname', '')} {suffix}"
                .replace("  ", " ").strip()
            )

        collab_node = contrib.find("./collab")
        if collab_node is not None:
            collab_parts = [collab_node.text] if collab_node.text else []
            for child in collab_node.iterchildren():
                if self._local_name(child.tag) != "contrib-group":
                    collab_parts.append("".join(child.itertext()))
            author["collab"] = "".join(collab_parts).strip()

        affs = []
        for aff in contrib.xpath("./aff"):
            aff_data: Dict[str, str] = {}
            inst = aff.find(".//institution")
            if inst is not None:
                aff_data["institution"] = "".join(inst.itertext()).strip()
            addr = aff.find(".//addr-line")
            if addr is not None:
                aff_data["city"] = "".join(addr.itertext()).strip()
            country = aff.find(".//country")
            if country is not None and country.text:
                aff_data["country"] = country.text.strip()
            if aff_data:
                affs.append(aff_data)
        if affs:
            author["affiliations"] = affs

        if contrib.find("./xref[@ref-type='corresp']") is not None:
            author["is_corresponding"] = True

        note_refs = [x.get("rid") for x in contrib.xpath("./xref[@ref-type='author-notes']") if x.get("rid")]
        if note_refs:
            author["note_refs"] = note_refs
        return author

    def _parse_author_notes(self, a_meta) -> Dict[str, Any]:
        notes: Dict[str, Any] = {}
        corresp = a_meta.find(".//author-notes/corresp")
        if corresp is not None:
            notes["correspondence"] = "".join(corresp.itertext()).strip()
        fns = ["".join(fn.itertext()).strip() for fn in a_meta.xpath(".//author-notes/fn/p")]
        if fns:
            notes["footnotes"] = fns
        return notes

    def _format_date(self, node) -> str:
        if node is None:
            return ""
        iso = node.get("iso-8601-date")
        if iso:
            return iso
        year = node.findtext("year")
        if not year:
            return ""
        parts = [year]
        month = node.findtext("month")
        day = node.findtext("day")
        if month:
            parts.append(month.zfill(2))
        if day:
            parts.append(day.zfill(2))
        return "-".join(parts)

    def _parse_pub_dates(self, a_meta) -> Dict[str, str]:
        return {
            pd.get("pub-type", "unknown"): self._format_date(pd)
            for pd in a_meta.xpath("./pub-date")
            if self._format_date(pd)
        }

    def _parse_history(self, a_meta) -> Dict[str, str]:
        return {
            hd.get("date-type", "unknown"): self._format_date(hd)
            for hd in a_meta.xpath("./history/date")
            if self._format_date(hd)
        }

    def _parse_pub_history(self, a_meta) -> List[Dict[str, Any]]:
        events = []
        for event in a_meta.xpath("./pub-history/event"):
            ev_data: Dict[str, Any] = {"event_type": event.get("event-type", "unknown")}
            date_node = event.find("./date")
            if date_node is None:
                date_node = event.find("./pub-date")
            if date_node is not None:
                ev_data["date"] = self._format_date(date_node)
            art_ids = {
                aid.get("pub-id-type"): "".join(aid.itertext()).strip()
                for aid in event.xpath("./article-id")
                if aid.get("pub-id-type")
            }
            if art_ids:
                ev_data["article_ids"] = art_ids
            version = event.findtext("./article-version")
            if version:
                ev_data["version"] = version.strip()
            events.append(ev_data)
        return events

    def _parse_issue_info(self, a_meta) -> Dict[str, str]:
        return {
            tag: a_meta.findtext(f"./{tag}").strip()
            for tag in ["volume", "issue", "fpage", "lpage", "elocation-id"]
            if a_meta.findtext(f"./{tag}")
        }

    def _parse_permissions(self, a_meta) -> Dict[str, Any]:
        perms: Dict[str, Any] = {}
        stmt = a_meta.find(".//copyright-statement")
        if stmt is not None:
            perms["copyright_statement"] = "".join(stmt.itertext()).strip()
        year = a_meta.findtext(".//copyright-year")
        if year:
            perms["copyright_year"] = year.strip()
        license_node = a_meta.find(".//license")
        if license_node is not None:
            lic_url = lic_text = None
            for el in license_node.iter():
                tag = self._local_name(el.tag)
                if tag == "license_ref":
                    lic_url = "".join(el.itertext()).strip()
                elif tag == "ext-link":
                    href = self._get_href(el)
                    if href and not lic_url:
                        lic_url = href
                elif tag == "license-p":
                    lic_text = "".join(el.itertext()).strip()
            if lic_url:
                perms["license_url"] = lic_url
            if lic_text:
                perms["license_text"] = lic_text
        return perms

    def _parse_funding(self, a_meta) -> List[Dict[str, Any]]:
        funding = []
        for award in a_meta.xpath("./funding-group/award-group"):
            aw: Dict[str, Any] = {}
            src = award.find(".//funding-source//institution")
            if src is not None:
                aw["source"] = "".join(src.itertext()).strip()
            src_id = award.find(".//funding-source//institution-id")
            if src_id is not None:
                aw["source_id"] = "".join(src_id.itertext()).strip()
            ids = [aid.text.strip() for aid in award.xpath("./award-id") if aid.text]
            if ids:
                aw["award_ids"] = ids
            if aw:
                funding.append(aw)
        return funding

    # ==================================================================
    # REFERENCES / FORMULAE
    # ==================================================================

    def _extract_reference_ids(self, citation_node) -> Tuple[str, str, str]:
        """Return ``(doi, pmid, pmcid)`` from a citation node's ``pub-id``s."""
        doi = pmid = pmcid = ""
        for pub_id in citation_node.xpath(".//pub-id"):
            id_type = pub_id.get("pub-id-type")
            val = " ".join("".join(pub_id.itertext()).split())
            if id_type == "doi":
                doi = val
            elif id_type == "pmid":
                pmid = val
            elif id_type == "pmcid":
                pmcid = val
        return doi, pmid, pmcid

    def _format_reference_text(self, citation_node) -> str:
        authors = [
            " ".join("".join(p.itertext()).split())
            for p in citation_node.xpath(".//person-group/string-name")
        ]
        author_str = ", ".join(a for a in authors if a)

        title = " ".join(self._get_text(citation_node, "article-title").split())
        source = " ".join(self._get_text(citation_node, "source").split())
        year = " ".join(self._get_text(citation_node, "year").split())
        volume = " ".join(self._get_text(citation_node, "volume").split())
        fpage = " ".join(self._get_text(citation_node, "fpage").split())
        lpage = " ".join(self._get_text(citation_node, "lpage").split())

        doi, pmid, pmcid = self._extract_reference_ids(citation_node)

        parts = [p for p in [author_str, title] if p]

        pages = ""
        if fpage and lpage:
            pages = f":{fpage}–{lpage}"
        elif fpage:
            pages = f":{fpage}"
        journal_info = [
            p for p in [source, year, f";{volume}" if volume else "", pages] if p
        ]
        if journal_info:
            parts.append("".join(journal_info))

        id_parts = []
        if doi:
            id_parts.append(f"DOI: [{doi}](https://doi.org/{doi})")
        if pmid:
            id_parts.append(f"PMID: [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})")
        if pmcid:
            id_parts.append(f"PMCID: [{pmcid}](https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid})")
        if id_parts:
            parts.append(" ".join(id_parts))

        formula_parts = []
        for formula_node in citation_node.xpath(".//inline-formula | .//disp-formula"):
            latex = self._extract_formula_latex(formula_node)
            if latex:
                formula_parts.append(f"${latex}$")
        if formula_parts:
            parts.append(" ".join(formula_parts))

        return " ".join(". ".join(parts).split())

    def _extract_formula_latex(self, node) -> str:
        tex = node.find(".//tex-math")
        if tex is not None:
            formula = "".join(tex.itertext()).strip()
            if formula.startswith("<![CDATA["):
                formula = formula[9:-3]
            formula = formula.strip()
            if formula:
                return formula

        math_node = node.find(".//{http://www.w3.org/1998/Math/MathML}math")
        if math_node is None:
            math_node = node.find(".//math")
        if math_node is not None:
            return " ".join(math_node.itertext()).strip()
        return ""

    # ==================================================================
    # ABSTRACT HELPERS
    # ==================================================================

    def _pick_primary_abstract(self, nodes) -> int:
        for i, node in enumerate(nodes):
            if (node.get("abstract-type") or "") in ("", "abstract") and self._abstract_has_content(node):
                return i
        for i, node in enumerate(nodes):
            if self._abstract_has_content(node):
                return i
        return 0

    def _abstract_has_content(self, abs_node) -> bool:
        for child in abs_node:
            if self._local_name(child.tag) != "title":
                return True
        text = "".join(abs_node.itertext()).strip()
        return bool(text) and text.lower() != "abstract"

    def _abstract_heading(self, abs_node) -> str:
        tag = self._local_name(abs_node.tag)
        if tag == "trans-abstract":
            return "Translated Abstract"
        at = (abs_node.get("abstract-type") or "").strip()
        return at.replace("-", " ").title() if at else "Abstract"

    # ==================================================================
    # XML / TEXT UTILITIES
    # ==================================================================

    @staticmethod
    def _local_name(tag) -> str:
        """Strip the XML namespace from a tag name."""
        if not isinstance(tag, str):
            return ""
        return tag.rsplit("}", 1)[-1]

    def _get_href(self, elem) -> str:
        if elem is None:
            return ""
        href = elem.get("{http://www.w3.org/1999/xlink}href") or elem.get("href")
        if href:
            return href
        for key, val in elem.attrib.items():
            if key.endswith("href") or key.endswith("}href"):
                return val
        return ""

    def _resolve_href(self, href: str) -> str:
        """Resolve a relative file href to an absolute URL."""
        if not href:
            return ""
        href = href.strip()
        if not href or href.startswith("#") or href.startswith(("http://", "https://")):
            return href
        if self.image_base_url and self._article_dir:
            return f"{self.image_base_url}/{self._article_dir}/{href}"
        if self.image_base_url:
            return f"{self.image_base_url}/{href}"
        return href

    def _resolve_image_href(self, href: str) -> str:
        """Resolve a relative image path to an absolute URL (empty if not an image)."""
        if not href:
            return ""
        href = href.strip()
        if not href or href.startswith("#") or href.startswith(("http://", "https://")):
            return href
        if self._is_non_image_href(href):
            return ""
        if Path(href).suffix.lower() not in self.IMAGE_EXTENSIONS:
            return ""
        if self.image_base_url and self._article_dir:
            return f"{self.image_base_url}/{self._article_dir}/{href}"
        if self.image_base_url:
            return f"{self.image_base_url}/{href}"
        return href

    def _is_non_image_href(self, href: str) -> bool:
        for pattern in self._NON_IMAGE_PATTERNS:
            if pattern.search(href):
                return True
        return "." not in href

    def _is_valid_image_url(self, url: str) -> bool:
        return bool(url) and not url.startswith("#") and url.startswith("http")

    def _get_text(self, node, xpath: str) -> str:
        """Safely extract text from the first node matching an XPath."""
        els = node.xpath(xpath)
        if not els:
            return ""
        el = els[0]
        return el.strip() if isinstance(el, str) else "".join(el.itertext()).strip()

    def _extract_text(self, node) -> str:
        """Extract plain text (no markup) from a node."""
        if node is None:
            return ""
        return "".join(node.itertext())

    def _clean_metadata(self, obj):
        """Recursively remove empty values from metadata structures."""
        if isinstance(obj, dict):
            cleaned = {k: self._clean_metadata(v) for k, v in obj.items()}
            return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
        if isinstance(obj, list):
            cleaned = [self._clean_metadata(i) for i in obj]
            return [i for i in cleaned if i not in (None, "", [], {})]
        return obj



class PMCASTParser(JATSBaseParser):
    """Parse PMC JATS XML into a structure-aware ``Document`` AST."""

    # Structural signals used to recognise a group/category row inside a
    # table (section 17 of the specification). These are deterministic
    # heuristics; no LLM is involved.
    CATEGORY_KEYWORDS = (
        "demographic", "clinical", "laboratory", "measurement",
        "medication", "therapy", "smoking", "alcohol", "history",
        "characteristic", "concomitant", "baseline", "comorbid",
        "prior disease", "cause of",
    )

    METRIC_UNIT_RE = re.compile(
        r"\([^)]*(%|mmhg|mg|ng|ml|kg|cm|mm|mmol|µmol|umol|m2|/l|/g|/d|/m|/dl)"
        r"[^)]*\)",
        re.IGNORECASE,
    )
    N_PERCENT_RE = re.compile(r"\bn\s*\(%\)", re.IGNORECASE)

    # Unicode em-space-like characters used for table row indentation.
    EM_SPACE_CHARS = ("\u2003", "\u2002", "\u2007", "\u2009", "\u200a")

    SECTION_TAGS = frozenset({
        "sec", "ack", "app", "notes", "bio", "fn-group",
    })

    PROSE_TAGS = frozenset({
        "p", "disp-quote", "code", "preformat", "verse-group",
        "speech", "statement", "attrib",
    })

    def parse(self, source: Path) -> Document:
        """Parse a single PMC XML file into a ``Document`` AST."""
        root = etree.parse(
            str(source),
            parser=etree.XMLParser(recover=True),
        ).getroot()

        self._resolve_article_dir(root)

        metadata = self._parse_front_matter(root)
        article_meta = metadata.get("article_meta", {})
        article_ids = article_meta.get("article_ids", {})

        pmcid = article_ids.get("pmcid") or ""
        title = article_meta.get("title") or ""

        if not pmcid:
            # Deterministic fallback (the XML file stem, never "unknown").
            pmcid = source.stem

        if not title:
            title_nodes = root.xpath(
                "//front/article-meta/title-group/article-title"
            )
            if title_nodes:
                title = self._clean_block_text(
                    "".join(title_nodes[0].itertext())
                )

        if not title:
            title = pmcid

        document = Document(
            pmcid=pmcid,
            title=title,
            metadata=metadata,
        )
        document.sections = self._build_document_sections(root)
        return document

    # ==================================================================
    # SECTION ASSEMBLY
    # ==================================================================

    def _build_document_sections(self, root) -> List[Section]:
        sections: List[Section] = []

        graphical = self._build_graphical_abstract_section(root)
        if graphical is not None:
            sections.append(graphical)

        abstract = self._build_abstract_section(root)
        if abstract is not None:
            sections.append(abstract)

        body = root.find(".//body")
        if body is not None:
            sections.extend(self._build_body_sections(body))

        floats = root.find(".//floats-group")
        if floats is not None:
            float_section = self._build_floats_section(floats)
            if float_section is not None:
                sections.append(float_section)

        back = root.find(".//back")
        if back is not None:
            sections.extend(self._build_back_sections(back))

        refs = self._build_references_section(root)
        if refs is not None:
            sections.append(refs)

        return sections

    def _build_body_sections(self, body) -> List[Section]:
        return self._build_loose_section_groups(body, "Body", "content")

    def _build_back_sections(self, back) -> List[Section]:
        return self._build_loose_section_groups(back, "Back Matter", "administrative")

    def _build_loose_section_groups(
        self,
        container,
        fallback_title: str,
        section_type: str,
    ) -> List[Section]:
        """Assemble a container's loose blocks and sub-sections.

        Shared by the document body (``content``) and back matter
        (``administrative``); the two are identical except for the fallback
        title and section classification.
        """
        sections: List[Section] = []
        loose: List[Block] = []

        def flush_loose() -> None:
            if loose:
                sections.append(
                    Section(
                        title=fallback_title,
                        level=1,
                        breadcrumb=[fallback_title],
                        blocks=list(loose),
                        section_type=section_type,
                    )
                )
                loose.clear()

        for child in container:
            tag = self._local_name(child.tag)
            if tag == "ref-list":
                # references are assembled separately
                continue
            if tag in self.SECTION_TAGS:
                flush_loose()
                section = self._build_section(child, 1, [])
                if section is not None:
                    sections.append(section)
                continue
            if tag == "boxed-text":
                flush_loose()
                section = self._build_boxed_text_section(child, 1, [])
                if section is not None:
                    sections.append(section)
                continue

            block = self._build_block(child, [])
            if block is not None:
                loose.append(block)

        flush_loose()
        return sections

    def _build_floats_section(self, floats) -> Optional[Section]:
        blocks: List[Block] = []
        for child in floats:
            tag = self._local_name(child.tag)
            if tag == "fig-group":
                for fig in child.findall("fig"):
                    block = self._build_figure_block(fig)
                    if block is not None:
                        blocks.append(block)
                continue
            block = self._build_block(child, ["Figures and Tables"])
            if block is not None:
                blocks.append(block)
        if not blocks:
            return None
        return Section(
            title="Figures and Tables",
            level=1,
            breadcrumb=["Figures and Tables"],
            blocks=blocks,
            section_type="content",
        )

    def _build_section(
        self,
        node,
        level: int,
        breadcrumb: List[str],
    ) -> Optional[Section]:
        title = self._section_title(node)
        if not title:
            title = self._local_name(node.tag).capitalize()

        full_breadcrumb = list(breadcrumb) + [title]
        section_type = classify_section_title(title)

        blocks, children = self._build_children(node, level, full_breadcrumb)

        return Section(
            title=title,
            level=level,
            breadcrumb=full_breadcrumb,
            blocks=blocks,
            children=children,
            metadata={"section_type": section_type},
            section_type=section_type,
            source_id=node.get("id") or None,
        )

    def _build_boxed_text_section(
        self,
        node,
        level: int,
        breadcrumb: List[str],
    ) -> Optional[Section]:
        title = self._section_title(node)
        if not title:
            label_node = node.find("label")
            if label_node is not None:
                title = self._clean_block_text(
                    "".join(label_node.itertext())
                )
        if not title:
            title = "Key Points"

        full_breadcrumb = list(breadcrumb) + [title]

        child_nodes = list(node)
        blocks: List[Block] = []
        for idx, child in enumerate(child_nodes):
            tag = self._local_name(child.tag)
            if tag in ("title", "label"):
                continue
            if tag == "list":
                label = None
                if idx > 0 and self._local_name(child_nodes[idx - 1].tag) == "p":
                    label = self._detect_list_label(child_nodes[idx - 1])
                    if label is not None and blocks and blocks[-1].block_type == "paragraph":
                        blocks.pop()
                block = self._build_list_block(child, label)
                if block is not None:
                    blocks.append(block)
                continue
            block = self._build_block(child, full_breadcrumb)
            if block is not None:
                blocks.append(block)

        return Section(
            title=title,
            level=level,
            breadcrumb=full_breadcrumb,
            blocks=blocks,
            section_type="content",
            source_id=node.get("id") or None,
        )

    def _build_children(
        self,
        node,
        level: int,
        breadcrumb: List[str],
    ) -> Tuple[List[Block], List[Section]]:
        """Build (blocks, child_sections) for a section node.

        Labelled-list detection happens here so that a short bold/strong
        paragraph immediately preceding a list is attached to the list
        rather than emitted as a standalone paragraph block.
        """
        blocks: List[Block] = []
        children: List[Section] = []
        child_nodes = list(node)

        for idx, child in enumerate(child_nodes):
            tag = self._local_name(child.tag)
            if tag == "title":
                continue

            if tag in self.SECTION_TAGS:
                sub = self._build_section(child, level + 1, breadcrumb)
                if sub is not None:
                    children.append(sub)
                continue

            if tag == "boxed-text":
                sub = self._build_boxed_text_section(child, level + 1, breadcrumb)
                if sub is not None:
                    children.append(sub)
                continue

            if tag == "list":
                label = None
                if idx > 0 and self._local_name(child_nodes[idx - 1].tag) == "p":
                    label = self._detect_list_label(child_nodes[idx - 1])
                    if label is not None and blocks and blocks[-1].block_type == "paragraph":
                        blocks.pop()
                block = self._build_list_block(child, label)
                if block is not None:
                    blocks.append(block)
                continue

            if tag == "fig-group":
                for fig in child.findall("fig"):
                    block = self._build_figure_block(fig)
                    if block is not None:
                        blocks.append(block)
                continue

            block = self._build_block(child, breadcrumb)
            if block is not None:
                blocks.append(block)

        return blocks, children

    # ==================================================================
    # BLOCK BUILDERS
    # ==================================================================

    def _build_block(
        self,
        node,
        breadcrumb: List[str],
    ) -> Optional[Block]:
        tag = self._local_name(node.tag)

        if tag == "p":
            return self._build_paragraph_block(node)
        if tag in self.PROSE_TAGS - {"p"}:
            block = self._build_paragraph_block(node)
            if block is not None and tag != "p":
                block.metadata["original_block_type"] = tag
            return block
        if tag == "table-wrap":
            return self._build_table_block(node)
        if tag == "fig":
            return self._build_figure_block(node)
        if tag == "disp-formula":
            return self._build_equation_block(node)
        if tag == "supplementary-material":
            return self._build_supplementary_block(node)
        if tag == "list":
            return self._build_list_block(node, None)
        if tag in ("media", "graphic", "inline-graphic", "caption", "label"):
            # Bare graphics/captions/labels outside a figure/table are not
            # standalone retrieval objects; they belong to their container.
            return None

        # Unknown / novel block type -> safe fallback. Record the original
        # type so the chunker and validator never lose the information.
        text = self._clean_block_text(self._plain_text(node)[0])
        if not text:
            return None
        return Block(
            block_type=tag,
            content=text,
            metadata={
                "original_block_type": tag,
                "block_id": node.get("id") or None,
                "xml_id": node.get("id") or None,
            },
        )

    def _build_paragraph_block(self, node) -> Optional[Block]:
        text, xrefs = self._plain_text(node)
        text = self._clean_block_text(text)
        if not text:
            return None

        metadata: Dict[str, Any] = {
            "block_id": node.get("id") or None,
            "xml_id": node.get("id") or None,
        }
        if xrefs.get("bibr"):
            metadata["citation_refs"] = xrefs["bibr"]
        if xrefs.get("fn"):
            metadata["footnote_refs"] = xrefs["fn"]
        if xrefs.get("fig"):
            metadata["figure_refs"] = xrefs["fig"]
        if xrefs.get("table"):
            metadata["table_refs"] = xrefs["table"]
        if xrefs.get("supplementary-material"):
            metadata["supplementary_refs"] = xrefs["supplementary-material"]

        return Block("paragraph", text, metadata)

    def _build_list_block(
        self,
        node,
        label: Optional[str],
    ) -> Optional[Block]:
        items: List[str] = []
        for li in node.findall("list-item"):
            item_text = self._clean_block_text(self._plain_text(li)[0])
            if item_text:
                items.append(item_text)

        if items:
            content = "\n".join(f"- {item}" for item in items)
        else:
            content = self._clean_block_text(self._plain_text(node)[0])

        if not content:
            return None

        return Block(
            "list",
            content,
            metadata={
                "block_id": node.get("id") or None,
                "xml_id": node.get("id") or None,
                "list_type": node.get("list-type", "bullet") or "bullet",
                "label": label or "",
            },
        )

    def _build_supplementary_block(self, node) -> Optional[Block]:
        supp_id = node.get("id", "")
        label = (
            self._get_text(node, "label").strip()
            or self._get_text(node, "title").strip()
            or "Supplementary Material"
        )
        caption = self._caption_plain(node)
        media = node.find(".//media")
        if media is None:
            media = node.find(".//graphic")
        href = self._get_href(media) if media is not None else ""
        full_href = self._resolve_href(href) if href else ""

        parts = [label]
        if caption:
            parts.append(caption)
        if full_href:
            parts.append(full_href)
        content = "\n\n".join(parts)

        return Block(
            "administrative",
            content,
            metadata={
                "block_id": supp_id or None,
                "xml_id": supp_id or None,
                "supplementary": True,
                "href": full_href,
                "label": label,
                "caption": caption,
            },
        )

    # ==================================================================
    # LIST LABEL DETECTION
    # ==================================================================

    def _detect_list_label(self, prev_node) -> Optional[str]:
        """Return the preceding paragraph as a list label if it is a short
        bold/strong paragraph immediately preceding a list."""
        if self._local_name(prev_node.tag) != "p":
            return None

        text = self._clean_block_text(self._plain_text(prev_node)[0])
        if not text or len(text) > 200:
            return None

        has_bold = (
            prev_node.find(".//bold") is not None
            or prev_node.find(".//strong") is not None
        )
        ends_colon = text.rstrip().endswith(":")

        if has_bold or ends_colon:
            return text
        return None

    # ==================================================================
    # FIGURES
    # ==================================================================

    def _build_figure_block(self, node) -> Optional[Block]:
        fig_id = node.get("id", "")
        label = self._get_text(node, "label").strip().rstrip(".:")
        caption = self._caption_plain(node)
        image_ref = ""
        panels: List[Dict[str, Any]] = []

        for visual in list(node.iter("graphic")) + list(node.iter("media")):
            parent = visual.getparent()
            nested = False
            while parent is not None and parent != node:
                if self._local_name(parent.tag) in ("fig", "supplementary-material"):
                    nested = True
                    break
                parent = parent.getparent()
            if nested:
                continue

            href = self._get_href(visual)
            full = (
                self._resolve_href(href)
                if self._local_name(visual.tag) == "media"
                else self._resolve_image_href(href)
            )
            if self._is_valid_image_url(full):
                image_ref = full
                break

        for sub in node.findall("fig"):
            panels.append({
                "figure_id": sub.get("id", ""),
                "label": self._get_text(sub, "label").strip(),
            })

        description = ""
        alt = node.find(".//alt-text")
        if alt is not None:
            description = self._clean_block_text("".join(alt.itertext()))

        text_parts: List[str] = []
        if label:
            text_parts.append(label)
        if caption:
            text_parts.append(caption)
        if description:
            text_parts.append(description)
        content = "\n\n".join(text_parts)

        return Block(
            "figure",
            content,
            metadata={
                "figure_id": fig_id or None,
                "block_id": fig_id or None,
                "xml_id": fig_id or None,
                "label": label,
                "caption": caption,
                "image_ref": image_ref,
                "description": description,
                "panels": panels,
            },
        )

    # ==================================================================
    # EQUATIONS
    # ==================================================================

    def _build_equation_block(self, node) -> Optional[Block]:
        latex = self._extract_formula_latex(node)
        eid = node.get("id", "")
        if latex:
            content = latex
        else:
            content = self._clean_block_text("".join(node.itertext()))
        if not content:
            return None
        return Block(
            "formula",
            content,
            metadata={
                "equation_id": eid or None,
                "block_id": eid or None,
                "xml_id": eid or None,
                "latex": latex,
            },
        )

    # ==================================================================
    # TABLES
    # ==================================================================

    def _build_table_block(self, node) -> Optional[Block]:
        table_id = node.get("id", "")
        label = self._get_text(node, "label").strip().rstrip(".:")
        caption = self._caption_plain(node)

        data = self._extract_table_data(node, table_id, label, caption)

        # Faithful fallback content (used only if structured metadata is
        # missing); never raw HTML/XML.
        if label and caption:
            content = f"{label}: {caption}"
        elif label:
            content = label
        else:
            content = caption

        return Block(
            "table",
            content,
            metadata={
                "block_id": table_id or None,
                "xml_id": node.get("id") or None,
                "table": data,
            },
        )

    def _extract_table_data(
        self,
        node,
        table_id: str,
        label: str,
        caption: str,
    ) -> Dict[str, Any]:
        table_node = node.find(".//table")

        if table_node is None:
            # Image-based table (no structured grid available).
            return {
                "table_id": table_id,
                "label": label,
                "caption": caption,
                "columns": [],
                "categories": [],
                "rows": [],
                "footnotes": self._extract_table_footnotes(node),
                "structure_degraded": True,
                "source_block_id": table_id or None,
            }

        # A <table> element whose cells contain only graphics (and no text)
        # is an image-based table. Preserve the image references and mark
        # the structure degraded rather than emitting empty data rows.
        text_content = "".join(table_node.itertext()).strip()
        image_refs: List[str] = []
        for g in (
            list(table_node.findall(".//graphic"))
            + list(table_node.findall(".//media"))
            + list(table_node.findall(".//inline-graphic"))
        ):
            href = self._get_href(g)
            full = (
                self._resolve_href(href)
                if self._local_name(g.tag) == "media"
                else self._resolve_image_href(href)
            )
            if self._is_valid_image_url(full):
                image_refs.append(full)

        if not text_content and image_refs:
            return {
                "table_id": table_id,
                "label": label,
                "caption": caption,
                "columns": [],
                "categories": [],
                "rows": [],
                "footnotes": self._extract_table_footnotes(node),
                "structure_degraded": True,
                "images": image_refs,
                "source_block_id": table_id or None,
            }

        columns, header_degraded = self._extract_table_columns(table_node)
        rows, categories, rows_degraded = self._extract_table_body_rows(
            table_node, columns
        )
        footnotes = self._extract_table_footnotes(node)

        source_block_id = table_id or None
        for row in rows:
            row["source_block_ids"] = [source_block_id] if source_block_id else []

        return {
            "table_id": table_id,
            "label": label,
            "caption": caption,
            "columns": columns,
            "categories": categories,
            "rows": rows,
            "footnotes": footnotes,
            "structure_degraded": header_degraded or rows_degraded,
            "source_block_id": source_block_id,
        }

    def _extract_table_columns(
        self,
        table_node,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        header_rows: List[List[Dict[str, Any]]] = []

        for child in table_node:
            if self._local_name(child.tag) != "thead":
                continue
            for tr in child:
                if self._local_name(tr.tag) != "tr":
                    continue
                row: List[Dict[str, Any]] = []
                for cell in tr:
                    tag = self._local_name(cell.tag)
                    if tag not in ("th", "td"):
                        continue
                    row.append({
                        "text": self._normalize_cell_text(
                            self._plain_text(cell)[0]
                        ),
                        "colspan": int(cell.get("colspan", 1) or 1),
                        "rowspan": int(cell.get("rowspan", 1) or 1),
                    })
                header_rows.append(row)
            break

        if not header_rows:
            return [], True

        grid, max_cols = self._expand_cells(header_rows)
        columns: List[Dict[str, Any]] = []
        for c in range(max_cols):
            names: List[str] = []
            for r in range(len(grid)):
                cell = grid[r][c]
                text = (cell["text"] if cell else "").strip()
                if text and text not in names:
                    names.append(text)
            name = " - ".join(names) if names else f"column_{c}"
            columns.append({"index": c, "name": name, "unit": ""})

        return columns, False

    def _extract_table_body_rows(
        self,
        table_node,
        columns: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str], bool]:
        tr_nodes = table_node.xpath("./tbody/tr")
        if not tr_nodes:
            tr_nodes = [
                c for c in table_node
                if self._local_name(c.tag) == "tr"
            ]
        if not tr_nodes:
            return [], [], True

        raw_rows: List[List[Dict[str, Any]]] = []
        for tr in tr_nodes:
            row: List[Dict[str, Any]] = []
            for cell in tr:
                tag = self._local_name(cell.tag)
                if tag not in ("th", "td"):
                    continue
                row.append({
                    "text": self._normalize_cell_text(
                        self._plain_text(cell)[0]
                    ),
                    "colspan": int(cell.get("colspan", 1) or 1),
                    "rowspan": int(cell.get("rowspan", 1) or 1),
                })
            raw_rows.append(row)

        grid, max_cols = self._expand_cells(raw_rows)

        degraded = False
        if not columns:
            columns = [
                {"index": c, "name": f"column_{c}", "unit": ""}
                for c in range(max_cols)
            ]
            degraded = True

        rows: List[Dict[str, Any]] = []
        categories: List[str] = []
        category: Optional[str] = None
        variable: Optional[str] = None
        last_data_label_at_indent: Dict[int, str] = {}

        for r in range(len(grid)):
            cells = grid[r]
            first_norm = self._cell_text(cells, 0)
            indent = self._leading_indent(first_norm)
            first = first_norm.strip()

            other_values = [
                self._cell_text(cells, c) for c in range(1, max_cols)
            ]
            is_group = bool(first) and all(v == "" for v in other_values)

            if is_group:
                kind = self._classify_group_header(first)
                if kind == "category":
                    category = first
                    variable = None
                    last_data_label_at_indent = {}
                else:
                    variable = first
                if first not in categories:
                    categories.append(first)
                continue

            if indent == 0:
                # Direct child of the current category; clear any lingering
                # sub-variable header.
                variable = None

            group_path = [g for g in (category, variable) if g]
            if indent >= 2 and (indent - 1) in last_data_label_at_indent:
                group_path = group_path + [last_data_label_at_indent[indent - 1]]

            values: Dict[str, str] = {}
            for c in range(1, max_cols):
                col_name = (
                    columns[c]["name"] if c < len(columns) else f"column_{c}"
                )
                values[col_name] = self._cell_text(cells, c)

            # Skip fully-empty separator rows (no content is dropped: the
            # row has no text). Mark the table degraded so validation knows
            # the structure could not be fully normalized.
            if not first and all(v == "" for v in values.values()):
                degraded = True
                continue

            ordered_cells = [first]
            for c in range(1, max_cols):
                col_name = (
                    columns[c]["name"] if c < len(columns) else f"column_{c}"
                )
                ordered_cells.append(values.get(col_name, ""))

            rows.append({
                "row_label": first,
                "group_path": group_path,
                "values": values,
                "cells": ordered_cells,
                "source_id": None,
                "is_group_row": False,
            })
            last_data_label_at_indent[indent] = first

        # Raw rows existed but produced no data rows (e.g. a table used for
        # layout or an algorithm box). Mark degraded rather than pretending
        # the structure is a normal data table.
        if raw_rows and not rows:
            degraded = True

        return rows, categories, degraded

    def _extract_table_footnotes(self, node) -> List[Dict[str, str]]:
        footnotes: List[Dict[str, str]] = []
        for foot in node.findall(".//table-wrap-foot"):
            for child in foot:
                tag = self._local_name(child.tag)
                if tag == "fn":
                    label_node = child.find("label")
                    marker = (
                        self._clean_block_text("".join(label_node.itertext()))
                        if label_node is not None
                        else ""
                    )
                    p_nodes = child.findall("p")
                    if p_nodes:
                        text = " ".join(
                            self._clean_block_text(self._plain_text(p)[0])
                            for p in p_nodes
                        )
                    else:
                        full = self._plain_text(child)[0]
                        text = self._clean_block_text(full)
                        if marker and text.startswith(marker):
                            text = text[len(marker):].strip()
                    footnotes.append({"marker": marker, "text": text})
                elif tag == "p":
                    footnotes.append({
                        "marker": "",
                        "text": self._clean_block_text(self._plain_text(child)[0]),
                    })
                elif tag == "attrib":
                    footnotes.append({
                        "marker": "",
                        "text": self._clean_block_text(self._plain_text(child)[0]),
                    })
        return footnotes

    # ==================================================================
    # TABLE HELPERS
    # ==================================================================

    def _expand_cells(self, raw_rows):
        """Expand colspan/rowspan into a 2D grid.

        Returns ``(grid, max_cols)`` where ``grid[r][c]`` is a cell dict or
        ``None`` for unfilled gaps.
        """
        if not raw_rows:
            return [], 0

        max_cols = max(
            sum(c["colspan"] for c in row) for row in raw_rows
        )
        n_rows = len(raw_rows)
        grid: List[List[Optional[Dict[str, Any]]]] = [
            [None] * max_cols for _ in range(n_rows)
        ]

        for r, row in enumerate(raw_rows):
            c = 0
            for cell in row:
                while c < max_cols and grid[r][c] is not None:
                    c += 1
                if c >= max_cols:
                    break
                for i in range(cell["rowspan"]):
                    for j in range(cell["colspan"]):
                        rr, cc = r + i, c + j
                        if rr < n_rows and cc < max_cols:
                            grid[rr][cc] = cell
                c += cell["colspan"]

        return grid, max_cols

    def _cell_text(self, cells, c: int) -> str:
        if c >= len(cells) or cells[c] is None:
            return ""
        return cells[c]["text"] if cells[c]["text"] else ""

    def _leading_indent(self, text: str) -> int:
        count = 0
        for ch in text:
            if ch == "\u2003":
                count += 1
            else:
                break
        return count

    def _normalize_cell_text(self, raw: str) -> str:
        """Normalize a table cell while preserving leading em-space indent.

        Empty cells (``&nbsp;`` / whitespace only) become ``""``. Data cells
        keep their numeric/unit content verbatim (no value rewriting).
        """
        em = 0
        i = 0
        while i < len(raw) and raw[i] in self.EM_SPACE_CHARS:
            em += 1
            i += 1
        # Skip residual leading regular whitespace after the em indentation.
        while i < len(raw) and raw[i] in ("\u00a0", " ", "\t", "\u2009"):
            i += 1

        body = raw[i:].replace("\u00a0", " ")
        body = " ".join(body.split())
        return ("\u2003" * em) + body

    def _classify_group_header(self, label: str) -> str:
        """Classify a table group row as ``"category"`` or ``"variable"``.

        Deterministic heuristic (section 17). Category keywords take
        precedence; otherwise metric/unit markers indicate a variable
        sub-header. Unrecognised headers default to ``"category"`` (safe
        top-level fallback).
        """
        low = label.lower()
        if any(keyword in low for keyword in self.CATEGORY_KEYWORDS):
            return "category"
        if self.N_PERCENT_RE.search(low) or self.METRIC_UNIT_RE.search(low):
            return "variable"
        return "category"

    # ==================================================================
    # REFERENCES
    # ==================================================================

    def _build_references_section(self, root) -> Optional[Section]:
        ref_lists = root.xpath("//ref-list[not(ancestor::sub-article)]")
        if not ref_lists:
            return None

        blocks: List[Block] = []
        seen_ids = set()
        counter = 1
        for rl in ref_lists:
            for ref in rl.findall(".//ref"):
                ref_id = ref.get("id", "")
                if ref_id and ref_id in seen_ids:
                    continue
                if ref_id:
                    seen_ids.add(ref_id)

                citation_node = ref.find(".//mixed-citation")
                if citation_node is None:
                    citation_node = ref.find(".//element-citation")

                if citation_node is not None:
                    text = self._format_reference_text(citation_node)
                    if not text.strip():
                        text = self._clean_block_text(
                            "".join(citation_node.itertext())
                        )
                else:
                    text = self._clean_block_text("".join(ref.itertext()))

                if not text.strip():
                    text = f"Reference {counter}"

                doi, pmid, pmcid = (
                    self._extract_reference_ids(citation_node)
                    if citation_node is not None
                    else ("", "", "")
                )

                blocks.append(Block(
                    "reference",
                    text,
                    metadata={
                        "reference_id": ref_id or f"ref_{counter}",
                        "id": ref_id or None,
                        "xml_id": ref_id or None,
                        "citation_number": counter,
                        "doi": doi,
                        "pmid": pmid,
                        "pmcid": pmcid,
                    },
                ))
                counter += 1

        if not blocks:
            return None

        return Section(
            title="References",
            level=1,
            breadcrumb=["References"],
            blocks=blocks,
            section_type="references",
        )

    # ==================================================================
    # ABSTRACT / GRAPHICAL ABSTRACT
    # ==================================================================

    def _build_graphical_abstract_section(self, root) -> Optional[Section]:
        gas = root.xpath(
            "//front//abstract[@abstract-type='graphical'][not(ancestor::sub-article)]"
        )
        blocks: List[Block] = []
        for ga in gas:
            # A graphical abstract is a figure/visual object. Find the
            # figure element(s) wherever they sit inside the abstract.
            figs = list(ga.iter("fig"))
            if figs:
                for fig in figs:
                    block = self._build_figure_block(fig)
                    if block is not None:
                        blocks.append(block)
                continue

            # Fallback: bare graphic/media with no enclosing <fig>.
            for g in list(ga.iter("graphic")) + list(ga.iter("media")):
                href = self._get_href(g)
                full = (
                    self._resolve_href(href)
                    if self._local_name(g.tag) == "media"
                    else self._resolve_image_href(href)
                )
                blocks.append(Block(
                    "figure",
                    "Graphical Abstract",
                    metadata={
                        "figure_id": g.get("id") or "graphical_abstract",
                        "label": "Graphical Abstract",
                        "image_ref": full if self._is_valid_image_url(full) else "",
                    },
                ))

        if not blocks:
            return None
        return Section(
            title="Graphical Abstract",
            level=1,
            breadcrumb=["Graphical Abstract"],
            blocks=blocks,
            section_type="content",
        )

    def _build_abstract_section(self, root) -> Optional[Section]:
        nodes = root.xpath(
            "//abstract[not(@abstract-type='graphical') and not(ancestor::sub-article)] "
            "| //trans-abstract[not(ancestor::sub-article)]"
        )
        if not nodes:
            nodes = root.xpath(
                "//abstract[not(@abstract-type='graphical') and not(ancestor::sub-article)]"
            )
        if not nodes:
            return None

        primary_idx = self._pick_primary_abstract(nodes)

        def build_blocks_and_children(
            abs_node,
            crumb: List[str],
            level: int,
        ) -> Tuple[List[Block], List[Section]]:
            blocks: List[Block] = []
            children: List[Section] = []
            for child in abs_node:
                tag = self._local_name(child.tag)
                if tag == "title":
                    continue
                if tag == "sec":
                    sub = self._build_section(child, level, crumb)
                    if sub is not None:
                        children.append(sub)
                    continue
                block = self._build_block(child, crumb)
                if block is not None:
                    blocks.append(block)
            if not blocks and not children:
                text = self._clean_block_text("".join(abs_node.itertext()))
                if text and text.lower() != "abstract":
                    blocks.append(Block("paragraph", text))
            return blocks, children

        primary = nodes[primary_idx]
        primary_blocks, primary_children = build_blocks_and_children(
            primary, ["Abstract"], 2
        )

        section = Section(
            title="Abstract",
            level=1,
            breadcrumb=["Abstract"],
            blocks=primary_blocks,
            children=primary_children,
            section_type="content",
        )

        for i, absn in enumerate(nodes):
            if i == primary_idx:
                continue
            heading = self._abstract_heading(absn)
            sub_blocks, sub_children = build_blocks_and_children(
                absn, ["Abstract", heading], 3
            )
            section.children.append(Section(
                title=heading,
                level=2,
                breadcrumb=["Abstract", heading],
                blocks=sub_blocks,
                children=sub_children,
                section_type="content",
            ))

        return section

    # ==================================================================
    # TEXT HELPERS
    # ==================================================================

    def _section_title(self, node) -> str:
        title_node = node.find("title")
        if title_node is not None:
            text = self._clean_block_text(self._plain_text(title_node)[0])
            if text:
                return text
        label_node = node.find("label")
        if label_node is not None:
            text = self._clean_block_text("".join(label_node.itertext()))
            if text:
                return text
        return ""

    def _caption_plain(self, node) -> str:
        cap = node.find("caption")
        if cap is None:
            return ""
        parts: List[str] = []
        if cap.text:
            parts.append(cap.text)
        for child in cap:
            tag = self._local_name(child.tag)
            if tag in ("fig", "table-wrap", "supplementary-material"):
                continue
            parts.append(self._plain_text(child)[0])
            if child.tail:
                parts.append(child.tail)
        return self._clean_block_text("".join(parts))

    def _plain_text(self, node, xrefs=None) -> Tuple[str, Dict[str, List[str]]]:
        """Extract plain text (no Markdown) while collecting cross references.

        Returns ``(text, xrefs)``. ``xrefs`` maps ref-type -> list of rids.
        """
        if xrefs is None:
            xrefs = {
                "bibr": [], "fig": [], "table": [], "fn": [],
                "supplementary-material": [], "disp-formula": [],
            }

        parts: List[str] = []
        if node.text:
            parts.append(node.text)

        for child in node:
            tag = self._local_name(child.tag)

            if tag == "xref":
                rid = child.get("rid", "")
                ref_type = child.get("ref-type", "")
                for r in rid.split():
                    r = r.lstrip("#")
                    if r and r not in xrefs.get(ref_type, []):
                        xrefs.setdefault(ref_type, []).append(r)
                parts.append("".join(child.itertext()))

            elif tag in ("media", "graphic", "inline-graphic"):
                pass

            elif tag in ("inline-formula", "disp-formula"):
                latex = self._extract_formula_latex(child)
                if latex:
                    parts.append(f"${latex}$")
                else:
                    parts.append("".join(child.itertext()))

            elif tag in ("table-wrap", "table", "thead", "tbody", "tfoot",
                         "tr", "th", "td", "col", "colgroup"):
                pass

            elif tag == "list":
                items = []
                for li in child.findall("list-item"):
                    items.append(self._plain_text(li, xrefs)[0])
                parts.append("; ".join(i for i in items if i))

            else:
                parts.append(self._plain_text(child, xrefs)[0])

            if child.tail:
                parts.append(child.tail)

        return "".join(parts), xrefs

    def _clean_block_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\u00a0", " ")
        return " ".join(text.split())

