import re
import yaml
from lxml import etree
from pathlib import Path
from typing import Any, Dict, List, Set


class PMCParser:
    """
    PMC JATS XML to Markdown parser.
    NOTE: Tested only on Cardiology 2025 data. Might contain errors
    This parser converts complex, nested JATS XML into clean, LLM-friendly
    Markdown with YAML frontmatter. It operates in four distinct phases:
    1. Initialization: Parse the XML tree and build a global registry of element IDs.
    2. Metadata Extraction: Pull journal, author, and article data into YAML frontmatter.
    3. AST Walking: Recursively traverse the XML body using structural pattern-matching.
    4. Post-Processing: Append orphaned images and generate missing HTML anchors.
    """

    PMC_OA_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com"

    IMAGE_EXTENSIONS = frozenset({
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".tif", ".tiff", ".bmp", ".eps",
    })

    LINKABLE_REF_TYPES = frozenset({
        "bibr", "fig", "table", "fn", "table-fn",
        "supplementary-material", "disp-formula", "app",
    })

    _NON_IMAGE_PATTERNS = [
        re.compile(r'^G[SEPL]\d+$'),
        re.compile(r'^[NX][MR]_\d+(\.\d+)?$'),
        re.compile(r'^[A-Z]{2}\d{6,}(\.\d+)?$'),
        re.compile(r'^10\.\d{4,}/'),
        re.compile(r'^[A-Z]\d{5}$'),
        re.compile(r'^[A-Z]{2,}\d{4,}$'),
        re.compile(r'^www\.'),
        re.compile(r'biorender', re.IGNORECASE),
        re.compile(r'^Movies?\s', re.IGNORECASE),
    ]

    def __init__(self, image_base_url: str = PMC_OA_BASE_URL):
        self.image_base_url = image_base_url.rstrip("/") if image_base_url else ""
        self._article_dir = ""
        self._article_doi = ""
        self._known_ids: Set[str] = set()
        self._id_to_elem: Dict[str, Any] = {}

    def parse(self, source: Path) -> str:
        """Main entry point. Orchestrates the multi-phase conversion process."""
        root = etree.parse(str(source), parser=etree.XMLParser(recover=True)).getroot()
        self._resolve_article_dir(root)
        self._known_ids = self._collect_all_ids(root)

        title_nodes = root.xpath("//front/article-meta/title-group/article-title")
        title = self._extract_text(title_nodes[0], 2).strip() if title_nodes else ""

        metadata = self._parse_front_matter(root)
        graphical_abstract = self._parse_graphical_abstract_md(root)
        abstract = self._parse_abstract_md(root)
        body = self._parse_body_md(root)
        floats = self._parse_floats_group_md(root)
        back = self._parse_back_md(root)
        references = self._parse_references_md(root)

        md_content = self._assemble_markdown(
            title, metadata, graphical_abstract, abstract,
            body, floats, back, references,
        )

        md_content = self._generate_missing_anchors(md_content)
        md_content = self._append_missing_images(root, md_content)

        return md_content

    def _assemble_markdown(self, title, metadata, graphical_abstract, abstract,
                           body, floats, back, references) -> str:
        """Combine YAML frontmatter with the rendered Markdown sections."""
        yaml_str = yaml.dump(metadata, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000)
        parts = [f"---\n{yaml_str}---"]
        if title: parts.append(f"# {title}")
        if graphical_abstract: parts.append(graphical_abstract)
        if abstract: parts.append(abstract)
        if body: parts.append(body)
        if floats: parts.append(floats)
        if back: parts.append(back)
        if references: parts.append(references)
        return "\n\n".join(parts)

    def _collect_all_ids(self, root) -> Set[str]:
        """Build a lookup dictionary of all IDs in the document."""
        ids: Set[str] = set()
        self._id_to_elem = {}
        for elem in root.iter():
            eid = elem.get("id")
            if eid:
                ids.add(eid)
                self._id_to_elem[eid] = elem
        return ids

    def _resolve_article_dir(self, root):
        """Determine the article directory based on PMC ID."""
        ids = {aid.get("pub-id-type"): "".join(aid.itertext()).strip() for aid in root.xpath("//front/article-meta/article-id")}
        self._article_dir = ids.get("pmcid-ver") or ids.get("pmcid") or ""
        self._article_doi = ids.get("doi") or ""

    def _parse_front_matter(self, root) -> Dict[str, Any]:
        """Parse journal, article, and supplementary metadata."""
        meta = {
            "journal_meta": self._parse_journal_meta(root),
            "article_meta": self._parse_article_meta(root),
            "supplementary_assets": self._parse_supplementary_assets(root),
        }
        return self._clean_metadata(meta)

    def _parse_journal_meta(self, root) -> Dict[str, Any]:
        """Extract journal-level metadata."""
        j_meta = root.xpath("//front/journal-meta")
        if not j_meta: return {}
        j_meta = j_meta[0]
        return {
            "journal_ids": {jid.get("journal-id-type"): "".join(jid.itertext()).strip() for jid in j_meta.xpath("./journal-id") if jid.get("journal-id-type")},
            "title": self._get_text(j_meta, ".//journal-title"),
            "issns": {issn.get("pub-type"): "".join(issn.itertext()).strip() for issn in j_meta.xpath("./issn") if issn.get("pub-type")},
            "publisher": {"name": self._get_text(j_meta, ".//publisher-name"), "loc": self._get_text(j_meta, ".//publisher-loc")},
        }

    def _parse_article_meta(self, root) -> Dict[str, Any]:
        """Extract article-level metadata."""
        a_meta = root.xpath("//front/article-meta")
        if not a_meta: return {}
        a_meta = a_meta[0]
        return {
            "article_ids": {aid.get("pub-id-type"): "".join(aid.itertext()).strip() for aid in a_meta.xpath("./article-id") if aid.get("pub-id-type")},
            "categories": [subj.text.strip() for sg in a_meta.xpath("./article-categories/subj-group") for subj in sg.xpath("./subject") if subj.text],
            "title": self._get_text(a_meta, ".//article-title"),
            "subtitle": self._get_text(a_meta, ".//subtitle"),
            "authors": self._parse_contributors(a_meta),
            "author_notes": self._parse_author_notes(a_meta),
            "publication_dates": self._parse_pub_dates(a_meta),
            "history": self._parse_history(a_meta),
            "pub_history": self._parse_pub_history(a_meta),
            "issue_info": self._parse_issue_info(a_meta),
            "permissions": self._parse_permissions(a_meta),
            "keywords": [kw.text.strip() for kw in a_meta.xpath(".//kwd-group/kwd") if kw.text],
            "funding": self._parse_funding(a_meta),
            "counts": {cnt.tag: cnt.get("count") for cnt in a_meta.xpath("./counts/*")},
            "custom_meta": self._parse_custom_meta(a_meta),
        }

    def _parse_custom_meta(self, node) -> Dict[str, str]:
        """Extract custom metadata key-value pairs."""
        return {m.findtext("meta-name"): m.findtext("meta-value") for m in node.xpath(".//custom-meta-group/custom-meta") if m.findtext("meta-name") and m.findtext("meta-value")}

    def _parse_supplementary_assets(self, root) -> List[Dict[str, str]]:
        """Extract supplementary material assets for the frontmatter."""
        assets = []
        for supp in root.xpath("//supplementary-material"):
            asset: Dict[str, str] = {}

            label_node = supp.find("label")
            if label_node is None:
                label_node = supp.find("title")

            label = " ".join(self._extract_text(label_node, 2).split()) if label_node is not None else ""
            caption_node = supp.find("caption")
            caption_text = " ".join(self._extract_text(caption_node, 2).split()) if caption_node is not None else ""

            if not label and caption_text:
                label = caption_text.split('.')[0] if '.' in caption_text else caption_text[:60]
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
            if asset:
                assets.append(asset)
        return assets

    def _parse_contributors(self, a_meta) -> List[Dict[str, Any]]:
        """Parse all contributors from the article metadata."""
        return [self._parse_single_contrib(c) for c in a_meta.xpath("./contrib-group/contrib")]

    def _parse_single_contrib(self, contrib) -> Dict[str, Any]:
        """Parse a single contributor's details."""
        author: Dict[str, Any] = {cid.get("contrib-id-type", "unknown"): "".join(cid.itertext()).strip() for cid in contrib.xpath("./contrib-id")}

        name_node = contrib.find("./name")
        if name_node is not None:
            author["given_names"] = self._get_text(name_node, "given-names")
            author["surname"] = self._get_text(name_node, "surname")
            prefix = self._get_text(name_node, "prefix")
            suffix = self._get_text(name_node, "suffix")
            author["name"] = f"{prefix} {author.get('given_names', '')} {author.get('surname', '')} {suffix}".replace("  ", " ").strip()

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
            if inst is not None: aff_data["institution"] = "".join(inst.itertext()).strip()
            addr = aff.find(".//addr-line")
            if addr is not None: aff_data["city"] = "".join(addr.itertext()).strip()
            country = aff.find(".//country")
            if country is not None and country.text: aff_data["country"] = country.text.strip()
            if aff_data: affs.append(aff_data)
        if affs: author["affiliations"] = affs

        if contrib.find("./xref[@ref-type='corresp']") is not None:
            author["is_corresponding"] = True

        note_refs = [x.get("rid") for x in contrib.xpath("./xref[@ref-type='author-notes']") if x.get("rid")]
        if note_refs: author["note_refs"] = note_refs
        return author

    def _parse_author_notes(self, a_meta) -> Dict[str, Any]:
        """Extract author notes and correspondence details."""
        notes: Dict[str, Any] = {}
        corresp = a_meta.find(".//author-notes/corresp")
        if corresp is not None: notes["correspondence"] = "".join(corresp.itertext()).strip()
        fns = ["".join(fn.itertext()).strip() for fn in a_meta.xpath(".//author-notes/fn/p")]
        if fns: notes["footnotes"] = fns
        return notes

    def _format_date(self, node) -> str:
        """Format an XML date node into an ISO-8601 string."""
        if node is None: return ""
        iso = node.get("iso-8601-date")
        if iso: return iso
        y = node.findtext("year")
        if not y: return ""
        parts = [y]
        m = node.findtext("month")
        d = node.findtext("day")
        if m: parts.append(m.zfill(2))
        if d: parts.append(d.zfill(2))
        return "-".join(parts)

    def _parse_pub_dates(self, a_meta) -> Dict[str, str]:
        """Extract publication dates."""
        return {pd.get("pub-type", "unknown"): self._format_date(pd) for pd in a_meta.xpath("./pub-date") if self._format_date(pd)}

    def _parse_history(self, a_meta) -> Dict[str, str]:
        """Extract manuscript history dates."""
        return {hd.get("date-type", "unknown"): self._format_date(hd) for hd in a_meta.xpath("./history/date") if self._format_date(hd)}

    def _parse_pub_history(self, a_meta) -> List[Dict[str, Any]]:
        """Extract detailed publication history events."""
        events = []
        for event in a_meta.xpath("./pub-history/event"):
            ev_data: Dict[str, Any] = {"event_type": event.get("event-type", "unknown")}
            date_node = event.find("./date")
            if date_node is None:
                date_node = event.find("./pub-date")
            if date_node is not None: ev_data["date"] = self._format_date(date_node)
            art_ids = {aid.get("pub-id-type"): "".join(aid.itertext()).strip() for aid in event.xpath("./article-id") if aid.get("pub-id-type")}
            if art_ids: ev_data["article_ids"] = art_ids
            version = event.findtext("./article-version")
            if version: ev_data["version"] = version.strip()
            events.append(ev_data)
        return events

    def _parse_issue_info(self, a_meta) -> Dict[str, str]:
        """Extract volume, issue, and pagination info."""
        return {tag: a_meta.findtext(f"./{tag}").strip() for tag in ["volume", "issue", "fpage", "lpage", "elocation-id"] if a_meta.findtext(f"./{tag}")}

    def _parse_permissions(self, a_meta) -> Dict[str, Any]:
        """Extract copyright and license information."""
        perms: Dict[str, Any] = {}
        stmt = a_meta.find(".//copyright-statement")
        if stmt is not None: perms["copyright_statement"] = "".join(stmt.itertext()).strip()
        year = a_meta.findtext(".//copyright-year")
        if year: perms["copyright_year"] = year.strip()
        license_node = a_meta.find(".//license")
        if license_node is not None:
            lic_url, lic_text = None, None
            for el in license_node.iter():
                tag = self._local_name(el.tag)
                if tag == "license_ref": lic_url = "".join(el.itertext()).strip()
                elif tag == "ext-link":
                    href = self._get_href(el)
                    if href and not lic_url: lic_url = href
                elif tag == "license-p": lic_text = "".join(el.itertext()).strip()
            if lic_url: perms["license_url"] = lic_url
            if lic_text: perms["license_text"] = lic_text
        return perms

    def _parse_funding(self, a_meta) -> List[Dict[str, Any]]:
        """Extract funding and award information."""
        funding = []
        for award in a_meta.xpath("./funding-group/award-group"):
            aw: Dict[str, Any] = {}
            src = award.find(".//funding-source//institution")
            if src is not None: aw["source"] = "".join(src.itertext()).strip()
            src_id = award.find(".//funding-source//institution-id")
            if src_id is not None: aw["source_id"] = "".join(src_id.itertext()).strip()
            ids = [aid.text.strip() for aid in award.xpath("./award-id") if aid.text]
            if ids: aw["award_ids"] = ids
            if aw: funding.append(aw)
        return funding

    def _parse_graphical_abstract_md(self, root) -> str:
        """Render the graphical abstract section."""
        graphical_abs = root.xpath("//front/article-meta/abstract[@abstract-type='graphical']")
        if not graphical_abs: return ""
        md_parts = ["## Graphical Abstract"]
        for child in graphical_abs[0]:
            md_parts.append(self._format_node(child, depth=3))
        return "\n\n".join([p for p in md_parts if p])

    def _parse_abstract_md(self, root) -> str:
        """Render the main and translated abstracts."""
        nodes = root.xpath("//front//abstract[not(@abstract-type='graphical') and not(ancestor::sub-article)] | //front//trans-abstract[not(ancestor::sub-article)]")
        if not nodes:
            nodes = root.xpath("//abstract[not(@abstract-type='graphical') and not(ancestor::sub-article)] | //trans-abstract[not(ancestor::sub-article)]")
        if not nodes: return ""

        primary_idx = self._pick_primary_abstract(nodes)
        sections = []
        for i, abs_node in enumerate(nodes):
            heading = "Abstract" if i == primary_idx else self._abstract_heading(abs_node)
            parts = [f"## {heading}"]
            for child in abs_node:
                if self._local_name(child.tag) == "title": continue
                # Use depth=3 so that <sec> tags inside the abstract render as ### (e.g., ### Background)
                # instead of ##, preventing them from being parsed as top-level paper sections.
                rendered = self._format_node(child, depth=3)
                if rendered.strip(): parts.append(rendered)
            if len(parts) == 1:
                text = self._extract_text(abs_node, depth=2).strip()
                if text and text.lower() != "abstract": parts.append(f"\n\n{text}\n\n")
            if len(parts) > 1: sections.append("\n\n".join([p for p in parts if p]))
        return "\n\n".join(sections)

    def _pick_primary_abstract(self, nodes) -> int:
        """Identify the primary abstract when multiple are present."""
        for i, node in enumerate(nodes):
            if (node.get("abstract-type") or "") in ("", "abstract") and self._abstract_has_content(node): return i
        for i, node in enumerate(nodes):
            if self._abstract_has_content(node): return i
        return 0

    def _abstract_has_content(self, abs_node) -> bool:
        """Check if an abstract node contains actual text or structural content."""
        for child in abs_node:
            if self._local_name(child.tag) != "title": return True
        text = "".join(abs_node.itertext()).strip()
        return bool(text) and text.lower() != "abstract"

    def _abstract_heading(self, abs_node) -> str:
        """Generate a heading for secondary or translated abstracts."""
        tag = self._local_name(abs_node.tag)
        if tag == "trans-abstract": return "Translated Abstract"
        at = (abs_node.get("abstract-type") or "").strip()
        return at.replace("-", " ").title() if at else "Abstract"

    def _parse_body_md(self, root) -> str:
        """Render the main article body."""
        body = root.find(".//body")
        if body is None: return ""
        return "\n\n".join([p for p in [self._format_node(c, 2) for c in body] if p])

    def _parse_floats_group_md(self, root) -> str:
        """Render the floats group (figures/tables outside main body)."""
        floats_group = root.find(".//floats-group")
        if floats_group is None: return ""
        return "\n\n".join([p for p in [self._format_node(c, 2) for c in floats_group] if p.strip()])

    def _parse_back_md(self, root) -> str:
        """Render the back matter (acknowledgments, appendices, etc.)."""
        back = root.find(".//back")
        if back is None: return ""
        return "\n\n".join([p for p in [self._format_node(c, 2) for c in back if self._local_name(c.tag) != "ref-list"] if p])

    def _parse_references_md(self, root) -> str:
        """Merge multiple reference lists into a single sequentially numbered section."""
        ref_lists = root.xpath("//ref-list[not(ancestor::sub-article)]")
        if not ref_lists: return ""

        refs = []
        seen_ids: Set[str] = set()
        for rl in ref_lists:
            for ref in rl.findall(".//ref"):
                ref_id = ref.get("id", "")
                if ref_id and ref_id in seen_ids: continue
                if ref_id: seen_ids.add(ref_id)
                refs.append(ref)

        if not refs: return ""

        md_parts = ["## References"]
        counter = 1
        for ref in refs:
            ref_id = ref.get("id", "")
            citation_node = ref.find(".//mixed-citation")
            if citation_node is None:
                citation_node = ref.find(".//element-citation")

            if citation_node is not None:
                text = self._format_reference_text(citation_node)
                if not text.strip(): text = " ".join(self._extract_text(citation_node, 2).split())
            else:
                text = " ".join(self._extract_text(ref, 2).split())

            if not text.strip(): text = f"Reference {counter}"
            md_parts.append(f"<a id=\"{ref_id}\"></a>\n{counter}. {text}")
            counter += 1
        return "\n\n".join(md_parts)

    def _format_reference_text(self, citation_node) -> str:
        """Format a single citation node into a readable string."""
        authors = [" ".join("".join(p.itertext()).split()) for p in citation_node.xpath(".//person-group/string-name")]
        author_str = ", ".join([a for a in authors if a])
        title = " ".join(self._get_text(citation_node, "article-title").split())
        source = " ".join(self._get_text(citation_node, "source").split())
        year = " ".join(self._get_text(citation_node, "year").split())
        volume = " ".join(self._get_text(citation_node, "volume").split())
        fpage = " ".join(self._get_text(citation_node, "fpage").split())
        lpage = " ".join(self._get_text(citation_node, "lpage").split())

        doi = pmid = pmcid = ""
        for pub_id in citation_node.xpath(".//pub-id"):
            id_type = pub_id.get("pub-id-type")
            val = " ".join("".join(pub_id.itertext()).split())
            if id_type == "doi": doi = val
            elif id_type == "pmid": pmid = val
            elif id_type == "pmcid": pmcid = val

        parts = [p for p in [author_str, title] if p]
        journal_info = [p for p in [source, year, f";{volume}" if volume else "", f":{fpage}–{lpage}" if fpage and lpage else f":{fpage}" if fpage else ""] if p]
        if journal_info: parts.append("".join(journal_info))

        id_parts = []
        if doi: id_parts.append(f"DOI: [{doi}](https://doi.org/{doi})")
        if pmid: id_parts.append(f"PMID: [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})")
        if pmcid: id_parts.append(f"PMCID: [{pmcid}](https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid})")
        if id_parts: parts.append(" ".join(id_parts))

        formula_parts = []
        for formula_node in citation_node.xpath(".//inline-formula | .//disp-formula"):
            rendered = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', self._render_formula(formula_node, 2)).strip()
            if "$" in rendered:
                formula_parts.append(rendered)
        if formula_parts: parts.append(" ".join(formula_parts))

        result = ". ".join(parts)
        return " ".join(result.split())

    def _format_node(self, node, depth=2) -> str:
        """Route XML tags to their specific rendering handlers via pattern matching."""
        tag = self._local_name(node.tag)

        match tag:
            case "italic" | "i": return f"*{self._extract_text(node, depth)}*"
            case "bold" | "b": return f"**{self._extract_text(node, depth)}**"
            case "sup": return f"<sup>{self._extract_text(node, depth)}</sup>"
            case "sub": return f"<sub>{self._extract_text(node, depth)}</sub>"
            case "monospace" | "code": return f"`{self._extract_text(node, depth)}`"
            case "underline": return f"<u>{self._extract_text(node, depth)}</u>"
            case "strike": return f"~~{self._extract_text(node, depth)}~~"
            case "ext-link": return self._render_ext_link(node, depth)
            case "xref": return self._render_xref(node, depth)
            case "uri": return self._render_uri(node, depth)
            case "email": return f"<{self._extract_text(node, depth)}>"
            case "p": return self._render_paragraph(node, depth)
            case "sec" | "ack" | "app" | "notes" | "bio" | "fn-group": return self._render_section(node, depth)
            case "list": return self._render_list(node, depth)
            case "disp-quote": return self._render_quote(node, depth)
            case "boxed-text": return self._render_boxed_text(node, depth)
            case "fig-group": return "".join(self._format_node(f, depth) for f in node.findall("fig"))
            case "fig": return self._render_fig(node, depth)
            case "media": return self._render_media(node, depth)
            case "graphic" | "inline-graphic": return self._render_graphic(node)
            case "inline-supplementary-material": return self._render_inline_supp(node, depth)
            case "supplementary-material": return self._render_supp_material(node, depth)
            case "table-wrap": return self._render_table_wrap(node, depth)
            case "disp-formula" | "inline-formula": return self._render_formula(node, depth)
            case "break": return "\n\n"
            case "fn": return self._render_footnote(node, depth)
            case "title": return self._extract_text(node, depth)
            case "app-group": return "".join(self._format_node(c, depth) for c in node)
            case _: return self._extract_text(node, depth)

    def _render_ext_link(self, node, depth) -> str:
        """Render external links."""
        href = self._get_href(node)
        text = self._extract_text(node, depth) or href
        return f"[{text}]({href})" if href.startswith(("http://", "https://")) else text

    def _render_xref(self, node, depth) -> str:
        """Render cross-references."""
        raw_rid = node.get("rid", "")
        text = self._extract_text(node, depth).strip()
        ref_type = node.get("ref-type", "")

        if ref_type == "aff" or ref_type not in self.LINKABLE_REF_TYPES: return text

        rids = [r.lstrip("#") for r in raw_rid.split()]
        valid_rids = [r for r in rids if r in self._known_ids]
        if not valid_rids: return text

        display_text = text if text else valid_rids[0]
        return f"[{display_text}](#{valid_rids[0]})"

    def _render_uri(self, node, depth) -> str:
        """Render URIs."""
        href = self._extract_text(node, depth)
        return f"[{href}]({href})" if href.startswith(("http://", "https://")) else href

    def _render_paragraph(self, node, depth) -> str:
        """Render paragraphs."""
        text = self._extract_text(node, depth).strip()
        text = re.sub(r'!\[[^\]]*\]\(#[^)]+\)', '', text)
        if text.startswith("#"): text = "\\" + text
        return f"\n\n{text}\n\n" if text else ""

    def _render_section(self, node, depth) -> str:
        """Render sections and headings."""
        sec_id = node.get("id", "")
        anchor = f'<a id="{sec_id}"></a>\n' if sec_id else ""
        title_node = node.find("title")
        title_text = self._extract_text(title_node, depth).strip() if title_node is not None else self._local_name(node.tag).capitalize()
        title_text = self._normalize_heading(title_text)

        md = f"\n\n{anchor}{'#' * min(depth, 6)} {title_text}\n\n"
        for child in node:
            if self._local_name(child.tag) != "title":
                md += self._format_node(child, depth + 1)
        return md

    def _render_list(self, node, depth) -> str:
        """Render lists."""
        list_type = node.get("list-type", "bullet")
        md, counter = "\n\n", 1
        for item in node.findall("list-item"):
            item_text = "".join(self._format_node(c, depth).strip() for c in item) or self._extract_text(item, depth).strip()
            item_text = item_text.replace("\n", " ")
            if list_type in ("order", "roman", "alpha"):
                md += f"{counter}. {item_text}\n"
                counter += 1
            else:
                md += f"- {item_text}\n"
        return md + "\n\n"

    def _render_quote(self, node, depth) -> str:
        """Render blockquotes."""
        text = self._extract_text(node, depth).strip().replace("\n\n\n", "\n\n").replace("\n", "\n> ")
        return f"\n\n> {text}\n\n"

    def _render_boxed_text(self, node, depth) -> str:
        """Render boxed text elements."""
        md = "\n\n"
        title_node = node.find("title")
        if title_node is not None:
            title_text = self._extract_text(title_node, depth).strip()
            if title_text:
                md += f"**{title_text}**\n\n"
        for child in node:
            if self._local_name(child.tag) != "title":
                rendered = self._format_node(child, depth + 1)
                if rendered.strip():
                    md += rendered
        return md + "\n\n"

    def _render_media(self, node, depth) -> str:
        """Render media elements."""
        media_id = node.get("id", "")
        href = self._get_href(node)
        caption_node = node.find("caption")
        caption = " ".join(self._extract_text(caption_node, depth).split()) if caption_node is not None else "Media File"
        full_href = self._resolve_href(href)
        anchor = f'<a id="{media_id}"></a>\n' if media_id else ""

        if self._is_valid_image_url(full_href):
            return f"\n\n{anchor}[📺 {caption}]({full_href})\n\n"
        return f"\n\n{anchor}**{caption}**\n\n" if anchor else ""

    def _render_graphic(self, node) -> str:
        """Render graphic elements."""
        href = self._get_href(node)
        full_href = self._resolve_image_href(href)
        return f"![]({full_href})" if self._is_valid_image_url(full_href) else ""

    def _render_inline_supp(self, node, depth) -> str:
        """Render inline supplementary materials."""
        href = self._get_href(node)
        text = self._extract_text(node, depth).strip() or href.split("/")[-1]
        full_href = self._resolve_href(href)
        return f"[{text}]({full_href})" if full_href.startswith("http") else text

    def _render_supp_material(self, node, depth) -> str:
        """Render supplementary material blocks."""
        supp_id = node.get("id", "")
        anchor = f'<a id="{supp_id}"></a>\n' if supp_id else ""

        label_node = node.find("label")
        if label_node is None:
            label_node = node.find("title")
        label = " ".join(self._extract_text(label_node, depth).split()) if label_node is not None else ""

        caption_node = node.find("caption")
        caption_text = " ".join(self._extract_text(caption_node, depth).split()) if caption_node is not None else ""

        if not label and caption_text:
            label = caption_text.split('.')[0] if '.' in caption_text else caption_text[:60]
        if not label:
            label = "Supplementary Material"
        label = self._normalize_heading(label)

        media = node.find(".//media")
        if media is None:
            media = node.find(".//graphic")
        href = self._get_href(media) if media is not None else ""
        full_href = self._resolve_href(href) if href and not self._is_non_image_href(href) else ""

        md = f"\n\n{anchor}**{label}**\n\n"
        if caption_text and caption_text not in label:
            md += f"*{caption_text}*\n\n"

        if full_href and full_href.startswith("http"):
            md += f"[Download supplementary file]({full_href})\n\n"
        elif href:
            md += f"*File: {href}*\n\n"

        for child in node:
            if self._local_name(child.tag) not in ("label", "title", "caption", "media", "graphic"):
                rendered = self._format_node(child, depth + 1)
                if rendered.strip():
                    md += rendered + "\n"
        return md

    def _render_formula(self, node, depth) -> str:
        """Render mathematical formulas from LaTeX or MathML."""
        formula_id = node.get("id", "")
        tag = self._local_name(node.tag)
        anchor = f'<a id="{formula_id}"></a>\n' if formula_id and tag == "disp-formula" else ""

        # Try LaTeX first
        tex = node.find(".//tex-math")
        formula = "".join(tex.itertext()).strip() if tex is not None else ""
        if formula.startswith("<![CDATA["): formula = formula[9:-3]

        # Fallback to MathML
        if not formula:
            math_node = node.find(".//{http://www.w3.org/1998/Math/MathML}math")
            if math_node is None:
                math_node = node.find(".//math")
            if math_node is not None: formula = " ".join(math_node.itertext()).strip()

        # Fallback to plain text
        if not formula:
            formula = self._extract_text(node, depth).strip()

        # Collect image fallbacks
        graphic_md = []
        for g in list(node.findall(".//graphic")) + list(node.findall(".//inline-graphic")):
            full_href = self._resolve_image_href(self._get_href(g))
            if self._is_valid_image_url(full_href):
                graphic_md.append(f"![{self._sanitize_image_alt(self._get_href(g).split('/')[-1])}]({full_href})")
        images = " ".join(graphic_md)

        if formula:
            formula = formula.replace("\n", " ").strip()
            if tag == "disp-formula":
                rendered = f"\n\n{anchor}$$\n{formula}\n$$\n\n"
                return rendered + f"{images}\n\n" if images else rendered
            return f"${formula}$ {images}".strip()

        return images if images else f"{anchor}{self._extract_text(node, depth)}"

    def _render_footnote(self, node, depth) -> str:
        """Render footnotes cleanly, separating labels from text."""
        fn_id = node.get("id", "")
        anchor = f'<a id="{fn_id}"></a>\n' if fn_id else ""

        # Extract the label (e.g., 'a', 'b', '*') if it exists
        label_node = node.find("label")
        label_text = self._extract_text(label_node, depth).strip() if label_node is not None else ""

        # Extract paragraph text
        p_nodes = node.findall("p")
        if p_nodes:
            p_text = " ".join(self._extract_text(p, depth).strip() for p in p_nodes)
        else:
            # Fallback if no <p> tags are present
            full_text = self._extract_text(node, depth).strip()
            if label_text and full_text.startswith(label_text):
                p_text = full_text[len(label_text):].strip()
            else:
                p_text = full_text

        # Format cleanly
        if label_text:
            return f"{anchor}**[{label_text}]** {p_text}"
        return f"{anchor}{p_text}"

    def _render_table_footnotes(self, node, depth) -> str:
        """Render table footnotes and attributions cleanly."""
        md = ""
        # Handle table attributions (often source notes)
        for attrib in node.findall("attrib"):
            rendered = self._format_node(attrib, depth).strip()
            if rendered:
                md += f"*{rendered}*\n\n"

        # Handle table-wrap-foot
        foots = node.findall(".//table-wrap-foot")
        if foots:
            md += "\n"
            for foot in foots:
                for child in foot:
                    tag = self._local_name(child.tag)
                    if tag == "fn":
                        # Use our new clean footnote renderer
                        rendered = self._render_footnote(child, depth)
                        if rendered.strip():
                            md += rendered + "\n\n"
                    elif tag == "p":
                        # Sometimes footnotes are just raw paragraphs
                        rendered = self._format_node(child, depth).strip()
                        if rendered:
                            md += rendered + "\n\n"
                    else:
                        rendered = self._format_node(child, depth).strip()
                        if rendered:
                            md += rendered + "\n\n"
        return md + "\n"

    def _render_fig(self, node, depth) -> str:
        """Render figures."""
        fig_id = node.get("id", "")
        label = self._get_text(node, "label").rstrip(".:") or "Figure"
        caption = self._get_caption_text(node, depth)
        alt = self._sanitize_image_alt(f"{label}: {caption}")

        md = f'\n\n<a id="{fig_id}"></a>\n' if fig_id else "\n\n"

        visual_md = self._extract_fig_visuals(node, alt)
        if visual_md:
            return md + visual_md + self._extract_fig_extras(node, depth)

        ext_link = node.find(".//ext-link")
        if ext_link is not None:
            href = self._get_href(ext_link)
            if href.startswith("http"):
                return md + f"![{alt}]({href})\n\n" + self._extract_fig_extras(node, depth)

        if self._article_doi:
            return md + f"![{alt}](https://doi.org/{self._article_doi})\n\n" + self._extract_fig_extras(node, depth)

        md += f"**{label}**\n"
        if caption: md += f"*{caption}*\n"
        return md + self._extract_fig_extras(node, depth)

    def _render_table_wrap(self, node, depth) -> str:
        """Render table wrappers."""
        table_id = node.get("id", "")
        label = self._get_text(node, "label")
        caption = self._get_caption_text(node, depth)

        md = f'\n\n<a id="{table_id}"></a>\n' if table_id else "\n\n"
        if label: md += f"**{label}**\n"
        if caption: md += f"*{caption}*\n\n"

        table_nodes = node.findall(".//table")
        if table_nodes:
            md += "\n".join(self._table_to_html(t) for t in table_nodes) + "\n"
        else:
            md += self._render_table_image_fallback(node, label)

        md += self._render_table_footnotes(node, depth)
        return md

    def _generate_missing_anchors(self, md_content: str) -> str:
        """Append missing HTML anchors for internal links at the end of the document."""
        internal_links = re.findall(r"\[[^\]]*\]\(#([^)]+)\)", md_content)
        existing_anchors = set(re.findall(r'<a id="([^"]+)"></a>', md_content))
        missing = list(dict.fromkeys([link for link in internal_links if link not in existing_anchors]))

        if not missing: return md_content

        rendered_parts = []
        rendered_ids = set()

        for missing_id in missing:
            if missing_id in rendered_ids: continue

            elem = self._id_to_elem.get(missing_id)
            if elem is None:
                for k, v in self._id_to_elem.items():
                    if k.lower() == missing_id.lower():
                        elem = v
                        break

            if elem is not None:
                tag = self._local_name(elem.tag)
                if tag == "ref":
                    citation_node = elem.find(".//mixed-citation")
                    if citation_node is None:
                        citation_node = elem.find(".//element-citation")
                    text = self._format_reference_text(citation_node) if citation_node is not None else " ".join(self._extract_text(elem, 2).split())
                    if not text.strip(): text = " ".join(self._extract_text(citation_node, 2).split()) if citation_node is not None else ""
                    rendered_parts.append(f'<a id="{missing_id}"></a>\n{text}\n')
                else:
                    rendered = self._format_node(elem, depth=2)
                    if f'<a id="{missing_id}"></a>' not in rendered: rendered = f'<a id="{missing_id}"></a>\n' + rendered
                    rendered_parts.append(rendered if rendered.strip() else f'<a id="{missing_id}"></a>\n')
            else:
                rendered_parts.append(f'<a id="{missing_id}"></a>\n')

            rendered_ids.add(missing_id)

        if rendered_parts: md_content += "\n\n" + "\n".join(rendered_parts)
        return md_content

    def _append_missing_images(self, root, md_content: str) -> str:
        """Append any unrendered graphics to the end of the document."""
        md_filenames = {url.split("/")[-1].split("?")[0] for url in set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_content)) if url.startswith("http")}
        rendered = []
        seen = set(md_filenames)

        xpath = "(//graphic | //inline-graphic)[not(ancestor::sub-article) and not(ancestor::supplementary-material)]"
        for g in root.xpath(xpath):
            href = self._get_href(g)
            if not href or href.startswith("#"): continue
            filename = href.split("/")[-1].split("?")[0]
            if not filename or filename in seen: continue

            resolved = self._resolve_image_href(href)
            if not resolved.startswith("http"): resolved = self._resolve_href(href)
            if not resolved.startswith("http"): continue

            rendered.append(f"![{self._sanitize_image_alt(filename)}]({resolved})")
            seen.add(filename)

        if rendered: md_content += "\n\n" + "\n\n".join(rendered)
        return md_content

    def _extract_text(self, node, depth=2) -> str:
        """Recursively extract and format text content from a node."""
        text = []
        if node.text: text.append(node.text)
        for child in node:
            child_text = self._format_node(child, depth)
            tag = self._local_name(child.tag)
            if tag in ("pub-id", "year", "volume", "issue", "fpage", "lpage", "elocation-id", "ext-link", "xref", "email"):
                if not child_text.endswith(" "): child_text += " "
            text.append(child_text)
            if child.tail: text.append(child.tail)
        return "".join(text)

    def _normalize_heading(self, text: str) -> str:
        """Normalize section headings to standard title casing."""
        if not text: return ""
        heading_map = {
            "FUNDING": "Funding", "ACKNOWLEDGEMENTS": "Acknowledgements", "ACKNOWLEDGMENTS": "Acknowledgments",
            "CONFLICT OF INTEREST": "Conflicts of Interest", "CONFLICT OF INTEREST STATEMENT": "Conflicts of Interest",
            "COMPETING INTERESTS": "Competing Interests", "AUTHOR CONTRIBUTIONS": "Author Contributions",
            "AUTHORS' CONTRIBUTIONS": "Author Contributions", "DATA AVAILABILITY": "Data Availability",
            "DATA AVAILABILITY STATEMENT": "Data Availability", "ETHICS APPROVAL": "Ethics Approval",
            "REFERENCES": "References", "SUPPLEMENTARY MATERIAL": "Supplementary Material", "ABSTRACT": "Abstract",
            "INTRODUCTION": "Introduction", "METHODS": "Methods", "RESULTS": "Results", "DISCUSSION": "Discussion",
            "CONCLUSION": "Conclusion", "CONCLUSIONS": "Conclusions",
        }
        upper_text = text.upper().strip()
        if upper_text in heading_map: return heading_map[upper_text]
        return text.title() if text.isupper() else text

    @staticmethod
    def _local_name(tag) -> str:
        """Strip namespace from an XML tag."""
        if not isinstance(tag, str): return ""
        return tag.rsplit("}", 1)[-1]

    def _get_href(self, elem) -> str:
        """Extract href attribute from an element, handling namespaces."""
        if elem is None: return ""
        href = elem.get("{http://www.w3.org/1999/xlink}href") or elem.get("href")
        if href: return href
        for key, val in elem.attrib.items():
            if key.endswith("href") or key.endswith("}href"): return val
        return ""

    def _resolve_image_href(self, href: str) -> str:
        """Resolve relative image paths to absolute URLs."""
        if not href: return ""
        href = href.strip()
        if not href or href.startswith("#") or href.startswith(("http://", "https://")): return href
        if self._is_non_image_href(href): return ""
        suffix = Path(href).suffix.lower()
        if suffix not in self.IMAGE_EXTENSIONS: return ""
        if self.image_base_url and self._article_dir: return f"{self.image_base_url}/{self._article_dir}/{href}"
        if self.image_base_url: return f"{self.image_base_url}/{href}"
        return href

    def _resolve_href(self, href: str) -> str:
        """Resolve any relative file href to an absolute URL."""
        if not href: return ""
        href = href.strip()
        if not href or href.startswith("#") or href.startswith(("http://", "https://")): return href
        if self.image_base_url and self._article_dir: return f"{self.image_base_url}/{self._article_dir}/{href}"
        if self.image_base_url: return f"{self.image_base_url}/{href}"
        return href

    def _is_non_image_href(self, href: str) -> bool:
        """Check if an href matches known non-image patterns."""
        for pattern in self._NON_IMAGE_PATTERNS:
            if pattern.search(href): return True
        return "." not in href

    def _is_valid_image_url(self, url: str) -> bool:
        """Validate if a URL is a properly formed absolute image link."""
        return bool(url) and not url.startswith("#") and url.startswith("http")

    def _sanitize_image_alt(self, text: str) -> str:
        """Remove characters that break markdown image alt parsing."""
        if not text: return ""
        text = text.replace("[", " ").replace("]", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _get_text(self, node, xpath: str) -> str:
        """Safely extract text from the first node matching an XPath."""
        els = node.xpath(xpath)
        if not els: return ""
        el = els[0]
        return el.strip() if isinstance(el, str) else "".join(el.itertext()).strip()

    def _get_caption_text(self, node, depth) -> str:
        """Extract and format caption text."""
        caption_node = node.find("caption")
        if caption_node is None: return ""
        parts = []
        if caption_node.text: parts.append(caption_node.text)
        for child in caption_node:
            tag = self._local_name(child.tag)
            if tag in ("fig", "supplementary-material", "table-wrap"): continue
            parts.append(self._format_node(child, depth))
            if child.tail: parts.append(child.tail)
        return " ".join("".join(parts).split())

    def _extract_fig_visuals(self, node, alt: str) -> str:
        """Extract and format visual elements from a figure."""
        md = ""
        for visual in list(node.iter("graphic")) + list(node.iter("media")):
            parent = visual.getparent()
            nested = False
            while parent is not None and parent != node:
                if self._local_name(parent.tag) in ("fig", "supplementary-material"):
                    nested = True
                    break
                parent = parent.getparent()
            if nested: continue

            href = self._get_href(visual)
            full_href = self._resolve_href(href) if self._local_name(visual.tag) == "media" else self._resolve_image_href(href)
            if self._is_valid_image_url(full_href):
                md += f"![{alt}]({full_href})\n\n"
        return md

    def _extract_fig_extras(self, node, depth) -> str:
        """Extract extra content from a figure."""
        md = ""
        for child in node.iterchildren():
            tag = self._local_name(child.tag)
            if tag in ("label", "caption", "graphic", "media"): continue
            rendered = self._format_node(child, depth)
            if rendered.strip(): md += f"\n{rendered}\n"
        return md

    def _render_table_image_fallback(self, node, label) -> str:
        """Render a table as an image if no XML table is present."""
        imgs = []
        for g in list(node.findall(".//graphic")) + list(node.findall(".//inline-graphic")):
            href = self._get_href(g)
            full_href = self._resolve_image_href(href)
            if not full_href.startswith("http"): full_href = self._resolve_href(href)
            if self._is_valid_image_url(full_href):
                imgs.append(f"![{self._sanitize_image_alt(label or 'Table')}]({full_href})")
        return "<table><tbody><tr><td>" + "<br>".join(imgs) + "</td></tr></tbody></table>\n" if imgs else ""


    def _table_to_html(self, table_node) -> str:
        """Convert an XML table node to an HTML string."""
        if table_node is None: return ""
        parts = ["<table", self._format_attrs(table_node), ">\n"]
        for child in table_node:
            tag = self._local_name(child.tag)
            if tag in ("thead", "tbody", "tfoot"): parts.append(self._format_table_section(child))
            elif tag == "tr": parts.append(self._format_table_row(child))
        parts.append("</table>")
        return "".join(parts)

    def _format_attrs(self, elem) -> str:
        """Format element attributes for HTML output."""
        parts = []
        for k, v in elem.attrib.items():
            key = self._local_name(k)
            parts.append(f' {key}="{v}"')
        return "".join(parts)

    def _format_table_section(self, node) -> str:
        """Format a table section (thead, tbody, tfoot)."""
        tag = self._local_name(node.tag)
        rows = [self._format_table_row(child) for child in node if self._local_name(child.tag) == "tr"]
        return f"<{tag}>\n" + "\n".join(rows) + f"\n</{tag}>\n"

    def _format_table_row(self, node) -> str:
        """Format a single table row."""
        cells = []
        for child in node:
            tag = self._local_name(child.tag)
            if tag in ("th", "td"):
                cells.append(f"<{tag}{self._format_attrs(child)}>{self._format_table_cell(child)}</{tag}>")
        return "<tr>" + "".join(cells) + "</tr>"

    def _format_table_cell(self, cell) -> str:
        """Format a single table cell."""
        content = self._extract_text(cell, depth=2)
        return re.sub(r"\s+", " ", content).strip()

    def _clean_metadata(self, obj):
        """Recursively remove empty values from metadata dictionaries."""
        if isinstance(obj, dict):
            cleaned = {k: self._clean_metadata(v) for k, v in obj.items()}
            return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
        elif isinstance(obj, list):
            cleaned = [self._clean_metadata(i) for i in obj]
            return [i for i in cleaned if i not in (None, "", [], {})]
        return obj