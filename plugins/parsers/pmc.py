import copy

import yaml
from lxml import etree
from pathlib import Path
from typing import Any, Dict, List


class PMCParser:
    """
    Production-grade JATS XML → Markdown parser.

    Architecture:
      1. YAML frontmatter: strictly structured document metadata.
      2. Markdown Abstract: structured text + graphical abstract.
      3. Markdown Body: recursive AST walk preserving hierarchy, inline formatting,
         figures (S3 URLs), tables (HTML), and lists.
      4. Markdown Back Matter: acknowledgments, funding, COI, etc.
      5. Markdown References: numbered list with ID anchors for xref linking.
    """

    # Default base for images/files hosted on the PMC Open Access bucket.
    # Structure: {base}/{PMCID.Version}/{filename}
    PMC_OA_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com"

    def __init__(self, image_base_url: str = PMC_OA_BASE_URL):
        # Pass "" to keep hrefs relative, or your own CDN base.
        self.image_base_url = image_base_url.rstrip("/") if image_base_url else ""
        self._article_dir = ""  # populated during parse (e.g. "PMC7616479.2")

    def parse(self, source: Path) -> str:
        root = etree.parse(str(source), parser=etree.XMLParser(recover=True)).getroot()
        self._resolve_article_dir(root)

        # 1. Metadata -> YAML
        metadata = self._parse_front_matter(root)

        # 2. Abstract -> Markdown
        abstract_md = self._parse_abstract_md(root)

        # 3. Body -> Markdown
        body_md = self._parse_body_md(root)

        # 4. Back Matter -> Markdown
        back_md = self._parse_back_md(root)

        # 5. References -> Markdown
        references_md = self._parse_references_md(root)

        yaml_str = yaml.dump(
            metadata,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=1000,
        )

        # Assemble final document
        parts = [f"---\n{yaml_str}---"]
        if abstract_md: parts.append(abstract_md)
        if body_md: parts.append(body_md)
        if back_md: parts.append(back_md)
        if references_md: parts.append(references_md)

        return "\n\n".join(parts)

    def _resolve_article_dir(self, root):
        """Resolves the article folder on the PMC bucket (e.g. 'PMC7616479.2')."""
        ids = {
            aid.get("pub-id-type"): "".join(aid.itertext()).strip()
            for aid in root.xpath("//front/article-meta/article-id")
        }
        self._article_dir = ids.get("pmcid-ver") or ids.get("pmcid") or ""

    def _clean_metadata(self, obj):
        """Recursively strips empty strings, lists, dicts, and None from YAML payload."""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_v = self._clean_metadata(v)
                # Keep if not empty (preserves 0 and False)
                if cleaned_v not in (None, "", [], {}):
                    cleaned[k] = cleaned_v
            return cleaned
        elif isinstance(obj, list):
            cleaned = [self._clean_metadata(item) for item in obj]
            return [item for item in cleaned if item not in (None, "", [], {})]
        return obj

    def _parse_front_matter(self, root) -> Dict[str, Any]:
        meta = {
            "journal_meta": self._parse_journal_meta(root),
            "article_meta": self._parse_article_meta(root),
        }
        return self._clean_metadata(meta)

    def _parse_journal_meta(self, root) -> Dict[str, Any]:
        j_meta = root.xpath("//front/journal-meta")
        if not j_meta: return {}
        j_meta = j_meta[0]

        return {
            "journal_ids": {
                jid.get("journal-id-type"): "".join(jid.itertext()).strip()
                for jid in j_meta.xpath("./journal-id") if jid.get("journal-id-type")
            },
            "title": self._get_text(j_meta, ".//journal-title"),
            "issns": {
                issn.get("pub-type"): "".join(issn.itertext()).strip()
                for issn in j_meta.xpath("./issn") if issn.get("pub-type")
            },
            "publisher": {
                "name": self._get_text(j_meta, ".//publisher-name"),
                "loc": self._get_text(j_meta, ".//publisher-loc"),
            },
        }

    def _parse_article_meta(self, root) -> Dict[str, Any]:
        a_meta = root.xpath("//front/article-meta")
        if not a_meta: return {}
        a_meta = a_meta[0]

        # Notice: Abstract and References are intentionally excluded here.
        meta = {
            "article_ids": {
                aid.get("pub-id-type"): "".join(aid.itertext()).strip()
                for aid in a_meta.xpath("./article-id") if aid.get("pub-id-type")
            },
            "categories": [
                subj.text.strip()
                for sg in a_meta.xpath("./article-categories/subj-group")
                for subj in sg.xpath("./subject") if subj.text
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
            "keywords": [kw.text.strip() for kw in a_meta.xpath(".//kwd-group/kwd") if kw.text],
            "funding": self._parse_funding(a_meta),
            "counts": {cnt.tag: cnt.get("count") for cnt in a_meta.xpath("./counts/*")},
            "custom_meta": self._parse_custom_meta(a_meta),
        }
        return meta

    def _parse_custom_meta(self, node) -> Dict[str, str]:
        return {
            m.findtext("meta-name"): m.findtext("meta-value")
            for m in node.xpath(".//custom-meta-group/custom-meta")
            if m.findtext("meta-name") and m.findtext("meta-value")
        }

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
            author["name"] = f"{prefix} {author.get('given_names', '')} {author.get('surname', '')} {suffix}".replace(
                "  ", " ").strip()

        collab_node = contrib.find("./collab")
        if collab_node is not None:
            collab_parts = [collab_node.text] if collab_node.text else []
            for child in collab_node.iterchildren():
                if child.tag != "contrib-group":
                    collab_parts.append("".join(child.itertext()))
            author["collab"] = "".join(collab_parts).strip()

            nested_groups = collab_node.xpath("./contrib-group")
            if nested_groups:
                author["members"] = [
                    self._parse_single_contrib(nc)
                    for group in nested_groups
                    for nc in group.xpath("./contrib")
                ]

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
        return {
            pd.get("pub-type", "unknown"): self._format_date(pd)
            for pd in a_meta.xpath("./pub-date") if self._format_date(pd)
        }

    def _parse_history(self, a_meta) -> Dict[str, str]:
        return {
            hd.get("date-type", "unknown"): self._format_date(hd)
            for hd in a_meta.xpath("./history/date") if self._format_date(hd)
        }

    def _parse_pub_history(self, a_meta) -> List[Dict[str, Any]]:
        events = []
        for event in a_meta.xpath("./pub-history/event"):
            ev_data: Dict[str, Any] = {"event_type": event.get("event-type", "unknown")}

            # Explicit None checks prevent lxml truth-testing bugs
            date_node = event.find("./date")
            if date_node is None:
                date_node = event.find("./pub-date")

            if date_node is not None:
                ev_data["date"] = self._format_date(date_node)

            art_ids = {
                aid.get("pub-id-type"): "".join(aid.itertext()).strip()
                for aid in event.xpath("./article-id") if aid.get("pub-id-type")
            }
            if art_ids: ev_data["article_ids"] = art_ids

            version = event.findtext("./article-version")
            if version: ev_data["version"] = version.strip()

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
        if stmt is not None: perms["copyright_statement"] = "".join(stmt.itertext()).strip()
        year = a_meta.findtext(".//copyright-year")
        if year: perms["copyright_year"] = year.strip()

        license_node = a_meta.find(".//license")
        if license_node is not None:
            lic_url, lic_text = None, None
            for el in license_node.iter():
                tag = el.tag if isinstance(el.tag, str) else ""
                if tag.endswith("license_ref"):
                    lic_url = "".join(el.itertext()).strip()
                elif tag.endswith("ext-link"):
                    href = el.get("{http://www.w3.org/1999/xlink}href") or el.get("href")
                    if href and not lic_url: lic_url = href
                elif tag.endswith("license-p"):
                    lic_text = "".join(el.itertext()).strip()
            if lic_url: perms["license_url"] = lic_url
            if lic_text: perms["license_text"] = lic_text
        return perms

    def _parse_funding(self, a_meta) -> List[Dict[str, Any]]:
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

    # ==================================================================
    # 2. ABSTRACT MARKDOWN
    # ==================================================================
    def _parse_abstract_md(self, root) -> str:
        md_parts = []

        # Graphical abstract
        graphical_abs = root.xpath("//front/article-meta/abstract[@abstract-type='graphical']")
        if graphical_abs:
            md_parts.append("## Graphical Abstract")
            for child in graphical_abs[0]:
                md_parts.append(self._format_node(child, depth=3))

        # Standard abstract
        standard_abs = root.xpath("//front/article-meta/abstract[not(@abstract-type='graphical')]")
        if standard_abs:
            md_parts.append("## Abstract")
            for child in standard_abs[0]:
                # Pass depth=3 so internal <sec> elements become ### headings
                md_parts.append(self._format_node(child, depth=3))

        return "\n\n".join([p for p in md_parts if p])

    def _parse_body_md(self, root) -> str:
        body = root.find(".//body")
        md_parts = []

        # Title as H1
        title = self._get_text(root, "//front/article-meta/title-group/article-title")
        if title:
            md_parts.append(f"# {title}")

        if body is not None:
            for child in body:
                md_parts.append(self._format_node(child, depth=2))

        return "\n\n".join([p for p in md_parts if p])

    def _parse_back_md(self, root) -> str:
        back = root.find(".//back")
        if back is None: return ""

        md_parts = []
        for child in back:
            # Skip reference list, it's handled in stage 5
            if child.tag == "ref-list": continue
            md_parts.append(self._format_node(child, depth=2))

        return "\n\n".join([p for p in md_parts if p])

    def _parse_references_md(self, root) -> str:
        ref_list = root.find(".//back/ref-list")
        if ref_list is None: return ""

        md_parts = ["## References"]
        counter = 1

        for ref in ref_list.findall("ref"):
            ref_id = ref.get("id", "")

            # Explicit None checks for citation nodes
            citation_node = ref.find(".//mixed-citation")
            if citation_node is None:
                citation_node = ref.find(".//element-citation")

            if citation_node is not None:
                text = self._extract_text(citation_node, 2).strip()
                text = " ".join(text.split())
            else:
                text = " ".join(self._extract_text(ref, 2).strip().split())

            md_parts.append(f"<a id=\"{ref_id}\"></a>\n{counter}. {text}")
            counter += 1

        return "\n\n".join(md_parts)

    def _format_node(self, node, depth=2) -> str:
        tag = node.tag if isinstance(node.tag, str) else ""

        # ---- Inline formatting ----
        if tag in ("italic", "i"): return f"*{self._extract_text(node, depth)}*"
        if tag in ("bold", "b"): return f"**{self._extract_text(node, depth)}**"
        if tag == "sup": return f"<sup>{self._extract_text(node, depth)}</sup>"
        if tag == "sub": return f"<sub>{self._extract_text(node, depth)}</sub>"
        if tag in ("monospace", "code"): return f"`{self._extract_text(node, depth)}`"
        if tag == "underline": return f"<u>{self._extract_text(node, depth)}</u>"
        if tag == "strike": return f"~~{self._extract_text(node, depth)}~~"

        if tag == "ext-link":
            href = node.get("{http://www.w3.org/1999/xlink}href") or node.get("href")
            text = self._extract_text(node, depth) or href
            return f"[{text}]({href})" if href else text

        if tag == "xref":
            rid = node.get("rid", "")
            text = self._extract_text(node, depth)
            # Preserves source brackets natively (e.g. [[15](#bib15)])
            return f"[{text}](#{rid})" if node.get("ref-type") in ("bibr", "fig", "table", "fn", "table-fn") else text

        if tag == "email": return f"<{self._extract_text(node, depth)}>"
        if tag == "uri":
            href = self._extract_text(node, depth)
            return f"[{href}]({href})"

        # ---- Block formatting ----
        if tag == "p":
            text = self._extract_text(node, depth).strip()
            return f"\n\n{text}\n\n" if text else ""

        if tag in ("sec", "ack", "app", "notes", "bio", "fn-group"):
            title_node = node.find("title")
            title_text = self._extract_text(title_node, depth).strip() if title_node is not None else ""
            if not title_text: title_text = tag.capitalize()
            md = f"\n\n{'#' * min(depth, 6)} {title_text}\n\n"
            for child in node:
                if child.tag != "title":
                    md += self._format_node(child, depth + 1)
            return md

        if tag == "app-group":
            return "".join(self._format_node(child, depth) for child in node)

        if tag == "list":
            list_type = node.get("list-type", "bullet")
            md, counter = "\n\n", 1
            for item in node.findall("list-item"):
                item_text = ""
                for child in item:
                    item_text += self._format_node(child, depth).strip()
                if not item_text:
                    item_text = self._extract_text(item, depth).strip()
                item_text = item_text.replace("\n", " ")
                if list_type in ("order", "roman", "alpha"):
                    md += f"{counter}. {item_text}\n"
                    counter += 1
                else:
                    md += f"- {item_text}\n"
            return md + "\n\n"

        if tag in ("disp-quote", "boxed-text"):
            text = self._extract_text(node, depth).strip()
            while "\n\n\n" in text: text = text.replace("\n\n\n", "\n\n")
            text = text.replace("\n", "\n> ")
            return f"\n\n> {text}\n\n"

        if tag == "fig-group":
            return "".join(self._format_node(fig, depth) for fig in node.findall("fig"))

        if tag == "fig":
            fig_id = node.get("id", "")
            label_node = node.find("label")
            label = " ".join(self._extract_text(label_node, depth).split()).rstrip(
                ".:") if label_node is not None else "Figure"
            caption_node = node.find("caption")
            caption = " ".join(self._extract_text(caption_node, depth).split()) if caption_node is not None else ""

            md = f"\n\n<a id=\"{fig_id}\"></a>\n"
            for graphic in node.findall(".//graphic"):
                href = graphic.get("{http://www.w3.org/1999/xlink}href") or graphic.get("href")
                full_href = self._resolve_image_href(href)
                if full_href:
                    md += f"![{label}: {caption}]({full_href})\n\n"
                else:
                    md += f"**{label}**\n*{caption}*\n\n"
            return md

        if tag == "media":
            media_id = node.get("id", "")
            href = node.get("{http://www.w3.org/1999/xlink}href") or node.get("href")
            caption_node = node.find("caption")
            caption = " ".join(
                self._extract_text(caption_node, depth).split()) if caption_node is not None else "Media File"
            full_href = self._resolve_image_href(href)
            return f"\n\n<a id=\"{media_id}\"></a>\n[📺 {caption}]({full_href})\n\n" if full_href else ""

        if tag == "inline-graphic":
            href = node.get("{http://www.w3.org/1999/xlink}href") or node.get("href")
            full_href = self._resolve_image_href(href)
            return f"![]({full_href})" if full_href else ""

        if tag == "supplementary-material":
            label_node = node.find("label")
            label = " ".join(
                self._extract_text(label_node, depth).split()) if label_node is not None else "Supplementary Material"
            media = node.find(".//media")
            href = (media.get("{http://www.w3.org/1999/xlink}href") or media.get("href")) if media is not None else None
            full_href = self._resolve_image_href(href) if href else ""

            md = f"\n\n## {label}\n\n" if depth == 2 else f"\n\n**{label}**\n\n"
            if full_href:
                md += f"[Download supplementary file]({full_href})\n\n"
            return md

        if tag == "table-wrap":
            table_id = node.get("id", "")
            label = " ".join(self._extract_text(node.find("label"), depth).split()) if node.find(
                "label") is not None else ""
            caption = " ".join(self._extract_text(node.find("caption"), depth).split()) if node.find(
                "caption") is not None else ""
            table_node = node.find(".//table")
            table_html = self._table_to_html(table_node) if table_node is not None else ""

            md = f"\n\n<a id=\"{table_id}\"></a>\n"
            if label: md += f"**{label}**\n"
            if caption: md += f"*{caption}*\n\n"
            if table_html: md += f"{table_html}\n"

            foot_notes = node.xpath(".//table-wrap-foot//fn")
            if foot_notes:
                md += "\n" + "\n\n".join(
                    f"<a id=\"{fn_node.get('id', '')}\"></a>*{' '.join(self._extract_text(fn_node, depth).split())}*"
                    for fn_node in foot_notes
                ) + "\n\n"
            else:
                md += "\n"
            return md

        if tag in ("disp-formula", "inline-formula"):
            tex = node.find(".//tex-math")
            formula = "".join(tex.itertext()).strip() if tex is not None else ""
            if formula.startswith("<![CDATA["): formula = formula[9:-3]
            if formula:
                return f"\n\n$$\n{formula}\n$$\n\n" if tag == "disp-formula" else f"${formula}$"
            return self._extract_text(node, depth)

        if tag == "break": return "\n\n"
        if tag == "fn":
            fn_id = node.get("id", "")
            return f"<a id=\"{fn_id}\"></a>^[{self._extract_text(node, depth).strip()}]"
        if tag == "title": return ""  # handled by parent section

        # Fallback for unhandled tags
        return self._extract_text(node, depth)

    def _extract_text(self, node, depth=2) -> str:
        text = []
        if node.text: text.append(node.text)
        for child in node:
            child_text = self._format_node(child, depth)
            # Prevent "smashed" text for adjacent tags that lack spaces in XML
            if isinstance(child.tag, str) and child.tag in (
                    "pub-id", "year", "volume", "issue", "fpage", "lpage",
                    "elocation-id", "ext-link", "xref", "email",
            ):
                if not child_text.endswith(" "): child_text += " "
            text.append(child_text)
            if child.tail: text.append(child.tail)
        return "".join(text)

    def _resolve_image_href(self, href: str) -> str:
        """Maps a relative JATS graphic href to its hosted location."""
        if not href: return ""
        if href.startswith(("http://", "https://")): return href
        if self.image_base_url and self._article_dir:
            return f"{self.image_base_url}/{self._article_dir}/{href}"
        if self.image_base_url:
            return f"{self.image_base_url}/{href}"
        return href

    def _table_to_html(self, table_node) -> str:
        if table_node is None: return ""
        table_copy = copy.deepcopy(table_node)
        for elem in table_copy.getiterator():
            if not hasattr(elem.tag, "find"): continue
            if isinstance(elem.tag, str) and elem.tag.startswith("{"):
                elem.tag = elem.tag.split("}", 1)[1]
            attribs = dict(elem.attrib)
            elem.attrib.clear()
            for k, v in attribs.items():
                new_k = k.split("}", 1)[1] if isinstance(k, str) and k.startswith("{") else k
                elem.attrib[new_k] = v
        return etree.tostring(table_copy, method="html", encoding="unicode", pretty_print=True)

    def _get_text(self, node, xpath: str) -> str:
        # Use .xpath() instead of .find() to support absolute paths and advanced predicates
        els = node.xpath(xpath)
        if not els:
            return ""
        el = els[0]
        # Handle cases where xpath returns a string/attribute directly instead of an element
        if isinstance(el, str):
            return el.strip()
        return "".join(el.itertext()).strip()