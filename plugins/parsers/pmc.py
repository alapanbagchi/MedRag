import copy

import yaml
from lxml import etree
from pathlib import Path
from typing import Any, Dict, List

from core.protocols import Parser


class PMCParser(Parser):
    """
    Production-grade JATS XML → Markdown parser.

    Output is a single Markdown document containing:
      - YAML frontmatter: journal metadata, article metadata, structured references
      - Markdown body: title, abstract, sections, figures (resolved against the
        PMC Open Access S3 bucket), tables (clean HTML), formulas, back matter
        (acknowledgements, funding, COI, etc.) and an anchored reference list.
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

        # Resolve the article folder on the PMC bucket (e.g. "PMC7616479.2")
        ids = {
            aid.get("pub-id-type"): "".join(aid.itertext()).strip()
            for aid in root.xpath("//front/article-meta/article-id")
        }
        self._article_dir = ids.get("pmcid-ver") or ids.get("pmcid") or ""

        metadata = self._parse_front_matter(root)
        body_md = self._parse_body_markdown(root, metadata)
        back_md = self._parse_back_markdown(root)

        yaml_str = yaml.dump(
            metadata,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=1000,
        )
        return f"---\n{yaml_str}---\n\n{body_md}\n{back_md}"

    def _parse_front_matter(self, root) -> Dict[str, Any]:
        """
        Parses the front matter from the JATS and returns it as YAML
        :param root:
        :return:
        """
        meta = {
            "journal_meta": self._parse_journal_meta(root),
            "article_meta": self._parse_article_meta(root),
            "references": self._parse_references(root),
        }
        return {k: v for k, v in meta.items() if v}

    def _parse_journal_meta(self, root) -> Dict[str, Any]:
        j_meta = root.xpath("//front/journal-meta")
        if not j_meta:
            return {}
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
        if not a_meta:
            return {}
        a_meta = a_meta[0]

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
            "abstract": self._parse_abstract(a_meta.find("./abstract")),
            "keywords": [kw.text.strip() for kw in a_meta.xpath(".//kwd-group/kwd") if kw.text],
            "funding": self._parse_funding(a_meta),
            "counts": {cnt.tag: cnt.get("count") for cnt in a_meta.xpath("./counts/*")},
            "custom_meta": self._parse_custom_meta(root),
        }
        return {k: v for k, v in meta.items() if v}

    def _parse_custom_meta(self, root) -> Dict[str, str]:
        return {
            m.findtext("meta-name"): m.findtext("meta-value")
            for m in root.xpath(".//custom-meta-group/custom-meta")
            if m.findtext("meta-name") and m.findtext("meta-value")
        }

    def _parse_references(self, root) -> List[Dict[str, Any]]:
        refs = []
        for ref in root.xpath("//back/ref-list/ref"):
            citation_node = ref.find(".//mixed-citation") or ref.find(".//element-citation")
            if citation_node is None:
                continue

            ref_data: Dict[str, Any] = {
                "id": ref.get("id", ""),
                "type": citation_node.get("publication-type", ""),
            }

            authors = [
                f"{self._get_text(p, 'given-names')} {self._get_text(p, 'surname')}".strip()
                for p in citation_node.xpath(".//person-group/string-name")
            ]
            if authors:
                ref_data["authors"] = authors

            collab = citation_node.findtext(".//person-group/collab")
            if collab:
                ref_data["collab"] = collab.strip()

            ref_data["title"] = self._get_text(citation_node, "article-title")
            ref_data["source"] = self._get_text(citation_node, "source")
            ref_data["year"] = self._get_text(citation_node, "year")
            ref_data["volume"] = self._get_text(citation_node, "volume")
            ref_data["pages"] = self._get_text(citation_node, "fpage")

            for pub_id in citation_node.xpath(".//pub-id"):
                id_type = pub_id.get("pub-id-type")
                if id_type:
                    ref_data[id_type] = "".join(pub_id.itertext()).strip()

            refs.append(ref_data)
        return refs

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
            author["name"] = f"{prefix} {author.get('given_names', '')} {author.get('surname', '')} {suffix}".replace("  ", " ").strip()

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

    def _parse_abstract(self, abstract_node) -> Dict[str, Any]:
        if abstract_node is None:
            return {}
        abs_data: Dict[str, Any] = {}
        title = self._get_text(abstract_node, "./title")
        if title:
            abs_data["title"] = title

        sections = [
            {
                "title": self._get_text(sec, "title"),
                "text": " ".join("".join(p.itertext()).strip() for p in sec.xpath("./p")),
            }
            for sec in abstract_node.xpath("./sec")
        ]
        if sections:
            abs_data["sections"] = sections
        else:
            abs_data["text"] = " ".join("".join(p.itertext()).strip() for p in abstract_node.xpath("./p"))
        return abs_data

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
        y = node.findtext("year")
        if not y:
            return ""
        parts = [y]
        m = node.findtext("month")
        d = node.findtext("day")
        if m:
            parts.append(m.zfill(2))
        if d:
            parts.append(d.zfill(2))
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
            date_node = event.find("./date") or event.find("./pub-date")
            if date_node is not None:
                ev_data["date"] = self._format_date(date_node)
            art_ids = {
                aid.get("pub-id-type"): "".join(aid.itertext()).strip()
                for aid in event.xpath("./article-id") if aid.get("pub-id-type")
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
            lic_url, lic_text = None, None
            for el in license_node.iter():
                tag = el.tag if isinstance(el.tag, str) else ""
                if tag.endswith("license_ref"):
                    lic_url = "".join(el.itertext()).strip()
                elif tag.endswith("ext-link"):
                    href = el.get("{http://www.w3.org/1999/xlink}href") or el.get("href")
                    if href and not lic_url:
                        lic_url = href
                elif tag.endswith("license-p"):
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

    # MARKDOWN PARSING FOR THE ACTUAL ARTICLE
    def _parse_body_markdown(self, root, metadata: Dict) -> str:
        body = root.find(".//body")
        md = ""

        title = metadata.get("article_meta", {}).get("title", "")
        if title:
            md += f"# {title}\n\n"

        abstract = metadata.get("article_meta", {}).get("abstract")
        if abstract:
            md += "## Abstract\n\n"
            if "sections" in abstract:
                for sec in abstract["sections"]:
                    md += f"**{sec['title']}**\n{sec['text']}\n\n"
            elif "text" in abstract:
                md += f"{abstract['text']}\n\n"

        if body is not None:
            for child in body:
                md += self._format_node(child, depth=2)
        return md.strip()

    def _parse_back_markdown(self, root) -> str:
        back = root.find(".//back")
        if back is None:
            return ""
        md = "\n\n---\n\n"
        for child in back:
            if child.tag == "ref-list":
                md += self._format_ref_list(child)
            else:
                md += self._format_node(child, depth=2)
        return md

    def _format_ref_list(self, ref_list_node) -> str:
        md = "\n\n## References\n\n"
        for ref in ref_list_node.findall("ref"):
            ref_id = ref.get("id", "")
            citation_node = ref.find(".//mixed-citation") or ref.find(".//element-citation")
            text = self._extract_text(citation_node if citation_node is not None else ref, 2).strip()
            text = " ".join(text.split())
            md += f"<a id=\"{ref_id}\"></a>\n{text}\n\n"
        return md

    def _format_node(self, node, depth=2) -> str:
        tag = node.tag if isinstance(node.tag, str) else ""

        # ---- Inline formatting ----
        if tag in ("italic", "i"):
            return f"*{self._extract_text(node, depth)}*"
        if tag in ("bold", "b"):
            return f"**{self._extract_text(node, depth)}**"
        if tag == "sup":
            return f"<sup>{self._extract_text(node, depth)}</sup>"
        if tag == "sub":
            return f"<sub>{self._extract_text(node, depth)}</sub>"
        if tag in ("monospace", "code"):
            return f"`{self._extract_text(node, depth)}`"
        if tag == "underline":
            return f"<u>{self._extract_text(node, depth)}</u>"
        if tag == "strike":
            return f"~~{self._extract_text(node, depth)}~~"

        if tag == "ext-link":
            href = node.get("{http://www.w3.org/1999/xlink}href") or node.get("href")
            text = self._extract_text(node, depth) or href
            return f"[{text}]({href})" if href else text

        if tag == "xref":
            rid = node.get("rid", "")
            text = self._extract_text(node, depth)
            return f"[{text}](#{rid})" if node.get("ref-type") in ("bibr", "fig", "table", "fn", "table-fn") else text

        if tag == "email":
            return f"<{self._extract_text(node, depth)}>"
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
            if not title_text:
                title_text = tag.capitalize()
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
            while "\n\n\n" in text:
                text = text.replace("\n\n\n", "\n\n")
            text = text.replace("\n", "\n> ")
            return f"\n\n> {text}\n\n"

        if tag == "fig-group":
            return "".join(self._format_node(fig, depth) for fig in node.findall("fig"))

        if tag == "fig":
            fig_id = node.get("id", "")
            label_node = node.find("label")
            label = " ".join(self._extract_text(label_node, depth).split()).rstrip(".:") if label_node is not None else "Figure"
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
            caption = " ".join(self._extract_text(caption_node, depth).split()) if caption_node is not None else "Media File"
            full_href = self._resolve_image_href(href)
            return f"\n\n<a id=\"{media_id}\"></a>\n[📺 {caption}]({full_href})\n\n" if full_href else ""

        if tag == "inline-graphic":
            href = node.get("{http://www.w3.org/1999/xlink}href") or node.get("href")
            full_href = self._resolve_image_href(href)
            return f"![]({full_href})" if full_href else ""

        if tag == "supplementary-material":
            label_node = node.find("label")
            label = " ".join(self._extract_text(label_node, depth).split()) if label_node is not None else "Supplementary Material"
            media = node.find(".//media")
            href = (media.get("{http://www.w3.org/1999/xlink}href") or media.get("href")) if media is not None else None
            full_href = self._resolve_image_href(href) if href else ""
            md = f"\n\n**{label}**\n\n"
            if full_href:
                md += f"[Download supplementary file]({full_href})\n\n"
            return md

        if tag == "table-wrap":
            table_id = node.get("id", "")
            label = " ".join(self._extract_text(node.find("label"), depth).split()) if node.find("label") is not None else ""
            caption = " ".join(self._extract_text(node.find("caption"), depth).split()) if node.find("caption") is not None else ""
            table_node = node.find(".//table")
            table_html = self._table_to_html(table_node) if table_node is not None else ""

            md = f"\n\n<a id=\"{table_id}\"></a>\n"
            if label:
                md += f"**{label}**\n"
            if caption:
                md += f"*{caption}*\n\n"
            if table_html:
                md += f"{table_html}\n"

            # Table footnotes with anchors so in-table xref links resolve
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
            if formula.startswith("<![CDATA["):
                formula = formula[9:-3]
            if formula:
                return f"\n\n$$\n{formula}\n$$\n\n" if tag == "disp-formula" else f"${formula}$"
            return self._extract_text(node, depth)

        if tag == "break":
            return "\n\n"
        if tag == "fn":
            fn_id = node.get("id", "")
            return f"<a id=\"{fn_id}\"></a>^[{self._extract_text(node, depth).strip()}]"
        if tag == "title":
            return ""  # handled by parent section

        # Fallback for unhandled tags
        return self._extract_text(node, depth)

    """HELPER"""
    def _extract_text(self, node, depth=2) -> str:
        text = []
        if node.text:
            text.append(node.text)
        for child in node:
            child_text = self._format_node(child, depth)
            # Prevent "smashed" text for adjacent tags that lack spaces in XML
            if isinstance(child.tag, str) and child.tag in (
                "pub-id", "year", "volume", "issue", "fpage", "lpage",
                "elocation-id", "ext-link", "xref", "email",
            ):
                if not child_text.endswith(" "):
                    child_text += " "
            text.append(child_text)
            if child.tail:
                text.append(child.tail)
        return "".join(text)

    def _resolve_image_href(self, href: str) -> str:
        """Maps a relative JATS graphic href to its hosted location."""
        if not href:
            return ""
        if href.startswith(("http://", "https://")):
            return href
        if self.image_base_url and self._article_dir:
            return f"{self.image_base_url}/{self._article_dir}/{href}"
        if self.image_base_url:
            return f"{self.image_base_url}/{href}"
        return href

    def _table_to_html(self, table_node) -> str:
        if table_node is None:
            return ""
        table_copy = copy.deepcopy(table_node)
        for elem in table_copy.getiterator():
            if not hasattr(elem.tag, "find"):
                continue
            if isinstance(elem.tag, str) and elem.tag.startswith("{"):
                elem.tag = elem.tag.split("}", 1)[1]
            attribs = dict(elem.attrib)
            elem.attrib.clear()
            for k, v in attribs.items():
                new_k = k.split("}", 1)[1] if isinstance(k, str) and k.startswith("{") else k
                elem.attrib[new_k] = v
        return etree.tostring(table_copy, method="html", encoding="unicode", pretty_print=True)

    def _get_text(self, node, xpath: str) -> str:
        el = node.find(xpath)
        return "".join(el.itertext()).strip() if el is not None else ""