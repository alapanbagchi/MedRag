# plugins/builders/ast_builder.py

import re
from typing import List
from core.models import Document, Section, Block


class MarkdownASTBuilder:
    """
    Converts generated Markdown into a structured Document AST.
    """

    # Regex to strip HTML tags but keep their inner text
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

    def build(self, markdown_text: str, metadata: dict) -> Document:
        article_meta = metadata.get("article_meta", {})
        pmcid = article_meta.get("article_ids", {}).get("pmcid", "unknown")
        title = article_meta.get("title", "Untitled Article")

        doc = Document(pmcid=pmcid, title=title, metadata=metadata)

        header_pattern = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
        matches = list(header_pattern.finditer(markdown_text))

        section_stack: List[Section] = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            section_title = match.group(2).strip()

            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
            content = markdown_text[start_idx:end_idx].strip()

            blocks = self._parse_blocks(content)

            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()

            breadcrumb = [s.title for s in section_stack] + [section_title]

            new_section = Section(
                title=section_title,
                level=level,
                breadcrumb=breadcrumb,
                blocks=blocks
            )

            if section_stack:
                section_stack[-1].children.append(new_section)
            else:
                doc.sections.append(new_section)

            section_stack.append(new_section)

        return doc

    def _parse_blocks(self, content: str) -> List[Block]:
        blocks = []
        if not content:
            return blocks

        parts = content.split('\n\n')

        anchor_pattern = re.compile(r'<a\s+id="[^"]*"\s*(?:></a>|/>)\s*', re.IGNORECASE)

        buffer = []
        in_table = False

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Strip anchor tags
            clean_part = anchor_pattern.sub('', part).strip()

            if not clean_part:
                continue

            # For non-table blocks, strip remaining HTML tags
            # (but keep table HTML intact for the TableBuilder)
            if not clean_part.startswith('<table'):
                clean_part = self.HTML_TAG_PATTERN.sub('', clean_part).strip()
                # Also clean up reference-style links like [1](#bib1) -> [1]
                clean_part = re.sub(r'\[([^\]]+)\]\(#[^)]+\)', r'[\1]', clean_part)

            # 1. Handle Tables
            if part.startswith('<table'):
                in_table = True
                buffer = [part]
                if '</table>' in part:
                    in_table = False
                    blocks.append(Block(block_type="table", content=part))
                    buffer = []
            elif in_table:
                buffer.append(part)
                if '</table>' in part:
                    in_table = False
                    blocks.append(Block(block_type="table", content='\n\n'.join(buffer)))
                    buffer = []

            # 2. Handle Figures
            elif clean_part.startswith('!['):
                blocks.append(Block(block_type="figure", content=clean_part))

            # 3. Handle Lists
            elif re.match(r'^[*\-+]\s', clean_part) or re.match(r'^\d+\.\s', clean_part):
                blocks.append(Block(block_type="list", content=clean_part))

            # 4. Handle Formulas
            elif clean_part.startswith('$$'):
                blocks.append(Block(block_type="formula", content=clean_part))

            # 5. Handle Paragraphs
            else:
                blocks.append(Block(block_type="paragraph", content=clean_part))

        if in_table and buffer:
            blocks.append(Block(block_type="table", content='\n\n'.join(buffer)))

        return blocks
