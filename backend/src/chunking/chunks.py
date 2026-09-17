"""Chunk record construction: raw facts -> Chunk objects.

The only job: given an id, text and provenance, build the Chunk record that
gets stored. No embedding logic - embedding_text is the chunk text itself
(encoding is a separate step that reads medpat.chunks.embedding_text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.lib.models import CHUNK_VERSION, Chunk
from src.chunking.ids import DocumentState
from src.chunking.parsing import RawBlock, _citation_numbers


@dataclass
class ChunkConfig:
    """Tuning knobs for one chunking run (validated in documents.chunk_document)."""

    max_tokens: int
    hard_max_tokens: int
    split_overlap_sentences: int
    split_overlap_effective: int


class ChunkFactory:
    """Builds Chunk records for one document.

    The per-document DocumentState (doc id, used ids, metadata) drives every
    builder. All builders go through make() so every chunk gets the same
    metadata, position, and shape.
    """

    def __init__(self, state: DocumentState):
        self.state = state

    def make(self, chunk_id: str, text: str, chunk_type: str,
             breadcrumb: List[str], *,
             position: int,
             retrieval_eligible: bool = True,
             parent_id: Optional[str] = None,
             object_id: Optional[str] = None,
             section: Optional[str] = None,
             subsection: Optional[str] = None,
             table_id: Optional[str] = None,
             figure_id: Optional[str] = None,
             figure_link: Optional[str] = None,
             equation_id: Optional[str] = None,
             reference_id: Optional[str] = None,
             row_label: Optional[str] = None,
             group_path: Optional[List[str]] = None,
             citation_refs: Optional[List[str]] = None,
             footnote_refs: Optional[List[str]] = None,
             extra_metadata: Optional[Dict[str, Any]] = None) -> Chunk:
        """Build one Chunk from facts + provenance.

        Normalizes whitespace, defaults section/subsection from the
        breadcrumb, attaches document metadata, and harvests bracketed
        citation numbers for paragraph/list chunks (resolved to reference ids
        later, once the reference list is chunked). embedding_text is the
        chunk text itself - encoding is a separate step.
        """
        text = " ".join((text or "").split()) if text else ""
        if not text:
            text = chunk_id  # never emit an empty chunk
        text = text.strip()
        metadata = dict(self.state.meta)
        metadata["unit_id"] = extra_metadata.pop("unit_id", "") if extra_metadata else ""
        if extra_metadata:
            metadata.update(extra_metadata)
        chunk = Chunk(
            id=chunk_id,
            document_id=self.state.doc_id,
            text=text,
            embedding_text=text,  # the encoder (medpat_embed) reads this column
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
            figure_link=figure_link,
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
        if chunk_type in ("paragraph", "list"):
            # bracketed citation numbers, raw; resolved to ref ids in a
            # post-pass once the reference list has been chunked
            nums = _citation_numbers(text)
            if nums:
                chunk.metadata["citation_numbers"] = nums
        return chunk

    # -- typed builders ------------------------------------------------

    def paragraph(self, text: str, path: List[str], unit_id: str,
                  extra_metadata: Optional[Dict[str, Any]] = None) -> Chunk:
        """One paragraph (or prose piece) chunk."""
        cid = self.state.unique(f"{self.state.doc_id}_{self.state.prose_counter}")
        self.state.prose_counter += 1
        return self.make(cid, text, "paragraph", path,
                         position=self.state.next_pos(),
                         parent_id=unit_id,
                         extra_metadata={**(extra_metadata or {}), "unit_id": unit_id})

    def list_chunk(self, b: RawBlock, path: List[str], unit_id: str) -> Optional[Chunk]:
        """One bullet/ordered list chunk."""
        text = "\n".join(f"- {it}" for it in b.items)
        if not text.strip():
            return None
        cid = self.state.unique(f"{self.state.doc_id}_{self.state.prose_counter}")
        self.state.prose_counter += 1
        return self.make(cid, text, "list", path,
                         position=self.state.next_pos(),
                         parent_id=unit_id,
                         extra_metadata={"unit_id": unit_id})

    def administrative(self, text: str, path: List[str], unit_id: str) -> Chunk:
        """Funding / acknowledgements / notes chunk (stored for completeness,
        never retrieval-eligible so boilerplate stays out of search)."""
        cid = self.state.unique(f"{self.state.doc_id}_{self.state.prose_counter}")
        self.state.prose_counter += 1
        return self.make(cid, text, "administrative", path,
                         position=self.state.next_pos(), parent_id=unit_id,
                         retrieval_eligible=False,
                         extra_metadata={"unit_id": unit_id})

    def reference(self, text: str, path: List[str], raw: str = "",
                  unit_id: str = "") -> Chunk:
        """One bibliography entry chunk (doi/pmid/pmcid extracted for the
        medpat.references table). Never retrieval-eligible."""
        ref_id = f"ref_{self.state.reference_counter}"
        self.state.reference_counter += 1
        doi = re.search(r"https://doi\.org/([^)\s]+)", raw or text)
        pmid = re.search(r"https://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", raw or text)
        pmcid = re.search(r"https://www\.ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)",
                          raw or text)
        return self.make(
            self.state.unique(f"{self.state.doc_id}_{ref_id}"), text, "reference", path,
            position=self.state.next_pos(), parent_id=unit_id or None,
            object_id=ref_id, reference_id=ref_id,
            retrieval_eligible=False,
            extra_metadata={"unit_id": unit_id,
                            "reference_id": ref_id,
                            "doi": doi.group(1) if doi else "",
                            "pmid": pmid.group(1) if pmid else "",
                            "pmcid": pmcid.group(1) if pmcid else ""},
        )

    def figure(self, b: RawBlock, path: List[str], unit_id: str) -> Chunk:
        """One figure chunk (label + caption + footnotes + image link).

        Figure footnotes (italic lines after the caption, see
        parsing._attach_figure_context) are appended to the chunk text so they
        travel with the figure; figure_link is the resolved image URL, stored
        first-class (new medpat.chunks.figure_link column) and in metadata.
        """
        figure_id = self.state.unique(DocumentState.slug(b.label or "figure"))
        if b.label and b.caption and b.caption.lower().startswith(b.label.lower()):
            text = b.caption
        else:
            text = " ".join([b.label or "", b.caption or ""]).strip()
        if not text:
            text = f"Figure {figure_id}"
        fn_text = "\n".join(t if not m else f"[{m}] {t}" for m, t in b.footnotes if t)
        if fn_text:
            text = f"{text}\n{fn_text}"
        return self.make(
            self.state.unique(f"{self.state.doc_id}_{figure_id}"), text, "figure", path,
            position=self.state.next_pos(), parent_id=unit_id,
            object_id=figure_id, figure_id=figure_id, figure_link=b.image_ref,
            extra_metadata={"unit_id": unit_id, "figure_id": figure_id,
                            "figure_link": b.image_ref, "image_ref": b.image_ref},
        )

    def equation(self, b: RawBlock, path: List[str], unit_id: str) -> Chunk:
        """One equation chunk (LaTeX body)."""
        eid = self.state.unique(f"equation_{self.state.equation_counter}")
        self.state.equation_counter += 1
        return self.make(
            self.state.unique(f"{self.state.doc_id}_{eid}"), b.text, "equation", path,
            position=self.state.next_pos(), parent_id=unit_id,
            object_id=eid, equation_id=eid,
            extra_metadata={"unit_id": unit_id, "equation_id": eid, "latex": b.text},
        )
