"""Medpat Postgres persistence: documents, units, chunks, references,
citations, load marks. Idempotent upserts + md_sha256 incremental skip.

No CLI here: the chunking pipeline (documents.chunk_document) is the single
entry point; the CLI calls store_file() per file with one connection.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

DEFAULT_DSN = os.environ.get("MEDPAT_DSN",
                            "postgresql://medpat:CHANGEME@localhost:5433/medpat")


INSERT_DOCUMENT = """
INSERT INTO medpat.documents (
    id, pmcid, title, journal, doi, pmid, publication_date,
    authors, keywords, categories, funding_sources, volume, issue, pages,
    source_format, front_matter, body_md, md_sha256, chunker_report, status,
    chunk_count, unit_count, eligible_count, updated_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s,%s,now())
ON CONFLICT (id) DO UPDATE SET
    pmcid=EXCLUDED.pmcid, title=EXCLUDED.title, journal=EXCLUDED.journal,
    doi=EXCLUDED.doi, pmid=EXCLUDED.pmid, publication_date=EXCLUDED.publication_date,
    authors=EXCLUDED.authors, keywords=EXCLUDED.keywords,
    categories=EXCLUDED.categories, funding_sources=EXCLUDED.funding_sources,
    volume=EXCLUDED.volume, issue=EXCLUDED.issue, pages=EXCLUDED.pages,
    source_format=EXCLUDED.source_format, front_matter=EXCLUDED.front_matter,
    body_md=EXCLUDED.body_md, md_sha256=EXCLUDED.md_sha256,
    chunker_report=EXCLUDED.chunker_report, status=EXCLUDED.status,
    chunk_count=EXCLUDED.chunk_count, unit_count=EXCLUDED.unit_count,
    eligible_count=EXCLUDED.eligible_count, updated_at=now()
"""


INSERT_UNIT = """
INSERT INTO medpat.units (unit_id, document_id, kind, title, breadcrumb, text,
                          parent_unit_id)
VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s)
ON CONFLICT (unit_id) DO UPDATE SET
    kind=EXCLUDED.kind, title=EXCLUDED.title, breadcrumb=EXCLUDED.breadcrumb,
    text=EXCLUDED.text, parent_unit_id=EXCLUDED.parent_unit_id
"""


INSERT_CHUNK = """
INSERT INTO medpat.chunks (
    id, document_id, chunk_type, section, subsection, breadcrumb, parent_id,
    object_id, table_id, figure_id, figure_link, equation_id, reference_id,
    row_label, group_path, text, embedding_text, metadata,
    citation_refs, footnote_refs, document_position, retrieval_eligible)
VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,
       %s::jsonb,%s::jsonb,%s,%s)
ON CONFLICT (id) DO UPDATE SET
    chunk_type=EXCLUDED.chunk_type, section=EXCLUDED.section,
    subsection=EXCLUDED.subsection, breadcrumb=EXCLUDED.breadcrumb,
    parent_id=EXCLUDED.parent_id, object_id=EXCLUDED.object_id,
    table_id=EXCLUDED.table_id, figure_id=EXCLUDED.figure_id,
    figure_link=EXCLUDED.figure_link,
    equation_id=EXCLUDED.equation_id, reference_id=EXCLUDED.reference_id,
    row_label=EXCLUDED.row_label, group_path=EXCLUDED.group_path,
    text=EXCLUDED.text, embedding_text=EXCLUDED.embedding_text,
    metadata=EXCLUDED.metadata, citation_refs=EXCLUDED.citation_refs,
    footnote_refs=EXCLUDED.footnote_refs,
    document_position=EXCLUDED.document_position,
    retrieval_eligible=EXCLUDED.retrieval_eligible
"""


INSERT_REFERENCE = """
INSERT INTO medpat.references (document_id, reference_id, position, text,
                               doi, pmid, pmcid, chunk_id)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (document_id, reference_id) DO UPDATE SET
    position=EXCLUDED.position, text=EXCLUDED.text, doi=EXCLUDED.doi,
    pmid=EXCLUDED.pmid, pmcid=EXCLUDED.pmcid, chunk_id=EXCLUDED.chunk_id
"""


INSERT_CITATION = """
INSERT INTO medpat.chunk_citations (chunk_id, document_id, reference_id)
VALUES (%s,%s,%s) ON CONFLICT DO NOTHING
"""


UPSERT_LOAD_MARK = """
INSERT INTO medpat.load_marks (source_key, status, chunker_config, chunks,
                               units, embeddings, started_at, finished_at, error)
VALUES (%s,%s,%s::jsonb,%s,%s,0,now(),now(),%s)
ON CONFLICT (source_key) DO UPDATE SET status=EXCLUDED.status,
    chunker_config=EXCLUDED.chunker_config, chunks=EXCLUDED.chunks,
    units=EXCLUDED.units, finished_at=now(), error=EXCLUDED.error
"""


def connect(dsn: Optional[str] = None):
    """Open a psycopg2 connection (one per ingest run)."""
    import psycopg2

    return psycopg2.connect(dsn or DEFAULT_DSN)


def compute_sha256(md_text: str) -> str:
    return hashlib.sha256(md_text.encode("utf-8")).hexdigest()


def existing_md_sha256(conn, doc_id: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT md_sha256 FROM medpat.documents WHERE id=%s", (doc_id,))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def _order_units(units):
    """Parents-first deterministic order so parent_unit_id FKs never dangle
    within a document's transaction (the chunker emits child units before
    their parent section)."""
    by_id = {u.unit_id: u for u in units}
    out = []
    seen = set()
    def visit(u):
        if u.unit_id in seen:
            return
        seen.add(u.unit_id)
        pu = by_id.get(u.parent_unit_id) if u.parent_unit_id else None
        if pu is not None:
            visit(pu)
        out.append(u)
    for u in units:
        visit(u)
    return out


def upsert_document(conn, md_text: str, doc_id: str, chunks, units,
                    report: Dict[str, Any]) -> Dict[str, int]:
    """Persist one chunked document (doc + units + chunks + refs + citations).

    Deterministic ids make this a converging upsert; the FTS trigger fills
    chunks.tsv. Returns counts.
    """
    from src.chunking.parsing import _parse_front_matter

    sha = compute_sha256(md_text)
    front, _ = _parse_front_matter(md_text)
    meta = report.get("meta") or {}
    eligible = sum(1 for c in chunks if c.retrieval_eligible)
    with conn:  # one transaction per document (commit on success)
        cur = conn.cursor()
        cur.execute(INSERT_DOCUMENT, (
            doc_id, str(meta.get("pmcid") or doc_id), str(meta.get("title") or ""),
            str(meta.get("journal") or ""), str(meta.get("doi") or ""),
            str(meta.get("pmid") or ""), str(meta.get("publication_date") or ""),
            json.dumps(meta.get("keywords") or [], ensure_ascii=False),
            json.dumps(meta.get("categories") or [], ensure_ascii=False),
            json.dumps(meta.get("authors") or [], ensure_ascii=False),
            json.dumps(meta.get("funding_sources") or [], ensure_ascii=False),
            str(meta.get("volume") or ""), str(meta.get("issue") or ""),
            str(meta.get("pages") or ""), "md",
            json.dumps(front, default=str, ensure_ascii=False),
            md_text, sha, json.dumps(report, default=str, ensure_ascii=False),
            "chunked", len(chunks), len(units), eligible,
        ))
        for u in _order_units(units):
            cur.execute(INSERT_UNIT, (
                u.unit_id, doc_id, u.kind, str(u.title or ""),
                json.dumps(u.breadcrumb, ensure_ascii=False), u.text,
                u.parent_unit_id,
            ))
        for c in chunks:
            cur.execute(INSERT_CHUNK, (
                c.id, doc_id, c.chunk_type, c.section, c.subsection,
                json.dumps(c.breadcrumb, ensure_ascii=False), c.parent_id,
                c.object_id, c.table_id, c.figure_id, c.figure_link,
                c.equation_id, c.reference_id, c.row_label,
                json.dumps(c.group_path, ensure_ascii=False),
                c.text, c.embedding_text,
                json.dumps(c.metadata, default=str, ensure_ascii=False),
                json.dumps(c.citation_refs, ensure_ascii=False),
                json.dumps(c.footnote_refs, ensure_ascii=False),
                c.document_position, bool(c.retrieval_eligible),
            ))
        # normalize reference chunks -> references rows
        for c in chunks:
            if c.chunk_type != "reference" or not c.reference_id:
                continue
            suffix = c.reference_id.split("_", 1)[-1]
            position = int(suffix) + 1 if suffix.isdigit() else 0
            md = c.metadata
            cur.execute(INSERT_REFERENCE, (
                doc_id, c.reference_id, position, c.text,
                str(md.get("doi") or ""), str(md.get("pmid") or ""),
                str(md.get("pmcid") or ""), c.id,
            ))
        # citation edges from bracketed citations
        for c in chunks:
            for ref in c.citation_refs or []:
                cur.execute(INSERT_CITATION, (c.id, doc_id, ref))
        cur.execute(UPSERT_LOAD_MARK, (
            doc_id, "done", json.dumps(report.get("chunker_config") or {},
                                       ensure_ascii=False),
            len(chunks), len(units), None,
        ))
        cur.close()
    return {"chunks": len(chunks), "units": len(units), "eligible": eligible}


def store_file(conn, md_text: str, doc_id: str,
               chunk_fn: Callable[[str, Optional[str]], Tuple[list, list, Dict[str, Any]]],
               overwrite: bool = False) -> Tuple[str, str]:
    """Chunk one Markdown document and upsert it into medpat.

    chunk_fn(text, doc_id) -> (chunks, units, report) - the single entry the
    CLI passes (a closure over chunk_document). md_sha256 skip makes re-runs
    idempotent. Returns (status, message); status in ok | skipped | failed.
    """
    try:
        if not overwrite:
            prev = existing_md_sha256(conn, doc_id)
            if prev is not None and prev == compute_sha256(md_text):
                return "skipped", doc_id
        chunks, units, report = chunk_fn(md_text, doc_id=doc_id)
        if not chunks:
            raise RuntimeError("no chunks produced")
        counts = upsert_document(conn, md_text, doc_id, chunks, units, report)
        return "ok", f"{doc_id}: {counts['chunks']} chunks, {counts['units']} units"
    except Exception as exc:  # noqa: BLE001 - per-file failures
        try:
            conn.rollback()
        except Exception:
            pass
        return "failed", f"{doc_id}: {exc}"


def corpus_stats(conn) -> Dict[str, Any]:
    """Top-level corpus counts (the CLI's summary line)."""
    cur = conn.cursor()
    cur.execute("SELECT count(*), coalesce(sum(chunk_count),0),"
                " coalesce(sum(unit_count),0), coalesce(sum(eligible_count),0)"
                " FROM medpat.documents WHERE status='chunked'")
    docs, chunks, units, eligible = cur.fetchone()
    cur.execute("SELECT count(*) FROM medpat.chunk_embeddings")
    embedded = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM medpat.chunks")
    total_chunks = cur.fetchone()[0]
    cur.close()
    return {"documents": docs, "chunks": total_chunks, "units": units,
            "eligible": eligible, "embedded": embedded}
