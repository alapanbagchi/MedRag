"""Table-aware evidence context (V2.1 parts 15-16, 19, 31).

When a candidate is a table row / summary / footnote, the retrieval layer must
assemble the WHOLE table as one contextual evidence object:

    section breadcrumb
    table caption / columns (headers)
    the target row
    relevant footnotes

This is the object scored by MedCPT and the intent layer. It fixes the current
failure where a row like "End-to-end | 2 (8) | 2 (50) | 0.04" is scored in
isolation and the meaning ("percentage", "p-value") - which lives only in the
table header - never reaches the scorer.

The module also implements the TABLE-AWARE requested-field detector (part 19):
whether "0.04" is a p-value depends on the column header, not on regexing the
bare row text. The detection stays deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medrag.retrieval_v2.config import V2Config, DEFAULT_CONFIG

_COLUMN_LIST_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s*(.+)\s*$")


def parse_summary_columns(summary_text: str) -> List[str]:
    """Extract the column headers from a table summary chunk.

    The XML-derived summary text looks like:

        Table 2: Characteristics of patients with and without early re-CoA.

        Columns:
        - Variables
        - No re-CoA (n = 24) [IQR] or n (%)
        - re-CoA (n = 4) [IQR] or n (%)
        - p

    Returns the header labels or [] when the shape is different.
    """
    if not summary_text:
        return []
    lines = summary_text.splitlines()
    cols: List[str] = []
    in_columns = False
    for line in lines:
        s = line.strip()
        low = s.lower()
        if low.startswith("columns") or low.startswith("column headers") or low.startswith("headers"):
            in_columns = True
            continue
        if not in_columns:
            continue
        if not s:
            if cols:
                break
            continue
        m = _COLUMN_LIST_RE.match(s)
        if m:
            cols.append(m.group(1).strip())
        elif cols:
            break
    return cols


def parse_row(row_text: str) -> Tuple[str, Dict[str, str]]:
    """Parse one table row chunk into (variable, {column_label: value}).

    Row text shape:
        Row: End-to-end
        No re-CoA (n = 24) [IQR] or n (%): 2 (8)
        re-CoA (n = 4) [IQR] or n (%): 2 (50)
        p: 0.04
    """
    variable = ""
    cells: Dict[str, str] = {}
    for line in (row_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("row:"):
            variable = s[len("row:"):].strip()
            continue
        idx = s.find(":")
        if idx > 0:
            label = s[:idx].strip()
            value = s[idx + 1:].strip()
            if label:
                cells[label] = value
    return variable, cells


_PVAL_CELL_RE = re.compile(r"^\s*(?:p\s*[=<>]\s*)?(0?\.\d+|\d+\.\d+|\d+)\s*$", re.IGNORECASE)
_PCT_IN_PAREN_RE = re.compile(r"\d+\s*\(\s*\d+\s*\)")
_PCT_SIGN_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_CI_RANGE_RE = re.compile(r"\b(?:\d+(?:\.\d+)?)\s*[-\u2013]\s*(?:\d+(?:\.\d+)?)\b")
_CI_LABEL_RE = re.compile(r"ci|confidence interval", re.IGNORECASE)
_OR_HR_SIGNAL_RE = re.compile(r"^(?:adjusted\s*)?(?:or|hr|rr|aor|ahr|ahrr)\b", re.IGNORECASE)
_OR_WORDED_RE = re.compile(r"\b(?:odds ratio|hazard ratio|relative risk|adjusted or|adjusted hr)\b", re.IGNORECASE)
_PVALUE_RE = re.compile(r"\bp(?:-value)?\s*[=<>]\s*0?\.?\d+|\bp\s*[=<>]\s*0?\.?\d+", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\d+\s*\(\s*\d+\s*\)")
_CI_GENERIC_RE = re.compile(r"95\s*%\s*ci|confidence interval", re.IGNORECASE)
_OR_GENERIC_RE = re.compile(r"\b(?:or|hr|rr)\s*[=:]\s*\d", re.IGNORECASE)
_SAMPLE_RE = re.compile(r"\bn\s*[=:]\s*\d+", re.IGNORECASE)


def detect_table_row_fields(variable: str, cells: Dict[str, str], headers: Sequence[str],
                            footnotes: Sequence[str]) -> Dict[str, bool]:
    """Deterministic, table-aware requested-field detection (part 19).

    A cell is a "percentage" only under a %-labeled column; a cell is a
    "p-value" only under the p column (or an explicit p= value). The variable
    name ("Lateral thoracotomy (%)") is also inspected.
    """
    out: Dict[str, bool] = {
        "percentage": False,
        "p-value": False,
        "confidence_interval": False,
        "effect_estimate": False,
        "sample_size": False,
    }
    # p-value: the "p" column (exact key) or header ending in p
    p_column = next((label for label in headers if re.fullmatch(r"p\s*", label, re.IGNORECASE) or label.strip().lower() in ("p", "p-value", "p value")), None)
    pct_columns = [label for label in headers if "%" in label or "(%" in label.lower() or "percent" in label.lower()]
    if p_column:
        val = cells.get(p_column, "")
        if _PVAL_CELL_RE.match(val.strip().lstrip("=<>")) or _PVALUE_RE.search(val):
            out["p-value"] = True
    # fall back to any explicit p= value in cells
    for label, val in cells.items():
        if re.fullmatch(r"p\s*", label, re.IGNORECASE) or label.strip().lower() in ("p", "p-value", "p value"):
            if _PVAL_CELL_RE.match(val.strip()) or _PVALUE_RE.search(val):
                out["p-value"] = True
    # percentage: under %-header columns or in the variable name
    if pct_columns:
        for label in pct_columns:
            val = cells.get(label, "")
            if _PCT_IN_PAREN_RE.search(val) or _PCT_SIGN_RE.search(val):
                out["percentage"] = True
                break
    if out.get("percentage") is None:
        pass
    if _PCT_IN_PAREN_RE.search(variable) or _PCT_SIGN_RE.search(variable):
        out["percentage"] = True
    # confidence interval
    for label, val in cells.items():
        if _CI_LABEL_RE.search(label) and (_CI_RANGE_RE.search(val) or _CI_GENERIC_RE.search(val)):
            out["confidence_interval"] = True
    if _CI_GENERIC_RE.search(" ".join(footnotes)):
        out["confidence_interval"] = True
    # effect estimate (OR/HR/RR): only short statistic labels or explicit
    # "odds ratio / hazard ratio / relative risk" wording - NOT descriptive
    # headers that merely contain the word "or" (e.g. "or n (%)").
    for label, val in cells.items():
        label_low = label.lower()
        is_short_stat = len(label.strip()) <= 14 and _OR_HR_SIGNAL_RE.match(label.strip())
        is_worded = bool(re.search(r"\b(?:odds ratio|hazard ratio|relative risk)\b", label_low))
        if (is_short_stat or is_worded) and re.search(r"\d", val):
            out["effect_estimate"] = True
    if _OR_WORDED_RE.search(" ".join(footnotes)):
        out["effect_estimate"] = True
    # sample size
    if _SAMPLE_RE.search(" ".join(cells.values())) or _SAMPLE_RE.search(variable):
        out["sample_size"] = True
    return out


def detect_paragraph_fields(text: str) -> Dict[str, bool]:
    """Text-level fallback detector for paragraph / figure candidates."""
    return {
        "percentage": bool(_PERCENT_RE.search(text or "")),
        "p-value": bool(_PVALUE_RE.search(text or "")),
        "confidence_interval": bool(_CI_GENERIC_RE.search(text or "")),
        "effect_estimate": bool(_OR_GENERIC_RE.search(text or "")),
        "sample_size": bool(_SAMPLE_RE.search(text or "")),
    }


def requested_field_fraction(detected: Dict[str, bool], requested: Sequence[str]) -> float:
    """Fraction of the requirement's requested fields detected on the candidate."""
    if not requested:
        return 1.0
    if not detected:
        return 0.0
    hits = sum(1 for f in requested if detected.get(f))
    return min(1.0, hits / len(requested))


def build_table_context(
    doc_index: Any,
    paper_id: str,
    chunk_id_or_node: Any,
    config: Optional[V2Config] = None,
) -> Tuple[Dict[str, Any], str, Dict[str, bool]]:
    """Assemble the table-aware evidence context for one table candidate.

    Returns (table_context_dict, context_text, detected_fields).

    table_context_dict = {
        "table_id", "caption", "headers", "variable", "row_values",
        "rows", "footnotes", "section"
    }
    context_text is the passage scored by MedCPT + intent (part 31):
        Section: ...
        Table: <caption>
        Headers: ...
        Row: <variable> | <values>
        Footnotes: ...
    """
    cfg = config or DEFAULT_CONFIG
    if isinstance(chunk_id_or_node, dict):
        node = chunk_id_or_node
        chunk_id = node.get("chunk_id", "")
    else:
        chunk_id = chunk_id_or_node
        node = doc_index.get_node(chunk_id) or {}

    table_id = node.get("table_id")
    if not table_id:
        return {}, "", {}

    tbl = doc_index.get_table(paper_id, table_id)
    summary = tbl.get("summary")
    rows = tbl.get("rows") or []
    footnotes = tbl.get("footnotes") or []

    summary_text = ""
    if summary:
        summary_text = doc_index.get_text(summary["chunk_id"]) or ""
    headers = parse_summary_columns(summary_text)
    caption = summary_text.splitlines()[0].strip() if summary_text else f"Table {table_id}"

    target_row = next((r for r in rows if r["chunk_id"] == chunk_id), None)
    variable, cells = parse_row(doc_index.get_text(chunk_id) or "")

    # context rows: include the target row plus a bounded window around it
    row_ids = [r["chunk_id"] for r in rows]
    try:
        pos = row_ids.index(chunk_id)
    except ValueError:
        pos = -1
    window = cfg.table_context_row_window
    lo = max(0, pos - window) if pos >= 0 else 0
    hi = min(len(rows), (pos + window + 1) if pos >= 0 else len(rows))
    context_rows = rows[lo:hi]
    context_chunk_ids = [summary["chunk_id"]] if summary else []
    context_chunk_ids += [r["chunk_id"] for r in context_rows]
    context_chunk_ids += [f_["chunk_id"] for f_ in footnotes[: cfg.table_context_footnotes]]
    context_texts = doc_index.get_texts(context_chunk_ids)

    fn_texts = [(doc_index.get_text(f_["chunk_id"]) or "").strip()
                for f_ in footnotes[: cfg.table_context_footnotes]]

    breadcrumb = list(node.get("breadcrumb") or [])
    section_line = " > ".join(breadcrumb) if breadcrumb else (node.get("section") or "")

    lines: List[str] = []
    if cfg.table_context_include_breadcrumb and section_line:
        lines.append("Section: " + section_line)
    if caption:
        lines.append("Table: " + caption)
    if cfg.table_context_include_headers and headers:
        lines.append("Headers: " + " | ".join(headers))
    if variable:
        line = "Row: " + variable
        if cells:
            line += " | " + " | ".join(f"{v}" for v in cells.values())
        lines.append(line)
    else:
        t = context_texts.get(chunk_id, "")
        if t:
            lines.append("Row: " + t.strip())
    if fn_texts:
        lines.append("Footnotes: " + " | ".join(fn_texts))
    context_text = "\n".join(lines)[: cfg.table_context_max_chars]

    detected = detect_table_row_fields(variable, cells, headers, fn_texts)

    table_context: Dict[str, Any] = {
        "table_id": table_id,
        "caption": caption,
        "headers": headers,
        "variable": variable,
        "row_values": cells,
        "context_rows": [
            {"chunk_id": r["chunk_id"], "text": (context_texts.get(r["chunk_id"]) or "").strip()}
            for r in context_rows
        ],
        "footnotes": fn_texts,
        "section": section_line,
    }
    return table_context, context_text, detected
