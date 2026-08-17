# plugins/builders/table_builder.py

import re
from lxml import etree
from typing import List, Dict, Any, Optional


class TableBuilder:
    """
    Parses raw HTML tables into a structured grid representation,
    then generates RAG-friendly text chunks (summary + row-level).
    """

    def __init__(self, pmcid: str):
        self.pmcid = pmcid

    def parse(self, table_html: str, caption: str = "", table_id: str = "") -> Optional[Dict[str, Any]]:
        wrapped = f"<root>{table_html}</root>"
        try:
            root = etree.fromstring(wrapped, parser=etree.HTMLParser())
        except Exception:
            return None

        table = root.find(".//table")
        if table is None:
            return None

        # 1. Extract all rows and expand colspan/rowspan into a 2D grid
        raw_rows = []
        for tr in table.xpath(".//tr"):
            row = []
            for cell in tr.xpath(".//th | .//td"):
                text = " ".join(cell.itertext()).strip()
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                row.append({"text": text, "colspan": colspan, "rowspan": rowspan, "is_header": cell.tag == "th"})
            raw_rows.append(row)

        if not raw_rows:
            return None

        max_cols = max(sum(c["colspan"] for c in row) for row in raw_rows)
        grid = [[{"text": "", "is_header": False} for _ in range(max_cols)] for _ in range(len(raw_rows))]

        for r_idx, row in enumerate(raw_rows):
            c_idx = 0
            for cell in row:
                while c_idx < max_cols and grid[r_idx][c_idx]["text"] != "":
                    c_idx += 1
                if c_idx >= max_cols:
                    break

                for i in range(cell["rowspan"]):
                    for j in range(cell["colspan"]):
                        if r_idx + i < len(grid) and c_idx + j < max_cols:
                            grid[r_idx + i][c_idx + j] = {"text": cell["text"], "is_header": cell["is_header"]}
                c_idx += cell["colspan"]

        # 2. Identify Header Rows
        header_rows = []
        for r_idx, row in enumerate(grid):
            if all(c["is_header"] or c["text"] == "" for c in row):
                header_rows.append(r_idx)
            else:
                if not header_rows and all(c["text"] == "" for c in row):
                    continue
                break

        if not header_rows and grid:
            header_rows = [0]

        # Flatten multi-level headers
        columns = []
        if header_rows:
            for c in range(max_cols):
                col_parts = []
                for r in header_rows:
                    val = grid[r][c]["text"].strip()
                    if val and val not in col_parts:
                        col_parts.append(val)
                columns.append(" - ".join(col_parts) if col_parts else f"Column {c + 1}")

        data_start_idx = max(header_rows) + 1 if header_rows else 0

        # 3. Track Group Headers & Build Structured Rows
        groups = []
        structured_rows = []

        for r_idx in range(data_start_idx, len(grid)):
            row_texts = [grid[r_idx][c]["text"].strip() for c in range(max_cols)]

            is_group = bool(row_texts[0]) and all(not t for t in row_texts[1:])

            if is_group:
                groups.append(row_texts[0])
                continue

            context_groups = groups[-2:]
            row_label = row_texts[0]

            row_data = {}
            for c in range(1, max_cols):
                col_name = columns[c] if c < len(columns) else f"Column {c + 1}"
                if row_texts[c]:
                    row_data[col_name] = row_texts[c]

            if row_data:
                structured_rows.append({
                    "groups": context_groups,
                    "label": row_label,
                    "data": row_data
                })

        return {
            "table_id": table_id,
            "caption": caption,
            "columns": columns,
            "groups": list(dict.fromkeys(groups)),
            "rows": structured_rows
        }

    def generate_chunks(self, parsed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not parsed_table:
            return []

        chunks = []
        t_id = parsed_table["table_id"]
        caption = parsed_table["caption"]

        # Summary chunk
        summary_text = f"Table {t_id} ({self.pmcid}): {caption}\n"
        summary_text += f"Columns: {', '.join(parsed_table['columns'][1:])}\n"
        if parsed_table["groups"]:
            summary_text += f"Categories: {', '.join(parsed_table['groups'])}\n"

        chunks.append({
            "chunk_type": "table_summary",
            "text": summary_text.strip(),
            "metadata": {"table_id": t_id, "pmcid": self.pmcid}
        })

        # Row-level chunks
        for row in parsed_table["rows"]:
            group_path = " > ".join(row["groups"]) if row["groups"] else "General"
            row_text = f"Table {t_id} ({self.pmcid}), Group: {group_path}\n"
            row_text += f"Row: {row['label']}\n"
            for col, val in row["data"].items():
                row_text += f"{col}: {val}\n"

            chunks.append({
                "chunk_type": "table_row",
                "text": row_text.strip(),
                "metadata": {
                    "table_id": t_id,
                    "pmcid": self.pmcid,
                    "row_label": row["label"],
                    "group_path": group_path
                }
            })

        return chunks
