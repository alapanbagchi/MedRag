"""Export the MedRAG codebase into a single, clearly separated Markdown document.

Order: project config -> package source -> scripts -> tests. Binary/build
artifacts (`.venv`, `__pycache__`, `data/`, `chunks/`, `embeddings/`, `index/`,
`eval/` reports) are intentionally excluded.

Usage:
    python scripts/export_code.py [--output OUTPUT.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# (display path, order index) per file. Order matters: config first, then the
# package in dependency order, then scripts, then tests.
FILE_SETS: List[Tuple[str, Iterable[Path]]] = [
    ("Project config", [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / ".gitignore",
    ]),
    ("Package root", sorted((ROOT / "src/medrag").glob("*.py"))),
    ("Retrieval layer", sorted((ROOT / "src/medrag/retrieval").glob("*.py"))),
    ("Scripts", sorted((ROOT / "scripts").glob("*.py"))),
    ("Tests", sorted((ROOT / "tests").glob("test_*.py"))),
]

FENCE_BY_SUFFIX = {
    ".py": "python",
    ".toml": "toml",
    ".md": "markdown",
    ".txt": "text",
}


def fence_lang(path: Path) -> str:
    return FENCE_BY_SUFFIX.get(path.suffix.lower(), "")


def build_document() -> List[str]:
    lines: List[str] = [
        "# MedRAG — Full Code Export",
        "",
        "Single-document export of the MedRAG codebase. Each file is clearly "
        "separated by a heading and a fenced code block. Data artifacts "
        "(`chunks/`, `embeddings/`, `index/`, `data/`, `eval/` reports) and "
        "build outputs (`.venv/`, `__pycache__/`) are excluded.",
        "",
    ]
    for section, files in FILE_SETS:
        lines.append("")
        lines.append(f"## {section}")
        lines.append("")
        for path in files:
            if not path.is_file():
                continue
            lang = fence_lang(path)
            raw = path.read_text(encoding="utf-8")
            # Use a longer fence when the file itself contains backticks
            # (e.g. README.md has ```bash blocks) so the inner fences do not
            # close the outer block early.
            fence = "`" * (4 if "```" in raw else 3)
            lines.append("")
            lines.append(f"### File: `{path.relative_to(ROOT)}`")
            lines.append("")
            lines.append(f"{fence}{lang}")
            lines.extend(raw.rstrip("\n").split("\n"))
            lines.append(fence)
            lines.append("")
    return lines


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(ROOT / "medrag_code_export.md"))
    args = ap.parse_args(argv)

    out = Path(args.output)
    out.write_text("\n".join(build_document()) + "\n", encoding="utf-8")
    total_lines = sum(1 for _ in out.open(encoding="utf-8"))
    print(f"Wrote {out.resolve()} ({total_lines} lines, {out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())