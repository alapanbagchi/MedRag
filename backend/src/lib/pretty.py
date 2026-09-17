"""Pretty terminal output for the agents CLI (stdlib only, no new deps).

Colors auto-disable when stdout is not a TTY or NO_COLOR is set, so
captured logs and pytest output stay plain.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap

# ANSI codes (kept minimal on purpose).
_BOLD = "1"
_DIM = "2"
_CYAN = "36"
_GREEN = "32"
_YELLOW = "33"
_MAGENTA = "35"
_RED = "31"

_FALLBACK_WIDTH = 80
_MAX_BODY_WIDTH = 100
_TOOL_PREVIEW_LIMIT = 800


def supports_color() -> bool:
    """Return True when ANSI colors are wanted on stdout."""
    if os.environ.get("NO_COLOR"):
        return False
    stream = sys.stdout
    return hasattr(stream, "isatty") and stream.isatty()


def style(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes, or return it unchanged without color support."""
    if not codes or not supports_color():
        return text
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"


def term_width(default: int = _FALLBACK_WIDTH) -> int:
    """Terminal width clamped to a readable range for body text."""
    try:
        width = shutil.get_terminal_width().columns
    except Exception:
        width = default
    return max(60, min(width, _MAX_BODY_WIDTH))


def rule(title: str = "", width: int | None = None) -> str:
    """A horizontal divider, optionally with an embedded title."""
    width = width or term_width()
    if not title:
        return "─" * width
    title = f" {title.strip()} "
    if len(title) >= width:
        return title.strip()
    side = (width - len(title)) // 2
    bar = "─" * side
    extra = "─" if (len(title) + side * 2) < width else ""
    return f"{bar}{title}{bar}{extra}"


def badge(deep: bool) -> str:
    """Colored [deep]/[shallow] tag for a plan item."""
    if deep:
        return style("[deep]", _BOLD, _MAGENTA)
    return style("[shallow]", _DIM)


def fold_line(prefix: str, text: str, width: int | None = None) -> str:
    """Fold a long 'prefix + text' line with continuation indent.

    Short lines stay on one row so greppable phrases (e.g.
    'deep agent replied: Hi!') remain contiguous.
    """
    width = width or term_width()
    text = " ".join(str(text).split())
    if len(prefix) + len(text) <= width:
        return f"{prefix}{text}"
    wrapped = textwrap.fill(
        text,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * min(len(prefix), 12),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped


def summarize_tool_text(content: object, limit: int = _TOOL_PREVIEW_LIMIT) -> str:
    """Trim long tool payloads, noting how much was cut."""
    text = content if isinstance(content, str) else str(content)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"
