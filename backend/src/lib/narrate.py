"""Run narration fan-out: terminal and UI thought stream stay 1:1.

Narration sites call :func:`say` instead of ``print``. With no sink
installed (CLI runs, tests) it behaves exactly like ``print``. The
streaming adapter installs a sink per run, so every narrated line is
also forwarded — plain, ANSI-stripped, unfolded — to the UI thought
stream, in the order it was said.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from src.lib.pretty import fold_line

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_Sink = Callable[[str], None]
_sink: ContextVar[_Sink | None] = ContextVar("medrag_narrate_sink", default=None)


def strip_ansi(text: object) -> str:
    """Remove ANSI color codes; terminal chrome never reaches the UI."""
    return _ANSI_RE.sub("", str(text))


def current_sink() -> _Sink | None:
    """The installed forward sink, if any (same task context)."""
    return _sink.get()


@contextmanager
def forwarding(sink: _Sink) -> Iterator[None]:
    """Forward every :func:`say` line to ``sink`` until the block exits."""
    token = install(sink)
    try:
        yield
    finally:
        uninstall(token)


def install(sink: _Sink) -> object:
    """Install a forward sink; pass the token to :func:`uninstall`."""
    return _sink.set(sink)


def uninstall(token: object) -> None:
    """Remove a sink installed with :func:`install`."""
    _sink.reset(token)  # type: ignore[arg-type]


def _forward(line: str) -> None:
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(strip_ansi(line))
    except Exception:  # noqa: BLE001 — narration never breaks a run
        pass


def say(line: object = "", *, forward: bool = True) -> None:
    """Print one line to the terminal; forward it to the UI unless told not to.

    ``forward=False`` is for full payload dumps (passage texts, verdict
    tables, raw replies): terminal-only detail, while the UI thought
    stream keeps just the thoughts about them.
    """
    print(line, flush=True)
    if forward:
        _forward(str(line) + "\n")


def say_chunk(chunk: object) -> None:
    """Print a partial chunk (no newline) and forward it raw (if sunk)."""
    print(chunk, end="", flush=True)
    _forward(str(chunk))


def say_folded(prefix: object, text: object, *, forward: bool = True) -> None:
    """Print ``prefix + text`` folded to terminal width; forward one line."""
    print(fold_line(str(prefix), text), flush=True)
    if forward:
        _forward(f"{prefix}{' '.join(str(text).split())}\n")
