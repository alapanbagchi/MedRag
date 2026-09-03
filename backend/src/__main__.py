"""Entry point: python -m src "query" (singular deepagents flow).

Delegates to the agents research CLI (src.agents.__main__).
"""

from __future__ import annotations

from src.agents.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
