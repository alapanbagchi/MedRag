"""Skill library: the MedRag synthesizer's writing skills.

One SKILL.md per writing style lives under backend/skills/writing/<style>/.
Every synthesis prompt is composed as: base (evidence discipline, ALWAYS)
+ the selected writing style (ALWAYS one - default concise-clinical) +
humanizer (ALWAYS). Skills are plain prompt text loaded at build time; the
agents never fetch at runtime. A skill must always be used, and the
humanizer pass is mandatory on every answer.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "writing"

# A skill must always be used: when the question gives no style signal,
# the default style applies. humanizer is mandatory on every answer.
DEFAULT_MODE = "concise-clinical"
MODES = ("concise-clinical", "structured-review", "patient-facing",
         "evidence-critique")
MANDATORY_TAIL = "humanizer"

_PATIENT_SIGNALS = (
    "i have", "my condition", "for me", "my doctor", "should i", "am i",
    "is it safe for me", "in plain", "simple terms", "tell me like",
    "my blood pressure", "my heart", "i am", "i was told",
)
_REVIEW_SIGNALS = (
    "review", "summarize the evidence", "map the evidence", "overview",
    "what does the literature", "state of the art", "comprehensive",
    "evidence on", "what is known about",
)
_CRITIQUE_SIGNALS = (
    "how strong is", "how strong are", "is the evidence", "are the findings",
    "consistent", "critical appraisal", "methodological", "reliable",
    "trustworthy", "limitations of the evidence", "is this evidence solid",
    "does the evidence support",
)


def list_skills() -> list[str]:
    """Names of every installed writing skill (mode + humanizer)."""
    out: list[str] = []
    for child in sorted(SKILLS_DIR.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            out.append(child.name)
    return out


@functools.lru_cache(maxsize=32)
def load_skill(name: str) -> str:
    """Read one skill's SKILL.md (frontmatter stripped, text kept)."""
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"skill not found: {name} ({path})")
    text = path.read_text(encoding="utf-8").strip()
    # Strip YAML frontmatter (--- ... ---) when present.
    if text.startswith("---"):
        m = re.match(r"^---" + chr(10) + r".*?" + chr(10) + r"---" + chr(10) + r"?", text, flags=re.S)
        if m:
            text = text[m.end():]
    return text.strip()


def detect_writing_mode(question: str) -> str:
    """Pick the writing style from the question's signals (heuristic).

    Order matters: concrete patient language beats review keywords
    ("what should I know about X" reads patient-first), and explicit
    appraisal language ("how strong is the evidence") beats a general
    "review" signal.
    """
    q = " " + " ".join((question or "").lower().split()) + " "
    if any(s in q for s in _PATIENT_SIGNALS):
        return "patient-facing"
    if any(s in q for s in _CRITIQUE_SIGNALS):
        return "evidence-critique"
    if any(s in q for s in _REVIEW_SIGNALS):
        return "structured-review"
    return DEFAULT_MODE


def build_synthesis_prompt(
    mode: str | None = None,
    question: str = "",
) -> str:
    """Compose the synthesizer's system prompt from the skill library.

    Always: base + one writing style + humanizer. Unknown modes fall
    back to the default style rather than failing the run.
    """
    mode = mode or detect_writing_mode(question)
    if mode not in MODES:
        mode = DEFAULT_MODE
    parts: list[str] = [load_skill("base")]
    parts.append(load_skill(mode))
    parts.append(load_skill(MANDATORY_TAIL))
    sep = chr(10) + chr(10) + "---" + chr(10) + chr(10)
    return sep.join(parts)


__all__ = ["DEFAULT_MODE", "MODES", "build_synthesis_prompt", "detect_writing_mode",
           "list_skills", "load_skill"]
