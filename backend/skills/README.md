# MedRag Skill Library — the writing skills the synthesizer always uses

The synthesizer NEVER writes without skills. Every system prompt is composed by
the loader (`backend/src/lib/skills.py`) as:

    base (evidence discipline — ALWAYS)
    + one writing-style skill (ALWAYS — default concise-clinical)
    + humanizer (MANDATORY — every answer)

Skills are plain SKILL.md prompt files under `backend/skills/writing/`. They are
load-time context, never runtime fetches. Missing/unknown styles fall back to
the default so synthesis never breaks.

## Layout

```
backend/skills/
├── README.md            ← you are here
├── registry.json        ← machine-readable index + trigger map
└── writing/
    ├── base/SKILL.md                  ← evidence discipline + structure + markdown presentation + self-review
    ├── concise-clinical/SKILL.md      ← default style: direct answer first
    ├── structured-review/SKILL.md     ← definition → themes → conflicts → uncertainty → gaps
    ├── patient-facing/SKILL.md        ← lay language, defined terms, next steps
    ├── evidence-critique/SKILL.md     ← verdict, strength per claim, limits, gaps
    └── humanizer/SKILL.md             ← mandatory anti-AI-tell prose pass
```

## How a style is picked

`src.lib.skills.detect_writing_mode(question)` (heuristic, deterministic):

| Question signals | Style |
|---|---|
| default / direct asks ("what dose", "does X work") | concise-clinical |
| "review / summarize / map the evidence on X" | structured-review |
| "I have…", "for me", "should I", plain-language asks | patient-facing |
| "how strong is the evidence", "are findings consistent", appraisal | evidence-critique |

The stream adapter detects the mode from the question before the synthesize leg;
the synthesizer (`build_agent(mode=…)`) composes `base + style + humanizer`.

## Calling the skills (the "place to call them")

```python
from src.lib.skills import list_skills, load_skill, detect_writing_mode, build_synthesis_prompt
build_synthesis_prompt()                                  # default style
build_synthesis_prompt(mode="patient-facing")             # explicit style
build_synthesis_prompt(question="I have hypertension…")   # auto-detected style
```

The loader strips YAML frontmatter; the composed prompt always ends with the
humanizer pass (mandatory). `registry.json` mirrors the same contract and is
kept in sync with the loader.

## History

The 10 generic vendor skills were collected, distilled to `medrag-answer`
(now the design notes behind the per-style tree here), then deleted as
useless for clinical answers. This tree is project-owned: every file was
written for MedRag, no third-party licenses, no AGPL, no network deps.
