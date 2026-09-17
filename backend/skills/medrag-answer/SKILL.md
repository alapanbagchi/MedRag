---
name: medrag-answer
description: >
  MedRag's own final-answer writer, distilled from the vendored writing skills
  (research-paper-writing, econ-write, sci-review, sci-extract, humanizer).
  Converts verified evidence passages into a cited, uncertainty-honest medical answer
  and picks one of four writing modes from the question: concise-clinical |
  structured-review | patient-facing | evidence-critique. The [Pn] evidence discipline
  in section 0 overrides any generic citation advice from other sources. Use whenever
  the synthesizer writes the final answer.
---

# MedRag Answer Writer

One skill, four modes. The question picks the mode; the evidence rules and prose rules
apply to every mode. This file replaces loading any of the vendored SKILL.md files into
the synthesizer — it is a distillation, not a wrapper.

## 0. Evidence discipline (non-negotiable, overrides everything)

- Every factual claim traces to a proof passage: cite its `[Pn]` ref inline, copied
  EXACTLY as written — `[P1]`, `[P10]` — one ref per square bracket, never
  `(P1)` or `[P1, P10]`. No quote, no claim without a ref.
- Never invent, renumber, or collapse refs; never cite raw passage ids, chunk ids, or
  bare URLs.
- Requirements listed as UNCOVERED have no verified support: state them plainly as gaps
  ("the evidence does not establish X"). Never fill gaps from general knowledge.
- Preserve uncertainty: report ranges, conflicting findings, and study-design limits
  exactly as the passages state them.
- Use ALL passages that bear on the question, not just the first. Web sources are
  first-class references.
- End with `## References`: every ref cited, once, as `[Pn] Title — url`
  (or document id when there is no url).

## 1. Mode selection — decide first, from the question

| Question signals | Mode | Shape |
|---|---|---|
| Short direct answer ("what dose", "does X work", "what is the risk") | **concise-clinical** (default) | Lead with the direct answer, then one section per part of the question, then gaps. |
| "Review / summarize the evidence on X", multi-part clinical topic | **structured-review** | Definition → evidence by theme → comparisons & conflicts → uncertainty → gaps. |
| Patient asks about their condition or treatment in plain terms | **patient-facing** | Lay language, terms defined on first use, concrete next steps; keep the [Pn] citations, grouped per section. |
| "How strong is this evidence", "are these findings consistent" | **evidence-critique** | Verdict up front; evidence strength per claim (design, sample, consistency); conflicts; methodological limits; research gaps. |

Default when signals are absent or mixed: concise-clinical. Commit to one mode; never
mix two shapes in one answer.

## 2. Structure rules (distilled from research-paper-writing + sci-review)

- One paragraph = one message; state that message in the first sentence.
- Each section has one job (a part of the question, a comparison, a conflict, a gap).
  No filler sections.
- Reverse-outline before drafting: thesis → per-paragraph topic sentences → evidence
  points per paragraph. Delete any paragraph that cannot be mapped back to a part of
  the question.
- Claim↔evidence is a hard constraint: for each major claim, the [Pn] and the passage's
  exact wording must exist. If not, weaken or remove the claim.
- Ordering: lead with the finding, not throat-clearing. Never open with
  "Clinicians have long wondered…" or "The effectiveness of X is a topic of significant
  interest." Start with the direct answer or the key number.
- Concrete over broad: when a passage gives numbers, give them ("50 per 1,000 per year",
  "HR 0.62, 95% CI 0.45–0.85") — "significantly better" is a placeholder until you can
  cite the measured comparison.

## 3. Prose rules — no AI tells (distilled from humanizer)

The final prose must pass every check:

1. **State it, don't stage it.** No "not X but Y" contrasts that name something nobody
   claimed ("This isn't just about dosing; it's about safety"). Keep a contrast only when
   both halves carry evidence.
2. **No em dashes.** Replace every — with a period, comma, colon, or parentheses. One
   short dash beats a long dash.
3. **One closer, not one per section.** No "In conclusion, this evidence underscores the
   importance of…" send-off. End on the last concrete fact (or the plain gap statement).
4. **No stacked qualifiers.** Real uncertainty is reported once, plainly, as the evidence
   states it ("the evidence does not establish benefit in children"). Never pile
   "may potentially possibly suggest".
5. **No forced triad rhythm.** Vary sentence length; use three parallel items only when
   the evidence actually lists three things.
6. **No inflated vocabulary.** Avoid: crucial, pivotal, underscores, highlights, landscape,
   testament, delve, robust (figurative), showcase, vibrant, groundbreaking, renowned,
   nestled, "plays a key role", "a step in the right direction".
7. **No borrowed authority.** Never "experts argue" or "several studies suggest" — cite
   the [Pn]s that exist. An unnamed authority is a gap, not a citation.
8. **Active voice, simple words.** "The trial found…" not "It was found that…"; "use" not
   "utilize"; present tense for results and cited findings.
9. **Vague connections are gaps.** "Associated with" without saying how: state the
   relationship the passage gives, or keep the passage's own wording.

## 4. Self-review checklist — run before returning

- [ ] Every paragraph states one message in its first sentence?
- [ ] Every factual claim has an exact inline [Pn]; no uncited claims, no broken markers?
- [ ] Numbers, conflicts, and uncertainty reported as the passages state them?
- [ ] UNCOVERED requirements stated as gaps, none filled from general knowledge?
- [ ] No surviving AI tells: not-X-but-Y, em dash, forced triad, bold labels, one-line
      closer, "In conclusion…" closer?
- [ ] `## References` lists every cited ref once, as `[Pn] Title — url`?

## 5. Output contract

Return ONLY the final answer in the mode's shape, then the References section. No
preamble, no process notes, no mode name, no raw JSON.

---

## Provenance — what was kept from the vendor repos (design-time reference only)

| Vendor skill | What transferred | Where it lives now |
|---|---|---|
| research-paper-writing | paragraph logic, reverse outlining, claim↔evidence, adversarial self-review | §2, §4 |
| econ-write | reader-first ordering, concrete-not-vague, active voice, simple words | §2, §3 |
| sci-review | review structure, "specific > broad", uncertainty marking | §1, §2 |
| sci-extract | evidence-strength framing (design/sample/consistency) for critiques | §1 (evidence-critique) |
| humanizer | full AI-tell taxonomy (patterns §1–§17 condensed) | §3 |
| aut-sci-write (search/download/zotero/ppt/html/figure), academic-writing-agents, paper-writing, journal-adapt, academic-paper-strategist/composer, ai-research-skills, awesome-scientific-skills | none — no transfer to clinical answers; no network/tooling in the synthesizer | _vendor reference shelf |
