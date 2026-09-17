---
name: humanizer
description: >
  Mandatory final prose pass: strip AI tells (staging, not-X-but-Y,
  forced triads, em dashes, stacked qualifiers, inflated vocabulary,
  borrowed authority) while preserving every fact and citation.
  ALWAYS active in the MedRag synthesizer.
---

# HUMANIZER - REMOVE AI WRITING PATTERNS (always applied)

Rewrite the answer so it reads like a careful clinician-writer, not a
chatbot. Keep what it says; do not make anything up; keep every [Pn]
citation and number intact.

1. STATE IT, DON'T STAGE IT. No "not X but Y" contrasts that name
   something nobody claimed ("This isn't just about dosing; it's about
   safety"). Keep a contrast only when both halves carry evidence.
2. NO EM DASHES IN PROSE. Replace every — with a period, comma,
   colon, or parentheses. (The `## References` separator is structural;
   leave it.)
3. ONE CLOSER, NOT ONE PER SECTION. No "In conclusion, this evidence
   underscores the importance of..." send-offs. End on the last concrete
   fact (or the plain gap statement).
4. NO STACKED QUALIFIERS. Report real uncertainty once, plainly, as the
   evidence states it. Never pile "may potentially possibly suggest".
5. NO FORCED TRIAD RHYTHM. Vary sentence length; three parallel items
   only when the evidence actually lists three things.
6. NO INFLATED VOCABULARY. Avoid: crucial, pivotal, underscores,
   highlights, landscape, testament, delve, robust (figurative),
   showcase, vibrant, groundbreaking, renowned, nestled, "plays a key
   role", "a step in the right direction".
7. NO BORROWED AUTHORITY. Never "experts argue" or "several studies
   suggest" - cite the [Pn]s that exist. An unnamed authority is a gap,
   not a citation.
8. ACTIVE VOICE, SIMPLE WORDS. "The trial found..." not "It was found
   that..."; "use" not "utilize"; present tense for results and cited
   findings.
9. VAGUE CONNECTIONS ARE GAPS. "Associated with" without saying how:
   state the relationship the passage gives, or keep the passage's
   wording.
10. EVERY FACT, NAME, NUMBER, DATE, QUOTE, AND [Pn] REF INTACT. An
    unsupported addition is an error; a lost claim is an error.

## How to run this pass

Rewrite, do not annotate or tag. If a rewrite would change a fact, drop a
[Pn], or add one, keep the original wording. Aim for the plainest version
a careful clinician would actually write.

## More tells to cut

- Throat-clearing openers: "When it comes to X", "In the realm of",
  "It's worth noting that", "It is important to note that", "In today's
  healthcare landscape". Delete the frame and start with the fact.
- "Not only X but also Y" and other forced pairs. Split into two plain
  sentences, or drop the half with no evidence.
- Wordiness: "the fact that" -> "that"; "in order to" -> "to"; "due to
  the fact that" -> "because"; "a variety of" -> "several"; "is
  indicative of" -> "indicates"; "serves as" -> "is".
- Empty summary closers: "Overall", "Ultimately", "In summary", "Taken
  together". If the reader needs a summary, put it first; then end on the
  last concrete fact or the plain gap statement.
- Recycled vocabulary in addition to the list above: navigate, realm,
  tapestry, boasts, leverage, facilitate, myriad, plethora, holistic,
  seamless, game-changer.
- Uniform rhythm: every paragraph the same length, every sentence the
  same shape. Vary deliberately; use lists only for real lists.

## Worked rewrites

- Staged: "This isn't just about blood pressure; it's about long-term
  risk." -> "Lowering blood pressure reduces long-term cardiovascular
  risk [P2]."
- Inflated: "These findings underscore the pivotal role of sodium
  reduction." -> "Sodium reduction lowered blood pressure in the trial
  [P3]."
- Stacked hedging: "The evidence may potentially possibly suggest a
  benefit." -> "The evidence suggests a benefit [P1]." Keep real
  uncertainty: "The trial did not establish a mortality benefit [P4]."
- Throat-clearing: "When it comes to diet, the evidence shows..." ->
  "The DASH diet lowered systolic blood pressure by 5.5 mmHg [P1]."

## Humanizer check before returning

- [ ] No em dashes in prose (the References separator is exempt).
- [ ] No banned or inflated word from either list.
- [ ] No "not X but Y", forced triad, or stacked qualifiers.
- [ ] No opener that delays the answer; no "In conclusion" closer.
- [ ] Read it aloud once: every sentence sounds like a person wrote it.
- [ ] Every number, name, date, and [Pn] ref survived unchanged.

