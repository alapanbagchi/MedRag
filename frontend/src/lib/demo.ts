import type { Source } from "./types";
import type { StepArgs } from "./xdeep";

/** Default demo question (also used by the "replay" entry). */
export const DEMO_QUESTION =
  "Can AI replace radiologists? What does the medical literature say?";

/**
 * Master deep-agent thinking trace, streamed line by line while running.
 * Shown inside the "thinking" panel as the agent reasons.
 */
export const DEMO_THOUGHTS: string[] = [
  "Understanding the question — this asks for a comparative evidence synthesis, not a single-paper lookup.",
  "Decomposing into sub-tasks: (1) diagnostic accuracy of AI vs radiologists, (2) workflow and workforce effects, (3) limits and failure modes.",
  "Planning retrieval: PMC open-access corpus first, then high-impact journals for recent trials.",
  "Querying local PMC index for radiology AI diagnostic accuracy studies…",
  "Reranking 42 candidate passages down to the strongest 10 sources.",
  "Cross-checking claims across sources — flagging one contradiction on mammography recall rates for the verdict step.",
  "Synthesizing a structured answer with inline citations.",
];

/** Scripted tool-call steps (tool-call arrangement around the thinking). */
export const DEMO_STEPS: StepArgs[] = [
  { kind: "decompose", label: "Decomposing the question", detail: "3 research tasks planned", done: true },
  { kind: "retrieve", label: "Searching literature", detail: "PMC index · “AI diagnostic accuracy radiology”", done: true },
  { kind: "retrieve", label: "Searching literature", detail: "PMC index · “radiologist workforce workflow AI”", done: true },
  { kind: "web_search", label: "Web search", detail: "Recent NEJM / Lancet trials on AI mammography", done: true },
  { kind: "reliability", label: "Reliability check", detail: "10 sources verified · 1 contradiction flagged", done: true },
  { kind: "synthesize", label: "Synthesizing answer", detail: "Structured breakdown with citations", done: true },
];

/**
 * 10 demo sources. `pmcid` set = local PMC corpus (gets the local PMC logo);
 * otherwise the badge shows the publisher / website logo.
 */
export const DEMO_SOURCES: Source[] = [
  {
    id: "pmc-demo-1",
    pmcid: "PMC10000019",
    title: "Deep learning for chest radiograph diagnosis: a multicentre validation study",
    journal: "PMC · Radiology: Artificial Intelligence",
    year: 2024,
    score: 0.94,
    snippet: "Model performance was non-inferior to board-certified radiologists across three health systems…",
  },
  {
    id: "pmc-demo-2",
    pmcid: "PMC10001459",
    title: "AI-assisted mammography screening: recall rates and cancer detection",
    journal: "PMC · JAMA Network Open",
    year: 2024,
    score: 0.91,
    snippet: "AI assistance reduced recall rates while maintaining cancer detection in a prospective cohort…",
  },
  {
    id: "pmc-demo-3",
    pmcid: "PMC10002462",
    title: "Workflow impact of AI triage in emergency head CT interpretation",
    journal: "PMC · Nature Digital Medicine",
    year: 2023,
    score: 0.88,
    snippet: "Time-to-critical-finding notification fell 41% after AI triage deployment…",
  },
  {
    id: "pmc-demo-4",
    pmcid: "PMC10003314",
    title: "Failure modes of diagnostic AI under distribution shift",
    journal: "PMC · Medical Image Analysis",
    year: 2023,
    score: 0.86,
    snippet: "Performance degraded on out-of-distribution scanners, underscoring the need for human oversight…",
  },
  {
    id: "pmc-demo-5",
    pmcid: "PMC10004280",
    title: "Radiologist workforce projections and the role of automation",
    journal: "PMC · Health Affairs",
    year: 2024,
    score: 0.83,
    snippet: "Surveyed departments report AI absorbing repetitive reads while hiring for complex procedures…",
  },
  {
    id: "pmc-demo-6",
    pmcid: "PMC10005889",
    title: "Ethical and liability frameworks for autonomous diagnostic tools",
    journal: "PMC · Bioethics",
    year: 2024,
    score: 0.8,
    snippet: "Current liability regimes assume a responsible physician in the loop…",
  },
  {
    id: "web-demo-1",
    title: "Can AI replace radiologists?",
    journal: "NEJM AI",
    year: 2024,
    score: 0.78,
    url: "https://ai.nejm.org/",
    snippet: "Editorial: augmentation, not replacement, matches the trial evidence so far.",
  },
  {
    id: "web-demo-2",
    title: "The future of AI in the reading room",
    journal: "The Lancet Digital Health",
    year: 2024,
    score: 0.76,
    url: "https://www.thelancet.com/digital-health",
    snippet: "Commentary on prospective trials and deployment lessons.",
  },
  {
    id: "web-demo-3",
    title: "Ethics of AI: a double-edged sword in clinical imaging",
    journal: "Harvard Business Review",
    year: 2023,
    score: 0.71,
    url: "https://hbr.org/",
    snippet: "How automation shifts, rather than erases, expert work.",
  },
  {
    id: "web-demo-4",
    title: "How AI is transforming healthcare delivery",
    journal: "Forbes",
    year: 2024,
    score: 0.68,
    url: "https://www.forbes.com/",
    snippet: "Industry view: productivity gains concentrated in high-volume screening.",
  },
];

/** Structured demo answer ( claim markers [n] map to DEMO_SOURCES order ). */
export const DEMO_ANSWER = `AI can replace some radiology tasks, but not radiologists as a whole. [1][2]

**Here's a breakdown:**

**What AI can replace:**
- Repetitive tasks: high-volume screening reads, triage queues, and preliminary measurements. [3]
- Data analysis at scale: flagging critical findings and identifying trends across populations. [1][3]
- Content generation (to an extent): draft reports and structured summaries for sign-off. [2]
- Pattern recognition: nodule and lesion detection matching specialists in controlled studies. [1][2]

**What AI cannot fully replace (at least for now):**
- Human creativity and originality: especially in complex interventions and novel cases. [4]
- Emotional intelligence and empathy: patient communication, consent, and breaking bad news. [5][7]
- Ethical reasoning and moral judgment: liability still assumes a responsible physician in the loop. [6]
- Contextual understanding and common sense: especially on out-of-distribution scanners and edge cases. [4]

> Think of AI as a tool — not a replacement. [7][8]

Just like PACS replaced film but not radiologists, AI is best seen as augmenting human abilities, not replacing the human experience itself. If you're asking this from a career or deployment perspective, I can give you more tailored advice based on your setting. [5][9][10]`;
