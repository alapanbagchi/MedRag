// Frontend render test: stubs the /v1/chat/stream fetch with a canned xdeep
// NDJSON sequence (payloads mirror the real backend events) and verifies
// every UI state renders: plan card, tool-card steps, sources, citations,
// completion + panel collapse. No backend / no LLM needed.
import puppeteer from "puppeteer-core";

const EVENTS = [
  { type: "status", stage: "understanding", message: "understanding the question" },
  { type: "pipeline", event: "progress", fields: { msg: "[decompose] orchestrator decomposing: Which blood pressure measurements define hypertension?" } },
  { type: "pipeline", event: "decompose_done", fields: { requirements: [
    { id: "R1", text: "Which specific blood pressure measurements define hypertension in routine adult care?", target_n: 3 },
    { id: "R2", text: "What do the WHO and ESC/ESH guidelines recommend for diagnostic thresholds?", target_n: 3 },
  ] } },
  { type: "status", stage: "decomposing", message: "2 independent tasks" },
  { type: "pipeline", event: "search_round", fields: { requirement_id: "R1", round_no: 1, queries: ["hypertension diagnostic thresholds", "systolic diastolic cutoff adults"], source: "corpus" } },
  { type: "pipeline", event: "query_start", fields: { requirement_id: "R1", query: "hypertension diagnostic thresholds", source: "corpus" } },
  { type: "pipeline", event: "retrieved", fields: { requirement_id: "R1", query: "hypertension diagnostic thresholds", count: 5, method: "pgfts+pgvector" } },
  { type: "status", stage: "retrieving", message: "retrieved 5 candidate(s)" },
  { type: "pipeline", event: "verdict", fields: { evidence_id: "R1.E1-1", requirement_id: "R1", document_id: "PMC11717708", status: "accepted", support: "supports", confidence: 1.0, note: "Passage states the 140/90 mmHg threshold directly." } },
  { type: "pipeline", event: "verdict", fields: { evidence_id: "R1.E1-2", requirement_id: "R1", document_id: "PMC11825899", status: "rejected", support: "neutral", confidence: 0.8, note: "Background only." } },
  { type: "status", stage: "reranking", message: "verifying evidence" },
  { type: "pipeline", event: "evidence_state", fields: { requirement_id: "R1", verified: 2, rejected: 3 } },
  { type: "pipeline", event: "gap_probe", fields: { requirement_id: "R1", question: "What are the WHO classification categories with numerical thresholds?" } },
  { type: "pipeline", event: "web_search_started", fields: { requirement_id: "R1", query: "WHO hypertension classification 140 90 categories", base: "http://127.0.0.1:8888" } },
  { type: "pipeline", event: "web_search_done", fields: { requirement_id: "R1", query: "WHO hypertension classification 140 90 categories", count: 6, urls: [] } },
  { type: "pipeline", event: "web_fetch", fields: { requirement_id: "R1", url: "https://www.who.int/news-room/fact-sheets/detail/hypertension", chars: 8002, ok: true } },
  { type: "pipeline", event: "reliability_verdict", fields: { requirement_id: "R1", url: "https://www.who.int/news-room/fact-sheets/detail/hypertension", reliability: "high", note: "Official WHO guidance." } },
  { type: "status", stage: "verifying", message: "checking contradictions" },
  { type: "pipeline", event: "contradiction", fields: { n: 1, c1: "guideline A vs guideline B" } },
  { type: "pipeline", event: "resolution", fields: { contradiction_id: "C1", status: "resolved", note: "harmonized by patient subgroups" } },
  { type: "pipeline", event: "synthesis_start", fields: { verified: 4, contradictions: 1, gap_resolutions: 0 } },
  { type: "status", stage: "synthesizing", message: "writing the answer" },
  { type: "sources", sources: [
    { id: "PMC11717708", pmcid: "PMC11717708", title: "Epidemiology and diagnosis of adult hypertension in primary care", journal: "PMC · BMC Fam Pract", year: 2024, snippet: "Hypertension is defined by office readings of 140/90 mmHg or higher on repeated measurement.", isWeb: false },
    { id: "who-factsheet", title: "Hypertension — fact sheet", journal: "WHO", year: 2025, url: "https://www.who.int/news-room/fact-sheets/detail/hypertension", snippet: "An adult is considered hypertensive when systolic is ≥ 140 mmHg or diastolic ≥ 90 mmHg.", isWeb: true },
    { id: "PMC11825899", pmcid: "PMC11825899", title: "ESC/ESH 2023 guidelines on the management of arterial hypertension", journal: "PMC · Eur Heart J", year: 2023, snippet: "Thresholds for diagnosis remain 140/90 mmHg, with office and out-of-office confirmation.", isWeb: false },
  ] },
  { type: "token", content: "## Answer\n\nAdult hypertension is defined by sustained office blood pressure readings of **≥ 140 mmHg systolic or ≥ 90 mmHg diastolic** [1]. The WHO fact sheet confirms the same cutoffs, adding that confirmation on repeated measurement is required before diagnosis [2]. European ESC/ESH guidelines retain the 140/90 mmHg diagnostic threshold and emphasize out-of-office confirmation for classification [3].\n\n## Limitations\n\n- Definitions differ slightly across guidelines in the classification of *elevated* (120–139/80–89 mmHg) versus *hypertensive* ranges." },
  { type: "pipeline", event: "synthesis_done", fields: { answer_len: 512, verified: 4, gaps: 0, contradictions: 1 } },
  { type: "done", timingMs: 21400 },
];

const STUB = `
(() => {
  const EVENTS = ${JSON.stringify(EVENTS)};
  const realFetch = window.fetch.bind(window);
  window.fetch = async (url, opts) => {
    if (typeof url === "string" && url.includes("/v1/chat/stream")) {
      const enc = new TextEncoder();
      let i = 0;
      const stream = new ReadableStream({
        start(controller) {
          const timer = setInterval(() => {
            if (i >= EVENTS.length) { clearInterval(timer); controller.close(); return; }
            controller.enqueue(enc.encode(JSON.stringify(EVENTS[i]) + "\\n"));
            i++;
          }, 160);
        },
      });
      return new Response(stream, { status: 200, headers: { "Content-Type": "application/x-ndjson" } });
    }
    return realFetch(url, opts);
  };
})();
`.trim();

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
const errs = [];
page.on("console", (m) => {
  if (m.type() === "error") errs.push(m.text().slice(0, 250));
});
page.on("pageerror", (e) => errs.push(`pageerror: ${e.message}`));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

await page.evaluateOnNewDocument(`(() => { ${STUB} })()`);
await page.goto("http://localhost:5174", { waitUntil: "load", timeout: 30000 });
await sleep(800);
await page.type("textarea", "Which blood pressure measurements define hypertension?", { delay: 2 });
await page.screenshot({ path: new URL("../.ui-shots/20-stub-composed.png", import.meta.url).pathname });
await page.keyboard.press("Enter");

// stream finishes in ~4.5s of event ticks
await sleep(7000);

const state = await page.evaluate(() => {
  const text = document.body.innerText;
  return {
    plan: /Research plan/.test(text) && /R1/.test(text) && /R2/.test(text),
    sources: text.includes("Sources") && text.includes("140 mmHg") && text.includes("WHO"),
    answer: text.includes("## Answer") || text.includes("Adult hypertension is defined"),
    citations: /\[\d\]/.test(text),
    collapsed: text.includes("View thoughts"),
  };
});

// expand the thinking panel to inspect the tool cards
const expandedClick = await page.evaluate(() => {
  const btn = [...document.querySelectorAll("button")].find((b) => (b.textContent ?? "").includes("View thoughts"));
  if (btn) { btn.click(); return true; }
  return false;
});
await sleep(500);
const cards = await page.evaluate(() => {
  const text = document.body.innerText;
  return {
    expanded: text.includes("Hide thoughts"),
    toolCards: ["Searching literature", "Web search", "Fetching source", "Reliability check", "Evidence verdict", "Verified evidence", "Contradiction", "Gap probe", "Gap resolved", "Synthesizing answer", "Decomposing the question", "Research task"].filter((s) => text.includes(s)),
    rows: (/retrieved 5 candidate/i.test(text) || /rejected|accepted/i.test(text)),
    stage: text.includes("understanding") || text.includes("Complete"),
  };
});
console.log(JSON.stringify({ ...state, panelExpanded: expandedClick, ...cards }, null, 2));
console.log("console errors:", errs.length ? errs.slice(0, 4) : "none");
await page.screenshot({ path: new URL("../.ui-shots/21-stub-complete.png", import.meta.url).pathname });
const chips = await page.evaluate(() => document.querySelectorAll(".cite-chip").length);
console.log("cite chips in answer:", chips);
await page.screenshot({ path: new URL("../.ui-shots/22-stub-expanded.png", import.meta.url).pathname });
await browser.close();