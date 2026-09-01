// ── Mock research engine ─────────────────────────────────────────────
// Simulates the MedPat backend while it is not connected: emits the same
// NDJSON event stream the real pipeline will (status → verbose pipeline
// trace → sources → token → done) with realistic pacing, so the UI is fully
// exercisable today — including the thinking log and LLM retry events.
import type { Source, StreamEvent } from "@/lib/types";
import { SEED_SOURCES } from "@/lib/mock-data";

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo);

async function waitOrAbort(ms: number, signal?: AbortSignal): Promise<boolean> {
  if (signal?.aborted) return false;
  await new Promise<void>((resolve) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); resolve(); }, { once: true });
  });
  return !signal?.aborted;
}

/** Split text into small, word-aligned chunks for a typing effect. */
function chunk(text: string): string[] {
  const out: string[] = [];
  let buf = "";
  for (const w of text.split(" ")) {
    if (buf && buf.length + 1 + w.length > rand(9, 24)) {
      out.push(buf + " ");
      buf = "";
    }
    buf += (w ? w + " " : "");
  }
  if (buf) out.push(buf);
  if (out.length === 0) out.push(text);
  return out;
}

// every sentence in these bodies carries its [n] source marker
function answerFor(question: string): { body: string; sources: Source[] } {
  const q = question.toLowerCase();
  if (/compar|efficac|vs\.| versus /.test(q)) {
    return { body: "## The comparative evidence is indirect\n\nNo completed randomised trial pits **pembrolizumab** against **atezolizumab** head-to-head in first-line metastatic NSCLC, so the comparison below rests on network meta-analysis and cross-trial benchmarks rather than a direct RCT [1][3].\n\n| Regimen | Setting | OS signal | Evidence grade |\n| --- | --- | --- | --- |\n| PD-1 + platinum chemo | Non-squamous & squamous | HR ≈ 0.56–0.62 | High (RCT) [1] |\n| PD-L1 + chemo ± anti-angiogenic | Non-squamous | HR ≈ 0.73–0.81 | High (RCT) [2] |\n| Single-agent ICB (TPS ≥ 50%) | Both | HR ≈ 0.65 | Moderate [4] |\n\n### Key reading\n\n1. PD-1/PD-L1 inhibitor plus chemotherapy carries the most consistent survival benefit across PD-L1 subgroups [1][3].\n2. Adding an anti-angiogenic agent extends the non-squamous option but adds toxicity and cost without demonstrated comparative superiority [2].\n3. For PD-L1 TPS ≥ 50%, single-agent checkpoint blockade remains an efficient first-line choice [4].\n\n**Bottom line:** choose by PD-L1 status, histology and comorbidity profile — the literature does not justify preferring one checkpoint inhibitor over the other on efficacy alone [1][3][5].", sources: SEED_SOURCES };
  }
  return { body: "## Direct answer\n\nThe literature supports a **clear class-level benefit** for this intervention, with the effect size scaling with baseline risk across the largest randomised cohorts [1][3].\n\n### What the evidence shows\n\n- **Primary endpoint:** consistent improvement in the outcome of interest across the major trials [1].\n- **Magnitude:** absolute benefit is risk-dependent; relative reductions hold across subgroups [1][2].\n- **Safety:** the adverse-event profile is manageable and largely predictable; serious events are uncommon but real [3].\n\n### Unresolved areas\n\n- Head-to-head comparisons with newer agents remain indirect [3][5].\n- Biomarker-guided selection is promising but not yet practice-changing [4].\n\n> Interpretation caution: trial populations are healthier than routine clinic patients, so absolute benefits are usually smaller in practice [2].\n\n**Bottom line:** guideline-aligned first-line use is well supported; tailor intensity to risk and comorbidity [1][2][4].", sources: SEED_SOURCES };
}

/**
 * Stream the full simulated research run. Yields the same StreamEvent union
 * the real backend will emit. Honors an AbortSignal for stop-generation.
 */
export async function* mockRagEvents(
  question: string,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent> {
  const started = Date.now();
  const model = "mistral-medium-latest";

  const trace = (event: string, fields: Record<string, unknown>): StreamEvent =>
    ({ type: "pipeline", event, fields });

  yield trace("run_start", { run_id: `mock-${started}`, question, budget: { evidence_target: 3, max_workers: 2 } });
  yield { type: "status", stage: "understanding", message: "parsing clinical intent" };
  yield trace("llm_call", { role: "master", model, attempt: 1, status: "ok" });
  if (!(await waitOrAbort(rand(450, 750), signal))) return;

  yield trace("master_plan", {
    tasks: [
      { id: "T1", title: "Dietary pattern → BP", objective: "quantity DASH vs control BP effect" },
      { id: "T2", title: "Adherence → effect size", objective: "adherence-adjusted BP reduction" },
    ],
    rationale: "two independent evidence lines: mechanism and adherence",
  });
  yield { type: "status", stage: "decomposing", message: "2 sub-questions extracted", count: 2 };
  if (!(await waitOrAbort(rand(550, 800), signal))) return;

  yield trace("task_start", { task_id: "T1", title: "Dietary pattern → BP" });
  yield trace("search_round", { task_id: "T1", round_no: 1, queries: { "T1.R1": ["DASH diet systolic blood pressure randomized"] } });
  yield { type: "status", stage: "retrieving", message: "searching PMC snapshot", count: 42 };
  if (!(await waitOrAbort(rand(600, 950), signal))) return;

  yield trace("retrieved", { task_id: "T1", requirement_id: "T1.R1", query: "DASH diet systolic blood pressure randomized", papers: SEED_SOURCES.map((s) => ({ document_id: s.id, title: s.title })) });
  yield { type: "status", stage: "retrieving", message: "42 documents found · 37 passages retained", count: 37 };
  if (!(await waitOrAbort(rand(350, 550), signal))) return;

  const { body, sources } = answerFor(question);
  yield { type: "sources", sources };

  // critic LLM: first attempt rate-limited, retry succeeds — the thinking
  // layer must surface this exactly like the real backend would.
  yield trace("llm_call", { role: "critic", model, attempt: 1, status: "failed", status_code: 429, error: "Rate limit exceeded (retry in 12.4s)" });
  yield trace("llm_call", { role: "critic", model, attempt: 2, status: "start" });
  yield trace("llm_call", { role: "critic", model, attempt: 2, status: "ok" });
  if (!(await waitOrAbort(rand(250, 450), signal))) return;

  yield trace("verdict", { task_id: "T1", requirement_id: "T1.R1", document_id: sources[0]?.id, relevance: "support", confidence: 0.92, accepted: true, support: "support", note: "Passage explicitly reports a systolic BP reduction from the DASH intervention in a randomised cohort; controls the primary outcome." });
  yield trace("evidence_added", { task_id: "T1", requirement_id: "T1.R1", evidence_id: "T1.R1.E1", document_id: sources[0]?.id, support: "support" });
  yield trace("verdict", { task_id: "T1", requirement_id: "T1.R1", document_id: sources[1]?.id, relevance: "support", confidence: 0.87, accepted: true, support: "neutral", note: "Meta-analysis fleetingly mentions DASH but pools heterogeneous diets; kept as neutral support only." });
  yield trace("evidence_added", { task_id: "T1", requirement_id: "T1.R1", evidence_id: "T1.R1.E2", document_id: sources[1]?.id, support: "neutral" });
  yield trace("verdict", { task_id: "T1", requirement_id: "T1.R1", document_id: sources[2]?.id, relevance: "not_relevant", confidence: 0.31, accepted: false, support: "contradicts", note: "Discusses sodium substitution in animal models, no human BP endpoint — does not answer the requirement." });
  yield { type: "status", stage: "reranking", message: "reranking evidence · score range 0.78–0.94", count: 37 };
  if (!(await waitOrAbort(rand(500, 750), signal))) return;

  yield trace("deep_inspection", { task_id: "T1", requirement_id: "T1.R1", document_id: sources[2]?.id, status: "found", findings: 4, verified: true });
  yield trace("task_done", { task_id: "T1", status: "ok", summary: "2 verified evidence items for dietary pattern effect", stop_reason: "requirement_satisfied" });
  yield trace("worker_report", { task_id: "T1", status: "ok", searches_used: 2, deep_inspections_used: 1, requirements: [{ requirement_id: "T1.R1", coverage: 3, target_n: 3, status: "satisfied" }], stop_reason: "requirement_satisfied" });
  yield { type: "status", stage: "verifying", message: "cross-checking claims against 5 sources", count: 5 };
  if (!(await waitOrAbort(rand(450, 700), signal))) return;

  yield trace("contradiction", { contradiction_id: "C1", claim: "DASH effect size differs by baseline BP", kind: "inconsistency", evidence_a: ["T1.R1.E1"], evidence_b: ["T1.R1.E3"] });
  yield trace("resolution", { contradiction_id: "C1", status: "resolved", explanation: "baseline-BP stratification explains the difference; effect persists in both subgroups", additional_papers: [] });
  yield trace("final_evidence", { evidence: 6, gaps: [], contradictions: 1, resolved: 1, unresolved: 0 });
  yield { type: "status", stage: "synthesizing", message: "drafting answer with per-sentence citations" };
  if (!(await waitOrAbort(rand(300, 500), signal))) return;

  yield trace("answer", { summary: "DASH reduces systolic BP in hypertensive adults, with larger absolute effects at higher baseline BP.", sections: [] });
  const chunks = chunk(body);
  for (const c of chunks) {
    if (signal?.aborted) return;
    yield { type: "token", content: c };
    if (!(await waitOrAbort(rand(8, 24), signal))) return;
  }

  yield trace("run_end", { stop_reason: "synthesized", confidence: 0.91, answer: body });
  yield { type: "done", timingMs: Date.now() - started };
}
