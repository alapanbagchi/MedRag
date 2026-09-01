// ── Mock research engine ─────────────────────────────────────────────
// Simulates the MedPat backend while it is not connected: emits the same
// NDJSON event stream the real pipeline will (status → sources → token →
// done) with realistic pacing, so the UI is fully exercisable today.
// Swap to the real client by setting NEXT_PUBLIC_RAG_API_URL (see rag-client).
import type { Source, StreamEvent } from "@/lib/types";
import { SEED_SOURCES } from "@/lib/mock-data";

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

async function waitOrAbort(ms: number, signal?: AbortSignal): Promise<boolean> {
  if (signal?.aborted) return false;
  await new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      resolve(); // resolve, caller checks abort
    }, { once: true });
  });
  return !signal?.aborted;
}

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo);

/** Split text into small, word-aligned-ish chunks for a typing effect. */
function chunk(text: string): string[] {
  const words = text.split(/(\s+)/);
  const out: string[] = [];
  let buf = "";
  for (const w of words) {
    buf += w;
    if (buf.length >= rand(6, 22)) {
      out.push(buf);
      buf = "";
    }
  }
  if (buf) out.push(buf);
  if (out.length === 0) out.push(text);
  return out;
}

function answerFor(question: string): { body: string; sources: Source[] } {
  const q = question.toLowerCase();
  const pick = SEED_SOURCES;
  if (/compar|efficac|vs\.| versus /.test(q)) {
    return { body: "## The comparative evidence is indirect\n\nNo single trial pits the two leading checkpoint-inhibitor backbones against each other head-to-head, so the comparison below rests on **network meta-analysis** and cross-trial benchmarks — treat it as indirect evidence [1][3].\n\n| Regimen | Setting | OS signal | Grade |\n| --- | --- | --- | --- |\n| PD-1 + platinum chemo | Non-squamous & squamous | HR ≈ 0.56–0.62 | High (RCT) [1] |\n| PD-L1 + chemo ± anti-angiogenic | Non-squamous | HR ≈ 0.73–0.81 | High (RCT) [2] |\n| Single-agent ICB (TPS ≥ 50%) | Both | HR ≈ 0.65 | Moderate [4] |\n\n### Key reading\n\n1. PD-1/PD-L1 inhibitor plus chemotherapy carries the most consistent survival benefit across PD-L1 subgroups [1][3].\n2. Adding an anti-angiogenic agent extends the non-squamous option but adds toxicity and cost without demonstrated comparative superiority [2].\n3. For PD-L1 TPS ≥ 50%, single-agent checkpoint blockade remains an efficient first-line choice [4].\n\n**Bottom line:** choose by PD-L1 status, histology, and comorbidity profile — the current literature does not justify preferring one checkpoint inhibitor over the other on efficacy alone [1][3][5].", sources: pick };
  }
  return { body: "## Direct answer\n\nThe literature supports a **clear class-level benefit** for this intervention, with effect size scaling with baseline risk — and the caveats below matter for individual decisions [1][3].\n\n### What the evidence shows\n\n- **Primary endpoint:** consistent improvement in the outcome of interest across the largest randomised cohorts [1].\n- **Magnitude:** absolute benefit is risk-dependent; relative reductions hold across subgroups [1][2].\n- **Safety:** the adverse-event profile is manageable and largely predictable; serious events are uncommon but real [3].\n\n### Unresolved areas\n\n- Head-to-head comparisons with newer agents remain indirect [3][5].\n- Biomarker-guided selection is promising but not yet practice-changing [4].\n\n> Interpretation caution: trial populations are healthier than routine clinic patients, so absolute benefits are usually smaller in practice [2].\n\n**Bottom line:** guideline-aligned first-line use is well supported; tailor intensity to risk and comorbidity [1][2][4].", sources: pick };
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
  yield { type: "status", stage: "understanding", message: "parsing clinical intent" };
  if (!(await waitOrAbort(rand(500, 850), signal))) return;

  yield { type: "status", stage: "decomposing", message: "2 sub-questions extracted", count: 2 };
  if (!(await waitOrAbort(rand(600, 950), signal))) return;

  yield { type: "status", stage: "retrieving", message: "searching PMC snapshot", count: 42 };
  if (!(await waitOrAbort(rand(700, 1100), signal))) return;

  yield { type: "status", stage: "retrieving", message: "42 documents found · 37 passages retained", count: 37 };
  if (!(await waitOrAbort(rand(400, 700), signal))) return;

  const { body, sources } = answerFor(question);
  yield { type: "sources", sources };
  if (!(await waitOrAbort(rand(300, 500), signal))) return;

  yield { type: "status", stage: "reranking", message: "reranking evidence · score range 0.78–0.94", count: 37 };
  if (!(await waitOrAbort(rand(600, 900), signal))) return;

  yield { type: "status", stage: "verifying", message: "cross-checking claims against 5 sources", count: 5 };
  if (!(await waitOrAbort(rand(550, 850), signal))) return;

  yield { type: "status", stage: "synthesizing", message: "drafting answer with inline citations" };
  if (!(await waitOrAbort(rand(350, 600), signal))) return;

  const chunks = chunk(body);
  for (const c of chunks) {
    if (signal?.aborted) return;
    yield { type: "token", content: c };
    if (!(await waitOrAbort(rand(8, 26), signal))) return;
  }

  yield { type: "done", timingMs: Date.now() - started };
}
