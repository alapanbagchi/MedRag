// ── Synthetic seed data. Replace with a real API later; shape is final. ──
import type { Conversation, Message, Source, StageEvent } from "@/lib/types";
import { uid } from "@/lib/utils";

export const ENGINE = {
  name: "medrag-pipeline",
  version: "0.5.0",
  corpus: "PMC snapshot",
  docsIndexed: 1_842_317,
  passages: 12_604_088,
  model: "medrag-agentic-v1",
  latencyP95: "2.9s",
};

export const SUGGESTED_QUERIES = [
  "What are the latest treatments for metastatic NSCLC?",
  "Compare the efficacy of pembrolizumab vs atezolizumab in first-line lung cancer.",
  "What does the literature say about statins and cardiovascular risk reduction?",
  "Find evidence for ketamine in treatment-resistant depression.",
];

const now = Date.now();
const MIN = 60_000;
const HOUR = 3_600_000;

function stage(stageName: StageEvent["stage"], label: string, message?: string, count?: number, offsetMs = 0): StageEvent {
  return { stage: stageName, label, message, count, at: now + offsetMs };
}

function src(
  id: string,
  title: string,
  journal: string,
  year: number,
  score: number,
  authors: string[],
  snippet: string
): Source {
  return { id, pmcid: id, pmid: "3" + id.slice(3), title, authors, journal, year, score, snippet, url: `https://pmc.ncbi.nlm.nih.gov/articles/${id}/` };
}

export const SEED_SOURCES: Source[] = [
  src("PMC123456", "Pembrolizumab plus chemotherapy in metastatic non-small-cell lung cancer", "New England Journal of Medicine", 2024, 0.94,
    ["R. Nakamura", "A. K. Weber", "L. S. Chen"], "In 616 patients with previously untreated metastatic NSCLC without EGFR/ALK alterations, pembrolizumab plus platinum-doublet chemotherapy improved overall survival and PFS across PD-L1 subgroups."),
  src("PMC987654", "Atezolizumab for first-line treatment of metastatic nonsquamous NSCLC", "The Lancet Oncology", 2023, 0.91,
    ["M. Delgado", "S. Okafor", "K. Johannsen"], "Cohort analysis: atezolizumab with bevacizumab and chemotherapy showed progression-free survival benefit across PD-L1 expression subgroups."),
  src("PMC456789", "Immune checkpoint inhibitors in lung cancer: a network meta-analysis of first-line regimens", "JAMA Oncology", 2025, 0.87,
    ["T. Hasegawa", "E. F. Rooney"], "Bayesian network meta-analysis of 14 randomised trials (n=9,312). PD-1/PD-L1 inhibitor-chemotherapy combinations ranked highest for PFS."),
  src("PMC112233", "Real-world outcomes of PD-1 blockade in advanced NSCLC by PD-L1 tumor proportion score", "Journal of Thoracic Oncology", 2024, 0.82,
    ["F. Almeida", "J. W. Park"], "Retrospective multicentre study (n=1,204): OS benefit concentrated in TPS ≥50%, with no significant effect in TPS 1–49%."),
  src("PMC445566", "Biomarker-driven selection for immunotherapy in non-small-cell lung cancer", "Annals of Oncology", 2023, 0.78,
    ["V. Ivanova", "G. D. Hughes"], "Review of companion diagnostics: PD-L1 IHC, TMB, and HLA class I expression as candidate predictive biomarkers; TMB remains exploratory."),
];

const STAGES_UNDERSTAND = (q: string, t: number): StageEvent[] => [
  stage("understanding", "Understanding query", "parsing clinical intent", undefined, t),
  stage("decomposing", "Decomposing question", "2 sub-questions extracted", 2, t + 900),
  stage("retrieving", "Retrieving evidence", "42 documents found", 42, t + 1800),
  stage("reranking", "Reranking evidence", "37 passages retained", 37, t + 2600),
  stage("verifying", "Verifying claims", "cross-checking 5 sources", 5, t + 3300),
];

function msg(role: Message["role"], content: string, status: Message["status"], extra?: Partial<Message>): Message {
  return { id: uid("m"), role, content, status, createdAt: now, ...extra };
}

export function seedConversations(): Conversation[] {
  const c1: Conversation = {
    id: "conv-onco",
    title: "First-line immunotherapy in metastatic NSCLC",
    pinned: true,
    createdAt: now - 3 * MIN,
    updatedAt: now - 2 * MIN,
    messages: [
      msg("user", "Compare the efficacy of pembrolizumab vs atezolizumab in first-line metastatic NSCLC without driver mutations.", "complete"),
      {
        id: uid("m"), role: "assistant",
        content: "## Head-to-head evidence is indirect\n\nNo completed randomised trial compares **pembrolizumab** with **atezolizumab** directly in first-line metastatic NSCLC. The comparison below is derived from network meta-analyses and cross-trial benchmarks; treat it as indirect evidence [1][2].\n\n### Summary of first-line regimens\n\n| Regimen | Setting | OS benefit | Evidence grade |\n| --- | --- | --- | --- |\n| Pembrolizumab + chemo | Non-squamous & squamous | HR 0.56–0.62 | High (RCT) [1] |\n| Atezolizumab + chemo + bevacizumab | Non-squamous | HR 0.73–0.81 | High (RCT) [2] |\n| PD-1/PD-L1 inhibitor alone | TPS ≥ 50% | HR 0.65 | Moderate [4] |\n\n### Practical reading\n\n1. Combination pembrolizumab regimens carry the strongest overall-survival signal across PD-L1 subgroups [1][3].\n2. Adding bevacizumab broadens the atezolizumab-based option but adds toxicity burden and an access cost without clear comparative superiority [2].\n3. For TPS ≥ 50% tumours, single-agent PD-1 blockade remains a reasonable, effective first-line choice [4].\n\n**Bottom line:** choose by PD-L1 status, histology, and comorbidity profile; the literature does not yet justify picking between the two checkpoint inhibitors on efficacy alone [1][3][5].",
        status: "complete", createdAt: now - 3 * MIN,
        startedAt: now - 3 * MIN + 200, finishedAt: now - 3 * MIN + 6200,
        sources: SEED_SOURCES, stages: STAGES_UNDERSTAND("compare", now - 3 * MIN), rating: "up",
      },
    ],
  };

  const c2: Conversation = {
    id: "conv-statin",
    title: "Statins and cardiovascular risk reduction — latest evidence",
    pinned: true,
    createdAt: now - 26 * HOUR,
    updatedAt: now - 25 * HOUR,
    messages: [
      msg("user", "What does the literature say about statins and cardiovascular risk reduction?", "complete"),
      {
        id: uid("m"), role: "assistant",
        content: "## The short answer\n\nStatins remain the best-evidenced lipid-lowering class for **primary and secondary** prevention of atherosclerotic cardiovascular disease (ASCVD); absolute benefit scales with baseline risk, not with baseline LDL alone [1].\n\n- Each ~1 mmol/L LDL reduction lowers major vascular events by ~20–25% per year of exposure [1].\n- Intensity matters: **high-intensity therapy** (atorvastatin 40–80 mg, rosuvastatin 20–40 mg) is first-line for established ASCVD [2].\n- Recent trials (e.g., newer PCSK9-siRNA agents) extend the LDL hypothesis but do not displace statins as the foundational therapy [3].\n\n### Unresolved areas\n\n- Statin intolerance is largely **nocebo-driven** when tested blind [4].\n- Primary prevention in low-risk, older populations remains a shared-decision zone [5].\n\n**Bottom line:** prescribe intensity by risk strata; residual risk is a separate conversation with the patient, not a reason to downgrade statins.",
        status: "complete", createdAt: now - 26 * HOUR,
        startedAt: now - 26 * HOUR + 300, finishedAt: now - 26 * HOUR + 5400,
        sources: SEED_SOURCES, stages: STAGES_UNDERSTAND("statins", now - 26 * HOUR),
      },
    ],
  };

  const c3: Conversation = {
    id: "conv-ket",
    title: "Ketamine in treatment-resistant depression",
    pinned: false,
    createdAt: now - 50 * HOUR,
    updatedAt: now - 49 * HOUR,
    messages: [
      msg("user", "Find evidence for ketamine in treatment-resistant depression, including dosing protocols.", "complete"),
      {
        id: uid("m"), role: "assistant",
        content: "## Evidence position\n\nIntravenous **esketamine/ketamine** has replicated, rapid antidepressant effects in treatment-resistant depression (TRD), with effect sizes largest in the first 24–72 hours [1].\n\n1. **Dose:** IV ketamine 0.5 mg/kg over 40 min is the best-studied protocol [1].\n2. **Efficacy:** response rates ~50–70% in RCTs versus ~30–40% for saline control at 24 h; durability requires maintenance dosing [1][2].\n3. **Guideline status:** esketamine nasal spray (FDA-approved with REMS) is the registered formulation; off-label IV racemic ketamine remains common [3].\n\n> Monitor dissociation and blood pressure; transient hemodynamic effects are expected and usually self-limited [2].\n\n**Bottom line:** ketamine is a rapid-onset, short-duration tool for TRD, best deployed inside a monitored treatment program and paired with ongoing antidepressant therapy [1][3].",
        status: "complete", createdAt: now - 50 * HOUR,
        startedAt: now - 50 * HOUR + 250, finishedAt: now - 50 * HOUR + 6100,
        sources: SEED_SOURCES, stages: STAGES_UNDERSTAND("ketamine", now - 50 * HOUR),
      },
    ],
  };

  const c4: Conversation = {
    id: "conv-gene",
    title: "Gene therapy for sickle cell disease — approval landscape",
    pinned: false,
    createdAt: now - 60 * HOUR,
    updatedAt: now - 59 * HOUR,
    messages: [
      msg("user", "What are the latest treatments for gene therapy in sickle cell disease?", "complete"),
      {
        id: uid("m"), role: "assistant",
        content: "## Approved ex vivo gene therapies\n\nTwo gene therapies are now approved for sickle cell disease, both using autologous edited/transduced CD34+ cells after myeloablative conditioning [1]:\n\n- **Exa-cel (Casgevy)** — CRISPR-Cas9 editing of *BCL11A* enhancer. Phase 3: 28/30 evaluable patients free of severe VOC events at 12 months [1][2].\n- **Lovo-cel (Lyfgenia)** — lentiviral *HBB* vector. Similar reported VOC freedom with a boxed warning for haematologic malignancy [2].\n\n### Candidates and cautions\n\n- Eligibility skews young and severe (recurrent VOC, no matched donor).\n- Conditioning-related infertility is a real consent burden [3].\n- Long-term genotoxicity data remain short (2–3 years) [1][2].\n\n**Bottom line:** transformative for eligible patients; counselling must frame conditioning toxicity and immature follow-up [1][3].",
        status: "complete", createdAt: now - 60 * HOUR,
        startedAt: now - 60 * HOUR + 350, finishedAt: now - 60 * HOUR + 5800,
        sources: SEED_SOURCES, stages: STAGES_UNDERSTAND("gene therapy", now - 60 * HOUR),
      },
    ],
  };

  const c5: Conversation = {
    id: "conv-glp",
    title: "GLP-1 receptor agonists — cardiovascular outcomes",
    pinned: false,
    createdAt: now - 8 * 24 * HOUR,
    updatedAt: now - 8 * 24 * HOUR + 20 * MIN,
    messages: [
      msg("user", "Summarise the cardiovascular outcome trials for GLP-1 receptor agonists.", "complete"),
      {
        id: uid("m"), role: "assistant",
        content: "## Class effect on MACE\n\nThe cardiovascular outcome trials (SELECT, SUSTAIN-6, LEADER, REWIND) converge on a **class-level MACE reduction** for GLP-1 RAs in high-risk populations, independent of baseline glycaemic control [1].\n\n- SELECT (semaglutide 2.4 mg, n=17,604, no diabetes): MACE HR 0.80 — the largest weight-driven signal [1].\n- Benefit appears within the first year and persists on-treatment [2].\n- Adverse events are GI-predominant; pancreatitis and cholelithiasis signals are small but present [3].\n\n**Bottom line:** GLP-1 RAs are now part of guideline-level ASCVD risk reduction, not merely glycaemic therapy [1][2][3].",
        status: "complete", createdAt: now - 8 * 24 * HOUR,
        startedAt: now - 8 * 24 * HOUR + 200, finishedAt: now - 8 * 24 * HOUR + 5600,
        sources: SEED_SOURCES, stages: STAGES_UNDERSTAND("glp-1", now - 8 * 24 * HOUR),
      },
    ],
  };

  const c6: Conversation = {
    id: "conv-echo",
    title: "Left atrial appendage occlusion vs DOACs",
    pinned: false,
    createdAt: now - 14 * 24 * HOUR,
    updatedAt: now - 13 * 24 * HOUR,
    messages: [
      msg("user", "How does left atrial appendage occlusion compare with DOACs for stroke prevention in atrial fibrillation?", "complete"),
      {
        id: uid("m"), role: "assistant",
        content: "## Headline\n\nPRAGUE-17-style evidence shows LAA occlusion is **non-inferior** to DOACs for the composite of stroke, systemic embolism, and major bleeding at 4 years, with most events occurring early (periprocedural) in the device arm [1].\n\n- Efficacy: stroke/SE rates broadly comparable after the periprocedural window [1].\n- Bleeding: device arm trends toward fewer non-procedural major bleeds, driven by avoiding long-term anticoagulation [1][2].\n- DOACs remain the default; occlusion is a reasonable alternative for patients with contraindications or intolerable bleeding risk [2].\n\n**Bottom line:** comparable outcomes, different risk profiles; patient selection is the whole game [1][2].",
        status: "complete", createdAt: now - 14 * 24 * HOUR,
        startedAt: now - 14 * 24 * HOUR + 300, finishedAt: now - 14 * 24 * HOUR + 5900,
        sources: SEED_SOURCES, stages: STAGES_UNDERSTAND("laa occlusion", now - 14 * 24 * HOUR),
      },
    ],
  };

  return [c1, c2, c3, c4, c5, c6];
}
