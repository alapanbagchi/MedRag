/** Friendly labels for backend stages / pipeline events. */

export const STAGE_LABELS: Record<string, string> = {
  understanding: "Understanding your question",
  decomposing: "Decomposing into sub-queries",
  planning: "Making a plan",
  retrieving: "Searching the literature",
  reranking: "Ranking retrieved evidence",
  verifying: "Verifying evidence against sources",
  synthesizing: "Synthesizing the answer",
  answering: "Composing the answer",
  started: "Starting",
  complete: "Complete",
};

export const EVENT_LABELS: Record<string, string> = {
  run_start: "Started research run",
  master_plan: "Built master plan",
  replan: "Re-planning",
  fallback_plan: "Falling back to deterministic plan",
  planner: "Planning",
  decompose: "Decomposing query",
  subquery: "Sub-query",
  search_round: "Search round",
  retrieve: "Retrieving",
  retrieved: "Retrieved evidence",
  verdict: "Verdict on passage",
  verify: "Verifying",
  evidence_added: "Evidence accepted",
  final_evidence: "Final evidence set",
  evidence_state: "Evidence state",
  requirement_state: "Requirement state",
  task_start: "Task started",
  task_done: "Task finished",
  worker_report: "Worker report",
  agent_spawn: "Agent spawned",
  agent_output: "Agent output",
  contradiction: "Contradiction check",
  resolution: "Resolving contradiction",
  deep_inspection: "Deep inspection",
  synthesize: "Synthesizing",
  answer: "Drafting answer",
  run_end: "Run finished",
  memory_prepare: "Preparing memory",
  memory_commit: "Committing memory",
  progress: "Progress",
  error: "Error",
};

export function labelForEvent(event: string): string {
  if (EVENT_LABELS[event]) return EVENT_LABELS[event]!;
  // prettify: snake_case -> Title Case
  return event
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

/** Pick a short human-readable summary out of an event's fields dict. */
export function summarizeFields(fields: Record<string, unknown> | undefined): string {
  if (!fields) return "";
  const preferred = [
    "msg",
    "message",
    "query",
    "subquery",
    "question",
    "title",
    "summary",
    "rationale",
    "tool",
    "reason",
    "status",
    "target",
  ];
  for (const key of preferred) {
    const value = fields[key];
    if (typeof value === "string" && value.trim()) {
      const clean = value.trim().replace(/\s+/g, " ");
      if (clean.length <= 96) return clean;
      return `${clean.slice(0, 93)}…`;
    }
  }
  for (const value of Object.values(fields)) {
    if (typeof value === "string" && value.trim()) {
      const clean = value.trim().replace(/\s+/g, " ");
      if (clean.length > 2 && clean.length <= 96) return clean;
    }
  }
  return "";
}