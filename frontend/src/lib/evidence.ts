/**
 * Shared evidence payload parsing: judge verdicts and planned requirements.
 */

export interface EvidencePassage {
  chunk_id?: string;
  document_id?: string;
  chunk_type?: string;
  section?: string;
  score?: number | string;
  text?: string;
  title?: string;
  journal?: string;
  url?: string;
  success?: boolean;
}

export interface PassageVerdict {
  passage_id?: string;
  intent_score?: number;
  coverage?: string[];
  reason?: string;
}

/**
 * The judge middleware appends verdicts to the tool output:
 * [EVIDENCE JUDGMENT] followed by the verdicts JSON (or a FAILED note).
 */
export function splitJudgment(raw: unknown): {
  passagesText: string;
  verdictsText: string | null;
  failed: string | null;
} {
  const text = typeof raw === "string" ? raw : "";
  const marker = "[EVIDENCE JUDGMENT]";
  const failedMarker = "[EVIDENCE JUDGMENT FAILED";
  const at = text.indexOf(marker);
  if (at >= 0) {
    return {
      passagesText: text.slice(0, at).trim(),
      verdictsText: text.slice(at + marker.length).trim(),
      failed: null,
    };
  }
  const failedAt = text.indexOf(failedMarker);
  if (failedAt >= 0) {
    return {
      passagesText: text.slice(0, failedAt).trim(),
      verdictsText: null,
      failed: text.slice(failedAt).trim(),
    };
  }
  return { passagesText: text, verdictsText: null, failed: null };
}

/** Verdict list from the [EVIDENCE JUDGMENT] payload. */
export function parseVerdicts(text: string | null): PassageVerdict[] | null {
  if (!text) return null;
  try {
    const value = JSON.parse(text) as { evidence_results?: unknown };
    const results = value?.evidence_results;
    if (!Array.isArray(results)) return null;
    return results.filter((v): v is PassageVerdict => !!v && typeof v === "object");
  } catch {
    return null;
  }
}

/** Total judge tokens recorded in the [EVIDENCE JUDGMENT] payload. */
export function parseJudgmentTokens(text: string | null): number | undefined {
  if (!text) return undefined;
  try {
    const value = JSON.parse(text) as { tokens?: unknown };
    const tokens = value?.tokens;
    return typeof tokens === "number" && Number.isFinite(tokens) && tokens > 0
      ? tokens
      : undefined;
  } catch {
    return undefined;
  }
}

/** Per-passage critic reason (kept and rejected) from the judgment payload. */
export function parseJudgmentReasons(text: string | null): Record<string, string> {
  if (!text) return {};
  try {
    const value = JSON.parse(text) as { reasons?: unknown };
    const reasons = value?.reasons;
    if (!reasons || typeof reasons !== "object" || Array.isArray(reasons)) return {};
    const out: Record<string, string> = {};
    for (const [id, reason] of Object.entries(reasons as Record<string, unknown>)) {
      if (typeof reason === "string" && reason.trim()) out[id] = reason.trim();
    }
    return out;
  } catch {
    return {};
  }
}

/** One evidence requirement, as returned by the requirements planner. */
export interface EvidenceCriterion {
  id: string;
  description: string;
}

/** Satisfaction of one criterion against the judge verdicts. */
export type CriterionStatus = "pending" | "satisfied" | "partial" | "unsatisfied";

/** Lenient parse of a {id, description} requirements list (string or list). */
export function parseRequirementList(raw: unknown): EvidenceCriterion[] {
  try {
    const value =
      typeof raw === "string" ? (raw.trim() ? JSON.parse(raw) : null) : raw;
    if (!Array.isArray(value)) return [];
    const out: EvidenceCriterion[] = [];
    for (const item of value) {
      if (!item || typeof item !== "object") continue;
      const o = item as Record<string, unknown>;
      const id =
        typeof o.id === "string"
          ? o.id
          : typeof o.requirement_id === "string"
            ? o.requirement_id
            : null;
      const description =
        typeof o.description === "string"
          ? o.description
          : typeof o.desc === "string"
            ? o.desc
            : "";
      if (!id || !description.trim()) continue;
      if (out.some((c) => c.id === id)) continue;
      out.push({ id, description: description.trim() });
    }
    return out;
  } catch {
    return [];
  }
}

/**
 * Score each criterion against every judge verdict.
 *
 * The best intent among verdicts covering the id wins — >= 0.5 is satisfied,
 * below is partial, never covered is unsatisfied. With no verdicts at all the
 * judge hasn't run yet, so everything stays pending.
 */
export function scoreCriteria(
  criteria: EvidenceCriterion[],
  verdicts: PassageVerdict[] | null,
): { id: string; status: CriterionStatus; best: number | null }[] {
  if (!verdicts || verdicts.length === 0) {
    return criteria.map((c) => ({ id: c.id, status: "pending" as const, best: null }));
  }
  return criteria.map((c) => {
    let best: number | null = null;
    for (const v of verdicts) {
      if (!Array.isArray(v.coverage) || !v.coverage.includes(c.id)) continue;
      if (typeof v.intent_score === "number" && Number.isFinite(v.intent_score)) {
        best = best === null ? v.intent_score : Math.max(best, v.intent_score);
      } else if (best === null) {
        best = 0;
      }
    }
    if (best === null) return { id: c.id, status: "unsatisfied" as const, best };
    return { id: c.id, status: best >= 0.5 ? "satisfied" : "partial", best };
  });
}
