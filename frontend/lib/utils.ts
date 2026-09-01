// ── tiny helpers ──────────────────────────────────────────────────────

export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export function uid(prefix = "id"): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

let counters: Record<string, number> = {};
export function tube(prefix: string): string {
  counters[prefix] = (counters[prefix] ?? 0) + 1;
  return `${prefix}-${String(counters[prefix]).padStart(3, "0")}`;
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

/** Strip markdown noise and cap length to make a conversation title. */
export function titleFromText(text: string, max = 60): string {
  const clean = text
    .replace(/[#*_~\[\]()`>|\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return clean.length > max ? clean.slice(0, max).trimEnd() + "…" : clean || "Untitled research";
}

const DAY = 86_400_000;

export type AgeGroup = "TODAY" | "YESTERDAY" | "PREVIOUS 7 DAYS" | "OLDER";

export function ageGroup(ts: number, now = Date.now()): AgeGroup {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const startOfTodayTs = startOfToday.getTime();
  if (ts >= startOfTodayTs) return "TODAY";
  if (ts >= startOfTodayTs - DAY) return "YESTERDAY";
  if (ts >= startOfTodayTs - 7 * DAY) return "PREVIOUS 7 DAYS";
  return "OLDER";
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s % 60)).padStart(2, "0")}m`;
}

export function fmtNum(n: number): string {
  return String(Math.trunc(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

export function formatClock(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function formatScore(score?: number): string {
  if (score == null) return "—";
  return `${Math.round(score * 100)}%`;
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}
