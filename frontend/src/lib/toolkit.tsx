import {
  AlertTriangleIcon,
  BookOpenIcon,
  CheckIcon,
  DatabaseIcon,
  ExternalLinkIcon,
  FlaskConicalIcon,
  GitMergeIcon,
  GlobeIcon,
  LanguagesIcon,
  NetworkIcon,
  PenLineIcon,
  PuzzleIcon,
  RefreshCwIcon,
  RouteIcon,
  ScaleIcon,
  ScanSearchIcon,
  SearchIcon,
  ShieldCheckIcon,
  TargetIcon,
  UserPlusIcon,
} from "lucide-react";
import { firstArray } from "./xdeep";
import type { ComponentType } from "react";

/** Step metadata: icon + display label per step kind. */
interface StepMeta {
  label: string;
  icon: ComponentType<{ className?: string }>;
  tone: string; // icon color class
}

export const STEP_META: Record<string, StepMeta> = {
  retrieve: { label: "Document retrieval", icon: SearchIcon, tone: "text-brand" },
  umls: { label: "UMLS Expansion", icon: LanguagesIcon, tone: "text-brand" },
  web_search: { label: "Web search", icon: GlobeIcon, tone: "text-success" },
  web_fetch: { label: "Fetching source", icon: ExternalLinkIcon, tone: "text-brand" },
  reliability: { label: "Reliability check", icon: ShieldCheckIcon, tone: "text-success" },
  verdict: { label: "Evidence verdict", icon: ScaleIcon, tone: "text-warning" },
  decompose: { label: "Decomposing the question", icon: NetworkIcon, tone: "text-brand" },
  research: { label: "Research task", icon: FlaskConicalIcon, tone: "text-brand" },
  search_round: { label: "Search round", icon: RefreshCwIcon, tone: "text-brand" },
  contradiction: { label: "Contradiction", icon: AlertTriangleIcon, tone: "text-warning" },
  resolution: { label: "Resolved", icon: CheckIcon, tone: "text-success" },
  gap_check: { label: "Gap agent", icon: ScanSearchIcon, tone: "text-brand" },
  delegate: { label: "Delegation", icon: UserPlusIcon, tone: "text-brand" },
  gap_probe: { label: "Evidence gap", icon: TargetIcon, tone: "text-warning" },
  gap_resolution: { label: "Gap resolved", icon: PuzzleIcon, tone: "text-success" },
  evidence: { label: "Verified evidence", icon: DatabaseIcon, tone: "text-brand" },
  synthesize: { label: "Synthesizing answer", icon: PenLineIcon, tone: "text-brand" },
  planner: { label: "Planning", icon: RouteIcon, tone: "text-brand" },
  join: { label: "Joining findings", icon: GitMergeIcon, tone: "text-brand" },
  done: { label: "Finished", icon: CheckIcon, tone: "text-success" },
  thought: { label: "Thought", icon: BookOpenIcon, tone: "text-muted-foreground" },
  error: { label: "Error", icon: AlertTriangleIcon, tone: "text-destructive" },
};

export function stepMeta(kind: string): StepMeta {
  return STEP_META[kind] ?? STEP_META.thought!;
}

export interface PlanStep {
  id: string;
  text: string;
  sub?: string;
  done?: boolean;
  started?: boolean;
  model?: string;
}

export function extractPlanSteps(fields: Record<string, unknown> | undefined): PlanStep[] {
  if (!fields) return [];
  const items = fields.items;
  if (Array.isArray(items)) {
    return items
      .map((t, i) => {
        if (t && typeof t === "object") {
          const o = t as Record<string, unknown>;
          const text =
            (typeof o.question === "string" && o.question.trim()) ||
            (typeof o.text === "string" && o.text.trim()) ||
            JSON.stringify(o).slice(0, 140);
          return {
            id: typeof o.id === "string" ? o.id : "T" + (i + 1),
            text: text.slice(0, 150),
            sub: o.deep_research === true ? "deep research" : undefined,
            done: o.done === true,
            started: o.started === true,
            model: typeof o.model === "string" ? o.model : undefined,
          };
        }
        return { id: "T" + (i + 1), text: String(t).slice(0, 150) };
      })
      .filter((s) => s.text.length > 0);
  }
  const requirements = fields.requirements;
  if (Array.isArray(requirements)) {
    return requirements
      .map((r, i) => {
        if (r && typeof r === "object") {
          const o = r as Record<string, unknown>;
          const id = typeof o.id === "string" ? o.id : "T" + (i + 1);
          const text =
            (typeof o.text === "string" && o.text.trim()) ||
            (typeof o.question === "string" && o.question.trim()) ||
            JSON.stringify(o).slice(0, 140);
          return { id, text: text.slice(0, 150), sub: typeof o.target_n === "number" ? "target" : undefined };
        }
        return { id: "T" + (i + 1), text: String(r).slice(0, 150) };
      })
      .filter((s) => s.text.length > 0);
  }
  const tasks = fields.tasks;
  if (Array.isArray(tasks)) {
    return tasks
      .map((t, i) => {
        if (t && typeof t === "object") {
          const o = t as Record<string, unknown>;
          const title =
            (typeof o.title === "string" && o.title) ||
            (typeof o.task === "string" && o.task) ||
            (typeof o.task_id === "string" && o.task_id) ||
            "";
          const desc = (typeof o.description === "string" && o.description) || "";
          return { id: (typeof o.id === "string" && o.id) || "T" + (i + 1), text: (title || JSON.stringify(o)).slice(0, 150), sub: desc || undefined };
        }
        return { id: "T" + (i + 1), text: String(t).slice(0, 150) };
      })
      .filter((s) => s.text.length > 0);
  }
  const arr = firstArray(fields);
  if (arr) {
    return arr.map((item, i) => ({
      id: "#" + (i + 1),
      text: (typeof item === "string" ? item : JSON.stringify(item)).slice(0, 150),
    }));
  }
  return [];
}
