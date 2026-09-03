import {
  defineToolkit,
  type ToolCallMessagePartComponent,
} from "@assistant-ui/react";
import {
  AlertTriangleIcon,
  BookOpenIcon,
  CheckIcon,
  DatabaseIcon,
  ExternalLinkIcon,
  FlaskConicalIcon,
  GitMergeIcon,
  GlobeIcon,
  ListChecksIcon,
  MemoryStickIcon,
  NetworkIcon,
  PenLineIcon,
  PuzzleIcon,
  RefreshCwIcon,
  RouteIcon,
  ScaleIcon,
  SearchIcon,
  ShieldCheckIcon,
  TargetIcon,
} from "lucide-react";
import { STAGE_LABELS } from "./labels";
import type { Source } from "./types";
import { firstArray, type StepArgs } from "./xdeep";
import type { ComponentType } from "react";

// ---------------------------------------------------------------------------
// Step metadata: icon + display label per step kind.
// ---------------------------------------------------------------------------

interface StepMeta {
  label: string;
  icon: ComponentType<{ className?: string }>;
  tone: string; // icon color class
}

export const STEP_META: Record<string, StepMeta> = {
  retrieve: { label: "Searching literature", icon: SearchIcon, tone: "text-[#4285f4]" },
  web_search: { label: "Web search", icon: GlobeIcon, tone: "text-[#0f9d58]" },
  web_fetch: { label: "Fetching source", icon: ExternalLinkIcon, tone: "text-[#4285f4]" },
  reliability: { label: "Reliability check", icon: ShieldCheckIcon, tone: "text-[#0f9d58]" },
  verdict: { label: "Evidence verdict", icon: ScaleIcon, tone: "text-[#f4b400]" },
  decompose: { label: "Decomposing the question", icon: NetworkIcon, tone: "text-[#9d7bfb]" },
  research: { label: "Research task", icon: FlaskConicalIcon, tone: "text-[#4285f4]" },
  search_round: { label: "Search round", icon: RefreshCwIcon, tone: "text-[#4285f4]" },
  contradiction: { label: "Contradiction", icon: AlertTriangleIcon, tone: "text-[#f4b400]" },
  resolution: { label: "Resolved", icon: CheckIcon, tone: "text-[#0f9d58]" },
  gap_probe: { label: "Evidence gap", icon: TargetIcon, tone: "text-[#f4b400]" },
  gap_resolution: { label: "Gap resolved", icon: PuzzleIcon, tone: "text-[#0f9d58]" },
  evidence: { label: "Verified evidence", icon: DatabaseIcon, tone: "text-[#9d7bfb]" },
  synthesize: { label: "Synthesizing answer", icon: PenLineIcon, tone: "text-[#9d7bfb]" },
  planner: { label: "Planning", icon: RouteIcon, tone: "text-[#9d7bfb]" },
  join: { label: "Joining findings", icon: GitMergeIcon, tone: "text-[#4285f4]" },
  done: { label: "Finished", icon: CheckIcon, tone: "text-[#0f9d58]" },
  thought: { label: "Thought", icon: BookOpenIcon, tone: "text-muted-foreground" },
  error: { label: "Error", icon: AlertTriangleIcon, tone: "text-destructive" },
};

export function stepMeta(kind: string): StepMeta {
  return STEP_META[kind] ?? STEP_META.thought!;
}

// ---------------------------------------------------------------------------
// StepUI — one row inside a tool card (detail + status).
// ---------------------------------------------------------------------------

export const StepUI: ToolCallMessagePartComponent<StepArgs> = ({ args }) => {
  const a = (args ?? {}) as StepArgs;
  const running = !a.done;
  return (
    <div className="flex items-center gap-2.5 px-3 py-1.5">
      <span className="min-w-0 flex-1 truncate text-[13px]" title={a.detail ?? a.label}>
        {a.detail || a.label}
      </span>
      {a.error ? (
        <span className="shrink-0 text-[11px] text-destructive" title={a.error}>
          {a.error.length > 40 ? `${a.error.slice(0, 40)}…` : a.error}
        </span>
      ) : running ? (
        <span className="shrink-0">
          <span className="block size-3 animate-spin rounded-full border-[1.5px] border-primary/30 border-t-primary" />
        </span>
      ) : (
        <CheckIcon className="size-3.5 shrink-0 text-[#0f9d58]" />
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// StageUI — pinned live-stage row inside the thinking panel.
// ---------------------------------------------------------------------------

type StatusArgs = { stage: string; message?: string };

export const StageUI: ToolCallMessagePartComponent<StatusArgs> = ({ args, status }) => {
  const stage = args?.stage ?? "understanding";
  const running = status.type === "running";
  const label = STAGE_LABELS[stage] ?? stage;
  return (
    <div className="flex items-center gap-2.5 px-3.5 py-2.5">
      {running ? (
        <span className="relative flex size-3 shrink-0">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-40" />
          <span className="relative inline-flex size-3 rounded-full bg-primary" />
        </span>
      ) : (
        <span className="flex size-3 shrink-0 items-center justify-center rounded-full bg-[#0f9d58]/10">
          <CheckIcon className="size-2.5 text-[#0f9d58]" />
        </span>
      )}
      <span className="truncate text-[13px] font-medium">{label}</span>
      {running && args?.message ? (
        <span className="min-w-0 flex-1 truncate text-right text-[11px] text-muted-foreground">
          {args.message.replace(/\s+/g, " ").slice(0, 90)}
        </span>
      ) : null}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Planner card — the decomposed research plan (standalone).
// ---------------------------------------------------------------------------

interface PlanStep {
  id: string;
  text: string;
  sub?: string;
}

function extractPlanSteps(fields: Record<string, unknown> | undefined): PlanStep[] {
  if (!fields) return [];
  const requirements = fields.requirements;
  if (Array.isArray(requirements)) {
    return requirements
      .map((r, i) => {
        if (r && typeof r === "object") {
          const o = r as Record<string, unknown>;
          const id = typeof o.id === "string" ? o.id : `T${i + 1}`;
          const text =
            (typeof o.text === "string" && o.text.trim()) ||
            (typeof o.question === "string" && o.question.trim()) ||
            JSON.stringify(o).slice(0, 140);
          return { id, text: text.slice(0, 150), sub: typeof o.target_n === "number" ? `target · ${o.target_n} sources` : undefined };
        }
        return { id: `T${i + 1}`, text: String(r).slice(0, 150) };
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
          return { id: (typeof o.id === "string" && o.id) || `T${i + 1}`, text: (title || JSON.stringify(o)).slice(0, 150), sub: desc || undefined };
        }
        return { id: `T${i + 1}`, text: String(t).slice(0, 150) };
      })
      .filter((s) => s.text.length > 0);
  }
  const arr = firstArray(fields);
  if (arr) {
    return arr.map((item, i) => ({
      id: `#${i + 1}`,
      text: (typeof item === "string" ? item : JSON.stringify(item)).slice(0, 150),
    }));
  }
  return [];
}

const PlanUI: ToolCallMessagePartComponent<{ fields?: Record<string, unknown> }> = ({ args }) => {
  const steps = extractPlanSteps(args?.fields);
  if (!steps.length) return null;
  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card">
      <div className="flex items-center gap-2 border-b border-border bg-accent/40 px-4 py-2.5">
        <ListChecksIcon className="size-4 text-primary" />
        <span className="text-sm font-semibold">Research plan</span>
        <span className="ml-auto rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
          {steps.length} task{steps.length === 1 ? "" : "s"}
        </span>
      </div>
      <ol className="divide-y divide-border">
        {steps.map((step, i) => (
          <li key={`${step.id}-${i}`} className="flex items-start gap-3 px-4 py-2.5 text-sm">
            <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-[11px] font-semibold text-primary">
              {step.id.replace(/^T/, "")}
            </span>
            <div className="min-w-0">
              <p className="leading-5">{step.text}</p>
              {step.sub ? (
                <p className="mt-0.5 text-xs text-muted-foreground">{step.sub}</p>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Sources card — verified evidence with citation numbers (standalone).
// ---------------------------------------------------------------------------

const SourcesUI: ToolCallMessagePartComponent<{ count?: number }> = ({ result }) => {
  const sources = (
    result && Array.isArray(result)
      ? result
      : Array.isArray((result as { sources?: Source[] } | undefined)?.sources)
        ? (result as { sources: Source[] }).sources
        : []
  ) as Source[];
  if (!sources.length) return null;
  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card">
      <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
        <BookOpenIcon className="size-4 text-primary" />
        <span className="text-sm font-semibold">Sources</span>
        <span className="ml-auto rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
          {sources.length}
        </span>
      </div>
      <ul className="grid grid-cols-1 gap-px sm:grid-cols-2">
        {sources.map((source, index) => {
          const href = source.url ?? (source.pmcid ? `https://pmc.ncbi.nlm.nih.gov/articles/${source.pmcid}/` : undefined);
          const meta = [source.journal, source.year ? String(source.year) : ""].filter(Boolean).join(" · ");
          const isWeb = (source as Source & { isWeb?: boolean }).isWeb;
          const inner = (
            <>
              <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border border-border text-[11px] font-semibold text-muted-foreground">
                {index + 1}
              </span>
              <span className="min-w-0">
                <span className="line-clamp-2 text-[13px] font-medium leading-5">
                  {source.title || source.id || "Untitled source"}
                </span>
                <span className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                  {isWeb ? <GlobeIcon className="size-3 shrink-0" /> : null}
                  <span className="truncate">{meta || (isWeb ? "web" : "source")}</span>
                </span>
                {source.snippet ? (
                  <span className="mt-1 line-clamp-2 text-xs leading-5 text-muted-foreground/90">
                    {source.snippet}
                  </span>
                ) : null}
              </span>
            </>
          );
          return (
            <li key={`${source.id ?? source.pmcid ?? index}-${index}`}>
              {href ? (
                <a href={href} target="_blank" rel="noreferrer" className="flex items-start gap-2.5 px-4 py-3 transition hover:bg-accent/40">
                  {inner}
                </a>
              ) : (
                <div className="flex items-start gap-2.5 px-4 py-3">{inner}</div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Memory card.
// ---------------------------------------------------------------------------

type MemoryArgs = {
  kind: string;
  session_title?: string;
  prior_claims?: number;
  prior_contradictions?: number;
  prior_gaps?: number;
};

const MemoryUI: ToolCallMessagePartComponent<MemoryArgs> = ({ args, status }) => {
  const kind = args?.kind ?? "";
  const running = status.type === "running";
  const label =
    kind === "prepare" ? args?.session_title || "Memory loaded" : kind === "commit" ? "Memory updated" : "Memory";
  const detail: string[] = [];
  if (typeof args?.prior_claims === "number") detail.push(`${args.prior_claims} prior claims`);
  if (typeof args?.prior_contradictions === "number") detail.push(`${args.prior_contradictions} contradictions`);
  if (typeof args?.prior_gaps === "number") detail.push(`${args.prior_gaps} gaps`);
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 text-[13px]">
      {running ? (
        <span className="block size-3 animate-spin rounded-full border-[1.5px] border-primary/30 border-t-primary" />
      ) : (
        <MemoryStickIcon className="size-3.5 shrink-0 text-muted-foreground" />
      )}
      <span className="font-medium">{label}</span>
      {detail.length ? <span className="truncate text-muted-foreground">{detail.join(" · ")}</span> : null}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Toolkit registration (render-only backend entries; the graph executes
// server-side — we only attach UI to matching tool-call parts).
// ---------------------------------------------------------------------------

export const toolkit = defineToolkit({
  step: { type: "backend", render: StepUI },
  status: { type: "backend", render: StageUI },
  memory: { type: "backend", render: MemoryUI },
  plan: { type: "backend", display: "standalone", render: PlanUI },
  sources: { type: "backend", display: "standalone", render: SourcesUI },
});