import {
  defineToolkit,
  type ToolCallMessagePartComponent,
} from "@assistant-ui/react";
import {
  AlertTriangleIcon,
  BookOpenIcon,
  CheckIcon,
  ChevronRightIcon,
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
import { SourceLogo, SourceOrigin } from "../components/SourceBadge";
import { useChatStore } from "./store";

// ---------------------------------------------------------------------------
// Step metadata: icon + display label per step kind.
// ---------------------------------------------------------------------------

interface StepMeta {
  label: string;
  icon: ComponentType<{ className?: string }>;
  tone: string; // icon color class
}

export const STEP_META: Record<string, StepMeta> = {
  retrieve: { label: "Searching literature", icon: SearchIcon, tone: "text-[#1883AE]" },
  web_search: { label: "Web search", icon: GlobeIcon, tone: "text-[#18AE95]" },
  web_fetch: { label: "Fetching source", icon: ExternalLinkIcon, tone: "text-[#1883AE]" },
  reliability: { label: "Reliability check", icon: ShieldCheckIcon, tone: "text-[#18AE95]" },
  verdict: { label: "Evidence verdict", icon: ScaleIcon, tone: "text-[#b07d10]" },
  decompose: { label: "Decomposing the question", icon: NetworkIcon, tone: "text-[#1883AE]" },
  research: { label: "Research task", icon: FlaskConicalIcon, tone: "text-[#1883AE]" },
  search_round: { label: "Search round", icon: RefreshCwIcon, tone: "text-[#1883AE]" },
  contradiction: { label: "Contradiction", icon: AlertTriangleIcon, tone: "text-[#b07d10]" },
  resolution: { label: "Resolved", icon: CheckIcon, tone: "text-[#18AE95]" },
  gap_probe: { label: "Evidence gap", icon: TargetIcon, tone: "text-[#b07d10]" },
  gap_resolution: { label: "Gap resolved", icon: PuzzleIcon, tone: "text-[#18AE95]" },
  evidence: { label: "Verified evidence", icon: DatabaseIcon, tone: "text-[#1883AE]" },
  synthesize: { label: "Synthesizing answer", icon: PenLineIcon, tone: "text-[#1883AE]" },
  planner: { label: "Planning", icon: RouteIcon, tone: "text-[#1883AE]" },
  join: { label: "Joining findings", icon: GitMergeIcon, tone: "text-[#1883AE]" },
  done: { label: "Finished", icon: CheckIcon, tone: "text-[#18AE95]" },
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
  if (a.kind === "thought") {
    return (
      <div className="flex items-start gap-2.5 px-4 py-1.5">
        <span className="mt-[7px] size-1.5 shrink-0 rounded-full bg-[#18AE95]" />
        <p className="min-w-0 flex-1 text-[13px] leading-5 text-[#3c4450]">
          {a.detail || a.label}
        </p>
      </div>
    );
  }
  const inspectable = typeof a.callId === "string" && a.callId.length > 0;
  const open = () => {
    if (inspectable) useChatStore.getState().openInspector(a.callId as string);
  };
  return (
    <div
      className={inspectable ? "flex cursor-pointer items-center gap-2.5 px-4 py-1.5 transition-colors hover:bg-muted/60" : "flex items-center gap-2.5 px-4 py-1.5"}
      onClick={inspectable ? open : undefined}
      onKeyDown={inspectable ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } } : undefined}
      role={inspectable ? "button" : undefined}
      tabIndex={inspectable ? 0 : undefined}
      title={inspectable ? "Open tool-call inspector" : undefined}
    >
      <span className="min-w-0 flex-1 truncate text-[13px] text-[#232838]" title={a.detail ?? a.label}>
        {a.detail || a.label}
      </span>
      {a.error ? (
        <span className="shrink-0 text-[11px] text-destructive" title={a.error}>
          {a.error.length > 40 ? `${a.error.slice(0, 40)}…` : a.error}
        </span>
      ) : running ? (
        <span className="shrink-0">
          <span className="block size-3 animate-spin rounded-full border-[1.5px] border-[#1883AE]/30 border-t-[#1883AE]" />
        </span>
      ) : (
        <CheckIcon className="size-3.5 shrink-0 text-[#18AE95]" />
      )}
      {inspectable ? (
        <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
      ) : null}
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
    <div className="flex items-center gap-2.5 px-4 py-2">
      {running ? (
        <span className="flex shrink-0 gap-1" aria-hidden>
          <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
          <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
          <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
        </span>
      ) : (
        <CheckIcon className="size-3.5 shrink-0 text-[#18AE95]" />
      )}
      <span className="truncate text-[13px] font-medium text-[#0D0E1A]">{label}</span>
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

export interface PlanStep {
  id: string;
  text: string;
  sub?: string;
}

export function extractPlanSteps(fields: Record<string, unknown> | undefined): PlanStep[] {
  if (!fields) return [];
  // Task-list planner shape: { items: [{ id, question, deep_research }] }.
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
            id: typeof o.id === "string" ? o.id : `T${i + 1}`,
            text: text.slice(0, 150),
            sub: o.deep_research === true ? "deep research" : undefined,
          };
        }
        return { id: `T${i + 1}`, text: String(t).slice(0, 150) };
      })
      .filter((s) => s.text.length > 0);
  }
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
    <div className="anim-rise overflow-hidden rounded-2xl border border-border bg-white shadow-[0_8px_30px_rgba(13,14,26,0.06)]">
      <div className="flex items-center gap-2 border-b border-border/70 bg-[#F3F8F9] px-4 py-2.5">
        <ListChecksIcon className="size-4 text-[#1883AE]" />
        <span className="text-sm font-semibold text-[#0D0E1A]">Research plan</span>
        <span className="ml-auto rounded-full bg-white px-2 py-0.5 text-[11px] text-muted-foreground ring-1 ring-border">
          {steps.length} task{steps.length === 1 ? "" : "s"}
        </span>
      </div>
      <ol className="divide-y divide-border/70">
        {steps.map((step, i) => (
          <li key={`${step.id}-${i}`} className="flex items-start gap-3 px-4 py-2.5 text-sm">
            <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-[#1883AE]/10 font-mono text-[11px] font-semibold text-[#1883AE]">
              {step.id.replace(/^T/, "")}
            </span>
            <div className="min-w-0">
              <p className="leading-5 text-[#232838]">{step.text}</p>
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
// Live state ("Reading N sources…") vs complete state share one card.
// ---------------------------------------------------------------------------

const SourcesUI: ToolCallMessagePartComponent<{ count?: number }> = ({ result, status }) => {
  const sources = (
    result && Array.isArray(result)
      ? result
      : Array.isArray((result as { sources?: Source[] } | undefined)?.sources)
        ? (result as { sources: Source[] }).sources
        : []
  ) as Source[];
  if (!sources.length) return null;
  const running = status.type === "running";
  const visible = sources.slice(0, 4);
  const overflow = sources.length - visible.length;
  return (
    <div className="anim-rise overflow-hidden rounded-2xl border border-[#d7e5e9] bg-[#F4F9FA] shadow-[0_8px_30px_rgba(24,131,174,0.08)]">
      <div className="flex items-center gap-2 px-4 pb-1 pt-3">
        {running ? (
          <span className="flex shrink-0 gap-1" aria-hidden>
            <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
            <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
            <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
          </span>
        ) : null}
        <span className="text-[13px] font-medium text-[#5b6672]">
          {running ? `Reading ${sources.length} sources…` : `${sources.length} sources`}
        </span>
      </div>
      <ul className="space-y-0.5 px-4 pb-3 pt-1">
        {visible.map((source, index) => {
          const href = source.url ?? (source.pmcid ? `https://pmc.ncbi.nlm.nih.gov/articles/${source.pmcid}/` : undefined);
          const label = source.title || source.id || "Untitled source";
          const row = (
            <>
              <span className="mt-[7px] size-1 shrink-0 rounded-full bg-[#9fb0ba]" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[14px] font-medium leading-5 text-[#0D0E1A]">
                  {label}
                </span>
                <span className="mt-0.5 flex items-center gap-1.5">
                  <SourceLogo source={source} size="sm" />
                  <SourceOrigin source={source} />
                </span>
              </span>
            </>
          );
          return (
            <li key={`${source.id ?? source.pmcid ?? index}-${index}`}>
              {href ? (
                <a href={href} target="_blank" rel="noreferrer" className="flex items-start gap-2 rounded-lg px-1 py-1.5 transition hover:bg-white">
                  {row}
                </a>
              ) : (
                <div className="flex items-start gap-2 px-1 py-1.5">{row}</div>
              )}
            </li>
          );
        })}
        {overflow > 0 ? (
          <li className="flex items-center gap-2 px-1 pt-0.5">
            <span className="mt-0 size-1 shrink-0 rounded-full bg-[#9fb0ba]" />
            <span className="rounded-full bg-white px-2 py-0.5 font-mono text-[11px] text-muted-foreground ring-1 ring-border">
              +{overflow}
            </span>
          </li>
        ) : null}
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
    <div className="flex items-center gap-2 px-4 py-1.5 text-[13px]">
      {running ? (
        <span className="block size-3 animate-spin rounded-full border-[1.5px] border-[#1883AE]/30 border-t-[#1883AE]" />
      ) : (
        <MemoryStickIcon className="size-3.5 shrink-0 text-muted-foreground" />
      )}
      <span className="font-medium text-[#0D0E1A]">{label}</span>
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
