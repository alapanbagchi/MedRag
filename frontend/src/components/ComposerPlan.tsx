import { useMemo, useState, type ReactNode } from "react";
import { useAuiState } from "@assistant-ui/react";
import {
  CheckIcon,
  ChevronDownIcon,
  ListChecksIcon,
  TriangleAlertIcon,
  XIcon,
} from "lucide-react";
import { cn } from "../lib/utils";
import { mono } from "@/lib/surfaces";
import {
  parseRequirementList,
  parseVerdicts,
  scoreCriteria,
  splitJudgment,
  type PassageVerdict,
} from "../lib/evidence";
import { extractPlanSteps } from "../lib/toolkit";
import { partArgs, partName, taskStatesOf, type Content } from "../lib/parts";
import type { StepArgs } from "../lib/xdeep";

export type EvidenceStatus = "pending" | "satisfied" | "partial" | "unsatisfied";
export type TaskStatus = "pending" | "running" | "done" | "failed";

export interface PlanEvidence {
  /** Stable React key (`<task>:<id>`). */
  key: string;
  id: string;
  text: string;
  status: EvidenceStatus;
}

export interface PlanTaskItem {
  id: string;
  text: string;
  status: TaskStatus;
  evidence: PlanEvidence[];
}

export interface ComposerPlanData {
  tasks: PlanTaskItem[];
  orphan: PlanEvidence[];
  running: boolean;
  doneCount: number;
  taskCount: number;
  satisfied: number;
  evidenceCount: number;
}

/** Message shape common to the legacy store and the AG-UI runtime. */
interface MessageLike {
  role?: string;
  content?: unknown;
  status?: unknown;
}

/**
 * Derive the live plan + per-task evidence from a message list. Reads parts
 * through `partName`/`partArgs` so the same logic serves the NDJSON
 * (`tool-call`) store and the AG-UI (`data`) runtime.
 */
export function deriveComposerPlan(messages: readonly MessageLike[]): ComposerPlanData | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (!message || message.role !== "assistant" || !Array.isArray(message.content)) continue;
    const content = message.content as unknown[];
    const planPart = content.find((p) => partName(p) === "plan");
    if (!planPart) continue;

    const planArgs = (partArgs(planPart) ?? {}) as {
      fields?: Record<string, unknown>;
      result?: { fields?: Record<string, unknown> };
    };
    const fields = planArgs.fields ?? planArgs.result?.fields;
    const planSteps = extractPlanSteps(fields);
    if (planSteps.length === 0) return null;

    const running = (message.status as { type?: string } | undefined)?.type === "running";
    const taskStates = taskStatesOf(content as Content);

    const steps = content.filter((p) => partName(p) === "step");
    const evidence = buildEvidence(fields, steps);
    const byTask = new Map<string, PlanEvidence[]>();
    const orphan: PlanEvidence[] = [];
    const knownIds = new Set(planSteps.map((s) => s.id));
    for (const item of evidence) {
      if (item.task && knownIds.has(item.task)) {
        byTask.set(item.task, [...(byTask.get(item.task) ?? []), item]);
      } else {
        orphan.push(item);
      }
    }

    const isClosed = (id: string): boolean => {
      const leg = taskStates.get(id)?.state;
      return leg === "done" || leg === "failed";
    };
    const firstOpen = planSteps.findIndex((s) => s.done !== true && !isClosed(s.id));

    const tasks: PlanTaskItem[] = planSteps.map((s, index) => {
      const leg = taskStates.get(s.id)?.state;
      let status: TaskStatus = "pending";
      if (leg === "failed") status = "failed";
      else if (s.done === true || leg === "done") status = "done";
      else if (leg === "started" || s.started === true) status = "running";
      else if (running && index === firstOpen) status = "running";
      return { id: s.id, text: s.text, status, evidence: byTask.get(s.id) ?? [] };
    });

    const doneCount = tasks.filter((t) => t.status === "done").length;
    const satisfied = evidence.filter((e) => e.status === "satisfied").length;
    return {
      tasks,
      orphan,
      running,
      doneCount,
      taskCount: tasks.length,
      satisfied,
      evidenceCount: evidence.length,
    };
  }
  return null;
}

/** Live plan for the AG-UI runtime surface. */
export function useAgUiPlan(): ComposerPlanData | null {
  const messages = useAuiState((s) => s.thread.messages);
  return useMemo(() => deriveComposerPlan(messages as unknown as MessageLike[]), [messages]);
}

/**
 * Card that owns the composer morph: the plan sheet slides open above the
 * input inside one surface, and the input (provided by `children`) collapses
 * to a single line / switches its placeholder via the `active` flag.
 */
export function PlanMorph({
  plan,
  className,
  children,
}: {
  plan: ComposerPlanData | null;
  className?: string;
  children: (active: boolean) => ReactNode;
}) {
  const active = plan !== null;
  return (
    <div className={cn("relative w-full", className)}>
      {/* Plan panel, absolutely anchored above the input container so the
          input's geometry never changes. It shares the container's border and
          squares off the card's top corners so the two read as one surface. */}
      <div
        aria-hidden={!active}
        className={cn(
          "absolute inset-x-0 bottom-full z-20 transition-opacity duration-300 ease-out motion-reduce:transition-none",
          active ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      >
        <div
          className={cn(
            "grid overflow-hidden rounded-t-[26px] border border-b-0 border-border bg-card transition-[grid-template-rows] duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none",
            active ? "grid-rows-[1fr] shadow-[0_-20px_50px_rgba(0,0,0,0.35)]" : "grid-rows-[0fr]",
          )}
        >
          <div className="overflow-hidden">{plan ? <ComposerPlan plan={plan} /> : null}</div>
        </div>
      </div>

      {/* Input container. Geometry is identical in both states — only the top
          corners are squared when the sheet is attached. */}
      <div
        className={cn(
          "w-full overflow-hidden rounded-[26px] border border-border bg-card shadow-[0_20px_60px_rgba(0,0,0,0.35)] backdrop-blur-2xl transition-[border-radius,box-shadow] duration-500 ease-out focus-within:border-[#4b8cf5]/50 focus-within:ring-2 focus-within:ring-[#4b8cf5]/20",
          active && "rounded-t-none",
        )}
      >
        {children(active)}
      </div>
    </div>
  );
}

/**
 * Plan sheet body: header + task tree. Rendered inside the morphing composer
 * card so the input and the plan share one surface.
 */
export function ComposerPlan({ plan }: { plan: ComposerPlanData }) {
  const [open, setOpen] = useState(true);
  return (
    <section aria-label="Plan of action">
      <header className="flex items-center gap-2 px-4 pt-3 pb-2.5">
        <ListChecksIcon className="size-4 shrink-0 text-[#4b8cf5]" />
        <h3 className="text-[15px] font-semibold tracking-[-0.01em] text-foreground">
          Plan of action
        </h3>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={open ? "Collapse plan" : "Expand plan"}
          className="ml-auto flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#4b8cf5]/40"
        >
          <ChevronDownIcon
            className={cn("size-4 transition-transform duration-300", !open && "-rotate-90")}
          />
        </button>
      </header>

      <div
        className={cn(
          "grid transition-[grid-template-rows] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none",
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className="overflow-hidden">
          <div className="max-h-[320px] overflow-y-auto overscroll-contain px-4 pb-4">
            <ol className="relative">
              {plan.tasks.map((task, i) => (
                <TaskRow key={task.id} task={task} last={i === plan.tasks.length - 1} />
              ))}
            </ol>

            {plan.orphan.length > 0 ? (
              <div className="relative mt-3">
                <div className="flex items-center gap-3">
                  <span className="relative z-10 flex size-[11px] shrink-0 items-center justify-center">
                    <span aria-hidden className="size-2.5 rounded-full bg-muted-foreground/40" />
                  </span>
                  <span className="text-[12.5px] font-medium text-muted-foreground">
                    General evidence
                  </span>
                </div>
                <ul className="relative ml-[5px] mt-1.5">
                  {plan.orphan.map((item, i) => (
                    <EvidenceRow
                      key={item.key}
                      evidence={item}
                      last={i === plan.orphan.length - 1}
                    />
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}

function TaskRow({ task, last }: { task: PlanTaskItem; last: boolean }) {
  return (
    <li className="relative pb-4 last:pb-0">
      {/* Spine linking this task's node to the next one, running behind
          the evidence branches so the tree reads as one continuous line. */}
      {!last ? (
        <span
          aria-hidden
          className="absolute left-[5px] top-[9px] -bottom-2 w-px bg-gradient-to-b from-tree-line-strong/70 to-tree-line-strong/35"
        />
      ) : null}

      <div className="relative flex items-start gap-3">
        <span className="relative z-10 mt-[3px] flex size-[11px] shrink-0 items-center justify-center">
          <StatusDot status={task.status} />
        </span>
        <span
          className={cn(
            "min-w-0 flex-1 text-[15px] font-semibold leading-[1.35] tracking-[-0.005em]",
            task.status === "done" && "text-foreground/50",
            task.status === "failed" && "text-foreground/75",
            task.status === "running" && "text-foreground",
            task.status === "pending" && "text-foreground/75",
          )}
        >
          <span className={cn(mono, "mr-1.5 text-[11.5px] font-normal text-[#4b8cf5]/60")}>
            {task.id}
          </span>
          {task.text}
        </span>
      </div>

      {task.evidence.length > 0 ? (
        <ul className="relative ml-[5px] mt-2">
          {task.evidence.map((item, i) => (
            <EvidenceRow
              key={item.key}
              evidence={item}
              last={i === task.evidence.length - 1}
            />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function EvidenceRow({ evidence, last }: { evidence: PlanEvidence; last: boolean }) {
  const satisfied = evidence.status === "satisfied";
  return (
    <li className="anim-rise relative flex items-start gap-2.5 py-[5px] pl-4 motion-reduce:animate-none">
      {/* Straight spine through the row (half height on the final branch so
          the line ends at the curve). */}
      <span
        aria-hidden
        className={cn(
          "absolute left-0 top-0 w-px bg-tree-line/60",
          last ? "h-1/2" : "h-full",
        )}
      />
      {/* Curved elbow: down the spine, then a rounded corner out to the text. */}
      <span
        aria-hidden
        className="absolute left-0 top-0 h-1/2 w-[13px] rounded-bl-[7px] border-b border-l border-tree-line/70"
      />
      <span
        className={cn(
          "min-w-0 flex-1 text-[13.5px] leading-[1.55] break-words",
          satisfied
            ? "text-muted-foreground/50 line-through decoration-muted-foreground/60 decoration-[1.4px]"
            : "text-foreground/75",
        )}
      >
        <span className={cn(mono, "mr-1.5 text-[10.5px] text-[#4b8cf5]/70")}>
          {evidence.id}
        </span>
        {evidence.text}
      </span>
      <span className="mt-1 flex size-4 shrink-0 items-center justify-center">
        <EvidenceGlyph status={evidence.status} />
      </span>
    </li>
  );
}

function EvidenceGlyph({ status }: { status: EvidenceStatus }) {
  if (status === "satisfied") return <CheckIcon className="size-4 text-[#18AE95]" />;
  if (status === "partial") return <TriangleAlertIcon className="size-3.5 text-amber-500" />;
  if (status === "unsatisfied") return <XIcon className="size-3.5 text-[#e2534b]" />;
  return <span aria-hidden className="size-2 rounded-full border border-muted-foreground/40" />;
}

function StatusDot({ status }: { status: TaskStatus }) {
  if (status === "running") {
    return (
      <span className="relative flex size-[11px] items-center justify-center" role="status" aria-label="running">
        <span aria-hidden className="absolute inline-flex size-2.5 animate-ping rounded-full bg-[#4b8cf5]/35 motion-reduce:hidden" />
        <span aria-hidden className="relative size-2.5 rounded-full bg-[#4b8cf5] shadow-[0_0_0_3px_rgba(75,140,245,0.14)]" />
      </span>
    );
  }
  const tone =
    status === "done"
      ? "bg-[#18AE95] shadow-[0_0_0_3px_rgba(24,174,149,0.14)]"
      : status === "failed"
        ? "bg-[#e2534b] shadow-[0_0_0_3px_rgba(226,83,75,0.14)]"
        : "bg-muted-foreground/40";
  return <span aria-hidden className={cn("size-2.5 rounded-full", tone)} />;
}

/** One evidence requirement, namespaced to its plan task. */
interface Requirement {
  task: string;
  id: string;
  description: string;
}

/**
 * Evidence requirements for one run message.
 *
 * The current pipeline embeds them in the plan items
 * (`items[].evidence_requirements`), so that is the primary source. Older
 * xdeep runs emitted them from a `planner` step's raw result instead; both
 * shapes are merged (plan first) so either generation nests correctly.
 * Verdicts come from `retrieve` steps' judge payloads and score each
 * requirement, keyed by the call's task namespace (`P1:…` -> `P1`).
 */
function buildEvidence(
  fields: Record<string, unknown> | undefined,
  steps: unknown[],
): (PlanEvidence & { task: string })[] {
  const requirements = dedupe([
    ...requirementsFromPlan(fields),
    ...requirementsFromSteps(steps),
  ]);
  if (requirements.length === 0) return [];
  const verdictsByTask = verdictsFromSteps(steps);
  return requirements.map((req) => {
    const [scored] = scoreCriteria(
      [{ id: req.id, description: req.description }],
      verdictsByTask.get(req.task) ?? null,
    );
    return {
      task: req.task,
      key: `${req.task}:${req.id}`,
      id: req.id,
      text: req.description,
      status: scored?.status ?? "pending",
    };
  });
}

function dedupe(list: Requirement[]): Requirement[] {
  const seen = new Set<string>();
  const out: Requirement[] = [];
  for (const req of list) {
    const key = `${req.task}\u0000${req.id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(req);
  }
  return out;
}

function requirementsFromPlan(fields: Record<string, unknown> | undefined): Requirement[] {
  const items = fields?.items;
  if (!Array.isArray(items)) return [];
  const out: Requirement[] = [];
  for (const item of items) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const task = typeof o.id === "string" ? o.id : "";
    for (const req of parseRequirementList(o.evidence_requirements)) {
      out.push({ task, id: req.id, description: req.description });
    }
  }
  return out;
}

function requirementsFromSteps(steps: unknown[]): Requirement[] {
  const out: Requirement[] = [];
  for (const part of steps) {
    const args = (partArgs(part) ?? {}) as Partial<StepArgs>;
    if (args.kind !== "planner" || !args.done) continue;
    const task = taskOfCall(args.callId);
    for (const req of parseRequirementList(args.rawResult)) {
      out.push({ task, id: req.id, description: req.description });
    }
  }
  return out;
}

function verdictsFromSteps(steps: unknown[]): Map<string, PassageVerdict[]> {
  const byTask = new Map<string, PassageVerdict[]>();
  for (const part of steps) {
    const args = (partArgs(part) ?? {}) as Partial<StepArgs>;
    if (args.kind !== "retrieve" || !args.done) continue;
    const list = parseVerdicts(splitJudgment(args.rawResult).verdictsText);
    if (!list || list.length === 0) continue;
    const task = taskOfCall(args.callId);
    byTask.set(task, [...(byTask.get(task) ?? []), ...list]);
  }
  return byTask;
}

/** Task namespace of a step call id (`P1:call_…` -> `P1`, else ""). */
function taskOfCall(callId: unknown): string {
  return typeof callId === "string" && callId.includes(":")
    ? callId.slice(0, callId.indexOf(":"))
    : "";
}
