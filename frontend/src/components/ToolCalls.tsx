import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import { ToolCall } from "./assistant-ui/elements/tool-call";
import {
  ToolTimeline,
  type TimelineStep,
} from "./assistant-ui/elements/tool-timeline";
import { stepMeta } from "../lib/toolkit";
import type { StepArgs } from "../lib/xdeep";
import { useChatStore } from "../lib/store";

/**
 * One streamed backend trace entry. `key` is the message part's toolCallId
 * (falling back to the backend callId) — it is what the detail sidebar uses
 * to look the entry back up. The backend tracing interface (StepArgs:
 * callId, rawArgs/rawResult, timeline, timestamps) is shared verbatim;
 * only this frontend mapping is per-tool customizable.
 */
export interface ToolEntry {
  key: string;
  step: StepArgs;
}

function shortText(value: string, max: number): string {
  const text = value
    .replace(/^https?:\/\/(www\.)?/, "")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

/** Chip next to the tool name: query, url, or detail — always short. */
function chipOf(step: StepArgs): string {
  return shortText(step.query ?? step.url ?? step.detail ?? step.label, 28);
}

function requestOf(step: StepArgs): string {
  const raw = step.rawArgs;
  const text =
    typeof raw === "string"
      ? raw
      : raw !== undefined
        ? JSON.stringify(raw)
        : (step.query ?? step.url ?? step.detail ?? step.label);
  return shortText(text, 160);
}

function resultOf(step: StepArgs): string {
  if (step.error) return step.error;
  return shortText(step.detail || step.label, 160);
}

/**
 * One tool-call row. A click opens the detail sidebar (the inline panel
 * stays closed — full input/output/timing live in the sidebar).
 */
function ToolCallRow({ entry }: { entry: ToolEntry }) {
  const openInspector = useChatStore((s) => s.openInspector);
  const { step } = entry;
  const running = !step.done;
  return (
    <ToolCall
      label={step.label}
      activeLabel={`${step.label}…`}
      query={chipOf(step)}
      request={requestOf(step)}
      result={resultOf(step)}
      running={running}
      open={false}
      onOpenChange={() => openInspector(entry.key)}
      className="max-w-none"
    />
  );
}

/** Map trace entries to timeline rows, keeping chips unique (element keys). */
function timelineSteps(entries: ToolEntry[]): TimelineStep[] {
  const seen = new Map<string, number>();
  return entries.map(({ step }) => {
    const meta = stepMeta(step.kind);
    const base = chipOf(step);
    const n = (seen.get(base) ?? 0) + 1;
    seen.set(base, n);
    return {
      verb: step.label,
      chip: n > 1 ? `${base} (${n})` : base,
      icon: meta.icon as unknown as LucideIcon,
    };
  });
}

/**
 * General tool-call architecture: one ToolCall row per trace entry
 * (name + purpose chip, click → sidebar) plus a ToolTimeline summary.
 * Per-tool display tuning happens in chipOf/requestOf/resultOf and the
 * timeline mapper above — one table to edit per tool, later.
 */
export function ToolCalls({
  entries,
  isRunning,
  stageLabel,
}: {
  entries: ToolEntry[];
  isRunning: boolean;
  stageLabel: string;
}) {
  const [summaryOpen, setSummaryOpen] = useState(false);
  if (entries.length === 0) return null;
  return (
    <div className="space-y-0.5 px-1 py-1">
      {entries.map((entry) => (
        <ToolCallRow key={entry.key} entry={entry} />
      ))}
      <ToolTimeline
        steps={timelineSteps(entries)}
        visibleSteps={entries.length}
        streaming={isRunning}
        open={summaryOpen}
        onOpenChange={setSummaryOpen}
        restingLabel={`${entries.length} tool call${entries.length === 1 ? "" : "s"}`}
        activeLabel={`${stageLabel}…`}
        stats={[]}
        className="max-w-none"
      />
    </div>
  );
}
