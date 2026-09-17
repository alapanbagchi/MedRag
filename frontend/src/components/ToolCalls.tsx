import { memo, useEffect, useRef, useState } from "react";
import { ToolCall } from "./assistant-ui/elements/tool-call";
import type { StepArgs } from "../lib/xdeep";
import { useAgUiUiStore } from "../agui/aguiStore";

/**
 * One streamed backend trace entry. `key` is the backend call id — the
 * inspector looks the entry back up by it.
 */
export interface ToolEntry {
  key: string;
  step: StepArgs;
}

function shortText(value: unknown, max: number): string {
  const text = String(value ?? "")
    .replace(/^https?:\/\/(www\.)?/, "")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > max ? text.slice(0, max) + "…" : text;
}

/** Chip next to the tool name: query, url, or detail — always short. */
function chipOf(step: StepArgs): string {
  return shortText(step.query ?? step.url ?? step.detail ?? step.label, 28);
}

/** One tool-call row; a click opens the AG-UI inspector. */
export function ToolCallRow({ entry }: { entry: ToolEntry }) {
  const inspect = useAgUiUiStore((s) => s.inspect);
  const { step } = entry;
  return (
    <ToolCall
      label={step.label}
      activeLabel={step.label + "…"}
      query={chipOf(step)}
      running={!step.done}
      onOpenChange={() => inspect(entry.key)}
      className="max-w-none"
    />
  );
}

function rowEqual(prev: { entry: ToolEntry }, next: { entry: ToolEntry }): boolean {
  return prev.entry.key === next.entry.key && prev.entry.step === next.entry.step;
}

export const MemoToolCallRow = memo(ToolCallRow, rowEqual);

const EXIT_MS = 320;

interface RenderedEntry {
  key: string;
  entry: ToolEntry;
  leaving: boolean;
}

function useAnimatedEntries(entries: ToolEntry[]): RenderedEntry[] {
  const [rendered, setRendered] = useState<RenderedEntry[]>(() =>
    entries.map((entry) => ({ key: entry.key, entry, leaving: false })),
  );
  const renderedRef = useRef(rendered);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  renderedRef.current = rendered;

  useEffect(() => {
    const incoming = new Map(entries.map((entry) => [entry.key, entry]));
    const seen = new Set<string>();
    const next: RenderedEntry[] = [];

    for (const item of renderedRef.current) {
      const live = incoming.get(item.key);
      if (live) {
        seen.add(item.key);
        next.push({ key: item.key, entry: live, leaving: false });
        const timer = timers.current.get(item.key);
        if (timer) {
          clearTimeout(timer);
          timers.current.delete(item.key);
        }
      } else if (!item.leaving) {
        next.push({ ...item, leaving: true });
        const timer = setTimeout(() => {
          timers.current.delete(item.key);
          setRendered((current) => current.filter((it) => it.key !== item.key));
        }, EXIT_MS);
        timers.current.set(item.key, timer);
      } else {
        next.push(item);
      }
    }
    for (const entry of entries) {
      if (!seen.has(entry.key)) next.push({ key: entry.key, entry, leaving: false });
    }

    const unchanged =
      next.length === renderedRef.current.length &&
      next.every((item, i) => {
        const prev = renderedRef.current[i];
        return (
          prev &&
          prev.key === item.key &&
          prev.entry === item.entry &&
          prev.leaving === item.leaving
        );
      });
    if (!unchanged) setRendered(next);
  }, [entries]);

  useEffect(
    () => () => {
      for (const timer of timers.current.values()) clearTimeout(timer);
    },
    [],
  );

  return rendered;
}

/** Live tool-call rows with enter/exit animation. */
export function ToolCalls({ entries }: { entries: ToolEntry[] }) {
  const rendered = useAnimatedEntries(entries);
  if (rendered.length === 0) return null;
  return (
    <div className="space-y-0.5 px-1 py-1">
      {rendered.map(({ key, entry, leaving }) => (
        <div key={key} className={leaving ? "anim-tool-out" : "anim-tool-in"}>
          <MemoToolCallRow entry={entry} />
        </div>
      ))}
    </div>
  );
}
