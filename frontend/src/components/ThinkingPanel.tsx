import { useAuiState } from "@assistant-ui/react";
import { CheckIcon, ChevronDownIcon } from "lucide-react";
import { useEffect, useRef, useState, type PropsWithChildren } from "react";
import { STAGE_LABELS } from "../lib/labels";
import { stepMeta } from "../lib/toolkit";

function currentStage(): string {
  // useAuiState must select a primitive (stable across renders).
  // eslint-disable-next-line react-hooks/rules-of-hooks
  return useAuiState((s) => {
    const parts = s.message.content;
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      if (p?.type === "tool-call" && p.toolName === "status") {
        return ((p as { args?: { stage?: string } }).args?.stage ?? "") as string;
      }
    }
    return "";
  });
}

function useElapsed(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  const startRef = useRef<number | null>(null);
  useEffect(() => {
    if (running) {
      startRef.current = Date.now();
      setSeconds(0);
      const id = setInterval(() => {
        if (startRef.current) setSeconds(Math.round((Date.now() - startRef.current) / 1000));
      }, 1000);
      return () => clearInterval(id);
    }
    startRef.current = null;
    return undefined;
  }, [running]);
  return seconds;
}

/**
 * The collapsible "thinking" panel: while a run is in flight it stays open
 * and streams tool cards live (with the current stage + elapsed time);
 * once the run settles it collapses to a compact "View thoughts" pill.
 */
export function ThinkingPanel({ count, children }: PropsWithChildren<{ count: number }>) {
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const stage = currentStage();
  const elapsed = useElapsed(isRunning);
  const [open, setOpen] = useState(false);

  // auto-open while working, auto-collapse when done
  useEffect(() => {
    setOpen(isRunning);
  }, [isRunning]);

  const stageLabel = STAGE_LABELS[stage] ?? (stage ? stage : "thinking");
  const meta = stepMeta("thought"); // fallback icon for the trigger

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-sm transition hover:bg-muted/50"
      >
        <ChevronDownIcon
          className={`size-4 shrink-0 text-muted-foreground transition-transform duration-200 ${
            open ? "" : "-rotate-90"
          }`}
        />
        {isRunning ? (
          <span className="flex shrink-0 gap-1" aria-hidden>
            <span className="dsh-dot size-1.5 rounded-full bg-primary" />
            <span className="dsh-dot size-1.5 rounded-full bg-primary" />
            <span className="dsh-dot size-1.5 rounded-full bg-primary" />
          </span>
        ) : (
          <CheckIcon className="size-3.5 shrink-0 text-[#0f9d58]" />
        )}
        <span className="min-w-0 truncate font-medium">
          {isRunning ? (
            <>
              Thinking · <span className="text-muted-foreground">{stageLabel}</span>
            </>
          ) : (
            "View thoughts"
          )}
        </span>
        <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
          {count}
        </span>
        {isRunning ? (
          <span className="shrink-0 font-mono text-[11px] text-muted-foreground tabular-nums">
            {elapsed}s
          </span>
        ) : null}
      </button>
      {open ? <div className="border-t border-border/60 pb-1.5 pt-1.5">{children}</div> : null}
    </div>
  );
}

/**
 * One grouped tool card (e.g. "Web search ×3"): header with icon + label +
 * count, then one row per invocation.
 */
export function StepCard(
  props: PropsWithChildren<{ kind: string; count: number; running: boolean }>,
) {
  const { kind, count, running } = props;
  const meta = stepMeta(kind);
  const Icon = meta.icon;
  return (
    <div className="my-1 overflow-hidden rounded-xl border border-border/70 bg-muted/20">
      <div className="flex items-center gap-2.5 border-b border-border/50 bg-muted/40 px-3 py-2">
        <span className="flex size-6 shrink-0 items-center justify-center rounded-lg bg-card">
          <Icon className={`size-3.5 ${meta.tone}`} />
        </span>
        <span className="text-[13px] font-semibold">{meta.label}</span>
        {count > 1 ? (
          <span className="rounded-full bg-muted px-1.5 py-0.5 text-[11px] leading-none text-muted-foreground">
            {count}
          </span>
        ) : null}
        <span className="ml-auto">
          {running ? (
            <span className="block size-3 animate-spin rounded-full border-[1.5px] border-primary/30 border-t-primary" />
          ) : (
            <CheckIcon className="size-3.5 text-[#0f9d58]" />
          )}
        </span>
      </div>
      <div className="py-1">{props.children}</div>
    </div>
  );
}