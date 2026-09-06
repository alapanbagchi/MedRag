import { useAuiState } from "@assistant-ui/react";
import { BrainIcon, CheckIcon, ChevronDownIcon } from "lucide-react";
import { useEffect, useRef, useState, type PropsWithChildren } from "react";
import { STAGE_LABELS } from "../lib/labels";

function currentStage(): string {
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
 * Master deep-agent thinking stream: while a run is in flight it stays open
 * and streams thoughts live (current stage + elapsed time); once the run
 * settles it collapses to a compact "Thoughts" pill.
 */
export function ThinkingPanel({ count, children }: PropsWithChildren<{ count: number }>) {
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const stage = currentStage();
  const elapsed = useElapsed(isRunning);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(isRunning);
  }, [isRunning]);

  const stageLabel = STAGE_LABELS[stage] ?? (stage ? stage : "thinking");

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-white shadow-[0_8px_30px_rgba(13,14,26,0.05)]">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left text-sm transition hover:bg-[#F3F8F9]"
      >
        <ChevronDownIcon
          className={`size-4 shrink-0 text-muted-foreground transition-transform duration-200 ${
            open ? "" : "-rotate-90"
          }`}
        />
        {isRunning ? (
          <span className="flex shrink-0 gap-1" aria-hidden>
            <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
            <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
            <span className="medrag-dot size-1.5 rounded-full bg-[#1883AE]" />
          </span>
        ) : (
          <BrainIcon className="size-4 shrink-0 text-[#1883AE]" />
        )}
        <span className="min-w-0 flex-1 truncate font-medium text-[#0D0E1A]">
          {isRunning ? (
            <>
              Thinking <span className="font-normal text-muted-foreground">· {stageLabel}</span>
            </>
          ) : (
            "Thoughts"
          )}
        </span>
        <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
          {count}
        </span>
        {isRunning ? (
          <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
            {elapsed}s
          </span>
        ) : null}
      </button>
      {open ? <div className="border-t border-border/60 py-1">{children}</div> : null}
    </div>
  );
}

/**
 * One grouped tool card (e.g. "Searching literature ×2"): header with icon +
 * label + count, then one row per invocation.
 */
export function StepCard(
  props: PropsWithChildren<{ kind: string; count: number; running: boolean; label: string; icon: React.ReactNode }>,
) {
  const { count, running, label, icon } = props;
  return (
    <div className="my-1 overflow-hidden rounded-xl border border-border/70 bg-white">
      <div className="flex items-center gap-2.5 border-b border-border/50 bg-[#F7FAFB] px-4 py-2">
        <span className="flex size-6 shrink-0 items-center justify-center rounded-lg bg-white ring-1 ring-border">
          {icon}
        </span>
        <span className="text-[13px] font-semibold text-[#0D0E1A]">{label}</span>
        {count > 1 ? (
          <span className="rounded-full bg-muted px-1.5 py-0.5 text-[11px] leading-none text-muted-foreground">
            ×{count}
          </span>
        ) : null}
        <span className="ml-auto">
          {running ? (
            <span className="block size-3 animate-spin rounded-full border-[1.5px] border-[#1883AE]/30 border-t-[#1883AE]" />
          ) : (
            <CheckIcon className="size-3.5 text-[#18AE95]" />
          )}
        </span>
      </div>
      <div className="py-1">{props.children}</div>
    </div>
  );
}
