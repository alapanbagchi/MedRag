// ── Research pipeline status: LED stage lamps ───────────────────────
"use client";

import { useEffect, useMemo, useState } from "react";
import type { Message, ResearchStatus, StageEvent } from "@/lib/types";
import { formatDuration } from "@/lib/utils";
import { CapsLabel, Led } from "@/components/ui/primitives";

const STAGE_ORDER: ResearchStatus[] = [
  "understanding", "decomposing", "retrieving", "reranking", "verifying", "synthesizing",
];

export const STAGE_LABEL: Record<ResearchStatus, string> = {
  idle: "Idle",
  understanding: "Understanding query",
  decomposing: "Decomposing question",
  retrieving: "Retrieving evidence",
  reranking: "Reranking evidence",
  verifying: "Verifying claims",
  synthesizing: "Synthesizing answer",
  complete: "Complete",
  error: "Error",
};

type StageState = "done" | "active" | "pending" | "error";

export function ThinkingStatus({ message }: { message: Message }) {
  const [elapsed, setElapsed] = useState(0);
  const startedAt = message.startedAt ?? message.createdAt;

  useEffect(() => {
    if (message.status === "queued" || message.status === "streaming") {
      const t = window.setInterval(() => setElapsed(Date.now() - startedAt), 250);
      return () => window.clearInterval(t);
    }
    setElapsed(Date.now() - startedAt);
  }, [message.status, startedAt]);

  const frosted = useMemo(() => {
    const events = message.stages ?? [];
    // Collapse repeated status events for the same stage so the LED lamps
    // advance one stage at a time, in true emitted order (never a future
    // stage shown as active, and never fake progress).
    const distinctOrder: ResearchStatus[] = [];
    for (const e of events) {
      if (distinctOrder[distinctOrder.length - 1] !== e.stage) distinctOrder.push(e.stage);
    }
    const lastDistinct = distinctOrder[distinctOrder.length - 1];
    return STAGE_ORDER.map((stage) => {
      const event = events.find((e) => e.stage === stage);
      const emitted = distinctOrder.includes(stage);
      let state: StageState = "pending";
      if (message.status === "error" && stage === lastDistinct) state = "error";
      else if (message.status === "streaming" || message.status === "queued") {
        if (stage === lastDistinct) state = "active";
        else if (emitted) state = "done";
      } else {
        // complete / stopped: every stage that actually emitted is done
        state = emitted ? "done" : "pending";
      }
      return { stage, label: STAGE_LABEL[stage], event, state };
    });
  }, [message.stages, message.status]);

  const isWorking = message.status === "queued" || message.status === "streaming";

  return (
    <div className="panel corner-ticks mb-4 overflow-hidden">
      <div className="flex items-center justify-between border-b border-line px-3 py-2">
        <CapsLabel>MedPat research pipeline</CapsLabel>
        <div className="flex items-center gap-2">
          {isWorking && <span className="mono text-[10px] uppercase tracking-[0.14em] text-accent">Working</span>}
          <span className="mono text-[10px] text-ink3 tnum">{formatDuration(elapsed)}</span>
        </div>
      </div>
      <ul className="grid gap-px bg-line text-[13px] sm:grid-cols-2">
        {frosted.map((row) => (
          <li key={row.stage} className="flex items-center gap-2.5 bg-panel px-3 py-2">
            <Led
              state={row.state === "active" ? "accent" : row.state === "done" ? "ok" : row.state === "error" ? "err" : "dim"}
              pulse={row.state === "active"}
            />
            <span className={row.state === "pending" ? "text-ink3" : "text-ink"}>{row.label}</span>
            {row.event?.count != null && (
              <span className="mono ml-auto text-[10px] text-ink3 tnum">{row.event.count}</span>
            )}
            {row.event?.message && row.state !== "pending" && (
              <span className="mono ml-auto hidden max-w-[40%] truncate text-[10px] text-ink3 md:block">
                {row.event.message}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
