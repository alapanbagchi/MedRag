import type { ThreadMessageLike } from "@assistant-ui/react";
import { useEffect } from "react";
import { XIcon } from "lucide-react";
import { useChatStore } from "../lib/store";
import type { StepArgs } from "../lib/xdeep";
import { stepMeta } from "../lib/toolkit";

const MAX_JSON_CHARS = 4000;

function formatTime(iso?: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString([], { hour12: false });
}

function formatDuration(start?: string, end?: string): string {
  if (!start || !end) return "";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "";
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function pretty(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value ?? null, null, 2);
  } catch {
    return String(value);
  }
}

function findStep(threadId: string | null, callId: string): StepArgs | null {
  if (!threadId) return null;
  const list = useChatStore.getState().messages[threadId] ?? [];
  for (const message of list) {
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (part?.type !== "tool-call") continue;
      const p = part as { toolName?: string; toolCallId?: string; args?: StepArgs };
      if (p.toolName !== "step") continue;
      const args = p.args;
      if (args && (args.callId === callId || p.toolCallId === callId)) return args;
    }
  }
  return null;
}

const EMPTY_MESSAGES: ThreadMessageLike[] = [];

/** Slide-over inspector for one backend tool call: inputs, live timeline, result. */
export function ToolInspector() {
  const selection = useChatStore((s) => s.inspector);
  const threadId = useChatStore((s) => s.currentThreadId);
  // NOTE: the fallback must be a stable reference — a fresh [] literal here
  // makes getSnapshot return a new value every time (infinite loop).
  const messages = useChatStore((s) => (s.currentThreadId ? (s.messages[s.currentThreadId] ?? EMPTY_MESSAGES) : EMPTY_MESSAGES));
  const closeInspector = useChatStore((s) => s.closeInspector);

  useEffect(() => {
    if (!selection) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeInspector();
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [selection, closeInspector]);

  if (!selection) return null;
  // Re-read on every store change so the timeline streams while open.
  void messages;
  const args = findStep(threadId, selection.callId);
  if (!args) return null;

  const meta = stepMeta(args.kind);
  const Icon = meta.icon;
  const running = !args.done;
  const failed = !!args.error;
  const status = failed ? "Failed" : running ? "Running" : "Completed";
  const timeline = args.timeline ?? [];
  const inputText = pretty(args.rawArgs ?? {});
  const resultText = pretty(args.rawResult ?? "(no result yet)");
  const duration = formatDuration(args.startedTs, args.finishedTs);

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Tool call inspector">
      <div className="absolute inset-0 bg-black/20" onClick={closeInspector} />
      <aside className="anim-slide-in absolute inset-y-0 right-0 flex w-full max-w-[420px] flex-col border-l bg-white shadow-xl">
        <div className="flex items-start gap-2.5 border-b px-5 py-4">
          <Icon className={`mt-0.5 size-4 shrink-0 ${meta.tone}`} />
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[14px] font-semibold text-[#0D0E1A]">{args.label}</h2>
            <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[12px] text-muted-foreground">
              <span
                className={
                  failed ? "font-medium text-destructive"
                  : running ? "font-medium text-[#1883AE]"
                  : "font-medium text-[#18AE95]"
                }
              >
                {status}
              </span>
              {duration ? <span>{duration}</span> : null}
              {args.startedTs ? <span>{formatTime(args.startedTs)}</span> : null}
            </p>
          </div>
          <button
            type="button"
            onClick={closeInspector}
            aria-label="Close inspector"
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <XIcon className="size-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <section>
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">Input</h3>
            <pre className="mt-1.5 overflow-x-auto rounded-md bg-muted/60 p-3 font-mono text-[12px] leading-5 text-[#232838]">
              {inputText.length > MAX_JSON_CHARS ? `${inputText.slice(0, MAX_JSON_CHARS)}…` : inputText}
            </pre>
          </section>

          <section className="mt-5">
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">Processing</h3>
            {timeline.length === 0 ? (
              <p className="mt-1.5 text-[13px] text-muted-foreground">Waiting for first update…</p>
            ) : (
              <ol className="mt-2 space-y-0">
                {timeline.map((item, i) => {
                  const last = i === timeline.length - 1;
                  return (
                    <li key={i} className="relative flex gap-2.5 pb-3 last:pb-0">
                      {i < timeline.length - 1 ? (
                        <span className="absolute left-[4px] top-4 h-[calc(100%-12px)] w-px bg-border" />
                      ) : null}
                      <span
                        className={
                          last && running
                            ? "mt-[7px] size-[9px] shrink-0 animate-pulse rounded-full bg-[#1883AE]"
                            : "mt-[7px] size-[9px] shrink-0 rounded-full border border-[#18AE95] bg-white"
                        }
                      />
                      <div className="min-w-0 flex-1">
                        <p className="text-[13px] leading-5 text-[#232838]">{item.text}</p>
                        {item.t ? (
                          <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">{formatTime(item.t)}</p>
                        ) : null}
                      </div>
                    </li>
                  );
                })}
              </ol>
            )}
          </section>

          <section className="mt-5">
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">Result</h3>
            <pre className="mt-1.5 overflow-x-auto whitespace-pre-wrap break-words rounded-md bg-muted/60 p-3 font-mono text-[12px] leading-5 text-[#232838]">
              {resultText.length > MAX_JSON_CHARS ? `${resultText.slice(0, MAX_JSON_CHARS)}…` : resultText}
            </pre>
          </section>
        </div>
      </aside>
    </div>
  );
}
