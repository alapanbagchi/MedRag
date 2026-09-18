/**
 * Tool-call inspector for the AG-UI surface.
 *
 * Steps arrive as data parts named "step"; clicking one opens this panel with
 * its arguments, result, and timeline.
 */

import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { useAuiState } from "@assistant-ui/react";
import { XIcon } from "lucide-react";
import { useAgUiUiStore } from "./aguiStore";

function Section({ title, children }: { title: string; children: ReactNode }) {
 return (
 <section>
 <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{title}</h3>
 <div className="text-[13px] leading-6 text-foreground">{children}</div>
 </section>
 );
}

/** Render any payload as text — never put an object in a React child. */
function Text({ value }: { value: unknown }) {
 if (value === undefined || value === null) return <em className="text-muted-foreground">—</em>;
 if (typeof value === "string") return <>{value}</>;
 try {
 return <>{JSON.stringify(value)}</>;
 } catch {
 return <>{String(value)}</>;
 }
}

function Pre({ value }: { value: unknown }) {
 if (value === undefined || value === null) {
 return <em className="text-muted-foreground">—</em>;
 }
 const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
 return (
 <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-composer p-3 font-mono text-[11.5px] leading-5 text-foreground">
 {text}
 </pre>
 );
}

/**
 * The always-mounted shell. It subscribes only to the UI store, so a closed
 * inspector never re-renders (or rescans every message) on each streamed
 * token; the panel that reads the thread mounts only while it is open.
 */
export function AgUiInspector() {
 const inspectedCallId = useAgUiUiStore((s) => s.inspectedCallId);
 const close = useAgUiUiStore((s) => s.closeInspector);
 if (!inspectedCallId) return null;
 return <InspectorPanel callId={inspectedCallId} onClose={close} />;
}

function InspectorPanel({ callId, onClose }: { callId: string; onClose: () => void }) {
 const messages = useAuiState((s) => s.thread.messages);
 const closeRef = useRef<HTMLButtonElement>(null);

 const step = useMemo(() => {
 for (const message of messages) {
 if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
 for (const part of message.content) {
 if (part.type === "data" && part.name === "step") {
 const data = part.data as Record<string, any> | undefined;
 if (data?.callId === callId) return data;
 }
 }
 }
 return null;
 }, [messages, callId]);

 // A step can disappear (thread switch / history reload): close, do not linger.
 useEffect(() => {
 if (!step) onClose();
 }, [step, onClose]);

 useEffect(() => {
 if (!step) return undefined;
 const onKey = (e: KeyboardEvent) => {
 if (e.key === "Escape") onClose();
 };
 window.addEventListener("keydown", onKey);
 closeRef.current?.focus();
 return () => window.removeEventListener("keydown", onKey);
 }, [step, onClose]);

 if (!step) return null;

 return (
 <>
 <div className="fixed inset-0 z-40 bg-overlay-soft" onClick={onClose} aria-hidden />
 <aside
 role="dialog"
 aria-modal="true"
 aria-label="Tool-call inspector"
 className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-border bg-card "
 >
 <header className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-3">
 <span className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
 {step.label ?? "Tool call"}
 </span>
 <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
 {String(step.kind ?? "tool")}
 </span>
 <button
 ref={closeRef}
 type="button"
 aria-label="Close inspector"
 onClick={onClose}
 className="flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition hover:bg-muted hover:text-foreground"
 >
 <XIcon className="size-4" />
 </button>
 </header>
 <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
 <Section title="Detail"><Text value={step.detail} /></Section>
 <Section title="Timeline">
 {Array.isArray(step.timeline) && step.timeline.length ? (
 <ul className="space-y-1">
 {step.timeline.map((entry: any, index: number) => (
 <li key={index} className="flex gap-2">
 <span className="mt-[7px] size-1 shrink-0 rounded-full bg-muted-foreground" aria-hidden />
 <span className="min-w-0"><Text value={entry?.text ?? entry} /></span>
 </li>
 ))}
 </ul>
 ) : (
 <em className="text-muted-foreground">—</em>
 )}
 </Section>
 <Section title="Arguments"><Pre value={step.rawArgs} /></Section>
 <Section title="Result"><Pre value={step.rawResult} /></Section>
 </div>
 </aside>
 </>
 );
}
