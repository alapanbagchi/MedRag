/**
 * AG-UI-native chat surface.
 *
 * Uses assistant-ui primitives over \`useAgUiRuntime\` and reuses the app's rich
 * message rendering (AssistantMessage → thinking stream, tool-call rows,
 * sub-agent cards, plan/sources).
 */

import { memo, useEffect, useRef } from "react";
import {
 AuiIf,
 ComposerPrimitive,
 ThreadPrimitive,
 useAuiState,
} from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { AgUiAssistantMessage } from "./AgUiAssistantMessage";
import { AgUiInterruptDock, usePendingQuestions } from "./AgUiInterruptDock";
import { useAgUiStore, useAgUiUiStore } from "./aguiStore";
import { PlanMorph, useAgUiPlan } from "../components/ComposerPlan";
import { ComposerControls } from "./ComposerControls";

const isRunning = (s: { thread: { isRunning: boolean } }) => s.thread.isRunning;
const isNotRunning = (s: { thread: { isRunning: boolean } }) => !s.thread.isRunning;

// Memoized: the thread re-renders on every streamed token, but a user bubble
// only changes when its own text does.
const UserBubble = memo(function UserBubble() {
 const text = useAuiState((s) => {
 const content = s.message.content;
 if (typeof content === "string") return content;
 let out = "";
 for (const part of content) if (part.type === "text") out += part.text;
 return out;
 });
 return (
 <div className="mb-5 flex justify-end">
 <div className="max-w-[82%] whitespace-pre-wrap rounded-3xl border border-border bg-user-bubble px-4 py-2.5 text-[15px] leading-7 text-foreground">
 {text}
 </div>
 </div>
 );
});

/** Titles a thread from its first user message (sidebar label). */
function TitleSync() {
 // Select the primitive, not the message array: the array changes on every
 // streamed token, so subscribing to it re-rendered this component (and ran
 // the effect) hundreds of times per turn for no reason.
 const firstUserText = useAuiState((s) => {
 for (const message of s.thread.messages) {
 if (message.role !== "user") continue;
 const content = message.content;
 if (typeof content === "string") return content;
 let out = "";
 for (const part of content) if (part.type === "text") out += part.text;
 return out;
 }
 return "";
 });
 const currentThreadId = useAgUiStore((s) => s.currentThreadId);
 const titled = useRef(false);
 useEffect(() => {
 if (titled.current || !currentThreadId) return;
 const thread = useAgUiStore.getState().threads.find((t) => t.id === currentThreadId);
 if (thread?.title) {
 titled.current = true;
 return;
 }
 const text = firstUserText.trim();
 if (text) {
 useAgUiStore.getState().renameThread(currentThreadId, text.slice(0, 56));
 titled.current = true;
 }
 }, [firstUserText, currentThreadId]);
 return null;
}

/** Centered landing shown before the first message: greeting + composer. */
function AgUiEmpty() {
 return (
 <div className="flex flex-1 flex-col items-center justify-center px-4 py-10">
 <AgUiInterruptDock />
 <h1 className="text-center text-[clamp(28px,3.4vw,40px)] font-normal tracking-tight text-foreground/90">
 Let&rsquo;s jump in
 </h1>
 <div className="mt-9 w-full max-w-[880px]">
 <AgUiComposer />
 </div>
 </div>
 );
}

/** Composer with the shared plan sheet attached above it on the AG-UI surface. */
function AgUiComposer() {
 const plan = useAgUiPlan();
 return (
 <PlanMorph plan={plan}>
 {(active) => (
 <ComposerPrimitive.Root className="flex w-full flex-col gap-1.5 rounded-2xl bg-popover px-5 pt-4 pb-3">
 <ComposerPrimitive.Input
 rows={1}
 autoFocus
 placeholder={active ? "Ask me anything" : "What do you want to know…"}
 className="max-h-64 min-h-[56px] w-full resize-none bg-transparent text-[16px] leading-7 outline-none placeholder:text-muted-foreground"
 />
 <div className="flex items-center gap-1.5">
 <ComposerControls />
 <div className="ml-auto flex items-center gap-1.5">
 <AuiIf condition={isNotRunning}>
 <ComposerPrimitive.Send
 aria-label="Send"
 className="flex size-9 shrink-0 items-center justify-center rounded-full bg-brand-gradient text-brand-foreground transition hover:brightness-105 disabled:opacity-40"
 >
 <ArrowUpIcon className="size-4" />
 </ComposerPrimitive.Send>
 </AuiIf>
 <AuiIf condition={isRunning}>
 <ComposerPrimitive.Cancel
 aria-label="Stop"
 className="flex size-9 shrink-0 items-center justify-center rounded-full border border-border bg-card text-foreground transition hover:bg-muted"
 >
 <SquareIcon className="size-4" />
 </ComposerPrimitive.Cancel>
 </AuiIf>
 </div>
 </div>
 </ComposerPrimitive.Root>
 )}
 </PlanMorph>
 );
}

/**
 * The footer input slot. While the agent is waiting on a clarification it
 * swaps the composer for the question form (inert on the hidden panel keeps
 * keyboard/AT users out of invisible controls).
 */
function AgUiComposerArea() {
 const hasQuestion = usePendingQuestions().length > 0;
 const ease = "ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none";
 return (
 <>
 <div
 inert={hasQuestion}
 aria-hidden={hasQuestion}
 className={"overflow-hidden transition-all duration-300 " + ease + " " + (
 hasQuestion
 ? "pointer-events-none max-h-0 -translate-y-2 opacity-0"
 : "max-h-[420px] translate-y-0 opacity-100"
 )}
 >
 <AgUiComposer />
 </div>
 <div
 inert={!hasQuestion}
 aria-hidden={!hasQuestion}
 className={"overflow-hidden transition-all duration-300 " + ease + " " + (
 hasQuestion
 ? "max-h-[80vh] translate-y-0 opacity-100"
 : "pointer-events-none max-h-0 translate-y-3 opacity-0"
 )}
 >
 <AgUiInterruptDock />
 </div>
 </>
 );
}

/** Visible run failure: a request that dies must not look like a no-op. */
function RunErrorBanner() {
 const runError = useAgUiUiStore((s) => s.runError);
 const clearRunError = useAgUiUiStore((s) => s.clearRunError);
 if (!runError) return null;
 return (
 <div
 role="alert"
 className="border-danger/30 bg-danger/10 text-danger mb-2 flex items-start gap-2 rounded-xl border px-3 py-2 text-[13px] leading-5"
 >
 <span className="min-w-0 flex-1 break-words [overflow-wrap:anywhere]">
 <span className="font-semibold">Run failed.</span> {runError}
 </span>
 <button
 type="button"
 aria-label="Dismiss error"
 onClick={clearRunError}
 className="text-danger/80 hover:bg-danger/10 shrink-0 rounded-md px-1.5 py-0.5 text-[12px] font-medium transition-colors"
 >
 Dismiss
 </button>
 </div>
 );
}

export function AgUiThread() {
 const isEmpty = useAuiState((s) => s.thread.isEmpty);
 const runActive = useAuiState((s) => s.thread.isRunning);
 const clearRunError = useAgUiUiStore((s) => s.clearRunError);
 // A new run clears the previous failure notice.
 useEffect(() => {
 if (runActive) clearRunError();
 }, [runActive, clearRunError]);
 return (
 <ThreadPrimitive.Root className="flex h-full min-h-0 flex-col bg-transparent">
 <TitleSync />
 <ThreadPrimitive.Viewport
 className={"flex min-h-0 flex-1 flex-col overflow-y-auto px-6 " + (isEmpty ? "" : "py-6")}
 >
 {isEmpty ? (
 <AgUiEmpty />
 ) : (
 <div className="mx-auto w-full max-w-[1200px]">
 <ThreadPrimitive.Messages>
 {({ message }) =>
 message.role === "user" ? <UserBubble /> : <AgUiAssistantMessage />
 }
 </ThreadPrimitive.Messages>
 </div>
 )}
 </ThreadPrimitive.Viewport>

 {isEmpty ? null : (
 <ThreadPrimitive.ViewportFooter
 data-slot="thread-footer"
 className="relative z-40 bg-transparent px-6 pt-3 pb-4"
 >
 <div className="mx-auto w-full max-w-[1200px]">
 <RunErrorBanner />
 <AgUiComposerArea />
 </div>
 </ThreadPrimitive.ViewportFooter>
 )}
 </ThreadPrimitive.Root>
 );
}
