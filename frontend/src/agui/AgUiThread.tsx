/**
 * AG-UI-native chat surface.
 *
 * Uses assistant-ui primitives over \`useAgUiRuntime\` and reuses the app's rich
 * message rendering (AssistantMessage → thinking stream, tool-call rows,
 * sub-agent cards, plan/sources).
 */

import { useEffect, useRef } from "react";
import {
  AuiIf,
  ComposerPrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { AgUiAssistantMessage } from "./AgUiAssistantMessage";
import { AgUiInterruptDock, usePendingQuestions } from "./AgUiInterruptDock";
import { useAgUiStore } from "./aguiStore";
import { PlanMorph, useAgUiPlan } from "../components/ComposerPlan";
import { AgentFlowTrigger } from "../components/assistant-ui/elements/subagent-stage";

const isRunning = (s: { thread: { isRunning: boolean } }) => s.thread.isRunning;
const isNotRunning = (s: { thread: { isRunning: boolean } }) => !s.thread.isRunning;

function UserBubble() {
  const text = useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return content;
    let out = "";
    for (const part of content) if (part.type === "text") out += part.text;
    return out;
  });
  return (
    <div className="mb-5 flex justify-end">
      <div className="max-w-[82%] whitespace-pre-wrap rounded-3xl border border-border bg-user-bubble px-4 py-2.5 text-[15px] leading-7 text-foreground shadow-[0_8px_30px_rgba(0,0,0,0.18)] backdrop-blur-xl">
        {text}
      </div>
    </div>
  );
}

/** Titles a thread from its first user message (sidebar label). */
function TitleSync() {
  const messages = useAuiState((s) => s.thread.messages);
  const currentThreadId = useAgUiStore((s) => s.currentThreadId);
  const titled = useRef(false);
  useEffect(() => {
    if (titled.current || !currentThreadId) return;
    const thread = useAgUiStore.getState().threads.find((t) => t.id === currentThreadId);
    if (thread?.title) {
      titled.current = true;
      return;
    }
    const firstUser = messages.find((m) => m.role === "user");
    if (!firstUser) return;
    const content = firstUser.content;
    let text = "";
    if (typeof content === "string") text = content;
    else for (const part of content) if (part.type === "text") text += part.text;
    text = text.trim();
    if (text) {
      useAgUiStore.getState().renameThread(currentThreadId, text.slice(0, 56));
      titled.current = true;
    }
  }, [messages, currentThreadId]);
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
      <div className="mt-9 w-full max-w-[720px]">
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
        <ComposerPrimitive.Root className="flex w-full items-end gap-2 px-4 py-2.5">
          <ComposerPrimitive.Input
            rows={1}
            autoFocus
            placeholder={active ? "Ask me anything" : "What do you want to know…"}
            className="max-h-64 min-h-[44px] flex-1 resize-none bg-transparent py-2.5 text-[15px] leading-6 outline-none placeholder:text-muted-foreground"
          />
          <AuiIf condition={isNotRunning}>
            <ComposerPrimitive.Send
              aria-label="Send"
              className="flex size-10 shrink-0 items-center justify-center rounded-full text-white shadow-[0_6px_18px_rgba(24,174,149,0.45)] transition hover:brightness-105 disabled:opacity-40"
              style={{ background: "linear-gradient(135deg, #4b8cf5 0%, #9d7bfb 100%)" }}
            >
              <ArrowUpIcon className="size-4" />
            </ComposerPrimitive.Send>
          </AuiIf>
          <AuiIf condition={isRunning}>
            <ComposerPrimitive.Cancel
              aria-label="Stop"
              className="flex size-10 shrink-0 items-center justify-center rounded-full border border-border bg-card text-foreground transition hover:bg-muted"
            >
              <SquareIcon className="size-4" />
            </ComposerPrimitive.Cancel>
          </AuiIf>
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

export function AgUiThread() {
  const isEmpty = useAuiState((s) => s.thread.isEmpty);
  return (
    <ThreadPrimitive.Root className="flex h-full min-h-0 flex-col bg-transparent">
      <TitleSync />
      <ThreadPrimitive.Viewport
        className={"flex min-h-0 flex-1 flex-col overflow-y-auto px-6 " + (isEmpty ? "" : "py-6")}
      >
        {isEmpty ? (
          <AgUiEmpty />
        ) : (
          <div className="mx-auto w-full max-w-[900px]">
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
          <div className="mx-auto w-full max-w-[900px]">
            <AgentFlowTrigger />
            <AgUiComposerArea />
          </div>
        </ThreadPrimitive.ViewportFooter>
      )}
    </ThreadPrimitive.Root>
  );
}
