import { ThreadPrimitive, useAuiState } from "@assistant-ui/react";
import { PlayIcon, ShareIcon } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AssistantMessage } from "./AssistantMessage";
import { Composer } from "./Composer";
import { EmptyState } from "./EmptyState";
import { PromptCards } from "./PromptCards";
import { UserMessage } from "./UserMessage";
import { DEMO_QUESTION } from "../lib/demo";
import { startDemoRun } from "../lib/demo-run";
import { DEMO_MEMORY_GRAPH } from "../lib/memory-graph";
import { useChatStore } from "../lib/store";
import { MemoryGraph } from "./MemoryGraph";
import { ToolInspector } from "./ToolInspector";

export type MainView = "chat" | "memory";

function TopBar({ view, onView }: { view: MainView; onView: (v: MainView) => void }) {
  const replayDemo = () => {
    const id = useChatStore.getState().currentThreadId;
    if (id) void startDemoRun(id, DEMO_QUESTION);
  };
  return (
    <div className="relative flex h-14 shrink-0 items-center gap-2 px-4">
      <div className="absolute left-1/2 flex -translate-x-1/2 items-center rounded-full border border-border bg-[#F7FAFB] p-0.5 text-[13px] font-medium" role="tablist" aria-label="Main view">
        {(["chat", "memory"] as MainView[]).map((v) => (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={view === v}
            onClick={() => onView(v)}
            className={`rounded-full px-4 py-1.5 transition ${
              view === v ? "bg-white text-[#0D0E1A] shadow-sm ring-1 ring-border" : "text-muted-foreground hover:text-[#0D0E1A]"
            }`}
          >
            {v === "chat" ? "Chat" : "Memory graph"}
          </button>
        ))}
      </div>
      <span className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={replayDemo}
          className="flex items-center gap-1.5 rounded-full border border-border bg-white px-3 py-1.5 text-xs font-medium text-[#3c4450] shadow-sm transition hover:border-[#1883AE]/50 hover:text-[#0D0E1A]"
        >
          <PlayIcon className="size-3.5 text-[#1883AE]" />
          Replay demo
        </button>
        <button
          type="button"
          aria-label="Share"
          className="flex size-8 items-center justify-center rounded-full text-[#8a97a3] transition hover:bg-muted hover:text-foreground"
        >
          <ShareIcon className="size-4" />
        </button>
        <span className="flex size-8 items-center justify-center rounded-full bg-[#0D0E1A] text-xs font-semibold text-white">
          M
        </span>
      </span>
    </div>
  );
}

type Stage = "empty" | "leaving" | "live";

/**
 * Owns the submit page transition: on the first message the greeting fades,
 * the (single, persistent) composer FLIP-flies from the center to the footer,
 * and only then does the thinking stream begin (demo-run holds its first
 * events back until the slide lands).
 */
function ChatShell() {
  const isEmpty = useAuiState((s) => s.thread.isEmpty);
  const [stage, setStage] = useState<Stage>(() => (isEmpty ? "empty" : "live"));
  const composerRef = useRef<HTMLDivElement>(null);
  const fromRect = useRef<DOMRect | null>(null);
  const reduceMotion = useMemo(
    () =>
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    [],
  );

  // A submit flips isEmpty while we're still showing the landing layout.
  useEffect(() => {
    if (!isEmpty && stage === "empty") {
      fromRect.current = composerRef.current?.getBoundingClientRect() ?? null;
      setStage(reduceMotion ? "live" : "leaving");
    } else if (isEmpty && stage !== "empty") {
      fromRect.current = null;
      setStage("empty");
    }
  }, [isEmpty, stage, reduceMotion]);

  // Let the greeting fade out before the layout switches.
  useEffect(() => {
    if (stage !== "leaving") return;
    const t = setTimeout(() => setStage("live"), 240);
    return () => clearTimeout(t);
  }, [stage]);

  // FLIP the persistent composer from the centered rect to the footer rect.
  useLayoutEffect(() => {
    if (stage !== "live" || !fromRect.current || !composerRef.current) return;
    const from = fromRect.current;
    fromRect.current = null;
    const el = composerRef.current;
    const to = el.getBoundingClientRect();
    const dx = from.left - to.left;
    const dy = from.top - to.top;
    if (Math.abs(dx) < 2 && Math.abs(dy) < 2) return;
    el.style.transition = "none";
    el.style.transform = `translate(${dx}px, ${dy}px)`;
    const frame = requestAnimationFrame(() => {
      el.style.transition = "transform 480ms cubic-bezier(0.22, 1, 0.36, 1)";
      el.style.transform = "translate(0px, 0px)";
    });
    const done = setTimeout(() => {
      el.style.transition = "";
      el.style.transform = "";
    }, 540);
    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(done);
    };
  }, [stage]);

  if (stage === "live") {
    return (
      <>
        <div className="anim-rise mx-auto flex w-full max-w-[900px] flex-1 flex-col gap-6 px-6 pt-2">
          <div className="flex justify-center pt-2">
            <span className="rounded-md bg-muted px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
              Today
            </span>
          </div>
          <ThreadPrimitive.Messages>
            {({ message }) => {
              if (message.composer.isEditing) return null;
              if (message.role === "user") return <UserMessage />;
              return <AssistantMessage />;
            }}
          </ThreadPrimitive.Messages>
        </div>

        <ThreadPrimitive.ViewportFooter className="sticky bottom-0 z-10 mt-auto w-full bg-gradient-to-t from-white via-white to-transparent px-6 pb-5 pt-8">
          <div ref={composerRef} className="mx-auto w-full max-w-[900px]">
            <Composer />
          </div>
        </ThreadPrimitive.ViewportFooter>
      </>
    );
  }

  return (
    <div className="flex flex-1 flex-col px-6 py-10">
      <div
        className={`relative -top-16 m-auto flex w-full max-w-[840px] flex-col items-center transition-all duration-200 lg:-top-24 ${
          stage === "leaving" ? "-translate-y-2 opacity-0" : "translate-y-0 opacity-100"
        }`}
      >
        <EmptyState />
        <div className="mt-6 w-full">
          <PromptCards />
        </div>
        <div ref={composerRef} className="mt-4 w-full">
          <Composer />
        </div>
      </div>
    </div>
  );
}

export function ThreadView() {
  const [view, setView] = useState<MainView>("chat");
  return (
    <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col bg-white">
      <TopBar view={view} onView={setView} />
      {view === "chat" ? (
        <ThreadPrimitive.Viewport className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          <ChatShell />
        </ThreadPrimitive.Viewport>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col px-6 pb-5">
          <MemoryGraph data={DEMO_MEMORY_GRAPH} />
        </div>
      )}
      <ToolInspector />
    </ThreadPrimitive.Root>
  );
}
