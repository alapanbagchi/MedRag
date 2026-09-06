import { MessagePrimitive, useAuiState } from "@assistant-ui/react";
import { ChevronDownIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ThinkingIndicator } from "./assistant-ui/elements/thinking-indicator";
import { StreamingText } from "./assistant-ui/elements/streaming-text";
import { TypingIndicator } from "./assistant-ui/elements/typing-indicator";
import { ToolCalls, type ToolEntry } from "./ToolCalls";
import { STAGE_LABELS } from "../lib/labels";
import type { StepArgs } from "../lib/xdeep";
import { MarkdownText } from "./MarkdownText";

/**
 * Subscribe to the message content by reference (stable across renders) and
 * derive everything with useMemo. Selectors must NOT build fresh arrays —
 * useSyncExternalStore treats each new reference as a change and loops.
 *
 * Split: `thought` steps feed the streaming thought text; every other step
 * is a tool trace entry for the ToolCall rows + timeline.
 */
function useThinking(): {
  lines: string[];
  tools: ToolEntry[];
  stage: string;
} {
  const content = useAuiState((s) => s.message.content);
  return useMemo(() => {
    const lines: string[] = [];
    const tools: ToolEntry[] = [];
    let stage = "";
    content.forEach((p, index) => {
      if (p?.type !== "tool-call") return;
      if (p.toolName === "step") {
        const args = (p as unknown as { args?: StepArgs }).args;
        if (!args) return;
        if (args.kind === "thought") {
          const text = (args.detail || args.label || "").trim();
          if (text) lines.push(text);
        } else {
          const key =
            (p as { toolCallId?: string }).toolCallId ??
            args.callId ??
            `step-${index}`;
          tools.push({ key, step: args });
        }
      } else if (p.toolName === "status") {
        stage = ((p as { args?: { stage?: string } }).args?.stage ?? "") as string;
      }
    });
    return { lines, tools, stage };
  }, [content]);
}

function useElapsed(running: boolean): string {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return undefined;
    const start = Date.now();
    setSeconds(0);
    const id = setInterval(
      () => setSeconds(Math.round((Date.now() - start) / 1000)),
      1000,
    );
    return () => clearInterval(id);
  }, [running]);
  return `${seconds}s`;
}

/**
 * Assistant message: thinking indicator + streaming thought lines directly
 * on the background below the question, then the final answer.
 * All tool-call cards, panels, pills, and action bars stay removed —
 * every non-text part renders nothing.
 */
export function AssistantMessage() {
  const { lines, tools, stage } = useThinking();
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const [thoughtsOpen, setThoughtsOpen] = useState(true);
  const elapsed = useElapsed(isRunning);

  const segments = useMemo(() => lines.map((text) => ({ text })), [lines]);
  const wordCount = useMemo(
    () => segments.reduce((n, s) => n + s.text.split(" ").length, 0),
    [segments],
  );
  const stageLabel = (STAGE_LABELS[stage] ?? stage).trim() || "Thinking";

  return (
    <MessagePrimitive.Root className="flex w-full">
      <div className="min-w-0 flex-1">
        {isRunning ? (
          <div className="px-1 py-1">
            <ThinkingIndicator label={stageLabel} elapsed={elapsed} />
          </div>
        ) : null}

        {lines.length === 0 && isRunning ? (
          <div className="px-1 py-1">
            <TypingIndicator variant="bare" />
          </div>
        ) : null}

        {segments.length > 0 ? (
          <div className="px-1 py-1">
            <button
              type="button"
              onClick={() => setThoughtsOpen((v) => !v)}
              aria-expanded={thoughtsOpen}
              className="flex items-center gap-2 rounded-md py-1 text-[13.5px] text-foreground/55 transition-colors outline-none hover:text-foreground/90"
            >
              <ChevronDownIcon
                className={`size-3.5 shrink-0 opacity-60 transition-transform duration-200 ${thoughtsOpen ? "" : "-rotate-90"}`}
              />
              Thinking
            </button>
            {thoughtsOpen ? (
              <div className="ms-2 mt-1 border-l-2 border-border/70 ps-4 opacity-60">
                <StreamingText
                  segments={segments}
                  count={wordCount}
                  streaming={isRunning}
                  className="max-w-none"
                />
              </div>
            ) : null}
          </div>
        ) : null}

        <ToolCalls entries={tools} isRunning={isRunning} stageLabel={stageLabel} />

        <MessagePrimitive.GroupedParts groupBy={() => null}>
          {({ part }) => {
            if (part.type === "text") {
              return (
                <div className="anim-rise py-1">
                  <div className="answer-body">
                    <MarkdownText />
                  </div>
                </div>
              );
            }
            return null;
          }}
        </MessagePrimitive.GroupedParts>
      </div>
    </MessagePrimitive.Root>
  );
}
