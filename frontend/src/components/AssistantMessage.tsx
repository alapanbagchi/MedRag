import { MessagePrimitive, TextMessagePartProvider, useAuiState } from "@assistant-ui/react";
import { useMemo } from "react";
import { ToolCalls, type ToolEntry } from "./ToolCalls";
import type { StepArgs } from "../lib/xdeep";
import { MarkdownText } from "./MarkdownText";
import { partArgs, partName } from "../lib/parts";

/**
 * Live in-flight tool calls for this message. Model thinking is not rendered
 * inline: the agentic-flow trigger below the chat opens the sub-agent sheet
 * where the thoughts live.
 */
function useTools(): ToolEntry[] {
  const content = useAuiState((s) => s.message.content);
  return useMemo(() => {
    const tools = new Map<string, ToolEntry>();
    content.forEach((p, index) => {
      if (partName(p) !== "step") return;
      const args = partArgs(p) as StepArgs | undefined;
      if (!args || args.kind === "thought") return;
      const key = String(args.callId ?? "step-" + index);
      tools.set(key, { key, step: args });
    });
    return [...tools.values()];
  }, [content]);
}

/** The citation guard's corrected answer, when the backend rewrote the draft. */
function useReplacedAnswer(): string {
  return useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return "";
    for (const part of content) {
      if (part.type !== "data") continue;
      if ((part as { name?: string }).name !== "answer_replace") continue;
      const text = (part.data as { text?: unknown } | undefined)?.text;
      if (typeof text === "string") return text;
    }
    return "";
  });
}

function AnswerBody() {
  const replaced = useReplacedAnswer();
  if (replaced) {
    return (
      <div className="anim-rise py-1">
        <div className="answer-body">
          <TextMessagePartProvider text={replaced}>
            <MarkdownText />
          </TextMessagePartProvider>
        </div>
      </div>
    );
  }
  return (
    <MessagePrimitive.GroupedParts groupBy={() => null}>
      {({ part }) => {
        if (part.type !== "text") return null;
        return (
          <div className="anim-rise py-1">
            <div className="answer-body">
              <MarkdownText />
            </div>
          </div>
        );
      }}
    </MessagePrimitive.GroupedParts>
  );
}

export function AssistantMessage() {
  const tools = useTools();
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const liveTools = useMemo(
    () => (isRunning ? tools.filter((entry) => !entry.step.done) : []),
    [tools, isRunning],
  );

  return (
    <MessagePrimitive.Root className="flex w-full">
      <div className="min-w-0 flex-1">
        <ToolCalls entries={liveTools} />
        <AnswerBody />
      </div>
    </MessagePrimitive.Root>
  );
}
