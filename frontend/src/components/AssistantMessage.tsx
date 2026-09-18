import { MessagePrimitive, useAuiState } from "@assistant-ui/react";
import { memo } from "react";
import { OpenUIAnswer } from "../openui/OpenUIAnswer";
import { MessageAgentFlowTrigger } from "./assistant-ui/elements/subagent-stage";

const EMPTY_PARTS: readonly unknown[] = [];

/**
 * The one-line "Thinking · Click to see the agentic flow" row, scoped to this
 * answer's own content so every message opens the thoughts it produced.
 * Tool-call rows are gone from the main screen: the sheet owns them.
 */
function MessageAgentFlow() {
  const content = useAuiState((s) =>
    typeof s.message.content === "string" ? EMPTY_PARTS : s.message.content,
  );
  const messageId = useAuiState((s) => s.message.id);
  const running = useAuiState((s) => s.thread.isRunning && s.message.isLast);
  return (
    <MessageAgentFlowTrigger
      content={content as readonly unknown[]}
      messageId={messageId}
      isRunning={running}
    />
  );
}

/** The citation guard corrected answer, when the backend rewrote the draft. */
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

/**
 * The backend signal for which answer contract this message carries.
 * The LAST one wins: the synthesizer announces openui before streaming so
 * the card renders live, then re-announces markdown if the program failed
 * validation and the answer falls back.
 */
function useAnswerFormat(): string {
  return useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return "";
    for (let i = content.length - 1; i >= 0; i -= 1) {
      const part = content[i];
      if (part.type !== "data") continue;
      if ((part as { name?: string }).name !== "answer_format") continue;
      const value = (part.data as { format?: unknown } | undefined)?.format;
      if (typeof value === "string") return value;
    }
    return "";
  });
}

/**
 * The assistant answer text.
 *
 * Read directly instead of walking the message parts: the stream appends one
 * data part per thinking/step event, and assistant-ui Parts/GroupedParts mounts
 * a provider per part on every render. For a long run that is thousands of
 * elements per token. Only text parts are the answer, so fold them in one pass.
 * Memoized: it re-renders on its own subscriptions, not on every parent render.
 */
const AnswerBody = memo(function AnswerBody() {
  const replaced = useReplacedAnswer();
  const format = useAnswerFormat();
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const text = useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return content;
    let out = "";
    for (const part of content) if (part.type === "text") out += part.text;
    return out;
  });

  const value = replaced || text;
  if (!value) return null;
  // Every answer gets the Gen UI / Text tabs, whatever mode produced it: the
  // OpenUI renderer when the backend wrote a program, the markdown otherwise.
  return (
    <div className="anim-rise py-1">
      <OpenUIAnswer text={value} isStreaming={isRunning} format={format} />
    </div>
  );
});

export const AssistantMessage = memo(function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex w-full">
      <div className="min-w-0 flex-1">
        <MessageAgentFlow />
        <AnswerBody />
      </div>
    </MessagePrimitive.Root>
  );
});
