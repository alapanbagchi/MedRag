/**
 * The verified source list for the current answer, read off the message.
 *
 * The backend emits the turn's judge-kept passages, in citation order, as an
 * `answer_sources` AG-UI data part. Both the OpenUI card and the derived
 * markdown view read it, so their references can never disagree.
 */

import { useAuiState } from "@assistant-ui/react";
import type { CardSource } from "@openuidev/react-ui";

/**
 * The backend chat id for this answer, from the answer_format part.
 *
 * The AG-UI thread id IS the chat id, and it is NOT the app's sidebar thread
 * id (that one is a client-only value), so the client cannot derive it - it
 * has to be told. Needed to ask for this turn's long-form text answer.
 */
export function useAnswerChatId(): string {
  return readIdPart("chat_id");
}

/** The backend run (turn) id for this answer, from the answer_format part. */
export function useAnswerRunId(): string {
  return readIdPart("run_id");
}

/**
 * Read one string id off the newest answer_format data part.
 *
 * Returns a primitive on purpose: a fresh object from the selector would make
 * useAuiState re-render on every store change.
 */
function readIdPart(key: "chat_id" | "run_id"): string {
  return useAuiState((state) => {
    const content = state.message.content;
    if (typeof content === "string") return "";
    for (let i = content.length - 1; i >= 0; i -= 1) {
      const part = content[i] as { type?: string; name?: string; data?: unknown };
      if (part.type !== "data" || part.name !== "answer_format") continue;
      const value = (part.data as Record<string, unknown> | undefined)?.[key];
      return typeof value === "string" ? value : "";
    }
    return "";
  });
}

export function useAnswerSources(): CardSource[] | undefined {
  return useAuiState((state) => {
    const content = state.message.content;
    if (typeof content === "string") return undefined;
    for (let i = content.length - 1; i >= 0; i -= 1) {
      const part = content[i] as { type?: string; name?: string; data?: unknown };
      if (part.type !== "data" || part.name !== "answer_sources") continue;
      const sources = (part.data as { sources?: unknown } | undefined)?.sources;
      if (Array.isArray(sources)) return sources as CardSource[];
    }
    return undefined;
  });
}
