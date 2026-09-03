import type { BackendEvent, ResearchRequest } from "./types";

/**
 * Streams one research run from `POST /v1/chat/stream` (NDJSON).
 * Parses line-by-line and forwards parsed events to `onEvent`.
 */
export async function streamResearch(
  request: ResearchRequest,
  signal: AbortSignal,
  onEvent: (event: BackendEvent) => void,
): Promise<void> {
  const response = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: request.question,
      conversation_id: request.conversationId ?? "",
      history: [],
      engine: request.engine ?? "",
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Backend responded ${response.status}${detail ? `: ${detail.slice(0, 200)}` : ""}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      try {
        onEvent(JSON.parse(line) as BackendEvent);
      } catch {
        // skip malformed lines instead of killing the stream
      }
    }
  }
}