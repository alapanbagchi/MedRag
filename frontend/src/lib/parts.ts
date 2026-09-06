import type { ThreadMessageLike } from "@assistant-ui/react";
import type { Source } from "./types";
import type { StepArgs } from "./xdeep";
import { uid } from "./store";

/** A message content part (as stored in our external store). */
export type MessagePartLike = Extract<
  ThreadMessageLike["content"],
  readonly unknown[]
>[number];

export type Content = MessagePartLike[];

export function contentOf(message: ThreadMessageLike): Content {
  return Array.isArray(message.content) ? (message.content as Content) : [];
}

export function toolPart(
  toolName: string,
  args: Record<string, unknown>,
  result?: unknown,
): MessagePartLike {
  return {
    type: "tool-call",
    toolCallId: uid(),
    toolName,
    args,
    ...(result !== undefined ? { result } : {}),
  } as MessagePartLike;
}

export function appendPart(content: Content, part: MessagePartLike): Content {
  return [...content, part];
}

/** Replace the part matched by `predicate` in place, or append `make()`'s result. */
export function upsertPart(
  content: Content,
  predicate: (part: MessagePartLike) => boolean,
  make: () => MessagePartLike,
): Content {
  const index = content.findIndex(predicate);
  if (index >= 0) {
    const next = [...content];
    next[index] = make();
    return next;
  }
  return [...content, make()];
}

/** Streamed answer text: append to the last text part (or create one). */
export function upsertText(content: Content, text: string): Content {
  let index = -1;
  for (let i = content.length - 1; i >= 0; i--) {
    if (content[i]?.type === "text") {
      index = i;
      break;
    }
  }
  if (index >= 0) {
    const next = [...content];
    const existing = next[index] as { type: "text"; text: string };
    next[index] = { ...existing, text: existing.text + text };
    return next;
  }
  return [...content, { type: "text", text }];
}

/** Live pipeline stage chip — only the newest `status` part survives. */
export function upsertStatus(content: Content, stage: string, count: number): Content {
  return upsertPart(
    content,
    (p) => p.type === "tool-call" && p.toolName === "status",
    () => toolPart("status", { stage, count }),
  );
}

/** One pipeline/LLM observation, surfaced as a "thought". */
export function pushThought(
  content: Content,
  event: string,
  fields?: Record<string, unknown>,
): Content {
  const data = { event, fields: fields ?? {} };
  return appendPart(content, toolPart("thought", data, data));
}

/** Streaming model thinking: append `delta` to the single open thought step
 * (created on first delta), so token-by-token reasoning renders as one row. */
export function upsertThought(content: Content, delta: string, done: boolean): Content {
  const MAX_THOUGHT = 4000;
  type ThoughtPart = { type: string; toolName: string; toolCallId: string; args: StepArgs };
  let index = -1;
  for (let i = content.length - 1; i >= 0; i--) {
    const p = content[i];
    if (p?.type === "tool-call" && (p as { toolName?: string }).toolName === "step") {
      const args = (p as unknown as ThoughtPart).args;
      if (args?.kind === "thought" && !args.done) {
        index = i;
        break;
      }
    }
  }
  if (index < 0) {
    if (!delta) return content;
    return appendPart(
      content,
      toolPart("step", { kind: "thought", label: "Thinking", detail: delta.slice(0, MAX_THOUGHT), done }),
    );
  }
  const next = [...content];
  const part = next[index] as unknown as ThoughtPart;
  const detail = `${part.args.detail ?? ""}${delta}`.slice(0, MAX_THOUGHT);
  next[index] = { ...part, args: { ...part.args, detail, done } } as unknown as MessagePartLike;
  return next;
}

/** A research step (tool-card row) — see lib/xdeep.ts for kinds. */
export function pushStep(content: Content, step: StepArgs, cap = 260): Content {
  const count = content.filter((p) => p.type === "tool-call" && p.toolName === "step").length;
  if (count >= cap && !step.done) return content; // avoid unbounded logs
  return appendPart(content, toolPart("step", { ...step }));
}

/** Update the LAST matching open step in place (e.g. web_search_done → query). */
export function updateStep(
  content: Content,
  predicate: (step: StepArgs) => boolean,
  patch: Partial<StepArgs>,
): Content {
  let index = -1;
  for (let i = content.length - 1; i >= 0; i--) {
    const p = content[i];
    if (p?.type === "tool-call" && p.toolName === "step") {
      const args = (p as { args?: StepArgs }).args as StepArgs;
      if (args && predicate(args)) {
        index = i;
        break;
      }
    }
  }
  if (index < 0) return content;
  const next = [...content];
  const part = next[index] as { type: string; args: Record<string, unknown>; toolName: string; toolCallId: string } & Record<string, unknown>;
  next[index] = { ...part, args: { ...part.args, ...patch } } as MessagePartLike;
  return next;
}

/** The master research plan (planner card). Prepend so the plan renders
 * above the thinking panel without splitting its step groups. */
export function pushPlan(content: Content, fields?: Record<string, unknown>): Content {
  const data = { fields: fields ?? {} };
  const part = toolPart("plan", data, data);
  // replace an existing plan part in place (single plan per run)
  const index = content.findIndex((p) => p.type === "tool-call" && p.toolName === "plan");
  if (index >= 0) {
    const next = [...content];
    next[index] = part;
    return next;
  }
  return [part, ...content];
}

/** Memory attachment / commit event. */
export function pushMemory(
  content: Content,
  kind: string,
  fields: Record<string, unknown>,
): Content {
  const data = { kind, ...fields };
  return appendPart(content, toolPart("memory", data, data));
}

/** Verified sources — merged and de-duplicated by document id. */
export function upsertSources(content: Content, sources: Source[]): Content {
  return upsertPart(
    content,
    (p) => p.type === "tool-call" && p.toolName === "sources",
    () => {
      const seen = new Map<string, Source>();
      const walk = (list: Source[]) => {
        for (const s of list) {
          const key = s.id ?? s.pmcid ?? s.title ?? JSON.stringify(s);
          seen.set(key, s);
        }
      };
      // the standalone sources part keeps its result as the source list
      walk(sources);
      const merged = [...seen.values()];
      return toolPart("sources", { count: merged.length }, merged);
    },
  );
}