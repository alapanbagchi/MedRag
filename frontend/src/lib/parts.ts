import type { ThreadMessageLike } from "@assistant-ui/react";

/** A message content part (as stored in the runtime message). */
export type MessagePartLike = Extract<
  ThreadMessageLike["content"],
  readonly unknown[]
>[number];

export type Content = MessagePartLike[];

/** Part name for an AG-UI `data` part. */
export function partName(part: unknown): string {
  const p = part as { type?: string; name?: string } | undefined;
  return p?.type === "data" ? p.name ?? "" : "";
}

/** Payload of an AG-UI `data` part. */
export function partArgs(part: unknown): unknown {
  const p = part as { type?: string; data?: unknown } | undefined;
  return p?.type === "data" ? p.data : undefined;
}

export type TaskState = 'started' | 'done' | 'failed';

/** Every leg's latest task state by id (planner S1 included). */
export function taskStatesOf(content: Content): Map<
  string,
  { state: TaskState; model?: string; question?: string; startedTs?: string; finishedTs?: string }
> {
  const states = new Map<
    string,
    { state: TaskState; model?: string; question?: string; startedTs?: string; finishedTs?: string }
  >();
  for (const p of content) {
    const name = partName(p);
    if (name !== 'task') continue;
    const args = partArgs(p) as
      | {
          id?: string;
          state?: TaskState;
          model?: string;
          question?: string;
          startedTs?: string;
          finishedTs?: string;
        }
      | undefined;
    if (!args || typeof args.id !== 'string') continue;
    const state = args.state;
    if (state !== 'started' && state !== 'done' && state !== 'failed') continue;
    const prev = states.get(args.id);
    states.set(args.id, {
      state,
      model: args.model ?? prev?.model,
      question: args.question ?? prev?.question,
      startedTs: args.startedTs ?? prev?.startedTs,
      finishedTs: args.finishedTs ?? prev?.finishedTs,
    });
  }
  return states;
}
