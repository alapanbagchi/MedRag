/**
 * AG-UI runtime provider: the useAgUiRuntime adapter over an HttpAgent pointed
 * at the backend's POST /v1/ag-ui.
 *
 * The agent carries the composer's options (model, web search, deep search) as
 * forwardedProps on every run, so the backend builds the run with them.
 */

import { useMemo, type ReactNode } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { HttpAgent } from "@ag-ui/client";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { createAgUiHistoryAdapter } from "./aguiHistory";
import { useAgUiStore, useAgUiUiStore } from "./aguiStore";
import { composerForwardedProps } from "./composerStore";

const AG_UI_URL = import.meta.env.VITE_AG_UI_URL ?? "/v1/ag-ui";

class MedRagHttpAgent extends HttpAgent {
  /** Stamp the composer's current options onto the run's forwardedProps. */
  protected override requestInit(input: any): RequestInit {
    const body = {
      ...input,
      forwardedProps: {
        ...((input?.forwardedProps as Record<string, unknown> | undefined) ?? {}),
        ...composerForwardedProps(),
      },
    };
    return {
      method: "POST",
      headers: {
        ...this.headers,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal: this.abortController.signal,
    };
  }
}

export function AgUiRuntimeProvider({ children }: { children: ReactNode }) {
  const threadId = useAgUiStore((s) => s.currentThreadId) ?? "";
  const agent = useMemo(() => new MedRagHttpAgent({ url: AG_UI_URL }), []);
  // History is captured when the runtime core is constructed, so this is
  // stable for the lifetime of the provider (remounted per thread).
  const history = useMemo(() => createAgUiHistoryAdapter(threadId), [threadId]);
  const runtime = useAgUiRuntime({
    agent,
    showThinking: true,
    adapters: { history },
    onError: (error) => {
      console.error("[ag-ui]", error);
      // Surface the failure in the UI: a run that dies (bad model id, missing
      // key, gateway 5xx) otherwise looks like "nothing happened".
      useAgUiUiStore
        .getState()
        .setRunError(error instanceof Error ? error.message : String(error));
    },
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
  );
}
