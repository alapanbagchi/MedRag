/**
 * AG-UI runtime provider: the `useAgUiRuntime` adapter over an `HttpAgent`
 * pointed at the backend's `POST /v1/ag-ui`.
 */

import { useMemo, type ReactNode } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { HttpAgent } from "@ag-ui/client";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { createAgUiHistoryAdapter } from "./aguiHistory";
import { useAgUiStore } from "./aguiStore";

const AG_UI_URL = import.meta.env.VITE_AG_UI_URL ?? "/v1/ag-ui";

export function AgUiRuntimeProvider({ children }: { children: ReactNode }) {
  const threadId = useAgUiStore((s) => s.currentThreadId) ?? "";
  const agent = useMemo(() => new HttpAgent({ url: AG_UI_URL }), []);
  // History is captured when the runtime core is constructed, so this is
  // stable for the lifetime of the provider (remounted per thread).
  const history = useMemo(() => createAgUiHistoryAdapter(threadId), [threadId]);
  const runtime = useAgUiRuntime({
    agent,
    showThinking: true,
    adapters: { history },
    onError: (error) => console.error("[ag-ui]", error),
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
  );
}
