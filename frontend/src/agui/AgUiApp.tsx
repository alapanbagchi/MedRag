/** Full AG-UI application shell: sidebar + thread, one runtime per thread. */

import { useEffect } from "react";
import { AgentPanel } from "../components/assistant-ui/elements/subagent-stage";
import { AgUiInspector } from "./AgUiInspector";
import { AgUiRuntimeProvider } from "./AgUiRuntimeProvider";
import { AgUiSidebar } from "./AgUiSidebar";
import { AgUiThread } from "./AgUiThread";
import { useAgUiStore, useAgUiUiStore } from "./aguiStore";

export function AgUiApp() {
  const currentThreadId = useAgUiStore((s) => s.currentThreadId);
  const flowOpen = useAgUiUiStore((s) => s.flowOpen);
  const closeFlow = useAgUiUiStore((s) => s.closeFlow);

  // The agent sheet is per chat: switching threads closes it so it never
  // carries one chat's thoughts over another.
  useEffect(() => {
    closeFlow();
  }, [currentThreadId, closeFlow]);

  return (
    <AgUiRuntimeProvider key={currentThreadId ?? "none"}>
      <div className="flex h-dvh overflow-hidden bg-canvas text-foreground">
        <AgUiSidebar />
        {/* Centered message column; the flow sheet overlays the whole viewport.
            Explicit backgrounds so the chat body is opaque on its own, not only
            because the root happens to paint behind it. */}
        <div className="bg-background relative flex min-w-0 flex-1 justify-center">
          <main className="relative flex w-full max-w-[1200px] min-w-0 flex-col bg-background">
            <AgUiThread />
          </main>
          <AgentPanel open={flowOpen} onClose={closeFlow} />
        </div>
      </div>
      <AgUiInspector />
    </AgUiRuntimeProvider>
  );
}
