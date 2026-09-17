/** Full AG-UI application shell: sidebar + thread, one runtime per thread. */

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

  return (
    <AgUiRuntimeProvider key={currentThreadId ?? "none"}>
      <div className="flex h-dvh overflow-hidden bg-transparent text-foreground">
        <AgUiSidebar />
        {/* Centered message column; the flow sheet overlays the whole viewport. */}
        <div className="relative flex min-w-0 flex-1 justify-center">
          <main className="relative flex w-[900px] max-w-full min-w-0 flex-col">
            <AgUiThread />
          </main>
          <AgentPanel open={flowOpen} onClose={closeFlow} />
        </div>
      </div>
      <AgUiInspector />
    </AgUiRuntimeProvider>
  );
}
