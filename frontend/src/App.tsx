import { MenuIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { RuntimeProvider } from "./components/RuntimeProvider";
import { Sidebar } from "./components/Sidebar";
import { ThreadView } from "./components/ThreadView";
import { DEMO_QUESTION } from "./lib/demo";
import { startDemoRun } from "./lib/demo-run";
import { useChatStore } from "./lib/store";

export default function App() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const seededDemo = useRef(false);

  // First visit with no history: stream the demo into a background thread
  // so the landing (greeting + input + capability cards) stays in front —
  // the finished stream is one click away in the sidebar.
  useEffect(() => {
    if (seededDemo.current) return;
    seededDemo.current = true;
    const st = useChatStore.getState();
    const hasMessages = Object.values(st.messages).some((list) => list.length > 0);
    if (!hasMessages && st.currentThreadId) {
      const landingId = st.currentThreadId;
      const demoId = st.createThread();
      st.renameThread(demoId, DEMO_QUESTION);
      st.selectThread(landingId);
      const t = setTimeout(() => void startDemoRun(demoId, DEMO_QUESTION), 600);
      return () => clearTimeout(t);
    }
    return undefined;
  }, []);

  return (
    <RuntimeProvider>
      <div className="flex h-dvh overflow-hidden bg-white text-foreground">
        <div className="hidden md:block">
          <Sidebar />
        </div>

        {mobileNavOpen ? (
          <div className="fixed inset-0 z-50 md:hidden">
            <div
              className="absolute inset-0 bg-black/40"
              onClick={() => setMobileNavOpen(false)}
            />
            <div className="absolute inset-y-0 left-0 shadow-2xl">
              <Sidebar />
            </div>
          </div>
        ) : null}

        <main className="relative flex min-w-0 flex-1 flex-col">
          <div className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-3 md:hidden">
            <button
              type="button"
              onClick={() => setMobileNavOpen(true)}
              className="flex size-8 items-center justify-center rounded-lg text-muted-foreground transition hover:bg-muted hover:text-foreground"
              aria-label="Open navigation"
            >
              <MenuIcon className="size-5" />
            </button>
            <span className="font-greeting text-base font-semibold tracking-tight">MedRAG</span>
          </div>
          <ThreadView />
        </main>
      </div>
    </RuntimeProvider>
  );
}
