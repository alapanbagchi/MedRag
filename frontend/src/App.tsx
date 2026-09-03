import { MenuIcon } from "lucide-react";
import { useState } from "react";
import { RuntimeProvider } from "./components/RuntimeProvider";
import { Sidebar } from "./components/Sidebar";
import { ThreadView } from "./components/ThreadView";

export default function App() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <RuntimeProvider>
      <div className="flex h-dvh overflow-hidden bg-background text-foreground">
        <div className="hidden md:block">
          <Sidebar />
        </div>

        {mobileNavOpen ? (
          <div className="fixed inset-0 z-50 md:hidden">
            <div
              className="absolute inset-0 bg-black/40"
              onClick={() => setMobileNavOpen(false)}
            />
            <div className="absolute inset-y-0 left-0 w-[280px] shadow-2xl">
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
            <span className="text-sm font-semibold tracking-tight">MedRAG</span>
          </div>
          <ThreadView />
        </main>
      </div>
    </RuntimeProvider>
  );
}