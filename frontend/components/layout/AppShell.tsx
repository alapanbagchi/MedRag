// ── App shell: sidebar + content column ─────────────────────────────
"use client";

import { Sidebar } from "@/components/layout/Sidebar";
import { useApp } from "@/lib/store";
import { cn } from "@/lib/utils";

export function AppShell({ children }: { children: React.ReactNode }) {
  const { sidebarCollapsed } = useApp();
  return (
    <div className="min-h-dvh">
      <Sidebar />
      <main
        className={cn(
          "min-h-dvh transition-[padding] duration-300 ease-out",
          sidebarCollapsed ? "lg:pl-16" : "lg:pl-[264px]"
        )}
      >
        {children}
      </main>
    </div>
  );
}
