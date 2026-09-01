// ── Home / landing: instrument hero ──────────────────────────────────
"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Menu, Moon, Sun } from "lucide-react";
import { useApp } from "@/lib/store";
import { useHotkey, useMediaQuery, useMounted } from "@/lib/hooks";
import { engineMode } from "@/lib/rag-client";
import { ENGINE, SUGGESTED_QUERIES } from "@/lib/mock-data";
import { cn, fmtNum, titleFromText } from "@/lib/utils";
import { Logo } from "@/components/Logo";
import { CapsLabel, Kbd, Led } from "@/components/ui/primitives";

function TelemetryColumn() {
  const rows: Array<[string, string]> = [
    ["ENGINE", engineMode()],
    ["CORPUS", ENGINE.corpus],
    ["DOCS INDEXED", fmtNum(ENGINE.docsIndexed)],
    ["PASSAGES", fmtNum(ENGINE.passages)],
    ["MODEL", ENGINE.model],
    ["LATENCY P95", ENGINE.latencyP95],
  ];
  return (
    <aside
      aria-label="System telemetry"
      className="fixed right-8 top-1/2 hidden w-64 -translate-y-1/2 xl:block"
    >
      <div className="panel corner-ticks">
        <div className="flex items-center justify-between border-b border-line px-3 py-2">
          <CapsLabel>System telemetry</CapsLabel>
          <Led state="ok" pulse />
        </div>
        <dl>
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between gap-3 border-b border-line px-3 py-2 last:border-b-0">
              <dt className="mono text-[10px] uppercase tracking-[0.12em] text-ink3">{k}</dt>
              <dd className="mono text-[11px] text-ink tnum">{v}</dd>
            </div>
          ))}
        </dl>
        <div className="border-t border-line px-3 py-2">
          <p className="mono text-[10px] leading-relaxed text-ink3">
            &lt; medpat/ready &gt;
            <span className="cursor-blink ml-1" aria-hidden />
          </p>
        </div>
      </div>
    </aside>
  );
}

export function HomeView() {
  const router = useRouter();
  const { createConversation, setMobileNavOpen, theme, toggleTheme } = useApp();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [value, setValue] = useState("");
  const mounted = useMounted();
  const busy = false;

  useHotkey("mod+k", () => inputRef.current?.focus(), true);
  useHotkey("mod+n", () => {
    router.push(`/chat/${createConversation()}`);
  }, true);

  const run = (q: string) => {
    const question = q.trim();
    if (!question) return;
    const id = createConversation(titleFromText(question));
    router.push(`/chat/${id}?q=${encodeURIComponent(question)}`);
  };

  const onSubmit = () => run(value);
  const onSuggestion = (q: string) => {
    setValue(q);
    run(q);
  };

  return (
    <div className="grid-surface noise relative flex min-h-dvh flex-col overflow-hidden">
      {/* top bar */}
      <header className="relative z-10 flex h-14 flex-none items-center gap-3 border-b border-line px-4 sm:px-6">
        <button
          type="button"
          onClick={() => setMobileNavOpen(true)}
          aria-label="Open navigation"
          className="icon-btn lg:hidden"
        >
          <Menu size={17} />
        </button>
        <div className="flex items-center gap-2.5">
          <Logo size={22} className="text-ink" accent="var(--accent)" />
          <span className="text-[13px] font-extrabold tracking-[0.2em]">MEDPAT</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="mono mr-1 inline-flex items-center gap-1.5 border border-line px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-ink2">
            <Led state={engineMode() === "MOCK" ? "accent" : "ok"} pulse className="!h-1.5 !w-1.5" />
            {engineMode()} ENGINE
          </span>
          <button type="button" onClick={toggleTheme} aria-label="Toggle theme" className="icon-btn">
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          <div className="hidden h-7 w-7 items-center justify-center border border-line-strong text-[10px] font-bold sm:flex" aria-hidden>
            DR
          </div>
        </div>
      </header>

      {/* hero */}
      <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-6 pb-16 pt-10 sm:px-10">
        <div className="w-full max-w-3xl">
          <h1 className="anim-rise text-[clamp(3rem,9vw,6.5rem)] font-black uppercase leading-[0.92] tracking-[-0.03em]">
            MEDPAT
          </h1>
          <p className="anim-rise mt-4 max-w-xl text-[15px] leading-relaxed text-ink2" style={{ animationDelay: "60ms" }}>
            Evidence-aware medical research, <span className="font-semibold text-ink">accelerated</span>.
            Ask the literature a question — watch the pipeline work, read a cited
            answer, inspect every source.
          </p>

          {/* search console */}
          <div className="anim-rise mt-8" style={{ animationDelay: "120ms" }}>
            <div className="input-frame">
              <div className="flex items-end gap-2 p-2 sm:p-3">
                <textarea
                  ref={inputRef}
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      onSubmit();
                    }
                  }}
                  rows={2}
                  placeholder="Ask a medical research question…"
                  aria-label="Research question"
                  className="max-h-40 min-h-[72px] w-full resize-none bg-transparent px-2 py-3 text-[15px] leading-relaxed text-ink placeholder:text-ink3 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={onSubmit}
                  disabled={!value.trim() || busy}
                  aria-label="Run research"
                  className="btn btn--accent btn--square h-10 w-10 flex-none !p-0"
                >
                  <ArrowUp size={17} strokeWidth={2.5} />
                </button>
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line px-3 py-2 sm:px-4">
                <span className="mono text-[10px] uppercase tracking-[0.14em] text-ink3">
                  Enter <span className="text-ink2">to run</span>
                </span>
                <span className="mono text-[10px] uppercase tracking-[0.14em] text-ink3">
                  Shift+Enter <span className="text-ink2">newline</span>
                </span>
                <span className="mono ml-auto hidden items-center gap-1.5 text-[10px] uppercase tracking-[0.14em] text-ink3 sm:flex">
                  Focus <Kbd>⌘K</Kbd>
                </span>
              </div>
            </div>
          </div>

          {/* suggested queries */}
          <div className="anim-rise mt-8" style={{ animationDelay: "180ms" }}>
            <div className="mb-2 flex items-center justify-between">
              <CapsLabel>Suggested research</CapsLabel>
              <CapsLabel className="tnum">{String(SUGGESTED_QUERIES.length).padStart(2, "0")}</CapsLabel>
            </div>
            <ul className="flex flex-col gap-[3px]">
              {SUGGESTED_QUERIES.map((q, i) => (
                <li key={q}>
                  <button
                    type="button"
                    onClick={() => onSuggestion(q)}
                    className="group flex w-full items-center gap-3 border border-transparent px-3 py-2.5 text-left transition-colors hover:border-line hover:bg-ground2/70"
                  >
                    <span className="mono w-6 flex-none text-[11px] text-ink3 tnum group-hover:text-accent">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="truncate text-[13.5px] text-ink2 group-hover:text-ink">{q}</span>
                    <ArrowUp size={13} className="ml-auto flex-none -rotate-45 text-ink3 opacity-0 transition-opacity group-hover:opacity-100" />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* bottom status strip */}
      <footer className="relative z-10 flex h-9 flex-none items-center justify-between border-t border-line px-4 text-[10px] uppercase tracking-[0.14em] text-ink3 sm:px-6">
        <span className="mono">RAG-AWARE SYNTHESIS · INLINE CITATIONS</span>
        <span className="mono hidden sm:block">SOURCES INSPECTABLE · v0.1</span>
      </footer>

      <TelemetryColumn />
    </div>
  );
}
