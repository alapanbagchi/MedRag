// ── small UI primitives ─────────────────────────────────────────────
"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
  return <kbd className={cn("kbd", className)}>{children}</kbd>;
}

export function CapsLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn("caps-label", className)}>{children}</span>;
}

export function Led({
  state,
  pulse = false,
  className,
}: {
  state: "accent" | "ok" | "err" | "dim";
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      aria-hidden
      className={cn(
        "led",
        state === "accent" && "led--accent",
        state === "ok" && "led--ok",
        state === "err" && "led--err",
        pulse && "led--pulse",
        className
      )}
    />
  );
}

/** A button that opens / closes a small anchored menu (three-dot style). */
export function Dropdown({
  trigger,
  items,
  align = "right",
  label = "More options",
}: {
  trigger: React.ReactNode;
  items: {
    key: string;
    label: string;
    icon?: React.ReactNode;
    danger?: boolean;
    onSelect: () => void;
  }[];
  align?: "left" | "right";
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="icon-btn"
      >
        {trigger}
      </button>
      {open && (
        <div
          role="menu"
          aria-label={label}
          className={cn(
            "anim-fade absolute z-50 mt-1 min-w-44 border border-line-strong bg-panel2 shadow-lg",
            align === "right" ? "right-0" : "left-0"
          )}
        >
          {items.map((item, i) => (
            <button
              key={item.key}
              role="menuitem"
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setOpen(false);
                item.onSelect();
              }}
              className={cn(
                "flex w-full items-center gap-2.5 px-3 py-2 text-left text-[12.5px] font-medium transition-colors",
                item.danger
                  ? "text-err hover:bg-err/10"
                  : "text-ink hover:bg-ground2"
              )}
              style={i > 0 ? { borderTop: "1px solid var(--line)" } : undefined}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
