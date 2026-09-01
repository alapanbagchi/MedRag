// ── Chat input console ──────────────────────────────────────────────
"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { Kbd } from "@/components/ui/primitives";

export function ChatInput({
  onSend,
  onStop,
  busy,
  autoFocus,
  shortcutRef,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  autoFocus?: boolean;
  /** register a focus function so ⌘K can target this input */
  shortcutRef?: React.MutableRefObject<(() => void) | null>;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  useEffect(() => {
    if (shortcutRef) shortcutRef.current = () => ref.current?.focus();
    return () => {
      if (shortcutRef) shortcutRef.current = null;
    };
  }, [shortcutRef]);

  const submit = () => {
    const v = value.trim();
    if (!v || busy) return;
    setValue("");
    onSend(v);
    requestAnimationFrame(() => {
      if (ref.current) {
        ref.current.style.height = "auto";
        ref.current.focus();
      }
    });
  };

  return (
    <div className="input-frame">
      <div className="flex items-end gap-2 p-2 sm:p-3">
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = Math.min(e.target.scrollHeight, 168) + "px";
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
          placeholder={busy ? "Research in progress…" : "Ask a follow-up…"}
          aria-label="Follow-up question"
          className="max-h-[168px] w-full resize-none bg-transparent !py-2.5 px-2 text-[14.5px] leading-relaxed text-ink placeholder:text-ink3 focus:outline-none"
        />
        {busy ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generation"
            className="btn btn--square h-10 w-10 flex-none !border-err !p-0 !text-err hover:!bg-err/10"
          >
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!value.trim()}
            aria-label="Send message"
            className="btn btn--accent btn--square h-10 w-10 flex-none !p-0"
          >
            <ArrowUp size={17} strokeWidth={2.5} />
          </button>
        )}
      </div>
      <div className="flex items-center gap-4 border-t border-line px-3 py-1.5 sm:px-4">
        <span className="mono text-[9.5px] uppercase tracking-[0.14em] text-ink3">
          Enter <span className="text-ink2">run</span>
        </span>
        <span className="mono text-[9.5px] uppercase tracking-[0.14em] text-ink3">
          Shift+Enter <span className="text-ink2">newline</span>
        </span>
        <span className="mono ml-auto hidden items-center gap-1.5 text-[9.5px] uppercase tracking-[0.14em] text-ink3 sm:flex">
          Focus <Kbd>⌘K</Kbd>
        </span>
      </div>
    </div>
  );
}
