"use client";

/**
 * Chat message surface for the agent sheet.
 *
 * Modelled on assistant-ui's message content: a rounded fill for the sender's
 * side carrying an iMessage-style tail and an optional timestamp underneath.
 * `ios-pop` supplies the spring entrance (see globals.css); the tail is a real
 * SVG path so it blends into the bubble corner instead of leaving a notch.
 */

import type { ReactNode } from "react";

export type MessageSide = "in" | "out";

/** The iMessage tail, drawn out of the bubble's bottom sender corner. */
function BubbleTail({ side }: { side: MessageSide }) {
  const out = side === "out";
  return (
    <svg
      aria-hidden
      viewBox="0 0 9 12"
      className={`absolute bottom-0 h-[12px] w-[9px] ${out ? "right-[-8px]" : "left-[-8px]"}`}
      style={{
        fill: out ? "var(--color-chat-tool)" : "var(--color-chat-ai)",
        ...(out ? { transform: "scaleX(-1)" } : {}),
      }}
    >
      <path d="M9 0 C9 6 5.5 10 0 12 C3 12 6 12 9 12 Z" />
    </svg>
  );
}

export function Message({
  side,
  time,
  avatar,
  delayMs = 0,
  className,
  children,
}: {
  side: MessageSide;
  /** Timestamp shown beneath the bubble (already formatted). */
  time?: string;
  /** Optional avatar, rendered beside an incoming bubble. */
  avatar?: ReactNode;
  delayMs?: number;
  className?: string;
  children: ReactNode;
}) {
  const out = side === "out";
  return (
    <div
      className={`ios-pop flex gap-2.5 ${out ? "ios-pop--out justify-end" : ""}`}
      style={{ animationDelay: `${delayMs}ms` }}
    >
      {avatar ? <div className="w-[36px] shrink-0">{avatar}</div> : null}
      <div
        className={`flex min-w-0 max-w-[50%] flex-col gap-1 ${out ? "items-end" : "items-start"}`}
      >
        <div
          className={`ios-bubble rounded-[20px] px-4 py-3 text-[15px] leading-[22px] ${
            out
              ? "rounded-br-none bg-chat-tool text-chat-tool-ink"
              : "rounded-bl-none bg-chat-ai text-chat-ai-ink"
          } ${className ?? ""}`}
        >
          {children}
          <BubbleTail side={side} />
        </div>
        {time ? (
          <span className="px-1 text-[11px] leading-none text-muted-foreground/70">{time}</span>
        ) : null}
      </div>
    </div>
  );
}
