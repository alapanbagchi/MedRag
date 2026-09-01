// ── Message list with user/assistant framing ─────────────────────────
"use client";

import type { Message } from "@/lib/types";
import { formatClock } from "@/lib/utils";
import { CapsLabel } from "@/components/ui/primitives";
import { AssistantMessage } from "@/components/chat/AssistantMessage";

function UserMessage({ message, index }: { message: Message; index: number }) {
  return (
    <section aria-label="Your question" className="mb-9">
      <div className="mb-1.5 flex items-center gap-2">
        <CapsLabel>You</CapsLabel>
        <span className="mono text-[10px] text-ink3 tnum">#{String(index + 1).padStart(2, "0")}</span>
        <span aria-hidden className="h-px flex-1 bg-line" />
        <span className="mono text-[10px] text-ink3 tnum">{formatClock(message.createdAt)}</span>
      </div>
      <p className="max-w-[72ch] whitespace-pre-wrap text-[15px] font-medium leading-relaxed">{message.content}</p>
    </section>
  );
}

export function MessageList({
  messages,
  onCite,
  onOpenSources,
  onRegenerate,
  onRetry,
  onRate,
}: {
  messages: Message[];
  onCite: (n: number) => void;
  onOpenSources: () => void;
  onRegenerate: (msgId: string) => void;
  onRetry: (msgId: string) => void;
  onRate: (msgId: string, rating: "up" | "down" | null) => void;
}) {
  let userIndex = 0;
  return (
    <div className="anim-fade">
      {messages.map((m) => {
        if (m.role === "user") {
          const idx = userIndex;
          userIndex += 1;
          return <UserMessage key={m.id} message={m} index={idx} />;
        }
        return (
          <AssistantMessage
            key={m.id}
            message={m}
            onCite={onCite}
            onOpenSources={onOpenSources}
            onRegenerate={() => onRegenerate(m.id)}
            onRetry={() => onRetry(m.id)}
            onRate={(rating) => onRate(m.id, rating)}
          />
        );
      })}
    </div>
  );
}
