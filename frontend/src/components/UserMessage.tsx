import { MessagePrimitive } from "@assistant-ui/react";

export function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end">
      <div className="max-w-[85%] rounded-3xl rounded-br-md bg-[var(--user-bubble-bg)] px-4 py-2.5 text-[15px] leading-6 text-[var(--user-bubble-ink)]">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}