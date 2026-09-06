import { MessagePrimitive } from "@assistant-ui/react";

export function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-[#F1F4F5] px-5 py-3 text-[16px] leading-7 text-[#0D0E1A]">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}
