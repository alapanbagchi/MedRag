import { AuiIf, ThreadPrimitive } from "@assistant-ui/react";
import { AssistantMessage } from "./AssistantMessage";
import { Composer } from "./Composer";
import { EmptyState } from "./EmptyState";
import { UserMessage } from "./UserMessage";

export function ThreadView() {
  return (
    <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col">
      <ThreadPrimitive.Viewport className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 pt-8">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <EmptyState />
          </AuiIf>
          <AuiIf condition={(s) => !s.thread.isEmpty}>
            <ThreadPrimitive.Messages>
              {({ message }) => {
                if (message.composer.isEditing) return null;
                if (message.role === "user") return <UserMessage />;
                return <AssistantMessage />;
              }}
            </ThreadPrimitive.Messages>
          </AuiIf>
        </div>

        <ThreadPrimitive.ViewportFooter className="sticky bottom-0 z-10 mt-auto w-full bg-gradient-to-t from-background via-background to-transparent px-4 pb-5 pt-8">
          <div className="mx-auto w-full max-w-3xl">
            <Composer />
          </div>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}