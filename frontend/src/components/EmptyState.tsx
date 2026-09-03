import { SuggestionPrimitive, ThreadPrimitive } from "@assistant-ui/react";
import { SparklesIcon } from "lucide-react";

export function EmptyState() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-9 py-10">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-[#4b8cf5] to-[#9d7bfb] text-white shadow-lg shadow-indigo-500/25">
          <SparklesIcon className="size-7" />
        </div>
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">MedRAG</h1>
          <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted-foreground">
            Agentic medical literature search over PMC open-access research.
            Ask a clinical question — get a cited, evidence-grounded answer.
          </p>
        </div>
      </div>

      <ThreadPrimitive.Suggestions>
        {() => (
          <div className="grid w-full max-w-2xl grid-cols-1 gap-2.5 sm:grid-cols-2">
            <SuggestionCard />
          </div>
        )}
      </ThreadPrimitive.Suggestions>
    </div>
  );
}

function SuggestionCard() {
  return (
    <SuggestionPrimitive.Trigger className="flex flex-col gap-1 rounded-2xl border border-border bg-card px-4 py-3 text-left transition hover:border-primary/40 hover:bg-accent/50">
      <SuggestionPrimitive.Title className="text-sm font-medium" />
      <SuggestionPrimitive.Description className="text-xs text-muted-foreground" />
    </SuggestionPrimitive.Trigger>
  );
}