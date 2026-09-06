import {
  ActivityIcon,
  FlaskConicalIcon,
  MapIcon,
  ScaleIcon,
} from "lucide-react";
import { startDemoRun } from "../lib/demo-run";
import { useChatStore } from "../lib/store";

interface Prompt {
  text: string;
  icon: React.ReactNode;
}

const PROMPTS: Prompt[] = [
  { text: "Can AI replace radiologists?", icon: <FlaskConicalIcon className="size-3.5" /> },
  { text: "Is PENK predictive of AKI after cardiac surgery?", icon: <ActivityIcon className="size-3.5" /> },
  { text: "Compare radial vs vein graft vasospasm risk", icon: <ScaleIcon className="size-3.5" /> },
  { text: "Map AKI prediction models in cardiac surgery", icon: <MapIcon className="size-3.5" /> },
];

/** Example-prompt card row above the input. Clicking a card runs it. */
export function PromptCards() {
  const run = (text: string) => {
    const id = useChatStore.getState().currentThreadId;
    if (id) void startDemoRun(id, text);
  };

  return (
    <div className="grid w-full grid-cols-2 gap-2.5 lg:grid-cols-4">
      {PROMPTS.map((p) => (
        <button
          key={p.text}
          type="button"
          onClick={() => run(p.text)}
          className="flex min-h-[104px] flex-col justify-between rounded-xl border border-border bg-white p-3 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-[#1883AE]/50 hover:shadow-[0_12px_28px_rgba(24,131,174,0.12)]"
        >
          <span className="text-[13px] leading-5 text-[#232838]">{p.text}</span>
          <span className="mt-2 text-[#b3bec7]">{p.icon}</span>
        </button>
      ))}
    </div>
  );
}
