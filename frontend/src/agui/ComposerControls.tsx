"use client";

import { useEffect, useMemo, useState, type ComponentType } from "react";
import {
  CheckIcon,
  ChevronDownIcon,
  CpuIcon,
  GlobeIcon,
  SearchIcon,
  SparklesIcon,
} from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { CitationStylePicker } from "../components/CitationStylePicker";
import { useComposerStore } from "./composerStore";
import { useModelsStore } from "./modelStore";

const MAX_MODELS = 60;

const SEARCH_TIP =
  "We strongly encourage keeping search on to ground your answers with trusted websites";
const DEEP_TIP =
  "Keeping this option on will allow us to ground your answers on actual facts. " +
  "Keeping this off is the same as going to chatgpt and getting an answer with no " +
  "sources whatsoever. You can not trust the answers when this is turned off. Use " +
  "only for the most common general knowledge stuff when you need fast answers";

function ToggleButton({
  icon: Icon,
  label,
  tooltip,
  active,
  onClick,
}: {
  icon: ComponentType<{ className?: string }>;
  label: string;
  tooltip: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-pressed={active}
          aria-label={label}
          className={cn(
            "flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-2.5 text-[12.5px] font-medium transition-colors",
            active
              ? "border-brand/40 bg-brand/10 text-brand"
              : "border-border/70 text-muted-foreground hover:border-border hover:text-foreground",
          )}
        >
          <Icon className="size-3.5" />
          <span className="hidden sm:inline">{label}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        sideOffset={8}
        className="max-w-[340px] text-[12px] leading-relaxed text-pretty"
      >
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

/** Model picker fed by the backend's OpenAI-compatible /v1/models proxy. */
function ModelSelector() {
  const model = useComposerStore((s) => s.model);
  const setModel = useComposerStore((s) => s.setModel);
  const models = useModelsStore((s) => s.models);
  const defaultModel = useModelsStore((s) => s.defaultModel);
  const loaded = useModelsStore((s) => s.loaded);
  const error = useModelsStore((s) => s.error);
  const load = useModelsStore((s) => s.load);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (open) void load();
  }, [open, load]);
  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return models.slice(0, MAX_MODELS);
    return models.filter((m) => m.toLowerCase().includes(needle)).slice(0, MAX_MODELS);
  }, [models, query]);

  const active = model || defaultModel;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="Model"
          className="flex h-8 max-w-[220px] shrink-0 items-center gap-1.5 rounded-full border border-border/70 px-2.5 text-[12.5px] font-medium text-muted-foreground transition-colors hover:border-border hover:text-foreground"
        >
          <CpuIcon className="size-3.5 shrink-0" />
          <span className="truncate">{active || "Model"}</span>
          <ChevronDownIcon className="size-3 shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" side="top" className="w-[320px] p-0">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <SearchIcon className="size-3.5 shrink-0 text-foreground/40" />
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={loaded ? "Search " + models.length + " models" : "Loading models"}
            className="w-full bg-transparent text-[13px] outline-none placeholder:text-foreground/35"
          />
        </div>
        <div className="max-h-64 overflow-y-auto p-1">
          <button
            type="button"
            onClick={() => {
              setModel("");
              setOpen(false);
            }}
            className="flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px] text-foreground/75 hover:bg-foreground/[0.04]"
          >
            <span className="truncate">
              Default{defaultModel ? " (" + defaultModel + ")" : ""}
            </span>
            {!model ? <CheckIcon className="size-3.5 shrink-0 text-primary" /> : null}
          </button>
          {shown.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => {
                setModel(m);
                setOpen(false);
              }}
              className="flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px] text-foreground/75 hover:bg-foreground/[0.04]"
            >
              <span className="truncate">{m}</span>
              {m === active ? (
                <CheckIcon className="size-3.5 shrink-0 text-primary" />
              ) : null}
            </button>
          ))}
          {!loaded ? (
            <p className="px-3 py-4 text-center text-[12px] text-foreground/40">
              Loading models
            </p>
          ) : null}
          {loaded && shown.length === 0 ? (
            <p className="px-3 py-4 text-center text-[12px] text-foreground/40">
              No matching model
            </p>
          ) : null}
          {error ? <p className="text-danger px-3 py-2 text-[11px]">{error}</p> : null}
        </div>
      </PopoverContent>
    </Popover>
  );
}

/** The composer's bottom row: model picker + web/deep search toggles. */
export function ComposerControls() {
  const webSearch = useComposerStore((s) => s.webSearch);
  const deepSearch = useComposerStore((s) => s.deepSearch);
  const setWebSearch = useComposerStore((s) => s.setWebSearch);
  const setDeepSearch = useComposerStore((s) => s.setDeepSearch);
  return (
    <>
      <ModelSelector />
      <ToggleButton
        icon={GlobeIcon}
        label="Search"
        tooltip={SEARCH_TIP}
        active={webSearch}
        onClick={() => setWebSearch(!webSearch)}
      />
      <ToggleButton
        icon={SparklesIcon}
        label="Deep Search"
        tooltip={DEEP_TIP}
        active={deepSearch}
        onClick={() => setDeepSearch(!deepSearch)}
      />
      <CitationStylePicker />
    </>
  );
}
