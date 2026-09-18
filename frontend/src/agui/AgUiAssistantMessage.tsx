/**
 * AG-UI assistant message: the app rich rendering (thinking stream, tool-call
 * rows, sub-agent cards), the standalone cards (plan, verdict, run stats) and
 * the answer's citation reference list.
 *
 * The verified source list, the style catalog and the reader's chosen style
 * are resolved once here and provided to every renderer, so the markdown
 * answer, the OpenUI visual answer, the inline badges, the picker and the
 * bottom reference list all agree.
 */

import { memo, useEffect, useMemo, type ComponentType } from "react";
import { useAuiState } from "@assistant-ui/react";
import { AssistantMessage } from "../components/AssistantMessage";
import { AnswerReferences } from "../components/AnswerReferences";
import {
  CitationActiveStyleContext,
  CitationSourcesContext,
  CitationStylesContext,
  FALLBACK_CITATION_STYLE,
} from "../components/CitationBadge";
import { useCitationStyle } from "../lib/citationStyle";
import { useCitationRender } from "../lib/citationRender";
import { useCslCatalog } from "../lib/cslCatalog";
import {
  EMPTY_CATALOG,
  EMPTY_SOURCES,
  applyRender,
  catalogDescriptor,
  citationStyles,
  messageSources,
  type CitationRenderResult,
  type CitationStyleCatalog,
} from "../lib/citations";
import type { CitationStyleDef, Source } from "../lib/types";
import { AGUI_DATA_BY_NAME } from "./renderers";

const EMPTY_PARTS: readonly unknown[] = [];
const EMPTY_IDS = { chatId: "", runId: "" };

/**
 * Standalone cards still rendered under the answer. Tool calls and their
 * results (retrieval progress, verdict tables, run stats) moved into the
 * agent sheet, reached from the in-message thinking row; only real errors
 * stay on the main screen.
 */
const OPENUI_NAMES = ["error_detail"];

function RenderData({ name, data }: { name: string; data: unknown }) {
  const Renderer = AGUI_DATA_BY_NAME[name] as unknown as
    | ComponentType<{ data: unknown }>
    | undefined;
  if (!Renderer) return null;
  return <Renderer data={data} />;
}

const standaloneCache = new WeakMap<
  readonly unknown[],
  { key: string; parts: readonly unknown[] }
>();

/**
 * The standalone card parts for a content array, cached by array identity so a
 * stream with no standalone cards keeps returning the same EMPTY_PARTS. That
 * keeps this component from re-rendering on every thinking token.
 */
function standalonePartsFor(
  content: readonly unknown[],
  names: string[],
): readonly unknown[] {
  const key = names.join(",");
  const hit = standaloneCache.get(content);
  if (hit && hit.key === key) return hit.parts;
  const parts = content.filter((raw) => {
    const part = raw as { type?: string; name?: string };
    return part.type === "data" && names.includes(part.name ?? "");
  });
  const result = parts.length === 0 ? EMPTY_PARTS : parts;
  standaloneCache.set(content, { key, parts: result });
  return result;
}

function Standalone({ names, position }: { names: string[]; position: "before" | "after" }) {
  const parts = useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return EMPTY_PARTS;
    return standalonePartsFor(content, names);
  });
  if (parts.length === 0) return null;
  return (
    <div className={position === "before" ? "mb-3 flex flex-col gap-2" : "mt-3 flex flex-col gap-2"}>
      {parts.map((part, index) => (
        <RenderData
          key={index}
          name={(part as { name?: string }).name ?? ""}
          data={(part as { data?: unknown }).data}
        />
      ))}
    </div>
  );
}

/** The verified source list for this message, in citation order. */
function useMessageSources(): Source[] {
  return useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return EMPTY_SOURCES;
    return messageSources(content);
  });
}

/** The styles precomputed in this message (the picker pins them as common). */
function useMessageCitationStyles(): CitationStyleCatalog {
  return useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return EMPTY_CATALOG;
    return citationStyles(content);
  });
}

// The backend chat/run ids for this answer, cached by content identity so the
// selector returns a stable object (a fresh object would re-render per token).
const idsCache = new WeakMap<object, { chatId: string; runId: string }>();

function answerIds(content: unknown): { chatId: string; runId: string } {
  if (!Array.isArray(content)) return EMPTY_IDS;
  const key = content as unknown as object;
  const hit = idsCache.get(key);
  if (hit) return hit;
  let result = EMPTY_IDS;
  for (let i = content.length - 1; i >= 0; i -= 1) {
    const part = content[i] as { type?: string; name?: string; data?: unknown };
    if (part.type !== "data" || part.name !== "answer_format") continue;
    const data = part.data as Record<string, unknown> | undefined;
    result = {
      chatId: typeof data?.chat_id === "string" ? data.chat_id : "",
      runId: typeof data?.run_id === "string" ? data.run_id : "",
    };
    break;
  }
  idsCache.set(key, result);
  return result;
}

function useAnswerIds(): { chatId: string; runId: string } {
  return useAuiState((s) => {
    const content = s.message.content;
    if (typeof content === "string") return EMPTY_IDS;
    return answerIds(content);
  });
}

/** The active style: chosen when valid, else the message default (APA). */
function resolveActiveStyle(
  messageStyles: CitationStyleDef[],
  defaultStyleId: string,
  catalog: readonly { id: string; title: string; format?: string }[],
  selected: string,
  render: CitationRenderResult | undefined,
): CitationStyleDef {
  const wanted = selected || defaultStyleId;
  const fromMessage = messageStyles.find((style) => style.id === wanted);
  const entry = fromMessage ? undefined : catalog.find((style) => style.id === wanted);
  const base: CitationStyleDef =
    fromMessage ??
    (entry ? catalogDescriptor(entry) : undefined) ??
    messageStyles.find((style) => style.id === defaultStyleId) ??
    messageStyles[0] ??
    FALLBACK_CITATION_STYLE;
  if (!render) return base;
  return {
    id: base.id,
    label: render.label || base.label,
    authorYear: render.authorYear,
    separator: render.separator,
    prefix: render.prefix,
    suffix: render.suffix,
  };
}

export const AgUiAssistantMessage = memo(function AgUiAssistantMessage() {
  const baseSources = useMessageSources();
  const messageCatalog = useMessageCitationStyles();
  const { chatId, runId } = useAnswerIds();
  const selected = useCitationStyle((state) => state.styleId);
  const catalog = useCslCatalog((state) => state.styles);
  const ensureRender = useCitationRender((state) => state.ensure);

  const messageStyles = messageCatalog.styles;
  const known =
    messageStyles.some((style) => style.id === selected) ||
    catalog.some((style) => style.id === selected);
  const activeId =
    selected && known
      ? selected
      : messageCatalog.defaultStyleId || messageStyles[0]?.id || FALLBACK_CITATION_STYLE.id;

  // A style the answer already carries needs no request; anything else is
  // rendered on demand for this turn (and cached by the render store).
  const precomputed = baseSources.some(
    (source) => source.citations && activeId in source.citations,
  );
  const render = useCitationRender(
    (state) => state.cache[chatId + ":" + runId + ":" + activeId],
  );
  useEffect(() => {
    if (!precomputed && chatId && activeId) {
      ensureRender(chatId, runId, activeId);
    }
  }, [precomputed, chatId, runId, activeId, ensureRender]);

  const active = useMemo(
    () => resolveActiveStyle(messageStyles, messageCatalog.defaultStyleId, catalog, activeId, render),
    [messageStyles, messageCatalog.defaultStyleId, catalog, activeId, render],
  );
  const sources = useMemo(() => applyRender(baseSources, render), [baseSources, render]);

  return (
    <CitationSourcesContext.Provider value={sources}>
      <CitationStylesContext.Provider value={messageCatalog}>
        <CitationActiveStyleContext.Provider value={active}>
          <div
            className="mb-5 flex w-full min-w-0 flex-1 flex-col"
            data-citation-scope=""
          >
            <AssistantMessage />
            <Standalone names={OPENUI_NAMES} position="after" />
            <AnswerReferences />
          </div>
        </CitationActiveStyleContext.Provider>
      </CitationStylesContext.Provider>
    </CitationSourcesContext.Provider>
  );
});
