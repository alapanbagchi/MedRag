/**
 * The verified source list for the current answer, shared with the OpenUI
 * component library.
 *
 * The backend emits the turn's judge-kept passages, in citation order, as an
 * \`answer_sources\` AG-UI data part. The app bridges that into this context
 * (see answerSources.ts) and the library's root Card reads it, so the
 * Sources strip and every inline [n] chip resolve to the ledger - never to
 * anything the model wrote.
 *
 * This module is deliberately free of app/runtime imports: it is evaluated by
 * the OpenUI CLI when generating the model-facing prompt.
 */

import { createContext, useContext } from "react";
import type { CardSource } from "@openuidev/react-ui";

export const MedRagSourcesContext = createContext<CardSource[] | undefined>(
  undefined,
);

/** The verified sources for this answer, in [n]-citation order. */
export function useMedRagSources(): CardSource[] | undefined {
  return useContext(MedRagSourcesContext);
}
