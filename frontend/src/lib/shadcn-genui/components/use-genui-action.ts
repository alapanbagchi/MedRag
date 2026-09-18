"use client";

import {
  BuiltinActionType,
  useFormName,
  useTriggerAction,
} from "@openuidev/react-lang";

import type { ActionSchema } from "../action";

/**
 * Dispatches a model-authored action (open_url / continue_conversation) from a
 * non-Button interactive component such as a Command row or DropdownMenu item.
 * Mirrors the Button component's action wiring so every clickable primitive
 * reports the same way.
 */
export function useGenuiAction() {
  const triggerAction = useTriggerAction();
  const formName = useFormName();

  return (label: string, action?: ActionSchema) => {
    if (!action) return;
    const actionType = action.type ?? BuiltinActionType.ContinueConversation;
    const params =
      action.type === BuiltinActionType.OpenUrl
        ? { url: (action as { url: string }).url }
        : (action as { params?: Record<string, unknown> }).params;
    triggerAction(label, formName, { type: actionType, params });
  };
}
