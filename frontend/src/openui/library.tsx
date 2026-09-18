/**
 * Entry point the OpenUI CLI and the prompt generator read: the shadcn-backed
 * answer library, plus the prompt options that describe it to the model.
 */
export { shadcnChatLibrary as library } from "@/lib/shadcn-genui";
export { promptOptions } from "./prompt-options.mjs";
