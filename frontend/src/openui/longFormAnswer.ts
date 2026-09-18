/**
 * The on-demand long-form text answer.
 *
 * The OpenUI program is the product; opening the Text view asks the backend
 * to re-write the turn's verified evidence as a detailed, point-wise
 * paper-style markdown answer. It runs once, the first time that view is
 * opened, and the result is cached per turn so toggling the tabs is free.
 *
 * The ids come from the answer_format part, not the app's thread store: the
 * backend's chat id is the AG-UI thread id, which is a different value from
 * the sidebar's client-side thread id. Asking the backend is the only way to
 * name the right chat.
 */
import { useCallback, useEffect, useState } from "react";

export type LongFormState =
  | { status: "loading"; markdown: "" }
  | { status: "ready"; markdown: string }
  | { status: "error"; markdown: ""; error: string };

/** Finished answers, keyed chat:run. Cleared on reload, like the server cache. */
const cache = new Map<string, string>();

export function useLongFormAnswer(
  chatId: string,
  runId: string,
): { state: LongFormState; reload: () => void } {
  const key = chatId + ":" + runId;
  const [nonce, setNonce] = useState(0);
  const [state, setState] = useState<LongFormState>(() => {
    const hit = cache.get(key);
    return hit ? { status: "ready", markdown: hit } : { status: "loading", markdown: "" };
  });

  useEffect(() => {
    if (!chatId) {
      setState({
        status: "error",
        markdown: "",
        error: "this answer carries no chat id",
      });
      return;
    }
    const hit = cache.get(key);
    if (hit) {
      setState({ status: "ready", markdown: hit });
      return;
    }
    let live = true;
    setState({ status: "loading", markdown: "" });
    const query = runId ? "?run_id=" + encodeURIComponent(runId) : "";
    fetch(
      "/v1/chats/" + encodeURIComponent(chatId) + "/text-answer" + query,
      { method: "POST" },
    )
      .then(async (res) => {
        if (!res.ok) {
          const body = (await res.json().catch(() => ({}))) as { detail?: string };
          throw new Error(body.detail || res.statusText || "request failed");
        }
        return (await res.json()) as { markdown?: unknown };
      })
      .then((data) => {
        if (!live) return;
        const markdown = typeof data.markdown === "string" ? data.markdown : "";
        if (!markdown.trim()) throw new Error("the agent returned an empty answer");
        cache.set(key, markdown);
        setState({ status: "ready", markdown });
      })
      .catch((error: unknown) => {
        if (!live) return;
        setState({
          status: "error",
          markdown: "",
          error: error instanceof Error ? error.message : String(error),
        });
      });
    return () => {
      live = false;
    };
  }, [chatId, key, runId, nonce]);

  const reload = useCallback(() => {
    cache.delete(key);
    setNonce((value) => value + 1);
  }, [key]);

  return { state, reload };
}
