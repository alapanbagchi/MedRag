/**
 * Per-thread message persistence for the AG-UI surface.
 *
 * The adapter is bound to one thread id at construction (the provider is
 * remounted per thread), keeps the parsed history in memory, and debounces
 * localStorage writes, so a streaming run does not re-parse/rewrite the whole
 * thread on every message update.
 */

import type {
  ExportedMessageRepositoryItem,
  ThreadHistoryAdapter,
} from "@assistant-ui/react";
import { HISTORY_KEY_PREFIX } from "./aguiStore";

type StoredItem = { parentId: string | null; message: Record<string, any> };

const cache = new Map<string, StoredItem[]>();
const pending = new Map<string, ReturnType<typeof setTimeout>>();

function readThread(threadId: string): StoredItem[] {
  const cached = cache.get(threadId);
  if (cached) return cached;
  let items: StoredItem[] = [];
  try {
    const raw = localStorage.getItem(HISTORY_KEY_PREFIX + threadId);
    const parsed = raw ? (JSON.parse(raw) as unknown) : [];
    items = Array.isArray(parsed)
      ? parsed.filter((i): i is StoredItem => !!i && typeof i === "object")
      : [];
  } catch {
    items = [];
  }
  cache.set(threadId, items);
  return items;
}

function scheduleWrite(threadId: string): void {
  if (pending.has(threadId)) return;
  const timer = setTimeout(() => {
    pending.delete(threadId);
    try {
      localStorage.setItem(
        HISTORY_KEY_PREFIX + threadId,
        JSON.stringify(cache.get(threadId) ?? []),
      );
    } catch (err) {
      console.warn("[agui] history persist failed for", threadId, err);
    }
  }, 300);
  pending.set(threadId, timer);
}

/** Revive JSON-persisted items: `createdAt` returns as a Date. */
function revive(item: StoredItem): ExportedMessageRepositoryItem {
  const message = item.message ?? {};
  return {
    parentId: item.parentId ?? null,
    message: {
      ...(message as object),
      createdAt: message.createdAt ? new Date(message.createdAt) : new Date(),
    },
  } as ExportedMessageRepositoryItem;
}

function persist(threadId: string, item: ExportedMessageRepositoryItem): void {
  if (!threadId) return;
  const items = readThread(threadId);
  const messageId = (item.message as { id?: string }).id;
  const index = messageId ? items.findIndex((x) => x.message?.id === messageId) : -1;
  const stored: StoredItem = {
    parentId: item.parentId ?? null,
    message: item.message as unknown as Record<string, any>,
  };
  if (index >= 0) items[index] = stored;
  else items.push(stored);
  scheduleWrite(threadId);
}

export function createAgUiHistoryAdapter(threadId: string): ThreadHistoryAdapter {
  return {
    async load() {
      const items = threadId ? readThread(threadId) : [];
      const messages = items.map(revive);
      return { messages, headId: messages.at(-1)?.message.id ?? null };
    },
    async append(item) {
      persist(threadId, item);
    },
    async update(item) {
      persist(threadId, item);
    },
  };
}
