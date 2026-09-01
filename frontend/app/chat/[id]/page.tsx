import { ChatView } from "@/components/chat/ChatView";

export default async function ChatPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ q?: string }>;
}) {
  const [{ id }, { q }] = await Promise.all([params, searchParams]);
  return <ChatView conversationId={id} initialQuestion={q ?? undefined} />;
}
