/** http(s) URL safe to put in an href, or null for any other scheme. */
export function safeHref(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const { protocol } = new URL(url);
    return protocol === "http:" || protocol === "https:" ? url : null;
  } catch {
    return null;
  }
}

/** Display host for a URL (lowercased, no port, no leading www). */
export function domainOf(url: string | null | undefined): string {
  if (!url) return "";
  try {
    return new URL(url).hostname.replace(/^www[.]/, "");
  } catch {
    return url.replace(/^https?:\/\//, "").split("/")[0] ?? "";
  }
}
