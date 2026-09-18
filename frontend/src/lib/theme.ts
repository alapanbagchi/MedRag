/**
 * Light/dark theme: the .dark class on <html> drives the CSS custom
 * properties in globals.css. Persisted in localStorage; an inline script in
 * index.html applies the stored value before paint so there is no flash.
 */

export type Theme = "light" | "dark";

const STORAGE_KEY = "medrag:theme";

export function readTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.style.colorScheme = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // storage unavailable (private mode) — the class still applies
  }
}

/** Toggle and persist the theme; returns the new value. */
export function toggleTheme(): Theme {
  const next: Theme = readTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  return next;
}

/**
 * Fixed fallbacks for the chart palette (light theme). The live values come
 * from the --series-N tokens in globals.css; these are only used when the CSS
 * is not available (SSR or a test), so nothing here is a second source of
 * truth for the UI.
 */
const SERIES_FALLBACK = [
  "#4b8cf5",
  "#18ae95",
  "#f5a623",
  "#e5484d",
  "#8b5cf6",
  "#0ea5e9",
  "#64748b",
];

/**
 * Read a CSS custom property from :root. Canvas-based charts and inline SVG
 * need concrete color strings at render time, so they read the central tokens
 * here rather than hardcoding a hex value.
 */
export function cssVar(name: string, fallback = ""): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || fallback;
}

/** The categorical chart palette, resolved from the central --series-N tokens. */
export function seriesPalette(count = SERIES_FALLBACK.length): string[] {
  return SERIES_FALLBACK.slice(0, Math.max(0, count)).map((fallback, index) =>
    cssVar("--series-" + (index + 1), fallback),
  );
}
