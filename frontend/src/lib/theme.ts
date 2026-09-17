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
