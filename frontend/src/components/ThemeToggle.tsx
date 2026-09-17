import { useState } from "react";
import { MoonIcon, SunIcon } from "lucide-react";
import { readTheme, toggleTheme, type Theme } from "../lib/theme";
import { cn } from "../lib/utils";

/**
 * Light/dark switch. Defaults to dark (product default); the inline script in
 * index.html applies the persisted choice before first paint.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>(() => readTheme());
  const dark = theme === "dark";

  return (
    <button
      type="button"
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      title={dark ? "Light theme" : "Dark theme"}
      onClick={() => setTheme(toggleTheme())}
      className={cn(
        "flex size-9 shrink-0 items-center justify-center rounded-full text-muted-foreground outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#4b8cf5]/40",
        className,
      )}
    >
      <SunIcon className={cn("size-4", dark ? "hidden" : "block")} />
      <MoonIcon className={cn("size-4", dark ? "block" : "hidden")} />
    </button>
  );
}
