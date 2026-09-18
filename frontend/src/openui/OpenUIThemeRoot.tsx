/**
 * Mounts the OpenUI ThemeProvider once, driven by the app own .dark class.
 *
 * There must be exactly one provider: a second one would also target body and
 * fight over the injected --openui-* variables.
 */
import { useEffect, useState, type ReactNode } from "react";
import { ThemeProvider, type ThemeMode } from "@openuidev/react-ui";
import { readTheme } from "../lib/theme";
import { openuiTheme } from "./theme";

/** The app light/dark mode, followed live off the html class list. */
function useAppThemeMode(): ThemeMode {
  const [mode, setMode] = useState<ThemeMode>(() => readTheme());
  useEffect(() => {
    // The toggle flips the class on <html>; no global store owns it, so
    // observe the attribute instead of adding a second source of truth.
    const observer = new MutationObserver(() => setMode(readTheme()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);
  return mode;
}

export function OpenUIThemeRoot({ children }: { children: ReactNode }) {
  const mode = useAppThemeMode();
  return (
    <ThemeProvider mode={mode} lightTheme={openuiTheme}>
      {children}
    </ThemeProvider>
  );
}
