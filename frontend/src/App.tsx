import { AgUiApp } from "./agui/AgUiApp";
import { Toaster } from "./components/ui/sonner";
import { OpenUIThemeRoot } from "./openui/OpenUIThemeRoot";

/**
 * The AG-UI surface (`POST /v1/ag-ui`, pydantic-ai AGUIAdapter +
 * `@assistant-ui/react-ag-ui`) is the only application shell.
 *
 * OpenUIThemeRoot is mounted once here so every generated interface inherits
 * the app design tokens from a single provider.
 */
export default function App() {
  return (
    <OpenUIThemeRoot>
      <AgUiApp />
      <Toaster />
    </OpenUIThemeRoot>
  );
}
