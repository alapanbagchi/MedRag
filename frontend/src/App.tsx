import { AgUiApp } from "./agui/AgUiApp";

/**
 * The AG-UI surface (`POST /v1/ag-ui`, pydantic-ai AGUIAdapter +
 * `@assistant-ui/react-ag-ui`) is the only application shell.
 */
export default function App() {
  return <AgUiApp />;
}
