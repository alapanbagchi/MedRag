import { Loader2Icon } from "lucide-react";
import * as React from "react";

import { clsx } from "clsx";

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  return (
    <Loader2Icon
      role="status"
      aria-label="Loading"
      data-slot="spinner"
      className={clsx("size-4 animate-spin", className)}
      {...props}
    />
  );
}

export { Spinner };
