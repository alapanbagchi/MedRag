import * as React from "react";

import { clsx } from "clsx";

function ButtonGroup({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      role="group"
      data-slot="button-group"
      className={clsx(
        "flex w-fit items-stretch [&>*]:focus-visible:relative [&>*]:focus-visible:z-10 [&>input]:flex-1 has-[>[data-slot=button-group]]:gap-2 [&>*:not(:first-child)]:rounded-l-none [&>*:not(:last-child)]:rounded-r-none [&>*:not(:first-child)]:border-l-0",
        className,
      )}
      {...props}
    />
  );
}

function ButtonGroupText({
  className,
  ...props
}: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="button-group-text"
      className={clsx(
        "bg-muted flex items-center gap-2 rounded-md border px-2.5 text-sm font-medium",
        className,
      )}
      {...props}
    />
  );
}

function ButtonGroupSeparator({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="button-group-separator"
      role="separator"
      className={clsx("bg-input !h-auto !w-px self-stretch", className)}
      {...props}
    />
  );
}

export { ButtonGroup, ButtonGroupSeparator, ButtonGroupText };
