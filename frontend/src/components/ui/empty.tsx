import * as React from "react";

import { clsx } from "clsx";

function Empty({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty"
      className={clsx(
        "flex min-w-0 flex-1 flex-col items-center justify-center gap-4 rounded-lg border border-dashed p-6 text-center text-balance",
        className,
      )}
      {...props}
    />
  );
}

function EmptyHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-header"
      className={clsx("flex max-w-sm flex-col items-center gap-2", className)}
      {...props}
    />
  );
}

function EmptyMedia({
  className,
  variant = "default",
  ...props
}: React.ComponentProps<"div"> & { variant?: "default" | "icon" }) {
  return (
    <div
      data-slot="empty-media"
      data-variant={variant}
      className={clsx(
        "flex shrink-0 items-center justify-center",
        variant === "icon" &&
          "bg-muted text-muted-foreground mb-2 flex size-10 items-center justify-center rounded-lg [&_svg:not([class*='size-'])]:size-5",
        className,
      )}
      {...props}
    />
  );
}

function EmptyTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-title"
      className={clsx("text-sm font-medium", className)}
      {...props}
    />
  );
}

function EmptyDescription({ className, ...props }: React.ComponentProps<"p">) {
  return (
    <p
      data-slot="empty-description"
      className={clsx("text-muted-foreground text-sm", className)}
      {...props}
    />
  );
}

function EmptyContent({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="empty-content"
      className={clsx("flex w-full max-w-sm flex-col items-center gap-2 text-sm", className)}
      {...props}
    />
  );
}

export { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle };
