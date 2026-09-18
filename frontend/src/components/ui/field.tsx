"use client";

import * as React from "react";

import { clsx } from "clsx";

function FieldSet({ className, ...props }: React.ComponentProps<"fieldset">) {
  return (
    <fieldset
      data-slot="field-set"
      className={clsx("flex flex-col gap-4", className)}
      {...props}
    />
  );
}

function FieldLegend({ className, ...props }: React.ComponentProps<"legend">) {
  return (
    <legend
      data-slot="field-legend"
      className={clsx("mb-1.5 text-sm font-medium", className)}
      {...props}
    />
  );
}

function FieldGroup({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="field-group"
      className={clsx("flex w-full flex-col gap-4", className)}
      {...props}
    />
  );
}

function Field({
  className,
  orientation = "vertical",
  ...props
}: React.ComponentProps<"div"> & { orientation?: "vertical" | "horizontal" | "responsive" }) {
  return (
    <div
      role="group"
      data-slot="field"
      data-orientation={orientation}
      className={clsx(
        "flex w-full gap-2 data-[orientation=vertical]:flex-col data-[orientation=horizontal]:flex-row data-[orientation=horizontal]:items-center",
        className,
      )}
      {...props}
    />
  );
}

function FieldContent({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="field-content"
      className={clsx("flex flex-1 flex-col gap-1", className)}
      {...props}
    />
  );
}

function FieldLabel({ className, ...props }: React.ComponentProps<"label">) {
  return (
    <label
      data-slot="field-label"
      className={clsx("flex items-center gap-2 text-sm font-medium", className)}
      {...props}
    />
  );
}

function FieldTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="field-title"
      className={clsx("text-sm font-medium", className)}
      {...props}
    />
  );
}

function FieldDescription({ className, ...props }: React.ComponentProps<"p">) {
  return (
    <p
      data-slot="field-description"
      className={clsx("text-muted-foreground text-sm", className)}
      {...props}
    />
  );
}

function FieldError({ className, ...props }: React.ComponentProps<"p">) {
  return (
    <p
      data-slot="field-error"
      role="alert"
      className={clsx("text-destructive text-sm", className)}
      {...props}
    />
  );
}

function FieldSeparator({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="field-separator"
      className={clsx("bg-border h-px w-full", className)}
      {...props}
    />
  );
}

export {
  Field,
  FieldContent,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSeparator,
  FieldSet,
  FieldTitle,
};
