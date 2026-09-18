"use client";

import * as React from "react";

import { clsx } from "clsx";

function InputGroup({ className, ...props }: React.ComponentProps<"div">) {
 return (
 <div
 data-slot="input-group"
 role="group"
 className={clsx(
 "border-input focus-within:border-ring focus-within:ring-ring/50 relative flex w-full items-center rounded-md border transition-[color,box-shadow] focus-within:ring-[3px] has-[input:disabled]:opacity-50",
 className,
 )}
 {...props}
 />
 );
}

function InputGroupAddon({
 className,
 align = "inline-start",
 ...props
}: React.ComponentProps<"div"> & { align?: "inline-start" | "inline-end" | "block-start" | "block-end" }) {
 return (
 <div
 data-slot="input-group-addon"
 data-align={align}
 className={clsx(
 "text-muted-foreground flex cursor-text items-center justify-center gap-2 px-2.5 text-sm [&_svg:not([class*='size-'])]:size-4",
 align === "inline-start" && "order-first",
 align === "inline-end" && "order-last",
 className,
 )}
 {...props}
 />
 );
}

function InputGroupInput({
 className,
 ...props
}: React.ComponentProps<"input">) {
 return (
 <input
 data-slot="input-group-input"
 className={clsx(
 "placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md bg-transparent px-2.5 py-1 text-sm outline-none disabled:cursor-not-allowed",
 className,
 )}
 {...props}
 />
 );
}

export { InputGroup, InputGroupAddon, InputGroupInput };
