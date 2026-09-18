"use client";

import * as React from "react";
import { Toaster as Sonner, toast } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

function Toaster({ ...props }: ToasterProps) {
 return (
 <Sonner
 className="toaster group"
 position="bottom-right"
 toastOptions={{
 classNames: {
 toast:
 "group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:",
 description: "group-[.toast]:text-muted-foreground",
 actionButton: "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
 cancelButton: "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
 },
 }}
 {...props}
 />
 );
}

export { Toaster, toast };
export type { ToasterProps };
