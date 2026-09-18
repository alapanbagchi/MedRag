"use client";

import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/sonner";
import { defineComponent } from "@openuidev/react-lang";
import { BellIcon } from "lucide-react";
import * as React from "react";
import { z } from "zod";

const SonnerSchema = z.object({
  message: z.string(),
  variant: z.enum(["default", "success", "error", "warning", "info"]).optional(),
  label: z.string().optional(),
  description: z.string().optional(),
});

export const Sonner = defineComponent({
  name: "Sonner",
  props: SonnerSchema,
  description:
    'Transient toast for confirmations ("Citation copied", "Filter applied"). message: toast body; variant: tone; label: button text (default "Notify"); description: secondary line. Never use a permanent Alert for a transient confirmation.',
  component: ({ props }) => {
    const notify = () => {
      const message = String(props.message ?? "");
      const opts = props.description ? { description: String(props.description) } : undefined;
      switch (props.variant) {
        case "success":
          toast.success(message, opts);
          break;
        case "error":
          toast.error(message, opts);
          break;
        case "warning":
          toast.warning(message, opts);
          break;
        case "info":
          toast.info(message, opts);
          break;
        default:
          toast(message, opts);
      }
    };

    return (
      <Button variant="outline" size="sm" onClick={notify}>
        <BellIcon className="size-3.5" />
        {props.label ?? "Notify"}
      </Button>
    );
  },
});
