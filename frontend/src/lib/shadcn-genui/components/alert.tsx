"use client";

import { AlertDescription, AlertTitle, Alert as ShadcnAlert } from "@/components/ui/alert";
import { defineComponent } from "@openuidev/react-lang";
import { AlertCircle, CheckCircle2, Info, TriangleAlert } from "lucide-react";
import { z } from "zod";
import { CitationText } from "@/components/CitationMarkdown";

const AlertSchema = z.object({
  title: z.string(),
  description: z.string(),
  variant: z.enum(["default", "destructive", "info", "success", "warning"]).optional(),
});

const variantStyles: Record<string, string> = {
  info: "border-info/30 bg-info/10 text-foreground [&>svg]:text-info",
  success:
    "border-success/30 bg-success/10 text-foreground [&>svg]:text-success",
  warning:
    "border-warning/30 bg-warning/10 text-foreground [&>svg]:text-warning",
};

const iconMap = {
  default: null,
  destructive: AlertCircle,
  info: Info,
  success: CheckCircle2,
  warning: TriangleAlert,
};

export const Alert = defineComponent({
  name: "Alert",
  props: AlertSchema,
  description:
    'Alert banner with icon, title, and description. variant: "default" | "destructive" | "info" | "success" | "warning".',
  component: ({ props }) => {
    const v = props.variant ?? "default";
    const shadcnVariant = v === "destructive" ? "destructive" : "default";
    const extraClass = variantStyles[v] ?? "";
    const Icon = iconMap[v as keyof typeof iconMap];

    return (
      <ShadcnAlert variant={shadcnVariant} className={extraClass}>
        {Icon && <Icon className="size-4" />}
        <AlertTitle>
          <CitationText text={String(props.title ?? "")} />
        </AlertTitle>
        <AlertDescription>
          <CitationText text={String(props.description ?? "")} />
        </AlertDescription>
      </ShadcnAlert>
    );
  },
});
