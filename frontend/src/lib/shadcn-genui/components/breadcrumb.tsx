"use client";

import {
  Breadcrumb as ShadcnBreadcrumb,
  BreadcrumbItem as ShadcnBreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

import { actionSchema } from "../action";
import { useGenuiAction } from "./use-genui-action";

const BreadcrumbItemSchema = z.object({
  label: z.string(),
  href: z.string().optional(),
  action: actionSchema,
});

export const BreadcrumbItem = defineComponent({
  name: "BreadcrumbItem",
  props: BreadcrumbItemSchema,
  description:
    "One step in a Breadcrumb. label: step text; href: link target; action: fired when a non-link step is clicked.",
  component: () => null,
});

const BreadcrumbSchema = z.object({
  items: z.array(BreadcrumbItem.ref).default([]),
});

export const Breadcrumb = defineComponent({
  name: "Breadcrumb",
  props: BreadcrumbSchema,
  description:
    "Research trail showing how the evidence set was narrowed (Question > Filtered evidence (12) > Included studies (5) > Synthesis). items: BreadcrumbItem[]; the last item renders as the current page.",
  component: ({ props }) => {
    const dispatch = useGenuiAction();
    const items = (props.items ?? []).filter((item) => item?.props?.label != null);

    return (
      <ShadcnBreadcrumb>
        <BreadcrumbList>
          {items.map((item, i) => {
            const label = String(item.props.label);
            const isLast = i === items.length - 1;
            return (
              <React.Fragment key={i}>
                <ShadcnBreadcrumbItem>
                  {isLast ? (
                    <BreadcrumbPage>{label}</BreadcrumbPage>
                  ) : item.props.href ? (
                    <BreadcrumbLink href={String(item.props.href)}>{label}</BreadcrumbLink>
                  ) : (
                    <button
                      type="button"
                      className="hover:text-foreground transition-colors"
                      onClick={() => dispatch(label, item.props.action)}
                    >
                      {label}
                    </button>
                  )}
                </ShadcnBreadcrumbItem>
                {!isLast && <BreadcrumbSeparator />}
              </React.Fragment>
            );
          })}
        </BreadcrumbList>
      </ShadcnBreadcrumb>
    );
  },
});
