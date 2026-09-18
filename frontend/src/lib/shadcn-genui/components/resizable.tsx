"use client";

import {
  ResizableHandle,
  ResizablePanel as ShadcnResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

function pct(value: number): string {
  return value + "%";
}

const ResizablePanelSchema = z.object({
  content: z.array(z.any()).default([]),
});

export const ResizablePanel = defineComponent({
  name: "ResizablePanel",
  props: ResizablePanelSchema,
  description: "One pane of a Resizable. content: nodes rendered inside the pane.",
  component: () => null,
});

const ResizableSchema = z.object({
  panels: z.array(ResizablePanel.ref).default([]),
  direction: z.enum(["horizontal", "vertical"]).optional(),
  initialRatio: z.number().optional(),
});

export const Resizable = defineComponent({
  name: "Resizable",
  props: ResizableSchema,
  description:
    'Side-by-side evidence comparison with a draggable divider. panels: ResizablePanel[]; direction: "horizontal" | "vertical"; initialRatio: first-pane percent (2 panes only). Use instead of stacking two tables.',
  component: ({ props, renderNode }) => {
    const panels = (props.panels ?? []).filter((panel) => panel != null);
    const direction = props.direction ?? "horizontal";
    const even = panels.length > 0 ? 100 / panels.length : 100;
    const first = panels.length === 2 && props.initialRatio ? props.initialRatio : even;
    const rest = (100 - first) / Math.max(panels.length - 1, 1);

    return (
      <ResizablePanelGroup
        orientation={direction}
        className="my-1 min-h-[12rem] rounded-md border"
        style={direction === "vertical" ? { height: "16rem" } : undefined}
      >
        {panels.map((panel, i) => (
          <React.Fragment key={i}>
            {i > 0 && <ResizableHandle withHandle />}
            <ShadcnResizablePanel defaultSize={pct(i === 0 ? first : rest)}>
              <div className="h-full space-y-3 overflow-auto p-3">
                {renderNode(panel?.props?.content)}
              </div>
            </ShadcnResizablePanel>
          </React.Fragment>
        ))}
      </ResizablePanelGroup>
    );
  },
});
