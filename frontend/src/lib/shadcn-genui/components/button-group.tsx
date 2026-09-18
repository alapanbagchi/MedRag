"use client";

import { ButtonGroup as ShadcnButtonGroup } from "@/components/ui/button-group";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

import { Button } from "./button";

const ButtonGroupSchema = z.object({
  buttons: z.array(Button.ref).default([]),
});

export const ButtonGroup = defineComponent({
  name: "ButtonGroup",
  props: ButtonGroupSchema,
  description:
    'Group related actions without spacing, e.g. "Include / Exclude / Maybe" as one control. buttons: Button[].',
  component: ({ props, renderNode }) => (
    <ShadcnButtonGroup>{renderNode(props.buttons)}</ShadcnButtonGroup>
  ),
});
