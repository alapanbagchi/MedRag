"use client";

import {
  InputGroup as ShadcnInputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

const InputGroupSchema = z.object({
  placeholder: z.string().optional(),
  addonStart: z.string().optional(),
  addonEnd: z.string().optional(),
});

export const InputGroup = defineComponent({
  name: "InputGroup",
  props: InputGroupSchema,
  description:
    "Input with addons, e.g. a search field with a leading filter label and a trailing hint. placeholder: input hint; addonStart/addonEnd: text on either side.",
  component: ({ props }) => (
    <ShadcnInputGroup>
      {props.addonStart && (
        <InputGroupAddon>
          <span>{props.addonStart}</span>
        </InputGroupAddon>
      )}
      <InputGroupInput placeholder={props.placeholder ?? "Search..."} />
      {props.addonEnd && (
        <InputGroupAddon align="inline-end">
          <span>{props.addonEnd}</span>
        </InputGroupAddon>
      )}
    </ShadcnInputGroup>
  ),
});
