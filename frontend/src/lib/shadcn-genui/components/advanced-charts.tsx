"use client";

import { EChart, type EChartOption } from "@/components/ui/echart";
import { defineComponent } from "@openuidev/react-lang";
import * as React from "react";
import { z } from "zod";

import { ChartFrame } from "./chart-frame";
import { cssVar } from "@/lib/theme";

const metaSchema = z
  .object({
    title: z.string().optional(),
    caption: z.string().optional(),
    unit: z.string().optional(),
    legend: z.boolean().optional(),
  })
  .optional();

type Meta = { title?: string; caption?: string; unit?: string; legend?: boolean };

type Virtual = { props: Record<string, unknown> };

function nodesOf(value: unknown): Virtual[] {
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is Virtual => !!x && typeof x === "object" && "props" in (x as object));
}

function text(value: unknown): string {
  if (value == null) return "";
  return typeof value === "string" ? value : String(value);
}

function num(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function fmt(value: unknown, unit?: string): string {
  const base = typeof value === "number" ? String(value) : text(value);
  return unit ? base + " " + unit : base;
}

function optionOf(value: Record<string, unknown>): EChartOption {
  return value as unknown as EChartOption;
}

// ── virtual item components ──

const WaterfallItemSchema = z.object({
  category: z.string(),
  value: z.number(),
  total: z.boolean().optional(),
});

export const WaterfallItem = defineComponent({
  name: "WaterfallItem",
  props: WaterfallItemSchema,
  description:
    "One step in a WaterfallChart. category: step label; value: signed change (negative decreases); total: true renders the running cumulative total bar.",
  component: () => null,
});

const SunburstNodeSchema = z.object({
  name: z.string(),
  value: z.number().optional(),
  parent: z.string().optional(),
});

export const SunburstNode = defineComponent({
  name: "SunburstNode",
  props: SunburstNodeSchema,
  description:
    "One node of a SunburstChart hierarchy. name: unique label; value: leaf size (omit for branch nodes); parent: parent node name (omit for a root).",
  component: () => null,
});

const TreemapNodeSchema = z.object({
  name: z.string(),
  value: z.number().optional(),
  parent: z.string().optional(),
});

export const TreemapNode = defineComponent({
  name: "TreemapNode",
  props: TreemapNodeSchema,
  description:
    "One node of a TreemapChart hierarchy. name: unique label; value: leaf size; parent: parent node name (omit for a root).",
  component: () => null,
});

const FunnelStageSchema = z.object({
  name: z.string(),
  value: z.number(),
});

export const FunnelStage = defineComponent({
  name: "FunnelStage",
  props: FunnelStageSchema,
  description: "One stage in a FunnelChart. name: stage label; value: stage count.",
  component: () => null,
});

const FlowNodeSchema = z.object({ name: z.string() });

export const FlowNode = defineComponent({
  name: "FlowNode",
  props: FlowNodeSchema,
  description: "One node (source or target) in a SankeyChart. name: unique label used by FlowLink.",
  component: () => null,
});

const FlowLinkSchema = z.object({
  source: z.string(),
  target: z.string(),
  value: z.number(),
});

export const FlowLink = defineComponent({
  name: "FlowLink",
  props: FlowLinkSchema,
  description: "One flow in a SankeyChart. source/target: FlowNode names; value: flow magnitude.",
  component: () => null,
});

const HeatCellSchema = z.object({
  x: z.string(),
  y: z.string(),
  value: z.number(),
});

export const HeatCell = defineComponent({
  name: "HeatCell",
  props: HeatCellSchema,
  description: "One cell in a HeatmapChart. x: column label; y: row label; value: cell magnitude.",
  component: () => null,
});

// ── hierarchy builder for sunburst / treemap ──

type Tree = { name: string; value?: number; children: Tree[] };

function buildTree(items: Virtual[]): Tree[] {
  const map = new Map<string, Tree>();
  const order: Tree[] = [];
  for (const item of items) {
    const name = text(item.props.name);
    if (!name) continue;
    const tree: Tree = { name, children: [] };
    if (item.props.value != null) tree.value = num(item.props.value);
    map.set(name, tree);
    order.push(tree);
  }
  const roots: Tree[] = [];
  for (const item of items) {
    const name = text(item.props.name);
    const parent = item.props.parent ? text(item.props.parent) : "";
    const tree = map.get(name);
    if (!tree) continue;
    const parentTree = parent ? map.get(parent) : undefined;
    if (parentTree && parentTree !== tree) parentTree.children.push(tree);
    else roots.push(tree);
  }
  return roots.length ? roots : order;
}

// ── WaterfallChart ──

export const WaterfallChart = defineComponent({
  name: "WaterfallChart",
  props: z.object({
    items: z.array(WaterfallItem.ref),
    xLabel: z.string().optional(),
    yLabel: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "Waterfall: how a starting value moves through signed increments to a final total. Use for additive effect decomposition. Mark the last bar total: true. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const items = nodesOf(props.items);
    if (!items.length) return null;
    const m = (props.meta ?? {}) as Meta;

    const categories: string[] = [];
    const base: (number | null)[] = [];
    const increase: (number | null)[] = [];
    const decrease: (number | null)[] = [];
    const total: (number | null)[] = [];
    let running = 0;

    for (const item of items) {
      const value = num(item.props.value);
      categories.push(text(item.props.category));
      if (item.props.total) {
        base.push(0);
        increase.push(null);
        decrease.push(null);
        total.push(running);
      } else if (value >= 0) {
        base.push(running);
        increase.push(value);
        decrease.push(null);
        total.push(null);
        running += value;
      } else {
        base.push(running + value);
        increase.push(null);
        decrease.push(-value);
        total.push(null);
        running += value;
      }
    }

    const yName = [props.yLabel, m.unit ? "(" + m.unit + ")" : ""].filter(Boolean).join(" ");
    const option = optionOf({
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        valueFormatter: (v: unknown) => fmt(v, m.unit),
      },
      legend: { show: m.legend !== false, data: ["Increase", "Decrease", "Total"], top: 0 },
      grid: { left: 8, right: 12, top: 36, bottom: props.xLabel ? 34 : 22, containLabel: true },
      xAxis: { type: "category", data: categories, name: props.xLabel },
      yAxis: { type: "value", name: yName },
      series: [
        { name: "Base", type: "bar", stack: "wf", silent: true, itemStyle: { color: "transparent" }, emphasis: { itemStyle: { color: "transparent" } }, tooltip: { show: false }, data: base },
        { name: "Increase", type: "bar", stack: "wf", itemStyle: { color: cssVar("--series-2", "#18ae95") }, data: increase },
        { name: "Decrease", type: "bar", stack: "wf", itemStyle: { color: cssVar("--series-4", "#e5484d") }, data: decrease },
        { name: "Total", type: "bar", stack: "wf", itemStyle: { color: cssVar("--series-1", "#4b8cf5") }, data: total },
      ],
    });

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={300} />
      </ChartFrame>
    );
  },
});

// ── SunburstChart ──

export const SunburstChart = defineComponent({
  name: "SunburstChart",
  props: z.object({ nodes: z.array(SunburstNode.ref), meta: metaSchema }),
  description:
    "Sunburst: a two-to-four level part-of-whole hierarchy as concentric rings. Use for nested categories; use a pie chart for a flat one-level split. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const data = buildTree(nodesOf(props.nodes));
    if (!data.length) return null;
    const m = (props.meta ?? {}) as Meta;
    const option = optionOf({
      tooltip: { trigger: "item", valueFormatter: (v: unknown) => fmt(v, m.unit) },
      series: [
        {
          type: "sunburst",
          data,
          radius: [0, "92%"],
          label: { rotate: "radial", fontSize: 10 },
          emphasis: { focus: "ancestor" },
          itemStyle: { borderColor: cssVar("--background", "#fff"), borderWidth: 1 },
        },
      ],
    });
    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={320} />
      </ChartFrame>
    );
  },
});

// ── TreemapChart ──

export const TreemapChart = defineComponent({
  name: "TreemapChart",
  props: z.object({ nodes: z.array(TreemapNode.ref), meta: metaSchema }),
  description:
    "Treemap: nested part-of-whole as area-proportional rectangles. Use when many categories or two levels must fit at once; pair with a Table for exact values. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const data = buildTree(nodesOf(props.nodes));
    if (!data.length) return null;
    const m = (props.meta ?? {}) as Meta;
    const option = optionOf({
      tooltip: { trigger: "item", valueFormatter: (v: unknown) => fmt(v, m.unit) },
      series: [
        {
          type: "treemap",
          data,
          leafDepth: 2,
          breadcrumb: { show: false },
          label: { show: true, fontSize: 11 },
          upperLabel: { show: true, height: 18 },
          itemStyle: { borderColor: cssVar("--background", "#fff"), borderWidth: 1 },
        },
      ],
    });
    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={320} />
      </ChartFrame>
    );
  },
});

// ── FunnelChart ──

export const FunnelChart = defineComponent({
  name: "FunnelChart",
  props: z.object({ stages: z.array(FunnelStage.ref), meta: metaSchema }),
  description:
    "Funnel: progressive attrition through ordered stages (screened > eligible > included > analysed). Use for any screening or selection flow. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const stages = nodesOf(props.stages);
    if (!stages.length) return null;
    const m = (props.meta ?? {}) as Meta;
    const option = optionOf({
      tooltip: { trigger: "item", valueFormatter: (v: unknown) => fmt(v, m.unit) },
      legend: { show: m.legend !== false, top: 0, data: stages.map((s) => text(s.props.name)) },
      series: [
        {
          type: "funnel",
          left: "8%",
          right: "8%",
          top: 36,
          bottom: 8,
          sort: "descending",
          gap: 2,
          label: { show: true, position: "inside", formatter: "{b}: {c}" },
          data: stages.map((s) => ({ name: text(s.props.name), value: num(s.props.value) })),
        },
      ],
    });
    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={300} />
      </ChartFrame>
    );
  },
});

// ── SankeyChart ──

export const SankeyChart = defineComponent({
  name: "SankeyChart",
  props: z.object({
    nodes: z.array(FlowNode.ref),
    links: z.array(FlowLink.ref),
    meta: metaSchema,
  }),
  description:
    "Sankey: how quantities flow between stages or groups. Use for movement/allocation, never simple category totals. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const nodes = nodesOf(props.nodes);
    const links = nodesOf(props.links);
    if (!nodes.length || !links.length) return null;
    const m = (props.meta ?? {}) as Meta;
    const option = optionOf({
      tooltip: { trigger: "item", valueFormatter: (v: unknown) => fmt(v, m.unit) },
      series: [
        {
          type: "sankey",
          left: 8,
          right: 8,
          top: 24,
          bottom: 8,
          nodeWidth: 14,
          nodeGap: 10,
          emphasis: { focus: "adjacency" },
          label: { fontSize: 11 },
          data: nodes.map((n) => ({ name: text(n.props.name) })),
          links: links.map((l) => ({
            source: text(l.props.source),
            target: text(l.props.target),
            value: num(l.props.value),
          })),
        },
      ],
    });
    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={320} />
      </ChartFrame>
    );
  },
});

// ── GaugeChart ──

export const GaugeChart = defineComponent({
  name: "GaugeChart",
  props: z.object({
    value: z.number(),
    max: z.number().optional(),
    label: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "Gauge: one headline number against a maximum (pooled estimate, certainty score, completion). Use only for a single KPI, never comparisons. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const m = (props.meta ?? {}) as Meta;
    const max = props.max && props.max > 0 ? props.max : 100;
    const detail = m.unit ? "{value} " + m.unit : "{value}";
    const option = optionOf({
      tooltip: { trigger: "item" },
      series: [
        {
          type: "gauge",
          min: 0,
          max,
          startAngle: 210,
          endAngle: -30,
          progress: { show: true, width: 16, itemStyle: { color: cssVar("--series-1", "#4b8cf5") } },
          axisLine: { lineStyle: { width: 16, color: [[1, cssVar("--chart-track", "#e8eef5")]] } },
          pointer: { show: true, length: "60%", width: 4 },
          axisTick: { show: false },
          splitLine: { length: 12, lineStyle: { color: cssVar("--chart-line", "#c9d4d9") } },
          axisLabel: { distance: 16, fontSize: 10 },
          detail: { valueAnimation: true, fontSize: 22, offsetCenter: [0, "72%"], formatter: detail },
          data: [{ value: num(props.value), name: props.label ?? "" }],
        },
      ],
    });
    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={280} />
      </ChartFrame>
    );
  },
});

// ── HeatmapChart ──

export const HeatmapChart = defineComponent({
  name: "HeatmapChart",
  props: z.object({
    cells: z.array(HeatCell.ref),
    xLabel: z.string().optional(),
    yLabel: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "Heatmap matrix: a value at every row x column combination (e.g. outcome x study). Use to expose patterns and gaps across two categorical axes. Always label both axes. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const cells = nodesOf(props.cells);
    if (!cells.length) return null;
    const m = (props.meta ?? {}) as Meta;
    const xCats: string[] = [];
    const yCats: string[] = [];
    const values: number[] = [];
    for (const cell of cells) {
      const x = text(cell.props.x);
      const y = text(cell.props.y);
      if (!xCats.includes(x)) xCats.push(x);
      if (!yCats.includes(y)) yCats.push(y);
      values.push(num(cell.props.value));
    }
    const option = optionOf({
      tooltip: { position: "top", valueFormatter: (v: unknown) => fmt(v, m.unit) },
      grid: { left: 8, right: 16, top: 16, bottom: props.xLabel ? 40 : 24, containLabel: true },
      xAxis: { type: "category", data: xCats, name: props.xLabel, splitArea: { show: true } },
      yAxis: { type: "category", data: yCats, name: props.yLabel, splitArea: { show: true } },
      visualMap: {
        min: Math.min.apply(null, values),
        max: Math.max.apply(null, values),
        calculable: true,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        inRange: {
          color: [
            cssVar("--chart-heat-low", "#e8f3f1"),
            cssVar("--series-2", "#18ae95"),
            cssVar("--chart-heat-green-high", "#0b6b5c"),
          ],
        },
      },
      series: [
        {
          type: "heatmap",
          data: cells.map((cell) => [text(cell.props.x), text(cell.props.y), num(cell.props.value)]),
          label: { show: true, fontSize: 10 },
        },
      ],
    });
    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <EChart option={option} height={300} />
      </ChartFrame>
    );
  },
});
