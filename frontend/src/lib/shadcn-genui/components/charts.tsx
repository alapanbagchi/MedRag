"use client";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { defineComponent } from "@openuidev/react-lang";
import {
  Area,
  Bar,
  CartesianGrid,
  Cell,
  Label,
  LabelList,
  Legend,
  Line,
  Pie,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadialBar,
  AreaChart as RechartsAreaChart,
  BarChart as RechartsBarChart,
  LineChart as RechartsLineChart,
  PieChart as RechartsPieChart,
  RadarChart as RechartsRadarChart,
  RadialBarChart as RechartsRadialBarChart,
  ScatterChart as RechartsScatterChart,
  Scatter,
  XAxis,
  YAxis,
} from "recharts";
import * as React from "react";
import { z } from "zod";

import { buildChartData, buildSliceData, hasAllProps } from "../helpers";
import { ChartFrame } from "./chart-frame";

const COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

/** Shared chart chrome. The model passes one object instead of trailing optionals. */
const metaSchema = z
  .object({
    title: z.string().optional(),
    caption: z.string().optional(),
    unit: z.string().optional(),
    legend: z.boolean().optional(),
    showValues: z.boolean().optional(),
  })
  .optional();

type Meta = {
  title?: string;
  caption?: string;
  unit?: string;
  legend?: boolean;
  showValues?: boolean;
};

function buildConfig(keys: string[]): ChartConfig {
  const config: ChartConfig = {};
  keys.forEach((key, i) => {
    config[key] = { label: key, color: COLORS[i % COLORS.length] };
  });
  return config;
}

function getSeriesKeys(data: Record<string, string | number>[]): string[] {
  if (!data.length) return [];
  return Object.keys(data[0]).filter((k) => k !== "category");
}

/** "Concentration (mg/mL)" — the value axis always carries its unit. */
function axisLabel(label?: string, unit?: string): string {
  if (label && unit) return label + " (" + unit + ")";
  if (label) return label;
  return unit ? "(" + unit + ")" : "";
}

function CartesianChrome({
  xLabel,
  yValue,
  legend,
}: {
  xLabel?: string;
  yValue?: string;
  legend?: boolean;
}) {
  return (
    <>
      <CartesianGrid vertical={false} />
      <XAxis dataKey="category" tickLine={false} axisLine={false} tickMargin={8}>
        {xLabel && <Label value={xLabel} position="insideBottom" offset={-14} />}
      </XAxis>
      <YAxis tickLine={false} axisLine={false} tickMargin={8}>
        {yValue && (
          <Label value={yValue} angle={-90} position="insideLeft" style={{ textAnchor: "middle" }} />
        )}
      </YAxis>
      <ChartTooltip content={<ChartTooltipContent />} />
      {legend !== false && <Legend verticalAlign="top" height={28} />}
    </>
  );
}

// ── Virtual sub-components ──

const SeriesSchema = z.object({
  category: z.string(),
  values: z.array(z.number()),
});

export const Series = defineComponent({
  name: "Series",
  props: SeriesSchema,
  description: "One named data series with values matching labels.",
  component: () => null,
});

const SliceSchema = z.object({
  category: z.string(),
  value: z.number(),
});

export const Slice = defineComponent({
  name: "Slice",
  props: SliceSchema,
  description: "A single slice in a PieChart or RadialChart.",
  component: () => null,
});

const PointSchema = z.object({
  x: z.number(),
  y: z.number(),
  label: z.string().optional(),
});

export const Point = defineComponent({
  name: "Point",
  props: PointSchema,
  description: "A single data point in a ScatterChart series.",
  component: () => null,
});

const ScatterSeriesSchema = z.object({
  category: z.string(),
  points: z.array(Point.ref),
});

export const ScatterSeries = defineComponent({
  name: "ScatterSeries",
  props: ScatterSeriesSchema,
  description: "Named scatter series with Point references.",
  component: () => null,
});

// ── BarChart ──

export const BarChartCondensed = defineComponent({
  name: "BarChart",
  props: z.object({
    labels: z.array(z.string()),
    series: z.array(SeriesSchema),
    variant: z.enum(["grouped", "stacked"]).optional(),
    xLabel: z.string().optional(),
    yLabel: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "Vertical bars comparing a value across categories, or a stacked composition. Use to rank or compare discrete groups. Set meta.title, meta.unit, and both axis labels.",
  component: ({ props }) => {
    if (!hasAllProps(props as Record<string, unknown>, "labels", "series")) return null;
    const data = buildChartData(props.labels, props.series);
    if (!data.length) return null;
    const keys = getSeriesKeys(data);
    const config = buildConfig(keys);
    const m = (props.meta ?? {}) as Meta;
    const stacked = props.variant === "stacked";
    const yValue = axisLabel(props.yLabel, m.unit);

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <ChartContainer config={config} className="min-h-[220px] w-full">
          <RechartsBarChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 18 }}>
            <CartesianChrome xLabel={props.xLabel} yValue={yValue} legend={m.legend} />
            {keys.map((key, i) => (
              <Bar
                key={key}
                dataKey={key}
                fill={COLORS[i % COLORS.length]}
                radius={4}
                stackId={stacked ? "stack" : undefined}
                isAnimationActive={false}
              >
                {m.showValues && (
                  <LabelList dataKey={key} position="top" className="fill-foreground text-[10px]" />
                )}
              </Bar>
            ))}
          </RechartsBarChart>
        </ChartContainer>
      </ChartFrame>
    );
  },
});

// ── LineChart ──

export const LineChartCondensed = defineComponent({
  name: "LineChart",
  props: z.object({
    labels: z.array(z.string()),
    series: z.array(SeriesSchema),
    xLabel: z.string().optional(),
    yLabel: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "Line chart for a trend across an ordered axis, often time. Use when the shape of change matters more than individual points. Set meta.title, meta.unit, and both axis labels.",
  component: ({ props }) => {
    if (!hasAllProps(props as Record<string, unknown>, "labels", "series")) return null;
    const data = buildChartData(props.labels, props.series);
    if (!data.length) return null;
    const keys = getSeriesKeys(data);
    const config = buildConfig(keys);
    const m = (props.meta ?? {}) as Meta;
    const yValue = axisLabel(props.yLabel, m.unit);

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <ChartContainer config={config} className="min-h-[220px] w-full">
          <RechartsLineChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 18 }}>
            <CartesianChrome xLabel={props.xLabel} yValue={yValue} legend={m.legend} />
            {keys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2}
                dot={m.showValues ? { r: 3 } : false}
                isAnimationActive={false}
              />
            ))}
          </RechartsLineChart>
        </ChartContainer>
      </ChartFrame>
    );
  },
});

// ── AreaChart ──

export const AreaChartCondensed = defineComponent({
  name: "AreaChart",
  props: z.object({
    labels: z.array(z.string()),
    series: z.array(SeriesSchema),
    xLabel: z.string().optional(),
    yLabel: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "Filled area chart emphasising magnitude or cumulative volume over an ordered axis. Use for totals/volume, not precise comparisons. Set meta.title, meta.unit, and both axis labels.",
  component: ({ props }) => {
    if (!hasAllProps(props as Record<string, unknown>, "labels", "series")) return null;
    const data = buildChartData(props.labels, props.series);
    if (!data.length) return null;
    const keys = getSeriesKeys(data);
    const config = buildConfig(keys);
    const m = (props.meta ?? {}) as Meta;
    const yValue = axisLabel(props.yLabel, m.unit);

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <ChartContainer config={config} className="min-h-[220px] w-full">
          <RechartsAreaChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 18 }}>
            <CartesianChrome xLabel={props.xLabel} yValue={yValue} legend={m.legend} />
            {keys.map((key, i) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                fill={COLORS[i % COLORS.length]}
                stroke={COLORS[i % COLORS.length]}
                fillOpacity={0.2}
                isAnimationActive={false}
              />
            ))}
          </RechartsAreaChart>
        </ChartContainer>
      </ChartFrame>
    );
  },
});

// ── PieChart ──

export const PieChartComponent = defineComponent({
  name: "PieChart",
  props: z.object({
    slices: z.array(SliceSchema),
    donut: z.boolean().optional(),
    meta: metaSchema,
  }),
  description:
    "Pie or donut for parts of a whole at one point in time (2-7 slices). Use a bar chart or Table when the reader must compare exact values. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const data = buildSliceData(props.slices);
    if (!data.length) return null;
    const config = buildConfig(data.map((d) => d.category as string));
    const m = (props.meta ?? {}) as Meta;
    const showValues = m.showValues !== false;

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <ChartContainer
          config={config}
          className="mx-auto aspect-square max-h-[300px] min-h-[220px] w-full"
        >
          <RechartsPieChart>
            <ChartTooltip content={<ChartTooltipContent nameKey="category" />} />
            {m.legend !== false && <Legend verticalAlign="bottom" height={28} />}
            <Pie
              data={data}
              dataKey="value"
              nameKey="category"
              innerRadius={props.donut ? "52%" : 0}
              isAnimationActive={false}
              label={
                showValues
                  ? (entry: { name?: string; percent?: number }) =>
                      String(entry.name ?? "") + " " + Math.round((entry.percent ?? 0) * 100) + "%"
                  : false
              }
              labelLine={showValues}
            >
              {data.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
          </RechartsPieChart>
        </ChartContainer>
      </ChartFrame>
    );
  },
});

// ── RadarChart ──

export const RadarChartComponent = defineComponent({
  name: "RadarChart",
  props: z.object({
    labels: z.array(z.string()),
    series: z.array(SeriesSchema),
    meta: metaSchema,
  }),
  description:
    "Radar/spider comparing entities across the same 3-8 dimensions on one 0-100 scale. Never mix units on a radar. Set meta.title and meta.unit.",
  component: ({ props }) => {
    if (!hasAllProps(props as Record<string, unknown>, "labels", "series")) return null;
    const data = buildChartData(props.labels, props.series);
    if (!data.length) return null;
    const keys = getSeriesKeys(data);
    const config = buildConfig(keys);
    const m = (props.meta ?? {}) as Meta;

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <ChartContainer
          config={config}
          className="mx-auto aspect-square max-h-[300px] min-h-[220px] w-full"
        >
          <RechartsRadarChart data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="category" />
            <ChartTooltip content={<ChartTooltipContent />} />
            {m.legend !== false && <Legend verticalAlign="top" height={28} />}
            {keys.map((key, i) => (
              <Radar
                key={key}
                dataKey={key}
                fill={COLORS[i % COLORS.length]}
                fillOpacity={0.3}
                stroke={COLORS[i % COLORS.length]}
                isAnimationActive={false}
              />
            ))}
          </RechartsRadarChart>
        </ChartContainer>
      </ChartFrame>
    );
  },
});

// ── RadialChart ──

export const RadialChartComponent = defineComponent({
  name: "RadialChart",
  props: z.object({
    slices: z.array(SliceSchema),
    meta: metaSchema,
  }),
  description:
    "Radial bar (concentric rings) for a small ranked set on independent scales or progress per category. Use a bar chart when values share one scale. Set meta.title and meta.unit.",
  component: ({ props }) => {
    const data = buildSliceData(props.slices);
    if (!data.length) return null;
    const colored = data.map((d, i) => ({ ...d, fill: COLORS[i % COLORS.length] }));
    const config = buildConfig(data.map((d) => d.category as string));
    const m = (props.meta ?? {}) as Meta;

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <ChartContainer
          config={config}
          className="mx-auto aspect-square max-h-[300px] min-h-[220px] w-full"
        >
          <RechartsRadialBarChart data={colored} innerRadius={30} outerRadius={110}>
            <PolarAngleAxis type="number" domain={[0, "dataMax"]} tick={false} />
            <ChartTooltip content={<ChartTooltipContent nameKey="category" />} />
            {m.legend !== false && <Legend verticalAlign="bottom" height={28} />}
            <RadialBar dataKey="value" isAnimationActive={false} label />
          </RechartsRadialBarChart>
        </ChartContainer>
      </ChartFrame>
    );
  },
});

// ── ScatterChart ──

export const ScatterChartComponent = defineComponent({
  name: "ScatterChart",
  props: z.object({
    series: z.array(ScatterSeriesSchema),
    xLabel: z.string().optional(),
    yLabel: z.string().optional(),
    meta: metaSchema,
  }),
  description:
    "Scatter relating two numeric variables, one point per study. Use to show correlation, clusters, or effect against a baseline. Label both axes and their units in meta.",
  component: ({ props }) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const seriesArr = ((props.series ?? []) as any[]).map((s) => ({
      category: String(s?.props?.category ?? ""),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      points: ((s?.props?.points ?? []) as any[]).map((p: any) => ({
        x: Number(p?.props?.x ?? 0),
        y: Number(p?.props?.y ?? 0),
      })),
    }));
    const config = buildConfig(seriesArr.map((s) => s.category));
    const m = (props.meta ?? {}) as Meta;
    const yValue = axisLabel(props.yLabel, m.unit);

    return (
      <ChartFrame title={m.title} caption={m.caption}>
        <ChartContainer config={config} className="min-h-[220px] w-full">
          <RechartsScatterChart margin={{ left: 8, right: 8, top: 8, bottom: 18 }}>
            <CartesianGrid />
            <XAxis
              type="number"
              dataKey="x"
              name={props.xLabel ?? "x"}
              tickLine={false}
              tickMargin={8}
            >
              {props.xLabel && <Label value={props.xLabel} position="insideBottom" offset={-14} />}
            </XAxis>
            <YAxis
              type="number"
              dataKey="y"
              name={props.yLabel ?? "y"}
              tickLine={false}
              tickMargin={8}
            >
              {yValue && (
                <Label value={yValue} angle={-90} position="insideLeft" style={{ textAnchor: "middle" }} />
              )}
            </YAxis>
            <ChartTooltip content={<ChartTooltipContent />} />
            {m.legend !== false && <Legend verticalAlign="top" height={28} />}
            {seriesArr.map((s, i) => (
              <Scatter
                key={s.category}
                name={s.category}
                data={s.points}
                fill={COLORS[i % COLORS.length]}
                isAnimationActive={false}
              />
            ))}
          </RechartsScatterChart>
        </ChartContainer>
      </ChartFrame>
    );
  },
});
