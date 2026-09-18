"use client";

import {
  BarChart,
  EffectScatterChart,
  FunnelChart,
  GaugeChart,
  HeatmapChart,
  LineChart,
  MapChart,
  PieChart,
  RadarChart,
  SankeyChart,
  ScatterChart,
  SunburstChart,
  TreemapChart,
} from "echarts/charts";
import {
  DatasetComponent,
  GeoComponent,
  GraphicComponent,
  GridComponent,
  LegendComponent,
  PolarComponent,
  RadarComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import * as React from "react";

import { clsx } from "clsx";

echarts.use([
  CanvasRenderer,
  BarChart,
  LineChart,
  PieChart,
  RadarChart,
  ScatterChart,
  EffectScatterChart,
  SunburstChart,
  TreemapChart,
  FunnelChart,
  SankeyChart,
  GaugeChart,
  HeatmapChart,
  MapChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  VisualMapComponent,
  GeoComponent,
  DatasetComponent,
  GraphicComponent,
  PolarComponent,
  RadarComponent,
]);

export type EChartOption = echarts.EChartsCoreOption;

export function EChart({
  option,
  className,
  height = 280,
}: {
  option: EChartOption;
  className?: string;
  height?: number | string;
}) {
  const hostRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<echarts.EChartsType | null>(null);

  React.useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const chart = echarts.init(host, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host);
    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  React.useEffect(() => {
    chartRef.current?.setOption(option, true);
  }, [option]);

  return (
    <div
      ref={hostRef}
      data-slot="echart"
      className={clsx("w-full", className)}
      style={{ height }}
    />
  );
}
