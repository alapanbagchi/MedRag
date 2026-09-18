import { z } from "zod";

import { Alert } from "./components/alert";
import { AlertDialogBlock } from "./components/alert-dialog-block";
import { ShadcnBadgeComponent } from "./components/badge";
import { CalendarBlock } from "./components/calendar-block";
import { CodeBlock } from "./components/code-block";
import { DialogBlock } from "./components/dialog-block";
import { DrawerBlock } from "./components/drawer-block";
import { FollowUpBlock } from "./components/follow-up-block";
import { Image, ImageBlock } from "./components/image";
import { MarkDownRenderer } from "./components/markdown-renderer";
import { PaginationBlock } from "./components/pagination-block";
import { Progress } from "./components/progress";
import { Separator } from "./components/separator";
import { TextContent } from "./components/text-content";
import { Blockquote, Heading, InlineCode } from "./components/typography";

import {
  AreaChartCondensed,
  BarChartCondensed,
  LineChartCondensed,
  PieChartComponent,
  RadarChartComponent,
  RadialChartComponent,
  ScatterChartComponent,
} from "./components/charts";

import {
  FunnelChart,
  GaugeChart,
  HeatmapChart,
  SankeyChart,
  SunburstChart,
  TreemapChart,
  WaterfallChart,
} from "./components/advanced-charts";
import { MapChart } from "./components/map-chart";

import { Table } from "./components/table";
import { TagBlock } from "./components/tag";

import { Avatar } from "./components/avatar";
import { Buttons } from "./components/buttons";
import { CardHeader } from "./components/card-header";
import { Form } from "./components/form";

import { AspectRatio } from "./components/aspect-ratio";
import { Breadcrumb } from "./components/breadcrumb";
import { ButtonGroup } from "./components/button-group";
import { Collapsible } from "./components/collapsible";
import { Combobox } from "./components/combobox";
import { Command } from "./components/command";
import { ContextMenu } from "./components/context-menu";
import { DropdownMenu } from "./components/dropdown-menu";
import { Empty } from "./components/empty";
import { Field } from "./components/field";
import { HoverCard } from "./components/hover-card";
import { InputGroup } from "./components/input-group";
import { Kbd } from "./components/kbd";
import { Menubar } from "./components/menubar";
import { NavigationMenu } from "./components/navigation-menu";
import { Popover } from "./components/popover";
import { Resizable } from "./components/resizable";
import { ScrollArea } from "./components/scroll-area";
import { Sheet } from "./components/sheet";
import { Skeleton } from "./components/skeleton";
import { Sonner } from "./components/sonner";
import { Spinner } from "./components/spinner";
import { ToggleGroup } from "./components/toggle-group";
import { Tooltip } from "./components/tooltip";

export const ContentChildUnion = z.union([
  TextContent.ref,
  MarkDownRenderer.ref,
  CardHeader.ref,
  Alert.ref,
  ShadcnBadgeComponent.ref,
  Avatar.ref,
  CodeBlock.ref,
  Image.ref,
  ImageBlock.ref,
  Progress.ref,
  Separator.ref,
  BarChartCondensed.ref,
  LineChartCondensed.ref,
  AreaChartCondensed.ref,
  PieChartComponent.ref,
  RadarChartComponent.ref,
  RadialChartComponent.ref,
  ScatterChartComponent.ref,
  WaterfallChart.ref,
  SunburstChart.ref,
  TreemapChart.ref,
  FunnelChart.ref,
  SankeyChart.ref,
  GaugeChart.ref,
  HeatmapChart.ref,
  MapChart.ref,
  Table.ref,
  TagBlock.ref,
  Form.ref,
  Buttons.ref,
  Heading.ref,
  Blockquote.ref,
  InlineCode.ref,
  PaginationBlock.ref,
  DialogBlock.ref,
  AlertDialogBlock.ref,
  DrawerBlock.ref,
  CalendarBlock.ref,
  Command.ref,
  Combobox.ref,
  HoverCard.ref,
  Tooltip.ref,
  Popover.ref,
  Collapsible.ref,
  ToggleGroup.ref,
  Sheet.ref,
  Resizable.ref,
  ScrollArea.ref,
  Sonner.ref,
  Skeleton.ref,
  Spinner.ref,
  Empty.ref,
  Field.ref,
  Breadcrumb.ref,
  NavigationMenu.ref,
  DropdownMenu.ref,
  ContextMenu.ref,
  Menubar.ref,
  Kbd.ref,
  AspectRatio.ref,
  ButtonGroup.ref,
  InputGroup.ref,
]);

export const ChatContentChildUnion = z.union([...ContentChildUnion.options, FollowUpBlock.ref]);
