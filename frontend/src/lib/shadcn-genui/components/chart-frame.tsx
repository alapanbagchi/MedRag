"use client";

import * as React from "react";

/**
 * Shared chrome for every chart: a visible title that says what the chart
 * shows, the plot itself, and an optional caption (source, n, caveat).
 * Keeps charts informative instead of bare axes.
 */
export function ChartFrame({
  title,
  caption,
  children,
}: {
  title?: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="space-y-2">
      {title && (
        <figcaption className="text-foreground text-sm font-medium">{title}</figcaption>
      )}
      <div className="min-w-0">{children}</div>
      {caption && <p className="text-muted-foreground text-xs leading-relaxed">{caption}</p>}
    </figure>
  );
}
