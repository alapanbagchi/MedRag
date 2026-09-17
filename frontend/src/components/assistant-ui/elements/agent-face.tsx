"use client";

/**
 * Agent face — a soft iridescent blob with hand-drawn ink features.
 *
 * Every agent gets a *unique* look: its accent colour anchors a random
 * multi-hue gradient, and a seeded RNG picks the body silhouette, gradient
 * direction, hair, eyes and mouth. The seed comes from the agent's stable
 * roster slot + id hash, so a face never changes while it is on screen and no
 * two agents land on the same colour.
 *
 * All motion is CSS (see `.agent-face*` in globals.css) so a list of faces
 * animates without JS cost, and reduced-motion users get a static face.
 */

import { useId, type CSSProperties, type ReactNode } from "react";

/** Ink used for every facial stroke. */
const INK = "#17171c";

type EyeKind = "dots" | "happy" | "sleepy" | "wink" | "line" | "side";
type MouthKind = "smile" | "open" | "wave" | "tongue" | "flat" | "o" | "grin";

const EYE_KINDS: readonly EyeKind[] = ["dots", "happy", "sleepy", "wink", "line", "side"];
const MOUTH_KINDS: readonly MouthKind[] = [
  "smile",
  "open",
  "wave",
  "tongue",
  "flat",
  "o",
  "grin",
];

/** Crown strokes, from bald to a full wavy fringe. */
const HAIR: readonly (readonly string[])[] = [
  [],
  ["M32 18 Q36 11 41 16 Q46 10 51 16 Q56 10 61 16", "M62 18 Q67 12 72 17"],
  ["M39 16 Q44 9 49 15 Q54 9 59 15"],
  ["M36 19 C44 10 55 9 63 16", "M60 19 Q65 13 70 18"],
  ["M34 18 Q40 10 46 17", "M53 16 Q59 9 65 16"],
  ["M31 20 C35 12 43 10 49 15", "M50 13 C56 8 65 9 70 15", "M63 18 Q68 13 73 18"],
  ["M40 15 Q45 8 50 14"],
  ["M30 17 Q35 10 41 15", "M44 14 Q50 8 56 14", "M59 16 Q64 11 69 16"],
  ["M33 19 C39 11 48 10 54 16"],
  ["M37 17 C43 9 54 9 60 15", "M58 18 Q64 13 69 18"],
];

export type FaceMood = "idle" | "talking";

/** Tiny deterministic PRNG (mulberry32) so a seed always yields one face. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

/** Minimal hex → HSL so a colour can be pushed into neighbouring hues. */
function hexToHsl(hex: string): { h: number; s: number; l: number } {
  let value = hex.replace("#", "");
  if (value.length === 3) value = value.split("").map((c) => c + c).join("");
  const num = Number.parseInt(value, 16);
  const r = ((num >> 16) & 255) / 255;
  const g = ((num >> 8) & 255) / 255;
  const b = (num & 255) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
  }
  return { h, s: s * 100, l: l * 100 };
}

function hsl(h: number, s: number, l: number): string {
  const hue = ((h % 360) + 360) % 360;
  return `hsl(${hue.toFixed(1)} ${clamp(s, 0, 100).toFixed(1)}% ${clamp(l, 0, 100).toFixed(1)}%)`;
}

function Eyes({
  kind,
  cxL,
  cxR,
  cy,
  r,
}: {
  kind: EyeKind;
  cxL: number;
  cxR: number;
  cy: number;
  r: number;
}) {
  const arc = (cx: number) =>
    `M${(cx - r - 1.6).toFixed(1)} ${(cy + 1).toFixed(1)} Q${cx.toFixed(1)} ${(cy - r - 1.8).toFixed(1)} ${(cx + r + 1.6).toFixed(1)} ${(cy + 1).toFixed(1)}`;
  const cup = (cx: number) =>
    `M${(cx - r - 1.6).toFixed(1)} ${(cy - 1).toFixed(1)} Q${cx.toFixed(1)} ${(cy + r + 1.8).toFixed(1)} ${(cx + r + 1.6).toFixed(1)} ${(cy - 1).toFixed(1)}`;
  const line = (cx: number) =>
    `M${(cx - r - 0.8).toFixed(1)} ${cy.toFixed(1)} L${(cx + r + 0.8).toFixed(1)} ${cy.toFixed(1)}`;
  // Every eye is wrapped in `.agent-face__eye` so each one blinks.
  const blink = (child: ReactNode, key: string) => (
    <g className="agent-face__eye" key={key}>
      {child}
    </g>
  );
  const dot = (cx: number, key: string) =>
    blink(<circle cx={cx} cy={cy} r={r * 0.85} fill={INK} stroke="none" />, key);

  switch (kind) {
    case "happy":
      return (
        <>
          {blink(<path d={arc(cxL)} />, "l")}
          {blink(<path d={arc(cxR)} />, "r")}
        </>
      );
    case "sleepy":
      return (
        <>
          {blink(<path d={cup(cxL)} />, "l")}
          {blink(<path d={cup(cxR)} />, "r")}
        </>
      );
    case "wink":
      return (
        <>
          {dot(cxL, "l")}
          {blink(<path d={arc(cxR)} />, "r")}
        </>
      );
    case "line":
      return (
        <>
          {blink(<path d={line(cxL)} />, "l")}
          {blink(<path d={line(cxR)} />, "r")}
        </>
      );
    case "side":
      return (
        <>
          {blink(
            <path
              d={`M${(cxL - r).toFixed(1)} ${(cy - 2.6).toFixed(1)} L${(cxL + r * 0.6).toFixed(1)} ${(cy + 2.6).toFixed(1)}`}
            />,
            "l",
          )}
          {blink(
            <path
              d={`M${(cxR - r * 0.6).toFixed(1)} ${(cy - 2.6).toFixed(1)} L${(cxR + r).toFixed(1)} ${(cy + 2.6).toFixed(1)}`}
            />,
            "r",
          )}
        </>
      );
    default:
      return (
        <>
          {dot(cxL, "l")}
          {dot(cxR, "r")}
        </>
      );
  }
}

function Mouth({ kind, y, talking }: { kind: MouthKind; y: number; talking: boolean }) {
  if (talking) {
    return (
      <ellipse
        cx={50}
        cy={y + 1.5}
        rx={4.2}
        ry={3.4}
        fill={INK}
        stroke="none"
        className="agent-face__mouth-open"
      />
    );
  }
  // Idle mouths still move (gentle chatter) so every face feels alive.
  return <g className="agent-face__mouth">{mouthShape(kind, y)}</g>;
}

function mouthShape(kind: MouthKind, y: number): ReactNode {
  switch (kind) {
    case "open":
      return <ellipse cx={50} cy={y + 1.5} rx={3.8} ry={3} fill={INK} stroke="none" />;
    case "wave":
      return (
        <path
          d={`M43.5 ${(y + 2).toFixed(1)} Q46.5 ${(y - 2).toFixed(1)} 50 ${(y + 2).toFixed(1)} Q53.5 ${(y + 6).toFixed(1)} 56.5 ${(y + 2).toFixed(1)}`}
        />
      );
    case "flat":
      return <path d={`M45.5 ${(y + 1).toFixed(1)} L54.5 ${(y + 1).toFixed(1)}`} />;
    case "o":
      return <circle cx={50} cy={y + 1} r={2.6} fill={INK} stroke="none" />;
    case "grin":
      return (
        <>
          <path d={`M43 ${y.toFixed(1)} Q50 ${(y + 7).toFixed(1)} 57 ${y.toFixed(1)}`} />
          <path d={`M43 ${y.toFixed(1)} l-1.6 -1.6`} />
          <path d={`M57 ${y.toFixed(1)} l1.6 -1.6`} />
        </>
      );
    case "tongue":
      return (
        <>
          <path d={`M44.5 ${(y - 0.5).toFixed(1)} Q50 ${(y + 5).toFixed(1)} 55.5 ${(y - 0.5).toFixed(1)}`} />
          <path
            d={`M47.6 ${(y + 2.9).toFixed(1)} Q50 ${(y + 7.1).toFixed(1)} 52.4 ${(y + 2.9).toFixed(1)} Z`}
            fill={INK}
            stroke="none"
          />
        </>
      );
    default:
      return <path d={`M45 ${y.toFixed(1)} Q50 ${(y + 5.5).toFixed(1)} 55 ${y.toFixed(1)}`} />;
  }
}

export function AgentFace({
  color,
  size = 52,
  mood = "idle",
  seed = 0,
  className,
  style,
}: {
  color: string;
  size?: number;
  mood?: FaceMood;
  /** Stable per-agent seed: same seed → same face, different seed → new face. */
  seed?: number;
  className?: string;
  style?: CSSProperties;
}) {
  // Gradient ids must be unique per instance or every face shares the first.
  const gradientId = `agent-grad-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const rand = mulberry32((seed || 1) >>> 0);
  const base = hexToHsl(color);
  const talking = mood === "talking" && size >= 32;

  // Body silhouette: rounded blob with a random corner radius, aspect and tilt.
  const inset = 3 + rand() * 4;
  const inner = 100 - inset * 2;
  const width = inner * (0.93 + rand() * 0.1);
  const height = inner * (0.93 + rand() * 0.1);
  const rx = Math.min(26 + rand() * 20, width / 2, height / 2);
  const tilt = (rand() - 0.5) * 14;

  // Gradient: random axis, then the agent colour flanked by two bent hues.
  const angle = rand() * Math.PI * 2;
  const dx = Math.cos(angle) * 50;
  const dy = Math.sin(angle) * 50;
  const dir = {
    x1: `${(50 - dx).toFixed(1)}%`,
    y1: `${(50 - dy).toFixed(1)}%`,
    x2: `${(50 + dx).toFixed(1)}%`,
    y2: `${(50 + dy).toFixed(1)}%`,
  };
  const stops = [
    hsl(base.h - (18 + rand() * 30), clamp(base.s + rand() * 18 - 6, 42, 96), clamp(base.l + 12 + rand() * 12, 58, 93)),
    hsl(base.h + rand() * 10 - 5, base.s, base.l),
    hsl(base.h + 36 + rand() * 46, clamp(base.s + rand() * 18 - 4, 44, 95), clamp(base.l - (4 + rand() * 16), 38, 72)),
  ];
  const midStop = 44 + rand() * 14;

  // Face: placement, expression and hair all vary per seed.
  const eyeY = 43.5 + rand() * 5;
  const spread = 11 + rand() * 5;
  const eyeR = 2.1 + rand() * 0.9;
  const mouthY = 60.5 + rand() * 4.5;
  const eyeKind = EYE_KINDS[Math.floor(rand() * EYE_KINDS.length)]!;
  const mouthKind = MOUTH_KINDS[Math.floor(rand() * MOUTH_KINDS.length)]!;
  const hair = HAIR[Math.floor(rand() * HAIR.length)]!;
  const hairShift = (rand() - 0.5) * 5;

  // Unique per agent so faces bob/blink/chatter out of sync.
  const faceDelay = `${(((seed >>> 0) % 1300) / 1000).toFixed(2)}s`;

  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      className={className}
      style={{ ...style, "--face-delay": faceDelay } as CSSProperties}
      aria-hidden
      focusable="false"
    >
      <defs>
        <linearGradient id={gradientId} x1={dir.x1} y1={dir.y1} x2={dir.x2} y2={dir.y2}>
          <stop offset="0%" stopColor={stops[0]} />
          <stop offset={`${midStop.toFixed(1)}%`} stopColor={stops[1]} />
          <stop offset="100%" stopColor={stops[2]} />
        </linearGradient>
      </defs>
      <g className="agent-face">
        <g className="agent-face__bob">
          <rect
            x={50 - width / 2}
            y={50 - height / 2}
            width={width}
            height={height}
            rx={rx}
            fill={`url(#${gradientId})`}
            transform={`rotate(${tilt.toFixed(2)} 50 50)`}
            style={{ filter: "drop-shadow(0 2px 3px rgba(13,14,26,0.18))" }}
          />
          <g
            fill="none"
            stroke={INK}
            strokeWidth={2.4}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <g transform={hairShift ? `translate(0 ${hairShift.toFixed(1)})` : undefined}>
              {hair.map((d, i) => (
                <path key={i} d={d} />
              ))}
            </g>
            <g className="agent-face__eyes">
              <Eyes kind={eyeKind} cxL={50 - spread} cxR={50 + spread} cy={eyeY} r={eyeR} />
            </g>
            <Mouth kind={mouthKind} y={mouthY} talking={talking} />
          </g>
        </g>
      </g>
    </svg>
  );
}
