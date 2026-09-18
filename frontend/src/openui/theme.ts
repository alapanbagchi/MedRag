/**
 * OpenUI semantic tokens mapped onto the app design system.
 *
 * Every value is a var(--app-token) reference, and the app redefines those
 * tokens under its .dark class, so ONE theme covers both modes: the values
 * resolve at use time against whichever scheme is active. That is also why
 * no darkTheme is passed - the reference says a shared override set is
 * reapplied over the dark defaults.
 *
 * Only semantic roles are mapped. Typography sizes and weights stay at the
 * OpenUI defaults on purpose: changing a primitive such as fontBody does not
 * regenerate the compound textBody* tokens, so a partial typography override
 * makes the scale inconsistent rather than branded. Fonts and radius do map,
 * because those tokens are consumed directly.
 */
import { createTheme } from "@openuidev/react-ui";

export const openuiTheme = createTheme({
  // Surfaces
  background: "var(--background)",
  foreground: "var(--foreground)",
  popoverBackground: "var(--popover)",
  elevated: "var(--card)",
  elevatedLight: "var(--card)",
  elevatedStrong: "var(--secondary)",
  sunk: "var(--muted)",
  sunkLight: "var(--card)",
  overlay: "var(--popover)",
  highlight: "var(--accent)",
  highlightSubtle: "var(--muted)",
  invertedBackground: "var(--foreground)",

  // Text
  textNeutralPrimary: "var(--foreground)",
  textNeutralSecondary: "var(--muted-foreground)",
  textNeutralTertiary: "color-mix(in oklab, var(--muted-foreground) 70%, transparent)",
  textNeutralLink: "var(--primary)",
  textBrand: "var(--brand)",
  textAccentPrimary: "var(--primary-foreground)",
  textInfoPrimary: "var(--brand)",
  textSuccessPrimary: "var(--success)",
  textAlertPrimary: "var(--warning)",
  textDangerPrimary: "var(--destructive)",

  // Status surfaces, tinted from the same hues so they follow the mode
  infoBackground: "color-mix(in oklab, var(--brand) 12%, transparent)",
  successBackground: "color-mix(in oklab, var(--success) 14%, transparent)",
  alertBackground: "color-mix(in oklab, var(--warning) 16%, transparent)",
  dangerBackground: "color-mix(in oklab, var(--destructive) 12%, transparent)",

  // Interactive
  interactiveAccentDefault: "var(--primary)",
  interactiveAccentHover: "color-mix(in oklab, var(--primary) 88%, var(--foreground))",
  interactiveAccentPressed: "color-mix(in oklab, var(--primary) 76%, var(--foreground))",
  interactiveAccentDisabled: "var(--muted)",
  interactiveDestructiveDefault: "var(--destructive)",
  interactiveDestructiveHover: "color-mix(in oklab, var(--destructive) 88%, var(--foreground))",
  interactiveDestructivePressed: "color-mix(in oklab, var(--destructive) 76%, var(--foreground))",

  // Borders
  borderDefault: "var(--border)",
  borderInteractive: "var(--border)",
  borderInteractiveEmphasis: "var(--ring)",
  borderAccent: "var(--primary)",
  borderInfo: "var(--brand)",
  borderSuccess: "var(--success)",
  borderAlert: "var(--warning)",
  borderDanger: "var(--destructive)",

  // Chat bubbles
  chatUserResponseBg: "var(--user-bubble-bg)",
  chatUserResponseText: "var(--user-bubble-ink)",

  // Radius: the shadcn convention off the app single --radius token
  radiusXs: "calc(var(--radius) - 6px)",
  radiusS: "calc(var(--radius) - 4px)",
  radiusM: "calc(var(--radius) - 2px)",
  radiusL: "var(--radius)",
  radiusXl: "calc(var(--radius) + 4px)",
  radius2xl: "calc(var(--radius) + 8px)",
  radius3xl: "calc(var(--radius) + 12px)",
  radiusFull: "9999px",

  // Fonts
  fontBody: "var(--font-sans)",
  fontHeading: "var(--font-sans)",
  fontLabel: "var(--font-sans)",
  fontNumbers: "var(--font-sans)",
  fontCode: "var(--font-mono)",
});
