// ── MedPat mark ───────────────────────────────────────────────────────
// Abstract: instrument frame + ECG trace + molecular nodes. Pure SVG.

export function Logo({
  size = 28,
  className,
  accent = "currentColor",
}: {
  size?: number;
  className?: string;
  accent?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
      role="img"
      aria-label="MedPat"
    >
      <rect x="1" y="1" width="30" height="30" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5 16h4.4l2.1-5.6 4.4 11.2 2.1-5.6H27" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx="9.4" cy="16" r="1.7" fill="currentColor" />
      <circle cx="13.5" cy="10.4" r="1.7" fill="currentColor" />
      <circle cx="15.9" cy="21.6" r="1.7" fill={accent} />
      <circle cx="23.6" cy="16" r="1.4" fill="currentColor" />
    </svg>
  );
}
