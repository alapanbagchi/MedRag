/** Centered ambient greeting + subtext (cards + composer live in the shell). */
export function EmptyState() {
  return (
    <div className="flex flex-col items-center text-center">
      <h1 className="font-greeting text-[36px] font-medium leading-tight tracking-tight text-[#0D0E1A] sm:text-[44px]">
        What can I do for you today?
      </h1>
      <p className="mt-3 max-w-md text-[14px] leading-6 text-muted-foreground">
        Use one of the example prompts below, or ask your own question to begin.
      </p>
    </div>
  );
}
