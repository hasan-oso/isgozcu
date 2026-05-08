// Pulsing red dot + "CANLI" label.
import { cn } from "@/lib/utils";

interface LiveBadgeProps {
  className?: string;
  label?: string;
}

export function LiveBadge({ className, label = "CANLI" }: LiveBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-2 py-0.5",
        "bg-[color:var(--bg-panel-2)] text-xs font-semibold text-text-primary",
        className
      )}
    >
      <span className="relative inline-flex h-2 w-2">
        <span className="absolute inset-0 rounded-full bg-danger opacity-70 animate-pulse-slow" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-danger" />
      </span>
      {label}
    </span>
  );
}
