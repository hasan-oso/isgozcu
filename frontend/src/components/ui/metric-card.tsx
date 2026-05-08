import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type MetricVariant = "default" | "danger";

interface MetricCardProps {
  label: string;
  value: string | number;
  variant?: MetricVariant;
  icon?: LucideIcon;
  hint?: string;
}

export function MetricCard({
  label,
  value,
  variant = "default",
  icon: Icon,
  hint,
}: MetricCardProps) {
  const isDanger = variant === "danger";
  return (
    <div
      className={cn(
        "flex flex-col justify-between rounded-lg border p-4 min-h-[112px]",
        isDanger
          ? "bg-danger-soft border-[color:var(--danger)] text-text-primary"
          : "bg-bg-panel border-[color:var(--border-subtle)]"
      )}
    >
      <div className="flex items-start justify-between">
        <span
          className={cn(
            "text-xs uppercase tracking-wider",
            isDanger ? "text-[color:var(--danger)] font-semibold" : "text-text-muted"
          )}
        >
          {label}
        </span>
        {Icon ? (
          <Icon
            size={18}
            className={cn(
              isDanger ? "text-[color:var(--danger)]" : "text-text-muted"
            )}
          />
        ) : null}
      </div>
      <div className="flex items-baseline gap-2 mt-3">
        <span
          className={cn(
            "text-3xl font-bold tabular-nums",
            isDanger ? "text-[color:var(--danger)]" : "text-text-primary"
          )}
        >
          {value}
        </span>
        {hint ? (
          <span className="text-xs text-text-muted">{hint}</span>
        ) : null}
      </div>
    </div>
  );
}
