"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";

const RANGES: { value: string; label: string }[] = [
  { value: "today", label: "Bugün" },
  { value: "week", label: "Bu Hafta" },
  { value: "month", label: "Bu Ay" },
  { value: "30d", label: "Son 30 Gün" },
];

export function RangeTabs({ current }: { current: string }) {
  const params = useSearchParams();
  return (
    <div className="inline-flex rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel p-1">
      {RANGES.map((r) => {
        const next = new URLSearchParams(params.toString());
        next.set("range", r.value);
        const active = current === r.value;
        return (
          <Link
            key={r.value}
            href={`/raporlar?${next.toString()}`}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              active
                ? "bg-accent text-white"
                : "text-text-muted hover:bg-bg-panel-2 hover:text-text-primary"
            )}
          >
            {r.label}
          </Link>
        );
      })}
    </div>
  );
}
