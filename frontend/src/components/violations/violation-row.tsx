import Image from "next/image";
import { formatDistanceToNow } from "date-fns";
import { tr } from "date-fns/locale";
import { AlertTriangle } from "lucide-react";
import type { ViolationEventWithWorker } from "@/lib/supabase/types";
import { cn } from "@/lib/utils";

interface ViolationRowProps {
  event: ViolationEventWithWorker;
  showSnapshot?: boolean;
}

function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}d ${s}s`;
}

function trackerLabel(ev: ViolationEventWithWorker): string {
  if (ev.workers?.tracker_id) return ev.workers.tracker_id;
  const meta = ev.metadata as { tracker_id?: string } | null;
  return meta?.tracker_id ?? `W?${ev.worker_id.slice(0, 4)}`;
}

export function ViolationRow({ event, showSnapshot = true }: ViolationRowProps) {
  const tracker = trackerLabel(event);
  const sinceStart = formatDistanceToNow(new Date(event.started_at), {
    locale: tr,
    addSuffix: true,
  });
  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-md border-l-2 p-2.5 transition-colors",
        "bg-bg-panel-2 hover:bg-[color:var(--border-subtle)]",
        event.is_active
          ? "border-l-[color:var(--danger)]"
          : "border-l-[color:var(--accent)]"
      )}
    >
      {showSnapshot ? (
        <div className="relative h-8 w-8 flex-shrink-0 overflow-hidden rounded bg-bg-page ring-1 ring-[color:var(--border-subtle)]">
          {event.snapshot_url ? (
            <Image
              src={event.snapshot_url}
              alt={event.violation_label_tr}
              fill
              sizes="32px"
              className="object-cover"
              unoptimized
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-text-muted">
              <AlertTriangle size={14} />
            </div>
          )}
        </div>
      ) : null}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "text-sm font-medium truncate",
              event.is_active
                ? "text-[color:var(--danger)]"
                : "text-text-primary"
            )}
          >
            {event.violation_label_tr}
          </span>
          <span className="text-[11px] uppercase tracking-wider text-text-muted">
            {tracker}
          </span>
        </div>
        <div className="text-[11px] text-text-muted">
          {fmtDuration(event.duration_seconds)} • {sinceStart}
        </div>
      </div>
      <span
        className={cn(
          "rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
          event.is_active
            ? "bg-[color:var(--danger)] text-white"
            : "bg-[color:var(--bg-page)] text-text-muted"
        )}
      >
        {event.is_active ? "Aktif" : "Bitti"}
      </span>
    </div>
  );
}
