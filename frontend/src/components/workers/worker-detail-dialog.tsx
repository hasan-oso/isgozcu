"use client";

import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  formatDuration,
  formatRelativeTime,
  formatTrTimestamp,
  formatViolationType,
  getViolationColor,
} from "@/lib/utils/format";
import type { ViolationEvent, Worker } from "@/lib/supabase/types";

interface WorkerDetailDialogProps {
  worker: Worker | null;
  events: ViolationEvent[];
  loading: boolean;
  onOpenChange: (open: boolean) => void;
}

function avgDuration(events: ViolationEvent[]): number {
  if (events.length === 0) return 0;
  return (
    events.reduce((acc, e) => acc + (e.duration_seconds || 0), 0) /
    events.length
  );
}

function mostCommonType(events: ViolationEvent[]): string | null {
  if (events.length === 0) return null;
  const counts: Record<string, number> = {};
  for (const e of events) {
    counts[e.violation_type] = (counts[e.violation_type] ?? 0) + 1;
  }
  return (
    Object.entries(counts).sort(([, a], [, b]) => b - a)[0]?.[0] ?? null
  );
}

export function WorkerDetailDialog({
  worker,
  events,
  loading,
  onOpenChange,
}: WorkerDetailDialogProps) {
  const open = worker !== null;
  const top = mostCommonType(events);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {worker ? (
          <>
            <DialogHeader>
              <DialogTitle className="text-base font-semibold text-text-primary">
                İşçi: {worker.tracker_id}
              </DialogTitle>
              <DialogDescription className="text-xs text-text-muted">
                {worker.id}
              </DialogDescription>
            </DialogHeader>
            <DialogBody>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Stat label="Toplam İhlal" value={String(worker.total_violations)} />
                <Stat
                  label="İlk Görülme"
                  value={
                    worker.first_seen
                      ? formatRelativeTime(worker.first_seen)
                      : "—"
                  }
                />
                <Stat
                  label="Son Görülme"
                  value={
                    worker.last_seen
                      ? formatRelativeTime(worker.last_seen)
                      : "—"
                  }
                />
                <Stat
                  label="Ort. Süre"
                  value={formatDuration(avgDuration(events))}
                />
              </div>

              {top ? (
                <div className="rounded bg-bg-panel-2 px-3 py-2 text-sm">
                  <span className="text-text-muted">En sık ihlal: </span>
                  <span
                    className="ml-1 inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-semibold"
                    style={{
                      background: getViolationColor(top).bg,
                      color: getViolationColor(top).fg,
                      border: `1px solid ${getViolationColor(top).border}`,
                    }}
                  >
                    {formatViolationType(top)}
                  </span>
                </div>
              ) : null}

              <div>
                <h3 className="mb-2 text-sm font-semibold text-text-primary">
                  Geçmiş İhlaller ({events.length})
                </h3>
                {loading ? (
                  <div className="rounded bg-bg-panel-2 p-3 text-center text-xs text-text-muted">
                    Yükleniyor…
                  </div>
                ) : events.length === 0 ? (
                  <div className="rounded bg-bg-panel-2 p-3 text-center text-xs text-text-muted">
                    Bu işçi için ihlal kaydı yok.
                  </div>
                ) : (
                  <ul className="flex flex-col gap-1.5 max-h-[300px] overflow-y-auto pr-1">
                    {events.map((e) => {
                      const color = getViolationColor(e.violation_type);
                      return (
                        <li
                          key={e.id}
                          className="flex items-center justify-between rounded border border-[color:var(--border-subtle)] bg-bg-panel-2 px-3 py-2 text-xs"
                        >
                          <div className="flex flex-col">
                            <span
                              className="font-semibold"
                              style={{ color: color.fg }}
                            >
                              {formatViolationType(e.violation_type)}
                            </span>
                            <span className="text-text-muted">
                              {formatTrTimestamp(e.started_at)} —{" "}
                              {formatDuration(e.duration_seconds)}
                            </span>
                          </div>
                          <span
                            className={
                              e.is_active
                                ? "rounded bg-[color:var(--danger)] px-2 py-0.5 text-[10px] font-semibold uppercase text-white"
                                : "rounded bg-[color:var(--accent)] px-2 py-0.5 text-[10px] font-semibold uppercase text-white"
                            }
                          >
                            {e.is_active ? "Aktif" : "Çözüldü"}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            </DialogBody>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-bg-panel-2 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-text-muted">
        {label}
      </div>
      <div className="mt-0.5 text-sm font-medium text-text-primary truncate">
        {value}
      </div>
    </div>
  );
}
