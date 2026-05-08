"use client";

import Image from "next/image";
import { AlertTriangle } from "lucide-react";
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
  formatTrTimestamp,
  formatViolationType,
  getViolationColor,
} from "@/lib/utils/format";
import type { ViolationEventWithWorker } from "@/lib/supabase/types";

interface EventDetailDialogProps {
  event: ViolationEventWithWorker | null;
  onOpenChange: (open: boolean) => void;
}

function tracker(ev: ViolationEventWithWorker): string {
  if (ev.workers?.tracker_id) return ev.workers.tracker_id;
  const meta = ev.metadata as { tracker_id?: string } | null;
  return meta?.tracker_id ?? "—";
}

export function EventDetailDialog({ event, onOpenChange }: EventDetailDialogProps) {
  const open = event !== null;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {event ? (
          <>
            <DialogHeader>
              <DialogTitle className="text-base font-semibold text-text-primary">
                İhlal Detayı
              </DialogTitle>
              <DialogDescription className="text-xs text-text-muted">
                {event.id}
              </DialogDescription>
            </DialogHeader>
            <DialogBody>
              {event.snapshot_url ? (
                <div className="relative w-full overflow-hidden rounded border border-[color:var(--border-subtle)] bg-bg-page">
                  <Image
                    src={event.snapshot_url}
                    alt={event.violation_label_tr}
                    width={1200}
                    height={800}
                    className="h-auto w-full object-contain"
                    unoptimized
                  />
                </div>
              ) : (
                <div className="flex h-40 items-center justify-center rounded bg-bg-page text-text-muted">
                  <AlertTriangle size={28} />
                </div>
              )}

              <div className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
                <Field
                  label="Tür"
                  value={
                    <span
                      className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-semibold"
                      style={{
                        background: getViolationColor(event.violation_type).bg,
                        color: getViolationColor(event.violation_type).fg,
                        borderColor: getViolationColor(event.violation_type).border,
                        borderWidth: 1,
                      }}
                    >
                      {formatViolationType(event.violation_type)}
                    </span>
                  }
                />
                <Field label="İşçi" value={tracker(event)} />
                <Field
                  label="Durum"
                  value={
                    event.is_active ? (
                      <span className="rounded bg-[color:var(--danger)] px-2 py-0.5 text-xs font-semibold text-white">
                        Aktif
                      </span>
                    ) : (
                      <span className="rounded bg-[color:var(--accent)] px-2 py-0.5 text-xs font-semibold text-white">
                        Çözüldü
                      </span>
                    )
                  }
                />
                <Field label="Süre" value={formatDuration(event.duration_seconds)} />
                <Field
                  label="Başlangıç"
                  value={formatTrTimestamp(event.started_at)}
                />
                <Field
                  label="Bitiş"
                  value={
                    event.ended_at ? formatTrTimestamp(event.ended_at) : "—"
                  }
                />
                <Field
                  label="Güven"
                  value={`${(event.confidence_avg * 100).toFixed(0)}%`}
                />
                <Field label="Kamera ID" value={event.camera_id.slice(0, 8) + "…"} />
              </div>
            </DialogBody>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded bg-bg-panel-2 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-text-muted">
        {label}
      </div>
      <div className="mt-0.5 text-sm text-text-primary">{value}</div>
    </div>
  );
}
