"use client";

import { Search, AlertTriangle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { WorkerDetailDialog } from "./worker-detail-dialog";
import {
  formatRelativeTime,
  formatViolationType,
  getViolationColor,
} from "@/lib/utils/format";
import { getBrowserClient } from "@/lib/supabase/client";
import type {
  ViolationEvent,
  Worker,
} from "@/lib/supabase/types";
import type { WorkerWithLatest } from "@/lib/supabase/queries";
import { cn } from "@/lib/utils";

interface WorkersGridProps {
  workers: WorkerWithLatest[];
}

export function WorkersGrid({ workers }: WorkersGridProps) {
  const [query, setQuery] = useState("");
  const [openWorker, setOpenWorker] = useState<Worker | null>(null);
  const [openEvents, setOpenEvents] = useState<ViolationEvent[]>([]);
  const [loadingEvents, setLoadingEvents] = useState(false);

  const filtered = useMemo(() => {
    if (!query.trim()) return workers;
    const q = query.trim().toLowerCase();
    return workers.filter((w) => w.tracker_id.toLowerCase().includes(q));
  }, [workers, query]);

  // Lazy-load full event history when a worker is opened.
  useEffect(() => {
    let cancelled = false;
    if (!openWorker) {
      setOpenEvents([]);
      return;
    }
    setLoadingEvents(true);
    const sb = getBrowserClient();
    sb.from("events")
      .select("*")
      .eq("worker_id", openWorker.id)
      .order("started_at", { ascending: false })
      .then((r) => {
        if (cancelled) return;
        setOpenEvents((r.data ?? []) as unknown as ViolationEvent[]);
        setLoadingEvents(false);
      });
    return () => {
      cancelled = true;
    };
  }, [openWorker]);

  if (workers.length === 0) {
    return (
      <div className="rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel p-12 text-center text-sm text-text-muted">
        Henüz işçi kaydı yok. Live simulator çalıştırıldığında otomatik oluşturulur.
      </div>
    );
  }

  return (
    <>
      <div className="mb-4 flex items-center gap-2">
        <div className="relative flex-1 max-w-sm">
          <Search
            size={14}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted"
          />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="İşçi ID ara (W007...)"
            className="h-9 w-full rounded border border-[color:var(--border-subtle)] bg-bg-panel-2 pl-8 pr-3 text-sm text-text-primary outline-none focus:border-[color:var(--accent)] focus:ring-1 focus:ring-[color:var(--accent)]"
          />
        </div>
        <span className="text-xs text-text-muted">
          {filtered.length} / {workers.length}
        </span>
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel p-12 text-center text-sm text-text-muted">
          Bu sorguya uygun işçi bulunamadı.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((w) => (
            <WorkerCard
              key={w.id}
              worker={w}
              onOpen={() => setOpenWorker(w)}
            />
          ))}
        </div>
      )}

      <WorkerDetailDialog
        worker={openWorker}
        events={openEvents}
        loading={loadingEvents}
        onOpenChange={(open) => {
          if (!open) setOpenWorker(null);
        }}
      />
    </>
  );
}

function WorkerCard({
  worker,
  onOpen,
}: {
  worker: WorkerWithLatest;
  onOpen: () => void;
}) {
  const hasViolations = worker.total_violations > 0;
  const latest = worker.latest_event;
  return (
    <button
      onClick={onOpen}
      className={cn(
        "flex flex-col items-stretch text-left rounded-lg border-l-2 bg-bg-panel p-4 transition-colors hover:bg-bg-panel-2",
        "border border-[color:var(--border-subtle)]",
        hasViolations
          ? "border-l-[color:var(--danger)]"
          : "border-l-[color:var(--accent)]"
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="rounded bg-bg-panel-2 px-2.5 py-1 text-base font-bold tracking-wider text-text-primary">
          {worker.tracker_id}
        </span>
        <span
          className={cn(
            "rounded px-2 py-0.5 text-xs font-semibold",
            hasViolations
              ? "bg-[color:var(--danger-soft)] text-[color:var(--danger)]"
              : "bg-bg-panel-2 text-text-muted"
          )}
        >
          {worker.total_violations} ihlal
        </span>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-1 text-[11px]">
        <dt className="text-text-muted">İlk görülme</dt>
        <dd className="text-right text-text-primary">
          {worker.first_seen ? formatRelativeTime(worker.first_seen) : "—"}
        </dd>
        <dt className="text-text-muted">Son görülme</dt>
        <dd className="text-right text-text-primary">
          {worker.last_seen ? formatRelativeTime(worker.last_seen) : "—"}
        </dd>
      </dl>

      {latest ? (
        <div
          className="mt-3 flex items-center gap-2 rounded bg-bg-panel-2 px-2 py-1.5 text-xs"
          style={{ borderLeft: `2px solid ${getViolationColor(latest.violation_type).fg}` }}
        >
          <AlertTriangle
            size={12}
            style={{ color: getViolationColor(latest.violation_type).fg }}
          />
          <span style={{ color: getViolationColor(latest.violation_type).fg }}>
            {formatViolationType(latest.violation_type)}
          </span>
          <span className="ml-auto text-text-muted">
            {formatRelativeTime(latest.started_at)}
          </span>
        </div>
      ) : (
        <div className="mt-3 rounded bg-bg-panel-2 px-2 py-1.5 text-[11px] text-text-muted">
          İhlal kaydı yok
        </div>
      )}

      <span className="mt-3 self-end text-xs text-[color:var(--accent)] hover:underline">
        Detay →
      </span>
    </button>
  );
}
