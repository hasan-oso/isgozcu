"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { EventDetailDialog } from "./event-detail-dialog";
import {
  formatDuration,
  formatTrTimestamp,
  formatViolationType,
  getViolationColor,
} from "@/lib/utils/format";
import type { ViolationEventWithWorker } from "@/lib/supabase/types";
import { cn } from "@/lib/utils";

interface ViolationsTableProps {
  rows: ViolationEventWithWorker[];
  total: number;
  page: number;
  pageSize: number;
}

function tracker(ev: ViolationEventWithWorker): string {
  if (ev.workers?.tracker_id) return ev.workers.tracker_id;
  const meta = ev.metadata as { tracker_id?: string } | null;
  return meta?.tracker_id ?? "—";
}

export function ViolationsTable({
  rows,
  total,
  page,
  pageSize,
}: ViolationsTableProps) {
  const [openEvent, setOpenEvent] =
    useState<ViolationEventWithWorker | null>(null);
  const params = useSearchParams();
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const buildPageHref = (p: number) => {
    const next = new URLSearchParams(params.toString());
    if (p > 1) next.set("page", p.toString());
    else next.delete("page");
    return `/ihlaller?${next.toString()}`;
  };

  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel p-12 text-center text-sm text-text-muted">
        Bu kriterlere uygun ihlal bulunamadı.
      </div>
    );
  }

  return (
    <>
      <div className="overflow-hidden rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[color:var(--border-subtle)] text-left text-[11px] uppercase tracking-wider text-text-muted">
              <th className="px-3 py-2 w-12"></th>
              <th className="px-3 py-2">İşçi</th>
              <th className="px-3 py-2">Tür</th>
              <th className="px-3 py-2">Kamera</th>
              <th className="px-3 py-2">Başlangıç</th>
              <th className="px-3 py-2">Süre</th>
              <th className="px-3 py-2">Durum</th>
              <th className="px-3 py-2 text-right">İşlem</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const color = getViolationColor(row.violation_type);
              return (
                <tr
                  key={row.id}
                  className={cn(
                    "border-b border-[color:var(--border-subtle)] last:border-b-0 transition-colors hover:bg-bg-panel-2",
                    row.is_active &&
                      "border-l-2 border-l-[color:var(--danger)]"
                  )}
                >
                  <td className="px-3 py-2">
                    <div className="relative h-10 w-10 flex-shrink-0 overflow-hidden rounded border border-[color:var(--border-subtle)] bg-bg-page">
                      {row.snapshot_url ? (
                        <Image
                          src={row.snapshot_url}
                          alt={row.violation_label_tr}
                          fill
                          sizes="40px"
                          className="object-cover"
                          unoptimized
                        />
                      ) : null}
                    </div>
                  </td>
                  <td className="px-3 py-2 font-medium text-text-primary">
                    {tracker(row)}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-semibold"
                      style={{
                        background: color.bg,
                        color: color.fg,
                        border: `1px solid ${color.border}`,
                      }}
                    >
                      {formatViolationType(row.violation_type)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-text-muted">
                    {row.camera_id.slice(0, 8) + "…"}
                  </td>
                  <td className="px-3 py-2 text-text-muted whitespace-nowrap">
                    {formatTrTimestamp(row.started_at)}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {formatDuration(row.duration_seconds)}
                  </td>
                  <td className="px-3 py-2">
                    {row.is_active ? (
                      <span className="rounded bg-[color:var(--danger)] px-2 py-0.5 text-[10px] font-semibold uppercase text-white">
                        Aktif
                      </span>
                    ) : (
                      <span className="rounded bg-[color:var(--accent)] px-2 py-0.5 text-[10px] font-semibold uppercase text-white">
                        Çözüldü
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <button
                      onClick={() => setOpenEvent(row)}
                      className="rounded bg-[color:var(--bg-panel-2)] px-2.5 py-1 text-xs text-text-primary hover:bg-[color:var(--accent)] hover:text-white transition-colors"
                    >
                      Detay
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <nav className="mt-3 flex items-center justify-between text-xs text-text-muted">
        <span>
          Toplam {total} kayıt — sayfa {page} / {totalPages}
        </span>
        <div className="flex items-center gap-1">
          <PaginationButton
            href={buildPageHref(Math.max(1, page - 1))}
            disabled={page <= 1}
          >
            <ChevronLeft size={14} />
            Önceki
          </PaginationButton>
          <PaginationButton
            href={buildPageHref(Math.min(totalPages, page + 1))}
            disabled={page >= totalPages}
          >
            Sonraki
            <ChevronRight size={14} />
          </PaginationButton>
        </div>
      </nav>

      <EventDetailDialog
        event={openEvent}
        onOpenChange={(open) => {
          if (!open) setOpenEvent(null);
        }}
      />
    </>
  );
}

function PaginationButton({
  href,
  disabled,
  children,
}: {
  href: string;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  if (disabled) {
    return (
      <span className="inline-flex items-center gap-1 rounded border border-[color:var(--border-subtle)] px-2 py-1 text-text-muted opacity-40">
        {children}
      </span>
    );
  }
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-1 rounded border border-[color:var(--border-subtle)] px-2 py-1 text-text-primary hover:bg-bg-panel-2 transition-colors"
    >
      {children}
    </Link>
  );
}
