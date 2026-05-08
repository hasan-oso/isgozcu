import { AlertTriangle, Camera, Users, Video, ShieldCheck } from "lucide-react";
import { format } from "date-fns";
import { tr } from "date-fns/locale";
import { MetricCard } from "@/components/ui/metric-card";
import { ViolationRow } from "@/components/violations/violation-row";
import { LiveBadge } from "@/components/ui/live-badge";
import { getServerClient } from "@/lib/supabase/server";
import type {
  ViolationEventWithWorker,
} from "@/lib/supabase/types";

// Always re-fetch on each request — no caching of dashboard data.
export const dynamic = "force-dynamic";
export const revalidate = 0;

interface DashboardData {
  cameraCount: number;
  activeWorkerCount: number;
  activeViolationCount: number;
  complianceRate: number; // 0..100
  recentEvents: ViolationEventWithWorker[];
  newestActiveEvent: ViolationEventWithWorker | null;
  errorMessage: string | null;
}

const ONE_HOUR_AGO = () => new Date(Date.now() - 60 * 60 * 1000).toISOString();

const COMPLIANT_VIOLATION_TYPES = ["no_hardhat", "no_vest", "no_mask"];

async function loadDashboard(): Promise<DashboardData> {
  let cameraCount = 0;
  let activeWorkerCount = 0;
  let activeViolationCount = 0;
  let complianceRate = 100;
  let recentEvents: ViolationEventWithWorker[] = [];
  let newestActiveEvent: ViolationEventWithWorker | null = null;
  let errorMessage: string | null = null;
  try {
    const sb = getServerClient();
    const [cams, workers, activeEv, totalEv, recent] = await Promise.all([
      sb.from("cameras").select("id", { count: "exact", head: true }),
      sb
        .from("workers")
        .select("id", { count: "exact", head: true })
        .gte("last_seen", ONE_HOUR_AGO()),
      sb
        .from("events")
        .select("id", { count: "exact", head: true })
        .eq("is_active", true),
      sb
        .from("events")
        .select("violation_type", { count: "exact" })
        .in("violation_type", COMPLIANT_VIOLATION_TYPES),
      sb
        .from("events")
        .select("*, workers(tracker_id)")
        .order("started_at", { ascending: false })
        .limit(10),
    ]);
    cameraCount = cams.count ?? 0;
    activeWorkerCount = workers.count ?? 0;
    activeViolationCount = activeEv.count ?? 0;
    // Compliance rate: rough proxy — share of *all* events that are resolved
    // (= compliance restored). With no events, default to 100%.
    const totalCount = totalEv.count ?? 0;
    if (totalCount > 0) {
      const stillActive = activeViolationCount;
      complianceRate = Math.round(((totalCount - stillActive) / totalCount) * 100);
    }
    recentEvents = (recent.data ?? []) as ViolationEventWithWorker[];
    newestActiveEvent = recentEvents.find((e) => e.is_active) ?? null;
  } catch (err) {
    errorMessage = err instanceof Error ? err.message : "Veri yüklenemedi.";
  }
  return {
    cameraCount,
    activeWorkerCount,
    activeViolationCount,
    complianceRate,
    recentEvents,
    newestActiveEvent,
    errorMessage,
  };
}

export default async function HomePage() {
  const data = await loadDashboard();
  const nowLabel = format(new Date(), "dd MMMM yyyy, HH:mm", { locale: tr });
  return (
    <div className="px-6 py-5">
      <div className="mb-4 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold text-text-primary">Genel Bakış</h1>
        <span className="text-xs text-text-muted">{nowLabel}</span>
      </div>

      {data.errorMessage ? (
        <div className="mb-4 rounded border border-[color:var(--danger)] bg-danger-soft px-3 py-2 text-sm text-[color:var(--danger)]">
          Veri yüklenemedi: {data.errorMessage}
        </div>
      ) : null}

      {/* Metric grid */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Aktif Kamera"
          value={data.cameraCount}
          icon={Camera}
        />
        <MetricCard
          label="Aktif İşçi"
          value={data.activeWorkerCount}
          icon={Users}
          hint="(son 1 saat)"
        />
        <MetricCard
          label="Aktif İhlal"
          value={data.activeViolationCount}
          variant={data.activeViolationCount > 0 ? "danger" : "default"}
          icon={AlertTriangle}
        />
        <MetricCard
          label="Uyum Oranı"
          value={`%${data.complianceRate}`}
          icon={ShieldCheck}
        />
      </div>

      {/* Two-column: live video (left) + recent violations (right) */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[1.6fr_1fr]">
        <LiveStreamCard latestActive={data.newestActiveEvent} />
        <RecentViolationsCard events={data.recentEvents} />
      </div>
    </div>
  );
}

function LiveStreamCard({
  latestActive,
}: {
  latestActive: ViolationEventWithWorker | null;
}) {
  return (
    <section className="rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel">
      <div className="flex items-center justify-between border-b border-[color:var(--border-subtle)] px-4 py-2.5">
        <h2 className="text-sm font-semibold text-text-primary">Canlı Yayın</h2>
        <span className="text-xs text-text-muted">CCTV_01</span>
      </div>
      <div className="relative aspect-video w-full bg-bg-page">
        {/* Placeholder video area — real WebRTC / HLS feed comes in Phase 3.3+. */}
        <div className="flex h-full w-full items-center justify-center text-text-muted">
          <Video size={56} className="opacity-30" />
        </div>
        <span className="absolute right-3 top-3">
          <LiveBadge />
        </span>
        <span className="absolute left-3 top-3 rounded bg-black/60 px-2 py-1 text-[11px] text-text-primary">
          CCTV_01
        </span>
        {latestActive ? (
          <div className="absolute left-3 right-3 top-12 flex items-start gap-2 rounded border border-[color:var(--danger)] bg-[rgba(216,59,59,0.18)] px-3 py-2 text-xs text-text-primary backdrop-blur-sm">
            <AlertTriangle
              size={16}
              className="mt-0.5 flex-shrink-0 text-[color:var(--danger)]"
            />
            <div className="min-w-0">
              <div className="font-semibold text-[color:var(--danger)]">
                Aktif ihlal: {latestActive.violation_label_tr}
              </div>
              <div className="truncate text-text-muted">
                {latestActive.workers?.tracker_id ??
                  (latestActive.metadata as { tracker_id?: string } | null)
                    ?.tracker_id ??
                  "—"}
              </div>
            </div>
          </div>
        ) : null}
        <span className="absolute bottom-3 right-3 rounded bg-black/60 px-2 py-1 text-[11px] text-text-muted">
          {format(new Date(), "dd.MM.yyyy HH:mm:ss")}
        </span>
      </div>
    </section>
  );
}

function RecentViolationsCard({
  events,
}: {
  events: ViolationEventWithWorker[];
}) {
  return (
    <section className="flex h-full flex-col rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel">
      <div className="flex items-center justify-between border-b border-[color:var(--border-subtle)] px-4 py-2.5">
        <h2 className="text-sm font-semibold text-text-primary">Son İhlaller</h2>
        <span className="text-xs text-text-muted">{events.length}</span>
      </div>
      <div className="flex flex-1 flex-col gap-1.5 px-3 py-3 max-h-[480px] overflow-y-auto">
        {events.length === 0 ? (
          <div className="my-auto py-8 text-center text-sm text-text-muted">
            Henüz ihlal kaydı yok.
          </div>
        ) : (
          events.map((event) => (
            <ViolationRow key={event.id} event={event} />
          ))
        )}
      </div>
    </section>
  );
}
