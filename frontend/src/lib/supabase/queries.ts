// Supabase queries used by the dashboard pages. Centralizing them here keeps
// the page components focused on layout / state.
//
// supabase-js's column-projection inference returns `never` for our local
// `Database` generic (see Phase 3.1 build notes), so we cast results to our
// own typed shapes.

import { getServerClient } from "./server";
import type {
  Camera,
  Site,
  ViolationEvent,
  ViolationEventWithWorker,
  Worker,
} from "./types";

// ---------- events ----------

export interface EventFilters {
  from?: string; // ISO datetime
  to?: string;
  type?: string; // 'no_hardhat' | 'no_vest' | 'no_mask'
  cameraId?: string;
  status?: "active" | "resolved";
  limit?: number;
  offset?: number;
  countOnly?: boolean;
}

export interface EventsPage {
  rows: ViolationEventWithWorker[];
  total: number;
}

export async function getEvents(filters: EventFilters = {}): Promise<EventsPage> {
  const sb = getServerClient();
  let q = sb
    .from("events")
    .select("*, workers(tracker_id)", { count: "exact" })
    .order("started_at", { ascending: false });

  if (filters.from) q = q.gte("started_at", filters.from);
  if (filters.to) q = q.lte("started_at", filters.to);
  if (filters.type) q = q.eq("violation_type", filters.type);
  if (filters.cameraId) q = q.eq("camera_id", filters.cameraId);
  if (filters.status === "active") q = q.eq("is_active", true);
  if (filters.status === "resolved") q = q.eq("is_active", false);

  const limit = filters.limit ?? 20;
  const offset = filters.offset ?? 0;
  q = q.range(offset, offset + limit - 1);

  const { data, count } = await q;
  const rows = (data ?? []) as unknown as ViolationEventWithWorker[];
  return { rows, total: count ?? rows.length };
}

export async function getEventById(
  id: string
): Promise<ViolationEventWithWorker | null> {
  const sb = getServerClient();
  const { data } = await sb
    .from("events")
    .select("*, workers(tracker_id)")
    .eq("id", id)
    .limit(1);
  const rows = (data ?? []) as unknown as ViolationEventWithWorker[];
  return rows[0] ?? null;
}

// ---------- workers ----------

export async function getWorkers(): Promise<Worker[]> {
  const sb = getServerClient();
  const { data } = await sb
    .from("workers")
    .select("*")
    .order("total_violations", { ascending: false })
    .order("last_seen", { ascending: false });
  return (data ?? []) as unknown as Worker[];
}

export interface WorkerWithLatest extends Worker {
  latest_event?: ViolationEvent | null;
}

export async function getWorkersWithLatestEvent(): Promise<WorkerWithLatest[]> {
  const workers = await getWorkers();
  if (workers.length === 0) return [];
  const ids = workers.map((w) => w.id);
  const sb = getServerClient();
  const { data } = await sb
    .from("events")
    .select("*")
    .in("worker_id", ids)
    .order("started_at", { ascending: false });
  const events = (data ?? []) as unknown as ViolationEvent[];
  const latestByWorker = new Map<string, ViolationEvent>();
  for (const e of events) {
    if (!latestByWorker.has(e.worker_id)) {
      latestByWorker.set(e.worker_id, e);
    }
  }
  return workers.map((w) => ({
    ...w,
    latest_event: latestByWorker.get(w.id) ?? null,
  }));
}

export async function getEventsForWorker(
  workerId: string
): Promise<ViolationEvent[]> {
  const sb = getServerClient();
  const { data } = await sb
    .from("events")
    .select("*")
    .eq("worker_id", workerId)
    .order("started_at", { ascending: false });
  return (data ?? []) as unknown as ViolationEvent[];
}

// ---------- cameras / sites ----------

export async function getCameras(): Promise<Camera[]> {
  const sb = getServerClient();
  const { data } = await sb.from("cameras").select("*").order("name");
  return (data ?? []) as unknown as Camera[];
}

export async function getSites(): Promise<Site[]> {
  const sb = getServerClient();
  const { data } = await sb
    .from("sites")
    .select("*")
    .order("created_at", { ascending: true });
  return (data ?? []) as unknown as Site[];
}

// ---------- aggregations for /raporlar ----------

export interface ViolationStats {
  total: number;
  byType: Record<string, number>;
  byHour: number[]; // 24 buckets
  uniqueWorkers: number;
  avgDurationSeconds: number;
  mostCommonType: string | null;
}

export interface DailyTrendPoint {
  date: string; // YYYY-MM-DD
  total: number;
}

export async function getViolationStats(
  fromIso: string,
  toIso: string
): Promise<ViolationStats> {
  const sb = getServerClient();
  const { data } = await sb
    .from("events")
    .select("violation_type, started_at, duration_seconds, worker_id")
    .gte("started_at", fromIso)
    .lte("started_at", toIso);
  type Slim = {
    violation_type: string;
    started_at: string;
    duration_seconds: number;
    worker_id: string;
  };
  const rows = (data ?? []) as unknown as Slim[];

  const byType: Record<string, number> = {};
  const byHour = new Array(24).fill(0) as number[];
  const workers = new Set<string>();
  let totalDuration = 0;
  for (const r of rows) {
    byType[r.violation_type] = (byType[r.violation_type] ?? 0) + 1;
    const h = new Date(r.started_at).getHours();
    if (h >= 0 && h < 24) byHour[h]++;
    workers.add(r.worker_id);
    totalDuration += r.duration_seconds ?? 0;
  }
  let mostCommonType: string | null = null;
  let mostCommonCount = -1;
  for (const [t, n] of Object.entries(byType)) {
    if (n > mostCommonCount) {
      mostCommonCount = n;
      mostCommonType = t;
    }
  }
  return {
    total: rows.length,
    byType,
    byHour,
    uniqueWorkers: workers.size,
    avgDurationSeconds: rows.length > 0 ? totalDuration / rows.length : 0,
    mostCommonType,
  };
}

export async function getDailyTrend(
  fromIso: string,
  toIso: string
): Promise<DailyTrendPoint[]> {
  const sb = getServerClient();
  const { data } = await sb
    .from("events")
    .select("started_at")
    .gte("started_at", fromIso)
    .lte("started_at", toIso)
    .order("started_at", { ascending: true });
  type Slim = { started_at: string };
  const rows = (data ?? []) as unknown as Slim[];

  const dayCounts = new Map<string, number>();
  // Pre-fill day buckets so the chart shows 0s for empty days.
  const start = new Date(fromIso);
  const end = new Date(toIso);
  for (
    let d = new Date(start);
    d <= end;
    d.setDate(d.getDate() + 1)
  ) {
    const key = d.toISOString().slice(0, 10);
    dayCounts.set(key, 0);
  }
  for (const r of rows) {
    const key = new Date(r.started_at).toISOString().slice(0, 10);
    dayCounts.set(key, (dayCounts.get(key) ?? 0) + 1);
  }
  return Array.from(dayCounts.entries())
    .sort(([a], [b]) => (a < b ? -1 : 1))
    .map(([date, total]) => ({ date, total }));
}
