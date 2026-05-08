import {
  AlertTriangle,
  Clock3,
  Download,
  TrendingUp,
  Users,
} from "lucide-react";
import { MetricCard } from "@/components/ui/metric-card";
import { RangeTabs } from "@/components/reports/range-tabs";
import {
  DailyTrendChart,
  HourlyBarChart,
  TypePieChart,
} from "@/components/reports/reports-charts";
import { getDailyTrend, getViolationStats } from "@/lib/supabase/queries";
import { formatDuration, formatViolationType } from "@/lib/utils/format";

export const dynamic = "force-dynamic";

interface PageProps {
  searchParams: Promise<{
    range?: string;
  }>;
}

function rangeBounds(range: string): { fromIso: string; toIso: string } {
  const now = new Date();
  const to = new Date(now);
  to.setHours(23, 59, 59, 999);
  const from = new Date(now);
  switch (range) {
    case "today":
      from.setHours(0, 0, 0, 0);
      break;
    case "week": {
      // ISO week starts Monday in tr locale; we just use last 7 days for simplicity.
      from.setDate(from.getDate() - 6);
      from.setHours(0, 0, 0, 0);
      break;
    }
    case "month": {
      from.setDate(1);
      from.setHours(0, 0, 0, 0);
      break;
    }
    case "30d":
    default:
      from.setDate(from.getDate() - 29);
      from.setHours(0, 0, 0, 0);
      break;
  }
  return { fromIso: from.toISOString(), toIso: to.toISOString() };
}

export default async function ReportsPage({ searchParams }: PageProps) {
  const sp = await searchParams;
  const range = sp.range ?? "30d";
  const { fromIso, toIso } = rangeBounds(range);

  const [stats, trend] = await Promise.all([
    getViolationStats(fromIso, toIso),
    getDailyTrend(fromIso, toIso),
  ]);

  return (
    <div className="px-6 py-5">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold text-text-primary">Raporlar</h1>
        <RangeTabs current={range} />
      </header>

      {/* 4 stat cards */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Toplam İhlal"
          value={stats.total}
          icon={AlertTriangle}
          variant={stats.total > 0 ? "danger" : "default"}
        />
        <MetricCard
          label="Ortalama Süre"
          value={formatDuration(stats.avgDurationSeconds)}
          icon={Clock3}
        />
        <MetricCard
          label="Etkilenen İşçi"
          value={stats.uniqueWorkers}
          icon={Users}
        />
        <MetricCard
          label="En Sık İhlal"
          value={
            stats.mostCommonType
              ? formatViolationType(stats.mostCommonType)
              : "—"
          }
          icon={TrendingUp}
        />
      </div>

      {/* line + pie */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Section title="Günlük İhlal Eğilimi">
          <DailyTrendChart points={trend} />
        </Section>
        <Section title="Türlere Göre Dağılım">
          <TypePieChart byType={stats.byType} />
        </Section>
      </div>

      {/* hourly bars */}
      <div className="mt-4">
        <Section title="Saatlere Göre Dağılım">
          <HourlyBarChart byHour={stats.byHour} />
        </Section>
      </div>

      <footer className="mt-6 flex justify-end">
        <button
          disabled
          className="inline-flex items-center gap-2 rounded border border-[color:var(--border-subtle)] bg-bg-panel px-3 py-2 text-sm text-text-muted opacity-60"
          title="PDF dışa aktarma Phase 3.3'te eklenecek"
        >
          <Download size={14} />
          PDF Olarak İndir
        </button>
      </footer>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel">
      <header className="border-b border-[color:var(--border-subtle)] px-4 py-2.5">
        <h2 className="text-sm font-semibold text-text-primary">{title}</h2>
      </header>
      <div className="px-4 py-3">{children}</div>
    </section>
  );
}
