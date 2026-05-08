"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { format, parseISO } from "date-fns";
import { tr } from "date-fns/locale";
import { formatViolationType } from "@/lib/utils/format";
import type { DailyTrendPoint } from "@/lib/supabase/queries";

// Skip recharts during SSR — it tries to measure DOM dimensions before mount
// and logs "width(-1) height(-1)" warnings. After hydration we render normally.
function useMounted() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);
  return mounted;
}

function ChartSkeleton({ height = "h-64" }: { height?: string }) {
  return <div className={`${height} w-full animate-pulse rounded bg-bg-panel-2`} />;
}

const ACCENT = "#378ADD";
const ACCENT_DARK = "#185FA5";
const TYPE_COLORS: Record<string, string> = {
  no_hardhat: "#E24B4A",
  no_vest: "#F08C3F",
  no_mask: "#E2C24B",
};
const GRID = "#1f2942";
const TICK = "#85B7EB";

const tooltipStyle: React.CSSProperties = {
  backgroundColor: "#0d1424",
  border: "1px solid #1f2942",
  borderRadius: 6,
  fontSize: 12,
  color: "#E6F1FB",
};

const tooltipItemStyle: React.CSSProperties = { color: "#E6F1FB" };
const tooltipLabelStyle: React.CSSProperties = { color: "#85B7EB" };

interface DailyTrendChartProps {
  points: DailyTrendPoint[];
}

export function DailyTrendChart({ points }: DailyTrendChartProps) {
  const mounted = useMounted();
  if (!mounted) return <ChartSkeleton />;
  if (points.length === 0) {
    return <ChartEmpty />;
  }
  const data = points.map((p) => ({
    date: p.date,
    label: format(parseISO(p.date), "dd MMM", { locale: tr }),
    total: p.total,
  }));
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            stroke={TICK}
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: GRID }}
          />
          <YAxis
            stroke={TICK}
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            itemStyle={tooltipItemStyle}
            labelStyle={tooltipLabelStyle}
          />
          <Line
            type="monotone"
            dataKey="total"
            name="İhlal sayısı"
            stroke={ACCENT}
            strokeWidth={2}
            dot={{ r: 3, fill: ACCENT, strokeWidth: 0 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface TypePieChartProps {
  byType: Record<string, number>;
}

export function TypePieChart({ byType }: TypePieChartProps) {
  const mounted = useMounted();
  if (!mounted) return <ChartSkeleton />;
  const data = Object.entries(byType).map(([k, v]) => ({
    name: formatViolationType(k),
    value: v,
    type: k,
  }));
  if (data.length === 0) {
    return <ChartEmpty />;
  }
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            innerRadius={48}
            outerRadius={84}
            paddingAngle={2}
            dataKey="value"
            nameKey="name"
            stroke="none"
          >
            {data.map((d) => (
              <Cell
                key={d.type}
                fill={TYPE_COLORS[d.type] ?? "#85B7EB"}
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={tooltipStyle}
            itemStyle={tooltipItemStyle}
            labelStyle={tooltipLabelStyle}
          />
          <Legend
            verticalAlign="bottom"
            iconType="circle"
            wrapperStyle={{ color: TICK, fontSize: 12 }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

interface HourlyBarChartProps {
  byHour: number[]; // 24 entries
}

export function HourlyBarChart({ byHour }: HourlyBarChartProps) {
  const mounted = useMounted();
  if (!mounted) return <ChartSkeleton height="h-72" />;
  const data = byHour.map((v, i) => ({
    hour: i,
    label: `${i.toString().padStart(2, "0")}:00`,
    total: v,
  }));
  const allZero = byHour.every((v) => v === 0);
  if (allZero) {
    return <ChartEmpty />;
  }
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            stroke={TICK}
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            interval={1}
          />
          <YAxis
            stroke={TICK}
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            itemStyle={tooltipItemStyle}
            labelStyle={tooltipLabelStyle}
            cursor={{ fill: "rgba(55, 138, 221, 0.08)" }}
          />
          <Bar dataKey="total" name="İhlal sayısı" fill={ACCENT_DARK}>
            {data.map((d) => (
              <Cell key={d.hour} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ChartEmpty() {
  return (
    <div className="flex h-64 items-center justify-center rounded bg-bg-panel-2 text-sm text-text-muted">
      Bu aralık için veri yok.
    </div>
  );
}
