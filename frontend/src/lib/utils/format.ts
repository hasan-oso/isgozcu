import { format, formatDistanceToNow, parseISO } from "date-fns";
import { tr } from "date-fns/locale";

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m < 60) return `${m}d ${s}s`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h}sa ${mm}d`;
}

export function formatRelativeTime(date: string | Date): string {
  const d = typeof date === "string" ? parseISO(date) : date;
  return formatDistanceToNow(d, { locale: tr, addSuffix: true });
}

export function formatTrTimestamp(date: string | Date): string {
  const d = typeof date === "string" ? parseISO(date) : date;
  return format(d, "dd MMM yyyy HH:mm", { locale: tr });
}

const VIOLATION_LABELS_TR: Record<string, string> = {
  no_hardhat: "Baret yok",
  no_vest: "Yelek yok",
  no_mask: "Maske yok",
};

export function formatViolationType(type: string): string {
  return VIOLATION_LABELS_TR[type] ?? type;
}

const VIOLATION_COLORS: Record<string, { fg: string; bg: string; border: string }> = {
  no_hardhat: { fg: "#FF7A7A", bg: "#2a1620", border: "#501313" },
  no_vest: { fg: "#FFB36B", bg: "#2a1f10", border: "#5a3d10" },
  no_mask: { fg: "#FFD86B", bg: "#2a2510", border: "#5a4d10" },
};

export function getViolationColor(type: string) {
  return (
    VIOLATION_COLORS[type] ?? {
      fg: "#93a5c2",
      bg: "#111a2e",
      border: "#1f2942",
    }
  );
}
