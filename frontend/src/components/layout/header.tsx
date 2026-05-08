"use client";

import { useEffect, useState } from "react";
import { Eye } from "lucide-react";
import { LiveBadge } from "@/components/ui/live-badge";

interface HeaderProps {
  siteName: string;
  location?: string | null;
}

function nowHHMMSS(): string {
  const d = new Date();
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function Header({ siteName, location }: HeaderProps) {
  // Avoid hydration mismatch: start blank, then tick once mounted.
  const [time, setTime] = useState<string>("");
  useEffect(() => {
    setTime(nowHHMMSS());
    const id = setInterval(() => setTime(nowHHMMSS()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header
      className="fixed top-0 right-0 left-0 z-40 flex h-14 items-center justify-between px-4 bg-bg-panel"
      style={{ borderBottom: "0.5px solid var(--border-subtle)" }}
    >
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded bg-success">
          <Eye size={18} className="text-bg-page" />
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-text-primary">
            İşGözcü
          </span>
          <span className="text-[11px] text-text-muted">
            {siteName}
            {location ? ` — ${location}` : ""}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <LiveBadge />
        <span className="font-mono tabular-nums text-sm text-text-muted min-w-[64px] text-right">
          {time || "--:--:--"}
        </span>
      </div>
    </header>
  );
}
