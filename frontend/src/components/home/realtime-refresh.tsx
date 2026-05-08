"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { getBrowserClient } from "@/lib/supabase/client";
import { cn } from "@/lib/utils";

type Status = "connecting" | "connected" | "disconnected";

/**
 * Subscribes to the `events` table and triggers a server-component refresh
 * whenever a row is inserted, updated, or deleted. Lightweight: we don't
 * track individual changes — Next's RSC re-render brings the latest data.
 *
 * Renders a small status pill: "Canlı bağlantı" (green dot) when subscribed.
 */
export function RealtimeRefresh() {
  const router = useRouter();
  const [status, setStatus] = useState<Status>("connecting");
  const lastRefresh = useRef<number>(0);

  useEffect(() => {
    const sb = getBrowserClient();

    const refresh = () => {
      // Throttle: at most one refresh per 1000 ms even on bursts.
      const now = Date.now();
      if (now - lastRefresh.current < 1000) return;
      lastRefresh.current = now;
      router.refresh();
    };

    const channel = sb
      .channel("home_events")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "events" },
        refresh
      )
      .subscribe((state) => {
        if (state === "SUBSCRIBED") setStatus("connected");
        else if (state === "CLOSED" || state === "CHANNEL_ERROR")
          setStatus("disconnected");
      });

    return () => {
      sb.removeChannel(channel);
    };
  }, [router]);

  const dotColor =
    status === "connected"
      ? "bg-[color:var(--success)]"
      : status === "connecting"
      ? "bg-yellow-400"
      : "bg-[color:var(--danger)]";
  const label =
    status === "connected"
      ? "Canlı bağlantı"
      : status === "connecting"
      ? "Bağlanıyor…"
      : "Bağlantı yok";

  return (
    <span className="inline-flex items-center gap-1.5 rounded bg-bg-panel-2 px-2 py-0.5 text-[11px] text-text-muted">
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          dotColor,
          status === "connected" && "animate-pulse-slow"
        )}
      />
      {label}
    </span>
  );
}
