"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * Silent background poller that calls `router.refresh()` every 2 seconds
 * so the home page re-fetches its server-component data (metric cards,
 * Aktif İhlal, Son İhlaller).
 *
 * Phase 3.5 replaced the previous Supabase `postgres_changes` realtime
 * subscription with this simple polling loop. We trade ~0.5 s of latency
 * for rock-solid behaviour over USB-tethered 4G, WebSocket-hostile
 * proxies, and intermittent RLS hiccups — the kinds of failure modes the
 * realtime channel was prone to during demos.
 *
 * Renders nothing — the status pill that lived here was tied to the
 * WebSocket state, which no longer exists.
 */
export function RealtimeRefresh() {
  const router = useRouter();

  useEffect(() => {
    // Poll every 2 seconds. Robust to network drops — a missed tick is
    // automatically retried on the next interval.
    const interval = setInterval(() => {
      router.refresh();
    }, 2000);

    return () => clearInterval(interval);
  }, [router]);

  return null; // No UI — silent background poller.
}
