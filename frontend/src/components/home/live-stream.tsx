"use client";

import { AlertTriangle, Video, VideoOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

type StreamState = "loading" | "live" | "offline" | "error";

const DEFAULT_STREAM_BASE =
  process.env.NEXT_PUBLIC_STREAM_BASE_URL ?? "http://localhost:8001";
const HEALTH_POLL_MS = 5_000;
const OFFLINE_TOLERANCE_S = 4; // /health.stale_seconds beyond which we mark offline
const RECONNECT_AFTER_MS = 10_000; // re-mount the <img> if the stream stalls visibly

interface LiveStreamProps {
  cameraId: string;
  /** Override the FastAPI base URL. Defaults to NEXT_PUBLIC_STREAM_BASE_URL or :8001. */
  streamBase?: string;
  className?: string;
}

/**
 * Live MJPEG view backed by `backend/streaming/mjpeg_server.py`.
 *
 * Behaviour:
 *   - polls /health/{cameraId} every 5 s to surface offline / live state,
 *   - shows the MJPEG `<img>` only when health says we're live,
 *   - re-mounts the <img> on reconnect so the browser drops the dead
 *     connection and starts a fresh multipart fetch.
 *
 * Browsers natively decode `multipart/x-mixed-replace` — no JS player needed.
 */
export function LiveStream({
  cameraId,
  streamBase = DEFAULT_STREAM_BASE,
  className,
}: LiveStreamProps) {
  const [state, setState] = useState<StreamState>("loading");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  // `nonce` bumps on reconnect so React re-mounts the <img> element.
  const [nonce, setNonce] = useState<number>(() => Date.now());
  const lastLiveAt = useRef<number>(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      try {
        const res = await fetch(`${streamBase}/health/${cameraId}`, {
          cache: "no-store",
        });
        if (cancelled) return;
        if (res.status === 404) {
          setState("offline");
          setErrorMsg(null);
        } else if (!res.ok) {
          setState("error");
          setErrorMsg(`HTTP ${res.status}`);
        } else {
          const body = (await res.json()) as {
            live?: boolean;
            stale_seconds?: number;
          };
          if (cancelled) return;
          const live =
            body.live === true ||
            (typeof body.stale_seconds === "number" &&
              body.stale_seconds < OFFLINE_TOLERANCE_S);
          if (live) {
            lastLiveAt.current = Date.now();
            setState("live");
            setErrorMsg(null);
          } else {
            setState("offline");
            setErrorMsg(null);
          }
        }
      } catch (err) {
        if (cancelled) return;
        setState("error");
        setErrorMsg(err instanceof Error ? err.message : "Bağlantı hatası");
      } finally {
        if (!cancelled) {
          timer = setTimeout(tick, HEALTH_POLL_MS);
        }
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [cameraId, streamBase]);

  // If we've been live and then dropped, re-mount the <img> after a beat —
  // browsers don't always recover the multipart connection on their own.
  useEffect(() => {
    if (state !== "offline" && state !== "error") return;
    const sinceLive = Date.now() - lastLiveAt.current;
    if (sinceLive > RECONNECT_AFTER_MS && lastLiveAt.current > 0) {
      const id = setTimeout(() => setNonce(Date.now()), 500);
      return () => clearTimeout(id);
    }
    return;
  }, [state]);

  const streamUrl = `${streamBase}/stream/${cameraId}?n=${nonce}`;

  return (
    <div
      className={cn(
        "relative h-full w-full overflow-hidden bg-black",
        className
      )}
    >
      {state === "live" ? (
        // Native browser MJPEG decoding — no <video>, no JS, no MediaSource.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={nonce}
          src={streamUrl}
          alt={`Canlı yayın - ${cameraId}`}
          className="h-full w-full object-contain"
          onError={() => setState("error")}
        />
      ) : (
        <Placeholder
          state={state}
          cameraId={cameraId}
          errorMsg={errorMsg}
          onRetry={() => {
            setNonce(Date.now());
            setState("loading");
          }}
        />
      )}

      <span
        className={cn(
          "absolute right-3 top-3 inline-flex items-center gap-1.5 rounded px-2 py-0.5",
          "bg-black/65 text-[11px] font-semibold backdrop-blur-sm",
          state === "live"
            ? "text-text-primary"
            : "text-text-muted"
        )}
      >
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            state === "live"
              ? "bg-danger animate-pulse-slow"
              : state === "loading"
              ? "bg-yellow-400"
              : "bg-text-muted"
          )}
        />
        {state === "live"
          ? "CANLI"
          : state === "loading"
          ? "BAĞLANIYOR"
          : state === "offline"
          ? "ÇEVRİMDIŞI"
          : "HATA"}
      </span>

      <span className="absolute left-3 top-3 rounded bg-black/65 px-2 py-1 text-[11px] text-text-primary backdrop-blur-sm">
        {cameraId}
      </span>
    </div>
  );
}

function Placeholder({
  state,
  cameraId,
  errorMsg,
  onRetry,
}: {
  state: Exclude<StreamState, "live">;
  cameraId: string;
  errorMsg: string | null;
  onRetry: () => void;
}) {
  if (state === "loading") {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-text-muted">
        <Video size={56} className="animate-pulse opacity-30" />
        <span className="text-xs">Yayına bağlanılıyor…</span>
      </div>
    );
  }
  if (state === "offline") {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-text-muted">
        <VideoOff size={56} className="opacity-30" />
        <span className="text-xs">{cameraId} çevrimdışı</span>
        <span className="text-[11px] text-text-muted/70">
          Live Simulator&apos;ı{" "}
          <code className="rounded bg-bg-panel-2 px-1 py-0.5 text-[11px]">
            --stream
          </code>{" "}
          ile başlatın.
        </span>
      </div>
    );
  }
  // error
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-2 px-4 text-center">
      <AlertTriangle size={48} className="text-[color:var(--danger)]" />
      <span className="text-xs text-text-primary">Yayın bağlantı hatası</span>
      {errorMsg ? (
        <span className="text-[11px] text-text-muted">{errorMsg}</span>
      ) : null}
      <button
        type="button"
        onClick={onRetry}
        className="mt-1 rounded bg-bg-panel-2 px-3 py-1 text-[11px] text-text-primary hover:bg-bg-panel-2/80"
      >
        Tekrar dene
      </button>
    </div>
  );
}
