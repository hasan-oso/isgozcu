"""MJPEG streaming server for the Live Simulator (Phase 3.3).

A tiny FastAPI app that serves the simulator's annotated frames to the
Next.js dashboard as `multipart/x-mixed-replace` (MJPEG). It is opt-in via
`--stream` on the simulator — when off, this module is never imported.

Architecture:

    LiveStreamSimulator  ─►  FrameBuffer.update(camera_id, jpg_bytes)
                                       │
                                       ▼
                           ┌──────────────────────────┐
                           │  FastAPI app (port 8001) │
                           │  /                       │  ← server info
                           │  /health/{camera_id}     │  ← liveness JSON
                           │  /snapshot/{camera_id}   │  ← single JPEG
                           │  /stream/{camera_id}     │  ← MJPEG multipart
                           └──────────────────────────┘
                                       ▲
                                       │
                          Browser <img src="…/stream/CCTV_01">

The simulator JPEG-encodes the annotated frame once per processed frame
and pushes the bytes into a per-camera `FrameBuffer` slot. The HTTP
endpoint streams those bytes to any number of connected viewers.

Thread-safety: `update` is called from the simulator's main thread,
`latest` is called from one or more uvicorn worker threads. A single
`threading.Lock` per camera serializes the swap.

Latency: typically ~50 ms end-to-end on localhost (one frame buffered).
Bandwidth: JPEG quality 75 keeps a 1280x720 frame around 60–120 KB; at
30 fps that's roughly 2–4 MB/s per viewer — fine for LAN.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional


# ---------- shared frame buffer (in-process, multi-camera) ----------


@dataclass
class _CameraSlot:
    """One camera's most recent JPEG frame + a mutex protecting it."""

    jpeg_bytes: Optional[bytes] = None
    updated_at: float = 0.0
    frame_seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class FrameBuffer:
    """Thread-safe per-camera latest-frame slots.

    Holds only the *latest* JPEG per camera (no queue) — late viewers see
    fresh frames immediately, slow viewers can't back-pressure the producer.
    """

    def __init__(self) -> None:
        self._cameras: dict[str, _CameraSlot] = {}
        self._registry_lock = threading.Lock()

    # producer side -----------------------------------------------------

    def register(self, camera_id: str) -> None:
        with self._registry_lock:
            if camera_id not in self._cameras:
                self._cameras[camera_id] = _CameraSlot()

    def update(self, camera_id: str, jpeg_bytes: bytes) -> None:
        """Push a new annotated JPEG into the camera's slot. O(1)."""
        slot = self._cameras.get(camera_id)
        if slot is None:
            # Auto-register so a typo'd camera id doesn't silently drop frames.
            self.register(camera_id)
            slot = self._cameras[camera_id]
        with slot.lock:
            slot.jpeg_bytes = jpeg_bytes
            slot.updated_at = time.time()
            slot.frame_seq += 1

    # consumer side -----------------------------------------------------

    def latest(self, camera_id: str) -> tuple[Optional[bytes], int, float]:
        """Return (jpeg_bytes, frame_seq, updated_at) for the camera, or
        (None, 0, 0.0) if nothing has been pushed yet."""
        slot = self._cameras.get(camera_id)
        if slot is None:
            return None, 0, 0.0
        with slot.lock:
            return slot.jpeg_bytes, slot.frame_seq, slot.updated_at

    def list_cameras(self) -> list[dict]:
        out: list[dict] = []
        for cam_id, slot in self._cameras.items():
            with slot.lock:
                out.append({
                    "camera_id": cam_id,
                    "has_frame": slot.jpeg_bytes is not None,
                    "frame_seq": slot.frame_seq,
                    "updated_at": slot.updated_at,
                    "stale_seconds": (
                        time.time() - slot.updated_at if slot.updated_at else None
                    ),
                })
        return out


# Process-wide singleton — the simulator and the FastAPI app share this.
_buffer = FrameBuffer()


def get_buffer() -> FrameBuffer:
    return _buffer


# Convenience wrappers so callers don't have to know about the singleton.
def register_camera(camera_id: str) -> None:
    _buffer.register(camera_id)


def update_frame(camera_id: str, jpeg_bytes: bytes) -> None:
    _buffer.update(camera_id, jpeg_bytes)


# ---------- FastAPI app ----------


def create_app():
    """Build a fresh FastAPI app wired to the singleton buffer.

    Imported lazily so projects that never use streaming don't pay the
    fastapi import cost.
    """
    from fastapi import FastAPI, HTTPException, Response  # noqa: WPS433
    from fastapi.middleware.cors import CORSMiddleware  # noqa: WPS433
    from fastapi.responses import StreamingResponse  # noqa: WPS433

    app = FastAPI(
        title="İşGözcü Live Stream",
        description="MJPEG streaming endpoint for the Next.js dashboard.",
        version="0.1.0",
    )

    # CORS: the dashboard runs on localhost:3000 in dev. We allow that
    # origin explicitly. Production deployments behind a reverse proxy
    # (same origin) wouldn't need this — leaving it on is harmless.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict:
        return {
            "service": "isgozcu-mjpeg",
            "version": "0.1.0",
            "cameras": _buffer.list_cameras(),
        }

    @app.get("/health/{camera_id}")
    def health(camera_id: str) -> dict:
        jpg, seq, ts = _buffer.latest(camera_id)
        if jpg is None:
            # Camera hasn't pushed yet — that's a 404, not a 5xx.
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not streaming.",
            )
        return {
            "camera_id": camera_id,
            "frame_seq": seq,
            "updated_at": ts,
            "stale_seconds": time.time() - ts,
            "live": (time.time() - ts) < 2.0,
        }

    @app.get("/snapshot/{camera_id}")
    def snapshot(camera_id: str) -> Response:
        jpg, _seq, _ts = _buffer.latest(camera_id)
        if jpg is None:
            raise HTTPException(status_code=404, detail="No frame yet.")
        return Response(
            content=jpg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/stream/{camera_id}")
    def stream(camera_id: str):
        boundary = "frame"

        def gen() -> Iterator[bytes]:
            last_seq = -1
            # ~30 fps cap on the wire — sleep when the producer hasn't
            # advanced. Don't busy-loop when a camera goes away.
            poll_interval = 0.015  # 15 ms — enough headroom for 60 fps producers
            stale_warn_after = 5.0
            last_emit = time.time()
            while True:
                jpg, seq, ts = _buffer.latest(camera_id)
                now = time.time()
                if jpg is None:
                    # Camera not yet registered / no frames. Hold the
                    # connection open; client can decide to drop.
                    if now - last_emit > stale_warn_after:
                        # Yield a tiny keepalive comment so proxies don't
                        # close the connection.
                        last_emit = now
                        yield (
                            f"--{boundary}\r\n"
                            "Content-Type: text/plain\r\n"
                            "Content-Length: 0\r\n\r\n\r\n"
                        ).encode("ascii")
                    time.sleep(0.1)
                    continue
                if seq == last_seq:
                    time.sleep(poll_interval)
                    continue
                last_seq = seq
                last_emit = now
                yield (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                ).encode("ascii") + jpg + b"\r\n"

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-store, no-transform"},
        )

    return app


# ---------- launch helper ----------


_server_thread: Optional[threading.Thread] = None
_server_started: bool = False


def start_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Launch uvicorn in a daemon thread.

    Idempotent — calling it twice is a no-op so the simulator can call
    it from `__init__` without worrying about double-binding.
    """
    global _server_thread, _server_started
    if _server_started:
        return
    import uvicorn  # noqa: WPS433

    app = create_app()
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",  # keep the simulator console clean
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        try:
            server.run()
        except Exception as e:  # pragma: no cover — diagnostic only
            print(f"[uyarı] MJPEG sunucusu hata: {e}")

    t = threading.Thread(target=_run, daemon=True, name="mjpeg-server")
    t.start()
    _server_thread = t
    _server_started = True
    # Tiny grace period so the bind error (if any) surfaces before the
    # simulator starts pumping frames into a void.
    time.sleep(0.25)
    print(f"[bilgi] MJPEG sunucusu http://{host}:{port} üzerinde aktif.")
