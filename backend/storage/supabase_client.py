"""Supabase persistence layer.

Four classes:
    SupabaseClient   — singleton wrapper, loads URL + service-role key from .env
    SiteManager      — get_or_create site / camera bootstrap
    WorkerRegistry   — maps tracker IDs ("W007") to worker UUIDs (cache → DB → insert)
    EventUploader    — uploads ViolationEvent + snapshot to Supabase

Privacy / security:
    * `SUPABASE_URL` is fine to log.
    * `SUPABASE_SERVICE_ROLE_KEY` is NEVER logged. The class does not expose it.
    * Only `tracker_id` strings ("W007") are stored — no biometrics, no real names.

Failure mode:
    Every method that talks to Supabase is wrapped in try/except. On failure, log
    at ERROR level and return None. The caller (live simulator) treats None as
    "skip this upload, keep running locally". The simulator must never crash if
    Supabase is unreachable.
"""

from __future__ import annotations

import logging
import os
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

# Avoid an import cycle: ViolationEvent lives in violation_engine.
from backend.inference.violation_engine import ViolationEvent


# ---------- logging ----------

logger = logging.getLogger("isgozcu.supabase")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


SNAPSHOT_BUCKET = "snapshots"


# ---------- 1. SupabaseClient (singleton) ----------


class SupabaseClient:
    """Singleton wrapper for the Supabase connection."""

    _instance: Optional["SupabaseClient"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url:
            raise RuntimeError("SUPABASE_URL not set in .env")
        if not key:
            raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY not set in .env")
        # Safe to log host — not the key.
        try:
            host = url.split("://", 1)[1].split(".", 1)[0]
        except Exception:
            host = "?"
        logger.info(f"Supabase client connecting to host '{host}.*'")
        self._client: Client = create_client(url, key)

    @property
    def client(self) -> Client:
        return self._client

    def health_check(self) -> bool:
        """Verify connection by querying the sites table. True if working."""
        try:
            self._client.table("sites").select("id").limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase health check failed: {e}")
            return False


# ---------- 2. SiteManager ----------


class SiteManager:
    """Get-or-create site / camera bootstrap for a session."""

    DEMO_SITE_NAME = "Demo Şantiye"
    DEMO_CAMERA_NAME = "CCTV_01"

    def __init__(self, supabase: SupabaseClient) -> None:
        self.client = supabase.client

    def get_or_create_site(self, name: str, location: Optional[str] = None) -> Optional[str]:
        """Returns site_id (uuid). Creates if it doesn't exist."""
        try:
            existing = (
                self.client.table("sites").select("id").eq("name", name).limit(1).execute()
            )
            if existing.data:
                return existing.data[0]["id"]
            payload: dict[str, Any] = {"name": name}
            if location:
                payload["location"] = location
            inserted = self.client.table("sites").insert(payload).execute()
            site_id = inserted.data[0]["id"]
            logger.info(f"Created site '{name}' id={site_id}")
            return site_id
        except Exception as e:
            logger.error(f"get_or_create_site('{name}') failed: {e}")
            return None

    def get_or_create_camera(self, site_id: str, camera_name: str) -> Optional[str]:
        """Returns camera_id (uuid). Creates if it doesn't exist."""
        try:
            existing = (
                self.client.table("cameras")
                .select("id")
                .eq("site_id", site_id)
                .eq("name", camera_name)
                .limit(1)
                .execute()
            )
            if existing.data:
                return existing.data[0]["id"]
            inserted = (
                self.client.table("cameras")
                .insert({"site_id": site_id, "name": camera_name})
                .execute()
            )
            camera_id = inserted.data[0]["id"]
            logger.info(f"Created camera '{camera_name}' under site {site_id} id={camera_id}")
            return camera_id
        except Exception as e:
            logger.error(f"get_or_create_camera({site_id}, '{camera_name}') failed: {e}")
            return None

    def get_default_demo_context(self) -> tuple[Optional[str], Optional[str]]:
        """Return (site_id, camera_id) for Demo Şantiye / CCTV_01.  Creates if missing."""
        site_id = self.get_or_create_site(self.DEMO_SITE_NAME, location="Ankara, Türkiye")
        if not site_id:
            return None, None
        camera_id = self.get_or_create_camera(site_id, self.DEMO_CAMERA_NAME)
        return site_id, camera_id


# ---------- 3. WorkerRegistry ----------


class WorkerRegistry:
    """Maps ByteTrack tracker IDs (e.g. "W007") to database worker UUIDs.

    A 'tracker_id' is whatever ByteTrack assigns (we format as "W%03d").
    A 'worker_id' (uuid) is the DB primary key.
    UNIQUE(site_id, tracker_id) means re-detecting the same physical worker
    later in the same site reuses the existing row.
    """

    def __init__(self, supabase: SupabaseClient, site_id: str) -> None:
        self.client = supabase.client
        self.site_id = site_id
        self._cache: dict[str, str] = {}

    def get_or_create_worker(self, tracker_id: str) -> Optional[str]:
        """Cache → DB lookup → insert. Returns worker uuid or None on failure."""
        if tracker_id in self._cache:
            self._touch_last_seen(self._cache[tracker_id])
            return self._cache[tracker_id]

        try:
            existing = (
                self.client.table("workers")
                .select("id")
                .eq("site_id", self.site_id)
                .eq("tracker_id", tracker_id)
                .limit(1)
                .execute()
            )
            if existing.data:
                worker_id = existing.data[0]["id"]
                self._cache[tracker_id] = worker_id
                self._touch_last_seen(worker_id)
                return worker_id

            now = datetime.now(timezone.utc).isoformat()
            inserted = (
                self.client.table("workers")
                .insert(
                    {
                        "site_id": self.site_id,
                        "tracker_id": tracker_id,
                        "first_seen": now,
                        "last_seen": now,
                        "total_violations": 0,
                    }
                )
                .execute()
            )
            worker_id = inserted.data[0]["id"]
            self._cache[tracker_id] = worker_id
            logger.info(f"Created worker tracker='{tracker_id}' id={worker_id}")
            return worker_id
        except Exception as e:
            logger.error(f"get_or_create_worker('{tracker_id}') failed: {e}")
            return None

    def _touch_last_seen(self, worker_uuid: str) -> None:
        try:
            self.client.table("workers").update(
                {"last_seen": datetime.now(timezone.utc).isoformat()}
            ).eq("id", worker_uuid).execute()
        except Exception as e:
            # Last-seen update is best-effort; don't surface as ERROR.
            logger.warning(f"_touch_last_seen({worker_uuid}) failed: {e}")

    def increment_violation_count(self, worker_uuid: str) -> None:
        """Increment total_violations counter (client-side read-modify-write).

        We don't have a Postgres `increment` RPC set up yet, so this races if
        multiple sessions write the same worker concurrently. For Phase 2C.2
        (single-process simulator) that's fine.
        """
        try:
            row = (
                self.client.table("workers")
                .select("total_violations")
                .eq("id", worker_uuid)
                .limit(1)
                .execute()
            )
            current = (row.data[0]["total_violations"] or 0) if row.data else 0
            self.client.table("workers").update(
                {"total_violations": current + 1}
            ).eq("id", worker_uuid).execute()
        except Exception as e:
            logger.warning(f"increment_violation_count({worker_uuid}) failed: {e}")

    def reset_session_cache(self) -> None:
        """Clear the in-memory tracker_id → worker_uuid map."""
        self._cache.clear()


# ---------- 4. EventUploader ----------


class EventUploader:
    """Upload ViolationEvent objects + snapshots to Supabase."""

    def __init__(
        self,
        supabase: SupabaseClient,
        camera_id: str,
        worker_registry: WorkerRegistry,
    ) -> None:
        self.client = supabase.client
        self.camera_id = camera_id
        self.worker_registry = worker_registry

    # ---- snapshot upload ----

    def upload_snapshot(self, local_path: str, event_id: str) -> Optional[str]:
        """Upload local JPEG to bucket as `violations/{date}/{event_id}.jpg`.

        Returns the public URL on success, None on failure. Defensive — never raises.
        """
        try:
            local = Path(local_path)
            if not local.exists():
                logger.error(f"snapshot file not found: {local}")
                return None
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            storage_path = f"violations/{date_str}/{event_id}.jpg"
            data = local.read_bytes()
            # `upsert='true'` (string) avoids "already exists" 409 on re-runs.
            self.client.storage.from_(SNAPSHOT_BUCKET).upload(
                path=storage_path,
                file=data,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
            public_url = self.client.storage.from_(SNAPSHOT_BUCKET).get_public_url(storage_path)
            return public_url
        except Exception as e:
            logger.error(f"upload_snapshot('{local_path}', {event_id}) failed: {e}")
            return None

    def delete_snapshot(self, storage_path: str) -> bool:
        """Test-only: delete a snapshot from the bucket. Returns True on success."""
        try:
            self.client.storage.from_(SNAPSHOT_BUCKET).remove([storage_path])
            return True
        except Exception as e:
            logger.warning(f"delete_snapshot('{storage_path}') failed: {e}")
            return False

    # ---- event row upload ----

    def upload_event(self, event: ViolationEvent) -> Optional[str]:
        """INSERT a NEW event into the events table.

        - Resolves tracker_id ("W%03d") → worker uuid (creating if needed)
        - Uploads the snapshot if event.snapshot_path is set
        - Inserts the row with id = event.event_id (so resolution updates can target it)
        Returns the event uuid on success, None on failure.
        """
        try:
            tracker_id = f"W{event.worker_id:03d}"
            worker_uuid = self.worker_registry.get_or_create_worker(tracker_id)
            if worker_uuid is None:
                return None

            snapshot_url: Optional[str] = None
            if event.snapshot_path:
                snapshot_url = self.upload_snapshot(event.snapshot_path, event.event_id)

            payload = {
                "id": event.event_id,
                "camera_id": self.camera_id,
                "worker_id": worker_uuid,
                "violation_type": event.violation_type,
                "violation_label_tr": event.violation_label_tr,
                "started_at": event.started_at.astimezone(timezone.utc).isoformat(),
                "ended_at": (
                    event.ended_at.astimezone(timezone.utc).isoformat()
                    if event.ended_at
                    else None
                ),
                "duration_seconds": float(event.duration_seconds),
                "snapshot_url": snapshot_url,
                "confidence_avg": float(event.confidence_avg),
                "is_active": bool(event.is_active),
                "metadata": {"tracker_id": tracker_id},
            }
            self.client.table("events").insert(payload).execute()
            logger.info(
                f"event INSERT id={event.event_id[:8]} type={event.violation_type} "
                f"worker={tracker_id} snap={'yes' if snapshot_url else 'no'}"
            )
            return event.event_id
        except Exception as e:
            logger.error(f"upload_event({event.event_id}) failed: {e}")
            return None

    def upload_event_resolution(self, event: ViolationEvent) -> bool:
        """UPDATE an existing events row to mark it resolved.

        Sets ended_at, duration_seconds, is_active=False, refreshes confidence_avg.
        Also increments worker.total_violations once the violation is fully closed.
        Returns True on success.
        """
        try:
            payload = {
                "ended_at": (
                    event.ended_at.astimezone(timezone.utc).isoformat()
                    if event.ended_at
                    else datetime.now(timezone.utc).isoformat()
                ),
                "duration_seconds": float(event.duration_seconds),
                "is_active": False,
                "confidence_avg": float(event.confidence_avg),
            }
            self.client.table("events").update(payload).eq("id", event.event_id).execute()
            # Bump worker's total_violations counter. Resolve via DB (not the
            # in-memory cache, which may have been reset by a loop boundary
            # between the INSERT and this UPDATE).
            tracker_id = f"W{event.worker_id:03d}"
            worker_uuid = self.worker_registry.get_or_create_worker(tracker_id)
            if worker_uuid:
                self.worker_registry.increment_violation_count(worker_uuid)
            logger.info(
                f"event UPDATE id={event.event_id[:8]} resolved duration={event.duration_seconds:.1f}s"
            )
            return True
        except Exception as e:
            logger.error(f"upload_event_resolution({event.event_id}) failed: {e}")
            return False


__all__ = ["SupabaseClient", "SiteManager", "WorkerRegistry", "EventUploader"]


# Suppress an unused-import warning when this module is consumed by callers
# that don't reference _uuid directly (it's here so ad-hoc test scripts can
# generate event ids without an extra import).
_ = _uuid
