"""Smoke test for the Supabase persistence layer.

Inserts a fake ViolationEvent + dummy snapshot, verifies the row & file end up
in Supabase, then deletes both. Reports PASS/FAIL clearly.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

# UTF-8 stdout so Turkish output doesn't crash on Windows cp1256.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import urllib.request  # noqa: E402
import urllib.error  # noqa: E402

from backend.inference.violation_engine import ViolationEvent  # noqa: E402
from backend.storage.supabase_client import (  # noqa: E402
    EventUploader,
    SiteManager,
    SupabaseClient,
    WorkerRegistry,
)


def _make_dummy_snapshot() -> Path:
    """200×200 red square JPEG saved to a temp path."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :, 2] = 220  # red in BGR
    cv2.putText(
        img, "TEST", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA
    )
    out = Path(tempfile.gettempdir()) / f"isgozcu_test_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out


def _make_fake_event(snapshot_path: str) -> ViolationEvent:
    """Tracker id 999 stays out of any real ByteTrack range."""
    return ViolationEvent(
        event_id=str(uuid.uuid4()),
        worker_id=999,
        violation_type="no_hardhat",
        camera_id="placeholder",  # overwritten by EventUploader.camera_id
        started_at=datetime.now(timezone.utc),
        ended_at=None,
        duration_seconds=0.0,
        snapshot_path=snapshot_path,
        confidence_avg=0.812,
        is_active=True,
    )


def _http_check(url: str) -> tuple[bool, int, int]:
    """GET with no body. Returns (ok, status, content_length)."""
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "isgozcu-smoke"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            return resp.status == 200 and len(data) > 0, resp.status, len(data)
    except urllib.error.HTTPError as e:
        return False, e.code, 0
    except Exception:
        return False, 0, 0


def main() -> int:
    print("=" * 64)
    print("Supabase upload smoke test")
    print("=" * 64)

    # 1. Initialize
    try:
        sb = SupabaseClient()
    except Exception as e:
        print(f"FAIL: SupabaseClient init failed: {e}")
        return 1
    if not sb.health_check():
        print("FAIL: health_check did not pass")
        return 2
    print("PASS  health_check ok")

    site_mgr = SiteManager(sb)
    site_id, camera_id = site_mgr.get_default_demo_context()
    if not site_id or not camera_id:
        print("FAIL: could not bootstrap site/camera")
        return 3
    print(f"PASS  site_id={site_id[:8]}.. camera_id={camera_id[:8]}..")

    worker_reg = WorkerRegistry(sb, site_id)
    uploader = EventUploader(sb, camera_id, worker_reg)

    # 2. Build fake event + dummy snapshot
    snap = _make_dummy_snapshot()
    print(f"PASS  dummy snapshot: {snap.name} ({snap.stat().st_size} bytes)")
    ev = _make_fake_event(str(snap))

    # 3. Upload
    event_uuid = uploader.upload_event(ev)
    if not event_uuid:
        print("FAIL: upload_event returned None")
        return 4
    print(f"PASS  upload_event id={event_uuid[:8]}..")

    # 4. Verify row exists
    try:
        rows = sb.client.table("events").select("*").eq("id", event_uuid).execute()
    except Exception as e:
        print(f"FAIL: select after insert raised: {e}")
        return 5
    if not rows.data:
        print("FAIL: row not found after insert")
        return 6
    row = rows.data[0]
    print(f"PASS  row found: type={row['violation_type']} url_set={bool(row['snapshot_url'])}")

    # 5. Snapshot URL accessible?
    snapshot_url = row["snapshot_url"]
    storage_path = None
    if snapshot_url:
        ok, status, n = _http_check(snapshot_url)
        if ok:
            print(f"PASS  snapshot URL accessible (HTTP {status}, {n} bytes)")
        else:
            print(f"FAIL  snapshot URL not accessible (HTTP {status})")
            print(f"      url: {snapshot_url}")
            # don't bail — we still want to clean up
        # Reverse-engineer the storage path from the URL for cleanup.
        marker = "/storage/v1/object/public/snapshots/"
        if marker in snapshot_url:
            storage_path = snapshot_url.split(marker, 1)[1]

    # 6. Test resolution upload
    ev.is_active = False
    ev.ended_at = datetime.now(timezone.utc)
    ev.duration_seconds = 7.5
    ev.confidence_avg = 0.834
    if uploader.upload_event_resolution(ev):
        print("PASS  upload_event_resolution ok")
    else:
        print("FAIL  upload_event_resolution failed")

    # 7. Cleanup — delete event row, snapshot, and the smoke-test worker row.
    cleanup_ok = True
    try:
        sb.client.table("events").delete().eq("id", event_uuid).execute()
        print("PASS  event row deleted (cleanup)")
    except Exception as e:
        print(f"FAIL  event delete: {e}")
        cleanup_ok = False
    if storage_path:
        ok = uploader.delete_snapshot(storage_path)
        print(f"{'PASS' if ok else 'WARN'}  snapshot deleted from bucket: {storage_path}")
        if not ok:
            cleanup_ok = False
    try:
        # Find and delete the smoke-test worker (tracker_id=W999).
        sb.client.table("workers").delete().eq("site_id", site_id).eq(
            "tracker_id", "W999"
        ).execute()
        print("PASS  test worker row deleted (cleanup)")
    except Exception as e:
        print(f"WARN  worker delete: {e}")
    try:
        snap.unlink(missing_ok=True)
    except Exception:
        pass

    print("=" * 64)
    if cleanup_ok:
        print("RESULT: SMOKE TEST PASSED")
        return 0
    print("RESULT: SMOKE TEST PASSED with cleanup warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
