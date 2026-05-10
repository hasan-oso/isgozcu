"""Side-by-side v2 vs v3 webcam shoot-out (Phase 2A.3.D).

Loads both RT-DETR checkpoints, runs them on the same webcam frames for
30 seconds, and reports per-class detection counts plus the headline
"Vest : NO-Vest" and "Hardhat : NO-Hardhat" ratios for each model.

Both ratios are coin-flip-style metrics — the model that's *closer to 1.0*
when the user is actually wearing the gear is the more confident one.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# UTF-8 console for Turkish output
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

from backend.inference.rtdetr_detector import RTDetrDetector  # noqa: E402
from backend.inference.text_render import put_text_outlined  # noqa: E402


# ---------- config ----------

V2_PATH = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v2"
V3_PATH = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v3"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reports"

CONF_THRESHOLD = 0.25
DURATION_SEC = 30.0

WINDOW_TITLE = "İşGözcü - v2 vs v3 Webcam Karşılaştırması"
WINDOW_W = 1600
WINDOW_H = 720
PANEL_W = WINDOW_W // 2  # 800
FRAME_PANEL_H = 450
STATS_PANEL_H = WINDOW_H - 60 - FRAME_PANEL_H  # banner is 60 px

CLASS_COLORS = {
    "Hardhat": (60, 220, 60),
    "Safety Vest": (60, 220, 60),
    "Mask": (60, 220, 60),
    "NO-Hardhat": (50, 60, 230),
    "NO-Safety Vest": (50, 60, 230),
    "NO-Mask": (50, 60, 230),
    "Person": (220, 130, 50),
    "Safety Cone": (0, 165, 255),
    "machinery": (200, 0, 200),
    "vehicle": (0, 200, 255),
}
DEFAULT_COLOR = (180, 180, 180)


# ---------- helpers ----------


def annotate(frame: np.ndarray, detections, names: dict[int, str]) -> np.ndarray:
    out = frame.copy()
    if len(detections) == 0:
        return out
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int).tolist()
        cid = int(detections.class_id[i])
        conf = float(detections.confidence[i])
        name = names.get(cid, str(cid))
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_stats_panel(
    label: str,
    counts: dict[str, int],
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (15, 22, 36)
    put_text_outlined(panel, label, (16, 28), scale=0.85, thickness=2, fg=(140, 220, 255))

    classes_to_show = [
        "Person",
        "Hardhat",
        "Safety Vest",
        "NO-Hardhat",
        "NO-Safety Vest",
        "NO-Mask",
        "Mask",
        "vehicle",
        "machinery",
        "Safety Cone",
    ]
    y = 56
    for cls in classes_to_show:
        n = counts.get(cls, 0)
        if n == 0 and cls not in ("Hardhat", "Safety Vest", "NO-Hardhat", "NO-Safety Vest", "Person"):
            continue  # skip noise classes that never fire
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        bar_w_max = width - 220
        max_count = max([1] + [counts.get(c, 0) for c in classes_to_show])
        bar_w = int(bar_w_max * (n / max_count)) if max_count > 0 else 0
        cv2.rectangle(panel, (200, y - 12), (200 + bar_w, y + 4), color, -1)
        put_text_outlined(panel, f"{cls}", (16, y), scale=0.55, fg=color)
        put_text_outlined(panel, f"{n}", (width - 60, y), scale=0.6, thickness=2)
        y += 22
        if y > height - 14:
            break
    return panel


def render_banner(elapsed: float, duration: float, status: str, fps: float) -> np.ndarray:
    banner = np.zeros((60, WINDOW_W, 3), dtype=np.uint8)
    banner[:] = (10, 16, 28)
    secs = max(0.0, duration - elapsed)
    timer = f"{int(secs // 60):02d}:{int(secs % 60):02d} / {int(duration // 60):02d}:{int(duration % 60):02d}"
    put_text_outlined(banner, "v2 vs v3 — 30 saniyelik karşılaştırma", (16, 38), scale=0.7, thickness=2)
    put_text_outlined(banner, timer, (WINDOW_W // 2 - 80, 38), scale=0.85, thickness=2, fg=(60, 220, 255))
    put_text_outlined(banner, f"FPS {fps:.1f}", (WINDOW_W - 240, 38), scale=0.6)
    put_text_outlined(banner, status, (WINDOW_W - 140, 38), scale=0.55, fg=(180, 220, 240))
    return banner


def ratio(positive: int, negative: int) -> Optional[float]:
    """Vest/(Vest + NO-Vest). Returns None if both are 0."""
    total = positive + negative
    return positive / total if total > 0 else None


def main() -> int:
    print("=" * 64)
    print("v2 vs v3 webcam karşılaştırma testi")
    print("=" * 64)

    if not V2_PATH.exists():
        print(f"[hata] v2 modeli bulunamadı: {V2_PATH}")
        return 1
    if not V3_PATH.exists():
        print(f"[hata] v3 modeli bulunamadı: {V3_PATH}")
        return 1

    print("Modeller yükleniyor...")
    t0 = time.time()
    det_v2 = RTDetrDetector(model_path=str(V2_PATH), conf_threshold=CONF_THRESHOLD)
    print(f"  v2 yüklendi ({time.time() - t0:.1f}s)")
    t0 = time.time()
    det_v3 = RTDetrDetector(model_path=str(V3_PATH), conf_threshold=CONF_THRESHOLD)
    print(f"  v3 yüklendi ({time.time() - t0:.1f}s)")

    names_v2: dict[int, str] = det_v2.class_names if isinstance(det_v2.class_names, dict) else {i: n for i, n in enumerate(det_v2.class_names)}
    names_v3: dict[int, str] = det_v3.class_names if isinstance(det_v3.class_names, dict) else {i: n for i, n in enumerate(det_v3.class_names)}

    print("Webcam açılıyor...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[hata] Webcam açılamadı.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, WINDOW_W, WINDOW_H)

    counts_v2: dict[str, int] = defaultdict(int)
    counts_v3: dict[str, int] = defaultdict(int)
    frames_processed = 0
    started = False
    start_time = 0.0
    fps_smooth = 0.0
    last_t = time.time()

    print("\nKameraya bak. Baret + yelek tak.")
    print("SPACE = başlat   |   Q = iptal\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # FPS smoothing for the banner
            now = time.time()
            inst_fps = 1.0 / max(now - last_t, 0.001)
            last_t = now
            fps_smooth = inst_fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst_fps

            # Run BOTH models on the same frame
            d_v2 = det_v2.detect(frame)
            d_v3 = det_v3.detect(frame)

            if started:
                for i in range(len(d_v2)):
                    cid = int(d_v2.class_id[i])
                    counts_v2[names_v2.get(cid, str(cid))] += 1
                for i in range(len(d_v3)):
                    cid = int(d_v3.class_id[i])
                    counts_v3[names_v3.get(cid, str(cid))] += 1
                frames_processed += 1

            ann_v2 = annotate(frame, d_v2, names_v2)
            ann_v3 = annotate(frame, d_v3, names_v3)
            ann_v2 = cv2.resize(ann_v2, (PANEL_W, FRAME_PANEL_H), interpolation=cv2.INTER_AREA)
            ann_v3 = cv2.resize(ann_v3, (PANEL_W, FRAME_PANEL_H), interpolation=cv2.INTER_AREA)

            # Label each panel top-left
            cv2.rectangle(ann_v2, (0, 0), (60, 28), (0, 0, 0), -1)
            cv2.putText(ann_v2, "v2", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 220, 255), 2, cv2.LINE_AA)
            cv2.rectangle(ann_v3, (0, 0), (60, 28), (0, 0, 0), -1)
            cv2.putText(ann_v3, "v3", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 220, 255), 2, cv2.LINE_AA)

            stats_v2 = render_stats_panel("v2 toplam", counts_v2, PANEL_W, STATS_PANEL_H)
            stats_v3 = render_stats_panel("v3 toplam", counts_v3, PANEL_W, STATS_PANEL_H)

            elapsed = (now - start_time) if started else 0.0
            status = "ÇALIŞIYOR" if started else "SPACE = başlat"
            banner = render_banner(elapsed, DURATION_SEC, status, fps_smooth)

            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
            canvas[0:60, 0:WINDOW_W] = banner
            canvas[60:60 + FRAME_PANEL_H, 0:PANEL_W] = ann_v2
            canvas[60:60 + FRAME_PANEL_H, PANEL_W:WINDOW_W] = ann_v3
            stats_y = 60 + FRAME_PANEL_H
            canvas[stats_y:stats_y + STATS_PANEL_H, 0:PANEL_W] = stats_v2
            canvas[stats_y:stats_y + STATS_PANEL_H, PANEL_W:WINDOW_W] = stats_v3

            cv2.imshow(WINDOW_TITLE, canvas)

            if started and elapsed >= DURATION_SEC:
                print("Süre doldu.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Kullanıcı iptal etti.")
                break
            if key == 32 and not started:  # SPACE
                started = True
                start_time = time.time()
                counts_v2.clear()
                counts_v3.clear()
                frames_processed = 0
                print("Başladı — 30 saniye boyunca her iki model de aynı kareleri işleyecek.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if frames_processed == 0:
        print("Hiç kare işlenmedi.")
        return 0

    # ---------- comparison report ----------
    print()
    print("=" * 64)
    print(f"SONUÇLAR — {frames_processed} kare işlendi")
    print("=" * 64)

    classes_in_either = sorted(set(counts_v2.keys()) | set(counts_v3.keys()))

    def pct_diff(a: int, b: int) -> str:
        if a == 0 and b == 0:
            return "—"
        if a == 0:
            return "+∞"
        return f"{(b - a) / a * 100:+.1f}%"

    print(f"{'Class':<18}{'v2':>10}{'v3':>10}  {'Δ':<10}")
    print("-" * 48)
    for cls in classes_in_either:
        a = counts_v2.get(cls, 0)
        b = counts_v3.get(cls, 0)
        print(f"{cls:<18}{a:>10}{b:>10}  {pct_diff(a, b):<10}")

    # Headline ratios
    print()
    print("Vest : NO-Vest oranı (1.0 = mükemmel uyum, 0.5 = yazı-tura)")
    v2_vest_ratio = ratio(counts_v2.get("Safety Vest", 0), counts_v2.get("NO-Safety Vest", 0))
    v3_vest_ratio = ratio(counts_v3.get("Safety Vest", 0), counts_v3.get("NO-Safety Vest", 0))
    print(f"  v2: {f'{v2_vest_ratio:.3f}' if v2_vest_ratio is not None else '—'}")
    print(f"  v3: {f'{v3_vest_ratio:.3f}' if v3_vest_ratio is not None else '—'}")
    if v2_vest_ratio is not None and v3_vest_ratio is not None:
        better = "v3" if abs(v3_vest_ratio - 1.0) < abs(v2_vest_ratio - 1.0) else "v2"
        print(f"  Daha güvenli: {better}")

    print()
    print("Hardhat : NO-Hardhat oranı (1.0 = mükemmel uyum)")
    v2_hat_ratio = ratio(counts_v2.get("Hardhat", 0), counts_v2.get("NO-Hardhat", 0))
    v3_hat_ratio = ratio(counts_v3.get("Hardhat", 0), counts_v3.get("NO-Hardhat", 0))
    print(f"  v2: {f'{v2_hat_ratio:.3f}' if v2_hat_ratio is not None else '—'}")
    print(f"  v3: {f'{v3_hat_ratio:.3f}' if v3_hat_ratio is not None else '—'}")
    if v2_hat_ratio is not None and v3_hat_ratio is not None:
        better = "v3" if abs(v3_hat_ratio - 1.0) < abs(v2_hat_ratio - 1.0) else "v2"
        print(f"  Daha güvenli: {better}")

    # Save JSON report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": DURATION_SEC,
        "frames_processed": frames_processed,
        "conf_threshold": CONF_THRESHOLD,
        "v2": {
            "model_path": str(V2_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "detections_per_class": dict(counts_v2),
            "vest_ratio": v2_vest_ratio,
            "hardhat_ratio": v2_hat_ratio,
        },
        "v3": {
            "model_path": str(V3_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "detections_per_class": dict(counts_v3),
            "vest_ratio": v3_vest_ratio,
            "hardhat_ratio": v3_hat_ratio,
        },
        "verdict": {
            "vest": (
                None
                if v2_vest_ratio is None or v3_vest_ratio is None
                else ("v3" if abs(v3_vest_ratio - 1.0) < abs(v2_vest_ratio - 1.0) else "v2")
            ),
            "hardhat": (
                None
                if v2_hat_ratio is None or v3_hat_ratio is None
                else ("v3" if abs(v3_hat_ratio - 1.0) < abs(v2_hat_ratio - 1.0) else "v2")
            ),
        },
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"v2_vs_v3_comparison_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Rapor kaydedildi: {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
