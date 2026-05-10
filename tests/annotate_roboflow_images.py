"""Roboflow re-annotation tool (Phase 2A.4).

Adapted from `tests/annotate_training_images.py` but with two important
differences:

1. **Dual-source seeding**. Each image is shown with TWO opinions:
     - the original Roboflow annotation (cyan dashed border)
     - the v3 model's prediction (yellow border)
   The user reconciles them — keep one, edit the other, draw their own,
   or any mix. The first edit a user makes "claims" the box (it becomes
   green and is treated as ground truth from then on).

2. **Skip key (K)**. If a Roboflow image is unsalvageable (mislabeled
   class confusion across the whole frame, or the image content itself
   is garbage), press K to mark it skipped and move on. Skipped images
   are tracked separately in `progress.json` so the COCO export can
   exclude them cleanly.

Storage layout (mirrors `select_images_for_reannotation.py`):

    data/roboflow_reannotation/
        ├── images/                            (input, written by the selector)
        ├── original_annotations.json          (input, Roboflow ground truth)
        ├── annotations/                       (output, per-image cleaned JSON)
        ├── progress.json                      (annotated_files, skipped_files)
        └── selection_report.json              (read-only metadata)

Bbox-validation guards (warn-only, don't auto-fix):
  * any box outside image bounds
  * any box smaller than 100 px²
  * any box covering more than 80 % of image area
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
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

ID2LABEL: dict[int, str] = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}

KEY_TO_CLASS: dict[int, str] = {
    ord("1"): "Hardhat",
    ord("2"): "NO-Hardhat",
    ord("3"): "Safety Vest",
    ord("4"): "NO-Safety Vest",
    ord("5"): "Mask",
    ord("6"): "NO-Mask",
    ord("7"): "Person",
    ord("8"): "machinery",
    ord("9"): "vehicle",
    ord("0"): "Safety Cone",
}

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
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

# Source-tagging colors for the dual-source UI
SRC_ROBOFLOW_COLOR = (255, 200, 60)  # cyan-ish
SRC_V3_COLOR = (60, 220, 255)        # yellow-ish
SRC_USER_COLOR = (60, 230, 60)       # green

ROOT = PROJECT_ROOT / "data" / "roboflow_reannotation"
IMG_DIR = ROOT / "images"
ANNO_DIR = ROOT / "annotations"
ORIG_ANNO_PATH = ROOT / "original_annotations.json"
PROGRESS_PATH = ROOT / "progress.json"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v3"
SEED_CONF_THRESHOLD = 0.20

WINDOW_W = 1600
WINDOW_H = 900
IMG_W = 1280
IMG_H = 720
PANEL_X = IMG_W
PANEL_W = WINDOW_W - IMG_W  # 320
STATUS_Y = IMG_H
STATUS_H = WINDOW_H - IMG_H  # 180

WINDOW_TITLE = "İşGözcü - Roboflow Yeniden Etiketleme"

MIN_BOX_PX = 12
WARN_TINY_AREA = 100
WARN_HUGE_FRACTION = 0.8


# ---------- data ----------


@dataclass
class AnnoBox:
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    source: str = "user"  # "roboflow" | "v3" | "user"

    @property
    def x(self) -> int:
        return self.x1

    @property
    def y(self) -> int:
        return self.y1

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def claim_user(self) -> None:
        self.source = "user"


def collect_images() -> list[Path]:
    if not IMG_DIR.exists():
        return []
    return sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.jpeg")) + sorted(IMG_DIR.glob("*.png"))


def load_progress() -> tuple[set[str], set[str]]:
    if not PROGRESS_PATH.exists():
        return set(), set()
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        return set(data.get("annotated_files", [])), set(data.get("skipped_files", []))
    except Exception:
        return set(), set()


def save_progress(annotated: set[str], skipped: set[str]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(
            {
                "annotated_files": sorted(annotated),
                "skipped_files": sorted(skipped),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def annotation_path_for(image_path: Path) -> Path:
    return ANNO_DIR / (image_path.stem + ".json")


def save_annotation(image_path: Path, boxes: list[AnnoBox], image_w: int, image_h: int) -> Path:
    ANNO_DIR.mkdir(parents=True, exist_ok=True)
    out = annotation_path_for(image_path)
    payload = {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "image_width": image_w,
        "image_height": image_h,
        "source": "roboflow_reannotated",
        "boxes": [
            {
                "class_id": b.class_id,
                "class_name": b.class_name,
                "bbox": [b.x, b.y, b.w, b.h],
            }
            for b in boxes
        ],
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_annotation(image_path: Path) -> Optional[list[AnnoBox]]:
    p = annotation_path_for(image_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        out: list[AnnoBox] = []
        for b in data.get("boxes", []):
            x, y, w, h = b["bbox"]
            out.append(
                AnnoBox(
                    class_id=int(b["class_id"]),
                    class_name=b["class_name"],
                    x1=int(x),
                    y1=int(y),
                    x2=int(x) + int(w),
                    y2=int(y) + int(h),
                    source="user",
                )
            )
        return out
    except Exception:
        return None


def load_roboflow_originals() -> dict[str, list[dict]]:
    if not ORIG_ANNO_PATH.exists():
        return {}
    try:
        return json.loads(ORIG_ANNO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_boxes(boxes: list[AnnoBox], image_w: int, image_h: int) -> list[str]:
    """Return human-readable warnings; never auto-fixes anything."""
    warns: list[str] = []
    img_area = max(1, image_w * image_h)
    for i, b in enumerate(boxes):
        if b.x1 < 0 or b.y1 < 0 or b.x2 > image_w or b.y2 > image_h:
            warns.append(f"#{i + 1} {b.class_name}: kutu sınır dışında")
        if b.area() < WARN_TINY_AREA:
            warns.append(f"#{i + 1} {b.class_name}: çok küçük ({b.area()}px²)")
        if b.area() / img_area > WARN_HUGE_FRACTION:
            pct = int(100 * b.area() / img_area)
            warns.append(f"#{i + 1} {b.class_name}: kare %{pct} kapsıyor")
    return warns


# ---------- detector + roboflow → AnnoBox ----------


def v3_initial_boxes(detector: RTDetrDetector, frame_bgr: np.ndarray) -> list[AnnoBox]:
    d = detector.detect(frame_bgr)
    out: list[AnnoBox] = []
    if len(d) == 0:
        return out
    for i in range(len(d)):
        x1, y1, x2, y2 = d.xyxy[i].astype(int).tolist()
        cid = int(d.class_id[i])
        name = ID2LABEL.get(cid, f"unknown_{cid}")
        out.append(
            AnnoBox(class_id=cid, class_name=name, x1=x1, y1=y1, x2=x2, y2=y2, source="v3")
        )
    return out


def roboflow_initial_boxes(orig_records: list[dict], image_w: int, image_h: int, scale_x: float, scale_y: float) -> list[AnnoBox]:
    """Convert Roboflow's COCO-style boxes (in original-image coords) into AnnoBox
    (in display-image coords)."""
    out: list[AnnoBox] = []
    for r in orig_records:
        x, y, w, h = r["bbox"]
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + w) * scale_x))
        y2 = int(round((y + h) * scale_y))
        # Clamp to image
        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        x2 = max(0, min(image_w, x2))
        y2 = max(0, min(image_h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(r["class_id"])
        name = r.get("class_name") or ID2LABEL.get(cid, f"unknown_{cid}")
        out.append(
            AnnoBox(class_id=cid, class_name=name, x1=x1, y1=y1, x2=x2, y2=y2, source="roboflow")
        )
    return out


# ---------- rendering ----------


def draw_translucent(canvas: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.6) -> None:
    H, W = canvas.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    sub = canvas[y1:y2, x1:x2]
    overlay = np.zeros_like(sub)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)


def draw_dashed_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int = 2, dash: int = 10) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for x in range(min(x1, x2), max(x1, x2), dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, max(x1, x2)), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + dash, max(x1, x2)), y2), color, thickness, cv2.LINE_AA)
    for y in range(min(y1, y2), max(y1, y2), dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, max(y1, y2))), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + dash, max(y1, y2))), color, thickness, cv2.LINE_AA)


def render_image_panel(
    image: np.ndarray,
    boxes: list[AnnoBox],
    selected_idx: Optional[int],
    drawing: bool,
    draw_p1: Optional[tuple[int, int]],
    draw_p2: Optional[tuple[int, int]],
) -> np.ndarray:
    canvas = image.copy()
    for i, b in enumerate(boxes):
        cls_color = CLASS_COLORS.get(b.class_name, DEFAULT_COLOR)
        is_sel = i == selected_idx
        thickness = 4 if is_sel else 2

        if b.source == "roboflow":
            # Dashed cyan border (historical reference)
            draw_dashed_rect(canvas, (b.x1, b.y1), (b.x2, b.y2), SRC_ROBOFLOW_COLOR, thickness, dash=8)
        elif b.source == "v3":
            # Solid yellow border (current model prediction)
            cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), SRC_V3_COLOR, thickness, cv2.LINE_AA)
        else:  # "user"
            # Solid box in CLASS color (committed truth)
            cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), cls_color, thickness, cv2.LINE_AA)

        if is_sel:
            draw_dashed_rect(canvas, (b.x1 - 4, b.y1 - 4), (b.x2 + 4, b.y2 + 4), (0, 255, 255), 2, dash=6)

        src_tag = {"roboflow": "RF", "v3": "v3", "user": "✓"}.get(b.source, "?")
        label = f"{b.class_name} [{src_tag}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(b.y1 - 4, th + 4)
        bg = cls_color if b.source == "user" else SRC_ROBOFLOW_COLOR if b.source == "roboflow" else SRC_V3_COLOR
        cv2.rectangle(canvas, (b.x1, ly - th - 6), (b.x1 + tw + 6, ly + 2), bg, -1)
        cv2.putText(canvas, label, (b.x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    if drawing and draw_p1 and draw_p2:
        cv2.rectangle(canvas, draw_p1, draw_p2, (0, 255, 255), 2, cv2.LINE_AA)
        draw_dashed_rect(canvas, draw_p1, draw_p2, (255, 255, 255), 1, 6)

    return canvas


def render_info_panel(
    image_path: Path,
    image_idx: int,
    total: int,
    boxes: list[AnnoBox],
    selected_idx: Optional[int],
    pending_class: bool,
    rf_count: int,
    v3_count: int,
    user_count: int,
    warnings_text: list[str],
) -> np.ndarray:
    panel = np.zeros((IMG_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (15, 22, 36)

    y = 30
    put_text_outlined(panel, "Yeniden Etiketleme", (12, y), scale=0.78, thickness=2)
    y += 26
    put_text_outlined(panel, f"{image_idx + 1} / {total}", (12, y), scale=0.65, fg=(140, 200, 255))
    y += 22
    put_text_outlined(panel, image_path.stem[:38], (12, y), scale=0.42, fg=(140, 165, 200))
    y += 22

    # Progress bar
    bar_w = PANEL_W - 24
    bar_x = 12
    bar_h = 8
    cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (40, 60, 90), -1)
    fill = int(bar_w * (image_idx / max(1, total)))
    cv2.rectangle(panel, (bar_x, y), (bar_x + fill, y + bar_h), (60, 200, 255), -1)
    y += bar_h + 18

    # Source counts
    put_text_outlined(panel, "Kaynaklar:", (12, y), scale=0.55, thickness=1)
    y += 22
    put_text_outlined(panel, f"  Roboflow: {rf_count}", (12, y), scale=0.5, fg=SRC_ROBOFLOW_COLOR)
    y += 18
    put_text_outlined(panel, f"  v3:       {v3_count}", (12, y), scale=0.5, fg=SRC_V3_COLOR)
    y += 18
    put_text_outlined(panel, f"  Onaylı:   {user_count}", (12, y), scale=0.5, fg=SRC_USER_COLOR)
    y += 22

    # Selected box info
    if selected_idx is not None and 0 <= selected_idx < len(boxes):
        b = boxes[selected_idx]
        put_text_outlined(panel, "Seçili:", (12, y), scale=0.55, fg=(0, 255, 255))
        y += 18
        put_text_outlined(panel, f"  {b.class_name}", (12, y), scale=0.5, fg=(0, 255, 255))
        y += 16
        put_text_outlined(panel, f"  ({b.x},{b.y}) {b.w}x{b.h}", (12, y), scale=0.42, fg=(180, 220, 240))
        y += 16
        put_text_outlined(panel, f"  kaynak: {b.source}", (12, y), scale=0.42, fg=(180, 220, 240))
        y += 22

    # Validation warnings
    if warnings_text:
        put_text_outlined(panel, "⚠ Uyarılar:", (12, y), scale=0.5, fg=(60, 180, 255))
        y += 18
        for w_text in warnings_text[:6]:
            put_text_outlined(panel, f"  {w_text}", (12, y), scale=0.42, fg=(60, 180, 255))
            y += 16

    if pending_class:
        cv2.rectangle(panel, (8, 6), (PANEL_W - 8, 70), (0, 220, 255), 2)
        put_text_outlined(panel, "Yeni kutu — sınıf seç!", (16, 36), scale=0.65, thickness=2, fg=(0, 220, 255))
        put_text_outlined(panel, "1-9 / 0 ile ata", (16, 58), scale=0.5, fg=(0, 220, 255))

    # Class hints (bottom of panel)
    y = IMG_H - 230
    put_text_outlined(panel, "Sınıf tuşları:", (12, y), scale=0.5, fg=(180, 200, 240))
    hints = [
        ("1", "Hardhat"),
        ("2", "NO-Hardhat"),
        ("3", "Safety Vest"),
        ("4", "NO-Safety Vest"),
        ("5", "Mask"),
        ("6", "NO-Mask"),
        ("7", "Person"),
        ("8", "machinery"),
        ("9", "vehicle"),
        ("0", "Safety Cone"),
    ]
    for i, (k, name) in enumerate(hints):
        col = CLASS_COLORS.get(name, DEFAULT_COLOR)
        put_text_outlined(panel, f"  [{k}] {name}", (12, y + 18 + i * 16), scale=0.42, fg=col)

    return panel


def render_status_strip(width: int, height: int, status: str, total_anno: int, total_skipped: int, total_imgs: int) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (10, 16, 28)
    put_text_outlined(strip, status, (16, 32), scale=0.65, thickness=2, fg=(220, 240, 255))
    keys = (
        "[SPACE] Onayla+İleri   [B] Geri   [K] Atla   [R] Modele döndür   "
        "[D / DEL] Seçili kutuyu sil   [Sağ tık] Hızlı sil   [Q] Kaydet+Çık"
    )
    put_text_outlined(strip, keys, (16, 64), scale=0.5, fg=(140, 180, 220))
    put_text_outlined(strip, f"Tamamlanan: {total_anno}   Atlanan: {total_skipped}   Toplam: {total_imgs}",
                      (16, 96), scale=0.5, fg=(140, 180, 220))
    return strip


# ---------- mouse interaction ----------


@dataclass
class UIState:
    boxes: list[AnnoBox] = field(default_factory=list)
    selected_idx: Optional[int] = None
    drawing: bool = False
    draw_p1: Optional[tuple[int, int]] = None
    draw_p2: Optional[tuple[int, int]] = None
    pending_new_box: Optional[tuple[int, int, int, int]] = None
    image: Optional[np.ndarray] = None
    image_w: int = IMG_W
    image_h: int = IMG_H
    dirty: bool = False


def hit_box(boxes: list[AnnoBox], x: int, y: int) -> Optional[int]:
    candidates = [(i, b) for i, b in enumerate(boxes) if b.contains(x, y)]
    if not candidates:
        return None
    return min(candidates, key=lambda kv: kv[1].area())[0]


def _on_mouse(event: int, x: int, y: int, flags: int, param: dict) -> None:
    state: UIState = param["state"]
    if x >= IMG_W or y >= IMG_H or x < 0 or y < 0:
        if state.drawing and event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
            state.draw_p1 = None
            state.draw_p2 = None
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = hit_box(state.boxes, x, y)
        if idx is not None:
            state.selected_idx = idx
            state.drawing = False
        else:
            state.selected_idx = None
            state.drawing = True
            state.draw_p1 = (x, y)
            state.draw_p2 = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.draw_p2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if state.drawing and state.draw_p1 is not None:
            x1, y1 = state.draw_p1
            x2, y2 = (x, y)
            state.draw_p1 = None
            state.draw_p2 = None
            state.drawing = False
            xa, ya = min(x1, x2), min(y1, y2)
            xb, yb = max(x1, x2), max(y1, y2)
            if (xb - xa) >= MIN_BOX_PX and (yb - ya) >= MIN_BOX_PX:
                state.pending_new_box = (xa, ya, xb, yb)
                state.dirty = True
                param["status"] = "Yeni kutu çizildi — 1-9 / 0 ile sınıf ata."
            else:
                param["status"] = "Çok küçük — yeniden çiz."
    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = hit_box(state.boxes, x, y)
        if idx is not None:
            removed = state.boxes.pop(idx)
            if state.selected_idx == idx:
                state.selected_idx = None
            elif state.selected_idx is not None and state.selected_idx > idx:
                state.selected_idx -= 1
            state.dirty = True
            param["status"] = f"Silindi: {removed.class_name} ({removed.source})"


# ---------- main loop ----------


def main() -> int:
    images = collect_images()
    if not images:
        print(f"[hata] Hiç görüntü yok: {IMG_DIR}")
        print("Önce: python tests/select_images_for_reannotation.py")
        return 1
    total = len(images)
    print(f"Bulunan görüntü: {total}")

    annotated, skipped = load_progress()
    print(f"Tamamlanan: {len(annotated)}   Atlanan: {len(skipped)}   Bekleyen: {total - len(annotated) - len(skipped)}")

    rf_originals = load_roboflow_originals()
    if not rf_originals:
        print(f"[uyarı] Roboflow orijinal etiketleri yüklenemedi: {ORIG_ANNO_PATH}")
        print("        Sadece v3 önerileri gösterilecek.")
    else:
        print(f"Roboflow orijinal etiket: {len(rf_originals)} dosya için.")

    # Find first un-handled image to resume.
    start_idx = 0
    for i, p in enumerate(images):
        if p.name not in annotated and p.name not in skipped:
            start_idx = i
            break
    else:
        start_idx = total - 1

    print(f"Model yükleniyor: {MODEL_PATH.name} ...")
    detector = RTDetrDetector(model_path=str(MODEL_PATH), conf_threshold=SEED_CONF_THRESHOLD)
    print()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, WINDOW_W, WINDOW_H)

    state = UIState()
    param = {"state": state, "status": "Hazır."}
    cv2.setMouseCallback(WINDOW_TITLE, _on_mouse, param)

    current_idx = start_idx

    def load_image(idx: int, force_redetect: bool = False) -> bool:
        path = images[idx]
        img = cv2.imread(str(path))
        if img is None:
            print(f"[hata] Okunamadı: {path}")
            return False

        # Display panel is fixed at 1280x720; scale image to fit while
        # remembering the scale factors so we can map Roboflow's original
        # coords correctly.
        orig_h, orig_w = img.shape[:2]
        if (orig_w, orig_h) != (IMG_W, IMG_H):
            disp = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        else:
            disp = img
        scale_x = IMG_W / orig_w
        scale_y = IMG_H / orig_h

        state.image = disp
        state.image_w = IMG_W
        state.image_h = IMG_H
        state.selected_idx = None
        state.drawing = False
        state.draw_p1 = None
        state.draw_p2 = None
        state.pending_new_box = None

        existing = None if force_redetect else load_annotation(path)
        if existing is not None:
            state.boxes = existing
            state.dirty = False
            param["status"] = f"Mevcut etiket yüklendi ({len(existing)} kutu)."
        else:
            rf_recs = rf_originals.get(path.name, [])
            rf_boxes = roboflow_initial_boxes(rf_recs, IMG_W, IMG_H, scale_x, scale_y)
            v3_boxes = v3_initial_boxes(detector, disp)
            state.boxes = rf_boxes + v3_boxes
            state.dirty = False
            param["status"] = (
                f"Roboflow: {len(rf_boxes)} kutu  |  v3: {len(v3_boxes)} kutu  →  doğru olanları onayla."
            )
        return True

    if not load_image(current_idx):
        return 1

    try:
        while True:
            path = images[current_idx]

            rf_count = sum(1 for b in state.boxes if b.source == "roboflow")
            v3_count = sum(1 for b in state.boxes if b.source == "v3")
            user_count = sum(1 for b in state.boxes if b.source == "user")
            warns = validate_boxes(state.boxes, state.image_w, state.image_h)

            img_panel = render_image_panel(
                state.image,
                state.boxes,
                state.selected_idx,
                state.drawing,
                state.draw_p1,
                state.draw_p2,
            )
            info_panel = render_info_panel(
                path,
                current_idx,
                total,
                state.boxes,
                state.selected_idx,
                state.pending_new_box is not None,
                rf_count,
                v3_count,
                user_count,
                warns,
            )
            status_strip = render_status_strip(
                WINDOW_W, STATUS_H, param["status"], len(annotated), len(skipped), total
            )

            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
            canvas[0:IMG_H, 0:IMG_W] = img_panel
            canvas[0:IMG_H, IMG_W:WINDOW_W] = info_panel
            canvas[STATUS_Y:WINDOW_H, 0:WINDOW_W] = status_strip

            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), ord("Q")):
                # Save in-flight edits, no advance.
                if state.dirty or load_annotation(path) is None:
                    # Mark all boxes user-claimed for save coherence
                    for b in state.boxes:
                        if b.source != "user":
                            b.claim_user()
                    save_annotation(path, state.boxes, state.image_w, state.image_h)
                save_progress(annotated, skipped)
                print("Kullanıcı çıkışı, ilerleme kaydedildi.")
                break

            if key == 32:  # SPACE = save & next
                if state.pending_new_box is not None:
                    param["status"] = "Önce yeni kutuya sınıf ata (1-9 / 0)."
                    continue
                # Convert any remaining roboflow/v3 boxes to user-claimed at save.
                for b in state.boxes:
                    if b.source != "user":
                        b.claim_user()
                save_annotation(path, state.boxes, state.image_w, state.image_h)
                annotated.add(path.name)
                skipped.discard(path.name)
                save_progress(annotated, skipped)
                if current_idx + 1 < total:
                    current_idx += 1
                    load_image(current_idx)
                else:
                    param["status"] = "Tüm görüntüler tamamlandı 🎉 — Q ile çık."
                continue

            if key in (ord("k"), ord("K")):
                # Mark image as skipped; don't write annotation.
                annotated.discard(path.name)
                skipped.add(path.name)
                save_progress(annotated, skipped)
                if current_idx + 1 < total:
                    current_idx += 1
                    load_image(current_idx)
                    param["status"] = f"Atlandı: {path.name}"
                else:
                    param["status"] = "Son görüntü atlandı — Q ile çık."
                continue

            if key in (ord("b"), ord("B")):
                if current_idx > 0:
                    if state.dirty or load_annotation(path) is None:
                        for b in state.boxes:
                            if b.source != "user":
                                b.claim_user()
                        save_annotation(path, state.boxes, state.image_w, state.image_h)
                    current_idx -= 1
                    load_image(current_idx)
                else:
                    param["status"] = "İlk görüntü."
                continue

            if key in (ord("r"), ord("R")):
                load_image(current_idx, force_redetect=True)
                continue

            if key in (ord("d"), ord("D"), 8, 127):
                if state.selected_idx is not None and 0 <= state.selected_idx < len(state.boxes):
                    removed = state.boxes.pop(state.selected_idx)
                    state.selected_idx = None
                    state.dirty = True
                    param["status"] = f"Silindi: {removed.class_name}"
                continue

            if key in KEY_TO_CLASS:
                cls_name = KEY_TO_CLASS[key]
                cls_id = LABEL2ID[cls_name]
                if state.pending_new_box is not None:
                    x1, y1, x2, y2 = state.pending_new_box
                    state.boxes.append(
                        AnnoBox(
                            class_id=cls_id,
                            class_name=cls_name,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            source="user",
                        )
                    )
                    state.selected_idx = len(state.boxes) - 1
                    state.pending_new_box = None
                    state.dirty = True
                    param["status"] = f"Eklendi: {cls_name}"
                elif state.selected_idx is not None and 0 <= state.selected_idx < len(state.boxes):
                    b = state.boxes[state.selected_idx]
                    if b.class_name != cls_name:
                        b.class_name = cls_name
                        b.class_id = cls_id
                    b.claim_user()
                    state.dirty = True
                    param["status"] = f"Sınıf ayarlandı → {cls_name} (onaylı)"
                else:
                    param["status"] = "Önce bir kutu seç ya da yeni kutu çiz."
                continue
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
