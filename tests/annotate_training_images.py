"""Interactive annotation tool for the self-captured training images
(Phase 2A.3.B).

Workflow per image:
  1. Run RT-DETR v2 to seed initial boxes (low conf threshold so we err
     toward more candidates the user can prune, not fewer).
  2. Show the image with boxes + a per-mode "expected" hint
     (`full_ppe` should have Person + Hardhat + Safety Vest, etc.).
  3. User edits:
       - left-click on a box → select it
       - left-click + drag in empty area → draw a new box
       - on mouseup of a new box → press 1-9 / 0 to assign its class
       - DELETE/D → remove selected box
       - 1-9 / 0 → change selected box's class
       - right-click on a box → quick-delete
       - SPACE → save current image, advance to next un-annotated image
       - B → go back one
       - R → re-run model on current image, replace current edits
       - Q → save progress + quit

Output (per image):
    data/training_images/annotations/{stem}.json

Progress index:
    data/training_images/progress.json  — list of completed file stems
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

# Class id ↔ name (mirrors the model's id2label exactly).
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

# Keypress digit → class name (ergonomic ordering for the UI).
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

MODE_EXPECTED: dict[str, list[str]] = {
    "full_ppe":   ["Person", "Hardhat", "Safety Vest"],
    "no_hardhat": ["Person", "NO-Hardhat", "Safety Vest"],
    "no_vest":    ["Person", "Hardhat", "NO-Safety Vest"],
    "no_ppe":     ["Person", "NO-Hardhat", "NO-Safety Vest"],
}

TRAIN_DIR = PROJECT_ROOT / "data" / "training_images"
ANNO_DIR = TRAIN_DIR / "annotations"
PROGRESS_PATH = TRAIN_DIR / "progress.json"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v2"
SEED_CONF_THRESHOLD = 0.15  # low so the user has plenty of candidates to prune

WINDOW_W = 1600
WINDOW_H = 900
IMG_W = 1280
IMG_H = 720
PANEL_X = IMG_W
PANEL_W = WINDOW_W - IMG_W  # 320
STATUS_Y = IMG_H
STATUS_H = WINDOW_H - IMG_H  # 180

WINDOW_TITLE = "İşGözcü - Eğitim Görüntüsü Etiketleme"

MIN_BOX_PX = 12  # reject newly-drawn boxes smaller than this


# ---------- data ----------


@dataclass
class AnnoBox:
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    auto: bool = False  # True if seeded by RT-DETR, False once user touches it

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


def collect_images() -> list[Path]:
    """Sorted list of training images (full_ppe / no_hardhat / no_vest / no_ppe)."""
    out: list[Path] = []
    for mode in MODE_EXPECTED.keys():
        folder = TRAIN_DIR / mode
        if folder.exists():
            out.extend(sorted(folder.glob("*.jpg")))
    return out


def image_mode(image_path: Path) -> str:
    return image_path.parent.name


def load_progress() -> set[str]:
    if not PROGRESS_PATH.exists():
        return set()
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        return set(data.get("annotated_files", []))
    except Exception:
        return set()


def save_progress(annotated: set[str]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(
            {
                "annotated_files": sorted(annotated),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def annotation_path_for(image_path: Path) -> Path:
    rel = image_path.relative_to(TRAIN_DIR).with_suffix(".json")
    return ANNO_DIR / rel.name  # flat folder, suffix already includes mode


def save_annotation(image_path: Path, boxes: list[AnnoBox], image_w: int, image_h: int, auto_only: bool) -> Path:
    ANNO_DIR.mkdir(parents=True, exist_ok=True)
    out = annotation_path_for(image_path)
    payload = {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "image_width": image_w,
        "image_height": image_h,
        "mode": image_mode(image_path),
        "boxes": [
            {
                "class_id": b.class_id,
                "class_name": b.class_name,
                "bbox": [b.x, b.y, b.w, b.h],
            }
            for b in boxes
        ],
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
        "auto_detected": auto_only,
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
                    auto=False,
                )
            )
        return out
    except Exception:
        return None


# ---------- detector wrapper ----------


def detect_initial_boxes(detector: RTDetrDetector, frame_bgr: np.ndarray) -> list[AnnoBox]:
    d = detector.detect(frame_bgr)
    out: list[AnnoBox] = []
    if len(d) == 0:
        return out
    for i in range(len(d)):
        x1, y1, x2, y2 = d.xyxy[i].astype(int).tolist()
        cid = int(d.class_id[i])
        name = ID2LABEL.get(cid, f"unknown_{cid}")
        out.append(AnnoBox(class_id=cid, class_name=name, x1=x1, y1=y1, x2=x2, y2=y2, auto=True))
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
    # top + bottom
    for x in range(min(x1, x2), max(x1, x2), dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, max(x1, x2)), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + dash, max(x1, x2)), y2), color, thickness, cv2.LINE_AA)
    # left + right
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
    # If image isn't exactly IMG_W x IMG_H, the cv2.imread we did already resized
    # in the caller. Mouse coords come in image coords directly.
    for i, b in enumerate(boxes):
        color = CLASS_COLORS.get(b.class_name, DEFAULT_COLOR)
        is_sel = i == selected_idx
        thickness = 4 if is_sel else 2
        cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), color, thickness, cv2.LINE_AA)
        if is_sel:
            draw_dashed_rect(canvas, (b.x1 - 4, b.y1 - 4), (b.x2 + 4, b.y2 + 4), (0, 255, 255), 2)
        # Class label background + text
        label = f"{b.class_name}"
        if b.auto:
            label += " (auto)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(b.y1 - 4, th + 4)
        cv2.rectangle(canvas, (b.x1, ly - th - 6), (b.x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(canvas, label, (b.x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Ghost rectangle while drawing.
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
) -> np.ndarray:
    panel = np.zeros((IMG_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (15, 22, 36)  # dark blue

    y = 30
    put_text_outlined(panel, "Etiketleme", (12, y), scale=0.85, thickness=2)
    y += 28
    put_text_outlined(panel, f"{image_idx + 1} / {total}", (12, y), scale=0.65, fg=(140, 200, 255))
    y += 22
    put_text_outlined(panel, image_path.name[:38], (12, y), scale=0.45, fg=(140, 165, 200))
    y += 18
    put_text_outlined(panel, f"Mod: {image_mode(image_path)}", (12, y), scale=0.5, fg=(140, 165, 200))

    # Progress bar
    y += 22
    bar_w = PANEL_W - 24
    bar_x = 12
    bar_h = 8
    cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (40, 60, 90), -1)
    fill = int(bar_w * (image_idx / max(1, total)))
    cv2.rectangle(panel, (bar_x, y), (bar_x + fill, y + bar_h), (60, 200, 255), -1)
    y += bar_h + 18

    # Expected vs detected
    expected = MODE_EXPECTED.get(image_mode(image_path), [])
    detected_classes = [b.class_name for b in boxes]
    put_text_outlined(panel, "Beklenen sınıflar:", (12, y), scale=0.55, thickness=1)
    y += 22
    for cls in expected:
        ok = cls in detected_classes
        mark = "✓" if ok else "⚠"
        color = (60, 220, 60) if ok else (60, 60, 230)
        put_text_outlined(panel, f"  {mark} {cls}", (12, y), scale=0.52, fg=color)
        y += 18
    y += 6

    # Detected count + extras (boxes whose class is not in the expected set)
    extras = [c for c in detected_classes if c not in expected]
    put_text_outlined(panel, f"Tespit: {len(boxes)} kutu", (12, y), scale=0.55)
    y += 18
    if extras:
        put_text_outlined(panel, "Beklenmeyen:", (12, y), scale=0.5, fg=(255, 180, 60))
        y += 18
        for cls in sorted(set(extras)):
            n = extras.count(cls)
            put_text_outlined(panel, f"  • {cls} ×{n}", (12, y), scale=0.45, fg=(255, 180, 60))
            y += 16
    y += 6

    # Selected box info
    if selected_idx is not None and 0 <= selected_idx < len(boxes):
        b = boxes[selected_idx]
        put_text_outlined(panel, "Seçili kutu:", (12, y), scale=0.55, fg=(0, 255, 255))
        y += 18
        put_text_outlined(panel, f"  {b.class_name}", (12, y), scale=0.5, fg=(0, 255, 255))
        y += 16
        put_text_outlined(panel, f"  ({b.x},{b.y}) {b.w}x{b.h}", (12, y), scale=0.42, fg=(180, 220, 240))
        y += 18

    # Class hints
    y = IMG_H - 280
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

    if pending_class:
        # Big yellow banner asking for class assignment
        cv2.rectangle(panel, (8, 6), (PANEL_W - 8, 70), (0, 220, 255), 2)
        put_text_outlined(panel, "Yeni kutu — sınıf seç!", (16, 36), scale=0.65, thickness=2, fg=(0, 220, 255))
        put_text_outlined(panel, "1-9 / 0 ile ata", (16, 58), scale=0.5, fg=(0, 220, 255))

    return panel


def render_status_strip(width: int, height: int, status: str, image_path: Path, total_anno: int, total_imgs: int) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (10, 16, 28)
    put_text_outlined(strip, status, (16, 32), scale=0.65, thickness=2, fg=(220, 240, 255))
    keys = (
        "[SPACE] Kaydet+İleri   [B] Geri   [R] Yeniden tespit   "
        "[D / DEL] Seçili kutuyu sil   [Sağ tık] Kutuyu sil   [Q] Kaydet+Çık"
    )
    put_text_outlined(strip, keys, (16, 64), scale=0.5, fg=(140, 180, 220))
    put_text_outlined(strip, f"Tamamlandı: {total_anno} / {total_imgs}", (16, 96), scale=0.5, fg=(140, 180, 220))
    return strip


# ---------- mouse interaction ----------


@dataclass
class UIState:
    boxes: list[AnnoBox] = field(default_factory=list)
    selected_idx: Optional[int] = None
    drawing: bool = False
    draw_p1: Optional[tuple[int, int]] = None
    draw_p2: Optional[tuple[int, int]] = None
    pending_new_box: Optional[tuple[int, int, int, int]] = None  # x1,y1,x2,y2 awaiting class
    image: Optional[np.ndarray] = None  # current frame in image-panel coords
    dirty: bool = False  # has the user touched anything since auto-detect?


def hit_box(boxes: list[AnnoBox], x: int, y: int) -> Optional[int]:
    """Return index of innermost (smallest-area) box containing (x, y)."""
    candidates = [(i, b) for i, b in enumerate(boxes) if b.contains(x, y)]
    if not candidates:
        return None
    return min(candidates, key=lambda kv: kv[1].area())[0]


def _on_mouse(event: int, x: int, y: int, flags: int, param: dict) -> None:
    state: UIState = param["state"]
    # Only react to events inside the image panel.
    if x >= IMG_W or y >= IMG_H or x < 0 or y < 0:
        # Reset any in-flight drag if the cursor leaves
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
                param["status"] = "Yeni kutu çizildi — sınıf seçmek için 1-9 / 0 tuşuna bas."
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
            param["status"] = f"Silindi: {removed.class_name}"


# ---------- main loop ----------


def main() -> int:
    images = collect_images()
    if not images:
        print(f"[hata] Hiç eğitim görüntüsü yok: {TRAIN_DIR}")
        return 1
    total = len(images)
    print(f"Bulunan görüntü: {total}")

    annotated = load_progress()
    print(f"Önceden tamamlanan: {len(annotated)} / {total}")

    # Find first un-annotated image to resume.
    start_idx = 0
    for i, p in enumerate(images):
        rel = str(p.relative_to(TRAIN_DIR)).replace("\\", "/")
        if rel not in annotated:
            start_idx = i
            break
    else:
        start_idx = total - 1

    print(f"Model yükleniyor: {MODEL_PATH.name} ...")
    detector = RTDetrDetector(model_path=str(MODEL_PATH), conf_threshold=SEED_CONF_THRESHOLD)
    print(f"Sınıflar: {detector.class_names}\n")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, WINDOW_W, WINDOW_H)

    state = UIState()
    param = {"state": state, "status": "Hazır."}
    cv2.setMouseCallback(WINDOW_TITLE, _on_mouse, param)

    current_idx = start_idx

    def load_image(idx: int, force_redetect: bool = False) -> bool:
        nonlocal_state = state  # captured
        path = images[idx]
        img = cv2.imread(str(path))
        if img is None:
            print(f"[hata] Okunamadı: {path}")
            return False
        # Ensure 1280x720 panel size — resize if needed (lossless keeps aspect).
        if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        nonlocal_state.image = img
        nonlocal_state.selected_idx = None
        nonlocal_state.drawing = False
        nonlocal_state.draw_p1 = None
        nonlocal_state.draw_p2 = None
        nonlocal_state.pending_new_box = None

        # Try to load existing annotation first; otherwise run model.
        existing = None if force_redetect else load_annotation(path)
        if existing is not None:
            nonlocal_state.boxes = existing
            nonlocal_state.dirty = False
            param["status"] = f"Mevcut etiket yüklendi ({len(existing)} kutu)."
        else:
            nonlocal_state.boxes = detect_initial_boxes(detector, img)
            nonlocal_state.dirty = False
            param["status"] = f"RT-DETR v2: {len(nonlocal_state.boxes)} aday kutu."
        return True

    if not load_image(current_idx):
        return 1

    try:
        while True:
            path = images[current_idx]
            # Compose canvas
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
            )
            status_strip = render_status_strip(
                WINDOW_W, STATUS_H, param["status"], path, len(annotated), total
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
                # Save current image's edits if there are any.
                if state.dirty or load_annotation(path) is None:
                    auto_only = not state.dirty and all(b.auto for b in state.boxes)
                    save_annotation(path, state.boxes, IMG_W, IMG_H, auto_only)
                save_progress(annotated)
                print("Kullanıcı çıkışı, ilerleme kaydedildi.")
                break

            if key == 32:  # SPACE = save & next
                if state.pending_new_box is not None:
                    param["status"] = "Önce yeni kutuya sınıf ata (1-9 / 0)."
                    continue
                auto_only = not state.dirty and all(b.auto for b in state.boxes)
                save_annotation(path, state.boxes, IMG_W, IMG_H, auto_only)
                rel = str(path.relative_to(TRAIN_DIR)).replace("\\", "/")
                annotated.add(rel)
                save_progress(annotated)
                if current_idx + 1 < total:
                    current_idx += 1
                    load_image(current_idx)
                else:
                    param["status"] = "Tüm görüntüler tamamlandı 🎉 — Q ile çık."
                continue

            if key in (ord("b"), ord("B")):
                if current_idx > 0:
                    # Save in-flight edits before stepping back
                    auto_only = not state.dirty and all(b.auto for b in state.boxes)
                    if state.dirty or load_annotation(path) is None:
                        save_annotation(path, state.boxes, IMG_W, IMG_H, auto_only)
                    current_idx -= 1
                    load_image(current_idx)
                else:
                    param["status"] = "İlk görüntü."
                continue

            if key in (ord("r"), ord("R")):
                load_image(current_idx, force_redetect=True)
                state.dirty = False
                continue

            if key in (ord("d"), ord("D"), 8, 127):  # D, BACKSPACE, DELETE
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
                            auto=False,
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
                        b.auto = False
                        state.dirty = True
                        param["status"] = f"Sınıf değişti → {cls_name}"
                else:
                    param["status"] = "Önce bir kutu seç ya da yeni kutu çiz."
                continue
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
