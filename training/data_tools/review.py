"""training/data_tools/review.py - manual correction of YOLO labels (OpenCV).

Opens each image with its model-seeded boxes; lets you delete wrong boxes,
draw missing ones, and reassign classes, then rewrites the YOLO .txt.

Scope: 3 classes -> number keys 1=Person, 2=Hardhat, 3=Safety Vest.
The .txt always stores the NUMERIC index (0/1/2); the Turkish names shown on
screen are display-only (Hershey font cannot draw s/g/I diacritics, so the
on-screen text uses ASCII: Isci / Baret / Guvenlik yelegi).

Run:
    python training/data_tools/review.py
    python training/data_tools/review.py --images training/captured --labels training/labels

Keys:
    D / Right     next image            A / Left   previous image
    left-drag     draw a new box        1 / 2 / 3  assign class to new/selected box
    left-click    select a box          X / DEL    delete selected box
    S             save current image     Q          quit (warns if unsaved)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES = PROJECT_ROOT / "training" / "captured"
DEFAULT_LABELS = PROJECT_ROOT / "training" / "labels"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# our index -> (ASCII on-screen label, BGR colour).
# Proper Turkish: 0=Isci->Isci(İşçi), 1=Baret, 2=Guvenlik yelegi(Güvenlik yeleği).
CLASSES: dict[int, tuple[str, tuple[int, int, int]]] = {
    0: ("Isci", (220, 130, 50)),          # blue   = Person
    1: ("Baret", (60, 200, 60)),          # green  = Hardhat
    2: ("Guvenlik yelegi", (40, 150, 250)),  # orange = Safety Vest
}
NUMKEY_TO_CLS = {ord("1"): 0, ord("2"): 1, ord("3"): 2}

BANNER_H = 56
MAX_W, MAX_H = 1280, 720
MIN_DRAG = 6  # minimum drag in original px to count as "draw a box"

# extended key codes (waitKeyEx) vary by OS; ASCII a/d/x always accepted too.
K_RIGHT = {2555904, 65363}
K_LEFT = {2424832, 65361}
K_DELETE = {3014656, 65535, 127, 255}


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class Reviewer:
    def __init__(self, images_root: Path, labels_root: Path) -> None:
        self.images_root = images_root
        self.labels_root = labels_root
        self.images = find_images(images_root)
        self.idx = 0
        self.win = "isgozcu review"
        self.boxes_by_idx: dict[int, list[dict]] = {}
        self.dirty: set[int] = set()
        # per-frame view state
        self.scale = 1.0
        self.W0 = 0
        self.H0 = 0
        self.cur_img = None
        # interaction state
        self.selected = -1
        self.drawing = False
        self.drag_start = (0.0, 0.0)
        self.drag_now = (0.0, 0.0)
        self.pending = None  # (x1,y1,x2,y2) original px, awaiting a class key

    # ---------- label IO ----------
    def label_path(self, img_path: Path) -> Path:
        return (self.labels_root / img_path.relative_to(self.images_root)).with_suffix(".txt")

    def load_boxes(self, i: int) -> list[dict]:
        if i in self.boxes_by_idx:
            return self.boxes_by_idx[i]
        img_path = self.images[i]
        lp = self.label_path(img_path)
        boxes: list[dict] = []
        if lp.exists():
            probe = cv2.imread(str(img_path))
            h, w = (probe.shape[:2] if probe is not None else (0, 0))
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    c = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                except ValueError:
                    continue
                boxes.append({
                    "cls": c,
                    "x1": (cx - bw / 2) * w, "y1": (cy - bh / 2) * h,
                    "x2": (cx + bw / 2) * w, "y2": (cy + bh / 2) * h,
                })
        self.boxes_by_idx[i] = boxes
        return boxes

    def save_current(self) -> None:
        img_path = self.images[self.idx]
        boxes = self.boxes_by_idx.get(self.idx, [])
        lp = self.label_path(img_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        W, H = self.W0, self.H0
        lines: list[str] = []
        for b in boxes:
            x1, x2 = sorted((b["x1"], b["x2"]))
            y1, y2 = sorted((b["y1"], b["y2"]))
            cx = min(max(((x1 + x2) / 2) / W, 0.0), 1.0)
            cy = min(max(((y1 + y2) / 2) / H, 0.0), 1.0)
            bw = min(max((x2 - x1) / W, 0.0), 1.0)
            bh = min(max((y2 - y1) / H, 0.0), 1.0)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f'{b["cls"]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.dirty.discard(self.idx)

    # ---------- coordinate mapping (display px -> original px) ----------
    def to_orig(self, mx: int, my: int) -> tuple[float, float]:
        return mx / self.scale, (my - BANNER_H) / self.scale

    # ---------- mouse ----------
    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.to_orig(x, y)
            if iy < 0:
                return
            self.drawing = True
            self.drag_start = (ix, iy)
            self.drag_now = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.drag_now = self.to_orig(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            ix, iy = self.to_orig(x, y)
            x0, y0 = self.drag_start
            if abs(ix - x0) >= MIN_DRAG and abs(iy - y0) >= MIN_DRAG:
                self.pending = (min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy))
                self.selected = -1
            else:
                self.selected = self.pick_box(ix, iy)

    def pick_box(self, ix: float, iy: float) -> int:
        best, best_area = -1, None
        for k, b in enumerate(self.boxes_by_idx.get(self.idx, [])):
            if min(b["x1"], b["x2"]) <= ix <= max(b["x1"], b["x2"]) and \
               min(b["y1"], b["y2"]) <= iy <= max(b["y1"], b["y2"]):
                area = abs(b["x2"] - b["x1"]) * abs(b["y2"] - b["y1"])
                if best_area is None or area < best_area:
                    best, best_area = k, area
        return best

    # ---------- rendering ----------
    def render(self):
        disp_w = max(1, int(self.W0 * self.scale))
        disp_h = max(1, int(self.H0 * self.scale))
        scaled = cv2.resize(self.cur_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((disp_h + BANNER_H, max(disp_w, 720), 3), dtype=np.uint8)
        canvas[BANNER_H:BANNER_H + disp_h, 0:disp_w] = scaled

        boxes = self.boxes_by_idx.get(self.idx, [])
        for k, b in enumerate(boxes):
            label, color = CLASSES.get(b["cls"], ("?", (200, 200, 200)))
            p1 = (int(min(b["x1"], b["x2"]) * self.scale), int(min(b["y1"], b["y2"]) * self.scale) + BANNER_H)
            p2 = (int(max(b["x1"], b["x2"]) * self.scale), int(max(b["y1"], b["y2"]) * self.scale) + BANNER_H)
            cv2.rectangle(canvas, p1, p2, color, 3 if k == self.selected else 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (p1[0] + 2, max(p1[1] - 4, BANNER_H + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if self.pending is not None:
            x1, y1, x2, y2 = self.pending
            cv2.rectangle(canvas,
                          (int(x1 * self.scale), int(y1 * self.scale) + BANNER_H),
                          (int(x2 * self.scale), int(y2 * self.scale) + BANNER_H),
                          (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "yeni kutu: 1=Isci 2=Baret 3=Yelek",
                        (int(x1 * self.scale) + 2, max(int(y1 * self.scale) + BANNER_H - 6, BANNER_H + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if self.drawing:
            x0, y0 = self.drag_start
            xn, yn = self.drag_now
            cv2.rectangle(canvas,
                          (int(x0 * self.scale), int(y0 * self.scale) + BANNER_H),
                          (int(xn * self.scale), int(yn * self.scale) + BANNER_H),
                          (180, 180, 180), 1, cv2.LINE_AA)

        name = self.images[self.idx].name
        status = "*KAYDEDILMEDI*" if self.idx in self.dirty else "kayitli"
        cv2.putText(canvas, f"[{self.idx + 1}/{len(self.images)}] {name}  kutu={len(boxes)}  {status}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "D/->ileri  A/<-geri  surukle=ciz  1/2/3=sinif  X=sil  S=kaydet  Q=cikis",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 255), 1, cv2.LINE_AA)
        return canvas

    # ---------- frame prep ----------
    def prepare(self) -> bool:
        img = cv2.imread(str(self.images[self.idx]))
        if img is None:
            print(f"[uyari] okunamadi: {self.images[self.idx]}")
            return False
        self.cur_img = img
        self.H0, self.W0 = img.shape[:2]
        self.scale = min(1.0, MAX_W / self.W0, MAX_H / self.H0)
        self.selected = -1
        self.pending = None
        self.drawing = False
        self.load_boxes(self.idx)
        return True

    def goto(self, new_idx: int) -> None:
        self.idx = new_idx % len(self.images)
        self.cur_img = None

    # ---------- main loop ----------
    def run(self) -> int:
        if not self.images:
            print(f"[HATA] Goruntu yok: {self.images_root}")
            return 1
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        print(f"[bilgi] {len(self.images)} goruntu. D/A gezin, S kaydet, Q cikis.")
        while True:
            if self.cur_img is None and not self.prepare():
                self.goto(self.idx + 1)
                continue
            cv2.imshow(self.win, self.render())
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            k8 = key & 0xFF

            if k8 in (ord("q"), ord("Q"), 27):
                if self.dirty:
                    print(f"[uyari] {len(self.dirty)} goruntude kaydedilmemis degisiklik var. "
                          "Cikmak icin tekrar Q; bu goruntuyu kaydetmek icin S.")
                    key2 = cv2.waitKeyEx(0) & 0xFF
                    if key2 in (ord("q"), ord("Q"), 27):
                        break
                    if key2 in (ord("s"), ord("S")):
                        self.save_current()
                    continue
                break
            elif key in K_RIGHT or k8 in (ord("d"), ord("D")):
                self.goto(self.idx + 1)
            elif key in K_LEFT or k8 in (ord("a"), ord("A")):
                self.goto(self.idx - 1)
            elif key in K_DELETE or k8 in (ord("x"), ord("X")):
                boxes = self.boxes_by_idx.get(self.idx, [])
                if 0 <= self.selected < len(boxes):
                    boxes.pop(self.selected)
                    self.selected = -1
                    self.dirty.add(self.idx)
            elif k8 in NUMKEY_TO_CLS:
                cls = NUMKEY_TO_CLS[k8]
                if self.pending is not None:
                    x1, y1, x2, y2 = self.pending
                    self.boxes_by_idx.setdefault(self.idx, []).append(
                        {"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    self.selected = len(self.boxes_by_idx[self.idx]) - 1
                    self.pending = None
                    self.dirty.add(self.idx)
                elif 0 <= self.selected < len(self.boxes_by_idx.get(self.idx, [])):
                    self.boxes_by_idx[self.idx][self.selected]["cls"] = cls
                    self.dirty.add(self.idx)
            elif k8 in (ord("s"), ord("S")):
                self.save_current()
                print(f"[kaydedildi] {self.label_path(self.images[self.idx])}")

        cv2.destroyAllWindows()
        print("[bilgi] Cikildi.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Manual YOLO label review / correction.")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    args = ap.parse_args()

    images_root = Path(args.images)
    if not images_root.is_absolute():
        images_root = PROJECT_ROOT / images_root
    labels_root = Path(args.labels)
    if not labels_root.is_absolute():
        labels_root = PROJECT_ROOT / labels_root
    if not images_root.exists():
        print(f"[HATA] Goruntu klasoru yok: {images_root}")
        return 2
    return Reviewer(images_root, labels_root).run()


if __name__ == "__main__":
    raise SystemExit(main())
