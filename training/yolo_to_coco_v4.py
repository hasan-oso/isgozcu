"""Convert filtered YOLO labels (6-class v4) → COCO JSON for HF RT-DETR.

Reads:
    data/construction_ppe_filtered/
        ├── images/{train,val,test}/
        └── labels/{train,val,test}/

Writes:
    data/construction_ppe_filtered/coco/
        ├── train.json
        ├── val.json
        └── test.json

COCO category IDs match the v4 schema (0..5):
    0 Person, 1 helmet, 2 vest, 3 gloves, 4 no_helmet, 5 no_gloves

Bbox conversion:
    YOLO line:  cls cx_norm cy_norm w_norm h_norm    (normalized 0..1)
    COCO bbox:  [x_abs, y_abs, w_abs, h_abs]         (absolute pixels)

Validation:
    * Every annotation must lie inside the image.
    * Per-image image dimensions are read from the actual image (not assumed).
    * Out-of-bounds annotations clamp + warn rather than crash.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# UTF-8 console for Turkish output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DST_ROOT = PROJECT_ROOT / "data" / "construction_ppe_filtered"
COCO_DIR = DST_ROOT / "coco"

SPLITS = ("train", "val", "test")

ID2NAME: dict[int, str] = {
    0: "Person",
    1: "helmet",
    2: "vest",
    3: "gloves",
    4: "no_helmet",
    5: "no_gloves",
}

CATEGORIES = [{"id": cid, "name": ID2NAME[cid], "supercategory": ""} for cid in sorted(ID2NAME)]


def yolo_line_to_coco_bbox(line: str, w: int, h: int) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cid = int(parts[0])
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
    except ValueError:
        return None
    x = cx - bw / 2.0
    y = cy - bh / 2.0
    # Clamp to image
    x = max(0.0, min(float(w), x))
    y = max(0.0, min(float(h), y))
    bw = min(bw, float(w) - x)
    bh = min(bh, float(h) - y)
    if bw <= 1 or bh <= 1:
        return None
    return cid, [x, y, bw, bh]


def convert_split(split: str) -> dict | None:
    img_dir = DST_ROOT / "images" / split
    lbl_dir = DST_ROOT / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[bilgi] {split}: dizin yok, atlanıyor.")
        return None

    images_meta: list[dict] = []
    annotations: list[dict] = []
    next_anno_id = 1
    skipped = 0
    out_of_bounds = 0

    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"\n[{split}] {len(image_files)} görüntü dönüştürülüyor...")
    for image_id, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [uyarı] okunamadı: {img_path.name}")
            skipped += 1
            continue
        h, w = img.shape[:2]
        images_meta.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": int(w),
                "height": int(h),
            }
        )
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            res = yolo_line_to_coco_bbox(raw, w, h)
            if res is None:
                out_of_bounds += 1
                continue
            cid, bbox = res
            if cid not in ID2NAME:
                # Shouldn't happen — prepare script already filtered.
                out_of_bounds += 1
                continue
            x, y, bw, bh = bbox
            annotations.append(
                {
                    "id": next_anno_id,
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            next_anno_id += 1
        if image_id % 100 == 0:
            print(f"  {image_id}/{len(image_files)} done")

    print(f"  → {len(images_meta)} görüntü, {len(annotations)} kutu, {out_of_bounds} atlandı, {skipped} okunamadı")

    return {
        "info": {
            "description": f"İşGözcü v4 — Construction-PPE filtered (6 classes), split={split}",
            "version": "1.0",
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": images_meta,
        "annotations": annotations,
    }


def main() -> int:
    if not DST_ROOT.exists():
        print(f"[hata] Filtreli dizin yok: {DST_ROOT}")
        print("       Önce: python tests/prepare_construction_ppe_for_v4.py")
        return 2

    COCO_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for split in SPLITS:
        coco = convert_split(split)
        if coco is None:
            continue
        out = COCO_DIR / f"{split}.json"
        out.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        per_class: dict[str, int] = defaultdict(int)
        for a in coco["annotations"]:
            per_class[ID2NAME[a["category_id"]]] += 1
        summary[split] = {
            "json": str(out.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "per_class": dict(per_class),
        }
        print(f"  yazıldı: {out}")

    print()
    print("=" * 64)
    print("COCO dönüştürme tamamlandı")
    print("=" * 64)
    for split, info in summary.items():
        print(f"  [{split}] {info['images']} görüntü, {info['annotations']} kutu  →  {info['json']}")
        for cls, n in sorted(info["per_class"].items()):
            print(f"      {cls:<12} {n}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
