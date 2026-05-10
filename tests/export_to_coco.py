"""Combine per-image annotations into COCO-format train/valid splits.

Reads:
    data/training_images/annotations/*.json     (one per image, written by
                                                  annotate_training_images.py)

Writes:
    data/training_images/coco/
        ├── images/                        (copies of the source JPEGs, flat layout)
        ├── train.json                     (80% — COCO format)
        └── valid.json                     (20% — COCO format)

The split is reproducible: a fixed seed is used, and the split is stratified
by mode (full_ppe / no_hardhat / no_vest / no_ppe) so each split sees every
PPE configuration in roughly the same proportion.

Category IDs match the v2 model's id2label exactly (0..9), so the exported
JSON drops straight into a HuggingFace RT-DETR fine-tuning loop without any
remapping.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# UTF-8 console for Turkish output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "training_images"
ANNO_DIR = TRAIN_DIR / "annotations"
COCO_DIR = TRAIN_DIR / "coco"
IMAGES_OUT = COCO_DIR / "images"

# Mirror the model's id2label.
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

CATEGORIES = [
    {"id": cid, "name": ID2LABEL[cid], "supercategory": ""}
    for cid in sorted(ID2LABEL)
]

VAL_FRACTION = 0.20
SEED = 42


def load_all_annotations() -> list[dict]:
    if not ANNO_DIR.exists():
        return []
    out = []
    for p in sorted(ANNO_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_anno_path"] = p
            out.append(data)
        except Exception as e:
            print(f"[uyarı] Bozuk dosya atlandı: {p.name} ({e})")
    return out


def stratified_split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_mode[r.get("mode", "_unknown")].append(r)
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for mode, group in by_mode.items():
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 5 else max(1, len(group) // 5)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def to_coco(records: list[dict]) -> dict:
    images = []
    annotations = []
    next_anno_id = 1
    for image_id, r in enumerate(records, start=1):
        src_image = PROJECT_ROOT / r["image_path"]
        if not src_image.exists():
            print(f"[uyarı] Görüntü bulunamadı, atlanıyor: {src_image}")
            continue
        # Flat filename inside coco/images/ (prefix with mode to avoid collisions).
        out_name = f"{r.get('mode', 'misc')}_{src_image.stem}.jpg"
        IMAGES_OUT.mkdir(parents=True, exist_ok=True)
        out_path = IMAGES_OUT / out_name
        if not out_path.exists():
            shutil.copy2(src_image, out_path)
        images.append(
            {
                "id": image_id,
                "file_name": out_name,
                "width": int(r["image_width"]),
                "height": int(r["image_height"]),
            }
        )
        for box in r.get("boxes", []):
            x, y, w, h = box["bbox"]
            annotations.append(
                {
                    "id": next_anno_id,
                    "image_id": image_id,
                    "category_id": int(box["class_id"]),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w) * float(h),
                    "iscrowd": 0,
                }
            )
            next_anno_id += 1

    return {
        "info": {
            "description": "İşGözcü self-captured PPE (Phase 2A.3.B export)",
            "version": "1.0",
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def main() -> int:
    records = load_all_annotations()
    if not records:
        print(f"[hata] Hiç etiket yok: {ANNO_DIR}")
        return 1
    print(f"Toplam etiketli görüntü: {len(records)}")

    train_recs, val_recs = stratified_split(records, VAL_FRACTION, SEED)
    print(f"Train: {len(train_recs)}  |  Val: {len(val_recs)}  (seed={SEED})")

    COCO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)

    train_coco = to_coco(train_recs)
    val_coco = to_coco(val_recs)

    train_path = COCO_DIR / "train.json"
    val_path = COCO_DIR / "valid.json"
    train_path.write_text(json.dumps(train_coco, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_coco, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("EXPORT TAMAMLANDI")
    print("=" * 60)
    print(f"  Train JSON:   {train_path}")
    print(f"  Valid JSON:   {val_path}")
    print(f"  Görüntüler:   {IMAGES_OUT}")
    print(f"  Train ann:    {len(train_coco['annotations'])} kutu, {len(train_coco['images'])} görüntü")
    print(f"  Valid ann:    {len(val_coco['annotations'])} kutu, {len(val_coco['images'])} görüntü")
    print(f"  Sınıflar (10):", ", ".join(c["name"] for c in CATEGORIES))
    print("=" * 60)
    print()
    print("Sonraki adım: bu klasörü ZIP'leyip Colab'a yükle (Phase 2A.3.C).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
