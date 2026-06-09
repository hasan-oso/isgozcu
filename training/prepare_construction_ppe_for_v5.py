"""Remap the Construction-PPE filtered dataset (6 classes) to v3's 10-class
schema, producing data/construction_ppe_v5/ as the v5 fine-tune corpus.

Class remap (Construction-PPE 6 → v3 10):
    0 (Person)    → 5  (Person)
    1 (helmet)    → 0  (Hardhat)
    2 (vest)      → 7  (Safety Vest)
    3 (gloves)    → DROPPED  (not in v3 schema)
    4 (no_helmet) → 2  (NO-Hardhat)
    5 (no_gloves) → DROPPED

We don't generate examples for v3's other classes (Mask, NO-Mask, NO-Safety
Vest, Safety Cone, machinery, vehicle) — those keep whatever weights v3
already learned. This is *continued training* on a small clean subset, not
a from-scratch retrain.
"""

from __future__ import annotations

import json
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
SRC_ROOT = PROJECT_ROOT / "data" / "construction_ppe_filtered"
DST_ROOT = PROJECT_ROOT / "data" / "construction_ppe_v5"

# Construction-PPE 6 → v3 10
CLASS_REMAP: dict[int, int] = {
    0: 5,  # Person
    1: 0,  # helmet → Hardhat
    2: 7,  # vest → Safety Vest
    4: 2,  # no_helmet → NO-Hardhat
    # 3 (gloves) and 5 (no_gloves) are intentionally dropped
}

V3_CATEGORIES = [
    {"id": 0, "name": "Hardhat", "supercategory": ""},
    {"id": 1, "name": "Mask", "supercategory": ""},
    {"id": 2, "name": "NO-Hardhat", "supercategory": ""},
    {"id": 3, "name": "NO-Mask", "supercategory": ""},
    {"id": 4, "name": "NO-Safety Vest", "supercategory": ""},
    {"id": 5, "name": "Person", "supercategory": ""},
    {"id": 6, "name": "Safety Cone", "supercategory": ""},
    {"id": 7, "name": "Safety Vest", "supercategory": ""},
    {"id": 8, "name": "machinery", "supercategory": ""},
    {"id": 9, "name": "vehicle", "supercategory": ""},
]
V3_NAMES = {c["id"]: c["name"] for c in V3_CATEGORIES}

SPLITS = ("train", "val")


def remap_split(split: str) -> dict:
    src_json = SRC_ROOT / "coco" / f"{split}.json"
    if not src_json.exists():
        raise FileNotFoundError(f"Source COCO missing: {src_json}")
    src_imgs_dir = SRC_ROOT / "images" / split
    dst_imgs_dir = DST_ROOT / "images" / split
    dst_imgs_dir.mkdir(parents=True, exist_ok=True)

    with open(src_json) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    next_anno_id = 1
    images_kept = 0
    images_dropped_no_anns = 0
    per_class_counts: dict[int, int] = defaultdict(int)

    for img_id, img_info in images_by_id.items():
        kept_anns = []
        for ann in anns_by_image.get(img_id, []):
            old_cid = int(ann["category_id"])
            if old_cid not in CLASS_REMAP:
                continue
            new_cid = CLASS_REMAP[old_cid]
            kept_anns.append(
                {
                    "id": next_anno_id,
                    "image_id": img_id,
                    "category_id": new_cid,
                    "bbox": [float(b) for b in ann["bbox"]],
                    "area": float(ann.get("area", 0.0)),
                    "iscrowd": 0,
                }
            )
            per_class_counts[new_cid] += 1
            next_anno_id += 1

        if not kept_anns:
            images_dropped_no_anns += 1
            continue

        # Copy image
        src_img = src_imgs_dir / img_info["file_name"]
        if not src_img.exists():
            images_dropped_no_anns += 1
            continue
        dst_img = dst_imgs_dir / img_info["file_name"]
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        new_images.append(
            {
                "id": img_id,
                "file_name": img_info["file_name"],
                "width": int(img_info["width"]),
                "height": int(img_info["height"]),
            }
        )
        new_annotations.extend(kept_anns)
        images_kept += 1

    out = {
        "info": {"description": f"İşGözcü v5 — Construction-PPE remapped to v3's 10-class schema, split={split}", "version": "1.0"},
        "licenses": [],
        "categories": V3_CATEGORIES,
        "images": new_images,
        "annotations": new_annotations,
    }

    coco_dir = DST_ROOT / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    out_json = coco_dir / f"{split}.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "split": split,
        "images_kept": images_kept,
        "images_dropped": images_dropped_no_anns,
        "annotations": len(new_annotations),
        "per_class": dict(per_class_counts),
        "json_path": str(out_json.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def main() -> int:
    print("=" * 64)
    print("Construction-PPE → v3 schema (v5 fine-tune corpus)")
    print("=" * 64)
    if not SRC_ROOT.exists():
        print(f"[hata] Kaynak yok: {SRC_ROOT}")
        return 2

    summaries = []
    for split in SPLITS:
        try:
            summary = remap_split(split)
        except Exception as e:
            print(f"[hata] {split}: {e}")
            return 1
        summaries.append(summary)

    print()
    for s in summaries:
        print(f"[{s['split']}] {s['images_kept']} görüntü, {s['annotations']} kutu (atlanan: {s['images_dropped']}) → {s['json_path']}")
        for cid in sorted(s["per_class"]):
            print(f"  v3 class {cid:>2} {V3_NAMES[cid]:<14} {s['per_class'][cid]}")
    print()
    print(f"Çıktı: {DST_ROOT}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
