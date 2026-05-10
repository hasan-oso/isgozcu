"""Select 500 diverse images from the Roboflow Construction-Site-Safety
dataset to be re-annotated by hand, then exported into a clean v4 training
set. (Phase 2A.4 — selection.)

Expects the original Roboflow COCO export at:

    data/roboflow_dataset/
        ├── train/
        │   ├── _annotations.coco.json
        │   └── *.jpg
        ├── valid/
        │   ├── _annotations.coco.json
        │   └── *.jpg
        └── test/
            ├── _annotations.coco.json
            └── *.jpg

Writes:

    data/roboflow_reannotation/
        ├── images/                       (the 500 selected JPEGs, copied
        │                                  with mode prefix to avoid collisions)
        ├── original_annotations.json     ({filename: [coco-style boxes]})
        └── selection_report.json         (counts + scoring debug info)

Selection is **deterministic** for reproducibility: we read images in sorted
order and the random shuffle uses a fixed seed.

Selection logic
---------------
1. Read every image from train/ + valid/ that has at least one box in the
   five "PPE-relevant" classes: Hardhat, NO-Hardhat, Safety Vest,
   NO-Safety Vest, Person.
2. Score each image on (object count, class diversity, aspect ratio, size).
3. Classify each image into one of four buckets:
       compliant     — has Hardhat/Safety Vest, no NO-*
       non_compliant — has NO-Hardhat or NO-Safety Vest, no compliant tags
       mixed         — has both compliant and NO-* tags in the same scene
       challenging   — many small boxes (workers far from camera) OR
                       extreme aspect ratio (likely fish-eye CCTV)
4. Pick the top 150 / 150 / 100 / 100 from each bucket by score.
5. Copy + record. Falls back to the next-bucket overflow if a bucket is
   short on candidates (so we always end up with exactly 500 if the dataset
   is large enough).
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# UTF-8 console for Turkish output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "roboflow_dataset"
OUT_ROOT = PROJECT_ROOT / "data" / "roboflow_reannotation"
OUT_IMAGES = OUT_ROOT / "images"
OUT_ORIG_ANNO = OUT_ROOT / "original_annotations.json"
OUT_REPORT = OUT_ROOT / "selection_report.json"

PPE_RELEVANT = {
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
    "Person",
}
COMPLIANT_CLASSES = {"Hardhat", "Safety Vest", "Mask"}
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}

TARGETS = {
    "compliant": 150,
    "non_compliant": 150,
    "mixed": 100,
    "challenging": 100,
}
TOTAL_TARGET = sum(TARGETS.values())  # 500

SUBSETS = ("train", "valid")  # we deliberately skip test/ to keep it as held-out


# ---------- types ----------


@dataclass
class ImageRecord:
    subset: str  # "train" / "valid"
    file_name: str
    width: int
    height: int
    boxes: list[dict] = field(default_factory=list)  # COCO-style with class name resolved
    score: float = 0.0
    bucket: str = ""

    @property
    def source_path(self) -> Path:
        return DATASET_ROOT / self.subset / self.file_name

    @property
    def out_filename(self) -> str:
        # Prefix subset to avoid filename collisions across train/valid.
        return f"{self.subset}__{self.file_name}"


# ---------- loading ----------


def load_subset(subset: str) -> dict[int, ImageRecord]:
    """Returns {image_id: ImageRecord} for one COCO subset directory."""
    coco_path = DATASET_ROOT / subset / "_annotations.coco.json"
    if not coco_path.exists():
        return {}
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    images: dict[int, ImageRecord] = {}
    for img in data.get("images", []):
        images[img["id"]] = ImageRecord(
            subset=subset,
            file_name=img["file_name"],
            width=int(img.get("width", 0)),
            height=int(img.get("height", 0)),
        )
    for ann in data.get("annotations", []):
        rec = images.get(ann["image_id"])
        if rec is None:
            continue
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        rec.boxes.append(
            {
                "class_id": int(ann["category_id"]),
                "class_name": cats.get(int(ann["category_id"]), "?"),
                "bbox": [float(b) for b in bbox],
                "area": float(ann.get("area", bbox[2] * bbox[3])),
            }
        )
    return images


def load_all() -> list[ImageRecord]:
    out: list[ImageRecord] = []
    for subset in SUBSETS:
        recs = load_subset(subset)
        out.extend(sorted(recs.values(), key=lambda r: r.file_name))
    return out


# ---------- scoring + bucketing ----------


def is_relevant(rec: ImageRecord) -> bool:
    return any(b["class_name"] in PPE_RELEVANT for b in rec.boxes)


def score_image(rec: ImageRecord) -> float:
    if rec.width <= 0 or rec.height <= 0:
        return 0.0
    object_count = len(rec.boxes)
    class_div = len({b["class_name"] for b in rec.boxes if b["class_name"] in PPE_RELEVANT})
    # Prefer landscape ~16:9 (CCTV-like). Punish portrait.
    ar = rec.width / max(1, rec.height)
    target_ar = 16 / 9
    ar_factor = math.exp(-((math.log(max(0.05, ar / target_ar))) ** 2))  # 1.0 at target, falls off
    # Prefer larger images (>= 800 px on the longer side).
    size_factor = min(1.0, max(rec.width, rec.height) / 1280.0)
    # Object density without rewarding spam (>20 boxes is usually a labelling mess).
    density_factor = math.log1p(min(object_count, 20))
    diversity_factor = 1.0 + 0.4 * class_div  # mix of classes is strictly more useful
    return density_factor * diversity_factor * (0.5 + 0.5 * ar_factor) * (0.4 + 0.6 * size_factor)


def bucket_for(rec: ImageRecord) -> str:
    classes = {b["class_name"] for b in rec.boxes}
    has_compliant = bool(classes & COMPLIANT_CLASSES)
    has_violation = bool(classes & VIOLATION_CLASSES)

    # "challenging" gate: lots of tiny boxes or strange aspect ratio.
    if rec.width > 0 and rec.height > 0:
        ar = rec.width / rec.height
        small_boxes = sum(1 for b in rec.boxes if b.get("area", 0) < 1500)
        if (
            (small_boxes >= max(3, len(rec.boxes) // 2))
            or ar < 0.7
            or ar > 2.4
        ):
            return "challenging"

    if has_compliant and has_violation:
        return "mixed"
    if has_compliant:
        return "compliant"
    if has_violation:
        return "non_compliant"
    # `Person` only with no PPE tags — useful for the "challenging" backlog
    return "challenging"


# ---------- selection ----------


def stratified_pick(records: list[ImageRecord]) -> list[ImageRecord]:
    by_bucket: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_bucket[r.bucket].append(r)
    for bucket in by_bucket:
        by_bucket[bucket].sort(key=lambda r: r.score, reverse=True)

    chosen: list[ImageRecord] = []
    chosen_set: set[str] = set()
    for bucket, target in TARGETS.items():
        for rec in by_bucket.get(bucket, []):
            if rec.out_filename in chosen_set:
                continue
            chosen.append(rec)
            chosen_set.add(rec.out_filename)
            if sum(1 for c in chosen if c.bucket == bucket) >= target:
                break

    # Fill any shortfall from the next-best across buckets.
    if len(chosen) < TOTAL_TARGET:
        remaining = sorted(
            (r for r in records if r.out_filename not in chosen_set),
            key=lambda r: r.score,
            reverse=True,
        )
        for rec in remaining:
            chosen.append(rec)
            chosen_set.add(rec.out_filename)
            if len(chosen) >= TOTAL_TARGET:
                break
    return chosen[:TOTAL_TARGET]


# ---------- main ----------


def main() -> int:
    print("=" * 64)
    print("Phase 2A.4 — Roboflow image selection (deterministic)")
    print("=" * 64)

    if not DATASET_ROOT.exists():
        print(f"[hata] Roboflow veri seti bulunamadı: {DATASET_ROOT}")
        print()
        print("Beklenen yapı:")
        print(f"  {DATASET_ROOT}\\train\\_annotations.coco.json + *.jpg")
        print(f"  {DATASET_ROOT}\\valid\\_annotations.coco.json + *.jpg")
        print()
        print("Çözümler:")
        print("  1) Roboflow Universe'den 'Construction Site Safety' veri setini")
        print("     COCO formatında indir, yukarıdaki yola çıkart.")
        print("  2) Veya .env dosyasına ROBOFLOW_API_KEY ekle (sonraki sürümde")
        print("     bu betik otomatik indirebilir).")
        return 2

    records = load_all()
    if not records:
        print(f"[hata] Hiç görüntü bulunamadı (train/ + valid/ boş). Veri seti tam mı?")
        return 2

    # Filter to PPE-relevant images
    relevant = [r for r in records if is_relevant(r)]
    print(f"Toplam görüntü:       {len(records)}")
    print(f"PPE içeren görüntü:   {len(relevant)}")

    # Score + bucket
    for rec in relevant:
        rec.score = score_image(rec)
        rec.bucket = bucket_for(rec)

    bucket_counts: dict[str, int] = defaultdict(int)
    for rec in relevant:
        bucket_counts[rec.bucket] += 1
    print()
    print("Kategori dağılımı (mevcut):")
    for b, t in TARGETS.items():
        print(f"  {b:<14} {bucket_counts.get(b, 0):>5} (hedef: {t})")
    print()

    chosen = stratified_pick(relevant)

    # Copy + record
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Görüntüler kopyalanıyor → {OUT_IMAGES}")

    original_annotations: dict[str, list[dict]] = {}
    final_bucket_counts: dict[str, int] = defaultdict(int)
    total_bytes = 0
    failures = 0
    for i, rec in enumerate(chosen, 1):
        src = rec.source_path
        dst = OUT_IMAGES / rec.out_filename
        try:
            if not dst.exists():
                shutil.copy2(src, dst)
            total_bytes += dst.stat().st_size
            original_annotations[rec.out_filename] = [
                {
                    "class_id": b["class_id"],
                    "class_name": b["class_name"],
                    "bbox": b["bbox"],  # already [x, y, w, h] COCO
                    "area": b["area"],
                }
                for b in rec.boxes
            ]
            final_bucket_counts[rec.bucket] += 1
        except Exception as e:
            failures += 1
            print(f"  [uyarı] kopyalama başarısız {src.name}: {e}")
        if i % 50 == 0 or i == len(chosen):
            print(f"  {i}/{len(chosen)}")

    OUT_ORIG_ANNO.write_text(
        json.dumps(original_annotations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "total_in_dataset": len(records),
        "relevant_images": len(relevant),
        "available_per_bucket": {b: bucket_counts.get(b, 0) for b in TARGETS},
        "selected_per_bucket": {b: final_bucket_counts.get(b, 0) for b in TARGETS},
        "selected_total": len(chosen) - failures,
        "copy_failures": failures,
        "output_dir": str(OUT_IMAGES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 64)
    print(f"SEÇİM TAMAMLANDI — {report['selected_total']} görüntü")
    print("=" * 64)
    for b, t in TARGETS.items():
        print(f"  {b:<14} {final_bucket_counts.get(b, 0):>3} / {t}")
    print(f"  Disk:          ~{report['total_mb']} MB")
    print(f"  Görüntüler:    {OUT_IMAGES}")
    print(f"  Etiketler:     {OUT_ORIG_ANNO}")
    print(f"  Rapor:         {OUT_REPORT}")
    print()
    print("Sonraki adım: python tests/annotate_roboflow_images.py")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
