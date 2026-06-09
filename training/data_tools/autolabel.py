"""training/data_tools/autolabel.py - seed YOLO labels with the v3 model.

Loads the production v3 RT-DETR checkpoint and writes YOLO-format labels for
the 3 POSITIVE classes only (Person / Hardhat / Safety Vest). The other 7 v3
classes are dropped entirely. Output is meant to be corrected by review.py.

Run:
    python training/data_tools/autolabel.py
    python training/data_tools/autolabel.py --images training/captured --out training/labels --conf 0.25

YOLO line format:  <our_class_id> <cx> <cy> <w> <h>   (all normalised 0..1)
An image with no kept detections produces an EMPTY .txt (this is not an error).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model" / "weights" / "v3"
CLASS_MAP_JSON = PROJECT_ROOT / "model" / "class_map.json"
DEFAULT_IMAGES = PROJECT_ROOT / "training" / "captured"
DEFAULT_LABELS = PROJECT_ROOT / "training" / "labels"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Pipeline scope: target class NAME -> our new YOLO index (0,1,2).
TARGET_NAME_TO_OURS: dict[str, int] = {"Person": 0, "Hardhat": 1, "Safety Vest": 2}
OUR_CLASSES_IN_ORDER = ["Person", "Hardhat", "Safety Vest"]  # index 0, 1, 2

# ---------------------------------------------------------------------------
# Explicit, documented v3 (10-class) -> ours (3-class) index table.
# The live mapping below is built BY NAME from the model's id2label /
# class_map.json, but for the shipped v3 checkpoint it resolves exactly to:
#
#       v3 idx 5  "Person"        -> 0   (Isci)
#       v3 idx 0  "Hardhat"       -> 1   (Baret)
#       v3 idx 7  "Safety Vest"   -> 2   (Guvenlik yelegi)
#
# Dropped v3 indices: 1=Mask 2=NO-Hardhat 3=NO-Mask 4=NO-Safety Vest
#                     6=Safety Cone 8=machinery 9=vehicle
# ---------------------------------------------------------------------------
EXPECTED_V3_TO_OURS = {5: 0, 0: 1, 7: 2}


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def build_v3_to_ours(id2label: dict[int, str]) -> dict[int, int]:
    """Model output index -> our index, matched by class NAME."""
    return {
        v3_idx: TARGET_NAME_TO_OURS[name]
        for v3_idx, name in id2label.items()
        if name in TARGET_NAME_TO_OURS
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed YOLO labels with the v3 model.")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--out", default=str(DEFAULT_LABELS))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    images_root = Path(args.images)
    if not images_root.is_absolute():
        images_root = PROJECT_ROOT / images_root
    labels_root = Path(args.out)
    if not labels_root.is_absolute():
        labels_root = PROJECT_ROOT / labels_root
    labels_root.mkdir(parents=True, exist_ok=True)

    if not MODEL_DIR.exists():
        print(f"[HATA] Model yok: {MODEL_DIR}")
        return 2
    if not images_root.exists():
        print(f"[HATA] Goruntu klasoru yok: {images_root}")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"[bilgi] Model : {MODEL_DIR}")
    print(f"[bilgi] device={device} fp16={use_fp16} conf={args.conf}")

    processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForObjectDetection.from_pretrained(str(MODEL_DIR)).to(device).eval()
    if use_fp16:
        model = model.half()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # Cross-check class names against the clean class_map.json (names must agree).
    if CLASS_MAP_JSON.exists():
        cmap = {int(k): v for k, v in json.loads(CLASS_MAP_JSON.read_text(encoding="utf-8")).items()}
        if cmap != id2label:
            print("[uyari] class_map.json model id2label ile birebir ayni degil; model esas alindi.")

    v3_to_ours = build_v3_to_ours(id2label)
    print(f"[bilgi] v3->ours (isimle): {v3_to_ours}  (beklenen {EXPECTED_V3_TO_OURS})")
    if v3_to_ours != EXPECTED_V3_TO_OURS:
        print("[uyari] Harita beklenenden farkli; isimle eslesen harita kullanildi.")
    if len(v3_to_ours) != len(TARGET_NAME_TO_OURS):
        missing = set(TARGET_NAME_TO_OURS) - set(id2label.values())
        print(f"[HATA] Hedef siniflar modelde yok: {missing}")
        return 2

    # classes.txt (our 3 classes, in index order)
    (labels_root / "classes.txt").write_text(
        "\n".join(OUR_CLASSES_IN_ORDER) + "\n", encoding="utf-8"
    )

    images = find_images(images_root)
    if not images:
        print(f"[uyari] {images_root} altinda goruntu bulunamadi.")
        return 0

    per_class = {name: 0 for name in OUR_CLASSES_IN_ORDER}
    empty_imgs = 0

    for img_path in images:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[uyari] {img_path.name} acilamadi: {e}")
            continue
        W, H = image.width, image.height
        inputs = processor(images=image, return_tensors="pt").to(device)
        if use_fp16:
            inputs = {
                k: (v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v)
                for k, v in inputs.items()
            }
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[H, W]], device=device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.conf
        )[0]

        lines: list[str] = []
        for label, box in zip(results["labels"].tolist(), results["boxes"].tolist()):
            v3_idx = int(label)
            if v3_idx not in v3_to_ours:
                continue  # drop the 7 non-target classes
            our_idx = v3_to_ours[v3_idx]
            x1, y1, x2, y2 = box
            cx = min(max(((x1 + x2) / 2.0) / W, 0.0), 1.0)
            cy = min(max(((y1 + y2) / 2.0) / H, 0.0), 1.0)
            bw = min(max((x2 - x1) / W, 0.0), 1.0)
            bh = min(max((y2 - y1) / H, 0.0), 1.0)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{our_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_class[OUR_CLASSES_IN_ORDER[our_idx]] += 1

        # mirror the relative sub-path under labels_root, same stem + .txt
        out_path = (labels_root / img_path.relative_to(images_root)).with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not lines:
            empty_imgs += 1

    print("\n==== AUTOLABEL OZET ====")
    print(f"Goruntu sayisi       : {len(images)}")
    for name in OUR_CLASSES_IN_ORDER:
        print(f"  {name:<12} kutu : {per_class[name]}")
    print(f"Kutusuz (bos) goruntu: {empty_imgs}")
    print(f"Etiket klasoru       : {labels_root}")
    print("========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
