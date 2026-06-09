"""TEMP — _sanity_check.py — proves the v3 RT-DETR model loads and infers.

Prints only; writes/saves NOTHING. Safe to delete after review.
DIAGNOSTIC ONLY — not part of the production code path.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v3"
THRESHOLD = 0.25


def pick_image() -> Path:
    """Return the first available self-captured image that should contain people."""
    base = ROOT / "data" / "training_images"
    for sub in ("full_ppe", "no_hardhat", "no_vest", "no_ppe"):
        imgs = sorted((base / sub).glob("*.jpg"))
        if imgs:
            return imgs[0]
    imgs = sorted(base.rglob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No .jpg under {base}")
    return imgs[0]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("ISGOZCU SANITY CHECK - v3 RT-DETR")
    print("=" * 60)
    print(f"torch            : {torch.__version__}")
    print(f"cuda_available   : {torch.cuda.is_available()}")
    print(f"device           : {device}")
    if device == "cuda":
        print(f"gpu              : {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print(f"model_dir        : {MODEL_DIR}")
    if not MODEL_DIR.exists():
        print(f"ERROR: model dir not found: {MODEL_DIR}")
        return 2

    processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForObjectDetection.from_pretrained(str(MODEL_DIR)).to(device).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"num_labels       : {len(id2label)}")
    cmap = {str(k): id2label[k] for k in sorted(id2label)}
    print("id2label         : " + json.dumps(cmap, ensure_ascii=False))

    img_path = pick_image()
    image = Image.open(img_path).convert("RGB")
    print(f"test_image       : {img_path.name} ({image.width}x{image.height})")

    inputs = processor(images=image, return_tensors="pt").to(device)

    # warmup (untimed) -- absorbs CUDA / transformers first-call JIT cost
    with torch.no_grad():
        _ = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    target_sizes = torch.tensor([[image.height, image.width]], device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=THRESHOLD
    )[0]

    n = len(results["boxes"])
    print(f"inference_ms_warm: {infer_ms:.1f}")
    if device == "cuda":
        print(f"peak_gpu_mem_MB  : {torch.cuda.max_memory_allocated() / (1024 * 1024):.1f}")
    print(f"detections@{THRESHOLD} : {n}")
    pairs = sorted(zip(results["scores"].tolist(), results["labels"].tolist()), reverse=True)
    for score, label in pairs:
        print(f"    - {id2label.get(int(label), '?'):<16} conf={score:.3f}")
    print("=" * 60)
    print("RESULT: MODEL IS ALIVE - loaded + inferred OK")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
