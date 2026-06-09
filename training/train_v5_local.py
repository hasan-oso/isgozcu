"""Local fine-tune of v3 → v5 on the Construction-PPE-remapped clean corpus.

Constraints:
    * Hardware: RTX 3050 Laptop, 4 GB VRAM (≈3.46 GB free at start).
    * Strategy: continue training from v3 weights (NOT from scratch) so the
      classes the new dataset doesn't cover (Mask, NO-Mask, NO-Safety Vest,
      Safety Cone, machinery, vehicle) keep what v3 already learned.
    * Memory: batch=2 + grad_accum=8 (effective 16), fp16, gradient
      checkpointing, image_size=640.
    * Robustness: if the first attempt OOMs, retry once with batch=1 +
      grad_accum=16. If that also OOMs, write the trace to logs/ and exit
      with code 2 so the orchestrator can surface it.

Output:
    backend/models/rtdetr_isgozcu_ppe_v5/        (final checkpoint)
    backend/models/rtdetr_isgozcu_ppe_v5_checkpoints/  (deleted after success)
    logs/v5_training.log                         (live log + final stats)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# UTF-8 console for Turkish output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import torch
import numpy as np
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image as HFImage,
    Sequence,
    Value,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)
import albumentations as A

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "construction_ppe_v5"
V3_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v3"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v5_checkpoints"
FINAL_DIR = PROJECT_ROOT / "backend" / "models" / "rtdetr_isgozcu_ppe_v5"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "v5_training.log"

CATEGORIES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]


# ---------- shared helpers ----------


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_split(split_name: str) -> list[dict[str, Any]]:
    json_path = DATASET_ROOT / "coco" / f"{split_name}.json"
    images_dir = DATASET_ROOT / "images" / split_name
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    img_id_to_anns: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
    examples = []
    for img_id, img_info in img_id_to_info.items():
        anns = img_id_to_anns.get(img_id, [])
        if not anns:
            continue
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        bboxes, cats, areas = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            bboxes.append([float(x), float(y), float(w), float(h)])
            cats.append(int(ann["category_id"]))
            areas.append(float(ann["area"]))
        if not bboxes:
            continue
        examples.append(
            {
                "image_id": img_id,
                "image": str(img_path),
                "width": int(img_info["width"]),
                "height": int(img_info["height"]),
                "objects": {
                    "id": list(range(len(bboxes))),
                    "area": areas,
                    "bbox": bboxes,
                    "category": cats,
                },
            }
        )
    return examples


# ---------- training run ----------


def run_training(per_device_batch: int, grad_accum: int) -> dict:
    log(f"=== Trying batch={per_device_batch} grad_accum={grad_accum} (effective {per_device_batch*grad_accum}) ===")

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    free_pre, total_vram = torch.cuda.mem_get_info()
    log(f"VRAM total: {total_vram / 1e9:.2f} GB | free: {free_pre / 1e9:.2f} GB")

    log("Loading datasets...")
    train_examples = load_split("train")
    val_examples = load_split("val")
    log(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    class_counts = {i: 0 for i in range(10)}
    for ex in train_examples:
        for cat in ex["objects"]["category"]:
            class_counts[cat] += 1
    log("Train class distribution:")
    for i, name in enumerate(CATEGORIES):
        log(f"  {i}: {name:<18} {class_counts[i]:>5}")

    features = Features(
        {
            "image_id": Value("int64"),
            "image": HFImage(),
            "width": Value("int32"),
            "height": Value("int32"),
            "objects": Sequence(
                {
                    "id": Value("int64"),
                    "area": Value("float32"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "category": ClassLabel(names=CATEGORIES),
                }
            ),
        }
    )
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_examples, features=features),
            "validation": Dataset.from_list(val_examples, features=features),
        }
    )

    log(f"Loading v3 from {V3_MODEL_PATH}")
    image_processor = AutoImageProcessor.from_pretrained(str(V3_MODEL_PATH))
    model = AutoModelForObjectDetection.from_pretrained(str(V3_MODEL_PATH))
    model = model.to("cuda")

    log(f"VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, fill=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    valid_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, fill=0),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1),
    )

    def format_annotations(image_id, categories, areas, bboxes):
        return [
            {"image_id": image_id, "category_id": int(cat), "isfinal": 1,
             "iscrowd": 0, "area": float(area), "bbox": [float(b) for b in bbox]}
            for cat, area, bbox in zip(categories, areas, bboxes)
        ]

    def transform_train(examples):
        images, annotations = [], []
        for image, image_id, objects in zip(examples["image"], examples["image_id"], examples["objects"]):
            image_array = np.array(image.convert("RGB"))
            out = train_transform(image=image_array, bboxes=objects["bbox"], category=objects["category"])
            anns = format_annotations(
                image_id,
                out["category"],
                [bbox[2] * bbox[3] for bbox in out["bboxes"]],
                out["bboxes"],
            )
            images.append(out["image"])
            annotations.append({"image_id": image_id, "annotations": anns})
        return image_processor(images=images, annotations=annotations, return_tensors="pt")

    def transform_valid(examples):
        images, annotations = [], []
        for image, image_id, objects in zip(examples["image"], examples["image_id"], examples["objects"]):
            image_array = np.array(image.convert("RGB"))
            out = valid_transform(image=image_array, bboxes=objects["bbox"], category=objects["category"])
            anns = format_annotations(
                image_id,
                out["category"],
                [bbox[2] * bbox[3] for bbox in out["bboxes"]],
                out["bboxes"],
            )
            images.append(out["image"])
            annotations.append({"image_id": image_id, "annotations": anns})
        return image_processor(images=images, annotations=annotations, return_tensors="pt")

    dataset["train"] = dataset["train"].with_transform(transform_train)
    dataset["validation"] = dataset["validation"].with_transform(transform_valid)

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "labels": [item["labels"] for item in batch],
        }

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=15,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        weight_decay=1e-4,
        max_grad_norm=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        fp16=True,
        # On Windows + Trainer, num_workers > 0 is fragile (multiprocessing).
        # 0 is slightly slower but rock-solid.
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        push_to_hub=False,
        # NOTE: RT-DETR (transformers 4.46) does not support gradient_checkpointing
        # — the model raises ValueError if you turn it on. We rely on fp16 +
        # small batch + the batch=1 fallback below for VRAM headroom instead.
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=image_processor,
        data_collator=collate_fn,
    )

    log("=" * 60)
    log("Starting v5 fine-tuning (continued from v3)")
    log("=" * 60)
    log(f"Effective batch size: {per_device_batch} × {grad_accum} = {per_device_batch * grad_accum}")
    log(f"Steps per epoch: ~{max(1, len(train_examples) // (per_device_batch * grad_accum))}")
    log(f"Total target steps: ~{max(1, len(train_examples) // (per_device_batch * grad_accum)) * 15}")

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    log("")
    log("=" * 60)
    log("Training completed.")
    log("=" * 60)
    log(f"Total time: {elapsed/60:.2f} minutes")
    log(f"Peak VRAM:  {peak_vram:.2f} GB")
    log(f"Final train loss: {train_result.metrics.get('train_loss', 0.0):.4f}")
    log(f"Samples/second:   {train_result.metrics.get('train_samples_per_second', 0.0):.2f}")

    # Pull eval-loss history out of trainer state for the final report.
    eval_history = []
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            eval_history.append(
                {"epoch": float(entry.get("epoch", 0)), "eval_loss": float(entry["eval_loss"])}
            )
    best = min(eval_history, key=lambda e: e["eval_loss"]) if eval_history else None
    final_eval = eval_history[-1] if eval_history else None
    if best:
        log(f"Best  val loss: {best['eval_loss']:.4f}  (epoch {best['epoch']:.2f})")
    if final_eval:
        log(f"Final val loss: {final_eval['eval_loss']:.4f}  (epoch {final_eval['epoch']:.2f})")

    # Save final
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_DIR))
    image_processor.save_pretrained(str(FINAL_DIR))
    log(f"Saved: {FINAL_DIR}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        log("Cleaned up checkpoint dir")

    return {
        "elapsed_minutes": elapsed / 60,
        "peak_vram_gb": peak_vram,
        "final_train_loss": float(train_result.metrics.get("train_loss", 0.0)),
        "samples_per_second": float(train_result.metrics.get("train_samples_per_second", 0.0)),
        "best_eval": best,
        "final_eval": final_eval,
        "eval_history": eval_history,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
    }


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")  # truncate

    if not V3_MODEL_PATH.exists():
        log(f"[hata] v3 modeli yok: {V3_MODEL_PATH}")
        return 2
    if not (DATASET_ROOT / "coco" / "train.json").exists():
        log(f"[hata] v5 dataset hazır değil: {DATASET_ROOT}")
        log("       Önce: python tests/prepare_construction_ppe_for_v5.py")
        return 2

    attempts = [
        {"per_device_batch": 2, "grad_accum": 8},   # primary
        {"per_device_batch": 1, "grad_accum": 16},  # fallback if first OOMs
    ]
    last_err: Exception | None = None
    for i, cfg in enumerate(attempts, 1):
        try:
            log(f"### Attempt {i}/{len(attempts)} ###")
            stats = run_training(**cfg)
            stats_path = LOG_DIR / "v5_training_stats.json"
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            log(f"Stats persisted to {stats_path}")
            return 0
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            log(f"[OOM] {e}")
            torch.cuda.empty_cache()
            if i < len(attempts):
                log("Retrying with smaller per-device batch...")
                continue
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                last_err = e
                log(f"[OOM RuntimeError] {e}")
                torch.cuda.empty_cache()
                if i < len(attempts):
                    log("Retrying with smaller per-device batch...")
                    continue
            log("[fatal RuntimeError]")
            log(traceback.format_exc())
            return 1
        except Exception as e:
            log(f"[fatal {type(e).__name__}]")
            log(traceback.format_exc())
            return 1

    log(f"[hata] Tüm denemeler OOM ile başarısız oldu — son: {last_err}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
