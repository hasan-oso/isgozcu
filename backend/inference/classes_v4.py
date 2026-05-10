"""Class-name registry for the v4 model (Phase 2A.5).

v4 is trained on the Ultralytics Construction-PPE dataset filtered to 6
production-relevant classes:

    0  Person      — İşçi
    1  helmet      — Baret
    2  vest        — Yelek
    3  gloves      — Eldiven
    4  no_helmet   — Baret yok
    5  no_gloves   — Eldiven yok

Compared to v1–v3 (10 classes), v4 drops `Mask`, `NO-Mask`, `Safety Cone`,
`machinery`, `vehicle`, `NO-Safety Vest`, and renames the helmet classes
to lowercase `helmet`/`no_helmet` and the vest class to `vest`.

This module is data only — no imports from the rest of the inference stack.
We'll wire it into `rtdetr_detector.py` / `violation_engine.py` once the v4
checkpoint is trained and validated.
"""

from __future__ import annotations

CLASS_NAMES_V4: list[str] = [
    "Person",
    "helmet",
    "vest",
    "gloves",
    "no_helmet",
    "no_gloves",
]

CLASS_NAMES_TR_V4: list[str] = [
    "İşçi",
    "Baret",
    "Yelek",
    "Eldiven",
    "Baret yok",
    "Eldiven yok",
]

# class_id ↔ name lookups (mirrors HF id2label / label2id convention).
ID2LABEL_V4: dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES_V4)}
LABEL2ID_V4: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES_V4)}
ID2LABEL_TR_V4: dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES_TR_V4)}

# Violation engine — non-compliance classes that fire alerts.
# Mapped to canonical violation_type strings used downstream by
# ViolationEvent + the dashboard's violation_label_tr lookups.
VIOLATION_CLASSES_V4: dict[str, str] = {
    "no_helmet": "Baret yok",
    "no_gloves": "Eldiven yok",
}

# Compliant-class reference set.
PPE_PRESENT_CLASSES_V4: set[str] = {"helmet", "vest", "gloves"}

# All class ids that count as a "person bbox" candidate for IoU matching.
PERSON_CLASSES_V4: set[str] = {"Person"}

# Bounding-box color palette (BGR) — matches the live_simulator + dashboard
# convention: green = compliant, red = violation, blue = person, gray = other.
CLASS_COLORS_V4: dict[str, tuple[int, int, int]] = {
    "Person": (220, 130, 50),       # blue
    "helmet": (60, 220, 60),        # green
    "vest": (60, 220, 60),          # green
    "gloves": (60, 220, 60),        # green
    "no_helmet": (50, 60, 230),     # red
    "no_gloves": (50, 60, 230),     # red
}


def is_violation(class_name: str) -> bool:
    return class_name in VIOLATION_CLASSES_V4


def is_compliant(class_name: str) -> bool:
    return class_name in PPE_PRESENT_CLASSES_V4


def violation_label_tr(class_name: str) -> str:
    return VIOLATION_CLASSES_V4.get(class_name, class_name)


__all__ = [
    "CLASS_NAMES_V4",
    "CLASS_NAMES_TR_V4",
    "ID2LABEL_V4",
    "LABEL2ID_V4",
    "ID2LABEL_TR_V4",
    "VIOLATION_CLASSES_V4",
    "PPE_PRESENT_CLASSES_V4",
    "PERSON_CLASSES_V4",
    "CLASS_COLORS_V4",
    "is_violation",
    "is_compliant",
    "violation_label_tr",
]
