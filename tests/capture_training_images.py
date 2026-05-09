"""Self-image capture tool for RT-DETR v3 fine-tuning (Phase 2A.3).

Cycles through four PPE-state modes (full_ppe / no_hardhat / no_vest / no_ppe).
For each mode, captures 50 images at fixed intervals with random Turkish pose
prompts to encourage dataset variation. Auto-skips frames that are too dark to
be useful (mean brightness < 30).

Output:
    data/training_images/{mode}/{mode}_{YYYYMMDD_HHMMSS}_{seq:03d}.jpg
    data/training_images/manifest.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

# UTF-8 console for Turkish print() output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Turkish-safe text overlay (Pillow under the hood)
from backend.inference.text_render import put_text_outlined  # noqa: E402


# ---------- config ----------


class Mode(str, Enum):
    FULL_PPE = "full_ppe"
    NO_HARDHAT = "no_hardhat"
    NO_VEST = "no_vest"
    NO_PPE = "no_ppe"


MODE_ORDER: list[Mode] = [Mode.FULL_PPE, Mode.NO_HARDHAT, Mode.NO_VEST, Mode.NO_PPE]

MODE_CONFIG: dict[Mode, dict] = {
    Mode.FULL_PPE: {
        "title": "FULL_PPE — Baret + Yelek (uyumlu)",
        "labels": ["Hardhat", "Safety Vest", "Person"],
    },
    Mode.NO_HARDHAT: {
        "title": "NO_HARDHAT — Yalnız yelek, baret yok",
        "labels": ["NO-Hardhat", "Safety Vest", "Person"],
    },
    Mode.NO_VEST: {
        "title": "NO_VEST — Yalnız baret, yelek yok",
        "labels": ["Hardhat", "NO-Safety Vest", "Person"],
    },
    Mode.NO_PPE: {
        "title": "NO_PPE — Hiçbir KKD yok",
        "labels": ["NO-Hardhat", "NO-Safety Vest", "Person"],
    },
}

KEY_TO_MODE: dict[int, Mode] = {
    ord("1"): Mode.FULL_PPE,
    ord("2"): Mode.NO_HARDHAT,
    ord("3"): Mode.NO_VEST,
    ord("4"): Mode.NO_PPE,
}

POSE_PROMPTS_TR: list[str] = [
    "Başınızı sola çevirin",
    "Başınızı sağa çevirin",
    "Doğal duruşunuzu alın",
    "Hafifçe öne eğilin",
    "Hafifçe geriye eğilin",
    "Kameraya bakın",
    "Kameranın yakınına gelin",
    "Kameradan biraz uzaklaşın",
    "Kollarınızı kaldırın",
    "Kollarınızı indirin",
]

OUTPUT_DIR = PROJECT_ROOT / "data" / "training_images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

CAPTURES_PER_MODE = 50
INTERVAL_SEC = 3.0  # 3-2-1 countdown between each capture
MIN_BRIGHTNESS = 30.0
JPEG_QUALITY = 95
WINDOW_TITLE = "İşGözcü - Eğitim Görüntüsü Toplama"
COMPLETE_PAUSE_SEC = 2.5
FLASH_FRAMES = 2  # number of display frames to draw the white flash overlay


# ---------- state ----------


class State(str, Enum):
    IDLE = "idle"
    MODE_SELECTED = "mode_selected"
    CAPTURING = "capturing"
    COMPLETE = "complete"


# ---------- helpers ----------


def count_existing_images(mode: Mode) -> int:
    folder = OUTPUT_DIR / mode.value
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


def update_manifest() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total_images": 0,
        "modes": {},
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "user_notes": "Self-captured webcam images for v3 fine-tuning",
    }
    for mode in MODE_ORDER:
        n = count_existing_images(mode)
        manifest["modes"][mode.value] = {
            "count": n,
            "labels": MODE_CONFIG[mode]["labels"],
        }
        manifest["total_images"] += n
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def is_too_dark(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < MIN_BRIGHTNESS


def save_capture(mode: Mode, frame: np.ndarray, seq: int) -> Path:
    folder = OUTPUT_DIR / mode.value
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{mode.value}_{ts}_{seq:03d}.jpg"
    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return path


# ---------- overlay ----------


def draw_translucent_band(canvas: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    H, W = canvas.shape[:2]
    x2, y2 = min(W, x + w), min(H, y + h)
    x1, y1 = max(0, x), max(0, y)
    if x2 <= x1 or y2 <= y1:
        return
    sub = canvas[y1:y2, x1:x2]
    overlay = np.zeros_like(sub)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)


def draw_overlay(
    canvas: np.ndarray,
    *,
    state: State,
    mode: Mode | None,
    captured_count: int,
    target_count: int,
    prompt: str,
    secs_until_capture: float,
    counts_per_mode: dict[Mode, int],
    flash_remaining: int,
) -> None:
    h, w = canvas.shape[:2]

    # White flash overlay right after a capture lands.
    if flash_remaining > 0:
        flash = np.full_like(canvas, 255)
        cv2.addWeighted(flash, 0.7, canvas, 0.3, 0, canvas)

    # Top mode-title band.
    draw_translucent_band(canvas, 0, 0, w, 64)
    if mode is not None:
        title = MODE_CONFIG[mode]["title"]
    else:
        title = "Mod seçilmedi — 1/2/3/4 tuşuna basın"
    put_text_outlined(canvas, title, (16, 42), scale=0.9, thickness=2)

    # Top-right progress / counts.
    if mode is not None and state in (State.MODE_SELECTED, State.CAPTURING, State.COMPLETE):
        progress = f"{captured_count}/{target_count}"
        put_text_outlined(canvas, progress, (w - 130, 42), scale=0.9, thickness=2)

    # Center countdown digit during CAPTURING (3..2..1).
    if state == State.CAPTURING and secs_until_capture > 0.05:
        digit = str(int(np.ceil(secs_until_capture)))
        # Big center number — ASCII so cv2.putText is fine and crisper than PIL.
        (tw, th), _ = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX, 8.0, 14)
        cx, cy = (w - tw) // 2, (h + th) // 2 - 30
        cv2.putText(canvas, digit, (cx + 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 8.0, (0, 0, 0), 22, cv2.LINE_AA)
        cv2.putText(canvas, digit, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 8.0, (60, 200, 255), 14, cv2.LINE_AA)

    # Prompt during CAPTURING (mid-lower, Turkish-safe).
    if state == State.CAPTURING and prompt:
        draw_translucent_band(canvas, 0, h - 120, w, 50)
        put_text_outlined(canvas, prompt, (16, h - 88), scale=0.9, thickness=2, fg=(60, 200, 255))

    # Mode-selected hint.
    if state == State.MODE_SELECTED:
        msg = "Hazır olduğunuzda SPACE tuşuna basarak başlayın"
        draw_translucent_band(canvas, 0, h - 120, w, 50)
        put_text_outlined(canvas, msg, (16, h - 88), scale=0.85, thickness=2, fg=(60, 220, 60))

    # Complete banner.
    if state == State.COMPLETE:
        msg = f"Mod tamamlandı! {captured_count} görüntü kaydedildi. 1/2/3/4 ile sonraki moda geçin."
        draw_translucent_band(canvas, 0, h // 2 - 30, w, 60)
        put_text_outlined(canvas, msg, (16, h // 2 + 8), scale=0.8, thickness=2, fg=(60, 220, 60))

    # IDLE summary of existing counts.
    if state == State.IDLE:
        draw_translucent_band(canvas, 0, h - 230, w, 110)
        put_text_outlined(
            canvas,
            "Mevcut görüntü sayıları:",
            (16, h - 200),
            scale=0.7,
            thickness=2,
        )
        for i, m in enumerate(MODE_ORDER):
            n = counts_per_mode.get(m, 0)
            color = (60, 220, 60) if n >= CAPTURES_PER_MODE else (200, 200, 200)
            put_text_outlined(
                canvas,
                f"{i + 1}) {m.value}: {n} / {CAPTURES_PER_MODE}",
                (40 + (i % 2) * 380, h - 170 + (i // 2) * 28),
                scale=0.6,
                thickness=1,
                fg=color,
            )

    # Bottom instructions strip — always visible.
    draw_translucent_band(canvas, 0, h - 56, w, 56)
    put_text_outlined(
        canvas,
        "[1] FULL_PPE  [2] NO_HARDHAT  [3] NO_VEST  [4] NO_PPE  [SPACE] Başlat  [Q] Çıkış",
        (16, h - 22),
        scale=0.65,
        thickness=2,
    )


# ---------- main loop ----------


def main() -> int:
    print("=" * 60)
    print("İşGözcü - Eğitim Görüntüsü Toplama (Phase 2A.3)")
    print("=" * 60)

    # Show existing counts
    counts_per_mode: dict[Mode, int] = {m: count_existing_images(m) for m in MODE_ORDER}
    total_existing = sum(counts_per_mode.values())
    print(f"Mevcut: {total_existing} / {CAPTURES_PER_MODE * len(MODE_ORDER)} görüntü")
    for m in MODE_ORDER:
        print(f"  {m.value:<12} {counts_per_mode[m]:>3} / {CAPTURES_PER_MODE}")
    print()

    print("Webcam açılıyor...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("HATA: Webcam açılamadı.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam {actual_w}x{actual_h} hazır.")
    print("1/2/3/4 ile mod seç, SPACE ile başla, Q ile çık.\n")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    state: State = State.IDLE
    current_mode: Mode | None = None
    captured_this_session = 0  # captured during this mode run
    target_count: int = CAPTURES_PER_MODE  # absolute target (existing + delta)
    next_capture_at: float = 0.0  # wall clock
    current_prompt: str = ""
    flash_remaining: int = 0
    complete_started_at: float = 0.0
    skipped_dark = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[uyarı] Webcam karesi okunamadı.")
                time.sleep(0.05)
                continue

            now = time.time()

            # State transitions / per-frame logic.
            if state == State.CAPTURING:
                secs_until_capture = next_capture_at - now
                if secs_until_capture <= 0:
                    if is_too_dark(frame):
                        skipped_dark += 1
                        # Don't increment count, just push timer for another shot.
                        next_capture_at = now + INTERVAL_SEC
                        if skipped_dark % 5 == 1:
                            print("[uyarı] Kare çok karanlık (parlaklık < 30); atlanıyor.")
                    else:
                        # Save raw frame (no overlay baked in).
                        existing = counts_per_mode.get(current_mode, 0)
                        seq = existing + 1
                        path = save_capture(current_mode, frame, seq)
                        counts_per_mode[current_mode] = seq
                        captured_this_session += 1
                        flash_remaining = FLASH_FRAMES
                        if captured_this_session % 10 == 0 or captured_this_session == 1:
                            print(
                                f"  ✓ {current_mode.value} #{seq} — {path.name}"
                            )
                        if seq >= target_count:
                            # Done with this mode.
                            update_manifest()
                            state = State.COMPLETE
                            complete_started_at = now
                            print(
                                f"[✓] {current_mode.value} tamamlandı — "
                                f"{captured_this_session} yeni görüntü kaydedildi.\n"
                            )
                        else:
                            next_capture_at = now + INTERVAL_SEC
                            current_prompt = random.choice(POSE_PROMPTS_TR)
                    secs_until_capture = max(0.0, next_capture_at - time.time())
            elif state == State.COMPLETE:
                if now - complete_started_at > COMPLETE_PAUSE_SEC:
                    state = State.IDLE
                    current_mode = None
                    captured_this_session = 0

            # Compose canvas (use a copy so we never bake overlay into the saved image).
            canvas = frame.copy()
            secs_until_capture_disp = (
                max(0.0, next_capture_at - now) if state == State.CAPTURING else 0.0
            )
            draw_overlay(
                canvas,
                state=state,
                mode=current_mode,
                captured_count=counts_per_mode.get(current_mode, 0) if current_mode else 0,
                target_count=target_count,
                prompt=current_prompt,
                secs_until_capture=secs_until_capture_disp,
                counts_per_mode=counts_per_mode,
                flash_remaining=flash_remaining,
            )
            if flash_remaining > 0:
                flash_remaining -= 1

            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Kullanıcı çıkışı.")
                break
            elif key in KEY_TO_MODE:
                picked = KEY_TO_MODE[key]
                # Allow switching mode any time except mid-capture.
                if state in (State.IDLE, State.MODE_SELECTED, State.COMPLETE):
                    current_mode = picked
                    state = State.MODE_SELECTED
                    target_count = CAPTURES_PER_MODE
                    captured_this_session = 0
                    skipped_dark = 0
                    print(
                        f"[mod] {picked.value} seçildi "
                        f"({counts_per_mode[picked]}/{target_count} mevcut). "
                        f"SPACE ile başlat."
                    )
            elif key == 32:  # SPACE
                if state == State.MODE_SELECTED and current_mode is not None:
                    if counts_per_mode[current_mode] >= target_count:
                        print(
                            f"[bilgi] {current_mode.value} zaten dolu "
                            f"({counts_per_mode[current_mode]}/{target_count})."
                        )
                        state = State.COMPLETE
                        complete_started_at = time.time()
                    else:
                        print(
                            f"[başlat] {current_mode.value} — "
                            f"{counts_per_mode[current_mode]} → {target_count} hedefleniyor"
                        )
                        state = State.CAPTURING
                        next_capture_at = time.time() + INTERVAL_SEC
                        current_prompt = random.choice(POSE_PROMPTS_TR)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        manifest = update_manifest()
        print()
        print("=" * 60)
        print("OTURUM ÖZETİ")
        print("=" * 60)
        print(f"Toplam görüntü:  {manifest['total_images']}")
        for mode_name, info in manifest["modes"].items():
            mark = "✓" if info["count"] >= CAPTURES_PER_MODE else "•"
            print(f"  {mark} {mode_name:<12} {info['count']:>3} / {CAPTURES_PER_MODE}")
        print(f"Manifest:        {MANIFEST_PATH}")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
