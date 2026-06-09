"""training/data_tools/capture.py - phone-camera capture for the v3 fine-tune dataset.

Captures full-resolution stills from an Android phone exposed as a webcam
(DroidCam / USB). No annotation happens here - that is autolabel.py + review.py.

Scope: positive PPE classes only (Person / Hardhat / Safety Vest). This tool
only grabs images; the case tag just organises them on disk.

Run:
    python training/data_tools/capture.py --person ali --case full
    python training/data_tools/capture.py --person ahmet --case no_helmet --out training/captured
    python training/data_tools/capture.py --person baba --case group --index 1

Keys:  SPACE = capture & save    Q / ESC = quit
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "training" / "captured"
VALID_CASES = ("full", "no_helmet", "no_vest", "none", "group")
MAX_PROBE_INDEX = 5
WINDOW = "isgozcu capture  [SPACE]=kaydet  [Q]=cikis"


def _open(index: int):
    """Open a camera index with the most reliable Windows backend, then fall back."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


def probe_cameras(max_index: int = MAX_PROBE_INDEX) -> list[tuple[int, int, int]]:
    """Return [(index, width, height), ...] for cameras that open and yield a frame."""
    found: list[tuple[int, int, int]] = []
    for idx in range(max_index + 1):
        cap = _open(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((idx, int(w), int(h)))
        cap.release()
    return found


def choose_camera(found: list[tuple[int, int, int]], default: int = 1) -> int:
    """Print available cameras and prompt for a choice (Enter = default if present)."""
    if not found:
        raise RuntimeError("Hic kamera bulunamadi (0..5). DroidCam/USB bagli mi?")
    print("\nBulunan kameralar:")
    for idx, w, h in found:
        tag = "  <- varsayilan (telefon genelde 1)" if idx == default else ""
        print(f"  [{idx}]  {w}x{h}{tag}")
    available = {idx for idx, _, _ in found}
    fallback = default if default in available else found[0][0]
    raw = input(f"\nKamera indeksi sec [{fallback}]: ").strip()
    if raw == "":
        return fallback
    try:
        choice = int(raw)
    except ValueError:
        print(f"  Gecersiz giris; {fallback} kullaniliyor.")
        return fallback
    if choice not in available:
        print(f"  {choice} mevcut degil; {fallback} kullaniliyor.")
        return fallback
    return choice


def _overlay(frame, person: str, case: str, count: int):
    """Return a PREVIEW copy with a translucent top banner (never saved to disk)."""
    disp = frame.copy()
    h, w = disp.shape[:2]
    band_h = min(64, h)
    band = disp[0:band_h, 0:w]
    disp[0:band_h, 0:w] = (band * 0.35).astype(band.dtype)
    cv2.putText(disp, f"Kisi: {person}   Durum: {case}", (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(disp, f"Bu oturumda kaydedilen: {count}", (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 230, 120), 2, cv2.LINE_AA)
    cv2.putText(disp, "[SPACE] kaydet  [Q] cikis", (max(16, w - 350), 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
    return disp


def main() -> int:
    ap = argparse.ArgumentParser(description="Phone-camera capture for the PPE dataset.")
    ap.add_argument("--person", required=True, help="Person tag, e.g. ali / ahmet / baba")
    ap.add_argument("--case", required=True, choices=VALID_CASES,
                    help="Scene case: " + " / ".join(VALID_CASES))
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output root (default training/captured/)")
    ap.add_argument("--index", type=int, default=None,
                    help="Camera index; if omitted, probe 0..5 and prompt.")
    args = ap.parse_args()

    person = args.person.strip().replace(" ", "_")
    case = args.case
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir = out_dir / case
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.index is None:
        index = choose_camera(probe_cameras(), default=1)
    else:
        index = args.index

    cap = _open(index)
    if not cap.isOpened():
        print(f"[HATA] Kamera {index} acilamadi.")
        return 1
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[bilgi] Kamera {index} acildi ({src_w}x{src_h}, orijinal cozunurluk - kucultme yok).")
    print(f"[bilgi] Kayit klasoru: {out_dir}")
    print("[bilgi] SPACE = kaydet, Q = cikis.\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[uyari] Kare okunamadi; tekrar deneniyor...")
                if (cv2.waitKey(50) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                continue
            cv2.imshow(WINDOW, _overlay(frame, person, case, count))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:  # SPACE
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                fpath = out_dir / f"{case}_{person}_{ts}_{count:03d}.jpg"
                # Save the CLEAN full-resolution frame (no overlay burned in,
                # no extra/EXIF metadata - just the JPEG pixels).
                cv2.imwrite(str(fpath), frame)
                count += 1
                print(f"[kaydedildi] {fpath.name}  ({count})")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[ozet] {count} goruntu kaydedildi -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
