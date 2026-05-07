"""Webcam smoke test: open default camera, show 50 frames in a window.

Press 'q' to quit early.
"""

from __future__ import annotations

import sys

import cv2


def main() -> int:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW = faster open on Windows
    if not cap.isOpened():
        print("Webcam açılamadı / Could not open webcam (cv2.VideoCapture(0)).")
        print("Webcam başka bir uygulama tarafından kullanılıyor olabilir.")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam açıldı / opened: {width}x{height} @ {fps:.2f} FPS")
    print("'q' tuşuna basarak çıkabilirsiniz / Press 'q' to quit.")

    window = "İşGözcü Webcam Test"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    shown = 0
    target_frames = 50
    try:
        while shown < target_frames:
            ok, frame = cap.read()
            if not ok:
                print(f"[uyarı] Kare okunamadı (frame {shown}). Çıkılıyor.")
                break

            cv2.putText(
                frame,
                f"frame {shown + 1}/{target_frames} - press q to quit",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, frame)
            shown += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                print("Kullanıcı çıkışı / user quit.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Toplam gösterilen kare / frames shown: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
