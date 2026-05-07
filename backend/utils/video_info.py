"""Print metadata for a video file using OpenCV."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


def fourcc_to_str(code: int) -> str:
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def print_video_info(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video bulunamadı / Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı / Could not open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = fourcc_to_str(codec_int) if codec_int else "unknown"
        duration = total_frames / fps if fps else 0.0
    finally:
        cap.release()

    size_mb = path.stat().st_size / (1024 * 1024)

    print(f"File:        {path}")
    print(f"Size:        {size_mb:.2f} MB")
    print(f"Resolution:  {width} x {height}")
    print(f"FPS:         {fps:.3f}")
    print(f"Frames:      {total_frames}")
    print(f"Duration:    {duration:.2f} s")
    print(f"Codec:       {codec}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "videos/samples/sample_01.mp4"
    print_video_info(target)
