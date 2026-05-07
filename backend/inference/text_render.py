"""Turkish-safe outlined text rendering on BGR numpy arrays.

cv2.putText doesn't support Unicode glyphs (Turkish ş, İ, ç, ğ, ö, ü …).
We use Pillow on a tight ROI around each text bbox so the round-trip is cheap
even when called many times per frame.

Origin convention matches cv2.putText: `org` is the baseline-bottom-left of
the text. This lets existing call sites that previously used cv2.putText be
swapped over without recomputing positions.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CACHE: dict[int, Any] = {}

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",  # Segoe UI: full Turkish coverage on Windows
    "C:/Windows/Fonts/arial.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
]


def get_font(size_px: int):
    if size_px in _FONT_CACHE:
        return _FONT_CACHE[size_px]
    for path in _FONT_CANDIDATES:
        try:
            f = ImageFont.truetype(path, size_px)
            _FONT_CACHE[size_px] = f
            return f
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def put_text_outlined(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.6,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> None:
    """Draw outlined Unicode-safe text on a BGR numpy array (in place).

    `org` is cv2-style baseline-bottom-left. Text is drawn via Pillow on a
    tight ROI around the text's bbox, then written back to the image.
    """
    size_px = max(11, int(round(scale * 22)))
    font = get_font(size_px)
    stroke = max(2, thickness + 1)
    try:
        bx0, by0, bx1, by1 = font.getbbox(text)
    except (AttributeError, TypeError):
        bx0, by0, bx1, by1 = 0, 0, len(text) * size_px // 2, size_px
    x, y = int(org[0]), int(org[1])
    pil_y = y - size_px
    pad = stroke + 1
    rx0 = x + bx0 - pad
    ry0 = pil_y + by0 - pad
    rx1 = x + bx1 + pad
    ry1 = pil_y + by1 + pad
    H, W = img.shape[:2]
    cx0, cy0 = max(0, rx0), max(0, ry0)
    cx1, cy1 = min(W, rx1), min(H, ry1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = img[cy0:cy1, cx0:cx1]
    pil = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(
        (x - cx0, pil_y - cy0),
        text,
        font=font,
        fill=(fg[2], fg[1], fg[0]),
        stroke_width=stroke,
        stroke_fill=(bg[2], bg[1], bg[0]),
    )
    sub[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
