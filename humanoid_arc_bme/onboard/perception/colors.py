"""Colour naming for segmented objects: the median HSV of the masked pixels, mapped onto a small named palette.

Stdlib + numpy only. Pixels are RGB, uint8 (or float in the same 0..255 units). Hue is in degrees [0, 360); saturation
and value are in [0, 1]. Hue and saturation do not change under uniform brightness scaling, so only the value
thresholds decide how a dimmed colour is named: white dims into gray below roughly 0.9 scale, and dark orange (brown)
is the orange hue family below BROWN_VAL_MAX.
"""

from __future__ import annotations

import numpy as np

PALETTE = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "white", "gray", "black")

# Achromatic split: saturation below SAT_ACHROMATIC, or max-min spread below CHROMA_ACHROMATIC (absolute, so dark
# noisy pixels whose saturation is ill-conditioned still read as achromatic), is white, gray or black by value.
SAT_ACHROMATIC = 0.18
CHROMA_ACHROMATIC = 0.12
VAL_BLACK = 0.20
VAL_WHITE = 0.80
# Chromatic split: red/magenta hues with saturation below PINK_SAT_MAX are pink; the rest of red/magenta is red/purple.
PINK_SAT_MAX = 0.60
# Hue edges in degrees. Red wraps through 0, so the red family is [H_RED_LO, 360) U [0, H_RED_HI).
H_RED_HI = 15.0
H_BROWN_HI = 29.0
H_ORANGE_HI = 45.0
H_YELLOW_HI = 70.0
H_GREEN_HI = 170.0
H_BLUE_HI = 260.0
H_PURPLE_LO = 300.0
H_RED_LO = 345.0
BROWN_VAL_MAX = 0.60

_INDEX = {name: i for i, name in enumerate(PALETTE)}


def erode_mask(mask: np.ndarray) -> np.ndarray:
    """One-pixel 4-neighbour erosion of a bool (H, W) mask. Pixels on the image border erode."""
    p = np.pad(mask, 1, constant_values=False)
    return mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def _hsv(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(N, 3) float32 RGB in [0, 1] -> hue (degrees, [0, 360)), saturation, value, chroma (max - min); float32."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    d = mx - np.minimum(np.minimum(r, g), b)
    s = d / np.where(mx > 0, mx, np.float32(1.0))
    safe_d = np.where(d > 0, d, np.float32(1.0))
    hr = (g - b) / safe_d
    hg = 2.0 + (b - r) / safe_d
    hb = 4.0 + (r - g) / safe_d
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb)) * np.float32(60.0)
    return np.mod(h, np.float32(360.0)), s, mx, d


def _classify(h: np.ndarray, s: np.ndarray, v: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Palette index for each element of hue (degrees), saturation, value and chroma. Vectorised; first rule wins."""
    achro = (s < SAT_ACHROMATIC) | (c < CHROMA_ACHROMATIC)
    pink = ((h >= H_PURPLE_LO) | (h < H_RED_HI)) & (s < PINK_SAT_MAX)
    red = (h >= H_RED_LO) | (h < H_RED_HI)
    conds = [
        achro & (v < VAL_BLACK),
        achro & (v > VAL_WHITE),
        achro,
        pink,
        red,
        (h >= H_RED_HI) & (h < H_BROWN_HI) & (v < BROWN_VAL_MAX),
        (h >= H_RED_HI) & (h < H_ORANGE_HI),
        (h >= H_ORANGE_HI) & (h < H_YELLOW_HI),
        (h >= H_YELLOW_HI) & (h < H_GREEN_HI),
        (h >= H_GREEN_HI) & (h < H_BLUE_HI),
        h >= H_BLUE_HI,
    ]
    choices = [_INDEX[n] for n in ("black", "white", "gray", "pink", "red", "brown", "orange", "yellow", "green", "blue", "purple")]
    return np.select(conds, choices, default=_INDEX["gray"])


def name_color(rgb: np.ndarray, mask: np.ndarray | None = None, min_pixels: int = 20) -> tuple[str, float]:
    """Name the dominant colour of the masked pixels. Returns (name, confidence), confidence in [0, 1].

    `rgb` is (H, W, 3) RGB, uint8 (or float in 0..255 units; non-finite values give "unknown"). `mask` is a bool
    (H, W) selection; it is eroded by one pixel to drop edge bleed, unless that leaves fewer than `min_pixels`, in
    which case the un-eroded mask is used. The winner is the palette name of the MEDIAN hue, saturation and value;
    confidence is the fraction of selected pixels that classify individually to that same name.

    Too few pixels, a bad shape or dtype, or non-finite input returns ("unknown", 0.0). Never raises on sensor data.
    """
    if not isinstance(min_pixels, (int, np.integer)) or isinstance(min_pixels, bool) or min_pixels < 1:
        raise ValueError("min_pixels: must be an int >= 1")
    unknown = ("unknown", 0.0)
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        return unknown
    if rgb.dtype.kind not in "uf" or (rgb.dtype.kind == "f" and not np.isfinite(rgb).all()):
        return unknown
    h_img, w_img = rgb.shape[:2]
    if mask is None:
        selected = np.ones((h_img, w_img), dtype=bool)
    else:
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (h_img, w_img):
            return unknown
        selected = erode_mask(mask)
        if np.count_nonzero(selected) < min_pixels:
            selected = mask
    if np.count_nonzero(selected) < min_pixels:
        return unknown

    x = rgb[selected].astype(np.float32) * np.float32(1.0 / 255.0)
    h, s, v, c = _hsv(x)
    chroma = ~((s < SAT_ACHROMATIC) | (c < CHROMA_ACHROMATIC))
    if chroma.any():
        hw = h[chroma]
        med_h = float(np.median(np.where(hw > 180.0, hw - 360.0, hw))) % 360.0
    else:
        med_h = 0.0
    med = _classify(
        np.array([med_h], dtype=np.float32),
        np.array([np.median(s)], dtype=np.float32),
        np.array([np.median(v)], dtype=np.float32),
        np.array([np.median(c)], dtype=np.float32),
    )
    winner = int(med[0])
    agree = np.count_nonzero(_classify(h, s, v, c) == winner) / x.shape[0]
    return PALETTE[winner], float(agree)
