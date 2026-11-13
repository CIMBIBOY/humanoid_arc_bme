"""Colour naming: palette patches, brightness and noise robustness, mask restriction, fail-closed inputs."""

import subprocess
import sys

import numpy as np
import pytest

from humanoid_arc_bme.onboard.perception.colors import PALETTE, erode_mask, name_color

# One representative RGB value per palette name. Hue and saturation are what the classifier keys on.
PATCH = {
    "red": (220, 20, 20),
    "orange": (255, 140, 0),
    "yellow": (240, 220, 0),
    "green": (30, 160, 40),
    "blue": (30, 60, 220),
    "purple": (130, 40, 200),
    "pink": (255, 150, 190),
    "brown": (150, 75, 25),
    "white": (245, 245, 245),
    "gray": (128, 128, 128),
    "black": (20, 20, 20),
}
# White is brighter than the 0.8 value threshold only near full scale; below ~0.9 it is indistinguishable from gray.
MIN_SCALE = {name: (0.9 if name == "white" else 0.5) for name in PALETTE}


def patch(rgb, h=24, w=24):
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = np.asarray(rgb, dtype=np.uint8)
    return img


def test_palette_and_patch_table_agree():
    assert PALETTE == ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "white", "gray", "black")
    assert set(PATCH) == set(PALETTE)


@pytest.mark.parametrize("name", PALETTE)
def test_flat_patch_named(name):
    assert name_color(patch(PATCH[name])) == (name, 1.0)


@pytest.mark.parametrize("name", PALETTE)
@pytest.mark.parametrize("scale", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_brightness_scaling(name, scale):
    if scale < MIN_SCALE[name]:
        pytest.skip("white dims into gray below the value threshold (documented)")
    rgb = np.round(np.asarray(PATCH[name], dtype=np.float64) * scale).astype(np.uint8)
    assert name_color(patch(rgb))[0] == name


@pytest.mark.parametrize("name", PALETTE)
def test_small_gaussian_noise_seeded(name):
    rng = np.random.default_rng(1234)
    base = np.asarray(PATCH[name], dtype=np.float64)
    noisy = np.clip(base + rng.normal(0.0, 6.0, size=(32, 32, 3)), 0, 255).astype(np.uint8)
    label, conf = name_color(noisy)
    assert label == name
    assert 0.5 < conf <= 1.0


def test_mask_restricts_colour():
    img = np.empty((20, 40, 3), dtype=np.uint8)
    img[:, :20] = PATCH["red"]
    img[:, 20:] = PATCH["blue"]
    right = np.zeros((20, 40), dtype=bool)
    right[2:18, 22:38] = True
    left = np.zeros((20, 40), dtype=bool)
    left[2:18, 2:18] = True
    assert name_color(img, right)[0] == "blue"
    assert name_color(img, left)[0] == "red"


def test_mixed_region_confidence_is_winner_fraction():
    img = np.empty((10, 10, 3), dtype=np.uint8)
    img[:, :7] = PATCH["green"]
    img[:, 7:] = PATCH["purple"]
    mask = np.ones((10, 10), dtype=bool)
    name, conf = name_color(img, mask, min_pixels=5)
    assert name == "green"
    assert conf == pytest.approx(0.7, abs=0.06)  # eroded mask keeps interior columns; the boundary column is excluded


def test_erosion_falls_back_to_full_mask_when_too_small():
    img = patch(PATCH["blue"], 12, 12)
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:8, 3:8] = True  # 25 px; eroded interior is 9 px < 20
    assert erode_mask(mask).sum() == 9
    assert name_color(img, mask, min_pixels=20) == ("blue", 1.0)


def test_erode_mask_removes_border_ring():
    m = np.ones((5, 6), dtype=bool)
    e = erode_mask(m)
    assert e.sum() == 3 * 4
    assert not e[0].any() and not e[-1].any() and not e[:, 0].any() and not e[:, -1].any()


def test_too_few_pixels_unknown():
    img = patch(PATCH["red"], 4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1:4] = True
    assert name_color(img, mask, min_pixels=20) == ("unknown", 0.0)
    assert name_color(img, np.zeros((4, 4), dtype=bool)) == ("unknown", 0.0)


@pytest.mark.parametrize(
    "rgb, mask",
    [
        (None, None),
        (np.zeros((8, 8), dtype=np.uint8), None),
        (np.zeros((8, 8, 4), dtype=np.uint8), None),
        (np.zeros((0, 8, 3), dtype=np.uint8), None),
        ("not an image", None),
        (np.zeros((8, 8, 3), dtype=np.uint8), np.ones((4, 4), dtype=bool)),
        (np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8), dtype=np.uint8)),
        (np.full((8, 8, 3), np.nan), None),
    ],
)
def test_bad_input_is_unknown_not_raise(rgb, mask):
    assert name_color(rgb, mask) == ("unknown", 0.0)


def test_rejects_bad_min_pixels():
    with pytest.raises(ValueError):
        name_color(patch(PATCH["red"]), min_pixels=0)


def test_no_heavy_imports():
    code = (
        "import sys, humanoid_arc_bme.onboard.perception.colors as c; "
        "bad = [m for m in ('torch', 'isaaclab', 'rclpy', 'onnxruntime', 'cv2') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
