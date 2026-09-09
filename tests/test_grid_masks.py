"""Grid-filtering masks for transmitted light.

The tracker keeps grid points where `filter_and_threshold` says the
image is interesting, then hands those to CoTracker. With plain Otsu
that means "brighter than the threshold" — correct for fluorescence,
but in brightfield the background IS the bright part, so CoTracker
tracks background instead of the embryo.

Measured on the reference frame t0000_BF.tif (1200x1200 uint16, embryo
centred, ~33% of the field) against an independent ground truth, for
the grid points a tracker would actually retain:

    otsu (as shipped)     44% purity,  49% recall
    inverted intensity   100% purity,  52% recall   <- loses half
    texture (local std)   95% purity,  91% recall

Inversion is pure but patchy: inverted intensity fragments into ~1900
blobs, so much of the embryo interior falls below threshold and its
points are dropped. That is the "missing features" symptom. Texture
separates embryo from background 2.7x (std 2089 vs 760) where
intensity manages only 1.2x (11685 vs 14262), and it does not care
about background polarity.
"""
from __future__ import annotations

import numpy as np
import pytest

from tracking_tools.utils.tracking_utils import (
    filter_and_threshold,
    invert_mask,
    texture_mask,
)


def synthetic_brightfield(size=256, radius=70, seed=0, uneven=True):
    """A dark, textured disc on a bright, smooth background.

    Mimics transmitted light: the object is only slightly darker than
    its surroundings but much more textured.

    With `uneven` (the default) the disc carries an intensity gradient
    so that parts of it are as bright as the background — which is what
    a real embryo looks like, and precisely why intensity thresholding
    produces a patchy mask while texture does not. A perfectly uniform
    disc makes inversion look flawless and hides the difference.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 14000.0, dtype=np.float32)
    img += rng.normal(0, 60, img.shape)            # smooth background

    yy, xx = np.mgrid[:size, :size]
    disc = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 < radius ** 2

    darkening = np.full((size, size), 2000.0, dtype=np.float32)
    if uneven:
        # Left side barely darker than background, right side clearly
        # darker: an intensity threshold can only capture part of it.
        darkening = 200.0 + 3000.0 * (xx / max(1, size - 1))
    img[disc] -= darkening[disc]
    img[disc] += rng.normal(0, 700, int(disc.sum()))  # more textured
    return np.clip(img, 0, 65535).astype(np.uint16), disc


def _scores(mask, truth):
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return 0.0, 0.0
    purity = float((mask & truth).sum()) / float(mask.sum())
    recall = float((mask & truth).sum()) / float(truth.sum())
    return purity, recall


# ------------------------------------------------------------ texture

def test_texture_finds_a_dark_textured_object():
    img, truth = synthetic_brightfield()
    purity, recall = _scores(texture_mask(img), truth)
    assert purity > 0.75, f"purity {purity:.2f} — too much background"
    assert recall > 0.75, f"recall {recall:.2f} — losing the object"


def test_texture_beats_inversion_on_recall():
    """The whole point: inversion is pure but drops half the points."""
    img, truth = synthetic_brightfield()
    _, tex_recall = _scores(texture_mask(img), truth)
    _, inv_recall = _scores(invert_mask(img), truth)
    assert tex_recall > inv_recall, (
        f"texture recall {tex_recall:.2f} should exceed inversion's "
        f"{inv_recall:.2f}"
    )


def test_texture_yields_one_region_not_speckle():
    """A fragmented mask is what scatters grid points."""
    from scipy import ndimage as ndi

    img, _ = synthetic_brightfield()
    _, n = ndi.label(texture_mask(img))
    assert n <= 3, f"{n} disconnected regions — mask is speckled"


def test_texture_ignores_background_polarity():
    """The same mode must work for a BRIGHT object on a dark field,
    i.e. fluorescence, so one setting covers both."""
    img, truth = synthetic_brightfield()
    flipped = (int(img.max()) + int(img.min()) - img.astype(np.int32))
    flipped = flipped.astype(np.uint16)
    purity, recall = _scores(texture_mask(flipped), truth)
    assert purity > 0.75 and recall > 0.75


def test_texture_survives_a_flat_frame():
    flat = np.full((64, 64), 12000, dtype=np.uint16)
    assert not np.asarray(texture_mask(flat)).any()


def test_texture_window_and_smoothing_are_tunable():
    img, truth = synthetic_brightfield()
    for window, smooth in ((9, 11), (21, 25), (41, 35)):
        purity, recall = _scores(
            texture_mask(img, window=window, smooth=smooth), truth
        )
        assert recall > 0.5, (
            f"window={window} smooth={smooth} recall {recall:.2f}"
        )


# ----------------------------------------------------------- inverted

def test_invert_mask_is_pure_even_if_incomplete():
    """Inversion's virtue is purity — what it selects really is the
    object. Its weakness is recall, covered above."""
    img, truth = synthetic_brightfield()
    purity, _ = _scores(invert_mask(img), truth)
    assert purity > 0.85, f"purity {purity:.2f}"


def test_invert_mask_survives_a_flat_frame():
    flat = np.full((32, 32), 500, dtype=np.uint16)
    assert not np.asarray(invert_mask(flat)).any()


# ------------------------------------------- wiring into the tracker

def test_plain_otsu_selects_the_background_in_brightfield():
    """The bug being fixed: with Otsu, most of what the grid keeps is
    NOT the object."""
    img, truth = synthetic_brightfield()
    purity, _ = _scores(
        filter_and_threshold(img, threshold_type="otsu"), truth
    )
    assert purity < 0.5, (
        f"otsu purity {purity:.2f} — expected it to favour background"
    )


@pytest.mark.parametrize("mode", ["texture", "invert"])
def test_transmitted_light_modes_select_the_object(mode):
    img, truth = synthetic_brightfield()
    purity, _ = _scores(
        filter_and_threshold(img, threshold_type=mode), truth
    )
    assert purity > 0.75, f"{mode} purity {purity:.2f}"


def test_filter_and_threshold_returns_an_integer_mask():
    """generate_grid multiplies it by the ROI box, so it must behave
    like the existing otsu path."""
    img, _ = synthetic_brightfield()
    for mode in ("otsu", "texture", "invert"):
        mask = filter_and_threshold(img, threshold_type=mode)
        assert mask.shape == img.shape
        assert set(np.unique(mask)).issubset({0, 1})


def test_unknown_threshold_type_still_behaves_like_otsu():
    """An older tracking_RoIs.json has no threshold_type at all."""
    img, _ = synthetic_brightfield()
    default = filter_and_threshold(img)
    explicit = filter_and_threshold(img, threshold_type="otsu")
    assert (default == explicit).all()


def test_grid_points_land_on_the_object_with_texture():
    """End to end: the mask is only useful if it changes which grid
    points are kept."""
    img, truth = synthetic_brightfield(size=256, radius=70)
    grid = np.zeros(img.shape, bool)
    grid[::8, ::8] = True

    def purity_of(mode):
        mask = np.asarray(
            filter_and_threshold(img, threshold_type=mode)
        ).astype(bool)
        kept = grid & mask
        if not kept.any():
            return 0.0
        return float((kept & truth).sum()) / float(kept.sum())

    assert purity_of("texture") > purity_of("otsu") + 0.3, (
        "texture should put far more grid points on the embryo"
    )
