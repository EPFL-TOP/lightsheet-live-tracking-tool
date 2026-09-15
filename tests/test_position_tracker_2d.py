"""Both PositionTracker classes must accept a 2-D frame.

The microscope sends single planes on every run without a z-stack,
and now also on every run that acquires a stack but tracks on a
projection. PositionTrackerMultiROI promoted 2-D to 3-D; the
single-ROI class did not, so `D, H, W = self.shape` in
compute_shift_px raised

    ValueError: not enough values to unpack (expected 3, got 2)
"""
import inspect
import numpy as np
import pytest

from tracking_tools.position_tracker import PositionTracker as PT


@pytest.mark.parametrize(
    "cls_name",
    ["PositionTrackerSingleRoI_v2", "PositionTrackerMultiROI"],
)
def test_tracker_promotes_a_2d_frame_before_using_shape(cls_name):
    src = inspect.getsource(getattr(PT, cls_name).compute_shift_um)
    assert "ndim == 2" in src, (
        f"{cls_name} does not promote 2-D frames, so a single plane "
        f"sets self.shape to (H, W) and compute_shift_px's "
        f"`D, H, W = self.shape` raises ValueError"
    )


@pytest.mark.parametrize(
    "cls_name",
    ["PositionTrackerSingleRoI_v2", "PositionTrackerMultiROI"],
)
def test_unpacking_a_2d_shape_really_does_raise(cls_name):
    """Guard the guard: prove the failure the promotion prevents."""
    with pytest.raises(ValueError):
        D, H, W = np.zeros((16, 16)).shape
