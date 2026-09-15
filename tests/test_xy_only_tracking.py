"""Acquire the full z-stack, correct in XY only.

The tracker measures XY and Z by DIFFERENT methods:

    XY   CoTracker, on a max projection the tracker makes itself
    Z    centre of mass across planes, on the thresholded volume

So "track in XY only" must be a FLAG, never a change to the pixels.
Flattening the stack before the tracker would destroy Z tracking
irreversibly while saving nothing on the XY path, which already runs
on a projection.

The setting belongs per position in tracking_RoIs.json, beside
`invert` and `threshold_type`: one embryo may sit still in Z while
another sinks.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from tracking_tools.tracker import BaseTracker as BT
from tracking_tools.tracking_runner import TrackingRunner as TR

_TRACKERS = ["SingleRoIBaseTracker_v2", "MultiRoIBaseTracker"]


# --- the tracker already projects for XY ----------------------------

@pytest.mark.parametrize("cls_name", _TRACKERS)
def test_xy_path_already_runs_on_a_max_projection(cls_name):
    """Which is why flattening the input beforehand saves nothing."""
    src = inspect.getsource(getattr(BT, cls_name))
    assert "np.max(self.current_frame, axis=0)" in src


@pytest.mark.parametrize("cls_name", _TRACKERS)
def test_z_is_measured_across_planes_by_a_separate_method(cls_name):
    """And is therefore destroyed by flattening the input."""
    src = inspect.getsource(
        getattr(BT, cls_name).compute_new_position_z
    )
    assert "D, H, W = self.current_frame.shape" in src
    assert "center_of_mass" in src


# --- the flag ------------------------------------------------------

@pytest.mark.parametrize("cls_name", _TRACKERS)
def test_tracker_accepts_and_stores_tracking_2d(cls_name):
    sig = inspect.signature(getattr(BT, cls_name).__init__)
    assert "tracking_2d" in sig.parameters
    assert sig.parameters["tracking_2d"].default is False
    assert "self.tracking_2d" in inspect.getsource(getattr(BT, cls_name))


@pytest.mark.parametrize("cls_name", _TRACKERS)
def test_z_computation_is_skipped_when_it_would_be_discarded(cls_name):
    """compute_new_position_z runs a 3-D Gaussian and a centre of mass
    over the cropped volume. On a deep stack that is not cheap, and
    with tracking_2d the answer is thrown away."""
    cls = getattr(BT, cls_name)
    obj = cls.__new__(cls)
    obj.tracking_2d = True
    obj.current_frame = np.zeros((51, 8, 8))

    def explode(roi):
        raise AssertionError(
            "computed a Z position that is then discarded"
        )

    obj.compute_new_position_z = explode
    # D//2 is exactly what compute_shift_px measures against, so the
    # resulting Z shift is zero.
    assert obj._position_z_or_reference(object()) == 25


@pytest.mark.parametrize("cls_name", _TRACKERS)
def test_z_is_computed_normally_when_tracking_in_3d(cls_name):
    cls = getattr(BT, cls_name)
    obj = cls.__new__(cls)
    obj.tracking_2d = False
    obj.current_frame = np.zeros((51, 8, 8))
    obj.compute_new_position_z = lambda roi: 7.5
    assert obj._position_z_or_reference(object()) == 7.5


# --- it reaches the tracker per position ---------------------------

def test_runner_takes_tracking_2d_from_the_roi_config():
    """Same route as threshold_type: the ROI selection dashboard
    writes it into tracking_RoIs.json, TrackingRunner merges that into
    positions_config and passes it down."""
    src = inspect.getsource(TR.TrackingRunner.initialize_tracker)
    assert 'positions_config[PosSetting].get(\n                "tracking_2d"' in src \
        or '"tracking_2d"' in src
    # Must be COPIED, not mutated: the run-wide dict is shared across
    # every position.
    assert "position_tracker_params = dict(" in src
    assert "position_tracker_params=position_tracker_params" in src


def test_runner_does_not_mutate_the_shared_params_dict():
    src = inspect.getsource(TR.TrackingRunner.initialize_tracker)
    assert "self.position_tracker_params[" not in src, (
        "writing into the run-wide dict would leak one position's "
        "setting into every other position"
    )


# --- the pre-existing threshold_type bug ---------------------------

@pytest.mark.parametrize("cls_name", _TRACKERS)
def test_REGRESSION_threshold_type_is_actually_stored(cls_name):
    """MultiRoIBaseTracker ACCEPTED threshold_type but never stored
    it, so _initialize_queries' getattr(..., "otsu") silently fell
    back to otsu forever. Multi-ROI brightfield therefore never got
    the texture mask chosen in the ROI dashboard — measured on
    t0000_BF.tif, otsu keeps 44% of its points on the embryo against
    texture's 95%."""
    src = inspect.getsource(getattr(BT, cls_name))
    assert "self.threshold_type = threshold_type" in src, (
        f"{cls_name} accepts threshold_type but never stores it"
    )


@pytest.mark.parametrize("cls_name", _TRACKERS)
def test_threshold_type_survives_to_the_grid_mask(cls_name):
    cls = getattr(BT, cls_name)
    obj = cls.__new__(cls)
    cls.__init__.__wrapped__ if hasattr(cls.__init__, "__wrapped__") else None
    obj.threshold_type = "texture"
    assert getattr(obj, "threshold_type", "otsu") == "texture"
