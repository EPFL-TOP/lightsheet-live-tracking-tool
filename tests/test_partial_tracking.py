"""Image every position, track the subset that has ROIs — and let more
join mid-run.

Recording twenty positions and drift-correcting three is a normal
experiment. Two things used to prevent it:

  * TrackingRunner built a tracker for EVERY position at startup, so a
    position without tracking_RoIs.json broke the run;
  * the Zeiss panel replaced the backend's positions_config and
    pos_names with only the trackable subset, which silently stopped
    the others being ACQUIRED at all — _acquisition_loop iterates
    pos_names.
"""
from __future__ import annotations

import inspect
import json
import os

import pytest

from tracking_tools.tracking_runner.TrackingRunner import TrackingRunner


ROI_KEYS = {
    "RoIs": [{"x": 10, "y": 10, "width": 4, "height": 4, "order": 1}],
    "detection": False,
    "tracking_mode": "single",
    "scaling_factor": 0,
    "blur_factor": 0,
    "grid_size": 5,
    "mask_kernel_size": 3,
}


def _runner(tmp_path, with_rois=("a",), positions=("a", "b", "c")):
    cfg = {}
    for name in positions:
        log_dir = tmp_path / name / "embryo_tracking"
        os.makedirs(log_dir, exist_ok=True)
        entry = {"xyz_um": (0.0, 0.0, 0.0), "log_dir": str(log_dir)}
        if name in with_rois:
            entry.update(ROI_KEYS)
            with open(log_dir / "tracking_RoIs.json", "w") as fh:
                json.dump(ROI_KEYS, fh)
        cfg[name] = entry

    r = TrackingRunner.__new__(TrackingRunner)
    r.positions_config = cfg
    r.trackers = {}
    r._untracked_reported = set()
    r._pending_reinit = set()
    r._tracks_history = {}
    r._reinit_epoch = {}
    r.logger = _QuietLogger()
    return r


class _QuietLogger:
    def __init__(self):
        self.messages = []

    def _log(self, msg):
        self.messages.append(str(msg))

    info = warning = error = debug = _log


# --- which positions are trackable ---------------------------------

def test_a_position_with_rois_is_trackable(tmp_path):
    r = _runner(tmp_path)
    assert r.is_trackable("a")


def test_a_position_without_rois_is_not_trackable_but_is_not_an_error(
        tmp_path):
    r = _runner(tmp_path)
    assert not r.is_trackable("b")
    assert not r.is_trackable("c")


def test_a_half_written_roi_config_is_not_trackable(tmp_path):
    """Missing keys must not become an AttributeError deep in the
    tracker."""
    r = _runner(tmp_path)
    del r.positions_config["a"]["grid_size"]
    assert not r.is_trackable("a")


def test_an_unknown_position_is_not_trackable(tmp_path):
    assert not _runner(tmp_path).is_trackable("nope")


# --- startup -------------------------------------------------------

def test_only_ready_positions_get_trackers(tmp_path, monkeypatch):
    r = _runner(tmp_path, with_rois=("a", "c"))
    built = []
    monkeypatch.setattr(
        type(r), "initialize_tracker",
        lambda self, name: built.append(name) or self.trackers.setdefault(name, object()),
    )
    r.initialize_ready_trackers()
    assert built == ["a", "c"]
    assert r.tracked_positions() == ["a", "c"]
    assert r.untracked_positions() == ["b"]


def test_startup_does_not_raise_when_nothing_has_rois(
        tmp_path, monkeypatch):
    """The panel attaches the tracker as soon as ANY position is
    ready, but a race must not crash the run."""
    r = _runner(tmp_path, with_rois=())
    monkeypatch.setattr(type(r), "initialize_tracker",
                        lambda self, name: None)
    r.initialize_ready_trackers()
    assert r.tracked_positions() == []


def test_startup_says_which_positions_are_only_imaged(
        tmp_path, monkeypatch):
    r = _runner(tmp_path, with_rois=("a",))
    monkeypatch.setattr(
        type(r), "initialize_tracker",
        lambda self, name: self.trackers.setdefault(name, object()),
    )
    r.initialize_ready_trackers()
    said = " ".join(r.logger.messages)
    assert "not tracking" in said
    assert "b" in said and "c" in said


# --- joining mid-run -----------------------------------------------

def test_a_position_joins_the_loop_when_its_rois_appear(
        tmp_path, monkeypatch):
    r = _runner(tmp_path, with_rois=("a",))
    monkeypatch.setattr(
        type(r), "initialize_tracker",
        lambda self, name: self.trackers.setdefault(name, object()),
    )
    r.initialize_ready_trackers()
    assert "b" not in r.trackers

    # The operator saves ROIs for b while the run is going.
    log_dir = r.positions_config["b"]["log_dir"]
    with open(os.path.join(log_dir, "tracking_RoIs.json"), "w") as fh:
        json.dump(ROI_KEYS, fh)

    r.microscope = None
    r.reinitialize_tracker("b")
    assert "b" in r.trackers, "the position never joined the loop"
    assert any("JOINED" in m for m in r.logger.messages)


def test_joining_is_not_counted_as_a_reinitialization(
        tmp_path, monkeypatch):
    """Epochs index tracks.pkl; a first tracker is epoch 0."""
    r = _runner(tmp_path, with_rois=())
    monkeypatch.setattr(
        type(r), "initialize_tracker",
        lambda self, name: self.trackers.setdefault(name, object()),
    )
    log_dir = r.positions_config["a"]["log_dir"]
    with open(os.path.join(log_dir, "tracking_RoIs.json"), "w") as fh:
        json.dump(ROI_KEYS, fh)
    r.microscope = None
    r.reinitialize_tracker("a")
    assert r._reinit_epoch.get("a", 0) == 0
    assert r._tracks_history.get("a", []) == []


def test_the_watcher_creates_folders_for_untracked_positions(tmp_path):
    """watchdog cannot watch a directory that does not exist, so a
    position whose folder was never created could never join."""
    src = inspect.getsource(TrackingRunner._start_roi_watcher)
    assert "makedirs" in src


def test_the_watcher_covers_every_position_not_just_tracked_ones(
        tmp_path):
    src = inspect.getsource(TrackingRunner._start_roi_watcher)
    assert "self.positions_config.items()" in src


# --- the panel must keep imaging everything ------------------------

def test_panel_does_not_narrow_the_backends_position_list():
    """_acquisition_loop iterates pos_names, so replacing it with the
    trackable subset stops the others being ACQUIRED."""
    pytest.importorskip("panel")
    from interactive_tools.mtb_setup import MTBSetupPanel
    src = inspect.getsource(MTBSetupPanel._build_tracker)
    assert "self._runner.pos_names = " not in src, (
        "the panel is narrowing what the backend acquires"
    )
    assert "self._runner.positions_config = " not in src, (
        "replacing the backend's config drops untracked positions"
    )
