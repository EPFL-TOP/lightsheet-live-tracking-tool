"""Unit tests for MicroscopeInterface_MTB.

No pythonnet, no Zeiss software, no microscope, no camera: the MTB
session and MTBMotion are faked. The regression tests at the bottom
guard the three bugs an adversarial review found in the Micro-Manager
backend, since this backend solves the same problem and could
reintroduce them.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tracking_tools.microscope_interface import mtb_backend
from tracking_tools.microscope_interface.mtb_backend import (
    MicroscopeInterface_MTB,
)


class FakeAxis:
    def __init__(self, position=0.0, lo=-1e6, hi=1e6):
        self.position = position
        self.limits = (lo, hi)
        self.moves = []

    def move_to(self, target, **kw):
        self.moves.append(target)
        self.position = target
        return target

    def move_by(self, delta, **kw):
        return self.move_to(self.position + delta)


class FakeMotion:
    """Stands in for MTBMotion, recording every commanded target."""

    def __init__(self, z_role="piezo"):
        self.z_role = z_role
        self.x = FakeAxis()
        self.y = FakeAxis()
        self.z = FakeAxis(position=250.0, lo=0.0, hi=500.0)
        self.move_to_calls = []

    def get_xyz(self):
        return (self.x.position, self.y.position, self.z.position)

    def move_to(self, x=None, y=None, z=None):
        self.move_to_calls.append((x, y, z))
        if x is not None:
            self.x.move_to(x)
        if y is not None:
            self.y.move_to(y)
        if z is not None:
            self.z.move_to(z)
        return self.get_xyz()

    def move_by(self, dx=0.0, dy=0.0, dz=0.0):
        if dx:
            self.x.move_by(dx)
        if dy:
            self.y.move_by(dy)
        if dz:
            self.z.move_by(dz)
        return self.get_xyz()

    def describe(self):
        return "fake motion"


class FakeSession:
    def __init__(self, *a, **kw):
        self.disconnected = False

    def connect(self):
        return self

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Patch the MTB layer and hand back a ready-to-connect backend."""
    motion = FakeMotion()
    monkeypatch.setattr(mtb_backend, "MTBSession", FakeSession)
    monkeypatch.setattr(
        mtb_backend, "MTBMotion", lambda session, z_axis="piezo": motion
    )

    positions = {
        "scene_000": {"xyz_um": (100.0, 200.0, 250.0)},
        "scene_001": {"xyz_um": (500.0, 600.0, 260.0)},
    }

    def synthetic(x, y, z, pos_name):
        return np.full((8, 8), 42, dtype=np.uint16)

    def make(**overrides):
        params = {
            "synthetic_source": synthetic,
            "interval_s": 0.01,
            "settle_s": 0.0,
            "stop_after_tp": 2,
        }
        params.update(overrides)
        return MicroscopeInterface_MTB(
            positions, str(tmp_path), params
        )

    return make, motion, tmp_path


# ------------------------------------------------------ construction

def test_baselines_read_from_positions_config(patched):
    make, _, _ = patched
    iface = make()
    assert iface._baseline_um["scene_000"] == (100.0, 200.0, 250.0)
    assert iface._baseline_um["scene_001"] == (500.0, 600.0, 260.0)


def test_missing_xyz_um_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="xyz_um"):
        MicroscopeInterface_MTB(
            {"scene_000": {}}, str(tmp_path), {}
        )


def test_empty_positions_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        MicroscopeInterface_MTB({}, str(tmp_path), {})


def test_cum_drift_starts_at_zero(patched):
    make, _, _ = patched
    iface = make()
    for name in ("scene_000", "scene_001"):
        assert iface.get_cum_drift(name) == (0.0, 0.0, 0.0)


# -------------------------------------------------------- corrections

def test_relative_move_accumulates(patched):
    make, _, _ = patched
    iface = make()
    iface.relative_move("scene_000", 1.0, 2.0, 0.5)
    iface.relative_move("scene_000", 0.5, -1.0, 0.25)
    assert iface.get_cum_drift("scene_000") == pytest.approx(
        (1.5, 1.0, 0.75)
    )


def test_relative_move_is_per_position(patched):
    make, _, _ = patched
    iface = make()
    iface.relative_move("scene_000", 5.0, 0.0, 0.0)
    assert iface.get_cum_drift("scene_001") == (0.0, 0.0, 0.0)


def test_relative_move_clamps_xy(patched):
    make, _, _ = patched
    iface = make(max_xy_um=10.0)
    iface.relative_move("scene_000", 999.0, -999.0, 0.0)
    dx, dy, _ = iface.get_cum_drift("scene_000")
    assert dx == pytest.approx(10.0)
    assert dy == pytest.approx(-10.0)


def test_relative_move_clamps_z(patched):
    make, _, _ = patched
    iface = make(max_z_um=3.0)
    iface.relative_move("scene_000", 0.0, 0.0, 50.0)
    assert iface.get_cum_drift("scene_000")[2] == pytest.approx(3.0)


def test_relative_move_ignores_unknown_position(patched):
    make, _, _ = patched
    iface = make()
    iface.relative_move("nope", 1.0, 1.0, 1.0)  # must not raise


def test_target_is_baseline_plus_drift(patched):
    make, _, _ = patched
    iface = make()
    iface.relative_move("scene_000", 10.0, -20.0, 5.0)
    assert iface._target_for("scene_000") == pytest.approx(
        (110.0, 180.0, 255.0)
    )


def test_refresh_filename_updates_baseline(patched):
    make, _, _ = patched
    iface = make()
    iface.positions_config["scene_000"]["xyz_um"] = (1.0, 2.0, 3.0)
    iface.refresh_filename("scene_000")
    assert iface._baseline_um["scene_000"] == (1.0, 2.0, 3.0)


def test_refresh_filename_tolerates_unknown_position(patched):
    make, _, _ = patched
    iface = make()
    iface.refresh_filename("nope")  # must not raise


# ----------------------------------------------------- contract stubs

def test_pause_stubs_exist_and_return_none(patched):
    make, _, _ = patched
    iface = make()
    assert iface.pause_after_position() is None
    assert iface.no_pause_after_position() is None
    assert iface.continue_from_pause() is None


def test_wait_for_image_returns_none_on_timeout(patched):
    make, _, _ = patched
    iface = make()
    assert iface.wait_for_image(timeout_ms=10) is None


# ---------------------------------------------------------- full loop

def test_loop_visits_every_position_each_timepoint(patched):
    make, motion, tmp_path = patched
    iface = make(stop_after_tp=2)
    iface.connect()
    try:
        seen = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            item = iface.wait_for_image(timeout_ms=200)
            if item is None:
                break
            _, tp, pos = item
            seen.append((tp, pos))
    finally:
        iface.disconnect()

    assert (0, "scene_000") in seen
    assert (0, "scene_001") in seen
    assert (1, "scene_000") in seen
    assert (1, "scene_001") in seen


def test_frames_are_written_with_the_shared_naming_convention(patched):
    make, _, tmp_path = patched
    iface = make(stop_after_tp=1, channel="BF")
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if iface.wait_for_image(timeout_ms=200) is None:
                break
    finally:
        iface.disconnect()

    assert (tmp_path / "scene_000" / "t0000_BF.tif").exists()
    assert (tmp_path / "scene_001" / "t0000_BF.tif").exists()


def test_loop_applies_accumulated_drift_on_the_next_visit(patched):
    make, motion, _ = patched
    iface = make(stop_after_tp=3)

    # Correct scene_000 before starting so the very first visit already
    # reflects it.
    iface.relative_move("scene_000", 7.0, -3.0, 1.0)
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if iface.wait_for_image(timeout_ms=200) is None:
                break
    finally:
        iface.disconnect()

    # baseline (100, 200, 250) + drift (7, -3, 1)
    assert (107.0, 197.0, 251.0) in motion.move_to_calls


def test_disconnect_releases_the_mtb_session(patched):
    make, _, _ = patched
    iface = make()
    iface.connect()
    session = iface.session
    iface.disconnect()
    assert session.disconnected
    assert iface.session is None


# ====================================================================
# Regression guards — the three bugs found in the Micro-Manager backend
# ====================================================================

def test_REGRESSION_zero_drift_does_not_deadlock(patched):
    """Bug 1: the MM backend only enqueued the next timepoint when a
    correction arrived, so a perfectly stable sample stalled the run.
    This loop must advance with no corrections at all."""
    make, _, _ = patched
    iface = make(stop_after_tp=3, interval_s=0.01)
    iface.connect()
    try:
        tps = set()
        deadline = time.monotonic() + 10.0   # generous wall budget
        while time.monotonic() < deadline:
            item = iface.wait_for_image(timeout_ms=200)
            if item is None:
                break
            tps.add(item[1])
    finally:
        iface.disconnect()

    assert tps == {0, 1, 2}, (
        f"expected timepoints 0,1,2 with zero drift, got {tps}"
    )


def test_REGRESSION_cum_drift_is_updated_before_return(patched):
    """Bug 2: the MM backend deferred the cum_drift update, so a target
    computed immediately after relative_move() used a stale value."""
    make, _, _ = patched
    iface = make()
    iface.relative_move("scene_000", 4.0, 5.0, 6.0)
    # No sleep, no loop iteration — must already be visible.
    assert iface.get_cum_drift("scene_000") == pytest.approx(
        (4.0, 5.0, 6.0)
    )
    assert iface._target_for("scene_000") == pytest.approx(
        (104.0, 205.0, 256.0)
    )


def test_REGRESSION_baselines_never_read_from_the_stage(patched,
                                                       monkeypatch):
    """Bug 3: reading the stage for baselines bakes existing drift into
    the reference. Construction must not touch the hardware at all."""
    make, motion, _ = patched

    def explode(*a, **kw):
        raise AssertionError(
            "backend read the stage position while establishing "
            "baselines — they must come from positions_config"
        )

    monkeypatch.setattr(motion, "get_xyz", explode)
    iface = make()
    assert iface._baseline_um["scene_000"] == (100.0, 200.0, 250.0)


def test_REGRESSION_a_failing_position_does_not_kill_the_run(patched):
    """One bad position must be logged and skipped, not abort the loop
    and strand every other scene."""
    make, motion, _ = patched
    iface = make(stop_after_tp=2)

    calls = {"n": 0}
    original = motion.move_to

    def flaky(x=None, y=None, z=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("stage hiccup")
        return original(x=x, y=y, z=z)

    motion.move_to = flaky
    iface.connect()
    try:
        seen = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            item = iface.wait_for_image(timeout_ms=200)
            if item is None:
                break
            seen.append((item[1], item[2]))
    finally:
        iface.disconnect()

    # The first visit blew up; everything afterwards must still run.
    assert len(seen) >= 3, f"run did not recover, only got {seen}"
