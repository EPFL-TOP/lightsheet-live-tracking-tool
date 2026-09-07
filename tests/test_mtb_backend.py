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
    """Stands in for MTBAxis.

    Must expose arrival_tolerance: the real axis grew it when the
    servoing piezo turned out to dither well above its step width, and
    the z-stack checks read it. A double that lags the real class hides
    exactly the bugs the tests exist to catch.
    """

    def __init__(self, position=0.0, lo=-1e6, hi=1e6,
                 arrival_tolerance=0.01):
        self.position = position
        self.limits = (lo, hi)
        self.arrival_tolerance = arrival_tolerance
        self.step = arrival_tolerance
        self.typical_deviation = 0.0
        self.max_deviation = 0.0
        self.label = "fake-z"
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
    """Stands in for MTBSession, including the shared() singleton.

    MTB permits one Login per process, so the backend must go through
    shared() rather than constructing its own session.
    """

    _shared = None

    def __init__(self, *a, **kw):
        self.disconnected = False

    def connect(self):
        return self

    def disconnect(self):
        self.disconnected = True

    @classmethod
    def shared(cls, **kwargs):
        if cls._shared is None:
            cls._shared = cls().connect()
        return cls._shared

    @classmethod
    def reset(cls):
        cls._shared = None


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Patch the MTB layer and hand back a ready-to-connect backend."""
    motion = FakeMotion()
    FakeSession.reset()
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


def test_REGRESSION_wait_for_image_returns_a_triple_on_timeout(patched):
    """TrackingRunner.run_zeiss() unpacks the result BEFORE testing it:
        image, tp, pos = microscope.wait_for_image(...)
        if image is None: ...
    so a bare None raised
        cannot unpack non-iterable NoneType object
    Every other backend returns (None, None, None). Observed
    2026-09-04, ~50 s into a closed-loop run."""
    make, _, _ = patched
    iface = make()
    result = iface.wait_for_image(timeout_ms=10)
    assert isinstance(result, tuple) and len(result) == 3
    image, tp, pos = result          # must not raise
    assert image is None and tp is None and pos is None


# ---------------------------------------------------------- full loop

def test_loop_visits_every_position_each_timepoint(patched):
    make, motion, tmp_path = patched
    iface = make(stop_after_tp=2)
    iface.connect()
    try:
        seen = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, tp, pos = iface.wait_for_image(timeout_ms=200)
            if image is None:
                if iface.stop_requested:
                    break
                continue
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
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
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
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
                break
    finally:
        iface.disconnect()

    # baseline (100, 200, 250) + drift (7, -3, 1)
    assert (107.0, 197.0, 251.0) in motion.move_to_calls


def test_REGRESSION_disconnect_must_NOT_log_out_the_mtb_session(patched):
    """Backend teardown must leave the MTB session alive.

    MTB cannot Login twice in one process, so logging out here would
    strand the GUI and every subsequent run. The backend drops its
    reference; only MTBSession.close_shared() logs out, at process
    exit.
    """
    make, _, _ = patched
    iface = make()
    iface.connect()
    session = iface.session
    iface.disconnect()
    assert not session.disconnected, (
        "backend logged out the shared MTB session — no further Login "
        "can succeed in this process"
    )
    assert iface.session is None, "backend should drop its reference"


def test_backend_uses_the_shared_session_not_a_new_one(patched):
    """Two backends in one process must share a single Login."""
    make, _, _ = patched
    a, b = make(), make()
    a.connect()
    b.connect()
    try:
        assert a.session is b.session
    finally:
        a.disconnect()
        b.disconnect()


def test_injected_session_is_used_and_left_open(patched):
    make, _, _ = patched
    caller_session = FakeSession()
    iface = make(session=caller_session)
    iface.connect()
    assert iface.session is caller_session
    iface.disconnect()
    assert not caller_session.disconnected, (
        "an injected session belongs to the caller"
    )


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
            image, tp, _ = iface.wait_for_image(timeout_ms=200)
            if image is None:
                if iface.stop_requested:
                    break
                continue
            tps.add(tp)
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
            image, tp, pos = iface.wait_for_image(timeout_ms=200)
            if image is None:
                if iface.stop_requested:
                    break
                continue
            seen.append((tp, pos))
    finally:
        iface.disconnect()

    # The first visit blew up; everything afterwards must still run.
    assert len(seen) >= 3, f"run did not recover, only got {seen}"


# ====================================================================
# Shared exclusive resources
#
# PVCAM allows ONE open handle per camera. The setup GUI opens it when
# the operator presses Connect, so a backend that opened its own failed
# with
#   [PVCAM] ERR: pl_cam_open failed, pvErr:12,
#   'This user has already opened this camera (C0_CAM_ALREADY_OPEN)'
# Observed 2026-09-04. Same shape as the MTB single-Login rule.
# ====================================================================

class FakeCore:
    def __init__(self):
        self.camera_device = None
        self.exposure = None
        self.unloaded = False

    def setCameraDevice(self, label):
        self.camera_device = label

    def setExposure(self, ms):
        self.exposure = ms

    def unloadAllDevices(self):
        self.unloaded = True

    def snapImage(self):
        pass

    def getImage(self):
        return np.zeros((4, 4), dtype=np.uint16)

    def getImageWidth(self):
        return 4

    def getImageHeight(self):
        return 4

    def getImageBitDepth(self):
        return 16

    def setCircularBufferMemoryFootprint(self, mb):
        pass


def test_REGRESSION_injected_camera_is_reused_not_reopened(patched):
    make, _, _ = patched
    core = FakeCore()
    iface = make(synthetic_source=None, mmc=core,
                 camera_label="SetupCam", stop_after_tp=1)
    iface._open_camera()
    assert iface.mmc is core
    assert iface._camera_label == "SetupCam"
    assert core.camera_device == "SetupCam"
    assert not iface._owns_camera


def test_REGRESSION_borrowed_camera_is_not_unloaded(patched):
    """Unloading the GUI's camera would leave it holding a dead handle
    and block every later PVCAM open until a process restart."""
    make, _, _ = patched
    core = FakeCore()
    iface = make(synthetic_source=None, mmc=core,
                 camera_label="SetupCam", stop_after_tp=1)
    iface._open_camera()
    iface.disconnect()
    assert not core.unloaded, "backend unloaded a camera it borrowed"


def test_injected_camera_gets_our_exposure(patched):
    make, _, _ = patched
    core = FakeCore()
    iface = make(synthetic_source=None, mmc=core,
                 camera_label="SetupCam", exposure_ms=7.5)
    iface._open_camera()
    assert core.exposure == pytest.approx(7.5)


def test_connect_is_idempotent(patched):
    """TrackingRunner.run_zeiss() calls connect() too, and the GUI has
    already started acquisition — a second thread would mean two
    consumers driving the same stage."""
    make, _, _ = patched
    iface = make(stop_after_tp=50, interval_s=0.05)
    iface.connect()
    first = iface._thread
    try:
        iface.connect()
        assert iface._thread is first, (
            "connect() started a second acquisition thread"
        )
    finally:
        iface.disconnect()


# ====================================================================
# wait_for_image contract
#
# run_zeiss() unpacks before testing and uses stop_requested to tell a
# timeout from end-of-run:
#     image, tp, pos = microscope.wait_for_image(timeout_ms=...)
#     if image is None:
#         if getattr(microscope, 'stop_requested', False): break
#         continue
# ====================================================================

def test_stop_requested_false_while_running(patched):
    make, _, _ = patched
    iface = make(stop_after_tp=50, interval_s=0.05)
    iface.connect()
    try:
        assert not iface.stop_requested
    finally:
        iface.disconnect()


def test_stop_requested_true_after_stop(patched):
    make, _, _ = patched
    iface = make()
    iface.connect()
    iface.stop()
    assert iface.stop_requested


def test_REGRESSION_stop_requested_true_once_frames_are_exhausted(
        patched):
    """Without this the tracker would spin forever after the final
    timepoint, since a timeout is indistinguishable from end-of-run."""
    make, _, _ = patched
    iface = make(stop_after_tp=1, interval_s=0.01)
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
                break
        assert iface.stop_requested, (
            "backend never signalled end-of-run"
        )
    finally:
        iface.disconnect()


def test_frames_still_unpack_as_a_triple(patched):
    make, _, _ = patched
    iface = make(stop_after_tp=1, interval_s=0.01)
    iface.connect()
    try:
        got = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, tp, pos = iface.wait_for_image(timeout_ms=200)
            if image is not None:
                got = (image, tp, pos)
                break
        assert got is not None, "no frame arrived"
        image, tp, pos = got
        assert image.shape == (8, 8)
        assert tp == 0
        assert pos in ("scene_000", "scene_001")
    finally:
        iface.disconnect()


# ====================================================================
# Z-stacks
#
# The stack path had no coverage at all. Both failure modes below
# produce DUPLICATE planes rather than an error, so the file looks like
# a valid stack and nothing flags it — worse than a refusal.
# ====================================================================

def test_plane_offsets_are_symmetric_about_zero():
    """An earlier version computed -range/2 + i*step, which left the
    last plane short and the stack lopsided."""
    make = None
    iface = MicroscopeInterface_MTB(
        {"p": {"xyz_um": (0.0, 0.0, 0.0)}}, "/tmp",
        {"z_stack": {"range_um": 10.0, "step_um": 1.0}},
    )
    offsets = iface.plane_offsets()
    assert len(offsets) == 11
    assert offsets[0] == pytest.approx(-5.0)
    assert offsets[-1] == pytest.approx(5.0)
    assert sum(offsets) == pytest.approx(0.0)


def test_plane_offsets_stay_symmetric_for_awkward_steps():
    iface = MicroscopeInterface_MTB(
        {"p": {"xyz_um": (0.0, 0.0, 0.0)}}, "/tmp",
        {"z_stack": {"range_um": 10.0, "step_um": 3.0}},
    )
    offsets = iface.plane_offsets()
    assert sum(offsets) == pytest.approx(0.0)
    assert offsets == pytest.approx([-4.5, -1.5, 1.5, 4.5])


def test_plane_offsets_single_plane():
    iface = MicroscopeInterface_MTB(
        {"p": {"xyz_um": (0.0, 0.0, 0.0)}}, "/tmp",
        {"z_stack": {"range_um": 0.0, "step_um": 1.0}},
    )
    assert iface.plane_offsets() == [0.0]


def test_plane_offsets_rejects_a_nonpositive_step():
    iface = MicroscopeInterface_MTB(
        {"p": {"xyz_um": (0.0, 0.0, 0.0)}}, "/tmp",
        {"z_stack": {"range_um": 10.0, "step_um": 0.0}},
    )
    with pytest.raises(ValueError, match="must be positive"):
        iface.plane_offsets()


def test_zstack_returns_a_3d_array_with_one_plane_per_offset(patched):
    make, motion, tmp_path = patched
    iface = make(z_stack={"range_um": 4.0, "step_um": 1.0},
                 stop_after_tp=1)
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        got = None
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is not None:
                got = image
                break
        assert got is not None, "no stack arrived"
        assert got.ndim == 3, f"expected a stack, got {got.shape}"
        assert got.shape[0] == 5, f"expected 5 planes, got {got.shape}"
    finally:
        iface.disconnect()


def test_REGRESSION_zstack_works_with_a_synthetic_source(patched):
    """_acquire() used to short-circuit on synthetic_source before
    reaching the stack branch, so the whole stack path was untestable
    without a camera — and therefore untested."""
    make, _, _ = patched
    seen_z = []

    def synthetic(x, y, z, pos_name):
        seen_z.append(z)
        return np.full((4, 4), 7, dtype=np.uint16)

    iface = make(synthetic_source=synthetic, stop_after_tp=1,
                 z_stack={"range_um": 2.0, "step_um": 1.0})
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is not None:
                assert image.ndim == 3
                break
    finally:
        iface.disconnect()
    # Distinct Z per plane, not the same value repeated.
    assert len(set(seen_z)) > 1, (
        f"synthetic source saw a single Z {seen_z} — planes duplicated"
    )


def test_zstack_restores_z_to_the_centre(patched):
    make, motion, _ = patched
    iface = make(z_stack={"range_um": 4.0, "step_um": 1.0},
                 stop_after_tp=1)
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
                break
    finally:
        iface.disconnect()
    # Last commanded Z must be a plane centre, not an outer plane.
    assert motion.z.moves, "Z never moved"
    assert motion.z.moves[-1] in (250.0, 260.0), (
        f"Z left at {motion.z.moves[-1]} instead of a centre"
    )


def test_check_z_stack_warns_when_step_is_below_axis_resolution(patched):
    """A step finer than the axis can resolve makes move_to() a no-op,
    so consecutive planes image the same slice."""
    make, motion, _ = patched
    motion.z.arrival_tolerance = 0.5
    iface = make(z_stack={"range_um": 2.0, "step_um": 0.05})
    iface.motion = motion
    warnings = iface.check_z_stack()
    assert any("arrival tolerance" in w for w in warnings), warnings


def test_check_z_stack_warns_when_the_range_exceeds_the_limits(patched):
    """move_to() clamps, so outer planes pile up at the same position."""
    make, motion, _ = patched
    motion.z.arrival_tolerance = 0.01
    motion.z.limits = (0.0, 500.0)
    motion.z.position = 250.0
    iface = make(z_stack={"range_um": 900.0, "step_um": 10.0})
    iface.motion = motion
    warnings = iface.check_z_stack()
    assert any("clamped" in w for w in warnings), warnings


def test_check_z_stack_warns_when_it_cannot_fit_the_interval(patched):
    make, motion, _ = patched
    motion.z.arrival_tolerance = 0.01
    iface = make(z_stack={"range_um": 100.0, "step_um": 1.0},
                 settle_s=0.5, interval_s=5.0)
    iface.motion = motion
    warnings = iface.check_z_stack()
    assert any("interval" in w for w in warnings), warnings


def test_check_z_stack_silent_when_everything_is_sane(patched):
    make, motion, _ = patched
    motion.z.arrival_tolerance = 0.01
    motion.z.limits = (0.0, 500.0)
    motion.z.position = 250.0
    iface = make(z_stack={"range_um": 10.0, "step_um": 1.0},
                 settle_s=0.05, interval_s=600.0)
    iface.motion = motion
    assert iface.check_z_stack() == []


def test_check_z_stack_empty_without_a_stack(patched):
    make, motion, _ = patched
    iface = make()
    iface.motion = motion
    assert iface.check_z_stack() == []


# ====================================================================
# Multi-channel acquisition
#
# One camera and one filter cube, so channels are inherently
# sequential. Each carries its own reflector cube, exposure and
# whether it is a stack.
# ====================================================================

def test_default_is_a_single_channel(patched):
    make, _, _ = patched
    iface = make()
    assert len(iface.channels) == 1
    assert iface.channels[0]["name"] == "BF"


def test_every_channel_is_written_to_its_own_file(patched):
    make, _, tmp_path = patched
    iface = make(
        stop_after_tp=1,
        channels=[
            {"name": "BF", "reflector_pos": 0, "exposure_ms": 5.0,
             "z_stack": False},
            {"name": "GFP", "reflector_pos": 3, "exposure_ms": 200.0,
             "z_stack": False},
        ],
    )
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
                break
    finally:
        iface.disconnect()

    for name in ("BF", "GFP"):
        assert (tmp_path / "scene_000" / f"t0000_{name}.tif").exists(), (
            f"channel {name} was not saved"
        )


def test_apply_channel_is_called_for_each_channel(patched):
    """The GUI supplies apply_channel so the reflector and exposure are
    set the same way whether snapping by hand or running the loop."""
    make, _, _ = patched
    applied = []
    iface = make(
        stop_after_tp=1,
        apply_channel=lambda ch: applied.append(ch["name"]),
        channels=[
            {"name": "BF", "reflector_pos": 0, "exposure_ms": 5.0},
            {"name": "GFP", "reflector_pos": 3, "exposure_ms": 200.0},
        ],
    )
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
                break
    finally:
        iface.disconnect()

    assert applied[:2] == ["BF", "GFP"]
    # Two positions x one timepoint x two channels.
    assert applied.count("GFP") == 2


def test_only_the_first_channel_reaches_the_tracker(patched):
    """Drift is measured once per position; correlating a brightfield
    frame against a fluorescence one would be meaningless."""
    make, _, _ = patched
    iface = make(
        stop_after_tp=1,
        channels=[
            {"name": "BF", "reflector_pos": 0, "exposure_ms": 5.0},
            {"name": "GFP", "reflector_pos": 3, "exposure_ms": 200.0},
        ],
    )
    iface.connect()
    try:
        frames = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, tp, pos = iface.wait_for_image(timeout_ms=200)
            if image is None:
                if iface.stop_requested:
                    break
                continue
            frames.append((tp, pos))
    finally:
        iface.disconnect()

    # One frame per position per timepoint, not one per channel.
    assert len(frames) == 2, f"expected 2 tracker frames, got {frames}"


def test_per_channel_z_stack_is_honoured(patched):
    """Fluorescence usually wants a stack, brightfield usually does
    not — so the flag is per channel, not global."""
    make, _, tmp_path = patched
    iface = make(
        stop_after_tp=1,
        z_stack={"range_um": 2.0, "step_um": 1.0},
        channels=[
            {"name": "BF", "reflector_pos": 0, "exposure_ms": 5.0,
             "z_stack": False},
            {"name": "GFP", "reflector_pos": 3, "exposure_ms": 200.0,
             "z_stack": True},
        ],
    )
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
                break
    finally:
        iface.disconnect()

    import tifffile
    bf = tifffile.imread(tmp_path / "scene_000" / "t0000_BF.tif")
    gfp = tifffile.imread(tmp_path / "scene_000" / "t0000_GFP.tif")
    assert bf.ndim == 2, f"BF should be a single plane, got {bf.shape}"
    assert gfp.ndim == 3, f"GFP should be a stack, got {gfp.shape}"
