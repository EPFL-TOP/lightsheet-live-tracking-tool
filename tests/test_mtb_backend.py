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


# ====================================================================
# Brightfield contrast inversion
#
# The ROI tracker thresholds for a BRIGHT object on a DARK background.
# True of fluorescence; exactly reversed in transmitted light, where
# the background is the brightest thing in the frame and survives
# every threshold. Reported 2026-09-08 after brightfield tracking
# worked but the masking did not.
# ====================================================================

def test_invert_makes_a_dark_object_bright():
    from tracking_tools.microscope_interface.mtb_backend import (
        invert_for_tracking,
    )
    img = np.full((8, 8), 14000, dtype=np.uint16)
    img[3:5, 3:5] = 11000              # dark embryo, bright field
    out = invert_for_tracking(img)
    assert out[4, 4] > out[0, 0], "object should now be the bright part"


def test_invert_uses_the_data_range_not_the_dtype_range():
    """Inverting against 65535 would push a frame spanning
    11000..14800 into the bottom of the scale and destroy contrast."""
    from tracking_tools.microscope_interface.mtb_backend import (
        invert_for_tracking,
    )
    img = np.array([[11000, 14800]], dtype=np.uint16)
    out = invert_for_tracking(img)
    assert set(out.ravel().tolist()) == {11000, 14800}
    assert out[0, 0] == 14800 and out[0, 1] == 11000


def test_invert_preserves_shape_and_dtype():
    from tracking_tools.microscope_interface.mtb_backend import (
        invert_for_tracking,
    )
    img = np.random.default_rng(0).integers(
        11000, 14800, size=(6, 7), dtype=np.uint16
    )
    out = invert_for_tracking(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_invert_survives_a_flat_frame():
    from tracking_tools.microscope_interface.mtb_backend import (
        invert_for_tracking,
    )
    flat = np.full((4, 4), 99, dtype=np.uint16)
    out = invert_for_tracking(flat)
    assert (out == 99).all()


def test_invert_is_its_own_inverse():
    from tracking_tools.microscope_interface.mtb_backend import (
        invert_for_tracking,
    )
    img = np.random.default_rng(1).integers(
        100, 900, size=(5, 5), dtype=np.uint16
    )
    assert (invert_for_tracking(invert_for_tracking(img)) == img).all()


def test_REGRESSION_tracker_gets_inverted_but_disk_keeps_raw(patched):
    """The saved file must stay raw — inverting on disk would confuse
    the ROI dashboard and anyone reviewing the data later."""
    make, _, tmp_path = patched
    seen = []

    def synthetic(x, y, z, pos_name):
        img = np.full((8, 8), 14000, dtype=np.uint16)
        img[3:5, 3:5] = 11000
        return img

    iface = make(
        synthetic_source=synthetic, stop_after_tp=1,
        channels=[{"name": "BF", "reflector_pos": 0, "led_nm": 0,
                   "exposure_ms": 5.0, "invert": True}],
    )
    iface.connect()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None:
                if iface.stop_requested:
                    break
                continue
            seen.append(image)
    finally:
        iface.disconnect()

    assert seen, "no frame reached the tracker"
    tracked = seen[0]
    assert tracked[4, 4] > tracked[0, 0], (
        "tracker did not receive an inverted frame"
    )

    import tifffile
    on_disk = tifffile.imread(
        tmp_path / "scene_000" / "t0000_BF.tif"
    )
    assert on_disk[4, 4] < on_disk[0, 0], (
        "the file on disk was inverted; it must stay raw"
    )


def test_no_inversion_when_the_channel_does_not_ask(patched):
    make, _, _ = patched

    def synthetic(x, y, z, pos_name):
        img = np.full((8, 8), 14000, dtype=np.uint16)
        img[3:5, 3:5] = 11000
        return img

    iface = make(
        synthetic_source=synthetic, stop_after_tp=1,
        channels=[{"name": "GFP", "reflector_pos": 3, "led_nm": 475,
                   "exposure_ms": 200.0, "invert": False}],
    )
    iface.connect()
    try:
        got = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is not None:
                got = image
                break
    finally:
        iface.disconnect()
    assert got is not None
    assert got[4, 4] < got[0, 0], "fluorescence must not be inverted"


# ====================================================================
# Preprocessing is authored in the ROI dashboard
#
# The transformation belongs where the ROIs are configured: that is
# where the operator sees the preview and decides what to track. It
# travels in tracking_RoIs.json, which TrackingRunner merges into
# positions_config before calling refresh_filename().
# ====================================================================

def test_invert_defaults_to_the_channel_flag(patched):
    """Before any ROIs exist the channel's flag is all we have."""
    make, _, _ = patched
    iface = make()
    assert iface.wants_invert("scene_000", {"invert": True}) is True
    assert iface.wants_invert("scene_000", {"invert": False}) is False


def test_roi_config_overrides_the_channel_flag(patched):
    """The dashboard wins once the operator has chosen."""
    make, _, _ = patched
    iface = make()
    iface.positions_config["scene_000"]["invert"] = False
    iface.refresh_filename("scene_000")
    # Channel says invert, ROI config says do not — ROI config wins.
    assert iface.wants_invert("scene_000", {"invert": True}) is False

    iface.positions_config["scene_000"]["invert"] = True
    iface.refresh_filename("scene_000")
    assert iface.wants_invert("scene_000", {"invert": False}) is True


def test_invert_is_read_from_positions_config_at_construction(tmp_path):
    iface = MicroscopeInterface_MTB(
        {"p": {"xyz_um": (0.0, 0.0, 0.0), "invert": True}},
        str(tmp_path), {},
    )
    assert iface.wants_invert("p") is True


def test_invert_is_per_position(patched):
    """Different positions can need different preprocessing."""
    make, _, _ = patched
    iface = make()
    iface.positions_config["scene_000"]["invert"] = True
    iface.positions_config["scene_001"]["invert"] = False
    iface.refresh_filename("scene_000")
    iface.refresh_filename("scene_001")
    assert iface.wants_invert("scene_000") is True
    assert iface.wants_invert("scene_001") is False


def test_refresh_filename_without_invert_key_leaves_the_default(patched):
    """An ROI file that predates the invert flag must not silently
    switch preprocessing off."""
    make, _, _ = patched
    iface = make()
    iface.positions_config["scene_000"].pop("invert", None)
    iface.refresh_filename("scene_000")
    assert iface.wants_invert("scene_000", {"invert": True}) is True


# ====================================================================
# Illumination is gated to the exposure
#
# Leaving the light on across the settle, the stage moves and the
# inter-timepoint interval illuminates a live sample continuously. For
# a 60 s interval with a 200 ms exposure that is roughly 300x the
# necessary dose — it bleaches and cooks the embryo. Reported
# 2026-09-09.
# ====================================================================

class LightLog:
    """Records every illuminate() call, and snap timing against it."""

    def __init__(self):
        self.events = []          # ('on'|'off', channel name)
        self.snaps_while_lit = 0
        self.snaps_while_dark = 0
        self.lit = False

    def illuminate(self, channel, on):
        self.events.append(("on" if on else "off",
                            channel.get("name")))
        self.lit = bool(on)

    def note_snap(self):
        if self.lit:
            self.snaps_while_lit += 1
        else:
            self.snaps_while_dark += 1


class LoggingCore(FakeCore):
    def __init__(self, light):
        super().__init__()
        self._light = light

    def snapImage(self):
        self._light.note_snap()


def _run(iface, seconds=5.0):
    iface.connect()
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            image, _, _ = iface.wait_for_image(timeout_ms=200)
            if image is None and iface.stop_requested:
                break
    finally:
        iface.disconnect()


def test_REGRESSION_light_is_on_only_during_the_exposure(patched):
    make, _, _ = patched
    light = LightLog()
    core = LoggingCore(light)
    iface = make(
        synthetic_source=None, mmc=core, camera_label="Cam",
        stop_after_tp=2, illuminate=light.illuminate,
        channels=[{"name": "GFP", "reflector_pos": 3, "led_nm": 475,
                   "led_percent": 20.0, "exposure_ms": 200.0}],
    )
    _run(iface)

    assert light.snaps_while_lit > 0, "no frame was taken with light on"
    assert light.snaps_while_dark == 0, (
        f"{light.snaps_while_dark} frame(s) taken in the dark"
    )
    # And the light must not be left on at the end.
    assert light.events[-1][0] == "off", (
        f"run ended with the light {light.events[-1][0]}"
    )


def test_light_is_switched_off_even_if_the_snap_fails(patched):
    """A raised snap must not leave the sample under continuous
    excitation until somebody notices."""
    make, _, _ = patched
    light = LightLog()

    class Exploding(LoggingCore):
        def snapImage(self):
            raise RuntimeError("camera hiccup")

    iface = make(
        synthetic_source=None, mmc=Exploding(light),
        camera_label="Cam", stop_after_tp=1,
        illuminate=light.illuminate,
        channels=[{"name": "GFP", "led_nm": 475, "exposure_ms": 200.0}],
    )
    _run(iface, seconds=3.0)

    ons = [e for e in light.events if e[0] == "on"]
    offs = [e for e in light.events if e[0] == "off"]
    assert ons, "light never came on"
    assert len(offs) >= len(ons), (
        "light was switched on more often than off — it was left lit"
    )


def test_each_channel_switches_its_own_light(patched):
    make, _, _ = patched
    light = LightLog()
    iface = make(
        synthetic_source=None, mmc=LoggingCore(light),
        camera_label="Cam", stop_after_tp=1,
        illuminate=light.illuminate,
        channels=[
            {"name": "BF", "led_nm": 0, "exposure_ms": 5.0},
            {"name": "GFP", "led_nm": 475, "exposure_ms": 200.0},
        ],
    )
    _run(iface)
    lit = [name for state, name in light.events if state == "on"]
    assert "BF" in lit and "GFP" in lit


def test_zstack_switches_light_per_plane_by_default(patched):
    """Per-plane switching is the lower-dose option: holding the light
    through the settle between planes multiplies the exposure."""
    make, _, _ = patched
    light = LightLog()
    iface = make(
        synthetic_source=None, mmc=LoggingCore(light),
        camera_label="Cam", stop_after_tp=1,
        illuminate=light.illuminate,
        z_stack={"range_um": 4.0, "step_um": 1.0},
        gate_light_per_plane=True,
        channels=[{"name": "GFP", "led_nm": 475, "exposure_ms": 200.0,
                   "z_stack": True}],
    )
    _run(iface)
    ons = len([e for e in light.events if e[0] == "on"])
    # 5 planes x 2 positions.
    assert ons >= 10, f"expected per-plane switching, saw {ons} on-events"


def test_zstack_can_hold_the_light_for_the_whole_stack(patched):
    """Fewer mechanical shutter cycles, at the cost of a larger dose."""
    make, _, _ = patched
    light = LightLog()
    iface = make(
        synthetic_source=None, mmc=LoggingCore(light),
        camera_label="Cam", stop_after_tp=1,
        illuminate=light.illuminate,
        z_stack={"range_um": 4.0, "step_um": 1.0},
        gate_light_per_plane=False,
        channels=[{"name": "GFP", "led_nm": 475, "exposure_ms": 200.0,
                   "z_stack": True}],
    )
    _run(iface)
    ons = len([e for e in light.events if e[0] == "on"])
    # One per stack per position, not one per plane.
    assert ons <= 4, f"expected per-stack switching, saw {ons} on-events"


def test_no_illuminate_callback_still_acquires(patched):
    """The backend must work standalone, without a GUI supplying the
    light callback."""
    make, _, tmp_path = patched
    iface = make(stop_after_tp=1)      # synthetic source, no illuminate
    _run(iface)
    assert (tmp_path / "scene_000" / "t0000_BF.tif").exists()


# ====================================================================
# DEFINITE FOCUS 2
#
# DF2 holds a learned plane optically. The two things that can go
# wrong are (a) one global reference being used for positions that sit
# at different heights, and (b) DF2 and the image-based tracker both
# aiming the Z axis, which makes them fight every timepoint.
# ====================================================================

class FakeDF2:
    """Stands in for MTBDefiniteFocus.

    Models the thing that matters: stabilizing MOVES THE FOCUS. A
    double that only records calls would let the backend image at the
    commanded Z and still pass.
    """

    def __init__(self, axis, planes=None, reject=()):
        self.axis = axis
        #: blob -> the Z that blob represents
        self.planes = planes or {}
        self.reject = set(reject)
        self.restored = []
        self.stabilized = 0
        self._plane = None

    def restore(self, data):
        if bytes(data) in self.reject:
            raise RuntimeError("stale reference")
        self.restored.append(bytes(data))
        self._plane = self.planes.get(bytes(data))
        return True

    def stabilize_now(self):
        self.stabilized += 1
        if self._plane is not None:
            self.axis.move_to(self._plane)
        return True

    def learn(self):
        return b"learned"


def _df2_backend(patched, **overrides):
    make, motion, tmp_path = patched
    df2 = FakeDF2(motion.z)
    overrides.setdefault("use_definite_focus", True)
    overrides.setdefault("df2", df2)
    iface = make(**overrides)
    iface.session = object()
    iface.motion = motion
    iface._connect_df2()
    return iface, df2, motion


def test_df2_reference_blobs_survive_the_json_round_trip():
    ref = bytes(range(256))
    encoded = mtb_backend.encode_df2_ref(ref)
    assert isinstance(encoded, str), "must be JSON-storable"
    assert mtb_backend.decode_df2_ref(encoded) == ref


def test_df2_decode_tolerates_rubbish():
    """A corrupt reference costs focus stabilisation, not the run."""
    assert mtb_backend.decode_df2_ref(None) is None
    assert mtb_backend.decode_df2_ref("") is None
    assert mtb_backend.decode_df2_ref("not base64!!!") is None
    assert mtb_backend.decode_df2_ref(12345) is None
    assert mtb_backend.decode_df2_ref([1, 2]) == b"\x01\x02"


def test_df2_references_are_read_from_positions_config(tmp_path):
    ref = mtb_backend.encode_df2_ref(b"\xaa\xbb")
    iface = MicroscopeInterface_MTB(
        {"scene_000": {"xyz_um": (1.0, 2.0, 3.0), "df2_ref": ref}},
        str(tmp_path), {},
    )
    assert iface._df2_refs["scene_000"] == b"\xaa\xbb"


def test_df2_is_off_by_default(patched):
    make, _, _ = patched
    iface = make()
    assert iface.use_definite_focus is False
    assert iface._stabilize("scene_000", 250.0) == 250.0


def test_df2_missing_hardware_downgrades_instead_of_failing(patched):
    """Losing focus stabilisation is a far smaller loss than losing the
    run, so an absent DF2 must not be fatal."""
    make, motion, _ = patched
    iface = make(use_definite_focus=True)

    class NoDF2:
        def definite_focus(self):
            raise RuntimeError("MTBFocusStabilizer2 is absent")

    iface.session = NoDF2()
    iface.motion = motion
    iface._connect_df2()
    assert iface.use_definite_focus is False
    assert iface._stabilize("scene_000", 250.0) == 250.0


def test_df2_restores_the_reference_belonging_to_THIS_position(patched):
    """Positions sit at different heights; one global reference would
    focus every position on the first one's plane."""
    make, motion, tmp_path = patched
    df2 = FakeDF2(motion.z, planes={b"A": 251.5, b"B": 262.5})
    iface = make(use_definite_focus=True, df2=df2)
    iface.session = object()
    iface.motion = motion
    iface._df2_refs = {"scene_000": b"A", "scene_001": b"B"}
    iface._connect_df2()

    assert iface._stabilize("scene_000", 250.0) == pytest.approx(251.5)
    assert iface._stabilize("scene_001", 260.0) == pytest.approx(262.5)
    assert df2.restored == [b"A", b"B"]


def test_df2_returns_the_z_it_actually_reached_not_the_commanded_one(
        patched):
    """The stack centres on the returned Z. DF2 moves the focus itself,
    so assuming the commanded Z would centre the stack in the wrong
    place by exactly the drift DF2 just removed."""
    iface, df2, motion = _df2_backend(patched)
    iface._df2_refs["scene_000"] = b"A"
    df2.planes[b"A"] = 253.75
    assert iface._stabilize("scene_000", 250.0) == pytest.approx(253.75)


def test_df2_falls_back_to_the_commanded_z_when_stabilizing_fails(
        patched):
    iface, df2, motion = _df2_backend(patched)

    def boom():
        raise RuntimeError("lost the surface")

    df2.stabilize_now = boom
    assert iface._stabilize("scene_000", 250.0) == 250.0


def test_df2_reports_a_stale_reference_once_then_carries_on(patched):
    """A blob DF2 rejects must not log on every visit of a long run."""
    make, motion, _ = patched
    df2 = FakeDF2(motion.z, reject={b"STALE"})
    iface = make(use_definite_focus=True, df2=df2)
    iface.session = object()
    iface.motion = motion
    iface._df2_refs = {"scene_000": b"STALE"}
    iface._connect_df2()

    for _ in range(3):
        iface._stabilize("scene_000", 250.0)
    assert "scene_000" in iface._df2_bad
    # Still stabilized every time, just on DF2's current plane.
    assert df2.stabilized == 3


def test_df2_owning_z_drops_tracked_z_corrections(patched):
    """Two controllers aiming the same axis at different measurements
    fight every timepoint."""
    iface, _, _ = _df2_backend(patched, df2_owns_z=True)
    iface.relative_move("scene_000", 1.0, 2.0, 3.0)
    dx, dy, dz = iface.get_cum_drift("scene_000")
    assert (dx, dy) == (1.0, 2.0)
    assert dz == 0.0, "DF2 and the tracker are both driving Z"


def test_tracked_z_is_kept_when_df2_does_not_own_z(patched):
    iface, _, _ = _df2_backend(patched, df2_owns_z=False)
    iface.relative_move("scene_000", 1.0, 2.0, 3.0)
    assert iface.get_cum_drift("scene_000") == (1.0, 2.0, 3.0)


def test_tracked_z_is_kept_when_df2_is_off(patched):
    make, _, _ = patched
    iface = make()
    iface.relative_move("scene_000", 1.0, 2.0, 3.0)
    assert iface.get_cum_drift("scene_000") == (1.0, 2.0, 3.0)


def test_specimen_offset_is_applied_on_top_of_the_stabilized_plane(
        patched):
    """With df2_owns_z off, DF2 removes the hardware drift and the
    tracked dz is specimen motion, so it belongs above the plane."""
    iface, df2, motion = _df2_backend(patched, df2_owns_z=False)
    iface._df2_refs["scene_000"] = b"A"
    df2.planes[b"A"] = 250.0
    iface.relative_move("scene_000", 0.0, 0.0, 4.0)

    assert iface._stabilize("scene_000", 250.0) == pytest.approx(254.0)


def test_set_df2_reference_writes_back_into_positions_config(tmp_path):
    """So a plane learned mid-run is saved with the rest of the
    position instead of dying with the process."""
    cfg = {"scene_000": {"xyz_um": (1.0, 2.0, 3.0)}}
    iface = MicroscopeInterface_MTB(cfg, str(tmp_path), {})
    iface.set_df2_reference("scene_000", b"\x07\x08")
    assert iface._df2_refs["scene_000"] == b"\x07\x08"
    assert cfg["scene_000"]["df2_ref"] == \
        mtb_backend.encode_df2_ref(b"\x07\x08")

    iface.set_df2_reference("scene_000", None)
    assert "scene_000" not in iface._df2_refs
    assert cfg["scene_000"]["df2_ref"] is None


def test_refresh_filename_picks_up_a_reference_learned_later(tmp_path):
    cfg = {"scene_000": {"xyz_um": (1.0, 2.0, 3.0)}}
    iface = MicroscopeInterface_MTB(cfg, str(tmp_path), {})
    assert "scene_000" not in iface._df2_refs

    cfg["scene_000"]["df2_ref"] = mtb_backend.encode_df2_ref(b"\x01")
    iface.refresh_filename("scene_000")
    assert iface._df2_refs["scene_000"] == b"\x01"


def test_visit_stabilizes_before_any_exposure(patched):
    """Every channel and every plane must reference the same plane, so
    DF2 runs once, after the move and before the first frame."""
    iface, df2, motion = _df2_backend(patched)
    order = []
    df2.stabilize_now = lambda: order.append("stabilize") or True
    iface.synthetic_source = (
        lambda x, y, z, n: order.append("grab")
        or np.zeros((4, 4), dtype=np.uint16)
    )
    iface._visit("scene_000", 0)
    assert order[0] == "stabilize"
    assert order.count("stabilize") == 1
    assert "grab" in order


def test_visit_centres_the_stack_on_the_stabilized_plane(patched):
    """The z-stack belongs to the position, and DF2 refines where that
    position's plane actually is."""
    iface, df2, motion = _df2_backend(
        patched, z_stack={"n_planes": 3, "step_um": 2.0},
        channels=[{"name": "BF", "reflector_pos": 0,
                   "exposure_ms": 5.0, "z_stack": True}],
    )
    iface._df2_refs["scene_000"] = b"A"
    df2.planes[b"A"] = 253.0

    grabbed = []
    iface.synthetic_source = (
        lambda x, y, z, n: grabbed.append(z)
        or np.zeros((4, 4), dtype=np.uint16)
    )
    iface._visit("scene_000", 0)
    # Baseline Z is 250.0; DF2 moved the plane to 253.0.
    assert grabbed == pytest.approx([251.0, 253.0, 255.0]), (
        "the stack centred on the commanded Z, not the stabilized one"
    )


# ====================================================================
# SAVING A Z-STACK
#
# Handing tifffile a bare (planes, H, W) array is NOT enough. With
# exactly 3 planes it guesses RGB and writes ONE page carrying three
# colour components, so a 3-plane stack silently became a colour
# snapshot. At other plane counts the axes came out 'Q' (unknown) with
# no z-step, so nothing downstream could tell a stack from a series.
# ====================================================================

def _read_tif(path):
    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffFile(path) as tf:
        return {
            "pages": len(tf.pages),
            "shape": tf.series[0].shape,
            "axes": tf.series[0].axes,
            "photometric": tf.pages[0].photometric.name,
            "imagej": tf.is_imagej,
            "meta": tf.imagej_metadata or {},
            "data": tf.asarray(),
        }


def test_REGRESSION_a_three_plane_stack_is_not_written_as_rgb(patched):
    """The exact case tifffile mis-guesses: 3 planes look like RGB."""
    make, _, tmp_path = patched
    iface = make(z_stack={"n_planes": 3, "step_um": 2.0})
    stack = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    path = iface._save(stack, "scene_000", 0, channel="BF")

    info = _read_tif(path)
    assert info["pages"] == 3, (
        f"3 planes were written as {info['pages']} page(s) — tifffile "
        f"collapsed the stack into an RGB image"
    )
    assert info["photometric"] == "MINISBLACK"
    assert np.array_equal(info["data"], stack)


def test_a_saved_stack_is_labelled_as_z_not_unknown(patched):
    make, _, tmp_path = patched
    iface = make(z_stack={"n_planes": 5, "step_um": 1.5})
    stack = np.zeros((5, 8, 8), dtype=np.uint16)
    info = _read_tif(iface._save(stack, "scene_000", 0, channel="BF"))

    assert info["axes"] == "ZYX", (
        f"axes are {info['axes']!r} — downstream cannot tell a z-stack "
        f"from a time series"
    )
    assert info["imagej"] is True
    assert info["meta"].get("spacing") == pytest.approx(1.5), (
        "the z-step is not recorded, so the stack has no real depth"
    )
    assert info["meta"].get("unit") == "um"


def test_a_single_plane_is_still_a_plain_2d_image(patched):
    make, _, tmp_path = patched
    iface = make()
    frame = np.full((8, 8), 7, dtype=np.uint16)
    info = _read_tif(iface._save(frame, "scene_000", 0, channel="BF"))
    assert info["shape"] == (8, 8)
    assert info["axes"] == "YX"
    assert np.array_equal(info["data"], frame)


def test_pixel_size_is_recorded_in_the_tiff(patched):
    """So analysis inherits the calibration instead of re-deriving it
    from the objective, which is where scale errors come from."""
    tifffile = pytest.importorskip("tifffile")
    make, _, tmp_path = patched
    iface = make(pixel_size_um=0.65)
    path = iface._save(np.zeros((8, 8), dtype=np.uint16),
                       "scene_000", 0, channel="BF")
    with tifffile.TiffFile(path) as tf:
        tags = tf.pages[0].tags
        xres = tags["XResolution"].value
    assert xres[0] / xres[1] == pytest.approx(1 / 0.65, rel=1e-3)


def test_stack_planes_keep_their_order(patched):
    """Plane i of the file must be plane i of the stack, or every
    depth measurement downstream is mirrored."""
    make, _, tmp_path = patched
    iface = make(z_stack={"n_planes": 4, "step_um": 1.0})
    stack = np.stack([
        np.full((8, 8), v, dtype=np.uint16) for v in (10, 20, 30, 40)
    ])
    info = _read_tif(iface._save(stack, "scene_000", 0, channel="BF"))
    assert [int(p.flat[0]) for p in info["data"]] == [10, 20, 30, 40]


def test_saving_still_works_without_a_z_step_configured(patched):
    make, _, tmp_path = patched
    iface = make()                      # z_stack is None
    stack = np.zeros((3, 8, 8), dtype=np.uint16)
    info = _read_tif(iface._save(stack, "scene_000", 0, channel="BF"))
    assert info["pages"] == 3


def test_REGRESSION_z_stack_with_no_channel_asking_for_one_is_flagged(
        patched):
    """The failure mode that produces NO stack at all rather than a
    corrupt one: the master switch is on, every channel says no, so
    every frame is a single plane and the setting looks ignored."""
    make, motion, _ = patched
    iface = make(
        z_stack={"n_planes": 5, "step_um": 1.0},
        channels=[{"name": "BF", "reflector_pos": 0,
                   "exposure_ms": 5.0, "z_stack": False}],
    )
    iface.motion = motion
    problems = " ".join(iface.check_z_stack())
    assert "NO channel" in problems, (
        "a z-stack that can never be acquired went unreported"
    )


def test_no_complaint_when_a_channel_does_ask_for_the_stack(patched):
    make, motion, _ = patched
    iface = make(
        z_stack={"n_planes": 5, "step_um": 1.0},
        channels=[{"name": "BF", "reflector_pos": 0,
                   "exposure_ms": 5.0, "z_stack": True}],
    )
    iface.motion = motion
    assert not any("NO channel" in w for w in iface.check_z_stack())


def test_check_z_stack_works_before_motion_exists(patched):
    """It runs at connect(), and the mismatch check needs no hardware."""
    make, _, _ = patched
    iface = make(
        z_stack={"n_planes": 5, "step_um": 1.0},
        channels=[{"name": "BF", "reflector_pos": 0,
                   "exposure_ms": 5.0, "z_stack": False}],
    )
    iface.motion = None
    assert any("NO channel" in w for w in iface.check_z_stack())


# ====================================================================
# Z-STACK SPEED
#
# A 100-plane stack took over a minute on hardware. ~75% of that was
# time.sleep: _acquire_stack used settle_s, which is sized for the XY
# STAGE between positions, for every PIEZO step. A 1 um piezo step
# settles in milliseconds.
# ====================================================================

def test_REGRESSION_plane_settle_is_not_the_stage_settle(patched):
    make, _, _ = patched
    iface = make(settle_s=0.3)
    assert iface.plane_settle_s < iface.settle_s, (
        "z-planes are waiting for the XY stage to stop ringing"
    )
    assert iface.plane_settle_s <= 0.02


def test_plane_settle_can_be_set_explicitly(patched):
    make, _, _ = patched
    assert make(settle_s=0.3, plane_settle_s=0.1).plane_settle_s == 0.1
    # Explicitly zero must stay zero, not fall back to the default.
    assert make(settle_s=0.3, plane_settle_s=0.0).plane_settle_s == 0.0


def test_plane_settle_never_exceeds_the_stage_settle(patched):
    """A tiny stage settle means a fast rig; do not slow the stack."""
    make, _, _ = patched
    assert make(settle_s=0.005).plane_settle_s == 0.005


def test_a_hundred_plane_stack_does_not_sleep_for_thirty_seconds(
        patched):
    """The actual complaint, as a number."""
    make, motion, _ = patched
    iface = make(settle_s=0.3, z_stack={"n_planes": 100,
                                        "step_um": 0.5})
    iface.motion = motion
    slept = []
    import tracking_tools.microscope_interface.mtb_backend as mod
    real_sleep = mod.time.sleep
    mod.time.sleep = lambda s: slept.append(s)
    try:
        iface._acquire_stack(250.0, lambda z: np.zeros((4, 4), np.uint16))
    finally:
        mod.time.sleep = real_sleep
    assert sum(slept) <= 2.5, (
        f"slept {sum(slept):.1f}s across 100 planes"
    )


def test_exposure_is_set_once_per_stack_not_once_per_plane(patched):
    """setExposure can reconfigure the camera; 100 planes meant 100
    needless round trips."""
    make, motion, _ = patched
    iface = make(z_stack={"n_planes": 20, "step_um": 1.0},
                 synthetic_source=None)
    iface.motion = motion
    iface.plane_settle_s = 0.0

    class CountingCore:
        def __init__(self):
            self.exposures = 0

        def setExposure(self, ms):
            self.exposures += 1

        def snapImage(self):
            pass

        def getImage(self):
            return np.zeros((4, 4), dtype=np.uint16)

    iface.mmc = CountingCore()
    iface._acquire("scene_000", 0.0, 0.0, 250.0, stack=True,
                   exposure_ms=5.0,
                   channel={"name": "BF", "z_stack": True})
    assert iface.mmc.exposures == 1, (
        f"camera exposure set {iface.mmc.exposures} times for 20 planes"
    )


def test_a_single_frame_still_sets_its_exposure(patched):
    make, motion, _ = patched
    iface = make(synthetic_source=None)
    iface.motion = motion

    class CountingCore:
        def __init__(self):
            self.exposures = 0

        def setExposure(self, ms):
            self.exposures += 1

        def snapImage(self):
            pass

        def getImage(self):
            return np.zeros((4, 4), dtype=np.uint16)

    iface.mmc = CountingCore()
    iface._acquire("scene_000", 0.0, 0.0, 250.0, stack=False,
                   exposure_ms=5.0, channel={"name": "BF"})
    assert iface.mmc.exposures == 1


def test_REGRESSION_an_unstacked_channel_is_lit_when_per_plane_is_off(
        patched):
    """gate_light_per_plane is about holding a mechanical shutter open
    ACROSS THE PLANES OF A STACK. It has no meaning for a single frame,
    but it was passed through anyway — so an unstacked channel with the
    box unticked was never lit at all and its frames came back black.

    The hold-open branch that would have lit it sits AFTER
    `if not want: return grab(z)`, so it was unreachable."""
    make, _, _ = patched
    light = LightLog()
    core = LoggingCore(light)
    iface = make(
        synthetic_source=None, mmc=core, camera_label="Cam",
        stop_after_tp=1, illuminate=light.illuminate,
        gate_light_per_plane=False,
        z_stack=None,
        channels=[{"name": "BF", "reflector_pos": 0, "tl": True,
                   "exposure_ms": 5.0, "z_stack": False}],
    )
    _run(iface, seconds=3.0)

    assert light.snaps_while_lit > 0, (
        "every frame was acquired in the dark — the light was never "
        "switched on"
    )
    assert light.snaps_while_dark == 0


def test_a_stacked_channel_still_holds_the_shutter_open(patched):
    """The flag must keep working for the case it exists for."""
    make, _, _ = patched
    light = LightLog()
    iface = make(
        synthetic_source=None, mmc=LoggingCore(light),
        camera_label="Cam", stop_after_tp=1,
        illuminate=light.illuminate,
        gate_light_per_plane=False,
        z_stack={"n_planes": 4, "step_um": 1.0},
        channels=[{"name": "BF", "reflector_pos": 0, "tl": True,
                   "exposure_ms": 5.0, "z_stack": True}],
    )
    _run(iface, seconds=3.0)
    ons = [e for e in light.events if e[0] == "on"]
    # Once per POSITION visit, not once per plane. The fixture has two
    # positions and the stack has four planes, so per-plane gating
    # would be 8 cycles and holding open is 2 — that gap is the whole
    # point of the flag.
    assert len(ons) == 2, (
        f"the shutter was cycled {len(ons)} times; expected one "
        f"open/close per position across the whole 4-plane stack"
    )
    assert light.snaps_while_dark == 0
    assert light.events[-1][0] == "off"
