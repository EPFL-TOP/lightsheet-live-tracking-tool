"""Unit tests for MicroscopeInterface_Micromanager.

These tests deliberately avoid any real Micro-Manager / pymmcore-plus /
useq installation by monkey-patching ``sys.modules`` with fake modules
BEFORE the interface module is imported.  A ``MagicMock`` stands in for
``CMMCorePlus`` so the entire class exercises its real code paths
(config load, event enqueue, frameReady handling, drift accounting,
device wait) without ever touching hardware.

Run with:

    pytest tests/test_mm_interface.py -v
"""

from __future__ import annotations

import os
import queue
import sys
import time
import types
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo root is on sys.path so ``tracking_tools`` resolves
# regardless of pytest's rootdir / invocation cwd.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Fake pymmcore_plus + useq modules
#
# MicroscopeInterface_Micromanager does:
#     import pymmcore_plus                       (probe, in __init__)
#     from useq import MDAEvent                  (in __init__)
#     from pymmcore_plus import CMMCorePlus      (late, in connect)
#
# We register both under sys.modules BEFORE importing the interface so
# every one of those statements resolves to our stand-ins.
# ---------------------------------------------------------------------------


class _FakeMDAEvent:
    """Minimal MDAEvent stand-in with just enough attributes for the
    interface to build events and for the tests to inspect them."""

    def __init__(self, index=None, x_pos=None, y_pos=None, z_pos=None,
                 exposure=None, channel=None, min_start_time=None,
                 metadata=None):
        self.index = index or {}
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        self.exposure = exposure
        self.channel = channel
        self.min_start_time = min_start_time
        self.metadata = metadata or {}

    def __repr__(self):
        return (f"_FakeMDAEvent(index={self.index}, pos=({self.x_pos},"
                f"{self.y_pos},{self.z_pos}), metadata={self.metadata})")


def _install_fake_modules():
    """Register fake ``pymmcore_plus`` + ``useq`` in sys.modules.

    Called at module import time so the interface module (imported next)
    resolves its inline imports against these fakes.  Safe to call
    repeatedly.
    """
    if 'pymmcore_plus' not in sys.modules:
        fake_pmp = types.ModuleType('pymmcore_plus')
        # CMMCorePlus is replaced per-test with a fresh MagicMock; the
        # stub attribute here is only what ``import pymmcore_plus``
        # sees when the interface's __init__ probes availability.
        fake_pmp.CMMCorePlus = mock.MagicMock(name='CMMCorePlus_stub')
        sys.modules['pymmcore_plus'] = fake_pmp

    if 'useq' not in sys.modules:
        fake_useq = types.ModuleType('useq')
        fake_useq.MDAEvent = _FakeMDAEvent
        sys.modules['useq'] = fake_useq


_install_fake_modules()

# Now safe to import the interface + synthetic source.
from tracking_tools.microscope_interface.MicroscopeInterface import (  # noqa: E402
    MicroscopeInterface_Micromanager,
)
from tracking_tools.microscope_interface.synthetic_source import (  # noqa: E402
    DriftingGaussianEmbryo,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def positions_config():
    """Two-scene positions_config with explicit xyz_um baselines."""
    return {
        'p1': {'xyz_um': (10.0, 20.0, 30.0), 'channel_preset': 'BF'},
        'p2': {'xyz_um': (100.0, 200.0, 300.0), 'channel_preset': 'BF'},
    }


@pytest.fixture
def mm_params(tmp_path):
    return {
        'cfg_path': str(tmp_path / 'fake.cfg'),
        'channel_group': 'Channel',
        'channel_preset': 'BF',
        'exposure_ms': 42.0,
        'max_xy_um': 500.0,
        'max_z_um': 100.0,
        'stop_after_tp': 100,  # very large; tests never rely on stop
    }


def _make_mock_cmmcore(getXYPosition_raises=True):
    """Build a fresh MagicMock CMMCorePlus.

    By default ``getXYPosition`` raises RuntimeError — Bug #3 regression
    guard: baselines must come from positions_config, never from the
    stage.  Individual tests can override via ``getXYPosition_raises=False``.
    """
    mmc = mock.MagicMock(name='CMMCorePlus_instance')
    if getXYPosition_raises:
        mmc.getXYPosition.side_effect = RuntimeError(
            "Bug #3 regression: getXYPosition() must never be called"
        )
    # deviceBusy returns False by default so _wait_for_device would exit
    # immediately.  Tests that want to force a timeout override this.
    mmc.deviceBusy.return_value = False
    return mmc


@pytest.fixture
def patched_cmm(positions_config, mm_params, tmp_path):
    """Instantiate the interface with a mocked ``CMMCorePlus``.

    Also creates an on-disk cfg_path so ``connect()`` doesn't take the
    "no cfg — load demo" branch.  Returns ``(iface, mmc)``.
    """
    Path(mm_params['cfg_path']).write_text('# fake mm config\n')

    mmc = _make_mock_cmmcore()

    # Patch CMMCorePlus.instance to return our mock every time the
    # interface calls it from connect().
    fake_cls = mock.MagicMock(name='CMMCorePlus_class')
    fake_cls.instance = mock.MagicMock(return_value=mmc)

    with mock.patch.dict(sys.modules, {
        'pymmcore_plus': types.SimpleNamespace(CMMCorePlus=fake_cls),
    }):
        iface = MicroscopeInterface_Micromanager(
            positions_config=positions_config,
            dirpath=str(tmp_path),
            mm_params=mm_params,
        )
        yield iface, mmc


def _drain_queue(q):
    """Non-blockingly drain a queue.Queue into a list."""
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def _make_event(pos_name, tp, scene_idx, n_scenes,
                x=0.0, y=0.0, z=0.0, z_idx=0, n_z=1):
    """Build a _FakeMDAEvent with the metadata dict frameReady expects."""
    return _FakeMDAEvent(
        index={'t': tp, 'p': scene_idx, 'z': z_idx},
        x_pos=x, y_pos=y, z_pos=z,
        exposure=100.0,
        channel={'group': 'Channel', 'config': 'BF'},
        metadata={
            'pos_name': pos_name,
            'tp': tp,
            'scene_idx': scene_idx,
            'z_idx': z_idx,
            'n_z': n_z,
            'is_last_scene': (scene_idx == n_scenes - 1),
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instantiation_loads_config(patched_cmm, mm_params):
    """connect() must forward the exact cfg_path to loadSystemConfiguration."""
    iface, mmc = patched_cmm
    iface.connect()
    mmc.loadSystemConfiguration.assert_called_once_with(mm_params['cfg_path'])


def test_baselines_come_from_positions_config_NOT_stage(patched_cmm,
                                                        positions_config):
    """Bug #3 regression.

    The mocked ``getXYPosition`` raises if invoked — instantiating and
    connecting must succeed, meaning baselines were read from
    ``positions_config[pos]['xyz_um']`` instead.
    """
    iface, mmc = patched_cmm
    # Instantiation already happened in the fixture; connect exercises
    # the additional startup code.
    iface.connect()

    assert not mmc.getXYPosition.called, (
        "getXYPosition() was called — baselines must come from "
        "positions_config, never from the stage (Bug #3)"
    )
    for pos_name, cfg in positions_config.items():
        expected = tuple(float(v) for v in cfg['xyz_um'])
        assert iface._baseline_um[pos_name] == expected


def test_relative_move_updates_cum_drift_immediately(patched_cmm):
    """Bug #2 regression: cum_drift must be updated synchronously."""
    iface, _ = patched_cmm
    iface.relative_move('p1', 1.5, -0.7, 0.2)
    # Read immediately — no MDA event needed to propagate.
    assert iface._cum_drift['p1'] == [1.5, -0.7, 0.2]


def test_relative_move_clamps_to_max_xy_um(patched_cmm):
    """Shifts beyond ``max_xy_um`` (500) clamp to ±500 on X and Y."""
    iface, _ = patched_cmm

    iface.relative_move('p1', 1000.0, -1000.0, 0.0)
    assert iface._cum_drift['p1'][0] == pytest.approx(500.0)
    assert iface._cum_drift['p1'][1] == pytest.approx(-500.0)
    assert iface._cum_drift['p1'][2] == pytest.approx(0.0)

    # Fresh scene — verify negative-direction clamp in isolation too.
    iface.relative_move('p2', -1000.0, 1000.0, 0.0)
    assert iface._cum_drift['p2'][0] == pytest.approx(-500.0)
    assert iface._cum_drift['p2'][1] == pytest.approx(500.0)


def test_relative_move_clamps_to_max_z_um(patched_cmm):
    """Z clamps at ``max_z_um`` (100 by default)."""
    iface, _ = patched_cmm
    iface.relative_move('p1', 0.0, 0.0, 1000.0)
    assert iface._cum_drift['p1'][2] == pytest.approx(100.0)

    iface.relative_move('p2', 0.0, 0.0, -1000.0)
    assert iface._cum_drift['p2'][2] == pytest.approx(-100.0)


def test_wait_for_image_returns_none_on_empty_queue(patched_cmm):
    """wait_for_image blocks up to timeout then returns triple-None."""
    iface, _ = patched_cmm
    t0 = time.monotonic()
    result = iface.wait_for_image(timeout_ms=100)
    elapsed = time.monotonic() - t0
    assert result == (None, None, None)
    # Must actually wait ~100 ms (with plenty of slack) — proves it hit
    # the Queue.get timeout path.
    assert 0.05 <= elapsed < 0.5


def test_connect_enqueues_t0_for_every_scene(patched_cmm, positions_config):
    """After connect(), the outgoing MDA queue has exactly one event per
    scene at t=0, ordered by scene index, with x/y/z_pos == baseline."""
    iface, _ = patched_cmm
    iface.connect()

    events = _drain_queue(iface._mda_queue)
    # No z-stack in fixture, one event per scene
    assert len(events) == len(positions_config)

    ordered_names = list(positions_config.keys())
    for scene_idx, (ev, pos_name) in enumerate(zip(events, ordered_names)):
        assert ev.metadata['tp'] == 0
        assert ev.metadata['scene_idx'] == scene_idx
        assert ev.metadata['pos_name'] == pos_name
        bx, by, bz = positions_config[pos_name]['xyz_um']
        assert ev.x_pos == pytest.approx(bx)
        assert ev.y_pos == pytest.approx(by)
        assert ev.z_pos == pytest.approx(bz)


def test_frame_ready_last_scene_auto_enqueues_next_tp(patched_cmm,
                                                      positions_config):
    """Bug #1 regression: after the LAST scene's frameReady, the queue
    must be extended by n_scenes new events for tp=1 UNCONDITIONALLY
    (no relative_move needed to trigger it)."""
    iface, _ = patched_cmm
    iface.connect()

    # Drain the tp=0 events primed by connect()
    _drain_queue(iface._mda_queue)

    pos_names = list(positions_config.keys())
    n = len(pos_names)

    # Feed each scene's tp=0 frameReady in order; make the images
    # non-trivial so we can also assert enqueue happens without drift.
    for scene_idx, pos_name in enumerate(pos_names):
        bx, by, bz = positions_config[pos_name]['xyz_um']
        img = np.zeros((32, 32), dtype=np.uint16)
        ev = _make_event(pos_name, tp=0, scene_idx=scene_idx, n_scenes=n,
                         x=bx, y=by, z=bz)
        iface._on_frame_ready(img, ev)

    new_events = _drain_queue(iface._mda_queue)
    assert len(new_events) == n, (
        f"Expected {n} events at tp=1 after last-scene frameReady "
        f"(Bug #1: unconditional next-tp enqueue), got {len(new_events)}"
    )
    for scene_idx, (ev, pos_name) in enumerate(zip(new_events, pos_names)):
        assert ev.metadata['tp'] == 1
        assert ev.metadata['scene_idx'] == scene_idx
        assert ev.metadata['pos_name'] == pos_name


def test_frame_ready_pushes_to_tracker_queue(patched_cmm, positions_config):
    """A frameReady call must push (image, tp, pos_name) to the tracker
    queue verbatim (image identity preserved, no synthetic in play)."""
    iface, _ = patched_cmm
    iface.connect()
    _drain_queue(iface._mda_queue)

    pos_name = list(positions_config.keys())[0]
    img = np.arange(64, dtype=np.uint16).reshape(8, 8)
    ev = _make_event(pos_name, tp=0, scene_idx=0, n_scenes=2,
                     x=10.0, y=20.0, z=30.0)
    iface._on_frame_ready(img, ev)

    got_image, got_tp, got_pos = iface._image_queue.get(timeout=1.0)
    assert got_tp == 0
    assert got_pos == pos_name
    # The interface calls np.asarray(image) before enqueuing but it's a
    # zero-copy view of the same underlying buffer.
    np.testing.assert_array_equal(got_image, img)


def test_synthetic_source_replaces_image(positions_config, mm_params,
                                         tmp_path):
    """When synthetic_source is set, the image landing on the tracker
    queue must be the sentinel returned by the callable — not whatever
    frameReady was originally given."""
    Path(mm_params['cfg_path']).write_text('# fake mm config\n')

    sentinel = np.full((16, 16), 42, dtype=np.uint16)

    def source(x, y, z, pos_name):
        return sentinel

    params = dict(mm_params)
    params['synthetic_source'] = source

    mmc = _make_mock_cmmcore()
    fake_cls = mock.MagicMock()
    fake_cls.instance = mock.MagicMock(return_value=mmc)
    with mock.patch.dict(sys.modules, {
        'pymmcore_plus': types.SimpleNamespace(CMMCorePlus=fake_cls),
    }):
        iface = MicroscopeInterface_Micromanager(
            positions_config=positions_config,
            dirpath=str(tmp_path),
            mm_params=params,
        )
        iface.connect()
        _drain_queue(iface._mda_queue)

        original = np.zeros((16, 16), dtype=np.uint16)  # NOT the sentinel
        pos_name = list(positions_config.keys())[0]
        ev = _make_event(pos_name, tp=0, scene_idx=0, n_scenes=2,
                         x=0.0, y=0.0, z=0.0)
        iface._on_frame_ready(original, ev)

        got_image, _, _ = iface._image_queue.get(timeout=1.0)
        assert got_image is sentinel or np.array_equal(got_image, sentinel)
        # And confirm the ORIGINAL image was NOT what landed.
        assert not np.array_equal(got_image, original)


def test_wait_for_device_times_out(patched_cmm):
    """_wait_for_device must raise TimeoutError within roughly the
    requested budget when deviceBusy never returns False."""
    iface, mmc = patched_cmm
    mmc.deviceBusy.return_value = True

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        iface._wait_for_device('SomeDevice', timeout_s=0.1)
    elapsed = time.monotonic() - t0
    # Generous upper bound but still << 200 ms as required
    assert elapsed < 0.2, f"_wait_for_device took {elapsed*1000:.1f} ms"


def test_refresh_filename_reloads_positions_config(patched_cmm,
                                                   positions_config):
    """Mutating positions_config in place and calling refresh_filename
    must update the interface's cached channel_preset and baseline."""
    iface, _ = patched_cmm

    pos_name = 'p1'
    # Sanity: pre-refresh values reflect the original fixture
    assert iface.channel_preset == 'BF'
    assert iface._baseline_um[pos_name] == (10.0, 20.0, 30.0)

    # Simulate the ROI JSON watcher rewriting the config in place
    positions_config[pos_name]['channel_preset'] = 'DAPI'
    positions_config[pos_name]['xyz_um'] = (11.0, 22.0, 33.0)

    iface.refresh_filename(pos_name)

    assert iface.channel_preset == 'DAPI'
    assert iface._baseline_um[pos_name] == (11.0, 22.0, 33.0)


def test_synthetic_drifting_embryo_advances_ground_truth():
    """DriftingGaussianEmbryo increments its ground-truth center by
    ``drift_um_per_tp`` on each call for the same pos_name, and the
    rendered blob's peak sits at the mathematically expected pixel."""
    shape = (128, 128)
    sigma = 8.0
    pixel_size_um = 0.5
    drift = (1.0, 2.0, 0.0)

    src = DriftingGaussianEmbryo(
        shape=shape, sigma=sigma,
        drift_um_per_tp=drift,
        pixel_size_um=pixel_size_um,
        noise_std=0.0,          # deterministic peak location
        seed=0,
    )

    # Call the source 3 times at fixed stage (0,0,0).  Verify:
    #   1. ground-truth center in µm advances by exactly (1, 2, 0) per call
    #   2. the returned image's argmax matches the analytical peak position
    H, W = shape
    center_x_px = W / 2.0
    center_y_px = H / 2.0

    for tp in range(3):
        img = src(0.0, 0.0, 0.0, 'p')
        assert img.shape == shape
        assert img.dtype == np.uint16

        # Ground-truth center after this call — the counter was
        # incremented AFTER we captured `tp`, so use the pre-increment
        # value.
        expected_true_um = (tp * drift[0], tp * drift[1], tp * drift[2])

        # Blob pixel = center + (true_um - stage_um) / pixel_size
        expected_px_x = center_x_px + (expected_true_um[0] - 0.0) / pixel_size_um
        expected_px_y = center_y_px + (expected_true_um[1] - 0.0) / pixel_size_um

        peak = np.unravel_index(np.argmax(img), img.shape)
        peak_y, peak_x = peak
        assert abs(peak_x - expected_px_x) <= 1, (
            f"tp={tp}: peak_x={peak_x} expected≈{expected_px_x}"
        )
        assert abs(peak_y - expected_px_y) <= 1, (
            f"tp={tp}: peak_y={peak_y} expected≈{expected_px_y}"
        )

    # After 3 calls, the internal tp counter for 'p' should be 3 —
    # confirms the per-call increment happened three times.
    assert src._tp['p'] == 3
