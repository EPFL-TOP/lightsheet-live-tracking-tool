"""Tests for the MTB setup panel's non-GUI logic.

Constructs the real panel (Panel widgets work headless) but never
touches hardware. Covers the parts that would otherwise only be
exercised by clicking at the microscope: validation gates, position
bookkeeping, and the config handed to the backend.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("panel")

from interactive_tools.mtb_setup import (  # noqa: E402
    MTBSetupPanel,
    _stretch_to_png,
)


@pytest.fixture
def panel_obj():
    return MTBSetupPanel()


class FakeMotion:
    def __init__(self, xyz=(1.0, 2.0, 3.0)):
        self._xyz = xyz
        self.moved_to = []
        self.moved_by = []

    def get_xyz(self):
        return self._xyz

    def move_to(self, x=None, y=None, z=None):
        self.moved_to.append((x, y, z))
        return self._xyz

    def move_by(self, dx=0.0, dy=0.0, dz=0.0):
        self.moved_by.append((dx, dy, dz))
        return self._xyz

    def describe(self):
        return "fake"


# ----------------------------------------------------------- building

def test_layout_builds_without_hardware(panel_obj):
    assert panel_obj.layout() is not None


def test_starts_disconnected(panel_obj):
    assert not panel_obj.connected
    assert panel_obj.positions_config() == {}


def test_stop_button_starts_disabled(panel_obj):
    assert panel_obj.btn_stop.disabled
    assert not panel_obj.btn_run.disabled


# --------------------------------------------------------- validation

def test_run_refused_when_disconnected(panel_obj):
    assert "not connected" in panel_obj._validate_run()


def test_run_refused_without_camera(panel_obj):
    panel_obj.motion = FakeMotion()
    assert "Camera" in panel_obj._validate_run()


def test_run_refused_without_positions(panel_obj):
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    assert "No positions" in panel_obj._validate_run()


def test_run_refused_without_output_directory(panel_obj, tmp_path):
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj.pos_table.value = pd.DataFrame(
        [{"name": "scene_000", "x_um": 1.0, "y_um": 2.0, "z_um": 3.0}]
    )
    panel_obj.outdir.value = ""
    assert "experiment root" in panel_obj._validate_run()


def test_run_refused_when_output_directory_is_non_empty(panel_obj,
                                                        tmp_path):
    """Frames are named t0000_<channel>.tif, so reusing a populated
    folder would silently collide with a previous run."""
    (tmp_path / "existing.tif").write_text("x")
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj.pos_table.value = pd.DataFrame(
        [{"name": "scene_000", "x_um": 1.0, "y_um": 2.0, "z_um": 3.0}]
    )
    panel_obj.outdir.value = str(tmp_path)
    assert "not empty" in panel_obj._validate_run()


def test_run_accepted_when_everything_is_set(panel_obj, tmp_path):
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj.pos_table.value = pd.DataFrame(
        [{"name": "scene_000", "x_um": 1.0, "y_um": 2.0, "z_um": 3.0}]
    )
    panel_obj.outdir.value = str(tmp_path / "fresh")
    assert panel_obj._validate_run() is None


# ---------------------------------------------------------- positions

def test_capture_appends_the_current_stage_position(panel_obj):
    panel_obj.motion = FakeMotion(xyz=(10.5, 20.25, 250.125))
    panel_obj._on_capture()
    df = panel_obj.pos_table.value
    assert len(df) == 1
    row = df.iloc[0]
    assert row["name"] == "scene_000"
    assert row["x_um"] == pytest.approx(10.5)
    assert row["z_um"] == pytest.approx(250.125)


def test_capture_numbers_positions_sequentially(panel_obj):
    panel_obj.motion = FakeMotion()
    for _ in range(3):
        panel_obj._on_capture()
    names = list(panel_obj.pos_table.value["name"])
    assert names == ["scene_000", "scene_001", "scene_002"]


def test_capture_without_connection_does_not_add_a_row(panel_obj):
    panel_obj._on_capture()
    assert panel_obj.pos_table.value.empty


def test_positions_config_matches_the_backend_contract(panel_obj):
    """The backend requires {name: {'xyz_um': (x, y, z)}} and rejects
    anything without an xyz_um baseline."""
    panel_obj.motion = FakeMotion(xyz=(1.0, 2.0, 3.0))
    panel_obj._on_capture()
    cfg = panel_obj.positions_config()
    assert cfg == {"scene_000": {"xyz_um": (1.0, 2.0, 3.0)}}


def test_remove_selected_drops_the_right_rows(panel_obj):
    panel_obj.pos_table.value = pd.DataFrame([
        {"name": "a", "x_um": 0.0, "y_um": 0.0, "z_um": 0.0},
        {"name": "b", "x_um": 1.0, "y_um": 1.0, "z_um": 1.0},
        {"name": "c", "x_um": 2.0, "y_um": 2.0, "z_um": 2.0},
    ])
    panel_obj.pos_table.selection = [1]
    panel_obj._on_remove()
    assert list(panel_obj.pos_table.value["name"]) == ["a", "c"]


def test_goto_requires_exactly_one_selection(panel_obj):
    motion = FakeMotion()
    panel_obj.motion = motion
    panel_obj.pos_table.value = pd.DataFrame([
        {"name": "a", "x_um": 5.0, "y_um": 6.0, "z_um": 7.0},
        {"name": "b", "x_um": 1.0, "y_um": 1.0, "z_um": 1.0},
    ])
    panel_obj.pos_table.selection = [0, 1]
    panel_obj._on_goto()
    assert not motion.moved_to, "ambiguous selection must not move"

    panel_obj.pos_table.selection = [0]
    panel_obj._on_goto()
    assert motion.moved_to == [(5.0, 6.0, 7.0)]


def test_save_and_load_round_trip(panel_obj, tmp_path):
    panel_obj.motion = FakeMotion(xyz=(11.0, 22.0, 33.0))
    panel_obj._on_capture()
    path = tmp_path / "positions.json"
    panel_obj.positions_file.value = str(path)
    panel_obj._on_save_positions()

    on_disk = json.loads(path.read_text())
    assert on_disk == {"scene_000": {"xyz_um": [11.0, 22.0, 33.0]}}

    fresh = MTBSetupPanel()
    fresh.positions_file.value = str(path)
    fresh._on_load_positions()
    assert fresh.positions_config() == {
        "scene_000": {"xyz_um": (11.0, 22.0, 33.0)}
    }


def test_load_reports_a_missing_file_instead_of_raising(panel_obj,
                                                       tmp_path):
    panel_obj.positions_file.value = str(tmp_path / "nope.json")
    panel_obj._on_load_positions()
    assert "No such file" in panel_obj.status.object


# --------------------------------------------------------------- jog

def test_jog_passes_signed_deltas_through(panel_obj):
    motion = FakeMotion()
    panel_obj.motion = motion
    panel_obj.xy_step.value = 10.0
    panel_obj.btn_xm.clicks += 1     # trigger the on_click handler
    assert motion.moved_by, "jog did not reach the motion layer"
    assert motion.moved_by[-1][0] == pytest.approx(-10.0)


def test_jog_without_connection_is_a_no_op(panel_obj):
    panel_obj._jog(dx=5.0)
    assert "Connect the hardware" in panel_obj.status.object


# ------------------------------------------------------ image render

def test_stretch_handles_a_low_contrast_16bit_frame():
    """Real frames sit near 11000-14800 of 65535, so a naive cast
    renders nearly black; the stretch must use the actual range."""
    rng = np.random.default_rng(0)
    img = rng.integers(11000, 14800, size=(64, 64), dtype=np.uint16)
    png = _stretch_to_png(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_stretch_projects_a_zstack():
    stack = np.zeros((5, 32, 32), dtype=np.uint16)
    stack[2] = 5000
    assert _stretch_to_png(stack)[:4] == b"\x89PNG"


def test_stretch_survives_a_constant_frame():
    """A saturated or dark frame has hi == lo; must not divide by zero."""
    flat = np.full((16, 16), 65535, dtype=np.uint16)
    assert _stretch_to_png(flat)[:4] == b"\x89PNG"


# ====================================================================
# Import path
#
# `panel serve interactive_tools/mtb_setup.py` puts only that file's
# DIRECTORY on sys.path, not the repo root, so `import tracking_tools`
# failed at runtime with
#   Cannot import the MTB layer: No module named 'tracking_tools'
# even though the tests passed (pytest runs from the repo root).
# Observed 2026-09-04.
# ====================================================================

def test_REGRESSION_module_puts_repo_root_on_sys_path():
    """Importing the panel must make tracking_tools importable."""
    import subprocess
    import sys as _sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    root = here.parent
    code = (
        "import sys\n"
        "sys.path = [p for p in sys.path "
        f"if {str(root)!r} not in p]\n"
        f"sys.path.insert(0, {str(root / 'interactive_tools')!r})\n"
        "import mtb_setup\n"
        "import tracking_tools.microscope_interface.mtb as m\n"
        "print('OK', bool(m.MTB_IDS))\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(root.parent),
    )
    assert "OK True" in result.stdout, (
        "panel app does not make tracking_tools importable; "
        f"stdout={result.stdout!r} stderr={result.stderr[-500:]!r}"
    )


def test_connect_reports_a_missing_dependency_clearly(panel_obj):
    """Without pythonnet the message must name the fix, and the panel
    must stay usable rather than raising."""
    panel_obj._on_connect()
    msg = panel_obj.status.object
    assert "❌" in msg
    # Either the import failed or the connection did; both must be
    # reported, never raised.
    assert "MTB" in msg
    assert not panel_obj.connected


# ====================================================================
# Bokeh restores sys.path after running the app script
#
# `panel serve` uses Bokeh's CodeHandler, which snapshots sys.path
# before executing the served script and RESTORES it afterwards. A
# module-level `sys.path.insert(repo_root)` therefore does NOT survive
# into button callbacks, so any repo-local import deferred into a
# callback fails at click time with
#   Cannot import the MTB layer: No module named 'tracking_tools'
#   ... on sys.path: False
# even though the same import succeeds at module scope.
# zeiss_panel_app.py imports at the top and works; deferring broke it.
# Observed 2026-09-04.
# ====================================================================

def test_REGRESSION_repo_imports_resolved_at_module_scope():
    """The MTB names must already be bound on the module, so callbacks
    never need sys.path to still contain the repo root."""
    import interactive_tools.mtb_setup as m

    assert hasattr(m, "MTBSession")
    assert hasattr(m, "MTBMotion")
    assert hasattr(m, "MicroscopeInterface_MTB")
    assert m.MTBSession is not None, (
        "MTBSession failed to import at module scope: "
        f"{m._MTB_IMPORT_ERROR}"
    )
    assert m.MicroscopeInterface_MTB is not None, (
        "backend failed to import at module scope: "
        f"{m._BACKEND_IMPORT_ERROR}"
    )


def test_REGRESSION_callbacks_work_after_sys_path_is_restored(
        monkeypatch):
    """Simulate Bokeh: import the module with the root on sys.path,
    then strip it, then exercise a callback."""
    import sys as _sys
    import interactive_tools.mtb_setup as m

    root = m._ROOT
    stripped = [p for p in _sys.path if os.path.abspath(p) != root]
    monkeypatch.setattr(_sys, "path", stripped)
    assert root not in _sys.path, "precondition: root must be gone"

    panel_obj = m.MTBSetupPanel()

    # Connect must get past the import stage and fail (if at all) on
    # the hardware, never on module resolution.
    panel_obj._on_connect()
    assert "No module named" not in (panel_obj.status.object or ""), (
        "a callback still performs a repo-local import at call time"
    )

    # And the validation path must not blame a missing backend.
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj.pos_table.value = pd.DataFrame(
        [{"name": "scene_000", "x_um": 1.0, "y_um": 2.0, "z_um": 3.0}]
    )
    panel_obj.outdir.value = "/tmp/does-not-exist-yet-xyz"
    assert panel_obj._validate_run() is None


def test_no_repo_local_imports_inside_methods():
    """Static guard: no `tracking_tools` import may sit inside a
    function body, because Bokeh restores sys.path before callbacks
    run. Module-level imports inside try/except are fine, so this
    walks the AST rather than matching indentation."""
    import ast
    import inspect
    import interactive_tools.mtb_setup as m

    tree = ast.parse(inspect.getsource(m))
    offenders = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = ", ".join(a.name for a in node.names)
            if mod and "tracking_tools" in mod:
                offenders.append(f"{func.name}() imports {mod}")

    assert not offenders, (
        "repo-local imports inside a function body fail under "
        f"panel serve: {offenders}"
    )


# ---------------------------------------------------- live mode

def test_live_starts_off(panel_obj):
    assert panel_obj.btn_live.value is False
    assert panel_obj._live_cb is None


def test_live_refused_without_a_camera(panel_obj):
    panel_obj.btn_live.value = True
    assert panel_obj.btn_live.value is False
    assert "Camera not available" in panel_obj.status.object


def test_live_refused_while_a_run_is_active(panel_obj):
    """The camera is exclusive; live snapping would interleave with the
    run's own acquisitions."""
    import threading

    panel_obj.mmc = object()
    started = threading.Event()
    stop = threading.Event()

    def busy():
        started.set()
        stop.wait(5.0)

    panel_obj._run_thread = threading.Thread(target=busy, daemon=True)
    panel_obj._run_thread.start()
    started.wait(1.0)
    try:
        panel_obj.btn_live.value = True
        assert panel_obj.btn_live.value is False
        assert "camera is exclusive" in panel_obj.status.object
    finally:
        stop.set()


def test_stop_live_is_idempotent(panel_obj):
    panel_obj._stop_live()
    panel_obj._stop_live()
    assert panel_obj._live_cb is None


# ------------------------------------------- two-phase workflow

def test_positions_config_includes_log_dir_when_root_given(panel_obj):
    panel_obj.motion = FakeMotion(xyz=(1.0, 2.0, 3.0))
    panel_obj._on_capture()
    cfg = panel_obj.positions_config(root="/exp")
    entry = cfg["scene_000"]
    assert entry["xyz_um"] == (1.0, 2.0, 3.0)
    assert entry["log_dir"] == os.path.join(
        "/exp", "scene_000", "embryo_tracking"
    )


def test_positions_config_omits_log_dir_without_root(panel_obj):
    panel_obj.motion = FakeMotion()
    panel_obj._on_capture()
    assert "log_dir" not in panel_obj.positions_config()["scene_000"]


def test_roi_status_detects_missing_and_present_rois(panel_obj,
                                                     tmp_path):
    panel_obj.motion = FakeMotion()
    panel_obj._on_capture()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path)

    roi_dir = tmp_path / "scene_000" / "embryo_tracking"
    roi_dir.mkdir(parents=True)
    (roi_dir / "tracking_RoIs.json").write_text("{}")

    state = panel_obj.roi_status()
    assert state == {"scene_000": True, "scene_001": False}


def test_REGRESSION_missing_rois_do_not_block_starting(panel_obj,
                                                       tmp_path):
    """Tracking runs ON TOP OF a running acquisition: the loop must
    start with no ROIs at all, because ROIs can only be drawn on
    frames that already exist. An earlier version refused to start
    without them, which made the workflow impossible."""
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path / "fresh")
    panel_obj.auto_track.value = True

    assert panel_obj._validate_run() is None, (
        "must be able to start acquiring before any ROI exists"
    )


def test_start_allowed_with_tracking_disabled(panel_obj, tmp_path):
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path / "fresh")
    panel_obj.auto_track.value = False
    assert panel_obj._validate_run() is None


def test_populated_folder_without_rois_is_refused(panel_obj, tmp_path):
    """Guards against overwriting a previous experiment's frames."""
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path)
    (tmp_path / "stale.tif").write_text("x")
    problem = panel_obj._validate_run()
    assert problem is not None and "not empty" in problem


def test_populated_folder_WITH_rois_is_allowed(panel_obj, tmp_path):
    """Resuming into the folder the acquisition already filled is the
    normal path once ROIs have been drawn."""
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path)

    roi_dir = tmp_path / "scene_000" / "embryo_tracking"
    roi_dir.mkdir(parents=True)
    (roi_dir / "tracking_RoIs.json").write_text("{}")
    (tmp_path / "scene_000" / "t0000_BF.tif").write_text("x")

    assert panel_obj._validate_run() is None


def test_tracking_requested_follows_the_auto_track_checkbox(panel_obj):
    panel_obj.auto_track.value = False
    assert not panel_obj.tracking_requested
    panel_obj.auto_track.value = True
    assert panel_obj.tracking_requested


def test_check_rois_explains_how_to_draw_them(panel_obj, tmp_path):
    panel_obj.motion = FakeMotion()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path)
    panel_obj._refresh_roi_state()
    assert "Selection" in panel_obj.roi_state.object
    assert "tracking_RoIs.json" in panel_obj.roi_state.object


# ====================================================================
# Partial ROI coverage
#
# get_pos_config() globs <root>/*/<log_dir_name> and opens
# tracking_RoIs.json UNCONDITIONALLY. The panel creates those folders
# up front so the ROI watcher can watch them, so a position with no
# ROIs yet has an EMPTY folder that the glob still finds:
#   ERROR building tracker: [Errno 2] No such file or directory:
#   '...\\scene_001\\embryo_tracking\\tracking_RoIs.json'
# Observed 2026-09-04 with ROIs drawn for scene_000 only.
# ====================================================================

def _two_positions(panel_obj, tmp_path):
    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj._on_capture()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path)
    for name in ("scene_000", "scene_001"):
        (tmp_path / name / "embryo_tracking").mkdir(parents=True)
    return tmp_path


def _write_rois(tmp_path, name):
    (tmp_path / name / "embryo_tracking"
     / "tracking_RoIs.json").write_text("{}")


def test_REGRESSION_empty_roi_folder_is_not_mistaken_for_ready(
        panel_obj, tmp_path):
    """The folder existing is not the same as ROIs existing — that
    conflation is what crashed the tracker build."""
    _two_positions(panel_obj, tmp_path)
    state = panel_obj.roi_status()
    assert state == {"scene_000": False, "scene_001": False}

    _write_rois(tmp_path, "scene_000")
    state = panel_obj.roi_status()
    assert state == {"scene_000": True, "scene_001": False}


def test_require_all_rois_defaults_on(panel_obj):
    """Attaching with a partial set leaves the rest untracked for the
    whole run, since the ROI watcher only watches folders known at
    attach time."""
    assert panel_obj.require_all_rois.value is True


def test_check_rois_names_the_positions_still_missing(panel_obj,
                                                      tmp_path):
    _two_positions(panel_obj, tmp_path)
    _write_rois(tmp_path, "scene_000")
    panel_obj._refresh_roi_state()
    text = panel_obj.roi_state.object
    assert "1/2" in text
    assert "scene_001" in text


def test_check_rois_reports_full_coverage(panel_obj, tmp_path):
    _two_positions(panel_obj, tmp_path)
    _write_rois(tmp_path, "scene_000")
    _write_rois(tmp_path, "scene_001")
    panel_obj._refresh_roi_state()
    assert "2/2" in panel_obj.roi_state.object


def test_start_still_allowed_with_partial_rois(panel_obj, tmp_path):
    """Partial coverage must not block starting — acquisition runs and
    tracking waits."""
    _two_positions(panel_obj, tmp_path)
    _write_rois(tmp_path, "scene_000")
    assert panel_obj._validate_run() is None


# ====================================================================
# Tracker parameters
#
# tracking_config.yaml does NOT contain `serverkit`; zeiss_panel_app.py
# injects it from a checkbox and MultiRoIBaseTracker requires it
# positionally. Omitting it failed only at tracker construction, after
# a full acquisition had already run:
#   ERROR MultiRoIBaseTracker.init() missing 1 required positional
#   argument: 'serverkit'
# Observed 2026-09-04.
# ====================================================================

def test_serverkit_widget_exists_and_defaults_on(panel_obj):
    assert hasattr(panel_obj, "serverkit")
    assert panel_obj.serverkit.value is True


def test_REGRESSION_serverkit_is_injected_into_roi_tracker_params(
        panel_obj, monkeypatch, tmp_path):
    """The YAML lacks serverkit, so the panel must add it."""
    import interactive_tools.mtb_setup as m

    if m.TrackingRunner is None or m.get_pos_config is None:
        pytest.skip("tracker dependencies unavailable")

    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(m, "TrackingRunner", fake_runner)
    monkeypatch.setattr(
        m, "get_pos_config",
        lambda root, log_dir_name, position_name=None: {
            position_name or "scene_000": {
                "log_dir": str(tmp_path / "scene_000"
                               / "embryo_tracking"),
                "RoIs": [], "detection": False,
                "tracking_mode": "x", "scaling_factor": 1,
                "blur_factor": 0, "grid_size": 1,
                "mask_kernel_size": 1,
            }
        },
    )

    panel_obj.motion = FakeMotion()
    panel_obj.mmc = object()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path)
    roi_dir = tmp_path / "scene_000" / "embryo_tracking"
    roi_dir.mkdir(parents=True)
    (roi_dir / "tracking_RoIs.json").write_text("{}")

    class FakeRunnerBackend:
        positions_config = {}
        pos_names = []

        def refresh_filename(self, name):
            pass

    panel_obj._runner = FakeRunnerBackend()
    panel_obj.serverkit.value = True
    panel_obj._build_tracker(str(tmp_path))

    roi_params = captured.get("roi_tracker_params", {})
    assert "serverkit" in roi_params, (
        "serverkit missing — tracker construction will fail"
    )
    assert roi_params["serverkit"] is True


def test_serverkit_checkbox_value_is_respected(panel_obj, monkeypatch,
                                               tmp_path):
    import interactive_tools.mtb_setup as m

    if m.TrackingRunner is None:
        pytest.skip("tracker dependencies unavailable")

    captured = {}
    monkeypatch.setattr(
        m, "TrackingRunner",
        lambda **kw: captured.update(kw) or object(),
    )
    monkeypatch.setattr(
        m, "get_pos_config",
        lambda root, log_dir_name, position_name=None: {
            position_name or "scene_000": {"log_dir": "x"}
        },
    )

    panel_obj.motion = FakeMotion()
    panel_obj._on_capture()
    panel_obj.outdir.value = str(tmp_path)
    roi_dir = tmp_path / "scene_000" / "embryo_tracking"
    roi_dir.mkdir(parents=True)
    (roi_dir / "tracking_RoIs.json").write_text("{}")

    class FakeRunnerBackend:
        positions_config = {}
        pos_names = []

        def refresh_filename(self, name):
            pass

    panel_obj._runner = FakeRunnerBackend()
    panel_obj.serverkit.value = False
    panel_obj._build_tracker(str(tmp_path))
    assert captured["roi_tracker_params"]["serverkit"] is False


def test_requirements_declare_the_tracker_dependencies():
    """A fresh install must not get all the way to tracker
    construction before discovering torch is missing."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    text = (root / "requirements.txt").read_text().lower()
    for pkg in ("torch", "scipy", "tifffile", "watchdog"):
        assert pkg in text, f"{pkg} missing from requirements.txt"


# ------------------------------------------------- layout / light path

def test_layout_is_tabbed_in_workflow_order(panel_obj):
    import panel as pn

    lay = panel_obj.layout()
    tabs = []

    def walk(o):
        if isinstance(o, pn.Tabs):
            tabs.append(list(o._names))
        for c in getattr(o, "objects", []) or []:
            walk(c)

    walk(lay)
    assert tabs, "layout is not tabbed"
    assert tabs[0] == [
        "Stage", "Optics", "Channels", "Acquisition", "Run",
    ]


def test_REGRESSION_camera_is_outside_the_tabs_always_visible():
    """The image must stay visible whatever tab is open — jogging the
    stage, setting the light path, editing channels. An earlier layout
    put the camera inside a tab, so it vanished during setup."""
    import panel as pn

    panel_obj = MTBSetupPanel()
    lay = panel_obj.layout()

    # Collect everything that lives INSIDE any Tabs container.
    inside_tabs = set()

    def collect(o, in_tabs):
        if in_tabs:
            inside_tabs.add(id(o))
        is_tabs = isinstance(o, pn.Tabs)
        for c in getattr(o, "objects", []) or []:
            collect(c, in_tabs or is_tabs)

    collect(lay, False)

    for name in ("preview", "btn_snap", "btn_live", "exposure",
                 "img_stats"):
        widget = getattr(panel_obj, name)
        assert id(widget) not in inside_tabs, (
            f"{name} is inside a tab — it would disappear when the "
            f"user switches tabs"
        )


def test_every_widget_still_reachable_after_the_reshuffle(panel_obj):
    """A tab rewrite must not orphan a control."""
    import panel as pn

    lay = panel_obj.layout()
    seen = set()

    def walk(obj):
        seen.add(id(obj))
        for attr in ("objects", "_names"):
            pass
        for child in getattr(obj, "objects", []) or []:
            walk(child)

    walk(lay)
    for name in ("btn_connect", "btn_run", "btn_stop", "btn_snap",
                 "btn_live", "btn_capture", "pos_table", "outdir",
                 "exposure", "pixel_xy", "serverkit", "zstack_on",
                 "btn_read_lightpath", "btn_read_objective",
                 "auto_track", "require_all_rois", "chan_table"):
        widget = getattr(panel_obj, name)
        assert id(widget) in seen, f"{name} is not in the layout"


def test_lightpath_starts_empty_and_needs_a_connection(panel_obj):
    assert panel_obj._lightpath_widgets == {}
    panel_obj._on_read_lightpath()
    assert "Connect the hardware" in panel_obj.status.object


def test_setting_a_lightpath_changer_without_connection_is_safe(
        panel_obj):
    panel_obj._set_lightpath("sideport", 1)
    assert "Connect the hardware" in panel_obj.status.object


def test_setting_a_lightpath_changer_with_no_selection_warns(
        panel_obj):
    panel_obj.motion = FakeMotion()
    panel_obj.session = object()
    panel_obj._set_lightpath("sideport", None)
    assert "Nothing selected" in panel_obj.status.object


def test_lightpath_roles_cover_the_transmitted_path():
    """Order matters for the UI: light travels shutter -> condenser ->
    objective -> reflector -> sideport."""
    from tracking_tools.microscope_interface.mtb import (
        LIGHT_PATH_ROLES,
    )
    assert LIGHT_PATH_ROLES == (
        "tl_shutter", "condenser", "objective", "reflector", "sideport",
    )


# ------------------------------------------------- LED button state

def test_led_label_is_unambiguous_in_both_states():
    """Colour alone is not enough — the glyph and the word read
    correctly in either theme and for colour-blind users."""
    from interactive_tools.mtb_setup import _led_label

    on = _led_label(475, True)
    off = _led_label(475, False)
    assert "475" in on and "475" in off
    assert "ON" in on and "OFF" in off
    assert on != off
    assert "●" in on and "○" in off


def test_led_label_covers_every_wavelength_on_this_colibri():
    from interactive_tools.mtb_setup import _led_label

    for wl in (385, 430, 475, 511, 567, 630):
        assert str(wl) in _led_label(wl, True)


# --------------------------------------------- preview performance

def test_preview_size_defaults_to_something_smaller_than_the_sensor(
        panel_obj):
    """Panel serialises every live frame to the browser, so sending a
    full 1200x1200 float array per update dominates the link."""
    assert panel_obj.preview_px.value < 1200
    assert 1200 in panel_obj.preview_px.options


def test_render_decimates_to_the_requested_size(panel_obj):
    """A frame far larger than the preview must be decimated, not sent
    whole."""
    img = np.random.default_rng(0).integers(
        11000, 14800, size=(1200, 1200), dtype=np.uint16
    )
    panel_obj.preview_px.value = 300
    panel_obj._render(img)
    shown = panel_obj._img_source.data["image"][0]
    assert max(shown.shape) <= 400, (
        f"preview was {shown.shape}, expected ~300 px"
    )


def test_render_does_not_upscale_a_small_frame(panel_obj):
    img = np.random.default_rng(0).integers(
        0, 255, size=(64, 64), dtype=np.uint16
    )
    panel_obj.preview_px.value = 600
    panel_obj._render(img)
    shown = panel_obj._img_source.data["image"][0]
    assert shown.shape == (64, 64)


def test_render_updates_in_place_rather_than_replacing_the_pane(
        panel_obj):
    """Mutating the ColumnDataSource is what avoids the white flash
    between live frames; swapping a PNG pane's bytes reloads the
    <img> and blanks it."""
    source = panel_obj._img_source
    img = np.random.default_rng(0).integers(
        0, 255, size=(32, 32), dtype=np.uint16
    )
    panel_obj._render(img)
    assert panel_obj._img_source is source, (
        "the data source was replaced instead of updated"
    )


def test_render_handles_a_zstack_by_projecting(panel_obj):
    stack = np.zeros((5, 64, 64), dtype=np.uint16)
    stack[2] = 5000
    panel_obj._render(stack)
    shown = panel_obj._img_source.data["image"][0]
    assert shown.ndim == 2
