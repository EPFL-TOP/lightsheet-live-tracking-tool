"""Tests for the MTB setup panel's non-GUI logic.

Constructs the real panel (Panel widgets work headless) but never
touches hardware. Covers the parts that would otherwise only be
exercised by clicking at the microscope: validation gates, position
bookkeeping, and the config handed to the backend.
"""
from __future__ import annotations

import json

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
