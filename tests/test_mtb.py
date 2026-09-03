"""Unit tests for the MTB motion layer.

Runs without pythonnet, without Zeiss software, and without a
microscope: fake components stand in for the .NET objects, and the
interface-cast helper is patched. Behaviours pinned here are the ones
verified against the real hardware on 2026-09-03.
"""
from __future__ import annotations

import pytest

from tracking_tools.microscope_interface import mtb
from tracking_tools.microscope_interface.mtb import (
    MODE_SYNCHRONOUS,
    MTBAxis,
    MTBError,
    MTBMotion,
    MTBSession,
)


class FakeContinual:
    """Stands in for a component cast to IMTBContinual."""

    def __init__(self, position=0.0, units=("µm",),
                 lo=-350000.0, hi=350000.0, step=0.25,
                 refuse=False, raise_on_set=False):
        self._pos = position
        self._units = list(units)
        self._lo, self._hi = lo, hi
        self._step = step
        self.refuse = refuse
        self.raise_on_set = raise_on_set
        self.set_calls = []

    # IMTBContinual surface
    def GetPositionUnitCount(self):
        return len(self._units)

    def GetPositionUnit(self, i):
        return self._units[i]

    def GetPosition(self, unit):
        return self._pos

    def GetMinPosition(self, unit):
        return self._lo

    def GetMaxPosition(self, unit):
        return self._hi

    def StepWidth(self, unit):
        return self._step

    def SetPosition(self, pos, unit, mode, timeout):
        self.set_calls.append((pos, unit, mode, timeout))
        if self.raise_on_set:
            raise RuntimeError("boom")
        if self.refuse:
            return False
        self._pos = pos
        return True


@pytest.fixture(autouse=True)
def no_cast(monkeypatch):
    """Bypass the .NET interface cast — fakes are already 'cast'."""
    monkeypatch.setattr(mtb, "_cast_to_continual", lambda c: c)


# --------------------------------------------------------------- axis

def test_axis_discovers_micrometre_unit():
    ax = MTBAxis(FakeContinual(units=("µm",)), "axis_x")
    assert ax.unit == "µm"


def test_axis_prefers_micrometre_when_several_units_offered():
    # The focus advertises both µm and nm; we must not pick nm.
    ax = MTBAxis(FakeContinual(units=("µm", "nm")), "focus")
    assert ax.unit == "µm"


def test_axis_falls_back_to_first_unit_when_no_micrometre():
    ax = MTBAxis(FakeContinual(units=("mm",)), "weird")
    assert ax.unit == "mm"


def test_axis_raises_when_no_units_advertised():
    with pytest.raises(MTBError, match="no position units"):
        MTBAxis(FakeContinual(units=()), "broken")


def test_axis_reports_position_limits_and_step():
    ax = MTBAxis(
        FakeContinual(position=123.456, lo=-10.0, hi=20.0, step=0.01),
        "focus",
    )
    assert ax.position == pytest.approx(123.456)
    assert ax.limits == (-10.0, 20.0)
    assert ax.step == pytest.approx(0.01)


def test_move_to_uses_synchronous_mode_and_timeout():
    fake = FakeContinual(position=0.0)
    ax = MTBAxis(fake, "axis_x", timeout_ms=4321)
    ax.move_to(50.0)
    pos, unit, mode, timeout = fake.set_calls[-1]
    assert pos == pytest.approx(50.0)
    assert mode == MODE_SYNCHRONOUS
    assert timeout == 4321


def test_move_by_is_relative_to_current_position():
    fake = FakeContinual(position=100.0)
    ax = MTBAxis(fake, "axis_y")
    ax.move_by(-25.0)
    assert fake.set_calls[-1][0] == pytest.approx(75.0)
    assert ax.position == pytest.approx(75.0)


def test_move_to_clamps_to_limits():
    # The piezo's real range is 0..500 µm; overshooting must clamp
    # rather than command an out-of-range target.
    fake = FakeContinual(position=250.0, lo=0.0, hi=500.0)
    ax = MTBAxis(fake, "piezo")
    ax.move_to(900.0)
    assert fake.set_calls[-1][0] == pytest.approx(500.0)
    ax.move_to(-900.0)
    assert fake.set_calls[-1][0] == pytest.approx(0.0)


def test_move_to_can_opt_out_of_clamping():
    fake = FakeContinual(position=250.0, lo=0.0, hi=500.0)
    ax = MTBAxis(fake, "piezo")
    ax.move_to(900.0, clamp=False)
    assert fake.set_calls[-1][0] == pytest.approx(900.0)


def test_refused_move_raises_with_df2_hint():
    # The real focus returned False (a refusal, not an error) while
    # Definite Focus 2 held the axis.
    fake = FakeContinual(refuse=True)
    ax = MTBAxis(fake, "focus")
    with pytest.raises(MTBError, match="Definite Focus 2"):
        ax.move_to(10.0)


def test_raising_move_is_wrapped_in_mtberror():
    ax = MTBAxis(FakeContinual(raise_on_set=True), "axis_x")
    with pytest.raises(MTBError, match="raised"):
        ax.move_to(10.0)


# ------------------------------------------------------------ session

class FakeRoot:
    def __init__(self, comps):
        self._comps = comps

    def GetComponent(self, mtb_id):
        return self._comps.get(mtb_id)


def _session_with(comps):
    s = MTBSession()
    s._root = FakeRoot(comps)
    s.client_id = "fake-client"
    return s


def test_component_resolves_role_names_to_mtb_ids():
    comp = FakeContinual()
    s = _session_with({"MTBStageAxisX": comp})
    assert s.component("axis_x") is comp
    assert s.component("MTBStageAxisX") is comp


def test_component_raises_when_absent():
    s = _session_with({})
    with pytest.raises(MTBError, match="absent"):
        s.component("axis_x")


def test_component_requires_connection():
    s = MTBSession()
    with pytest.raises(MTBError, match="not connected"):
        s.component("axis_x")


def test_axis_wrappers_are_cached():
    s = _session_with({"MTBStageAxisX": FakeContinual()})
    assert s.axis("axis_x") is s.axis("axis_x")


# ------------------------------------------------------------- motion

def _motion(z_axis="piezo"):
    comps = {
        "MTBStageAxisX": FakeContinual(position=10.0),
        "MTBStageAxisY": FakeContinual(position=20.0),
        "MTBPiezoFocusCan": FakeContinual(
            position=250.0, lo=0.0, hi=500.0, step=0.01),
        "MTBFocus": FakeContinual(
            position=5.0, lo=-14000.0, hi=14000.0, step=0.01),
    }
    return MTBMotion(_session_with(comps), z_axis=z_axis), comps


def test_motion_defaults_to_piezo_for_z():
    m, _ = _motion()
    assert m.z_role == "piezo"
    assert m.get_xyz() == pytest.approx((10.0, 20.0, 250.0))


def test_motion_can_use_motorized_focus_for_z():
    m, _ = _motion(z_axis="focus")
    assert m.get_xyz() == pytest.approx((10.0, 20.0, 5.0))


def test_motion_rejects_unknown_z_axis():
    with pytest.raises(ValueError, match="piezo"):
        _motion(z_axis="nonsense")


def test_move_to_leaves_none_axes_untouched():
    m, comps = _motion()
    m.move_to(x=99.0)
    assert comps["MTBStageAxisX"].set_calls
    assert not comps["MTBStageAxisY"].set_calls
    assert not comps["MTBPiezoFocusCan"].set_calls


def test_move_by_skips_zero_deltas_entirely():
    # A zero-drift correction must not command a pointless move; the
    # Micro-Manager backend had a bug in this area, so pin it here.
    m, comps = _motion()
    m.move_by(dx=0.0, dy=5.0, dz=0.0)
    assert not comps["MTBStageAxisX"].set_calls
    assert comps["MTBStageAxisY"].set_calls
    assert not comps["MTBPiezoFocusCan"].set_calls


def test_move_by_applies_all_nonzero_deltas():
    m, comps = _motion()
    m.move_by(dx=1.0, dy=2.0, dz=3.0)
    assert comps["MTBStageAxisX"].set_calls[-1][0] == pytest.approx(11.0)
    assert comps["MTBStageAxisY"].set_calls[-1][0] == pytest.approx(22.0)
    assert comps["MTBPiezoFocusCan"].set_calls[-1][0] == pytest.approx(253.0)


def test_describe_mentions_the_active_z_axis():
    m, _ = _motion()
    text = m.describe()
    assert "piezo" in text
    assert "axis_x" in text
