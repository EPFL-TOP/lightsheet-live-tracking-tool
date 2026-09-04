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
                 refuse=False, raise_on_set=False,
                 typ_dev=0.0, max_dev=0.0):
        self._pos = position
        self._units = list(units)
        self._lo, self._hi = lo, hi
        self._step = step
        self._typ_dev = typ_dev
        self._max_dev = max_dev
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

    def TypicalDeviation(self, unit):
        return self._typ_dev

    def MaxDeviation(self, unit):
        return self._max_dev

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


# ====================================================================
# Single-session-per-process constraint
#
# MTB tears down its realtime subsystem on Logout and cannot rebuild
# it: a second Login fails with
#   MTBException: MTB could not be initialized
#   ConnectToRtSystem(): OpenRtNet("localhost", 1966, ...)
#   'No such interface supported' (0x80000004)
# Observed on the real microscope 2026-09-04, when a tool read the
# stage through its own session, logged out, and the backend's Login
# then failed. These tests pin the mitigation.
# ====================================================================

@pytest.fixture
def reset_shared(monkeypatch):
    """Isolate the module-level session singleton per test."""
    monkeypatch.setattr(mtb, "_shared_session", None)
    monkeypatch.setattr(mtb, "_logout_happened", False)


class FakeConn:
    """Minimal stand-in for ZEISS.MTB.Api.MTBConnection."""

    def __init__(self):
        self.logouts = []

    def Logout(self, client_id):
        self.logouts.append(client_id)


def _fake_connect(monkeypatch, calls):
    """Make MTBSession.connect() succeed without any .NET.

    Note this bypasses the real connect(), so the _logout_happened
    guard inside it is not exercised here — the refusal test calls the
    real connect() for that.
    """
    def connect(self):
        calls.append(self)
        self._conn = FakeConn()
        self._root = FakeRoot({})
        self.client_id = f"client-{len(calls)}"
        return self
    monkeypatch.setattr(MTBSession, "connect", connect)


def test_REGRESSION_shared_returns_one_session(reset_shared,
                                               monkeypatch):
    calls = []
    _fake_connect(monkeypatch, calls)
    a = MTBSession.shared()
    b = MTBSession.shared()
    assert a is b
    assert len(calls) == 1, "shared() must Login exactly once"


def test_REGRESSION_second_connect_after_logout_is_refused(
        reset_shared, monkeypatch):
    """The real failure mode: connect, logout, connect again.

    Deliberately calls the REAL connect() for the second attempt so
    the guard is what refuses it, rather than the fake.
    """
    calls = []
    _fake_connect(monkeypatch, calls)

    s = MTBSession()
    s.connect()
    s.disconnect()
    assert s._conn is None
    assert mtb._logout_happened, "disconnect must mark the process"

    # Restore the real connect() — the guard lives inside it.
    monkeypatch.undo()
    monkeypatch.setattr(mtb, "_logout_happened", True)
    with pytest.raises(MTBError, match="cannot be re-initialized"):
        MTBSession().connect()


def test_context_manager_does_not_log_out(reset_shared, monkeypatch):
    """__exit__ must not burn the process's one session."""
    calls = []
    _fake_connect(monkeypatch, calls)
    with MTBSession() as s:
        assert s.is_connected
    # Still usable, and a later connect is still permitted.
    assert not mtb._logout_happened


def test_is_connected_reflects_state(reset_shared, monkeypatch):
    calls = []
    _fake_connect(monkeypatch, calls)
    s = MTBSession()
    assert not s.is_connected
    s.connect()
    assert s.is_connected


# ====================================================================
# Enum marshalling
#
# pythonnet >= 3.0 refuses implicit int -> Enum conversion. Passing the
# numeric MTBCmdSetModes value straight to SetPosition failed on the
# real microscope with:
#   System.ArgumentException: since Python.NET 3.0 int can not be
#   converted to Enum implicitly. Use Enum(int_value)
#   in method Boolean SetPosition(Double, String, MTBCmdSetModes, Int32)
# Observed 2026-09-04.
# ====================================================================

def test_resolve_mode_known_names():
    assert mtb.resolve_mode("Default") == mtb.MODE_DEFAULT
    assert mtb.resolve_mode("Synchronous") == mtb.MODE_SYNCHRONOUS
    assert mtb.resolve_mode("Relative") == mtb.MODE_RELATIVE


def test_resolve_mode_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown MTBCmdSetModes"):
        mtb.resolve_mode("Teleport")


def test_resolve_mode_prefers_the_dotnet_enum_member(monkeypatch):
    """When the Zeiss assembly is present, the real member must win
    over the integer fallback."""
    sentinel = object()

    class FakeModes:
        Synchronous = sentinel

    import sys
    import types
    fake_mod = types.ModuleType("ZEISS.MTB.Api")
    fake_mod.MTBCmdSetModes = FakeModes
    monkeypatch.setitem(sys.modules, "ZEISS", types.ModuleType("ZEISS"))
    monkeypatch.setitem(sys.modules, "ZEISS.MTB",
                        types.ModuleType("ZEISS.MTB"))
    monkeypatch.setitem(sys.modules, "ZEISS.MTB.Api", fake_mod)
    monkeypatch.setattr(mtb, "_mode_cache", {})

    assert mtb.resolve_mode("Synchronous") is sentinel


def test_REGRESSION_move_to_resolves_mode_via_resolve_mode(monkeypatch):
    """move_to must route the mode through resolve_mode(), never hand a
    bare int to .NET."""
    seen = []

    def spy(name="Synchronous"):
        seen.append(name)
        return f"<enum {name}>"

    monkeypatch.setattr(mtb, "resolve_mode", spy)
    fake = FakeContinual(position=0.0)
    ax = MTBAxis(fake, "axis_x")
    ax.move_to(10.0)

    assert seen == ["Synchronous"], "mode was not resolved"
    assert fake.set_calls[-1][2] == "<enum Synchronous>", (
        "a raw int reached SetPosition — pythonnet >=3.0 rejects that"
    )


def test_move_to_accepts_an_alternative_mode_name(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mtb, "resolve_mode",
        lambda name="Synchronous": seen.append(name) or f"<{name}>",
    )
    ax = MTBAxis(FakeContinual(), "axis_x")
    ax.move_to(1.0, mode_name="Fast")
    assert seen == ["Fast"]


# ====================================================================
# No-op moves
#
# MTB returns False from SetPosition when asked to move somewhere the
# axis already is. That is "nothing to do", not a refusal — and it
# happens constantly, because a tracking run with zero accumulated
# drift targets exactly the current position. The real microscope
# failed with
#   axis_x: SetPosition(3.508 µm) was refused
# where 3.508 was the live position. Observed 2026-09-04.
# ====================================================================

def test_REGRESSION_target_equal_to_current_position_is_a_noop():
    fake = FakeContinual(position=3.508, step=0.25)
    ax = MTBAxis(fake, "axis_x")
    landed = ax.move_to(3.508)
    assert landed == pytest.approx(3.508)
    assert not fake.set_calls, (
        "commanded a zero-distance move; MTB returns False for those"
    )


def test_REGRESSION_target_within_one_step_is_a_noop():
    # XY resolves 0.25 µm, so 0.1 µm away is unreachable anyway.
    fake = FakeContinual(position=100.0, step=0.25)
    ax = MTBAxis(fake, "axis_x")
    ax.move_to(100.1)
    assert not fake.set_calls


def test_move_beyond_one_step_is_commanded():
    fake = FakeContinual(position=100.0, step=0.25)
    ax = MTBAxis(fake, "axis_x")
    ax.move_to(100.5)
    assert fake.set_calls, "a resolvable move must still be commanded"
    assert fake.set_calls[-1][0] == pytest.approx(100.5)


def test_false_return_is_accepted_when_the_axis_did_arrive():
    """Some axes report False yet land correctly; trust the readback."""

    class ArrivesButReportsFalse(FakeContinual):
        def SetPosition(self, pos, unit, mode, timeout):
            self.set_calls.append((pos, unit, mode, timeout))
            self._pos = pos      # it really moved
            return False         # ...but says otherwise

    fake = ArrivesButReportsFalse(position=0.0, step=0.01)
    ax = MTBAxis(fake, "piezo")
    assert ax.move_to(5.0) == pytest.approx(5.0)


def test_false_return_still_raises_when_the_axis_did_not_move():
    fake = FakeContinual(position=0.0, step=0.01, refuse=True)
    ax = MTBAxis(fake, "focus")
    with pytest.raises(MTBError, match="still at"):
        ax.move_to(5.0)


def test_refusal_message_lists_plausible_causes():
    fake = FakeContinual(position=0.0, step=0.01, refuse=True)
    ax = MTBAxis(fake, "axis_x")
    with pytest.raises(MTBError) as err:
        ax.move_to(5.0)
    msg = str(err.value)
    # Must not blame DF2 unconditionally — this fired on axis_x.
    assert "emergency stop" in msg or "joystick" in msg
    assert "Definite Focus 2" in msg


def test_explicit_tolerance_overrides_step_width():
    fake = FakeContinual(position=100.0, step=0.25)
    ax = MTBAxis(fake, "axis_x")
    ax.move_to(102.0, tolerance=5.0)
    assert not fake.set_calls, "explicit tolerance should suppress it"


# ====================================================================
# Servoing axes and arrival tolerance
#
# A closed-loop piezo resolves 0.01 um but dithers well above that, so
# using step width as the arrival tolerance made normal settling look
# like a refusal. The real microscope failed with
#   piezo: SetPosition(250.000 um) was refused and the axis is still
#          at 250.190
# while XY corrections in the same run succeeded exactly.
# Observed 2026-09-04. MTB declares the real figures via
# TypicalDeviation() / MaxDeviation().
# ====================================================================

def test_arrival_tolerance_uses_typical_deviation_when_larger():
    fake = FakeContinual(step=0.01, typ_dev=0.15)
    ax = MTBAxis(fake, "piezo")
    assert ax.arrival_tolerance == pytest.approx(0.15)


def test_arrival_tolerance_falls_back_to_step_width():
    fake = FakeContinual(step=0.25, typ_dev=0.0)
    ax = MTBAxis(fake, "axis_x")
    assert ax.arrival_tolerance == pytest.approx(0.25)


def test_arrival_tolerance_is_never_zero():
    fake = FakeContinual(step=0.0, typ_dev=0.0)
    ax = MTBAxis(fake, "odd")
    assert ax.arrival_tolerance > 0


def test_REGRESSION_servoing_piezo_within_deviation_is_not_a_failure():
    """The exact failing case: commanded 250.000, sits at 250.190,
    step is only 0.010 — but the axis declares a larger deviation."""

    class Servoing(FakeContinual):
        def SetPosition(self, pos, unit, mode, timeout):
            self.set_calls.append((pos, unit, mode, timeout))
            return False          # refuses, and does not move

    fake = Servoing(position=250.190, step=0.01,
                    typ_dev=0.10, max_dev=0.25)
    ax = MTBAxis(fake, "piezo")
    landed = ax.move_to(250.000)
    assert landed == pytest.approx(250.190)


def test_deviation_beyond_max_still_raises():
    class WayOff(FakeContinual):
        def SetPosition(self, pos, unit, mode, timeout):
            self.set_calls.append((pos, unit, mode, timeout))
            return False

    fake = WayOff(position=280.0, step=0.01,
                  typ_dev=0.10, max_dev=0.25)
    ax = MTBAxis(fake, "piezo")
    with pytest.raises(MTBError, match="still at"):
        ax.move_to(250.0)


def test_deviations_default_to_zero_when_unsupported():
    """Axes whose adapter lacks the calls must not break."""

    class NoDeviations(FakeContinual):
        def TypicalDeviation(self, unit):
            raise RuntimeError("unsupported")

        def MaxDeviation(self, unit):
            raise RuntimeError("unsupported")

    ax = MTBAxis(NoDeviations(step=0.25), "axis_x")
    assert ax.typical_deviation == 0.0
    assert ax.max_deviation == 0.0
    assert ax.arrival_tolerance == pytest.approx(0.25)


def test_describe_reports_deviations_and_tolerance():
    m, _ = _motion()
    text = m.describe()
    assert "dev typ" in text and "tol" in text


# ------------------------------------------------ objective / pixel size

def test_parse_magnification_across_zeiss_naming():
    from tracking_tools.microscope_interface.mtb import (
        parse_magnification,
    )
    cases = {
        "Plan-Apochromat 20x/0.8 M27": 20.0,
        "LD C-Apochromat 40x/1.1 W Korr": 40.0,
        "EC Plan-Neofluar 10x/0.3 Ph1": 10.0,
        "Fluar 5x/0.25": 5.0,
        "63x/1.4 Oil": 63.0,
        "Plan-Apochromat 1.25X/0.04": 1.25,
        "objective 100 X oil": 100.0,
    }
    for text, expected in cases.items():
        assert parse_magnification(text) == pytest.approx(expected), text


def test_parse_magnification_returns_none_when_absent():
    from tracking_tools.microscope_interface.mtb import (
        parse_magnification,
    )
    for text in ("", None, "no numbers here", "Plan-Apochromat"):
        assert parse_magnification(text) is None


def test_sample_pixel_size_for_prime95b():
    """Prime 95B has 11 um pixels; this scope reports a 1x CSU adapter."""
    from tracking_tools.microscope_interface.mtb import (
        sample_pixel_size_um,
    )
    assert sample_pixel_size_um(11.0, 20.0) == pytest.approx(0.55)
    assert sample_pixel_size_um(11.0, 40.0) == pytest.approx(0.275)
    assert sample_pixel_size_um(11.0, 63.0) == pytest.approx(
        0.174603, abs=1e-5
    )


def test_sample_pixel_size_accounts_for_the_adapter():
    from tracking_tools.microscope_interface.mtb import (
        sample_pixel_size_um,
    )
    assert sample_pixel_size_um(11.0, 20.0, 0.5) == pytest.approx(1.1)
    assert sample_pixel_size_um(11.0, 20.0, 2.0) == pytest.approx(0.275)


def test_sample_pixel_size_rejects_nonsense_magnification():
    from tracking_tools.microscope_interface.mtb import (
        sample_pixel_size_um,
    )
    with pytest.raises(ValueError, match="must be positive"):
        sample_pixel_size_um(11.0, 0.0)


class FakeObjectiveElement:
    def __init__(self, name=None, mag=None, na=None):
        if name is not None:
            self.Name = name
        if mag is not None:
            self.Magnification = mag
        if na is not None:
            self.Aperture = na


class FakeChanger:
    def __init__(self, position=3, element=None):
        self.Position = position
        self._element = element

    def GetElement(self, pos):
        return self._element


def _objective(changer):
    from tracking_tools.microscope_interface.mtb import MTBObjective
    obj = MTBObjective.__new__(MTBObjective)
    obj.label = "objective"
    obj._raw = changer
    obj._changer = changer
    return obj


def test_objective_prefers_the_typed_magnification():
    el = FakeObjectiveElement(name="Plan-Apochromat 20x/0.8", mag=20.0)
    obj = _objective(FakeChanger(element=el))
    assert obj.magnification == pytest.approx(20.0)
    assert obj.position == 3


def test_objective_falls_back_to_parsing_the_name():
    """No typed Magnification member — the name must carry it."""
    el = FakeObjectiveElement(name="LD C-Apochromat 40x/1.1 W")
    obj = _objective(FakeChanger(element=el))
    assert obj.magnification == pytest.approx(40.0)


def test_objective_reports_none_when_nothing_is_readable():
    obj = _objective(FakeChanger(element=FakeObjectiveElement()))
    assert obj.magnification is None
    assert obj.name is None


def test_objective_probe_describes_what_it_found():
    el = FakeObjectiveElement(name="Fluar 5x/0.25", mag=5.0, na=0.25)
    info = _objective(FakeChanger(element=el)).probe()
    assert info["magnification"] == pytest.approx(5.0)
    assert info["aperture"] == pytest.approx(0.25)
    assert info["element_found"] is True
    assert "element_attrs" in info


def test_objective_survives_a_changer_without_elements():
    class Bare:
        Position = 1
    obj = _objective(Bare())
    assert obj.magnification is None
    assert obj.probe()["element_found"] is False
