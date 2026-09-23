"""MTB smoke test — connect, discover, read positions, optionally move.

The MTB equivalent of tools/hw_smoke_test.py. Because the Axio Observer
7 carries CAN29 over USB into CZCanSrv, Micro-Manager's serial
ZeissCAN29 adapter cannot reach it; MTB is the supported route. See
docs/bringup_runbook.md for the full story.

Design notes:
- Reading uses IMTBContinual.GetPosition(unit) on the per-axis
  components (MTBStageAxisX/Y), which returns a plain Double. The
  IMTBStage.GetPosition(out x, out y, unit) overload uses .NET out
  params, whose pythonnet marshalling is fiddlier; we avoid it.
- Writing uses IMTBStage.SetPosition(x, y, unit, mode, timeout) so XY
  moves are coordinated rather than two independent axis moves.
- Unit strings are DISCOVERED via GetPositionUnitCount/GetPositionUnit
  rather than hard-coded, because "um" vs "µm" varies.
- --move is opt-in and defaults to a tiny delta. Nothing moves unless
  you ask for it.

Usage:
    # Safe: connect, enumerate, read positions and limits. No motion.
    python tools/mtb_smoke_test.py

    # Also perform a small relative move and verify readback
    python tools/mtb_smoke_test.py --move --dxy 20 --dz 2

    # Non-default DLL location
    python tools/mtb_smoke_test.py --dll "C:\\path\\to\\MTBApi.dll"

Exit codes:
    0  all requested checks passed
    1  a check failed
    2  pythonnet missing
    3  MTBApi.dll not found / not loadable
    4  connection/login failed
"""
from __future__ import annotations

import argparse
import os
import sys

# Running from tools/ leaves the repo root off sys.path, so importing
# tracking_tools for the wrapper probe fails.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_DLL = (
    r"C:\Program Files\Carl Zeiss\MTB 2011 - 2.12.0.7\MTBApi\MTBApi.dll"
)

DEFAULT_CONFIG_XML = (
    r"C:\ProgramData\Carl Zeiss\MTB2011\2.12.0.7"
    r"\Active Configuration.xml"
)

# MTBIds confirmed present on this machine (probe_mtb.py --connect,
# 2026-09-03). Role -> (MTBId, description).
COMPONENTS = {
    "stage":      ("MTBStage", "Motorized Stage (SMC2009)"),
    "axis_x":     ("MTBStageAxisX", "x-Axis"),
    "axis_y":     ("MTBStageAxisY", "y-Axis"),
    "focus":      ("MTBFocus", "Motorized Focus"),
    "piezo":      ("MTBPiezoFocusCan", "Piezo Focus (WSB 500um)"),
    "df2":        ("MTBFocusStabilizer2", "Definite Focus 2"),
    "leds":       ("MTBFLLEDController", "Colibri 5/7"),
    "objective":  ("MTBObjectiveChanger", "6x Motorized Nosepiece"),
    "reflector":  ("MTBReflectorChanger", "6x Motorized Reflector"),
}

_failures: list[str] = []


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _hdr(title: str) -> None:
    _log()
    _log("=" * 70)
    _log(title)
    _log("=" * 70)


def _fail(msg: str) -> None:
    _failures.append(msg)
    _log(f"  FAIL: {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Connect to Zeiss MTB and exercise the hardware.",
    )
    p.add_argument("--dll", default=DEFAULT_DLL,
                   help=f"Path to MTBApi.dll (default: {DEFAULT_DLL})")
    p.add_argument("--locale", default="en",
                   help="Locale passed to MTB Login (default: en).")
    p.add_argument("--move", action="store_true",
                   help="Perform small test moves. Off by default — "
                        "without this the script only reads.")
    p.add_argument("--dxy", type=float, default=20.0,
                   help="XY test move in um (default: 20).")
    p.add_argument("--dz", type=float, default=2.0,
                   help="Z test move in um (default: 2).")
    p.add_argument("--timeout-ms", type=int, default=10000,
                   help="MTB SetPosition timeout in ms (default: 10000).")
    p.add_argument("--tol-um", type=float, default=3.0,
                   help="Acceptable position error in um (default: 3).")
    p.add_argument("--objective", action="store_true",
                   help="Dump everything readable about the nosepiece, "
                        "so the magnification API can be identified.")
    p.add_argument("--df2", action="store_true",
                   help="Dump the Definite Focus 2 interface so "
                        "autofocus can be wired up.")
    p.add_argument("--colibri", action="store_true",
                   help="Dump the Colibri LED controller exhaustively "
                        "— which interfaces it and each LED cast to, "
                        "the operation-mode enums, and the trigger "
                        "buffer where brightness lives.")
    p.add_argument("--lightpath", action="store_true",
                   help="Report shutters, reflector, sideport and lamps "
                        "— i.e. why no light may be reaching the "
                        "camera.")
    p.add_argument("--enumerate", dest="enumerate_all",
                   action="store_true",
                   help="Walk the whole device tree and list EVERY "
                        "component, not just the ids we thought to "
                        "guess. Use this to settle whether a "
                        "motorized field diaphragm exists.")
    p.add_argument("--config-xml", default=DEFAULT_CONFIG_XML,
                   help="MTB Active Configuration.xml, dumped with "
                        f"--enumerate (default: {DEFAULT_CONFIG_XML}).")
    p.add_argument("--xml-only", action="store_true",
                   help="Dump Active Configuration.xml and exit. "
                        "Needs no connection and no hardware, so it is "
                        "safe to run mid-experiment.")
    return p.parse_args()


def load_api(dll_path: str):
    """Load MTBApi.dll; return (clr, assembly)."""
    try:
        import clr
    except ImportError:
        _log("FAIL: pythonnet not installed.  pip install pythonnet")
        sys.exit(2)

    if not os.path.exists(dll_path):
        _log(f"FAIL: MTBApi.dll not found at:\n      {dll_path}")
        sys.exit(3)

    _log(f"Loading {dll_path}")
    asm = None
    try:
        asm = clr.AddReference(
            dll_path[:-4] if dll_path.endswith(".dll") else dll_path
        )
    except Exception:
        try:
            from System.Reflection import Assembly
            asm = Assembly.LoadFrom(dll_path)
        except Exception as e:
            _log(f"FAIL: could not load assembly: {type(e).__name__}: {e}")
            sys.exit(3)
    _log("Assembly loaded.")
    return clr, asm


def get_set_mode(asm):
    """Resolve MTBCmdSetModes.Synchronous, reporting what's available."""
    from System import Enum

    mode_type = None
    try:
        types = [t for t in asm.GetTypes() if t is not None]
    except Exception as e:
        types = [t for t in (getattr(e, "Types", None) or []) if t]
    for t in types:
        if t.Name == "MTBCmdSetModes":
            mode_type = t
            break

    if mode_type is None:
        _log("  WARN: MTBCmdSetModes enum not found; passing 0.")
        return 0

    names = list(Enum.GetNames(mode_type))
    _log(f"  MTBCmdSetModes values: {', '.join(names)}")

    # If it carries [Flags], modes can be OR-ed — e.g.
    # Synchronous | Relative would give us a blocking relative move,
    # which is exactly what relative_move() wants.
    try:
        from System import FlagsAttribute
        is_flags = mode_type.IsDefined(FlagsAttribute, False)
    except Exception:
        is_flags = None
    _log(f"  combinable ([Flags]): {is_flags}")
    try:
        pairs = [
            (n, int(Enum.Parse(mode_type, n))) for n in names
        ]
        _log("  numeric values: "
             + ", ".join(f"{n}={v}" for n, v in pairs))
    except Exception:
        pass

    # Synchronous means SetPosition blocks until the move completes,
    # which is what a closed tracking loop wants.
    for preferred in ("Synchronous", "Sync", "Default"):
        if preferred in names:
            val = Enum.Parse(mode_type, preferred)
            _log(f"  Using mode: {preferred}")
            return val
    val = Enum.Parse(mode_type, names[0])
    _log(f"  Using mode: {names[0]} (no Synchronous found)")
    return val


def connect(asm, locale: str):
    """Login to MTB. Returns (connection, client_id, root)."""
    from ZEISS.MTB.Api import MTBConnection

    conn = MTBConnection()
    # Login is Void Login(String culture, out String& ID). pythonnet
    # surfaces the out-param in the return; shape varies by version so
    # normalise defensively.
    res = conn.Login(locale, "")
    if isinstance(res, tuple):
        client_id = next(
            (v for v in reversed(res) if isinstance(v, str) and v), None
        )
    else:
        client_id = res if isinstance(res, str) else None
    if not client_id:
        _log(f"FAIL: Login returned no client id (got {res!r})")
        sys.exit(4)
    _log(f"  clientId = {client_id}")

    root = conn.GetRoot(client_id)
    if root is None:
        _log("FAIL: GetRoot returned None")
        sys.exit(4)
    _log(f"  root = {root}")
    return conn, client_id, root


def as_continual(comp, label: str):
    """Cast an MTB component to the interface exposing positions.

    root.GetComponent() hands back concrete classes from the
    ZEISS.MTB.MicControl assembly (e.g. MTBCtrlStageAxisAquila). Their
    IMTBContinual members are explicit interface implementations, so
    calling GetPosition() on the raw object raises AttributeError —
    pythonnet only surfaces them after an interface cast.

    Returns (casted_object, interface_name) or (None, None).
    """
    candidates = []
    try:
        from ZEISS.MTB.Api import IMTBContinual
        candidates.append(IMTBContinual)
    except ImportError:
        pass
    for name in ("IMTBStageAxis", "IMTBAxis", "IMTBFocus"):
        try:
            mod = __import__("ZEISS.MTB.Api", fromlist=[name])
            candidates.append(getattr(mod, name))
        except Exception:
            pass

    for iface in candidates:
        try:
            cast = iface(comp)
            # Probe that the cast actually exposes what we need.
            cast.GetPositionUnitCount()
            return cast, iface.__name__
        except Exception:
            continue

    # Last resort: maybe this build does surface members directly.
    try:
        comp.GetPositionUnitCount()
        return comp, "<direct>"
    except Exception:
        return None, None


def discover_units(comp, label: str) -> list[str]:
    """Ask a component which position units it accepts."""
    units: list[str] = []
    try:
        n = comp.GetPositionUnitCount()
        for i in range(n):
            try:
                units.append(comp.GetPositionUnit(i))
            except Exception:
                pass
    except Exception as e:
        _log(f"  {label}: unit discovery failed "
             f"({type(e).__name__})")
    return units


def pick_um(units: list[str]) -> str | None:
    """Pick the micrometre unit string from a discovered list."""
    for u in units:
        if u and u.strip().lower() in ("um", "\u00b5m", "micron",
                                       "micrometer", "micrometre"):
            return u
    # Fall back to anything containing 'm' but not 'mm'-only
    for u in units:
        if u and "m" in u.lower() and u.lower() != "mm":
            return u
    return units[0] if units else None


def report_axis(raw, label: str) -> tuple[str | None, float | None, object]:
    """Print units, limits, step and current position for one axis."""
    _log(f"  {label}")
    comp, iface = as_continual(raw, label)
    if comp is None:
        _fail(f"{label}: could not cast to a position interface — "
              f"no IMTBContinual/IMTBStageAxis/IMTBAxis/IMTBFocus "
              f"cast exposed GetPositionUnitCount")
        return None, None, None
    _log(f"    cast via:        {iface}")

    units = discover_units(comp, label)
    unit = pick_um(units)
    _log(f"    units available: {units or '<none>'}")
    if unit is None:
        _fail(f"{label}: no usable position unit")
        return None, None, comp
    _log(f"    using unit:      {unit!r}")

    pos = None
    try:
        pos = comp.GetPosition(unit)
        _log(f"    position:        {pos:.3f} {unit}")
    except Exception as e:
        _fail(f"{label}: GetPosition failed: {type(e).__name__}: {e}")

    for meth, name in (
        ("GetMinPosition", "min"),
        ("GetMaxPosition", "max"),
        ("StepWidth", "step"),
    ):
        try:
            val = getattr(comp, meth)(unit)
            _log(f"    {name:9s}        {val:.3f} {unit}")
        except Exception:
            pass

    return unit, pos, comp


def _attempt_set(comp, label, target, unit, mode, timeout, client_id):
    """Try the SetPosition overloads until one is accepted.

    MTB exposes both SetPosition(pos, unit, mode, timeout) and
    SetPosition(clientID, pos, unit, mode, timeout). Some axes require
    the clientID form to establish write ownership — XY accepted the
    short form but the motorized focus returned False, which is a
    refusal rather than an error, so ownership is the prime suspect.
    """
    attempts = [
        ("no-clientID", lambda: comp.SetPosition(
            target, unit, mode, timeout)),
        ("no-clientID, no-timeout", lambda: comp.SetPosition(
            target, unit, mode)),
    ]
    if client_id:
        attempts += [
            ("with-clientID", lambda: comp.SetPosition(
                client_id, target, unit, mode, timeout)),
            ("with-clientID, no-timeout", lambda: comp.SetPosition(
                client_id, target, unit, mode)),
        ]

    for name, call in attempts:
        try:
            ok = call()
        except Exception as e:
            _log(f"    [{name}] raised {type(e).__name__}: {e}")
            continue
        if ok is False:
            _log(f"    [{name}] returned False (refused)")
            continue
        _log(f"    [{name}] accepted")
        return True, name
    return False, None


def test_move_axis(comp, label, unit, delta, mode, timeout, tol,
                   client_id=None) -> None:
    """Relative move on a single IMTBContinual axis, then move back."""
    try:
        before = comp.GetPosition(unit)
    except Exception as e:
        _fail(f"{label}: cannot read start position: {e}")
        return

    target = before + delta
    _log(f"  {label}: {before:.3f} -> {target:.3f} {unit}")
    ok, how = _attempt_set(
        comp, label, target, unit, mode, timeout, client_id
    )
    if not ok:
        _fail(f"{label}: every SetPosition overload refused the move. "
              f"If this is the motorized focus, Definite Focus 2 is "
              f"probably engaged and holding the axis — check DF2 "
              f"state, or drive the piezo instead.")
        return

    try:
        after = comp.GetPosition(unit)
    except Exception as e:
        _fail(f"{label}: cannot read position after move: {e}")
        return

    err = abs(after - target)
    _log(f"    landed at {after:.3f} {unit} (error {err:.3f})")
    if err > tol:
        _fail(f"{label}: move error {err:.3f} {unit} exceeds "
              f"tolerance {tol}")

    # Always try to restore the original position.
    try:
        _attempt_set(comp, label, before, unit, mode, timeout,
                     client_id)
        back = comp.GetPosition(unit)
        _log(f"    restored to {back:.3f} {unit}")
    except Exception as e:
        _fail(f"{label}: could not restore original position: {e}")


# Candidate MTBIds for anything that can gate light to the camera.
# Probed by name because the set varies per stand; absent ones are
# simply reported as absent.
_LIGHTPATH_IDS = [
    "MTBTLShutter", "MTBRLShutter", "MTBLampShutter",
    "MTBTLLamp", "MTBTLHalogenLamp", "MTBRLLamp",
    "MTBTLLampControl", "MTBFLLEDController", "MTBFLLEDShutter",
    "MTBSideportChanger", "MTBBaseportChanger",
    "MTBReflectorChanger", "MTBFilterChanger",
    "MTBCondenser", "MTBCondenserContrastChanger",
    "MTBApertureChanger", "MTBTubeLensChanger",
    "MTBOptovar", "MTBTLAperture", "MTBTLFieldStop",
    "MTBObservationChanger", "MTBCamera",
]


def _dump_df2(root, asm, comp) -> None:
    """Dump everything about the Definite Focus 2.

    DF2 holds focus against thermal drift in hardware, which over a
    multi-hour run beats correcting Z from images. The MM wiki
    describes a 'Focus Method' with 'Measure' (learn the current
    plane) and 'Apply' (return to it), and warns the Core timeout must
    be raised to ~25 s because Measure is slow — so expect long calls.

    Enumerate rather than assume: the same reflection scan settled the
    objective and the Colibri.
    """
    import ZEISS.MTB.Api as api

    if comp is None:
        _log("  MTBFocusStabilizer2 absent")
        return

    try:
        _log(f"  component: {comp.Name!r}")
    except Exception:
        pass

    try:
        types = [t for t in asm.GetTypes() if t is not None]
    except Exception as e:
        types = [t for t in (getattr(e, "Types", None) or []) if t]
    ifaces = [t for t in types
              if t.IsInterface and t.IsPublic
              and (t.Name or "").startswith("IMTB")]

    _log(f"  scanning {len(ifaces)} IMTB* interfaces")
    hits = []
    for t in sorted(ifaces, key=lambda x: x.Name):
        try:
            cast = getattr(api, t.Name)(comp)
        except Exception:
            continue
        if cast is None:
            continue
        members = sorted(
            a for a in dir(cast)
            if not a.startswith(("_", "get_", "set_", "add_",
                                 "remove_"))
            and a not in ("Equals", "GetHashCode", "GetType",
                          "ToString")
        )
        hits.append(t.Name)
        _log(f"    {t.Name}:")
        _log(f"        {members}")
        # Read anything that looks like state; skip callables.
        for a in members:
            if not any(k in a.lower() for k in
                       ("focus", "state", "mode", "offset", "stab",
                        "period", "active", "busy", "method",
                        "result", "surface")):
                continue
            try:
                v = getattr(cast, a)
                if not callable(v):
                    _log(f"            {a} = {v!r}")
            except Exception:
                pass
    _log(f"  castable to ({len(hits)}): {hits}")

    _log()
    _log("  Relevant enums:")
    from System import Enum
    for t in types:
        nm = t.Name or ""
        if t.IsEnum and any(k in nm for k in
                            ("AutoFocus", "FocusStabilizer",
                             "Focus")):
            try:
                _log(f"    {nm}: {list(Enum.GetNames(t))}")
            except Exception:
                pass

    _log()
    _log("  Method signatures on the stabilizer interfaces:")
    for t in sorted(ifaces, key=lambda x: x.Name):
        if "FocusStabilizer" not in (t.Name or "") and \
                "AutoFocus" not in (t.Name or ""):
            continue
        try:
            for m in sorted(t.GetMethods(), key=lambda x: x.Name):
                params = ", ".join(
                    f"{p.ParameterType.Name} {p.Name}"
                    for p in m.GetParameters()
                )
                _log(f"    {t.Name}.{m.Name}({params}) "
                     f"-> {m.ReturnType.Name}")
        except Exception:
            pass

    _log()
    _log("  What we need: a call to LEARN the current plane and one to")
    _log("  RETURN to it, plus a way to read whether it is holding.")
    _log("  Expect Measure to take seconds — raise any timeout.")


def _dump_colibri(root, asm, comp) -> None:
    """Find how to switch the Colibri on and set its brightness.

    Why this is needed: changing the reflector cube alone produces no
    fluorescence — without excitation light every "channel" is just
    the transmitted-light image through a different filter, which is
    exactly what was observed.

    Neither IMTBFluorescenceLEDControl nor IMTBFluorescenceLED exposes
    a brightness or on/off member. Brightness DOES appear in
      GetTriggerBufferEntry(entry, out Double[] brightness, unit,
                            out UInt32[] duration)
      SetTriggerBufferEntry(entry, Double[] brightness, unit,
                            UInt32[] duration)
    so the control path is probably SetOperationMode(...) plus a
    trigger-buffer write. Individual LEDs may also cast to
    IMTBContinual, which would give a plain intensity axis. This dumps
    every route so the answer stops being a guess.
    """
    import ZEISS.MTB.Api as api

    if comp is None:
        _log("  MTBFLLEDController absent")
        return

    # --- interfaces the controller casts to ---
    try:
        types = [t for t in asm.GetTypes() if t is not None]
    except Exception as e:
        types = [t for t in (getattr(e, "Types", None) or []) if t]
    ifaces = [t for t in types
              if t.IsInterface and t.IsPublic
              and (t.Name or "").startswith("IMTB")]

    def scan(obj, indent="    "):
        hits = []
        for t in sorted(ifaces, key=lambda x: x.Name):
            try:
                cast = getattr(api, t.Name)(obj)
            except Exception:
                continue
            if cast is None:
                continue
            members = sorted(
                a for a in dir(cast)
                if not a.startswith(("_", "get_", "set_", "add_",
                                     "remove_"))
                and a not in ("Equals", "GetHashCode", "GetType",
                              "ToString")
            )
            hits.append(t.Name)
            _log(f"{indent}{t.Name}: {members}")
            for a in members:
                if any(k in a.lower() for k in
                       ("bright", "intens", "onoff", "active", "power",
                        "wavelength", "state", "level")):
                    try:
                        v = getattr(cast, a)
                        if not callable(v):
                            _log(f"{indent}    {a} = {v!r}")
                    except Exception:
                        pass
        return hits

    _log(f"  controller: {comp.Name!r}")
    _log("  controller casts to:")
    scan(comp)

    # --- operation / trigger mode enums ---
    _log()
    _log("  Relevant enums:")
    from System import Enum
    for t in types:
        nm = t.Name or ""
        if t.IsEnum and "FluorescenceLED" in nm:
            try:
                _log(f"    {nm}: {list(Enum.GetNames(t))}")
            except Exception:
                pass

    # --- current operation mode ---
    _log()
    try:
        ctrl = api.IMTBFluorescenceLEDControl(comp)
        _log(f"  NumberOfLEDs = {ctrl.NumberOfLEDs}")
        try:
            res = ctrl.GetOperationMode(None, None)
            _log(f"  GetOperationMode -> {res!r}")
        except Exception as e:
            _log(f"  GetOperationMode failed: {type(e).__name__}: {e}")

        # --- each LED, and what IT casts to ---
        for i in range(int(ctrl.NumberOfLEDs)):
            _log()
            try:
                led = ctrl.LED[i]
            except Exception as e:
                _log(f"  LED[{i}] unavailable: {e}")
                continue
            wl = ""
            try:
                wl = f" wavelength={led.Wavelength}"
            except Exception:
                pass
            _log(f"  LED[{i}]{wl} casts to:")
            scan(led, indent="      ")
            try:
                name, lo, hi = led.GetExcitationFilter(None, None, None)
                _log(f"      excitation filter: {name!r} {lo}-{hi} nm")
            except Exception:
                try:
                    r = led.GetExcitationFilter("", 0, 0)
                    _log(f"      GetExcitationFilter -> {r!r}")
                except Exception as e:
                    _log(f"      GetExcitationFilter failed: "
                         f"{type(e).__name__}")

        # --- the trigger buffer, where brightness lives ---
        _log()
        _log("  Trigger buffer (brightness lives here):")
        try:
            ptr = ctrl.GetTriggerBufferPointer()
            _log(f"    GetTriggerBufferPointer() = {ptr}")
        except Exception as e:
            _log(f"    pointer failed: {type(e).__name__}: {e}")
        for entry in range(0, 4):
            try:
                r = ctrl.GetTriggerBufferEntry(entry, None, "%", None)
                _log(f"    entry {entry} -> {r!r}")
            except Exception as e:
                _log(f"    entry {entry} failed: {type(e).__name__}: "
                     f"{e}")
    except Exception as e:
        _log(f"  controller cast failed: {type(e).__name__}: {e}")

    _log()
    _log("  What we are looking for: a member that turns an LED ON and "
         "sets its intensity. Candidates in order of preference —")
    _log("    1. an LED that casts to IMTBContinual (plain intensity)")
    _log("    2. SetOperationMode(...) + SetTriggerBufferEntry(...)")
    _log("    3. an IMTBShutter/IMTBShutterState on the controller")


def _dump_lightpath(root) -> None:
    """Report every component that can stop light reaching the camera.

    An exposure-independent frame (mean and std identical from 1 ms to
    50 ms) means ZERO photons, not few: that is the sensor's dark
    offset plus read noise. So the question is not exposure but what is
    closed, misdirected, or switched off.
    """
    import ZEISS.MTB.Api as api

    def _read(obj, names):
        for n in names:
            try:
                v = getattr(obj, n)
            except Exception:
                continue
            if callable(v):
                continue
            return n, v
        return None, None

    found = 0
    for mtb_id in _LIGHTPATH_IDS:
        try:
            comp = root.GetComponent(mtb_id)
        except Exception as e:
            _log(f"  {mtb_id:32s} ERROR {type(e).__name__}")
            continue
        if comp is None:
            # Say so. Silence here is indistinguishable from "we never
            # looked", and that ambiguity is exactly what left the
            # question of a field diaphragm unanswered.
            _log(f"  {mtb_id:32s} absent")
            continue
        found += 1

        name = ""
        try:
            name = f"  ({comp.Name})"
        except Exception:
            pass
        _log(f"  {mtb_id:32s} PRESENT{name}")

        # Shutter open/closed, changer position, lamp on/off.
        for iface_name, fields in (
            ("IMTBShutterState", ("Open",)),
            ("IMTBChanger", ("Position", "MinPosition",
                             "MaxPosition")),
            ("IMTBContinual", ()),
            ("IMTBLamp", ("IsActive", "Intensity", "OnOff")),
            ("IMTBLampControl", ("IsActive", "Intensity")),
        ):
            try:
                cast = getattr(api, iface_name)(comp)
            except Exception:
                continue
            if cast is None:
                continue
            bits = []
            for f in fields:
                try:
                    v = getattr(cast, f)
                    if not callable(v):
                        bits.append(f"{f}={v!r}")
                except Exception:
                    pass
            if iface_name == "IMTBContinual":
                try:
                    u = cast.GetPositionUnit(0)
                    bits.append(f"pos={cast.GetPosition(u):.3f} {u}")
                except Exception:
                    pass
            if iface_name == "IMTBChanger":
                # Name the current element AND every alternative, so
                # the right setting is visible rather than guessed.
                try:
                    el = cast.GetElement(cast.Position)
                    bits.append(f"element={el.Name!r}")
                except Exception:
                    pass
                try:
                    lo = int(cast.MinPosition)
                    hi = int(cast.MaxPosition)
                    cur = int(cast.Position)
                    # Scan from 0: EVERY changer fails at its
                    # MaxPosition, which means GetElement() indexes a
                    # 0-based array while Position is 1-based. Dumping
                    # from 0 makes the true mapping visible instead of
                    # leaving an off-by-one to guess at.
                    names = []
                    for i in range(0, hi + 1):
                        try:
                            nm = cast.GetElement(i).Name
                        except Exception:
                            nm = "?"
                        marks = []
                        if i == cur:
                            marks.append("<= Position")
                        if i == cur - 1:
                            marks.append("<= Position-1")
                        suffix = ("  " + " ".join(marks)) if marks else ""
                        names.append(f"{i}={nm!r}{suffix}")
                    if names:
                        bits.append("\n        options: "
                                    + ", ".join(names))
                except Exception:
                    pass
            if bits:
                _log(f"      {iface_name}: {'  '.join(bits)}")

    if not found:
        _log("  no light-path components matched the candidate list")
    _log()
    _log("  Interpretation:")
    _log("    - a shutter with Open=False blocks the path")
    _log("    - a sideport/observation changer on the EYEPIECE port")
    _log("      sends no light to the camera")
    _log("    - transmitted-light brightfield needs the TL lamp, not")
    _log("      the Colibri (which is fluorescence excitation)")
    _log("    - the SIDEPORT turret decides whether light reaches the")
    _log("      camera at all; this scope's camera is on the LEFT")
    _log("      sideport per MTB's MTBCameraAdapter_MTBSideportChanger"
         "_Left")
    _log("    - a CONDENSER on a phase ring (PH1/PH2/PH3) instead of")
    _log("      brightfield throws away most of the light")


# Every MTBId we have ever *guessed* at for restricting or attenuating
# light, beyond the ones in COMPONENTS. None of these has been
# confirmed on this stand. They are probed by NAME, which can only
# ever find an id somebody thought to type in — which is why
# --enumerate exists below.
_STOP_IDS = [
    # Transmitted-light field diaphragm (Leuchtfeldblende). This is
    # the one that decides how much of the WELL is lit.
    "MTBTLFieldStop", "MTBTLFieldDiaphragm", "MTBTLLuminousFieldStop",
    "MTBFieldStop", "MTBLuminousFieldStop",
    # Transmitted-light aperture diaphragm (condenser NA, not area).
    "MTBTLAperture", "MTBTLApertureDiaphragm", "MTBApertureChanger",
    "MTBApertureStop", "MTBCondenserAperture",
    # Reflected-light (epi / Colibri) side.
    "MTBRLFieldStop", "MTBRLFieldDiaphragm", "MTBRLAperture",
    "MTBRLShutter", "MTBFLLEDShutter", "MTBLampShutter",
    # Attenuators — a neutral-density wheel halves dose without
    # touching exposure or LED percentage.
    "MTBAttenuator", "MTBRLAttenuator", "MTBFLAttenuator",
    "MTBNDFilter", "MTBFilterWheel", "MTBFilterChanger",
]

# Substrings that mark a component as interesting for light DOSE.
# Zeiss ships German names in places, hence 'blende'.
_DOSE_KEYWORDS = (
    "field", "blende", "diaphragm", "aperture", "stop",
    "atten", "nd", "shutter", "lamp", "led", "light",
    "condenser", "filter",
)

def _ident(comp) -> dict:
    """Best-effort (ID, Name, Type, ...) for one component.

    GetComponent() returns concrete ZEISS.MTB.MicControl classes whose
    interface members are explicit implementations, so identity has to
    be read through an interface cast just like positions are.
    """
    import ZEISS.MTB.Api as api

    out: dict = {}
    sources = []
    for iface in ("IMTBIdent", "IMTBBase", "IMTBComponent"):
        try:
            cast = getattr(api, iface)(comp)
        except Exception:
            continue
        if cast is not None:
            sources.append(cast)
    sources.append(comp)          # some builds surface them directly

    for src in sources:
        for attr in ("ID", "Id", "Name", "Type", "DeviceID",
                     "Version", "SerialNumber", "MaterialNumber"):
            if attr in out:
                continue
            try:
                v = getattr(src, attr)
            except Exception:
                continue
            if v is None or callable(v):
                continue
            out[attr] = v
    return out


def _count_of(obj, names) -> tuple:
    """A count that may be a property on one MTB build, a method on
    the next. Try both rather than betting on one."""
    for n in names:
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        try:
            return int(v() if callable(v) else v), n
        except Exception:
            continue
    return None, None


def _item_of(obj, names, i):
    """Element i of an indexer that may be a method or a subscript."""
    for n in names:
        try:
            f = getattr(obj, n)
        except Exception:
            continue
        try:
            return f(i) if callable(f) else f[i]
        except Exception:
            continue
    return None


def _members_of(obj) -> list:
    """Public members, for when the walk API cannot be guessed."""
    try:
        return sorted(
            a for a in dir(obj)
            if not a.startswith(("_", "get_", "set_", "add_",
                                 "remove_"))
            and a not in ("Equals", "GetHashCode", "GetType",
                          "ToString", "Overloads")
        )
    except Exception:
        return []


def _dump_all_components(root, asm) -> None:
    """Enumerate EVERY component the stand exposes, by walking the
    device tree rather than by guessing MTBIds.

    Why this is not the same as --lightpath: that probes
    root.GetComponent(id) for a hand-written list, so a component
    whose id nobody guessed is indistinguishable from a component
    that is not fitted. The specific question this was written for —
    "is there a motorized transmitted-light FIELD DIAPHRAGM, so the
    illuminated area can be cut to the field of view instead of
    flooding the whole well?" — cannot be answered by a name probe,
    because the answer we fear is 'it is there under a name we never
    tried'.

    The walk API is DISCOVERED, not assumed: MTB builds differ on
    whether the count is a property or a method. If the walk fails
    entirely, the reflected members of the root are printed so the
    right call can be identified from the output.
    """
    import ZEISS.MTB.Api as api

    n_dev, how = _count_of(root, ("DeviceCount", "GetDeviceCount",
                                  "NumberOfDevices"))
    if n_dev is None:
        _log("  cannot enumerate: the root exposes no device count.")
        _log(f"  root members: {_members_of(root)}")
        for iface in ("IMTBRoot", "IMTBBase", "IMTBIdent"):
            try:
                cast = getattr(api, iface)(root)
            except Exception:
                continue
            if cast is not None:
                _log(f"  root as {iface}: {_members_of(cast)}")
        _log("  -> pick the enumeration member from the lists above "
             "and add it to _count_of/_item_of.")
        return

    _log(f"  {n_dev} device(s) via root.{how}")
    flagged: list = []
    total = 0

    for d in range(n_dev):
        dev = _item_of(root, ("GetDevice", "Device"), d)
        if dev is None:
            _log(f"  device {d}: could not be fetched")
            continue
        dev_id = _ident(dev)
        _log()
        _log(f"  device {d}: "
             f"{dev_id.get('Name', '?')!r}  id={dev_id.get('ID', '?')!r}")

        n_c, chow = _count_of(dev, ("ComponentCount",
                                    "GetComponentCount",
                                    "NumberOfComponents"))
        if n_c is None:
            _log(f"    no component count; members: {_members_of(dev)}")
            continue

        for c in range(n_c):
            comp = _item_of(dev, ("GetComponent", "Component"), c)
            if comp is None:
                _log(f"    component {c}: could not be fetched")
                continue
            total += 1
            ident = _ident(comp)
            cid = str(ident.get("ID", "?"))
            cname = str(ident.get("Name", "?"))
            ctype = str(ident.get("Type", ""))
            hay = f"{cid} {cname} {ctype}".lower()
            mark = ""
            if any(k in hay for k in _DOSE_KEYWORDS):
                mark = "   <== light dose"
                flagged.append((cid, cname))
            _log(f"    {cid:40s} {cname!r}{mark}")

            # What can it actually be DRIVEN as? A diaphragm is either
            # a continual axis (an opening in mm or %) or a changer
            # (discrete positions). Say which, so the next step is
            # obvious rather than another probe run.
            if not mark:
                continue
            for iface, fields in (
                ("IMTBContinual", ()),
                ("IMTBChanger", ("Position", "MinPosition",
                                 "MaxPosition")),
                ("IMTBShutterState", ("Open",)),
                ("IMTBLamp", ("IsActive", "Intensity")),
                ("IMTBLampControl", ("IsActive", "Intensity")),
            ):
                try:
                    cast = getattr(api, iface)(comp)
                except Exception:
                    continue
                if cast is None:
                    continue
                bits = []
                for f in fields:
                    try:
                        v = getattr(cast, f)
                        if not callable(v):
                            bits.append(f"{f}={v!r}")
                    except Exception:
                        pass
                if iface == "IMTBContinual":
                    # THE answer for a diaphragm: a settable opening
                    # with real limits and a unit.
                    try:
                        units = [cast.GetPositionUnit(i)
                                 for i in
                                 range(cast.GetPositionUnitCount())]
                        u = units[0] if units else None
                        bits.append(f"units={units}")
                        if u:
                            bits.append(
                                f"pos={cast.GetPosition(u)} "
                                f"[{cast.GetMinPosition(u)}"
                                f"..{cast.GetMaxPosition(u)}] {u}"
                            )
                    except Exception as e:
                        bits.append(f"<{type(e).__name__}>")
                if bits:
                    _log(f"        {iface}: {'  '.join(bits)}")

    _log()
    _log(f"  {total} component(s) across {n_dev} device(s)")
    _log()
    _log("  Name probe for stops and attenuators we never confirmed:")
    for mtb_id in _STOP_IDS:
        try:
            comp = root.GetComponent(mtb_id)
        except Exception as e:
            _log(f"    {mtb_id:28s} ERROR {type(e).__name__}")
            continue
        if comp is None:
            _log(f"    {mtb_id:28s} absent")
            continue
        name = ""
        try:
            name = f"  ({comp.Name})"
        except Exception:
            pass
        _log(f"    {mtb_id:28s} PRESENT{name}")

    _log()
    if flagged:
        _log("  Components that can change the delivered dose:")
        for cid, cname in flagged:
            _log(f"    {cid}  {cname!r}")
    else:
        _log("  NOTHING in the tree looks like a field or aperture "
             "stop. If that holds, the illuminated AREA is not under "
             "software control on this stand and has to be set by "
             "hand at the microscope — close the transmitted-light "
             "field diaphragm until its edge just clears the camera "
             "field.")


def _dump_config_xml(path: str) -> int:
    """Every MTBId in MTB's own Active Configuration.xml.

    No connection, no hardware, no risk, and it can be run on the
    microscope PC in ten seconds without interrupting an experiment.
    MTB wrote this file from what is actually fitted, so if a
    motorized field diaphragm exists its id is in here. Parsing is
    deliberately schema-agnostic — every attribute value that looks
    like an MTBId is collected — because guessing the schema is the
    same mistake as guessing the component names.
    """
    import re
    import xml.etree.ElementTree as ET

    if not os.path.exists(path):
        _log(f"  not found: {path}")
        _log("  Locate it with:")
        _log('    dir /s /b "C:\\ProgramData\\Carl Zeiss\\'
             'Active Configuration.xml"')
        return 3

    _log(f"  {path}")
    ids: dict = {}
    try:
        tree = ET.parse(path)
        for el in tree.iter():
            tag = el.tag.split("}")[-1]
            name = (el.get("Name") or el.get("name") or "")
            for k, v in el.attrib.items():
                if not isinstance(v, str):
                    continue
                if re.fullmatch(r"MTB[A-Za-z0-9_]+", v):
                    ids.setdefault(v, (tag, k, name))
            if re.fullmatch(r"MTB[A-Za-z0-9_]+", tag):
                ids.setdefault(tag, (tag, "<tag>", name))
    except Exception as e:
        _log(f"  XML parse failed ({type(e).__name__}: {e}); "
             f"falling back to a raw scan")
        try:
            with open(path, "r", encoding="utf-8",
                      errors="replace") as fh:
                for m in re.finditer(r"MTB[A-Za-z0-9_]+", fh.read()):
                    ids.setdefault(m.group(0), ("<raw>", "", ""))
        except Exception as e2:
            _log(f"  raw scan failed too: {type(e2).__name__}: {e2}")
            return 3

    if not ids:
        _log("  no MTB* identifiers found — wrong file?")
        return 1

    _log(f"  {len(ids)} distinct MTB identifier(s):")
    hits = []
    for mid in sorted(ids):
        tag, attr, name = ids[mid]
        hay = f"{mid} {name}".lower()
        mark = ""
        if any(k in hay for k in _DOSE_KEYWORDS):
            mark = "   <== light dose"
            hits.append(mid)
        extra = f"  {name!r}" if name else ""
        _log(f"    {mid:44s} [{tag}/{attr}]{extra}{mark}")

    _log()
    if hits:
        _log("  Anything above marked 'light dose' is worth casting "
             "to IMTBContinual — a diaphragm reads as an opening with "
             "min/max and a unit.")
    else:
        _log("  No field stop, aperture stop or attenuator is "
             "configured. The illuminated AREA is then a MANUAL "
             "setting on the stand, not something this software can "
             "change.")
    return 0

def _dump_objective(root, comp) -> None:
    """Print every route to the objective's magnification.

    The panel could not read it via the routes we guessed, so enumerate
    exhaustively: the concrete type, the interfaces it can be cast to,
    the changer position, whatever element sits there, and every public
    member of each. Whatever names appear here are what we wire up.
    """
    import json as _json

    if comp is None:
        _log("  objective component absent")
        return

    _log(f"  concrete type: {type(comp).__name__}")
    try:
        _log(f"  Name: {comp.Name!r}")
    except Exception as e:
        _log(f"  Name unreadable: {type(e).__name__}")

    # Which ZEISS.MTB.Api interfaces does this object accept a cast to?
    # Enumerate via ASSEMBLY REFLECTION: dir() on a pythonnet namespace
    # module resolves lazily and lists almost nothing, which is why an
    # earlier version of this scan reported zero castable interfaces
    # even for IMTBComponent, which the object certainly is.
    castable = []
    try:
        from System import AppDomain
        asm = None
        for a in AppDomain.CurrentDomain.GetAssemblies():
            try:
                if (a.GetName().Name or "") == "MTBApi":
                    asm = a
                    break
            except Exception:
                continue
        if asm is None:
            raise RuntimeError("MTBApi assembly not found in AppDomain")

        try:
            types = [t for t in asm.GetTypes() if t is not None]
        except Exception as e:
            types = [t for t in (getattr(e, "Types", None) or []) if t]

        iface_types = [
            t for t in types
            if t.IsInterface and t.IsPublic
            and (t.Name or "").startswith("IMTB")
        ]
        _log(f"  scanning {len(iface_types)} IMTB* interfaces")

        import ZEISS.MTB.Api as api
        for t in sorted(iface_types, key=lambda x: x.Name):
            try:
                iface = getattr(api, t.Name)
            except Exception:
                continue
            try:
                cast = iface(comp)
            except Exception:
                continue
            if cast is None:
                continue
            extra = sorted(
                a for a in dir(cast)
                if not a.startswith(("_", "get_", "set_", "add_",
                                     "remove_"))
            )
            castable.append(t.Name)
            _log(f"    {t.Name}")
            _log(f"        {extra}")
    except Exception as e:
        _log(f"  interface scan failed: {type(e).__name__}: {e}")
    _log(f"  castable to ({len(castable)}): {castable}")

    # Public members of the raw object.
    try:
        members = sorted(a for a in dir(comp) if not a.startswith("_"))
        _log(f"  raw members ({len(members)}):")
        for i in range(0, len(members), 4):
            _log("      " + "  ".join(f"{m:26s}"
                                      for m in members[i:i + 4]))
    except Exception as e:
        _log(f"  member listing failed: {e}")

    # Read anything that looks like magnification / position / element.
    _log()
    _log("  Value probe:")
    for attr in ("Position", "Magnification", "Name", "Aperture",
                 "NumericalAperture", "ElementCount", "Elements",
                 "Element", "Objective", "Objectives"):
        try:
            val = getattr(comp, attr)
        except Exception:
            continue
        if callable(val):
            _log(f"    {attr}() -> <callable>")
            continue
        _log(f"    {attr} = {val!r}")

    # Walk the changer's elements, which is where per-objective data
    # usually lives.
    for getter in ("GetElement", "GetElementAt", "GetObjective"):
        fn = getattr(comp, getter, None)
        if not callable(fn):
            continue
        _log()
        _log(f"  {getter}(1..6):")
        for pos in range(1, 7):
            try:
                el = fn(pos)
            except Exception as e:
                _log(f"    [{pos}] {type(e).__name__}: {e}")
                continue
            if el is None:
                _log(f"    [{pos}] None")
                continue
            bits = {}
            for a in ("Name", "Magnification", "Aperture",
                      "NumericalAperture", "Contrast", "WorkingDistance"):
                try:
                    v = getattr(el, a)
                    if not callable(v):
                        bits[a] = v
                except Exception:
                    pass
            _log(f"    [{pos}] {type(el).__name__} "
                 f"{_json.dumps(bits, default=str)}")
            if pos == 1:
                try:
                    em = sorted(a for a in dir(el)
                                if not a.startswith("_"))
                    _log(f"          members: {em}")
                except Exception:
                    pass

    # IMTBChangerElement exposes only ElementType, so the objective's
    # own data must sit behind a further cast. Apply the same
    # reflection scan to the ELEMENT.
    _log()
    _log("  ELEMENT interface scan (this is where magnification "
         "should live):")
    try:
        import ZEISS.MTB.Api as api
        from System import AppDomain

        asm = None
        for a in AppDomain.CurrentDomain.GetAssemblies():
            try:
                if (a.GetName().Name or "") == "MTBApi":
                    asm = a
                    break
            except Exception:
                continue
        try:
            types = [t for t in asm.GetTypes() if t is not None]
        except Exception as e:
            types = [t for t in (getattr(e, "Types", None) or []) if t]
        iface_types = [
            t for t in types
            if t.IsInterface and t.IsPublic
            and (t.Name or "").startswith("IMTB")
        ]

        changer = None
        for iname in ("IMTBObjectiveChanger", "IMTBChanger"):
            try:
                changer = getattr(api, iname)(comp)
                if changer is not None:
                    _log(f"    changer cast: {iname}")
                    break
            except Exception:
                continue

        if changer is None:
            _log("    could not cast to a changer")
        else:
            try:
                n = changer.GetElementCount()
                pos = changer.Position
                _log(f"    GetElementCount()={n}  Position={pos}")
            except Exception as e:
                n, pos = 0, None
                _log(f"    count/position failed: {e}")

            for idx in range(1, int(n or 0) + 1):
                try:
                    el = changer.GetElement(idx)
                except Exception as e:
                    _log(f"    [{idx}] GetElement failed: {e}")
                    continue
                if el is None:
                    _log(f"    [{idx}] None")
                    continue
                etype = ""
                try:
                    etype = f" ElementType={el.ElementType!r}"
                except Exception:
                    pass
                marker = "  <-- CURRENT" if idx == pos else ""
                _log(f"    [{idx}]{etype}{marker}")

                for t in sorted(iface_types, key=lambda x: x.Name):
                    try:
                        iface = getattr(api, t.Name)
                        cast = iface(el)
                    except Exception:
                        continue
                    if cast is None:
                        continue
                    members = sorted(
                        a for a in dir(cast)
                        if not a.startswith(("_", "get_", "set_",
                                             "add_", "remove_"))
                        and a not in ("Equals", "GetHashCode",
                                      "GetType", "ToString")
                    )
                    _log(f"        {t.Name}: {members}")
                    for a in members:
                        if any(k in a.lower() for k in
                               ("magnif", "aperture", "name", "type",
                                "contrast", "immers", "workdist")):
                            try:
                                v = getattr(cast, a)
                                if not callable(v):
                                    _log(f"            {a} = {v!r}")
                            except Exception:
                                pass
    except Exception as e:
        _log(f"    element scan failed: {type(e).__name__}: {e}")

    # GetConfiguration() is another plausible home for objective data.
    _log()
    _log("  GetConfiguration():")
    try:
        cfg = comp.GetConfiguration()
        _log(f"    -> {type(cfg).__name__}")
        members = sorted(a for a in dir(cfg)
                         if not a.startswith(("_", "get_", "set_")))
        _log(f"    members: {members}")
        for a in members:
            try:
                v = getattr(cfg, a)
                if not callable(v):
                    _log(f"      {a} = {v!r}")
            except Exception:
                pass
    except Exception as e:
        _log(f"    unavailable: {type(e).__name__}: {e}")

    # And what our own wrapper makes of it.
    try:
        from tracking_tools.microscope_interface.mtb import MTBObjective
        _log()
        _log("  MTBObjective.probe():")
        info = MTBObjective(comp).probe()
        for k, v in info.items():
            if k.endswith("_attrs"):
                continue
            _log(f"    {k} = {v!r}")
    except Exception as e:
        _log(f"  wrapper probe failed: {type(e).__name__}: {e}")


def main() -> int:
    args = parse_args()

    # Before anything touches the hardware: the configuration file is
    # MTB's own record of what is fitted, and reading it costs nothing.
    if args.xml_only:
        _hdr("MTB ACTIVE CONFIGURATION")
        return _dump_config_xml(args.config_xml)

    clr, asm = load_api(args.dll)

    _hdr("SET MODE ENUM")
    mode = get_set_mode(asm)

    _hdr("CONNECT")
    conn, client_id, root = connect(asm, args.locale)

    try:
        _hdr("COMPONENTS")
        comps = {}
        for role, (mtb_id, desc) in COMPONENTS.items():
            try:
                c = root.GetComponent(mtb_id)
            except Exception as e:
                _log(f"  {role:10s} {mtb_id:24s} ERROR "
                     f"{type(e).__name__}")
                continue
            if c is None:
                _log(f"  {role:10s} {mtb_id:24s} absent")
                continue
            name = ""
            try:
                name = f" — {c.Name}"
            except Exception:
                pass
            _log(f"  {role:10s} {mtb_id:24s} OK{name}")
            comps[role] = c

        for required in ("axis_x", "axis_y", "focus"):
            if required not in comps:
                _fail(f"required component missing: {required}")

        _hdr("AXIS REPORT (read-only)")
        units = {}
        casted = {}
        for role in ("axis_x", "axis_y", "focus", "piezo"):
            if role in comps:
                unit, _, cast = report_axis(comps[role], role)
                if unit:
                    units[role] = unit
                if cast is not None:
                    casted[role] = cast

        if args.objective:
            _hdr("OBJECTIVE / NOSEPIECE")
            _dump_objective(root, comps.get("objective"))

        if args.df2:
            _hdr("DEFINITE FOCUS 2")
            _dump_df2(root, asm, comps.get("df2"))

        if args.colibri:
            _hdr("COLIBRI / FLUORESCENCE LEDS")
            _dump_colibri(root, asm, comps.get("leds"))

        if args.lightpath:
            _hdr("LIGHT PATH")
            _dump_lightpath(root)

        if args.enumerate_all:
            _hdr("EVERY COMPONENT ON THE STAND")
            _dump_all_components(root, asm)
            _hdr("MTB ACTIVE CONFIGURATION")
            _dump_config_xml(args.config_xml)

        if args.move:
            _hdr("TEST MOVES")
            _log("Moving. Ensure the objective is clear of the sample.")
            for role, delta in (
                ("axis_x", args.dxy),
                ("axis_y", args.dxy),
                ("focus", args.dz),
                ("piezo", args.dz),
            ):
                if role in casted and role in units:
                    test_move_axis(
                        casted[role], role, units[role], delta,
                        mode, args.timeout_ms, args.tol_um,
                        client_id=client_id,
                    )
        else:
            _hdr("TEST MOVES")
            _log("Skipped (pass --move to enable).")

    finally:
        try:
            conn.Logout(client_id)
            _log()
            _log("Logged out cleanly.")
        except Exception as e:
            _log(f"(logout failed, harmless: {type(e).__name__}: {e})")

    _hdr("RESULT")
    if _failures:
        _log(f"{len(_failures)} failure(s):")
        for f in _failures:
            _log(f"  - {f}")
        return 1
    _log("MTB SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
