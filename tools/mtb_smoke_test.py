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

    # And what our own wrapper makes of it.
    try:
        from tracking_tools.microscope_interface.mtb import MTBObjective
        _log()
        _log("  MTBObjective.probe():")
        info = MTBObjective(comp).probe()
        for k, v in info.items():
            _log(f"    {k} = {v!r}")
    except Exception as e:
        _log(f"  wrapper probe failed: {type(e).__name__}: {e}")


def main() -> int:
    args = parse_args()
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
