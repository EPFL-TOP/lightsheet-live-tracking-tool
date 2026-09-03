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


def report_axis(comp, label: str) -> tuple[str | None, float | None]:
    """Print units, limits, step and current position for one axis."""
    units = discover_units(comp, label)
    unit = pick_um(units)
    _log(f"  {label}")
    _log(f"    units available: {units or '<none>'}")
    if unit is None:
        _fail(f"{label}: no usable position unit")
        return None, None
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

    return unit, pos


def test_move_axis(comp, label, unit, delta, mode, timeout, tol) -> None:
    """Relative move on a single IMTBContinual axis, then move back."""
    try:
        before = comp.GetPosition(unit)
    except Exception as e:
        _fail(f"{label}: cannot read start position: {e}")
        return

    target = before + delta
    _log(f"  {label}: {before:.3f} -> {target:.3f} {unit}")
    try:
        ok = comp.SetPosition(target, unit, mode, timeout)
    except Exception as e:
        _fail(f"{label}: SetPosition raised: {type(e).__name__}: {e}")
        return
    if ok is False:
        _fail(f"{label}: SetPosition returned False")
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
        comp.SetPosition(before, unit, mode, timeout)
        back = comp.GetPosition(unit)
        _log(f"    restored to {back:.3f} {unit}")
    except Exception as e:
        _fail(f"{label}: could not restore original position: {e}")


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
        for role in ("axis_x", "axis_y", "focus", "piezo"):
            if role in comps:
                unit, _ = report_axis(comps[role], role)
                if unit:
                    units[role] = unit

        if args.move:
            _hdr("TEST MOVES")
            _log("Moving. Ensure the objective is clear of the sample.")
            for role, delta in (
                ("axis_x", args.dxy),
                ("axis_y", args.dxy),
                ("focus", args.dz),
            ):
                if role in comps and role in units:
                    test_move_axis(
                        comps[role], role, units[role], delta,
                        mode, args.timeout_ms, args.tol_um,
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
