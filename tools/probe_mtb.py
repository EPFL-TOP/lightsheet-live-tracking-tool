"""Probe Zeiss MTB 2011 through pythonnet — API discovery + connection test.

Why this exists: the Axio Observer 7 carries its CAN29 bus over USB into
CZCanSrv, so Micro-Manager's serial ZeissCAN29 adapter can never reach
it (proved 2026-09-03 — every device in MTB's Active Configuration.xml
says PortType="USB"). MTB is Zeiss's hardware abstraction layer and the
only supported route to the stand. ZEN is merely another MTB client, so
using MTB does NOT mean running ZEN.

This script deliberately DISCOVERS the API by reflection rather than
assuming method signatures, so we build the backend against what is
actually there.

Usage:
    # Everything, in order, stopping at the first hard failure
    python tools/probe_mtb.py

    # Just reflect over the assembly — no connection attempt
    python tools/probe_mtb.py --dump-api

    # Just try to connect and enumerate components
    python tools/probe_mtb.py --connect

    # Narrow the reflection dump
    python tools/probe_mtb.py --dump-api --filter Stage

    # Non-default install location
    python tools/probe_mtb.py --dll "C:\\path\\to\\MTBApi.dll"

Exit codes:
    0  success
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

# Component MTBIds seen in this machine's Active Configuration.xml.
# These are what GetComponent() should accept.
KNOWN_COMPONENT_IDS = [
    "MTBStage",
    "MTBStageAxisX",
    "MTBStageAxisY",
    "MTBFocus",
    "MTBFocusStabilizer2",       # Definite Focus 2
    "MTBPiezoFocusCan",          # Wienecke & Sinske WSB 500 um
    "MTBFLLEDShutter",           # Colibri 5/7
    "MTBFLLEDController",
    "MTBObjectiveChanger",
    "MTBReflectorChanger",
    "MTBCamera_MTBCameraAdapter_MTBSideportChanger_Left",
]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _hdr(title: str) -> None:
    _log()
    _log("=" * 70)
    _log(title)
    _log("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe Zeiss MTB 2011 via pythonnet.",
    )
    p.add_argument(
        "--dll", default=DEFAULT_DLL,
        help=f"Path to MTBApi.dll (default: {DEFAULT_DLL})",
    )
    p.add_argument(
        "--dump-api", action="store_true",
        help="Reflect over the assembly and list types/members.",
    )
    p.add_argument(
        "--connect", action="store_true",
        help="Attempt MTB connection + login + component enumeration.",
    )
    p.add_argument(
        "--filter", default=None,
        help="Only show types whose name contains this substring "
             "(case-insensitive). Applies to --dump-api.",
    )
    p.add_argument(
        "--members", action="store_true",
        help="With --dump-api, also list methods/properties per type. "
             "Verbose — pair with --filter.",
    )
    p.add_argument(
        "--locale", default="en",
        help="Locale string passed to MTB Login (default: en).",
    )
    return p.parse_args()


def load_assembly(dll_path: str):
    """Load MTBApi.dll and return the clr module."""
    try:
        import clr  # noqa: F401  (pythonnet)
    except ImportError:
        _log("FAIL: pythonnet not installed.")
        _log("      pip install pythonnet")
        sys.exit(2)

    import clr

    if not os.path.exists(dll_path):
        _log(f"FAIL: MTBApi.dll not found at:\n      {dll_path}")
        _log("      Locate it with:")
        _log("        Get-ChildItem 'C:\\Program Files\\Carl Zeiss' "
             "-Recurse -Include MTBApi.dll")
        sys.exit(3)

    _log(f"Loading assembly: {dll_path}")
    asm = None
    try:
        # AddReference accepts a full path without the .dll suffix and
        # returns the loaded Assembly.
        asm = clr.AddReference(dll_path[:-4] if dll_path.endswith(".dll")
                               else dll_path)
    except Exception as e_path:
        # Fall back to loading the raw file, which is more permissive.
        try:
            from System.Reflection import Assembly
            asm = Assembly.LoadFrom(dll_path)
        except Exception as e_file:
            _log("FAIL: could not load assembly.")
            _log(f"      AddReference: {type(e_path).__name__}: {e_path}")
            _log(f"      LoadFrom:     {type(e_file).__name__}: {e_file}")
            _log()
            _log("      If this says the assembly targets an "
                 "incompatible runtime, MTB 2011 is .NET Framework — "
                 "pythonnet must be built against .NET Framework or "
                 ".NET Core with compat shims. Try a 32-bit Python if "
                 "CZCanSrv.exe being 32-bit turns out to matter.")
            sys.exit(3)
    _log("Assembly loaded.")

    if asm is None:
        # Some pythonnet versions return None from AddReference; recover
        # the Assembly by scanning the AppDomain. Note AppDomain is a
        # CLASS in the System namespace, not a module.
        asm = find_mtb_assembly()
    return asm


def find_mtb_assembly():
    """Return the loaded MTBApi Assembly object, or None."""
    try:
        from System import AppDomain
    except ImportError:
        return None
    for candidate in AppDomain.CurrentDomain.GetAssemblies():
        try:
            name = candidate.GetName().Name or ""
        except Exception:
            continue
        if "MTB" in name:
            return candidate
    return None


def dump_api(asm, filter_str: str | None, show_members: bool) -> None:
    _hdr("MTB API SURFACE (reflection)")

    if asm is None:
        asm = find_mtb_assembly()
    if asm is None:
        _log("Could not locate the loaded MTBApi assembly.")
        return

    try:
        _log(f"Assembly: {asm.GetName().Name} v{asm.GetName().Version}")
    except Exception:
        _log(f"Assembly: {asm}")
    _log()

    try:
        types = list(asm.GetTypes())
    except Exception as e:
        # ReflectionTypeLoadException still exposes the types it managed
        # to load, which is usually enough.
        types = []
        for attr in ("Types", "types"):
            got = getattr(e, attr, None)
            if got:
                types = [t for t in got if t is not None]
                break
        if not types:
            _log(f"GetTypes() failed: {type(e).__name__}: {e}")
            return
        _log(f"(partial type load: {len(types)} types recovered)")

    # Group by namespace so the shape of the API is legible.
    by_ns: dict[str, list] = {}
    for t in types:
        ns = t.Namespace or "<global>"
        by_ns.setdefault(ns, []).append(t)

    _log(f"Namespaces ({len(by_ns)}):")
    for ns in sorted(by_ns):
        _log(f"  {ns}  ({len(by_ns[ns])} types)")

    needle = filter_str.lower() if filter_str else None

    for ns in sorted(by_ns):
        shown = [
            t for t in sorted(by_ns[ns], key=lambda x: x.Name)
            if t.IsPublic and (needle is None or needle in t.Name.lower())
        ]
        if not shown:
            continue
        _log()
        _log(f"--- {ns} ---")
        for t in shown:
            kind = ("interface" if t.IsInterface
                    else "enum" if t.IsEnum
                    else "class")
            _log(f"  [{kind}] {t.Name}")
            if not show_members:
                continue
            try:
                for m in sorted(t.GetMethods(), key=lambda x: x.Name):
                    if m.DeclaringType != t:
                        continue  # skip inherited Object members
                    params = ", ".join(
                        f"{p.ParameterType.Name} {p.Name}"
                        for p in m.GetParameters()
                    )
                    _log(f"        {m.ReturnType.Name} "
                         f"{m.Name}({params})")
            except Exception as e:
                _log(f"        <members unavailable: {e}>")


def try_connect(asm, locale: str) -> int:
    _hdr("MTB CONNECTION TEST")

    # The documented MTB SDK entry point is ZEISS.MTB.Api.MTBConnection,
    # but do not trust that blindly — locate it by reflection first.
    conn_type = None
    if asm is None:
        asm = find_mtb_assembly()
    if asm is None:
        _log("Could not locate the MTBApi assembly.")
        return 4

    try:
        types = [t for t in asm.GetTypes() if t is not None]
    except Exception as e:
        types = [t for t in (getattr(e, "Types", None) or []) if t]

    candidates = [
        t for t in types
        if t.IsPublic and not t.IsInterface
        and "connection" in t.Name.lower()
    ]
    if not candidates:
        _log("No public *Connection* type found. Run --dump-api and "
             "look for the entry-point class.")
        return 4

    _log("Connection candidate types:")
    for t in candidates:
        _log(f"  {t.Namespace}.{t.Name}")
    conn_type = candidates[0]
    _log()
    _log(f"Using: {conn_type.Namespace}.{conn_type.Name}")

    # Instantiate.
    try:
        from System import Activator
        conn = Activator.CreateInstance(conn_type)
    except Exception as e:
        _log(f"FAIL: could not instantiate: {type(e).__name__}: {e}")
        return 4
    _log("Instantiated connection object.")

    # Login. Signature is typically Login(string locale, out string id).
    client_id = None
    for attempt in ("two_arg", "one_arg", "no_arg"):
        try:
            if attempt == "two_arg":
                # pythonnet returns out-params in a tuple.
                res = conn.Login(locale, "")
                client_id = res[1] if isinstance(res, tuple) else res
            elif attempt == "one_arg":
                client_id = conn.Login(locale)
            else:
                client_id = conn.Login()
            _log(f"Login OK ({attempt}); clientId = {client_id!r}")
            break
        except Exception as e:
            _log(f"  Login({attempt}) failed: {type(e).__name__}: {e}")
    else:
        _log()
        _log("FAIL: every Login signature failed.")
        _log("      If the error mentions the MTB server being "
             "unavailable, check the MTBService_2.12.0.7 service is "
             "Running and that the hardware is powered on.")
        return 4

    # Get the device tree root.
    root = None
    for meth, args in (("GetRoot", (client_id,)), ("GetRoot", ())):
        try:
            root = getattr(conn, meth)(*args)
            _log(f"{meth}{args} OK -> {root}")
            break
        except Exception as e:
            _log(f"  {meth}{args} failed: {type(e).__name__}: {e}")
    if root is None:
        _log("FAIL: could not obtain the MTB root object.")
        return 4

    # Enumerate the components we care about.
    _log()
    _log("Component lookup (IDs from this machine's "
         "Active Configuration.xml):")
    found = {}
    for cid in KNOWN_COMPONENT_IDS:
        try:
            comp = root.GetComponent(cid)
        except Exception as e:
            _log(f"  {cid:52s} ERROR {type(e).__name__}")
            continue
        if comp is None:
            _log(f"  {cid:52s} absent")
        else:
            _log(f"  {cid:52s} PRESENT -> {comp}")
            found[cid] = comp

    # Try to read live values off whatever we found.
    if found:
        _log()
        _log("Attempting live reads:")
        for cid, comp in found.items():
            for prop in ("Position", "PositionX", "PositionY",
                         "Name", "DevicePosition"):
                try:
                    val = getattr(comp, prop)
                    _log(f"  {cid}.{prop} = {val!r}")
                except Exception:
                    pass

    # Be a good citizen — MTB tracks client sessions.
    try:
        conn.Logout(client_id)
        _log()
        _log("Logged out cleanly.")
    except Exception as e:
        _log(f"(logout failed, harmless: {type(e).__name__}: {e})")

    return 0


def main() -> int:
    args = parse_args()

    # Default: do both, API dump first so a connection failure still
    # leaves us with the API map.
    do_dump = args.dump_api or not (args.dump_api or args.connect)
    do_conn = args.connect or not (args.dump_api or args.connect)

    asm = load_assembly(args.dll)

    if do_dump:
        dump_api(asm, args.filter, args.members)

    rc = 0
    if do_conn:
        rc = try_connect(asm, args.locale)

    _hdr("DONE")
    if rc == 0:
        _log("No hard failures.")
    else:
        _log(f"Exited with code {rc} — see above.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
