"""Discover which serial port a Micro-Manager device is wired to.

Motivation: the ZeissCAN29 adapter does NOT auto-detect — the MM wiki
is explicit that "hardware must be manually defined in the
Configuration Wizard; the microscope is not auto-queried". But once
the port IS right, initializing the hub succeeds and the adapter reads
back real labels from the stand. So a successful initializeDevice() is
the detection: we brute-force the (port, baud) grid and report which
combination answers.

Beats tracing cables behind a crowded microscope.

Usage:
    # Inventory only — what ports and libraries exist?
    python tools/probe_serial_devices.py --list

    # Probe for the Zeiss stand (the default target)
    python tools/probe_serial_devices.py

    # Probe a specific port/baud
    python tools/probe_serial_devices.py --ports COM1 --bauds 57600

    # Probe for a Marzhauser TANGO controller instead
    python tools/probe_serial_devices.py --library Marzhauser --device XYStage

    # Probe for a Marzhauser LStep controller
    python tools/probe_serial_devices.py --library MarzhauserLStep --device XYStage

Exit codes:
    0  at least one (port, baud) answered
    1  nothing answered
    2  bad arguments / missing adapter
    3  pymmcore-plus not installed
"""
from __future__ import annotations

import argparse
import sys

# ZeissCAN29 wiki documents 57600 as the CAN29 default. The others are
# the plausible fallbacks if someone reprogrammed the stand or we are
# probing a Marzhauser controller instead.
DEFAULT_BAUDS = ["57600", "9600", "19200", "38400", "115200"]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find which serial port an MM device is wired to.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Only inventory available serial ports + device libraries, "
             "then exit. Run this first.",
    )
    p.add_argument(
        "--library", default="ZeissCAN29",
        help="MM device library to probe with (default: ZeissCAN29).",
    )
    p.add_argument(
        "--device", default="ZeissScope",
        help="Device name inside that library — must be the one that "
             "owns the 'Port' property, i.e. the hub "
             "(default: ZeissScope).",
    )
    p.add_argument(
        "--ports", nargs="*", default=None,
        help="Serial ports to try (default: every port MM can see).",
    )
    p.add_argument(
        "--bauds", nargs="*", default=None,
        help=f"Baud rates to try (default: {' '.join(DEFAULT_BAUDS)}).",
    )
    p.add_argument(
        "--timeout-ms", default="500",
        help="Serial AnswerTimeout in ms (default: 500).",
    )
    p.add_argument(
        "--adapter-path", default=None,
        help="Override the MM device-adapter search path, e.g. "
             r"'C:\Program Files\Micro-Manager-2.0'.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show the full error for every failed combination.",
    )
    return p.parse_args()


def make_core(adapter_path: str | None):
    try:
        from pymmcore_plus import CMMCorePlus
    except ImportError:
        _log("FAIL: pymmcore-plus not installed. Run:")
        _log("  pip install -r requirements-mm.txt")
        _log("  mmcore install")
        sys.exit(3)

    mmc = CMMCorePlus()
    if adapter_path:
        _log(f"Adapter search path override: {adapter_path}")
        mmc.setDeviceAdapterSearchPaths([adapter_path])
    return mmc


def get_serial_ports(mmc) -> list[str]:
    """Ports MM can see, via the SerialManager pseudo-library."""
    try:
        return list(mmc.getAvailableDevices("SerialManager"))
    except Exception as e:
        _log(f"WARN: could not enumerate SerialManager ports: {e}")
        _log("      (On Windows this should work. If it doesn't, the "
             "adapter tree may be incomplete — try `mmcore install`.)")
        return []


def do_inventory(mmc, library: str) -> None:
    _log("=" * 62)
    _log("INVENTORY")
    _log("=" * 62)

    try:
        _log(f"MMCore API:  {mmc.getAPIVersionInfo()}")
    except Exception as e:
        _log(f"MMCore API:  <unavailable: {e}>")

    ports = get_serial_ports(mmc)
    _log()
    _log(f"Serial ports visible to MM ({len(ports)}):")
    for p in ports:
        _log(f"  - {p}")
    if not ports:
        _log("  <none>")

    _log()
    try:
        libs = sorted(mmc.getDeviceAdapterNames())
    except Exception as e:
        _log(f"Could not list device libraries: {e}")
        libs = []

    # Show the libraries relevant to this microscope, not all ~200.
    interesting = [
        lib for lib in libs
        if any(k in lib.lower() for k in
               ("zeiss", "marzhauser", "pvcam", "serial", "democamera"))
    ]
    _log(f"Relevant device libraries present ({len(interesting)} of "
         f"{len(libs)} total):")
    for lib in interesting:
        _log(f"  - {lib}")

    _log()
    for lib in ("ZeissCAN29", "Marzhauser", "MarzhauserLStep", "PVCAM"):
        try:
            devs = list(mmc.getAvailableDevices(lib))
            _log(f"{lib} sub-devices ({len(devs)}):")
            for d in devs:
                _log(f"  - {d}")
        except Exception as e:
            _log(f"{lib}: not loadable ({type(e).__name__})")
        _log()


def probe(mmc, library, device, port, baud, timeout_ms, verbose):
    """Try to bring up `device` on `port` at `baud`.

    Returns a dict of readback info on success, None on failure.
    """
    # Always start from a clean slate — a half-initialized device from
    # a previous attempt will poison the next one.
    try:
        mmc.unloadAllDevices()
    except Exception:
        pass

    port_label = port
    hub_label = "_probe_hub"

    try:
        # 1. Bring up the serial port itself.
        mmc.loadDevice(port_label, "SerialManager", port)
        for prop, val in (
            ("BaudRate", baud),
            ("DataBits", "8"),
            ("StopBits", "1"),
            ("Parity", "None"),
            ("Handshaking", "Off"),
            ("AnswerTimeout", timeout_ms),
            ("DelayBetweenCharsMs", "0"),
        ):
            try:
                mmc.setProperty(port_label, prop, val)
            except Exception:
                # Not every SerialManager build exposes every property.
                pass
        mmc.initializeDevice(port_label)

        # 2. Bring up the hub and point it at that port.
        mmc.loadDevice(hub_label, library, device)
        mmc.setProperty(hub_label, "Port", port_label)
        mmc.initializeDevice(hub_label)

        # 3. Success — scrape whatever the device will tell us. This is
        #    the payoff: real labels read back off the hardware.
        info = {"port": port, "baud": baud, "properties": {}}
        try:
            for prop in mmc.getDevicePropertyNames(hub_label):
                try:
                    info["properties"][prop] = mmc.getProperty(
                        hub_label, prop
                    )
                except Exception:
                    pass
        except Exception:
            pass
        return info

    except Exception as e:
        if verbose:
            _log(f"      {type(e).__name__}: {e}")
        return None
    finally:
        try:
            mmc.unloadAllDevices()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    mmc = make_core(args.adapter_path)

    if args.list:
        do_inventory(mmc, args.library)
        return 0

    ports = args.ports if args.ports else get_serial_ports(mmc)
    bauds = args.bauds if args.bauds else DEFAULT_BAUDS

    if not ports:
        _log("FAIL: no serial ports to probe.")
        _log("      Run with --list to inventory, or pass --ports COM1.")
        return 2

    # Confirm the target library is actually present before probing.
    try:
        available = list(mmc.getAvailableDevices(args.library))
    except Exception as e:
        _log(f"FAIL: device library {args.library!r} not loadable: {e}")
        _log("      Run with --list to see what IS available.")
        return 2
    if args.device not in available:
        _log(f"FAIL: {args.device!r} not found in library "
             f"{args.library!r}.")
        _log(f"      Available: {', '.join(available)}")
        return 2

    _log("=" * 62)
    _log(f"PROBING {args.library}/{args.device}")
    _log("=" * 62)
    _log(f"Ports: {', '.join(ports)}")
    _log(f"Bauds: {', '.join(bauds)}")
    _log(f"Timeout: {args.timeout_ms} ms")
    _log()

    hits = []
    for port in ports:
        for baud in bauds:
            _log(f"  trying {port} @ {baud} ... ", )
            info = probe(
                mmc, args.library, args.device, port, baud,
                args.timeout_ms, args.verbose,
            )
            if info:
                _log(f"  *** {port} @ {baud}: ANSWERED ***")
                hits.append(info)
                # Don't keep hammering a port that already worked.
                break
            else:
                _log(f"      {port} @ {baud}: no answer")
    _log()

    if not hits:
        _log("=" * 62)
        _log("RESULT: nothing answered.")
        _log("=" * 62)
        _log("Next things to check:")
        _log("  1. Is the stand powered on and finished booting?")
        _log("  2. Is ZEN (or its services) holding the port? Kill")
        _log("     zen.exe / zenblue.exe / MTB2.exe, stop 'Carl Zeiss'")
        _log("     services in services.msc.")
        _log("  3. Is a cable actually connected between the stand and")
        _log("     this PC? A port MM can see is not a port with")
        _log("     hardware on the other end.")
        _log("  4. Re-run with -v to see the actual error per attempt.")
        return 1

    _log("=" * 62)
    _log(f"RESULT: {len(hits)} combination(s) answered")
    _log("=" * 62)
    for info in hits:
        _log()
        _log(f"  Port {info['port']} @ {info['baud']} baud")
        props = info["properties"]
        if props:
            _log(f"  Readback ({len(props)} properties):")
            for k, v in sorted(props.items()):
                _log(f"    {k} = {v!r}")
        else:
            _log("  (no properties readable — initialized but silent)")
    _log()
    _log("Use the winning port + baud in the Configuration Wizard for")
    _log("the hub device, then add the sub-devices under it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
