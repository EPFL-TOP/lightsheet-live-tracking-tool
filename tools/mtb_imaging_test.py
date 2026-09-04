"""End-to-end imaging test: MicroscopeInterface_MTB on real hardware.

This is the first thing that drives the actual backend against the
actual microscope. Unit tests prove the logic against fakes; this
proves MTB motion and PVCAM acquisition cooperate inside one loop.

SAFETY: by default NOTHING MOVES. The current stage position is read
once and used as the only baseline, so the default run just images
where the microscope already is. Motion is strictly opt-in via --grid
or --test-drift.

Usage:
    # Safest: 3 frames where the stage already sits, no motion at all
    python tools/mtb_imaging_test.py

    # Find a workable exposure first if the field looks saturated
    python tools/mtb_imaging_test.py --exposure-scan

    # Several timepoints, still no motion
    python tools/mtb_imaging_test.py --n-timepoints 5 --interval 2

    # Two positions 100 um apart in X  (MOVES THE STAGE)
    python tools/mtb_imaging_test.py --grid 2 --spacing 100

    # Verify the closed loop: inject a correction and confirm the
    # stage actually applies it next visit  (MOVES THE STAGE)
    python tools/mtb_imaging_test.py --n-timepoints 3 --test-drift 5

Exit codes:
    0  frames acquired and look plausible
    1  something failed, or every frame was degenerate
    2  a dependency is missing
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _hdr(title: str) -> None:
    _log()
    _log("=" * 70)
    _log(title)
    _log("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drive MicroscopeInterface_MTB against real "
                    "hardware.",
    )
    p.add_argument("--outdir", default=None,
                   help="Where to write frames (default: a timestamped "
                        "folder under ./data).")
    p.add_argument("--n-timepoints", type=int, default=3,
                   help="Timepoints to acquire (default: 3).")
    p.add_argument("--interval", type=float, default=2.0,
                   help="Seconds between timepoints (default: 2).")
    p.add_argument("--exposure", type=float, default=5.0,
                   help="Exposure in ms. The Prime 95B saturates by "
                        "~100 ms at current light (default: 5).")
    p.add_argument("--channel", default="BF",
                   help="Channel label used in filenames "
                        "(default: BF).")
    p.add_argument("--z-axis", default="piezo",
                   choices=("piezo", "focus"),
                   help="Which Z actuator to use (default: piezo).")
    p.add_argument("--settle", type=float, default=0.3,
                   help="Seconds to settle after a move (default: 0.3).")
    p.add_argument("--grid", type=int, default=1,
                   help="Number of positions along X. >1 MOVES THE "
                        "STAGE (default: 1 = no motion).")
    p.add_argument("--spacing", type=float, default=100.0,
                   help="Spacing in um between grid positions "
                        "(default: 100).")
    p.add_argument("--test-drift", type=float, default=0.0,
                   help="Inject this XY correction (um) after the "
                        "first frame to verify the closed loop. "
                        "MOVES THE STAGE.")
    p.add_argument("--exposure-scan", action="store_true",
                   help="Before imaging, snap at a range of exposures "
                        "and report which avoid saturation. Implies no "
                        "motion.")
    p.add_argument("--verbose", action="store_true",
                   help="Show backend debug logging.")
    return p.parse_args()


def read_current_position(z_axis: str):
    """Read where the stage is now, to use as a safe baseline.

    Uses MTBSession.shared() and does NOT log out: MTB permits one
    Login per process, so the backend must reuse this very session.
    An earlier version opened its own session here, logged out, and
    then the backend's Login failed with
      ConnectToRtSystem(): OpenRtNet(...) 'No such interface supported'
    """
    from tracking_tools.microscope_interface.mtb import (
        MTBMotion, MTBSession,
    )
    session = MTBSession.shared()
    motion = MTBMotion(session, z_axis=z_axis)
    _log(motion.describe())
    return session, motion.get_xyz()


def exposure_scan(exposures=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0)):
    """Report which exposures give a usable, unsaturated image."""
    _hdr("EXPOSURE SCAN")
    try:
        from pymmcore_plus import CMMCorePlus
    except ImportError:
        _log("pymmcore-plus missing")
        return None

    mmc = CMMCorePlus()
    try:
        mmc.setCircularBufferMemoryFootprint(512)
    except Exception:
        pass
    mmc.loadDevice("Cam", "PVCAM", "Camera-1")
    mmc.initializeDevice("Cam")
    mmc.setCameraDevice("Cam")

    depth = mmc.getImageBitDepth()
    full = (1 << depth) - 1
    _log(f"sensor {mmc.getImageWidth()}x{mmc.getImageHeight()}, "
         f"{depth}-bit (saturates at {full})")
    _log()
    _log(f"{'exp (ms)':>9}  {'min':>7}  {'max':>7}  {'mean':>9}  "
         f"{'std':>8}  verdict")

    best = None
    try:
        for exp in exposures:
            mmc.setExposure(exp)
            mmc.snapImage()
            img = mmc.getImage()
            mn, mx = float(img.min()), float(img.max())
            mean, std = float(img.mean()), float(img.std())
            if mx >= full:
                verdict = "SATURATED"
            elif std < 1.0:
                verdict = "flat / no signal"
            elif mx < full * 0.1:
                verdict = "dim"
            else:
                verdict = "GOOD"
                if best is None:
                    best = exp
            _log(f"{exp:9.1f}  {mn:7.0f}  {mx:7.0f}  {mean:9.1f}  "
                 f"{std:8.1f}  {verdict}")
    finally:
        try:
            mmc.unloadAllDevices()
        except Exception:
            pass

    _log()
    if best is not None:
        _log(f"Suggested exposure: {best} ms  "
             f"(pass --exposure {best})")
    else:
        _log("No exposure produced a good image. Check illumination "
             "and that the light path is set to the camera sideport.")
    return best


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.exposure_scan:
        best = exposure_scan()
        if best is None:
            return 1
        _log()
        _log("Re-run without --exposure-scan to acquire a series.")
        return 0

    try:
        from tracking_tools.microscope_interface.mtb_backend import (
            MicroscopeInterface_MTB,
        )
    except ImportError as e:
        _log(f"FAIL: cannot import the backend: {e}")
        return 2

    # --- Establish baselines from where the stage actually is ---
    _hdr("CURRENT POSITION")
    try:
        session, (x0, y0, z0) = read_current_position(args.z_axis)
    except Exception as e:
        _log(f"FAIL: could not read the stage: {type(e).__name__}: {e}")
        return 1
    _log(f"baseline: x={x0:.3f} y={y0:.3f} z={z0:.3f} um")

    positions = {}
    for i in range(max(1, args.grid)):
        positions[f"scene_{i:03d}"] = {
            "xyz_um": (x0 + i * args.spacing, y0, z0)
        }

    moves = args.grid > 1 or args.test_drift
    _log()
    if moves:
        _log("*** THIS RUN WILL MOVE THE STAGE ***")
        if args.grid > 1:
            _log(f"    {args.grid} positions, {args.spacing} um apart "
                 f"in X")
        if args.test_drift:
            _log(f"    plus an injected {args.test_drift} um XY "
                 f"correction")
        _log("    Make sure the objective is clear of the sample.")
    else:
        _log("No motion this run — imaging at the current position "
             "only.")

    outdir = args.outdir or os.path.join(
        "data", f"mtb_imaging_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(outdir, exist_ok=True)
    _log(f"output: {os.path.abspath(outdir)}")

    iface = MicroscopeInterface_MTB(
        positions, outdir,
        {
            "z_axis": args.z_axis,
            "exposure_ms": args.exposure,
            "channel": args.channel,
            "interval_s": args.interval,
            "settle_s": args.settle,
            "stop_after_tp": args.n_timepoints,
            # Hand over the session we already hold — a second Login
            # would fail in this process.
            "session": session,
        },
    )

    _hdr("ACQUIRING")
    frames = []
    injected = False
    try:
        iface.connect()
        deadline = time.monotonic() + (
            args.n_timepoints * max(args.interval, 1.0) * len(positions)
            + 60.0
        )
        while time.monotonic() < deadline:
            item = iface.wait_for_image(timeout_ms=500)
            if item is None:
                if iface._thread is None or not iface._thread.is_alive():
                    break
                continue
            img, tp, pos_name = item
            mn, mx = float(img.min()), float(img.max())
            mean, std = float(img.mean()), float(img.std())
            frames.append((tp, pos_name, mn, mx, mean, std))
            _log(f"  t={tp} {pos_name:12s} shape={img.shape} "
                 f"min={mn:.0f} max={mx:.0f} mean={mean:.1f} "
                 f"std={std:.1f}")

            # Verify the closed loop: after the first frame, inject a
            # correction and let the loop apply it on the next visit.
            if args.test_drift and not injected:
                _log(f"    injecting {args.test_drift} um XY "
                     f"correction for {pos_name}")
                iface.relative_move(
                    pos_name, args.test_drift, args.test_drift, 0.0
                )
                injected = True
    except KeyboardInterrupt:
        _log("interrupted")
    except Exception as e:
        _log(f"FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        _log()
        _log("shutting down ...")
        try:
            iface.disconnect()
        except Exception as e:
            _log(f"(disconnect complained: {e})")

    # --- Report ---
    _hdr("RESULT")
    if not frames:
        _log("No frames acquired.")
        return 1

    _log(f"{len(frames)} frame(s) acquired.")
    degenerate = [f for f in frames if f[5] < 1.0]
    saturated = [f for f in frames if f[3] >= 65535]
    if saturated:
        _log(f"{len(saturated)} frame(s) SATURATED — reduce "
             f"--exposure (try --exposure-scan).")
    if degenerate:
        _log(f"{len(degenerate)} frame(s) had no variance.")

    if args.test_drift:
        for name in positions:
            _log(f"final cum_drift[{name}] = "
                 f"{iface.get_cum_drift(name)}")
        _log("Non-zero drift above means relative_move() was recorded "
             "and applied on the following visit.")

    _log()
    _log(f"Frames written under: {os.path.abspath(outdir)}")
    for name in positions:
        folder = os.path.join(outdir, name)
        if os.path.isdir(folder):
            got = sorted(os.listdir(folder))
            _log(f"  {name}: {len(got)} file(s)  {got[:4]}"
                 f"{' ...' if len(got) > 4 else ''}")

    if len(frames) and not degenerate:
        _log()
        _log("IMAGING TEST PASSED")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
