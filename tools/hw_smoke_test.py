"""Scripted hardware checkout for the Micro-Manager backend.

Runs the same sequence of calls a human would type into a REPL during
Phase 3 of docs/bringup_runbook.md — enumerate devices, snap the
camera, nudge XY, nudge Z, cycle each channel preset — but as a
non-interactive script that exits non-zero on the first hard failure.
Use this as a repeatable pre-flight before every session on real
hardware.

Usage:
    python tools/hw_smoke_test.py --cfg docs/axio_observer_7.cfg

    python tools/hw_smoke_test.py --cfg <path> --dxy 100 --dz 5 \
                                  --group Channel --skip-channels

Design notes:
- Deliberately NOT a GUI. The Java MMStudio is the canonical
  first-time bring-up tool; this script is the scripted checkout you
  run AFTER MMStudio has proven the .cfg. See docs/bringup_runbook.md
  for the full workflow.
- No Qt / napari / pymmcore-widgets imports. Adding those to the
  bring-up environment buys nothing and pulls in PyQt6-vs-5, PySide6
  wheel-size, and napari maintenance-mode risks. If we want an
  interactive Panel checkout tab later, that decision is tracked in
  docs/work_queue.md.
- Every stage/focus move is followed by waitForDevice() — the
  Marzhauser adapter serialises XY moves; a fast burst without waits
  drops updates silently.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _log(msg: str) -> None:
    print(f"[hw_smoke_test] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[hw_smoke_test] FAIL: {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scripted hardware checkout for the MM backend.",
    )
    p.add_argument(
        "--cfg", required=True,
        help="Path to the MM system configuration (.cfg) file.",
    )
    p.add_argument(
        "--dxy", type=float, default=100.0,
        help="XY nudge in µm (default: 100).",
    )
    p.add_argument(
        "--dz", type=float, default=5.0,
        help="Z nudge in µm (default: 5).",
    )
    p.add_argument(
        "--group", default="Channel",
        help="Channel group name in the .cfg (default: 'Channel').",
    )
    p.add_argument(
        "--skip-channels", action="store_true",
        help="Skip the channel-preset cycle (useful for a dark-room "
             "test where no light is desirable).",
    )
    p.add_argument(
        "--settle-s", type=float, default=0.2,
        help="Extra settle time after each waitForDevice in seconds.",
    )
    p.add_argument(
        "--buffer-mb", type=int, default=512,
        help=(
            "MMCore circular buffer size in MB (default: 512). "
            "PVCAM (Prime 95B) has a documented race between its "
            "notification queue and MMCore's sequence buffer — "
            "starving the buffer causes 'Camera image buffer read "
            "failed' from pymmcore-plus while the same .cfg is fine "
            "in MMStudio. 512-1024 MB is the safe range for 16-bit "
            "full-frame Prime 95B. See image.sc thread 107892."
        ),
    )
    p.add_argument(
        "--adapter-path", default=None,
        help=(
            "Escape hatch for the two-adapter-tree DIV mismatch. "
            "Point pymmcore-plus at an alternate adapter tree — "
            "typically 'C:\\Program Files\\Micro-Manager-2.0' so it "
            "uses the same DLLs MMStudio saved the .cfg against."
        ),
    )
    return p.parse_args()


def check_enumerate(mmc) -> dict:
    devs = {
        "Camera":     mmc.getCameraDevice(),
        "XY stage":   mmc.getXYStageDevice(),
        "Focus":      mmc.getFocusDevice(),
        "AutoFocus":  mmc.getAutoFocusDevice(),
    }
    _log("Enumerated devices:")
    for k, v in devs.items():
        _log(f"  {k:10s} = {v!r}")
    _log(f"  Image size = {mmc.getImageWidth()} x {mmc.getImageHeight()}")
    _log(f"  API version = {mmc.getAPIVersionInfo()}")
    if not devs["Camera"]:
        raise RuntimeError("No camera device configured — check .cfg.")
    if not devs["XY stage"]:
        raise RuntimeError("No XY stage device configured — check .cfg.")
    if not devs["Focus"]:
        raise RuntimeError("No focus device configured — check .cfg.")
    return devs


def check_snap(mmc) -> None:
    mmc.snapImage()
    img = mmc.getImage()
    _log(
        f"Snapped: shape={img.shape} dtype={img.dtype} "
        f"mean={float(img.mean()):.1f} std={float(img.std()):.1f}"
    )
    if img.max() == 0:
        raise RuntimeError(
            "Camera returned an all-zero image. Likely light path, "
            "closed shutter, or PVCAM ↔ ZEN contention (kill Zen.exe)."
        )


def check_xy(mmc, dxy: float, settle_s: float) -> None:
    dev = mmc.getXYStageDevice()
    x0, y0 = mmc.getXYPosition()
    _log(f"XY before: ({x0:.2f}, {y0:.2f}) µm; nudging +{dxy} µm in X")
    mmc.setXYPosition(x0 + dxy, y0)
    mmc.waitForDevice(dev)
    time.sleep(settle_s)
    x1, y1 = mmc.getXYPosition()
    _log(f"XY after:  ({x1:.2f}, {y1:.2f}) µm")
    if abs((x1 - x0) - dxy) > 5.0:
        raise RuntimeError(
            f"XY move mismatch: commanded +{dxy}, observed +{x1 - x0:.2f}. "
            f"Wrong Marzhauser adapter, or stage lock?"
        )
    mmc.setXYPosition(x0, y0)
    mmc.waitForDevice(dev)
    time.sleep(settle_s)


def check_z(mmc, dz: float, settle_s: float) -> None:
    dev = mmc.getFocusDevice()
    z0 = mmc.getPosition()
    _log(f"Z before: {z0:.2f} µm; nudging +{dz} µm")
    mmc.setPosition(z0 + dz)
    mmc.waitForDevice(dev)
    time.sleep(settle_s)
    z1 = mmc.getPosition()
    _log(f"Z after:  {z1:.2f} µm")
    if abs((z1 - z0) - dz) > 1.0:
        raise RuntimeError(
            f"Z move mismatch: commanded +{dz}, observed +{z1 - z0:.2f}."
        )
    mmc.setPosition(z0)
    mmc.waitForDevice(dev)
    time.sleep(settle_s)


def check_channels(mmc, group: str) -> None:
    try:
        presets = list(mmc.getAvailableConfigs(group))
    except Exception as e:
        _log(f"No channel group '{group}' in this .cfg ({e}) — skipping.")
        return
    if not presets:
        _log(f"Channel group '{group}' has no presets — skipping.")
        return
    dark_mean = None
    mmc.snapImage()
    dark_mean = float(mmc.getImage().mean())
    _log(f"Baseline (current preset) frame mean = {dark_mean:.1f}")
    for preset in presets:
        _log(f"Applying {group}/{preset}")
        mmc.setConfig(group, preset)
        mmc.waitForConfig(group, preset)
        mmc.snapImage()
        m = float(mmc.getImage().mean())
        _log(f"  frame mean = {m:.1f}")
        # If baseline was in the dark and this preset lights the sample,
        # mean should climb noticeably. Not asserted — some presets are
        # legitimately dim.


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.cfg):
        _fail(f"cfg not found: {args.cfg}")
        return 2

    try:
        from pymmcore_plus import CMMCorePlus
    except ImportError:
        _fail(
            "pymmcore-plus not installed. Run "
            "`pip install -r requirements-mm.txt` and `mmcore install`."
        )
        return 3

    _log(f"Loading MM cfg: {args.cfg}")
    mmc = CMMCorePlus()

    if args.adapter_path:
        if not os.path.isdir(args.adapter_path):
            _fail(f"--adapter-path is not a directory: {args.adapter_path}")
            return 5
        _log(f"Overriding adapter search path: {args.adapter_path}")
        mmc.setDeviceAdapterSearchPaths([args.adapter_path])

    _log(f"Setting circular buffer to {args.buffer_mb} MB (PVCAM race fix)")
    try:
        mmc.setCircularBufferMemoryFootprint(args.buffer_mb)
    except Exception as e:
        _log(f"WARN: could not set circular buffer: {e}")

    try:
        mmc.loadSystemConfiguration(args.cfg)
    except Exception as e:
        _fail(
            f"loadSystemConfiguration failed: {e}\n"
            f"  Common cause: device-interface-version mismatch. Check "
            f"`mmc.getAPIVersionInfo()` matches your installed MM nightly.\n"
            f"  Escape hatch: rerun with --adapter-path "
            f"'C:\\Program Files\\Micro-Manager-2.0'."
        )
        return 4

    try:
        check_enumerate(mmc)
        check_snap(mmc)
        check_xy(mmc, args.dxy, args.settle_s)
        check_z(mmc, args.dz, args.settle_s)
        if not args.skip_channels:
            check_channels(mmc, args.group)
    except Exception as e:
        _fail(str(e))
        return 1
    finally:
        try:
            mmc.reset()
        except Exception:
            pass

    _log("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
