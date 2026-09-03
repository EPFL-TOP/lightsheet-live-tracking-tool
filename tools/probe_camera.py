"""Is the Prime 95B actually there, or is PVCAM offering a phantom slot?

PVCAM advertises Camera-1..Camera-4 in Micro-Manager regardless of what
is physically connected, and initializeDevice() can succeed on an empty
slot. So "it initialized" is NOT evidence the camera is alive. This
script looks for positive proof: populated identity properties (model,
serial, sensor dimensions) and a snap that returns non-degenerate data.

Needs no .cfg file — it loads the PVCAM adapter directly, so it works
before any hardware configuration exists.

Usage:
    python tools/probe_camera.py                 # try Camera-1
    python tools/probe_camera.py --all            # try all four slots
    python tools/probe_camera.py --exposure 200   # longer exposure
    python tools/probe_camera.py --library DemoCamera --device DCam

Exit codes:
    0  a real camera responded with a plausible image
    1  camera loaded but looks like a phantom / snap failed
    2  pymmcore-plus not installed
    3  the device library could not be loaded
"""
from __future__ import annotations

import argparse
import sys

# Properties whose presence and non-emptiness suggest a real device
# rather than an unpopulated slot.
IDENTITY_HINTS = (
    "camera", "chip", "model", "serial", "sensor", "firmware",
    "name", "version", "product",
)

# Set from --sweep in main(); consulted by try_camera().
SWEEP_REQUESTED = False


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _hdr(title: str) -> None:
    _log()
    _log("=" * 66)
    _log(title)
    _log("=" * 66)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distinguish a real camera from a phantom PVCAM slot.",
    )
    p.add_argument("--library", default="PVCAM",
                   help="MM device library (default: PVCAM).")
    p.add_argument("--device", default=None,
                   help="Specific device name, e.g. Camera-1. Default: "
                        "first available (or all with --all).")
    p.add_argument("--all", action="store_true",
                   help="Try every device the library advertises.")
    p.add_argument("--exposure", type=float, default=100.0,
                   help="Exposure in ms for the snap (default: 100).")
    p.add_argument("--buffer-mb", type=int, default=512,
                   help="Circular buffer MB — PVCAM needs headroom "
                        "(default: 512).")
    p.add_argument("--sweep", action="store_true",
                   help="Systematically vary exposure, ShutterMode and "
                        "ReadoutRate to separate real saturation from "
                        "an unread buffer.")
    return p.parse_args()


def _snap_stats(mmc, label):
    """Snap once and return (min, max, mean, std) or None."""
    try:
        mmc.snapImage()
        img = mmc.getImage()
        return (float(img.min()), float(img.max()),
                float(img.mean()), float(img.std()))
    except Exception as e:
        _log(f"      snap failed: {type(e).__name__}: {e}")
        return None


def sweep(mmc, label, names) -> bool:
    """Vary the things that distinguish the plausible causes.

    Reasoning:
    - If a very SHORT exposure brings values down, the sensor is really
      being read and the field was simply saturated by light.
    - If the constant value tracks the READOUT BIT DEPTH (65535 at
      16-bit, 4095 at 12-bit), it is a fill pattern / unread buffer,
      because real optical saturation does not renormalise itself.
    - ShutterMode drives a shutter line that may not exist on this
      sideport; 'Never' removes it from the equation.
    """
    _hdr("SWEEP")
    found = False

    def has(prop):
        return prop in names

    _log("Exposure sweep (short exposures should darken a real image):")
    for exp in (1.0, 5.0, 20.0, 100.0):
        try:
            mmc.setExposure(exp)
        except Exception as e:
            _log(f"  {exp:6.1f} ms  <could not set: {e}>")
            continue
        st = _snap_stats(mmc, label)
        if st is None:
            continue
        mn, mx, mean, std = st
        flag = "  <-- VARYING" if mx != mn else ""
        _log(f"  {exp:6.1f} ms  min={mn:8.1f} max={mx:8.1f} "
             f"mean={mean:9.1f} std={std:7.1f}{flag}")
        if mx != mn:
            found = True

    if has("ShutterMode"):
        _log()
        _log("ShutterMode sweep:")
        try:
            original = mmc.getProperty(label, "ShutterMode")
        except Exception:
            original = None
        for val in ("Never", "Pre-Exposure"):
            try:
                mmc.setProperty(label, "ShutterMode", val)
            except Exception as e:
                _log(f"  {val:16s} <could not set: {e}>")
                continue
            st = _snap_stats(mmc, label)
            if st is None:
                continue
            mn, mx, mean, std = st
            flag = "  <-- VARYING" if mx != mn else ""
            _log(f"  {val:16s} min={mn:8.1f} max={mx:8.1f} "
                 f"std={std:7.1f}{flag}")
            if mx != mn:
                found = True
        if original is not None:
            try:
                mmc.setProperty(label, "ShutterMode", original)
            except Exception:
                pass

    if has("ReadoutRate"):
        _log()
        _log("ReadoutRate sweep — THE decisive test:")
        _log("  If the constant follows the bit depth (65535 at 16-bit,")
        _log("  4095 at 12-bit) it is an unread buffer, not light.")
        try:
            original = mmc.getProperty(label, "ReadoutRate")
            opts = list(
                mmc.getAllowedPropertyValues(label, "ReadoutRate")
            )
        except Exception:
            original, opts = None, []
        for val in opts:
            try:
                mmc.setProperty(label, "ReadoutRate", val)
            except Exception as e:
                _log(f"  {val:16s} <could not set: {e}>")
                continue
            st = _snap_stats(mmc, label)
            if st is None:
                continue
            mn, mx, mean, std = st
            note = ""
            if mx == mn:
                if abs(mn - 4095) < 1:
                    note = "  <-- 12-bit max: FILL PATTERN confirmed"
                elif abs(mn - 65535) < 1:
                    note = "  <-- 16-bit max"
            else:
                note = "  <-- VARYING"
                found = True
            _log(f"  {val:16s} min={mn:8.1f} max={mx:8.1f} "
                 f"std={std:7.1f}{note}")
        if original is not None:
            try:
                mmc.setProperty(label, "ReadoutRate", original)
            except Exception:
                pass

    return found


def make_core(buffer_mb: int):
    try:
        from pymmcore_plus import CMMCorePlus
    except ImportError:
        _log("FAIL: pymmcore-plus not installed.")
        _log("      pip install -r requirements-mm.txt")
        sys.exit(2)
    mmc = CMMCorePlus()
    try:
        mmc.setCircularBufferMemoryFootprint(buffer_mb)
    except Exception as e:
        _log(f"WARN: could not set circular buffer: {e}")
    return mmc


def _retry_with_internal_trigger(mmc, label, names, exposure) -> bool:
    """Force an internal/software trigger, then re-snap.

    A uniform frame almost always means the exposure never fired. PVCAM
    exposes the trigger source as an enum property whose exact name and
    values vary by adapter version, so search for it and try every
    value that looks internal.
    """
    candidates = [
        p for p in names
        if "trigger" in p.lower() and "mode" in p.lower()
    ] or [p for p in names if "trigger" in p.lower()]

    if not candidates:
        _log()
        _log("  No trigger property to adjust.")
        return False

    internal_hints = ("internal", "software", "timed", "normal", "free")

    for prop in candidates:
        try:
            opts = list(mmc.getAllowedPropertyValues(label, prop))
        except Exception:
            opts = []
        if not opts:
            continue
        targets = [
            o for o in opts
            if any(h in o.lower() for h in internal_hints)
        ]
        if not targets:
            continue

        for target in targets:
            _log()
            _log(f"  Retrying with {prop} = {target!r} ...")
            try:
                mmc.setProperty(label, prop, target)
                mmc.setExposure(exposure)
                mmc.snapImage()
                img = mmc.getImage()
            except Exception as e:
                _log(f"    failed: {type(e).__name__}: {e}")
                continue
            mn, mx = float(img.min()), float(img.max())
            _log(f"    min={mn:.1f} max={mx:.1f} "
                 f"mean={float(img.mean()):.1f} "
                 f"std={float(img.std()):.1f}")
            if mx != mn:
                _log(f"    *** SUCCESS — real image with "
                     f"{prop}={target!r} ***")
                _log(f"    Put this in the .cfg / backend config.")
                return True
    return False


def try_camera(mmc, library: str, device: str, exposure: float) -> bool:
    """Load one camera slot and look for proof it is real."""
    _hdr(f"{library} / {device}")
    label = "Cam"

    try:
        mmc.unloadAllDevices()
    except Exception:
        pass

    try:
        mmc.loadDevice(label, library, device)
    except Exception as e:
        _log(f"loadDevice failed: {type(e).__name__}: {e}")
        return False
    _log("loadDevice   OK")

    try:
        mmc.initializeDevice(label)
    except Exception as e:
        _log(f"initializeDevice failed: {type(e).__name__}: {e}")
        _log()
        _log("An empty slot usually fails HERE. If every slot fails "
             "this way, the camera is off, unplugged, or held by "
             "another process (ZEN).")
        return False
    _log("initialize   OK  <-- NOTE: also succeeds on empty slots")

    try:
        mmc.setCameraDevice(label)
    except Exception as e:
        _log(f"setCameraDevice failed: {type(e).__name__}: {e}")
        return False

    # --- Positive evidence #1: geometry ---
    w = h = depth = None
    try:
        w, h = mmc.getImageWidth(), mmc.getImageHeight()
        depth = mmc.getImageBitDepth()
        _log(f"geometry     {w} x {h}, {depth}-bit")
    except Exception as e:
        _log(f"geometry     <unavailable: {e}>")

    # --- Positive evidence #2: identity properties ---
    _log()
    _log("Identity properties:")
    identity_found = 0
    try:
        names = list(mmc.getDevicePropertyNames(label))
    except Exception as e:
        names = []
        _log(f"  <could not list properties: {e}>")
    for prop in names:
        if not any(h in prop.lower() for h in IDENTITY_HINTS):
            continue
        try:
            val = mmc.getProperty(label, prop)
        except Exception:
            continue
        marker = ""
        if val not in ("", "0", "Unknown", "N/A", "None"):
            identity_found += 1
            marker = "  <-- populated"
        _log(f"  {prop} = {val!r}{marker}")
    if not names:
        _log("  <none>")
    _log(f"  ({identity_found} populated identity properties)")

    # --- Positive evidence #3: a snap with real structure ---
    _log()
    ok_snap = False
    try:
        mmc.setExposure(exposure)
        mmc.snapImage()
        img = mmc.getImage()
        mn, mx = float(img.min()), float(img.max())
        mean, std = float(img.mean()), float(img.std())
        _log(f"snap         shape={img.shape} dtype={img.dtype}")
        _log(f"             min={mn:.1f} max={mx:.1f} "
             f"mean={mean:.1f} std={std:.1f}")
        if mx == mn:
            _log(f"             ^ CONSTANT image (every pixel "
                 f"{mn:.0f}).")
            full = (1 << (depth or 16)) - 1
            if mn >= full:
                _log("             Pegged at the 16-bit maximum with "
                     "zero variance. Real saturation from light still "
                     "shows hot pixels and edge falloff, so this is "
                     "most likely an UNREAD BUFFER, not bright light.")
                _log("             Prime suspect: the camera is in an "
                     "EXTERNAL TRIGGER mode (this scope has trigger "
                     "wiring per MTB's <Trigger>SVB1_Camera1Ports</>), "
                     "so the exposure never fires.")
            elif mn == 0:
                _log("             All zero — sensor not read, or "
                     "light path fully closed.")
        else:
            ok_snap = True
    except Exception as e:
        _log(f"snap failed: {type(e).__name__}: {e}")
        _log()
        _log("If this is a buffer-read error, it is the known PVCAM +"
             " pymmcore race. Retry with a bigger --buffer-mb, and "
             "compare against MMStudio's own Snap.")

    # --- Trigger / readout diagnosis ---
    # A constant frame is usually a trigger-mode problem, so dump the
    # properties that govern how an exposure is initiated and retry
    # after forcing an internal/software trigger.
    if not ok_snap:
        _log()
        _log("Trigger / readout properties:")
        trig_props = [
            p for p in names
            if any(k in p.lower() for k in
                   ("trigger", "exposure", "readout", "port", "speed",
                    "gain", "clear", "mode", "shutter", "binning"))
        ]
        for prop in trig_props:
            try:
                val = mmc.getProperty(label, prop)
            except Exception:
                continue
            allowed = ""
            try:
                opts = list(mmc.getAllowedPropertyValues(label, prop))
                if opts:
                    allowed = f"   allowed: {opts}"
            except Exception:
                pass
            _log(f"  {prop} = {val!r}{allowed}")
        if not trig_props:
            _log("  <none found>")

        ok_snap = _retry_with_internal_trigger(
            mmc, label, names, exposure
        )
        if not ok_snap and SWEEP_REQUESTED:
            ok_snap = sweep(mmc, label, names)
        elif not ok_snap:
            _log()
            _log("  Re-run with --sweep to vary exposure, ShutterMode "
                 "and ReadoutRate systematically.")

    # --- Verdict ---
    _log()
    real = ok_snap and identity_found > 0
    if real:
        _log("VERDICT: looks like a REAL camera "
             "(identity populated + varying image).")
    elif ok_snap:
        _log("VERDICT: image varies but identity is empty — "
             "probably real, worth confirming the model name.")
    elif identity_found:
        _log("VERDICT: identity populated but no usable image — "
             "camera present but not delivering frames. Check power "
             "cycle order and that nothing else holds it.")
    else:
        _log("VERDICT: looks like a PHANTOM slot. No identity, no "
             "image.")
    return real


def main() -> int:
    global SWEEP_REQUESTED
    args = parse_args()
    SWEEP_REQUESTED = args.sweep
    mmc = make_core(args.buffer_mb)

    try:
        available = list(mmc.getAvailableDevices(args.library))
    except Exception as e:
        _log(f"FAIL: library {args.library!r} not loadable: {e}")
        return 3

    _log(f"{args.library} advertises {len(available)} device(s): "
         f"{', '.join(available) or '<none>'}")
    _log("NOTE: PVCAM advertises fixed slots whether or not hardware "
         "is attached — this list proves nothing on its own.")

    if args.device:
        targets = [args.device]
    elif args.all:
        targets = available
    else:
        targets = available[:1]

    if not targets:
        _log("FAIL: no devices to try.")
        return 3

    any_real = False
    for dev in targets:
        if try_camera(mmc, args.library, dev, args.exposure):
            any_real = True

    try:
        mmc.unloadAllDevices()
    except Exception:
        pass

    _hdr("SUMMARY")
    if any_real:
        _log("At least one real camera responded.")
        return 0
    _log("No slot produced convincing evidence of a real camera.")
    _log()
    _log("Next steps, cheapest first:")
    _log("  1. Confirm the camera's own power LED is on.")
    _log("  2. Make sure ZEN is fully closed (it holds PVCAM "
         "exclusively): kill zen.exe / zenblue.exe.")
    _log("  3. Then, and only then, do the documented restart: "
         "PC -> camera -> microscope.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
