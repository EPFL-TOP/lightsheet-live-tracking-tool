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
    return p.parse_args()


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
            _log("             ^ CONSTANT image — no real signal. "
                 "Either the sensor is not being read, or the light "
                 "path is fully dark.")
        else:
            ok_snap = True
    except Exception as e:
        _log(f"snap failed: {type(e).__name__}: {e}")
        _log()
        _log("If this is a buffer-read error, it is the known PVCAM +"
             " pymmcore race. Retry with a bigger --buffer-mb, and "
             "compare against MMStudio's own Snap.")

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
    args = parse_args()
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
