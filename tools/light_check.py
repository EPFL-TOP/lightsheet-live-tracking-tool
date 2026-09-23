"""Does the software actually make this stand dark?

Every dose estimate in the codebase assumes that `all_off()` and
`set_open(False)` extinguish the light. Nothing has ever verified that
against the sensor. This does, with the camera as the witness:

  * prints the TL shutter's ELEMENT NAMES, because `set_open()` decides
    which position is closed by string-matching them ('open' /
    'closed'/'close'/'shut'/'blocked'). If they are named anything else
    it raises on every call, and `_illuminate` swallows that into a log
    line — the transmitted light would then never be gated at all, for
    a whole overnight run, invisibly.
  * commands each source on and off and snaps a frame each way. A
    frame that does not return to the read-noise floor when the light
    is commanded off is a light that stayed on.
  * times the MTB round trip, which is the one number the phototoxicity
    arithmetic needs and nobody has ever measured.
  * probes for a motorized field stop / aperture / attenuator, i.e.
    whether the ILLUMINATED AREA could be under software control.

Run it with NO SAMPLE on the stage (or the plate removed). It moves no
axis. Nothing is written.

    python tools\light_check.py                 # TL + every Colibri LED
    python tools\light_check.py --percent 20    # LED intensity to test
    python tools\light_check.py --no-camera     # names and timings only
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# Relative to THIS file, not the working directory: the other tools all
# do this so they run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

try:
    from tracking_tools.microscope_interface.mtb import MTBSession
except Exception as _e:          # pragma: no cover - env dependent
    MTBSession = None
    _IMPORT_ERROR = _e

# Candidate components that would put the illuminated AREA, or a second
# fluorescence gate, under software control. None is in MTB_IDS today.
CANDIDATES = [
    "MTBFLLEDShutter", "MTBRLShutter", "MTBLampShutter",
    "MTBTLFieldStop", "MTBTLFieldDiaphragm", "MTBTLLuminousFieldStop",
    "MTBRLFieldStop", "MTBTLAperture", "MTBApertureChanger",
    "MTBAttenuator", "MTBNDFilter", "MTBTLHalogenLamp", "MTBRLLamp",
]


def _hdr(t):
    print(f"\n=== {t} " + "=" * max(0, 60 - len(t)))


def _stats(mmc, exposure_ms):
    import numpy as np
    mmc.setExposure(float(exposure_ms))
    mmc.snapImage()
    img = np.asarray(mmc.getImage())
    return float(img.mean()), float(img.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percent", type=float, default=20.0)
    ap.add_argument("--exposure", type=float, default=20.0)
    ap.add_argument("--no-camera", action="store_true")
    a = ap.parse_args()

    if MTBSession is None:
        print(f"Cannot import the MTB layer: {_IMPORT_ERROR}",
              flush=True)
        return 2
    try:
        s = MTBSession.shared().connect()
    except Exception as e:
        # A readable line rather than a traceback, like the other
        # hardware tools.
        print(f"Cannot reach the microscope: {e}", flush=True)
        return 2
    mmc = None
    if not a.no_camera:
        try:
            from pymmcore_plus import CMMCorePlus
            mmc = CMMCorePlus()
            mmc.loadDevice("Camera", "PVCAM", "Camera-1")
            mmc.initializeDevice("Camera")
            mmc.setCameraDevice("Camera")
        except Exception as e:
            print(f"camera unavailable ({e}) — continuing without it")
            mmc = None

    sh = s.changer("tl_shutter")
    col = s.colibri()

    _hdr("WHAT set_open() HAS TO MATCH")
    # If neither name contains 'open'/'closed', set_open() raises every
    # time and the transmitted light is never gated.
    print(sh.describe())
    names = [(e["name"] or "").lower() for e in sh.elements()]
    ok = any("open" in n for n in names) and (
        any(k in n for n in names for k in ("closed", "close", "shut"))
        or len(names) == 2
    )
    print("set_open() can resolve these names:", "YES" if ok else "NO <== "
          "TL IS NEVER GATED; _illuminate swallows the error")

    _hdr("COLIBRI")
    print(col.describe())

    if mmc is not None:
        _hdr("DOES THE LIGHT ACTUALLY GO OUT?")
        col.all_off()
        sh.set_open(False)
        floor = _stats(mmc, a.exposure)
        print(f"everything commanded OFF   mean={floor[0]:8.1f} "
              f"std={floor[1]:6.1f}   <- read-noise floor")

        sh.set_open(True)
        lit = _stats(mmc, a.exposure)
        sh.set_open(False)
        back = _stats(mmc, a.exposure)
        print(f"TL commanded OPEN          mean={lit[0]:8.1f}")
        print(f"TL commanded CLOSED again  mean={back[0]:8.1f}"
              f"   {'OK' if back[0] < floor[0] * 1.2 else '<== STILL LIT'}")
        if lit[0] < floor[0] * 1.2:
            print("   TL never illuminated the sensor: check the sideport "
                  "(camera is on LEFT) and the condenser (a PH ring "
                  "blocks it).")

        for led in col.leds():
            try:
                col.select(led.wavelength, a.percent)
                on = _stats(mmc, a.exposure)
                col.all_off()
                off = _stats(mmc, a.exposure)
                flag = "OK" if off[0] < floor[0] * 1.2 else "<== STILL LIT"
                print(f"{led.wavelength:>4} nm @ {a.percent:g}%   "
                      f"on={on[0]:8.1f}  off={off[0]:8.1f}   {flag}")
            except Exception as e:
                print(f"{led.wavelength:>4} nm: {e}")
        col.all_off()
        sh.set_open(False)

    _hdr("MTB ROUND TRIP (the unmeasured number in every dose estimate)")
    t = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = sh.position
        t.append(time.perf_counter() - t0)
    print(f"property read      {1000 * statistics.median(t):7.2f} ms")
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        sh.elements()
        t.append(time.perf_counter() - t0)
    print(f"elements()         {1000 * statistics.median(t):7.2f} ms"
          "   <- set_open() does this 1-2x per call, uncached")
    t = []
    for _ in range(10):
        t0 = time.perf_counter()
        col.all_off()
        t.append(time.perf_counter() - t0)
    print(f"colibri.all_off()  {1000 * statistics.median(t):7.2f} ms"
          "   <- runs INSIDE the lit window")
    led = col.leds()[0] if col.leds() else None
    if led is not None:
        t = []
        for _ in range(10):
            t0 = time.perf_counter()
            led.set_on(False)
            t.append(time.perf_counter() - t0)
        print(f"one led.set_on()   {1000 * statistics.median(t):7.2f} ms")

    _hdr("IS THE ILLUMINATED AREA UNDER SOFTWARE CONTROL?")
    found = []
    for cid in CANDIDATES:
        try:
            comp = s.component(cid)
        except Exception:
            comp = None
        if comp is not None:
            found.append(cid)
            try:
                print(f"  PRESENT  {cid}  '{comp.Name}'")
            except Exception:
                print(f"  PRESENT  {cid}")
    if not found:
        print("  none of the field-stop / aperture / attenuator ids is "
              "configured.\n  The illuminated AREA is a manual knob on "
              "the stand. Close the\n  transmitted-light field diaphragm "
              "by hand; software cannot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
