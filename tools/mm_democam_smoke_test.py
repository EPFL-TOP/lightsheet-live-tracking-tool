"""
End-to-end smoke test for ``MicroscopeInterface_Micromanager``.

Drives the Micro-Manager (pymmcore-plus) backend against the bundled
DemoCamera adapter with a ``DriftingGaussianEmbryo`` synthetic source,
and verifies that the closed-loop tracker + stage machinery converges.

The test runs entirely in-process, requires NO real microscope, and
completes in roughly 30 seconds on a laptop.

────────────────────────────────────────────────────────────────────────
PREREQUISITES
────────────────────────────────────────────────────────────────────────

    pip install pymmcore-plus useq-schema scipy
    mmcore install    # fetches the DemoCamera adapter binaries once

The DemoCamera .cfg lives at ``docs/mm_demo_config.cfg`` relative to
the repo root.  If that file is absent, the script falls back to
pymmcore-plus' bundled default demo config (loaded with no cfg path).

────────────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT VERIFIES
────────────────────────────────────────────────────────────────────────

PRIMARY (closed-loop convergence)
  Three synthetic scenes are placed at (0,0,0), (200,0,0), (400,0,0) µm
  with a known 5 µm/tp Y-drift.  A trivial center-of-mass tracker
  counter-moves the stage via ``relative_move`` every timepoint; by
  tp=9 the RESIDUAL embryo displacement must be < 3 px per scene.

REGRESSION (three bugs surfaced during adversarial review)
  A. Zero-drift no-deadlock.
     A 5-cycle run with zero ground-truth drift must terminate in
     under 30s wall-clock — proves the MDA cycle-advance is
     UNCONDITIONAL (Bug #1).
  B. Immediate cum_drift.
     ``relative_move`` must update ``_cum_drift`` synchronously
     (before returning), not lazily on the next MDA event (Bug #2).
  C. Baselines from config.
     ``_baseline_um`` must come from ``positions_config[...]['xyz_um']``,
     never from ``mmc.getXYPosition``, so that a scene the stage has
     never physically visited still has the correct anchor (Bug #3).

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────

    python tools/mm_democam_smoke_test.py

Exit code:
    0  — every assertion passed
    1  — any assertion failed (or environment missing)
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import tempfile
import traceback
from pathlib import Path

# ─── Make repo root importable regardless of cwd ─────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Environment probe — fail loudly with an install hint rather than a
# cryptic ImportError deep inside the interface.
try:
    import numpy as np
    import pymmcore_plus  # noqa: F401
    import useq  # noqa: F401
    from scipy import ndimage
except ImportError as _e:  # pragma: no cover
    print(f"MISSING DEPENDENCY: {_e}")
    print("Install with:")
    print("    pip install pymmcore-plus useq-schema scipy")
    print("    mmcore install")
    sys.exit(1)

from tracking_tools.microscope_interface.MicroscopeInterface import (
    MicroscopeInterface_Micromanager,
)
from tracking_tools.microscope_interface.synthetic_source import (
    DriftingGaussianEmbryo,
)


# ─── Constants ───────────────────────────────────────────────────────────────

DEMO_CFG_REL   = Path("docs") / "mm_demo_config.cfg"
PIXEL_SIZE_UM  = 0.347          # matches DriftingGaussianEmbryo default
IMAGE_SHAPE    = (512, 512)     # DriftingGaussianEmbryo default
IMAGE_CENTER   = np.array([IMAGE_SHAPE[1] / 2.0, IMAGE_SHAPE[0] / 2.0])
CHANNEL_PRESET = "Brightfield"


# ─── Pretty printing (matches tools/smoke_test_multipos.py style) ────────────

_START_TS = time.time()


def _t():
    return f"+{time.time() - _START_TS:6.2f}s"


def _step(n, title):
    bar = "─" * max(1, 60 - len(title))
    print()
    print(f"━━━ STEP {n} — {title} {bar}")


def _ok(msg):    print(f"  + {_t()}  {msg}")
def _info(msg):  print(f"    {_t()}  {msg}")
def _warn(msg):  print(f"  ! {_t()}  {msg}")
def _fail(msg):  print(f"  x {_t()}  {msg}")


# ─── Config helpers ──────────────────────────────────────────────────────────

def _resolve_demo_cfg() -> str:
    """Return the demo cfg path, or "" to let pymmcore-plus use its bundled default."""
    p = _REPO_ROOT / DEMO_CFG_REL
    if p.is_file():
        return str(p)
    _warn(f"{DEMO_CFG_REL} not found — falling back to pymmcore-plus bundled demo config")
    return ""


def _make_positions_config(dirpath: str) -> dict:
    """Three synthetic scenes at (0,0,0), (200,0,0), (400,0,0) µm."""
    xyz_by_scene = [
        (0.0,   0.0, 0.0),
        (200.0, 0.0, 0.0),
        (400.0, 0.0, 0.0),
    ]
    cfg = {}
    for i, xyz in enumerate(xyz_by_scene):
        name = f"scene_{i}"
        scene_dir = os.path.join(dirpath, name)
        os.makedirs(scene_dir, exist_ok=True)
        cfg[name] = {
            "xyz_um":     xyz,
            "filename":   "t0000_Brightfield.tif",   # Viventis-style
            "images_dir": scene_dir,
        }
    return cfg


# ─── Trivial center-of-mass tracker ──────────────────────────────────────────

def _embryo_center_px(image: np.ndarray) -> np.ndarray:
    """Center of mass of the (thresholded) image, returned as [x, y] px."""
    arr = image.astype(np.float32)
    # Threshold at mean + 1σ — robust to noise + baseline bias
    thr = arr.mean() + arr.std()
    mask = arr > thr
    if not mask.any():
        # Fall back to unthresholded COM if the blob is very faint
        cy, cx = ndimage.center_of_mass(arr)
    else:
        cy, cx = ndimage.center_of_mass(arr, mask)
    return np.array([cx, cy], dtype=np.float64)


def _shift_um_from_image(image: np.ndarray, pixel_size_um: float):
    """Compute the µm shift that would center the blob.

    ``shift = (image_center - embryo_center) * pixel_size``.  Applied via
    ``relative_move`` — the synthetic source subtracts the stage position
    from the world position, so a positive shift in stage µm moves the
    stage toward the blob (blob appears to move toward image center).
    """
    embryo_px = _embryo_center_px(image)
    residual_px = embryo_px - IMAGE_CENTER
    shift_px = -residual_px
    shift_um = shift_px * pixel_size_um
    return float(shift_um[0]), float(shift_um[1]), embryo_px, residual_px


# ─── The three tests ─────────────────────────────────────────────────────────

def _run_convergence(demo_cfg: str, results: dict) -> None:
    """PRIMARY assertion: residual < 3 px per scene at t=9."""
    _step(1, "PRIMARY — closed-loop convergence with 5 µm/tp Y drift")

    dirpath = tempfile.mkdtemp(prefix="mm_smoke_primary_")
    _info(f"scratch dir: {dirpath}")

    try:
        positions_config = _make_positions_config(dirpath)
        _info(f"positions: {list(positions_config.keys())}")

        embryo = DriftingGaussianEmbryo(
            shape=IMAGE_SHAPE,
            sigma=15,
            drift_um_per_tp=(0.0, 5.0, 0.0),
            pixel_size_um=PIXEL_SIZE_UM,
            noise_std=5.0,
            seed=42,
        )

        # ── Regression C(bonus): monkey-patch mmc.getXYPosition to trip if the
        # interface tries to read the stage during baseline capture.  The guard
        # MUST be armed BEFORE MicroscopeInterface_Micromanager(...) is called,
        # because baselines are populated in __init__.
        from pymmcore_plus import CMMCorePlus
        _mmc = CMMCorePlus.instance()
        _orig_getXYPosition = _mmc.getXYPosition
        _baseline_capture_active = {"on": False}
        _bad_calls: list[str] = []

        def _guarded_getXYPosition(*a, **kw):
            if _baseline_capture_active["on"]:
                _bad_calls.append("getXYPosition called during baseline capture")
            return _orig_getXYPosition(*a, **kw)

        try:
            _mmc.getXYPosition = _guarded_getXYPosition
        except Exception as e:
            _warn(f"could not monkey-patch mmc.getXYPosition: {e}")

        _baseline_capture_active["on"] = True
        try:
            microscope = MicroscopeInterface_Micromanager(
                positions_config=positions_config,
                dirpath=dirpath,
                mm_params={
                    "cfg_path":         demo_cfg,
                    "channel_preset":   CHANNEL_PRESET,
                    "exposure_ms":      10.0,
                    "interval_s":       0.0,
                    "synthetic_source": embryo,
                    "stop_after_tp":    10,
                },
            )
            microscope.connect()
        finally:
            # After connect returns, baseline capture is over; allow subsequent
            # stage reads (there shouldn't be any either, but only the baseline
            # phase is what Bug #3 is about).
            _baseline_capture_active["on"] = False
        _ok("microscope.connect() returned")

        # ── Regression C: baseline anchors come from positions_config
        baseline_mismatch = None
        for pos_name, cfg in positions_config.items():
            want = tuple(cfg["xyz_um"])
            got = microscope._baseline_um.get(pos_name)
            if got is None or tuple(got) != want:
                baseline_mismatch = (
                    f"[{pos_name}] baseline {got} != positions_config {want}"
                )
                _fail(f"[{pos_name}] baseline mismatch: got {got}, want {want}")
            else:
                _info(f"[{pos_name}] baseline OK: {got}")
        if baseline_mismatch is None:
            results["C_baselines_from_config"] = (True, "all baselines match config")
        else:
            results["C_baselines_from_config"] = (False, baseline_mismatch)

        if _bad_calls:
            results["C_no_stage_read_during_baseline"] = (
                False, f"{len(_bad_calls)} bad calls: {_bad_calls[:3]}"
            )
        else:
            results["C_no_stage_read_during_baseline"] = (
                True, "mmc.getXYPosition NOT called during baseline capture"
            )

        # ── Driver loop
        last_residual_px = {p: None for p in positions_config}
        frames_seen = 0
        for _ in range(30):
            img, tp, pos = microscope.wait_for_image(timeout_ms=5000)
            if img is None:
                _info(f"wait_for_image returned None after {frames_seen} frames — stop")
                break
            frames_seen += 1
            shift_x_um, shift_y_um, embryo_px, residual_px = _shift_um_from_image(
                img, PIXEL_SIZE_UM
            )
            last_residual_px[pos] = residual_px
            _info(
                f"[{pos}] tp={tp} embryo_px=({embryo_px[0]:6.1f},{embryo_px[1]:6.1f}) "
                f"residual_px=({residual_px[0]:+6.2f},{residual_px[1]:+6.2f}) "
                f"→ move µm=({shift_x_um:+.2f},{shift_y_um:+.2f})"
            )
            microscope.relative_move(pos, shift_x_um, shift_y_um, 0.0)

        _info(f"driver loop finished after {frames_seen} frames")

        microscope.stop()
        _ok("microscope.stop() returned")

        # Restore
        try:
            _mmc.getXYPosition = _orig_getXYPosition
        except Exception:
            pass

        # ── PRIMARY assertion: residual < 3 px per scene at t=9
        for pos, res in last_residual_px.items():
            if res is None:
                results[f"PRIMARY_convergence[{pos}]"] = (
                    False, "no frame received for scene",
                )
                _fail(f"[{pos}] no frames received")
                continue
            mag = float(np.hypot(res[0], res[1]))
            ok = mag < 3.0
            results[f"PRIMARY_convergence[{pos}]"] = (
                ok,
                f"residual |Δpx| = {mag:.2f} (< 3.0 required)",
            )
            (_ok if ok else _fail)(f"[{pos}] residual |Δpx|={mag:.2f}")
    finally:
        shutil.rmtree(dirpath, ignore_errors=True)


def _run_zero_drift_no_deadlock(demo_cfg: str, results: dict) -> None:
    """Regression A: 5-cycle zero-drift run must finish in < 30 s."""
    _step(2, "REGRESSION A — zero-drift no-deadlock")

    dirpath = tempfile.mkdtemp(prefix="mm_smoke_zerodrift_")
    try:
        positions_config = _make_positions_config(dirpath)
        embryo = DriftingGaussianEmbryo(
            shape=IMAGE_SHAPE,
            sigma=15,
            drift_um_per_tp=(0.0, 0.0, 0.0),
            pixel_size_um=PIXEL_SIZE_UM,
            noise_std=5.0,
            seed=42,
        )
        microscope = MicroscopeInterface_Micromanager(
            positions_config=positions_config,
            dirpath=dirpath,
            mm_params={
                "cfg_path":         demo_cfg,
                "channel_preset":   CHANNEL_PRESET,
                "exposure_ms":      10.0,
                "interval_s":       0.0,
                "synthetic_source": embryo,
                "stop_after_tp":    5,
            },
        )
        microscope.connect()
        _info("microscope connected — driving zero-drift loop")

        start = time.time()
        deadline = start + 30.0
        frames = 0
        while time.time() < deadline:
            img, tp, pos = microscope.wait_for_image(timeout_ms=2000)
            if img is None:
                break
            frames += 1
            # Even with zero drift the tracker still calls relative_move(0,0,0)
            # every timepoint — this exercises the code path fully.
            microscope.relative_move(pos, 0.0, 0.0, 0.0)
        elapsed = time.time() - start

        microscope.stop()

        ok = (elapsed < 30.0) and (frames > 0)
        results["A_zero_drift_no_deadlock"] = (
            ok,
            f"elapsed={elapsed:.2f}s frames={frames} (< 30 s required, > 0 frames)",
        )
        (_ok if ok else _fail)(
            f"5-cycle zero-drift run: elapsed={elapsed:.2f}s frames={frames}"
        )
    finally:
        shutil.rmtree(dirpath, ignore_errors=True)


def _run_immediate_cum_drift(demo_cfg: str, results: dict) -> None:
    """Regression B: relative_move must update _cum_drift synchronously."""
    _step(3, "REGRESSION B — immediate _cum_drift update")

    dirpath = tempfile.mkdtemp(prefix="mm_smoke_bugB_")
    try:
        positions_config = _make_positions_config(dirpath)

        # Do NOT call connect() — we don't need the MDA engine to spin up
        # for this test.  __init__ is enough to set up _cum_drift.
        microscope = MicroscopeInterface_Micromanager(
            positions_config=positions_config,
            dirpath=dirpath,
            mm_params={
                "cfg_path":         demo_cfg,
                "channel_preset":   CHANNEL_PRESET,
                "synthetic_source": None,
                "stop_after_tp":    1,
            },
        )

        microscope.relative_move("scene_0", 3.14, -1.59, 0.0)
        got = list(microscope._cum_drift["scene_0"])
        want = [3.14, -1.59, 0.0]

        ok = all(abs(g - w) < 1e-9 for g, w in zip(got, want))
        results["B_immediate_cum_drift"] = (
            ok, f"_cum_drift[scene_0]={got} want={want}",
        )
        (_ok if ok else _fail)(f"_cum_drift['scene_0']={got} want={want}")
    finally:
        shutil.rmtree(dirpath, ignore_errors=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    demo_cfg = _resolve_demo_cfg()
    if demo_cfg:
        _info(f"demo cfg: {demo_cfg}")
    results: dict[str, tuple[bool, str]] = {}

    try:
        _run_immediate_cum_drift(demo_cfg, results)
    except Exception:
        traceback.print_exc()
        results["B_immediate_cum_drift"] = (False, "raised exception (see traceback)")

    try:
        _run_zero_drift_no_deadlock(demo_cfg, results)
    except Exception:
        traceback.print_exc()
        results["A_zero_drift_no_deadlock"] = (False, "raised exception (see traceback)")

    try:
        _run_convergence(demo_cfg, results)
    except Exception:
        traceback.print_exc()
        # If convergence blows up mid-way, still record what we can.
        results.setdefault(
            "PRIMARY_convergence[fatal]", (False, "raised exception (see traceback)")
        )

    # ── Summary
    _step(4, "SUMMARY")
    passes = fails = 0
    for name in sorted(results):
        ok, detail = results[name]
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}: {detail}")
        if ok:
            passes += 1
        else:
            fails += 1
    print()
    print(f"  {passes} passed, {fails} failed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
