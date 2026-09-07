"""MicroscopeInterface_MTB — Zeiss motion via MTB + camera via PVCAM.

The backend for the EPFL Axio Observer 7. Implements the same contract
as the LS1 / ZEN / Files / Micro-Manager backends, so TrackingRunner
and the panel need no special-casing.

WHY THIS EXISTS, AND WHY IT SOLVES THE ORIGINAL PROBLEM

The project's blocker was that ZEN cannot change stored positions while
an experiment runs, so a closed drift-correction loop was impossible.
Micro-Manager was the intended escape, but its serial ZeissCAN29
adapter cannot reach an Observer 7 (that stand carries CAN29 over USB
into Zeiss's CZCanSrv).

The resolution splits ownership across two INDEPENDENT subsystems:

    camera   -> Micro-Manager / PVCAM
    motion   -> Zeiss MTB 2011 (via pythonnet)
    timing   -> this module's own loop

Nothing arbitrates between them, so moving the stage mid-acquisition is
simply a call. Note MTB is Zeiss's hardware abstraction *service*,
architecturally equivalent to a Micro-Manager device adapter — using it
does not mean running ZEN, which is merely another MTB client.

Because we own the time loop, there is no MDA event queue. That removes
the machinery responsible for three regression bugs in the
Micro-Manager backend; the guards against reintroducing them are:

  1. The loop advances UNCONDITIONALLY, so zero drift cannot deadlock.
  2. relative_move() updates _cum_drift synchronously before returning.
  3. Baselines come only from positions_config[pos]['xyz_um'] — never
     from reading the stage, which would bake existing drift into the
     reference.

Verified hardware behaviour (2026-09-03): XY/focus move with 0.000 µm
error, piezo with 0.082 µm; camera is a Prime 95B (GS144BSI, 1200x1200,
16-bit) needing ~1-5 ms exposure at current illumination.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time

import numpy as np

from .mtb import MTBError, MTBMotion, MTBSession

logger = logging.getLogger(__name__)


class MicroscopeInterface_MTB:
    """Closed-loop tracking backend: MTB motion + PVCAM camera.

    mtb_params keys (all optional unless noted):

      dll_path        MTBApi.dll location (default: the standard path)
      z_axis          'piezo' (default) or 'focus'
      camera_library  MM device library     (default 'PVCAM')
      camera_device   MM device name        (default 'Camera-1')
      exposure_ms     camera exposure       (default 5.0 — the Prime
                      95B saturates by ~100 ms at current light)
      channel         label used in frame filenames (default 'BF')
      interval_s      seconds between timepoints (default 60)
      settle_s        pause after a move before snapping (default 0.3)
      max_xy_um       per-correction XY clamp (default 50)
      max_z_um        per-correction Z clamp  (default 20)
      z_stack         {'range_um': float, 'step_um': float} or None
      stop_after_tp   stop after this many timepoints, or None
      buffer_mb       MMCore circular buffer (default 512)
      synthetic_source  callable(x, y, z, pos_name) -> ndarray, used
                      instead of the camera for offline testing
      session         an existing MTBSession to reuse. Omit and the
                      process-wide MTBSession.shared() is used, which
                      is what you almost always want — MTB permits
                      only ONE Login per process, so a backend that
                      created its own session would break any GUI
                      holding one.
    """

    def __init__(self, positions_config: dict, dirpath: str,
                 mtb_params: dict | None = None):
        p = dict(mtb_params or {})
        self.positions_config = positions_config
        self.dirpath = dirpath

        self.dll_path = p.get("dll_path")
        self.z_axis = p.get("z_axis", "piezo")
        self.camera_library = p.get("camera_library", "PVCAM")
        self.camera_device = p.get("camera_device", "Camera-1")
        self.exposure_ms = float(p.get("exposure_ms", 5.0))
        self.channel = p.get("channel", "BF")
        self.interval_s = float(p.get("interval_s", 60.0))
        self.settle_s = float(p.get("settle_s", 0.3))
        self.max_xy_um = float(p.get("max_xy_um", 50.0))
        self.max_z_um = float(p.get("max_z_um", 20.0))
        self.z_stack = p.get("z_stack")
        self.stop_after_tp = p.get("stop_after_tp")
        self.buffer_mb = int(p.get("buffer_mb", 512))
        self.synthetic_source = p.get("synthetic_source")
        # Multi-channel: each entry is
        #   {name, reflector_pos, exposure_ms, z_stack}
        # Channels are acquired SEQUENTIALLY per position — one camera
        # and one filter cube means they cannot be simultaneous.
        # apply_channel(channel) puts the hardware into that channel's
        # configuration; the GUI supplies it so the reflector cube and
        # exposure are set the same way whether snapping by hand or
        # running the loop.
        self.channels = p.get("channels") or [{
            "name": p.get("channel", "BF"),
            "reflector_pos": 0,
            "exposure_ms": self.exposure_ms,
            "z_stack": bool(p.get("z_stack")),
        }]
        self.apply_channel = p.get("apply_channel")
        # An injected session belongs to the caller: we must not log it
        # out, because MTB cannot Login again in this process.
        self._injected_session = p.get("session")
        self._owns_session = False
        # PVCAM allows ONE open handle per camera, so a GUI that has
        # already opened it must hand the core over rather than let us
        # open a second one:
        #   pl_cam_open failed, pvErr:12,
        #   'This user has already opened this camera'
        self._injected_mmc = p.get("mmc")
        self._injected_cam_label = p.get("camera_label")
        self._owns_camera = False

        self.pos_names = list(positions_config.keys())
        if not self.pos_names:
            raise ValueError("positions_config is empty")

        # GUARD 3: baselines come from the config, never from the
        # stage. Reading the stage here would treat whatever drift has
        # already occurred as the reference position.
        self._baseline_um: dict[str, tuple[float, float, float]] = {}
        for name in self.pos_names:
            entry = positions_config[name]
            if "xyz_um" not in entry:
                raise ValueError(
                    f"position {name!r} has no 'xyz_um' baseline — "
                    f"the tracker cannot establish a reference"
                )
            x, y, z = entry["xyz_um"]
            self._baseline_um[name] = (float(x), float(y), float(z))

        self._cum_drift: dict[str, list[float]] = {
            name: [0.0, 0.0, 0.0] for name in self.pos_names
        }
        self._drift_lock = threading.Lock()
        self._config_lock = threading.Lock()

        self._image_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Set when the acquisition loop exits, so stop_requested can
        # distinguish "timed out" from "no more frames are coming".
        self._finished = False

        self.session: MTBSession | None = None
        self.motion: MTBMotion | None = None
        self.mmc = None
        self._camera_label = "TrackingCam"
        self._current_tp = 0

    # ------------------------------------------------------ lifecycle

    def connect(self) -> None:
        """Open MTB and the camera, then start the acquisition loop.

        Idempotent. TrackingRunner.run_zeiss() calls connect() itself,
        and the setup GUI starts acquisition BEFORE any ROIs exist so
        the operator can draw them on real frames — so connect() gets
        called twice on the same backend. Without this guard the second
        call would start a second acquisition thread and both would
        drive the stage.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.info(
                "acquisition already running — connect() is a no-op"
            )
            return

        logger.info("MTB backend connecting")

        if self._injected_session is not None:
            self.session = self._injected_session
            logger.info("reusing the caller's MTB session")
        else:
            kwargs = {"dll_path": self.dll_path} if self.dll_path else {}
            # shared() rather than MTBSession(...): a second Login in
            # this process would fail with E_NOINTERFACE.
            self.session = MTBSession.shared(**kwargs)
        self.motion = MTBMotion(self.session, z_axis=self.z_axis)
        logger.info("MTB motion ready:\n%s", self.motion.describe())

        if self.synthetic_source is None:
            self._open_camera()
        else:
            logger.info("synthetic_source supplied — camera not opened")

        for warning in self.check_z_stack():
            logger.warning("Z-STACK: %s", warning)

        self._stop_event.clear()
        self._finished = False
        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name="mtb-acquisition",
            daemon=True,
        )
        self._thread.start()
        logger.info("acquisition loop started")

    def _open_camera(self) -> None:
        if self._injected_mmc is not None:
            # Reuse the caller's already-open camera. Opening a second
            # PVCAM handle fails with C0_CAM_ALREADY_OPEN.
            self.mmc = self._injected_mmc
            if self._injected_cam_label:
                self._camera_label = self._injected_cam_label
            self._owns_camera = False
            with self._config_lock:
                try:
                    self.mmc.setCameraDevice(self._camera_label)
                    self.mmc.setExposure(self.exposure_ms)
                except Exception as e:
                    logger.warning(
                        "could not configure the shared camera: %s", e
                    )
            logger.info(
                "reusing the caller's camera %r, exposure %.1f ms",
                self._camera_label, self.exposure_ms,
            )
            return

        try:
            from pymmcore_plus import CMMCorePlus
        except ImportError as e:
            raise MTBError(
                "pymmcore-plus is required for the camera — "
                "pip install -r requirements-mm.txt"
            ) from e

        self._owns_camera = True
        self.mmc = CMMCorePlus()
        try:
            # PVCAM races MMCore's sequence buffer when it is tight.
            self.mmc.setCircularBufferMemoryFootprint(self.buffer_mb)
        except Exception as e:
            logger.warning("could not size circular buffer: %s", e)

        self.mmc.loadDevice(
            self._camera_label, self.camera_library, self.camera_device
        )
        self.mmc.initializeDevice(self._camera_label)
        self.mmc.setCameraDevice(self._camera_label)
        with self._config_lock:
            self.mmc.setExposure(self.exposure_ms)
        logger.info(
            "camera ready: %s %dx%d %d-bit, exposure %.1f ms",
            self.camera_device,
            self.mmc.getImageWidth(), self.mmc.getImageHeight(),
            self.mmc.getImageBitDepth(), self.exposure_ms,
        )

    def disconnect(self) -> None:
        self.stop()
        if self.mmc is not None:
            # Only tear down a camera we opened. Unloading a borrowed
            # one would leave the GUI holding a dead handle, and the
            # next PVCAM open would fail until the process restarted.
            if self._owns_camera:
                try:
                    self.mmc.unloadAllDevices()
                except Exception as e:
                    logger.warning("camera unload failed: %s", e)
            else:
                logger.info("leaving the borrowed camera open")
            self.mmc = None
        # Leave the MTB session alone. Logging out would make every
        # later Login in this process fail, stranding the GUI and any
        # subsequent run. The owner closes it at process exit via
        # MTBSession.close_shared().
        self.session = None
        self.motion = None
        with self._drift_lock:
            final = {k: list(v) for k, v in self._cum_drift.items()}
        logger.info("MTB backend disconnected; final drift: %s", final)

    def stop(self) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=max(5.0, self.settle_s * 4))
            if t.is_alive():
                logger.warning("acquisition thread did not stop cleanly")
        self._thread = None
        # Unblock any consumer parked in wait_for_image().
        self._image_queue.put(None)

    # ------------------------------------------------- acquisition

    def _target_for(self, pos_name: str) -> tuple[float, float, float]:
        """Absolute target = baseline + cumulative drift.

        Absolute rather than incremental: a dropped move costs one
        frame instead of permanently offsetting the series.
        """
        with self._config_lock:
            bx, by, bz = self._baseline_um[pos_name]
        with self._drift_lock:
            dx, dy, dz = self._cum_drift[pos_name]
        return (bx + dx, by + dy, bz + dz)

    def _acquisition_loop(self) -> None:
        """Visit every position each timepoint, forever.

        GUARD 1: this advances unconditionally. The Micro-Manager
        backend enqueued the next timepoint only when a correction
        arrived, which deadlocked whenever drift was exactly zero.
        """
        tp = 0
        try:
            while not self._stop_event.is_set():
                if (self.stop_after_tp is not None
                        and tp >= self.stop_after_tp):
                    logger.info("reached stop_after_tp=%s",
                                self.stop_after_tp)
                    break

                started = time.monotonic()
                self._current_tp = tp

                for pos_name in self.pos_names:
                    if self._stop_event.is_set():
                        break
                    try:
                        self._visit(pos_name, tp)
                    except Exception:
                        # One bad position must not kill the run.
                        logger.exception(
                            "timepoint %d, position %s failed",
                            tp, pos_name,
                        )

                tp += 1

                elapsed = time.monotonic() - started
                remaining = self.interval_s - elapsed
                if remaining > 0:
                    # wait() rather than sleep() so stop() is prompt.
                    self._stop_event.wait(remaining)
                else:
                    logger.warning(
                        "timepoint %d took %.1fs, longer than the "
                        "%.1fs interval — running without idle time",
                        tp - 1, elapsed, self.interval_s,
                    )
        finally:
            self._finished = True
            self._image_queue.put(None)
            logger.info("acquisition loop exited after %d timepoints",
                        tp)

    def _visit(self, pos_name: str, tp: int) -> None:
        """Move to one position, acquire every channel, save, publish.

        Only the FIRST channel is handed to the tracker: drift is
        measured once per position per timepoint, and mixing channels
        would correlate a brightfield frame against a fluorescence one.
        Put the channel you want to track on first — brightfield is
        usually the right choice, being brighter and structure-rich.
        Every channel is still written to disk.
        """
        x, y, z = self._target_for(pos_name)
        self.motion.move_to(x=x, y=y, z=z)
        if self.settle_s:
            time.sleep(self.settle_s)

        for idx, channel in enumerate(self.channels):
            if self._stop_event.is_set():
                return
            try:
                if self.apply_channel is not None:
                    self.apply_channel(channel)
                elif self.mmc is not None:
                    with self._config_lock:
                        self.mmc.setExposure(
                            float(channel["exposure_ms"])
                        )
            except Exception:
                logger.exception(
                    "could not apply channel %s", channel.get("name")
                )

            want_stack = bool(channel.get("z_stack")) and self.z_stack
            img = self._acquire(pos_name, x, y, z, stack=want_stack,
                                exposure_ms=channel.get("exposure_ms"))
            path = self._save(img, pos_name, tp,
                              channel=str(channel.get("name", "BF")))
            logger.debug("t=%d %s [%s] -> %s",
                         tp, pos_name, channel.get("name"), path)

            if idx == 0:
                self._image_queue.put((img, tp, pos_name))

    def _acquire(self, pos_name, x, y, z, stack=None,
                 exposure_ms=None) -> np.ndarray:
        """One frame, or one z-stack, for this position.

        The per-plane grabber is passed in so a z-stack works with the
        synthetic source too — an earlier version short-circuited on
        synthetic_source before reaching the stack branch, which made
        the stack path untestable without a camera.
        """
        if self.synthetic_source is not None:
            def grab(zz):
                return np.asarray(
                    self.synthetic_source(x, y, zz, pos_name)
                )
        else:
            def grab(zz):
                return self._snap(exposure_ms)

        want = self.z_stack if stack is None else (
            stack and self.z_stack
        )
        if want:
            return self._acquire_stack(z, grab)
        return grab(z)

    def _snap(self, exposure_ms=None) -> np.ndarray:
        with self._config_lock:
            self.mmc.setExposure(
                float(exposure_ms) if exposure_ms is not None
                else self.exposure_ms
            )
        self.mmc.snapImage()
        return np.asarray(self.mmc.getImage())

    def plane_offsets(self) -> list[float]:
        """Z offsets of the stack planes, symmetric about the centre.

        Derived from range and step. The plane count follows the step,
        and the offsets are then centred on zero so the stack is
        symmetric even when the range is not an exact multiple of the
        step — computing them as `-range/2 + i*step` left the last
        plane short and the stack lopsided.
        """
        rng = abs(float(self.z_stack.get("range_um", 10.0)))
        step = float(self.z_stack.get("step_um", 1.0))
        if step <= 0:
            raise ValueError(
                f"z_stack step_um must be positive, got {step}"
            )
        n = max(1, int(round(rng / step)) + 1)
        if n == 1:
            return [0.0]
        span = (n - 1) * step
        return [-span / 2.0 + i * step for i in range(n)]

    def check_z_stack(self) -> list[str]:
        """Problems that would silently corrupt a stack. Empty if fine.

        Both failure modes here produce DUPLICATE planes rather than an
        error, which is far worse than a refusal: the file looks like a
        valid stack and nothing flags it.
        """
        if not self.z_stack or self.motion is None:
            return []
        warnings = []
        offsets = self.plane_offsets()
        step = float(self.z_stack.get("step_um", 1.0))
        axis = self.motion.z

        # 1. A step finer than the axis can resolve: move_to() treats
        #    it as a no-op, so consecutive planes image the same slice.
        tol = axis.arrival_tolerance
        if step < tol:
            warnings.append(
                f"z-stack step {step:g} µm is below the {axis.label}'s "
                f"arrival tolerance ({tol:g} µm) — consecutive planes "
                f"would not actually move and would duplicate. Use a "
                f"step above {tol:g} µm, or switch Z actuator."
            )

        # 2. A range beyond the axis limits: move_to() clamps, so the
        #    outer planes pile up at the same clamped position.
        try:
            lo, hi = axis.limits
            centre = axis.position
            want_lo, want_hi = centre + offsets[0], centre + offsets[-1]
            if want_lo < lo or want_hi > hi:
                warnings.append(
                    f"z-stack spans {want_lo:.2f}..{want_hi:.2f} µm but "
                    f"the {axis.label} is limited to {lo:.2f}..{hi:.2f} "
                    f"— outer planes would be clamped and duplicated. "
                    f"Reduce the range or re-centre Z."
                )
        except Exception:
            pass

        # 3. Settling dominates long stacks; warn if it will not fit.
        est = len(offsets) * (self.settle_s + self.exposure_ms / 1000.0)
        total = est * len(self.pos_names)
        if total > self.interval_s:
            warnings.append(
                f"{len(offsets)} planes x {len(self.pos_names)} "
                f"position(s) needs about {total:.1f} s, longer than "
                f"the {self.interval_s:g} s interval — timepoints will "
                f"run back-to-back with no idle time."
            )
        return warnings

    def _acquire_stack(self, z_centre: float, grab) -> np.ndarray:
        """Z-stack around z_centre. Returns a (planes, H, W) array.

        The piezo suits this well: 10 nm steps and ~250 µm of travel
        either side of its mid-point. Z is always returned to z_centre,
        including on failure, so a bad plane cannot leave the focus
        parked somewhere unexpected for the next position.
        """
        offsets = self.plane_offsets()
        planes = []
        try:
            for off in offsets:
                self.motion.z.move_to(z_centre + off)
                if self.settle_s:
                    time.sleep(self.settle_s)
                planes.append(grab(z_centre + off))
        finally:
            try:
                self.motion.z.move_to(z_centre)
            except Exception as e:
                logger.warning(
                    "could not restore Z to %.3f after the stack: %s",
                    z_centre, e,
                )
        return np.stack(planes)

    def _save(self, img: np.ndarray, pos_name: str, tp: int,
              channel: str | None = None) -> str:
        """Write a frame using the convention every backend shares.

        t{tp:04d}_{channel}.tif — the ROI selection dashboard matches
        t{NNNN}<suffix>.tif, so per-channel files sit side by side and
        the suffix picks which one the tracker reads.
        """
        folder = os.path.join(self.dirpath, pos_name)
        os.makedirs(folder, exist_ok=True)
        label = channel if channel is not None else self.channel
        path = os.path.join(folder, f"t{tp:04d}_{label}.tif")
        try:
            import tifffile
            tifffile.imwrite(path, img)
        except ImportError:
            np.save(path.replace(".tif", ".npy"), img)
            logger.warning("tifffile missing — wrote .npy instead")
        return path

    # ---------------------------------------------- tracker contract

    @property
    def stop_requested(self) -> bool:
        """True once the acquisition loop has stopped or been asked to.

        TrackingRunner.run_zeiss() uses this to tell a plain timeout
        apart from end-of-run: on (None, None, None) it continues if
        this is False and breaks if True. Without it the tracker would
        spin forever after the last timepoint.
        """
        if self._stop_event.is_set():
            return True
        if self._finished:
            return True
        return False

    def wait_for_image(self, timeout_ms: int = 1000):
        """Pop the next frame as (image, timepoint, position_name).

        Returns (None, None, None) on timeout or shutdown — NOT a bare
        None. Every other backend uses that convention and
        TrackingRunner relies on it: run_zeiss() unpacks the result
        BEFORE testing it, so returning None raised
          cannot unpack non-iterable NoneType object
        Pair it with stop_requested to tell timeout from end-of-run.
        """
        try:
            item = self._image_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None, None, None
        if item is None:                 # shutdown sentinel
            self._finished = True
            return None, None, None
        return item

    def wait_for_pause(self, timeout_ms: int = 1000):
        return self.wait_for_image(timeout_ms)

    def relative_move(self, position_name: str, shift_x: float,
                      shift_y: float, shift_z: float) -> None:
        """Record a drift correction for the next visit.

        GUARD 2: _cum_drift is updated synchronously here, before
        returning. The Micro-Manager backend deferred this and briefly
        computed targets from a stale correction.

        The stage is NOT moved now — the loop applies the correction
        when it next visits this position, which keeps motion serialised
        on the acquisition thread.
        """
        if position_name not in self._cum_drift:
            logger.warning("relative_move for unknown position %r",
                           position_name)
            return

        sx = max(-self.max_xy_um, min(self.max_xy_um, float(shift_x)))
        sy = max(-self.max_xy_um, min(self.max_xy_um, float(shift_y)))
        sz = max(-self.max_z_um, min(self.max_z_um, float(shift_z)))
        if (sx, sy, sz) != (shift_x, shift_y, shift_z):
            logger.warning(
                "%s: correction (%.2f, %.2f, %.2f) clamped to "
                "(%.2f, %.2f, %.2f)",
                position_name, shift_x, shift_y, shift_z, sx, sy, sz,
            )

        with self._drift_lock:
            d = self._cum_drift[position_name]
            d[0] += sx
            d[1] += sy
            d[2] += sz
            total = tuple(d)
        logger.info("%s: cum_drift updated -> (%.2f, %.2f, %.2f)",
                    position_name, *total)

    def refresh_filename(self, pos_name: str) -> None:
        """Re-read a baseline after the config changed on disk."""
        entry = self.positions_config.get(pos_name)
        if not entry or "xyz_um" not in entry:
            return
        x, y, z = entry["xyz_um"]
        with self._config_lock:
            self._baseline_um[pos_name] = (float(x), float(y), float(z))
        logger.info("%s: baseline refreshed -> (%.1f, %.1f, %.1f)",
                    pos_name, x, y, z)

    # The MTB loop never pauses mid-run: corrections are applied on the
    # next visit, so there is nothing to hold. Present for contract
    # compatibility with the LS1 and ZEN backends.
    def pause_after_position(self) -> None:
        pass

    def no_pause_after_position(self) -> None:
        pass

    def continue_from_pause(self) -> None:
        pass

    # -------------------------------------------------------- helpers

    def get_cum_drift(self, pos_name: str) -> tuple[float, float, float]:
        with self._drift_lock:
            return tuple(self._cum_drift[pos_name])

    def set_exposure(self, exposure_ms: float) -> None:
        with self._config_lock:
            self.exposure_ms = float(exposure_ms)
        logger.info("exposure set to %.1f ms", exposure_ms)
