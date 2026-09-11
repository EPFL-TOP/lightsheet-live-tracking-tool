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

import base64
import binascii
import logging
import os
import queue
import threading
import time

import numpy as np

from .mtb import MTBError, MTBMotion, MTBSession

logger = logging.getLogger(__name__)


def encode_df2_ref(data: bytes | None) -> str | None:
    """Make a DF2 stabilizing blob storable in JSON.

    DF2 hands back an opaque Byte[] describing a learned focus plane.
    It has to survive a round trip through positions_config and
    tracking_RoIs.json, which are JSON, so it travels base64-encoded.
    """
    if not data:
        return None
    return base64.b64encode(bytes(data)).decode("ascii")


def decode_df2_ref(value) -> bytes | None:
    """Inverse of encode_df2_ref, tolerant of what JSON round-trips to.

    Returns None rather than raising on rubbish: a corrupt reference
    should cost focus stabilisation, not the whole run.
    """
    if not value:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, (list, tuple)):
        try:
            return bytes(bytearray(value))
        except (ValueError, TypeError):
            logger.warning("df2_ref list is not a byte sequence")
            return None
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            logger.warning("df2_ref is not valid base64 — ignoring it")
            return None
    logger.warning("df2_ref has unusable type %s", type(value).__name__)
    return None


def invert_for_tracking(img: np.ndarray) -> np.ndarray:
    """Flip contrast so a dark-on-bright object becomes bright-on-dark.

    The ROI tracker segments by thresholding for a bright object
    against a dark background. That holds for fluorescence but is
    exactly reversed in transmitted light, where the background is the
    brightest thing in the frame and survives every threshold.

    Inverts within the data's own range rather than the dtype's, so a
    16-bit frame spanning 11000..14800 still uses the full contrast
    instead of collapsing into the bottom of a 0..65535 scale.
    Geometry is untouched, so ROIs drawn on the raw image remain
    valid.
    """
    a = np.asarray(img)
    lo = a.min()
    hi = a.max()
    if hi == lo:
        return a.copy()
    return (hi + lo) - a


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
      settle_s        pause after an XY move, before snapping (0.3)
      plane_settle_s  pause between Z PLANES (default min(settle_s,
                      0.02)) - a piezo step settles in milliseconds,
                      so the stage figure here is pure waste
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
        # A SEPARATE settle for z-planes, and the difference matters
        # enormously. settle_s is sized for the XY stage between
        # positions, where a heavy stage genuinely rings. A 1 um piezo
        # step settles in milliseconds. Using the stage figure per
        # plane made a 100-plane stack sleep for 30 s — about 75% of
        # the minute it took — for no benefit whatsoever.
        plane_settle = p.get("plane_settle_s")
        self.plane_settle_s = (
            float(plane_settle) if plane_settle is not None
            else min(self.settle_s, 0.02)
        )
        self.max_xy_um = float(p.get("max_xy_um", 50.0))
        self.max_z_um = float(p.get("max_z_um", 20.0))
        self.z_stack = p.get("z_stack")
        # Written into the TIFF so downstream analysis inherits the
        # calibration instead of re-deriving it from the objective.
        self.pixel_size_um = p.get("pixel_size_um")
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
        # Light is gated to the EXPOSURE ONLY. Leaving it on across the
        # settle, the stage moves and the inter-timepoint interval
        # illuminates a live sample continuously — for a 60 s interval
        # with a 200 ms exposure that is ~300x the necessary dose, which
        # bleaches and cooks the embryo. illuminate(channel, on) is
        # supplied by the GUI so hardware details stay in one place.
        self.illuminate = p.get("illuminate")
        # Called for EVERY channel, unlike wait_for_image() which
        # yields only the first (the one drift is measured on). Lets a
        # GUI show the newest frame of each channel and name what is
        # being acquired.
        self.on_frame = p.get("on_frame")
        # A mechanical shutter has finite cycles, so a long z-stack can
        # optionally hold it open for the whole stack instead of per
        # plane. LEDs are solid state and always switch per exposure —
        # that is where the phototoxicity is.
        self.gate_light_per_plane = bool(
            p.get("gate_light_per_plane", True)
        )
        # --- Definite Focus 2 -------------------------------------
        # DF2 holds a learned plane optically. Over a multi-hour run it
        # beats correcting Z from images: it costs no light, no
        # exposure time, and it works on a specimen with nothing
        # trackable in it.
        self.use_definite_focus = bool(p.get("use_definite_focus", False))
        # Who owns Z when DF2 is running?
        #
        #   True  (default) - DF2 owns it. Z corrections from the
        #         tracker are DROPPED. Correct when the drift you are
        #         fighting is thermal: two controllers aiming at the
        #         same axis from different measurements will otherwise
        #         fight each other every timepoint.
        #   False - DF2 removes the hardware drift, and the tracker's
        #         accumulated dz is then applied as an offset from the
        #         stabilized plane. Use when the SPECIMEN genuinely
        #         moves in Z: because DF2 has already re-established
        #         the plane before each frame, the residual Z error
        #         the images show is specimen motion, so accumulating
        #         it on top of the DF2 plane composes correctly.
        self.df2_owns_z = bool(p.get("df2_owns_z", True))
        self._df2 = p.get("df2")
        # Per-position stabilizing blobs. One global DF2 reference
        # cannot serve several positions sitting at different heights,
        # so each position carries its own; see decode_df2_ref.
        self._df2_refs: dict[str, bytes] = {}
        # Positions whose stored blob DF2 has already rejected, so a
        # broken reference is reported once rather than every visit.
        self._df2_bad: set[str] = set()

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

        # Per-position preprocessing, authored in the ROI selection
        # dashboard and carried in tracking_RoIs.json. TrackingRunner
        # merges every key of that file into positions_config and then
        # calls refresh_filename(), which is how it reaches us.
        self._invert: dict[str, bool] = {
            name: bool(positions_config[name].get("invert", False))
            for name in self.pos_names
        }
        # Which positions had it stated explicitly, so the channel
        # default is not mistaken for an operator decision.
        self._invert_explicit: set[str] = {
            name for name in self.pos_names
            if "invert" in positions_config[name]
        }

        for name in self.pos_names:
            ref = decode_df2_ref(positions_config[name].get("df2_ref"))
            if ref:
                self._df2_refs[name] = ref

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

        self._connect_df2()

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

    # ------------------------------------------- definite focus 2

    def _connect_df2(self) -> None:
        """Attach DF2 if it was asked for. Never fatal.

        A stand without DF2, or a DF2 that will not cast, downgrades to
        image-based Z correction with a warning — losing focus
        stabilisation is a much smaller loss than losing the run.
        """
        if not self.use_definite_focus:
            return
        if self._df2 is not None:
            logger.info("using the caller's Definite Focus 2 handle")
        else:
            try:
                self._df2 = self.session.definite_focus()
            except Exception as e:
                logger.warning(
                    "Definite Focus 2 requested but unavailable (%s) — "
                    "falling back to image-based Z correction", e,
                )
                self.use_definite_focus = False
                return
        missing = [n for n in self.pos_names if n not in self._df2_refs]
        if missing:
            logger.warning(
                "no DF2 reference stored for %s — those positions will "
                "be stabilized on whatever plane DF2 currently holds. "
                "Learn a reference at each position in the setup panel.",
                ", ".join(missing),
            )
        logger.info("Definite Focus 2 active, df2_owns_z=%s",
                    self.df2_owns_z)

    def set_df2_reference(self, pos_name: str, data: bytes | None) -> None:
        """Store the plane learned at one position.

        Also written back into positions_config so it is saved with the
        rest of the position and survives a restart.
        """
        if pos_name not in self._cum_drift:
            logger.warning("DF2 reference for unknown position %r",
                           pos_name)
            return
        with self._config_lock:
            if data:
                self._df2_refs[pos_name] = bytes(data)
            else:
                self._df2_refs.pop(pos_name, None)
            self._df2_bad.discard(pos_name)
            entry = self.positions_config.get(pos_name)
            if entry is not None:
                entry["df2_ref"] = encode_df2_ref(data)
        logger.info("%s: DF2 reference %s", pos_name,
                    "stored" if data else "cleared")

    def _stabilize(self, pos_name: str, commanded_z: float) -> float:
        """Put DF2 on this position's plane. Returns the Z to image at.

        Called after the stage has arrived, so DF2 starts from roughly
        the right height. The returned Z is READ BACK rather than
        assumed: DF2 moves the focus itself, and a z-stack must be
        centred where the focus actually ended up, not where we asked
        it to go.
        """
        if not self.use_definite_focus or self._df2 is None:
            return commanded_z

        ref = self._df2_refs.get(pos_name)
        if ref and pos_name not in self._df2_bad:
            try:
                self._df2.restore(ref)
            except Exception as e:
                # Once only: a stale blob would otherwise log per visit.
                self._df2_bad.add(pos_name)
                logger.warning(
                    "%s: DF2 rejected the stored reference (%s) — "
                    "stabilizing on its current plane instead",
                    pos_name, e,
                )

        try:
            self._df2.stabilize_now()
        except Exception:
            logger.exception(
                "%s: DF2 stabilization failed — imaging at the "
                "commanded Z", pos_name,
            )
            return commanded_z

        try:
            actual = float(self.motion.z.position)
        except Exception:
            logger.exception("could not read Z back after DF2")
            return commanded_z

        if not self.df2_owns_z:
            # DF2 has removed the hardware drift, so the tracker's
            # accumulated dz is specimen motion and belongs on top of
            # the stabilized plane.
            with self._drift_lock:
                dz = self._cum_drift[pos_name][2]
            if dz:
                try:
                    self.motion.z.move_to(actual + dz)
                    actual = float(self.motion.z.position)
                except Exception:
                    logger.exception(
                        "%s: could not apply the tracked Z offset "
                        "after DF2", pos_name,
                    )

        if abs(actual - commanded_z) > 0.001:
            logger.debug("%s: DF2 moved focus %.3f -> %.3f um",
                         pos_name, commanded_z, actual)
        return actual

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

        # DF2 runs AFTER the move and before any exposure, so every
        # channel and every plane of the stack is referenced to the
        # same stabilized plane. z is rebound because DF2 moves the
        # focus itself and the stack centres on the result.
        z = self._stabilize(pos_name, z)

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
                                exposure_ms=channel.get("exposure_ms"),
                                channel=channel)
            path = self._save(img, pos_name, tp,
                              channel=str(channel.get("name", "BF")))
            logger.debug("t=%d %s [%s] -> %s",
                         tp, pos_name, channel.get("name"), path)

            if self.on_frame is not None:
                try:
                    self.on_frame(pos_name,
                                  str(channel.get("name", "BF")),
                                  tp, img)
                except Exception:
                    # A display callback must never break acquisition.
                    logger.exception("on_frame callback failed")

            if idx == 0:
                # Invert what the TRACKER sees; the file on disk stays
                # raw. The ROI dashboard is authoritative because that
                # is where the operator configures what to track and
                # sees the preview — the channel's flag is only the
                # default until ROIs are saved.
                self._image_queue.put(
                    (invert_for_tracking(img)
                     if self.wants_invert(pos_name, channel) else img,
                     tp, pos_name)
                )

    def _acquire(self, pos_name, x, y, z, stack=None,
                 exposure_ms=None, channel=None) -> np.ndarray:
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
            # Set the exposure ONCE per stack rather than per plane: it
            # can reconfigure the camera, and 100 planes meant 100
            # needless round trips.
            first = {"pending": True}

            def grab(zz):
                once = first["pending"]
                first["pending"] = False
                return self._snap(exposure_ms, channel=channel,
                                  gate=self.gate_light_per_plane,
                                  set_exposure=once)

        want = self.z_stack if stack is None else (
            stack and self.z_stack
        )
        if not want:
            return grab(z)

        # Holding a mechanical shutter open across the stack trades a
        # larger light dose for fewer shutter cycles.
        if not self.gate_light_per_plane and self.illuminate is not None \
                and channel is not None:
            self.illuminate(channel, True)
            try:
                return self._acquire_stack(z, grab)
            finally:
                try:
                    self.illuminate(channel, False)
                except Exception:
                    logger.exception(
                        "could not switch the light off after a stack"
                    )
        return self._acquire_stack(z, grab)

    def _snap(self, exposure_ms=None, channel=None,
              gate: bool = True, set_exposure: bool = True) -> np.ndarray:
        """Acquire one frame, illuminated only while integrating.

        The `finally` matters: if the snap raises, the light must still
        go out. Otherwise a failed frame leaves the sample under
        continuous excitation until someone notices.

        set_exposure=False skips the camera round trip, for planes
        within a z-stack where it was already set once for the whole
        stack. Acquisition is serialised on one thread, so nothing can
        change it underneath us mid-stack.
        """
        if set_exposure:
            with self._config_lock:
                self.mmc.setExposure(
                    float(exposure_ms) if exposure_ms is not None
                    else self.exposure_ms
                )
        if gate and self.illuminate is not None and channel is not None:
            self.illuminate(channel, True)
        try:
            self.mmc.snapImage()
            return np.asarray(self.mmc.getImage())
        finally:
            if gate and self.illuminate is not None \
                    and channel is not None:
                try:
                    self.illuminate(channel, False)
                except Exception:
                    logger.exception(
                        "could not switch the light off after a frame"
                    )

    def plane_offsets(self) -> list[float]:
        """Z offsets of the stack planes, symmetric about the centre.

        Specify either `n_planes` (preferred) or `range_um`. Plane
        count is the honest parameter: a range has to be divided by the
        step and rounded, so asking for 10 um in 3 um steps silently
        gives 4 planes spanning 9 um. n_planes x step says exactly what
        you get, and the range follows from it.
        """
        step = float(self.z_stack.get("step_um", 1.0))
        if step <= 0:
            raise ValueError(
                f"z_stack step_um must be positive, got {step}"
            )

        n = self.z_stack.get("n_planes")
        if n is None:
            rng = abs(float(self.z_stack.get("range_um", 10.0)))
            n = int(round(rng / step)) + 1
        n = max(1, int(n))
        if n == 1:
            return [0.0]
        span = (n - 1) * step
        return [-span / 2.0 + i * step for i in range(n)]

    @property
    def stack_range_um(self) -> float:
        """Total span of the stack, derived from planes and step."""
        if not self.z_stack:
            return 0.0
        offsets = self.plane_offsets()
        return offsets[-1] - offsets[0]

    def check_z_stack(self) -> list[str]:
        """Problems that would silently corrupt a stack. Empty if fine.

        Both failure modes here produce DUPLICATE planes rather than an
        error, which is far worse than a refusal: the file looks like a
        valid stack and nothing flags it.
        """
        if not self.z_stack:
            return []
        warnings = []

        # 0. The one that actually bites, and the only one that
        #    produces NO stack at all rather than a corrupt one: a
        #    z-stack is configured but no channel asks for it, so
        #    every frame comes back as a single plane and the setting
        #    appears to be ignored.
        if not any(c.get("z_stack") for c in self.channels):
            warnings.append(
                "a z-stack is configured but NO channel has z_stack "
                "enabled, so every frame will be a single plane. Tick "
                "z_stack on the channel(s) you want stacked in the "
                "Channels tab."
            )

        if self.motion is None:
            return warnings
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
        est = len(offsets) * (
            self.plane_settle_s + self.exposure_ms / 1000.0
        )
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
        started = time.monotonic()
        try:
            for off in offsets:
                self.motion.z.move_to(z_centre + off)
                if self.plane_settle_s:
                    time.sleep(self.plane_settle_s)
                planes.append(grab(z_centre + off))
        finally:
            try:
                self.motion.z.move_to(z_centre)
            except Exception as e:
                logger.warning(
                    "could not restore Z to %.3f after the stack: %s",
                    z_centre, e,
                )
        took = time.monotonic() - started
        if planes:
            logger.info(
                "z-stack: %d planes in %.2fs (%.0f ms/plane, "
                "%.0f ms of that settling)",
                len(planes), took, 1000 * took / len(planes),
                1000 * self.plane_settle_s,
            )
        return np.stack(planes)

    def _save(self, img: np.ndarray, pos_name: str, tp: int,
              channel: str | None = None) -> str:
        """Write a frame or a z-stack using the shared convention.

        t{tp:04d}_{channel}.tif — the ROI selection dashboard matches
        t{NNNN}<suffix>.tif, so per-channel files sit side by side and
        the suffix picks which one the tracker reads.

        A STACK IS WRITTEN AS A REAL Z-STACK. Handing tifffile a bare
        (planes, H, W) array is not enough: with exactly 3 planes it
        guesses RGB and writes ONE page with three colour components,

            data with shape (3, 8, 8) ... stored as RGB with separate
            component planes

        so a 3-plane stack silently became a colour snapshot. Even at
        other plane counts the axes came out as 'Q' — unknown — with no
        z-step, so nothing downstream could tell a stack from a time
        series. Writing ImageJ metadata fixes both: ImageJ, Fiji and
        tifffile all then read it as slices with a real spacing.
        """
        folder = os.path.join(self.dirpath, pos_name)
        os.makedirs(folder, exist_ok=True)
        label = channel if channel is not None else self.channel
        path = os.path.join(folder, f"t{tp:04d}_{label}.tif")
        try:
            import tifffile
        except ImportError:
            np.save(path.replace(".tif", ".npy"), img)
            logger.warning("tifffile missing — wrote .npy instead")
            return path

        kwargs = self._tiff_kwargs(img)
        try:
            tifffile.imwrite(path, img, **kwargs)
        except Exception:
            # Never lose data to a metadata problem.
            logger.exception(
                "writing %s with metadata failed — falling back", path
            )
            tifffile.imwrite(path, img, photometric="minisblack")
        return path

    def _tiff_kwargs(self, img: np.ndarray) -> dict:
        """ImageJ metadata describing what the array actually is."""
        meta = {"axes": "ZYX" if img.ndim == 3 else "YX"}
        if img.ndim == 3:
            meta["spacing"] = float(self.z_stack.get("step_um", 1.0)) \
                if isinstance(self.z_stack, dict) else 1.0
            meta["unit"] = "um"
        kwargs = {"imagej": True, "metadata": meta}
        if self.pixel_size_um:
            # ImageJ reads XY calibration from the TIFF resolution
            # tags, in pixels per unit.
            r = 1.0 / float(self.pixel_size_um)
            kwargs["resolution"] = (r, r)
        return kwargs

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

        # DF2 and the image-based correction would otherwise aim the
        # same axis at two different measurements and fight every
        # timepoint, so when DF2 owns Z the tracker's dz is dropped
        # here rather than accumulated and then ignored later.
        if self.use_definite_focus and self.df2_owns_z:
            if abs(float(shift_z)) > 1e-9:
                logger.debug(
                    "%s: dropping Z correction %.3f um — Definite "
                    "Focus 2 owns Z", position_name, float(shift_z),
                )
            shift_z = 0.0

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

    def wants_invert(self, pos_name: str, channel: dict | None = None):
        """Whether frames for this position are inverted for tracking.

        The ROI selection dashboard wins: it is where the operator
        decides what to track and previews the result, and its choice
        travels in tracking_RoIs.json. The channel's flag is only a
        default for before any ROIs exist.
        """
        with self._config_lock:
            if pos_name in self._invert_explicit:
                return self._invert[pos_name]
        return bool((channel or {}).get("invert", False))

    def refresh_filename(self, pos_name: str) -> None:
        """Re-read per-position config after it changed on disk.

        Called by TrackingRunner when tracking_RoIs.json changes, after
        it has merged that file into positions_config — so this is
        where preprocessing chosen in the ROI dashboard reaches us.
        """
        entry = self.positions_config.get(pos_name)
        if not entry:
            return

        if "invert" in entry:
            with self._config_lock:
                self._invert[pos_name] = bool(entry["invert"])
                self._invert_explicit.add(pos_name)
            logger.info("%s: invert-for-tracking = %s (from ROI config)",
                        pos_name, entry["invert"])

        if "df2_ref" in entry:
            ref = decode_df2_ref(entry["df2_ref"])
            with self._config_lock:
                if ref:
                    self._df2_refs[pos_name] = ref
                else:
                    self._df2_refs.pop(pos_name, None)
                self._df2_bad.discard(pos_name)
            logger.info("%s: DF2 reference %s (from config)", pos_name,
                        "refreshed" if ref else "cleared")

        if "xyz_um" not in entry:
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
