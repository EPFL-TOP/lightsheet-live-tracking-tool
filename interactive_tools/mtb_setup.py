"""Setup GUI for the Zeiss Axio Observer 7 (MTB motion + PVCAM camera).

Serve standalone:
    panel serve interactive_tools/mtb_setup.py --show --port 5023

Or embed the component in another Panel app:
    from interactive_tools.mtb_setup import MTBSetupPanel
    tabs.append(("Setup", MTBSetupPanel().layout()))

WHY PANEL AND NOT pymmcore-widgets

pymmcore-widgets drives MMCore devices. On this microscope the camera
is on MMCore but the STAGE IS NOT — it is on Zeiss MTB, because the
Axio Observer 7 carries CAN29 over USB into CZCanSrv where
Micro-Manager's serial adapter cannot reach it. So StageWidget and
friends cannot move this stage at all, and half the GUI would be
unbuildable. Panel also lets this sit alongside the existing ROI
Selection and Visualisation tabs.

THE ONE-SESSION RULE

MTB permits a single Login per process and cannot re-initialise after
a Logout. This panel therefore owns the process's MTBSession via
MTBSession.shared() and hands the same object to the tracking backend
through mtb_params['session']. Never call disconnect() on it here.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time

import numpy as np
import panel as pn

# `panel serve` puts THIS file's directory on sys.path, not the repo
# root, so `import tracking_tools` fails without this. Same pattern as
# zeiss_panel_app.py and the other apps here.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Python caches directory listings and failed lookups per sys.path
# entry. If anything attempted `import tracking_tools` before the line
# above ran, that miss is remembered and the fresh path entry is
# ignored — so drop the caches after mutating sys.path.
import importlib  # noqa: E402
importlib.invalidate_caches()

# Import the repo's own modules HERE, at module scope, and never inside
# a callback. Bokeh's CodeHandler (which `panel serve` uses) snapshots
# sys.path before running this script and RESTORES it afterwards, so
# by the time a button callback fires the repo root is off sys.path
# again and `import tracking_tools` raises ModuleNotFoundError. That
# is why zeiss_panel_app.py imports at the top and works; deferring
# the import is what broke.
try:
    from tracking_tools.microscope_interface.mtb import (  # noqa: E402
        LIGHT_PATH_ROLES,
        MTBMotion,
        MTBSession,
        sample_pixel_size_um,
    )
    _MTB_IMPORT_ERROR = None
except Exception as _e:            # pragma: no cover - env dependent
    MTBMotion = MTBSession = sample_pixel_size_um = None
    LIGHT_PATH_ROLES = ()
    _MTB_IMPORT_ERROR = _e

try:
    from tracking_tools.microscope_interface.mtb_backend import (  # noqa: E402,E501
        MicroscopeInterface_MTB,
    )
    _BACKEND_IMPORT_ERROR = None
except Exception as _e:            # pragma: no cover - env dependent
    MicroscopeInterface_MTB = None
    _BACKEND_IMPORT_ERROR = _e

# The tracker pulls in heavier dependencies (torch, imaging-server-kit)
# than motion does, so keep its failure separate: you can still set up
# and acquire without it.
try:
    import yaml  # noqa: E402
    from tracking_tools.tracking_runner.TrackingRunner import (  # noqa: E402,E501
        TrackingRunner,
    )
    from tracking_tools.utils.tracking_utils import (  # noqa: E402
        get_pos_config,
    )
    _TRACKER_IMPORT_ERROR = None
except Exception as _e:            # pragma: no cover - env dependent
    TrackingRunner = None
    get_pos_config = None
    yaml = None
    _TRACKER_IMPORT_ERROR = _e

pn.extension("tabulator", notifications=True)

logger = logging.getLogger(__name__)

# Jog step sizes offered in the UI (um). The stage's own repeatability
# is ~1.9 um, so anything below ~2 um is not meaningfully commandable
# on XY; the piezo resolves far finer.
XY_STEPS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
Z_STEPS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

DEFAULT_EXPOSURE_MS = 5.0     # 5-20 ms is the good range; 50 saturates


def _stretch(img: np.ndarray, lo_pct=1.0, hi_pct=99.5) -> np.ndarray:
    """Percentile-stretch to float 0..1, max-projecting a stack.

    Raw frames sit around 11000-14800 of 65535, so a naive cast renders
    almost black; stretching to percentiles reveals the structure.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:                       # z-stack -> max projection
        a = a.max(axis=0)
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _stretch_to_png(img: np.ndarray, lo_pct=1.0, hi_pct=99.5) -> bytes:
    """Percentile-stretch a frame to 8-bit PNG bytes for display.

    Raw 16-bit frames sit around 11000-14800 out of 65535, so a naive
    cast renders almost black. Stretching to percentiles makes the
    actual structure visible.
    """
    from PIL import Image

    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:                       # z-stack -> max projection
        a = a.max(axis=0)
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    im = Image.fromarray((a * 255).astype(np.uint8))
    im.thumbnail((600, 600))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class MTBSetupPanel:
    """Interactive setup: jog, preview, capture positions, run tracking."""

    def __init__(self):
        self.session = None
        self.motion = None
        self.mmc = None
        self._cam_label = "SetupCam"
        self._runner = None          # active MicroscopeInterface_MTB
        self._tracker = None         # active TrackingRunner
        self._run_thread = None
        self._stop_flag = threading.Event()
        self._live_cb = None         # Panel periodic callback handle

        self._build_widgets()
        self._wire()

    # ------------------------------------------------------- widgets

    def _build_widgets(self) -> None:
        # --- connection ---
        self.btn_connect = pn.widgets.Button(
            name="Connect hardware", button_type="primary", width=170
        )
        self.status = pn.pane.Markdown(
            "_Not connected._", width=560,
            styles={"font-size": "0.9em"},
        )
        self.hw_info = pn.pane.Markdown("", width=560,
                                        styles={"font-size": "0.8em"})

        # --- live position ---
        self.pos_readout = pn.pane.Markdown(
            "`x — y — z —`", width=380,
            styles={"font-family": "monospace"},
        )
        self.btn_refresh = pn.widgets.Button(name="↻", width=40)

        # --- jog ---
        self.xy_step = pn.widgets.Select(
            name="XY step (µm)", options=XY_STEPS, value=10.0, width=130
        )
        self.z_step = pn.widgets.Select(
            name="Z step (µm)", options=Z_STEPS, value=1.0, width=130
        )
        self.btn_xm = pn.widgets.Button(name="← X", width=70)
        self.btn_xp = pn.widgets.Button(name="X →", width=70)
        self.btn_ym = pn.widgets.Button(name="↓ Y", width=70)
        self.btn_yp = pn.widgets.Button(name="↑ Y", width=70)
        self.btn_zm = pn.widgets.Button(name="− Z", width=70)
        self.btn_zp = pn.widgets.Button(name="+ Z", width=70)
        self.z_axis = pn.widgets.RadioButtonGroup(
            name="Z actuator", options=["piezo", "focus"],
            value="piezo", width=180,
        )

        # --- camera ---
        self.exposure = pn.widgets.FloatInput(
            name="Exposure (ms)", value=DEFAULT_EXPOSURE_MS,
            start=0.1, end=5000.0, step=1.0, width=130,
        )
        self.btn_snap = pn.widgets.Button(
            name="Snap", button_type="success", width=100
        )
        # Label flips with state so it is obvious live is running.
        self.btn_live = pn.widgets.Toggle(
            name="● Start live", button_type="warning", width=130,
            value=False,
        )
        self.live_fps = pn.widgets.FloatInput(
            name="Live rate (fps)", value=2.0, start=0.2, end=20.0,
            step=0.5, width=120,
        )
        # A Bokeh image glyph rather than a PNG pane: replacing a PNG's
        # bytes makes the browser swap the <img>, which flashes white
        # between live frames. Updating a ColumnDataSource mutates the
        # existing glyph in place, so live view is smooth.
        from bokeh.models import ColumnDataSource, LinearColorMapper
        from bokeh.plotting import figure

        self._img_source = ColumnDataSource(
            data={"image": [np.zeros((2, 2), dtype=np.float32)]}
        )
        self._img_fig = figure(
            width=560, height=560, x_range=(0, 1), y_range=(0, 1),
            toolbar_location="above",
            tools="pan,box_zoom,wheel_zoom,reset",
        )
        self._img_fig.axis.visible = False
        self._img_fig.grid.visible = False
        self._img_fig.image(
            image="image", x=0, y=0, dw=1, dh=1,
            source=self._img_source,
            color_mapper=LinearColorMapper(palette="Greys256",
                                           low=0.0, high=1.0),
        )
        self.preview = pn.pane.Bokeh(self._img_fig)
        self.img_stats = pn.pane.Markdown(
            "", width=600, styles={"font-size": "0.8em",
                                   "font-family": "monospace"},
        )

        # --- light path ---
        # The camera is on the LEFT sideport, so a sideport set to
        # '100% R' starves it, and a condenser on a phase ring throws
        # away the transmitted light. Both happened, and produced
        # exposure-independent read-noise frames that looked like a
        # camera fault.
        self.btn_read_lightpath = pn.widgets.Button(
            name="Read light path", button_type="primary", width=160
        )
        self.lightpath_rows = pn.Column(width=620)
        self.lightpath_note = pn.pane.Markdown(
            "", width=620, styles={"font-size": "0.8em"},
        )
        self._lightpath_widgets = {}

        # --- LED live control (Optics tab) ---
        # Interactive tuning: pick an LED, slide the intensity, watch
        # it on Live. Channels are the saved recipe; this is the dial.
        self.btn_read_leds = pn.widgets.Button(
            name="Read LEDs", button_type="primary", width=130
        )
        self.led_rows = pn.Column(width=620)
        self.led_note = pn.pane.Markdown(
            "", width=620, styles={"font-size": "0.8em"},
        )
        self.btn_tl_on = pn.widgets.Button(
            name="TL light ON", width=130
        )
        self.btn_tl_off = pn.widgets.Button(
            name="TL light OFF", width=130
        )

        # --- positions ---
        import pandas as pd
        self.pos_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["name", "x_um", "y_um", "z_um"]),
            height=220, width=560, show_index=False,
            selectable="checkbox",
        )
        self.btn_capture = pn.widgets.Button(
            name="+ Capture current position", button_type="primary",
            width=220,
        )
        self.btn_goto = pn.widgets.Button(
            name="Go to selected", width=140
        )
        self.btn_remove = pn.widgets.Button(
            name="Remove selected", width=150
        )
        self.btn_clear = pn.widgets.Button(name="Clear all", width=110)
        self.positions_file = pn.widgets.TextInput(
            name="Positions file", value="positions.json", width=380
        )
        self.btn_save_pos = pn.widgets.Button(name="Save", width=80)
        self.btn_load_pos = pn.widgets.Button(name="Load", width=80)

        # --- channels ---
        # A channel is (reflector cube, exposure, z-stack): the
        # reflector position IS what distinguishes a fluorescence
        # channel, and it is settable through MTBReflectorChanger.
        # reflector_pos = 0 means "leave the reflector alone", which is
        # what brightfield wants.
        self.chan_table = pn.widgets.Tabulator(
            pd.DataFrame([
                {"name": "BF", "reflector_pos": 0, "led_nm": 0,
                 "led_percent": 0.0, "tl": True, "invert": True,
                 "exposure_ms": DEFAULT_EXPOSURE_MS, "z_stack": False},
            ]),
            height=200, width=620, show_index=False,
        )
        self.btn_add_channel = pn.widgets.Button(
            name="+ Add empty channel", width=170
        )
        # The natural workflow: set the optics up by hand until the
        # image looks right, then save THAT state as a channel. Mirrors
        # "capture position" for the stage, and removes the guesswork
        # of typing reflector position numbers.
        self.btn_capture_channel = pn.widgets.Button(
            name="Capture light path → channel",
            button_type="primary", width=230,
        )
        self.btn_snap_channels = pn.widgets.Button(
            name="Snap all channels", button_type="success", width=170
        )
        self.btn_del_channel = pn.widgets.Button(
            name="Delete selected", button_type="danger", width=150
        )
        self.btn_leds_off = pn.widgets.Button(
            name="All LEDs off", width=130
        )
        self.led_info = pn.pane.Markdown(
            "", width=600, styles={"font-size": "0.85em"},
        )
        self.channel_previews = pn.Row()
        self.chan_table.selectable = "checkbox"
        self.chan_note = pn.pane.Markdown(
            "`reflector_pos` picks the filter cube by **position "
            "number** (0 = leave it alone, for brightfield); read the "
            "positions under *Light path* on the Optics tab. Each "
            "channel also carries its own exposure and whether it is "
            "acquired as a z-stack — fluorescence usually wants a "
            "stack, brightfield usually does not.\n\n"
            "`led_nm` selects the Colibri LED by **wavelength** "
            "(0 = no LED, for brightfield) and `led_percent` its "
            "intensity. Available: **385 / 430 / 475 / 511 / 567 / "
            "630 nm**. Selecting an LED switches every other one off, "
            "so channels cannot bleed into each other.\n\n"
            "A fluorescence channel needs **both** the LED and the "
            "matching reflector cube — a cube change alone gives you "
            "the transmitted-light image through a filter, which is "
            "why earlier channels all looked like brightfield.",
            width=560, styles={"font-size": "0.8em"},
        )

        # --- acquisition ---
        self.interval_s = pn.widgets.FloatInput(
            name="Interval (s)", value=60.0, start=0.1, end=86400.0,
            width=130,
        )
        self.n_timepoints = pn.widgets.IntInput(
            name="Timepoints (0 = until stopped)", value=0, start=0,
            end=100000, width=200,
        )
        self.settle_s = pn.widgets.FloatInput(
            name="Settle after move (s)", value=0.3, start=0.0,
            end=10.0, step=0.1, width=160,
        )
        self.zstack_on = pn.widgets.Checkbox(
            name="Acquire Z-stack", value=False
        )
        self.zstack_range = pn.widgets.FloatInput(
            name="Z range (µm)", value=10.0, start=0.1, end=400.0,
            width=120,
        )
        self.zstack_step = pn.widgets.FloatInput(
            name="Z step (µm)", value=1.0, start=0.01, end=50.0,
            width=120,
        )
        self.max_xy = pn.widgets.FloatInput(
            name="Max XY correction (µm)", value=50.0, start=0.1,
            end=1000.0, width=180,
        )
        self.max_z = pn.widgets.FloatInput(
            name="Max Z correction (µm)", value=20.0, start=0.1,
            end=500.0, width=180,
        )
        # The tracker measures shifts in PIXELS; these convert to µm so
        # relative_move() gets stage units. Prime 95B pixels are 11 µm
        # on a 1200x1200 sensor, so the value depends on the objective.
        self.pixel_xy = pn.widgets.FloatInput(
            name="Pixel size XY (µm)", value=0.347, start=0.001,
            end=100.0, step=0.001, width=160,
        )
        self.pixel_z = pn.widgets.FloatInput(
            name="Pixel size Z (µm)", value=1.0, start=0.001,
            end=100.0, step=0.01, width=160,
        )
        self.tracking_2d = pn.widgets.Checkbox(
            name="2-D tracking (ignore Z shift)", value=True
        )
        # Not present in tracking_config.yaml: the runner injects it,
        # and MultiRoIBaseTracker.__init__ requires it positionally —
        # omitting it fails with
        #   MultiRoIBaseTracker.init() missing 1 required positional
        #   argument: 'serverkit'
        self.serverkit = pn.widgets.Checkbox(
            name="Use serverkit (CoTracker server)", value=True
        )
        # Pixel pitch is derived, not guessed: camera pixel / (objective
        # x adapter). MTB knows the objective, so read it rather than
        # making the operator remember.
        self.camera_pixel_um = pn.widgets.FloatInput(
            name="Camera pixel (µm)", value=11.0, start=0.1, end=100.0,
            step=0.1, width=150,
        )
        self.adapter_mag = pn.widgets.FloatInput(
            name="Adapter mag (×)", value=1.0, start=0.1, end=10.0,
            step=0.1, width=140,
        )
        self.btn_read_objective = pn.widgets.Button(
            name="Read objective from MTB", button_type="primary",
            width=210,
        )
        self.objective_info = pn.pane.Markdown(
            "", width=560, styles={"font-size": "0.85em"},
        )
        # Manual override: the current slot may be empty, or the
        # element indexing may be off by one, so let the operator pick
        # the objective that is actually in the light path.
        self.objective_choice = pn.widgets.Select(
            name="Objective", options=[], width=380,
        )
        self.btn_apply_objective = pn.widgets.Button(
            name="Rotate to it", button_type="primary", width=130,
        )
        self.outdir = pn.widgets.TextInput(
            name="Experiment root", value="", width=560,
            placeholder=r"D:\Users\zeiss\data\my_experiment",
        )

        # Tracking is not a separate run: acquisition starts first (ROIs
        # can only be drawn on frames that exist), and the tracker
        # attaches to the already-running loop as soon as ROIs are
        # saved. One Start button, one continuous acquisition.
        self.auto_track = pn.widgets.Checkbox(
            name="Engage tracking automatically when ROIs are saved",
            value=True,
        )
        self.roi_poll_s = pn.widgets.FloatInput(
            name="ROI check interval (s)", value=5.0, start=1.0,
            end=120.0, width=170,
        )
        # Attaching as soon as ANY position has ROIs means the rest
        # stay untracked for the whole run, since the ROI watcher only
        # watches folders known at attach time. Waiting for all is the
        # safer default.
        self.require_all_rois = pn.widgets.Checkbox(
            name="Wait until every position has ROIs before tracking",
            value=True,
        )
        self.roi_state = pn.pane.Markdown(
            "", width=560, styles={"font-size": "0.85em"},
        )
        self.btn_check_rois = pn.widgets.Button(
            name="Check ROIs", width=120
        )
        self.btn_run = pn.widgets.Button(
            name="▶ Start", button_type="success", width=140
        )
        self.btn_stop = pn.widgets.Button(
            name="■ Stop", button_type="danger", width=110,
            disabled=True,
        )
        self.run_status = pn.pane.Markdown("_Idle._", width=560)
        self.run_log = pn.pane.Markdown(
            "", width=560,
            styles={"font-size": "0.8em", "font-family": "monospace",
                    "max-height": "200px", "overflow-y": "auto"},
        )

    def _wire(self) -> None:
        self.btn_connect.on_click(self._on_connect)
        self.btn_refresh.on_click(lambda e: self._update_position())
        self.btn_snap.on_click(self._on_snap)
        self.btn_capture.on_click(self._on_capture)
        self.btn_goto.on_click(self._on_goto)
        self.btn_remove.on_click(self._on_remove)
        self.btn_clear.on_click(
            lambda e: self._set_positions_df(self.pos_table.value.iloc[0:0])
        )
        self.btn_save_pos.on_click(self._on_save_positions)
        self.btn_load_pos.on_click(self._on_load_positions)
        self.btn_run.on_click(self._on_run)
        self.btn_stop.on_click(self._on_stop)
        self.btn_check_rois.on_click(lambda e: self._refresh_roi_state())
        self.btn_read_objective.on_click(self._on_read_objective)
        self.btn_read_lightpath.on_click(self._on_read_lightpath)
        self.btn_read_leds.on_click(self._on_read_leds)
        self.btn_tl_on.on_click(lambda e: self._set_tl(True))
        self.btn_tl_off.on_click(lambda e: self._set_tl(False))
        self.btn_add_channel.on_click(self._on_add_channel)
        self.btn_capture_channel.on_click(self._on_capture_channel)
        self.btn_del_channel.on_click(self._on_del_channel)
        self.btn_leds_off.on_click(
            lambda e: self._all_leds_off()
        )
        self.btn_snap_channels.on_click(self._on_snap_channels)
        self.btn_apply_objective.on_click(self._on_apply_objective)
        self.btn_live.param.watch(self._on_live_toggle, "value")
        self.z_axis.param.watch(self._on_z_axis_change, "value")

        self.btn_xm.on_click(lambda e: self._jog(dx=-self.xy_step.value))
        self.btn_xp.on_click(lambda e: self._jog(dx=+self.xy_step.value))
        self.btn_ym.on_click(lambda e: self._jog(dy=-self.xy_step.value))
        self.btn_yp.on_click(lambda e: self._jog(dy=+self.xy_step.value))
        self.btn_zm.on_click(lambda e: self._jog(dz=-self.z_step.value))
        self.btn_zp.on_click(lambda e: self._jog(dz=+self.z_step.value))

    # -------------------------------------------------------- helpers

    def _say(self, msg: str, kind: str = "info") -> None:
        icon = {"info": "", "ok": "✅ ", "warn": "⚠️ ",
                "err": "❌ "}.get(kind, "")
        self.status.object = f"{icon}{msg}"
        logger.info("status: %s", msg)

    def _append_log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        prev = self.run_log.object or ""
        lines = (prev + f"\n{stamp}  {msg}").strip().split("\n")
        self.run_log.object = "\n".join(lines[-40:])

    @property
    def connected(self) -> bool:
        return self.motion is not None

    def _require_connection(self) -> bool:
        if not self.connected:
            self._say("Connect the hardware first.", "warn")
            return False
        return True

    # ----------------------------------------------------- connection

    def _on_connect(self, _event=None) -> None:
        if MTBSession is None:
            # Resolved at import time, not here — see the note at the
            # top about Bokeh restoring sys.path.
            expected = os.path.join(_ROOT, "tracking_tools",
                                    "microscope_interface", "mtb.py")
            self._say(
                f"MTB layer unavailable: {_MTB_IMPORT_ERROR}\n\n"
                f"- repo root: `{_ROOT}`\n"
                f"- `{expected}` exists: "
                f"`{os.path.exists(expected)}`\n\n"
                f"This is decided when the app module loads. If the "
                f"file exists, check the `panel serve` console for the "
                f"import traceback — a missing dependency of "
                f"`mtb.py` shows up here too.",
                "err",
            )
            return

        try:
            # shared(): MTB allows one Login per process, and the
            # tracking backend will reuse this very session.
            self.session = MTBSession.shared()
            self.motion = MTBMotion(self.session,
                                    z_axis=self.z_axis.value)
        except Exception as e:
            self._say(f"MTB connection failed: {e}", "err")
            logger.exception("MTB connect failed")
            return

        try:
            self._open_camera()
        except Exception as e:
            # Motion still usable without the camera.
            self._say(f"MTB connected, but the camera failed: {e}",
                      "warn")
            self._show_hw_info()
            self._update_position()
            return

        self._say("Hardware connected.", "ok")
        self._show_hw_info()
        self._update_position()
        self.btn_connect.disabled = True

    def _open_camera(self) -> None:
        from pymmcore_plus import CMMCorePlus

        self.mmc = CMMCorePlus()
        try:
            # PVCAM races MMCore's sequence buffer when it is tight.
            self.mmc.setCircularBufferMemoryFootprint(512)
        except Exception:
            pass
        self.mmc.loadDevice(self._cam_label, "PVCAM", "Camera-1")
        self.mmc.initializeDevice(self._cam_label)
        self.mmc.setCameraDevice(self._cam_label)
        self.mmc.setExposure(float(self.exposure.value))

    def _show_hw_info(self) -> None:
        parts = []
        if self.motion is not None:
            parts.append("```\n" + self.motion.describe() + "\n```")
        try:
            parts.append("```\n" + self.session.colibri().describe()
                         + "\n```")
        except Exception:
            pass
        if self.mmc is not None:
            try:
                parts.append(
                    f"Camera: {self.mmc.getImageWidth()}×"
                    f"{self.mmc.getImageHeight()}, "
                    f"{self.mmc.getImageBitDepth()}-bit"
                )
            except Exception:
                pass
        self.hw_info.object = "\n\n".join(parts)

    def _on_z_axis_change(self, event) -> None:
        if not self.connected or MTBMotion is None:
            return
        try:
            self.motion = MTBMotion(self.session, z_axis=event.new)
            self._say(f"Z actuator switched to {event.new}.", "ok")
            self._show_hw_info()
            self._update_position()
        except Exception as e:
            self._say(f"Could not switch Z actuator: {e}", "err")

    # ------------------------------------------------------- position

    def _update_position(self) -> None:
        if not self.connected:
            self.pos_readout.object = "`x — y — z —`"
            return
        try:
            x, y, z = self.motion.get_xyz()
            self.pos_readout.object = (
                f"`x {x:10.2f}   y {y:10.2f}   "
                f"z {z:8.2f}  µm`"
            )
        except Exception as e:
            self.pos_readout.object = f"`read failed: {e}`"

    def _jog(self, dx=0.0, dy=0.0, dz=0.0) -> None:
        if not self._require_connection():
            return
        try:
            self.motion.move_by(dx=dx, dy=dy, dz=dz)
        except Exception as e:
            self._say(f"Move failed: {e}", "err")
            return
        self._update_position()

    # --------------------------------------------------------- camera

    def _sync_live_label(self) -> None:
        self.btn_live.name = ("■ Stop live" if self.btn_live.value
                              else "● Start live")
        self.btn_live.button_type = ("danger" if self.btn_live.value
                                     else "warning")

    def _on_live_toggle(self, event) -> None:
        """Start/stop repeated snapping.

        The camera is a single exclusive resource: while a tracking run
        owns it, live snapping would interleave with the run's own
        acquisitions and both would get corrupted frames. So Live is
        refused during a run, and starting a run stops Live.
        """
        if event.new:
            if self._run_thread is not None and self._run_thread.is_alive():
                self.btn_live.value = False
                self._sync_live_label()
                self._say("Cannot go live while a run is active — the "
                          "camera is exclusive.", "warn")
                return
            if self.mmc is None:
                self.btn_live.value = False
                self._sync_live_label()
                self._say("Camera not available.", "warn")
                return
            period_ms = int(1000.0 / max(0.2, float(self.live_fps.value)))
            try:
                self._live_cb = pn.state.add_periodic_callback(
                    self._live_tick, period=period_ms
                )
                self._sync_live_label()
                self._say(f"Live at ~{self.live_fps.value:g} fps.", "ok")
            except Exception as e:
                # No server context (e.g. under pytest) — degrade to a
                # single snap rather than breaking.
                self.btn_live.value = False
                self._sync_live_label()
                self._say(f"Could not start live mode: {e}", "warn")
        else:
            self._stop_live()
            self._sync_live_label()
            self._say("Live stopped.")

    def _stop_live(self) -> None:
        if self._live_cb is not None:
            try:
                self._live_cb.stop()
            except Exception:
                pass
            self._live_cb = None
        if self.btn_live.value:
            self.btn_live.value = False
        self._sync_live_label()

    def _live_tick(self) -> None:
        """One live frame. Stops itself on repeated failure."""
        try:
            self._on_snap()
        except Exception as e:
            logger.warning("live tick failed: %s", e)
            self._stop_live()
            self._say(f"Live stopped after an error: {e}", "err")

    def _on_snap(self, _event=None) -> None:
        if self.mmc is None:
            self._say("Camera not available.", "warn")
            return
        try:
            self.mmc.setExposure(float(self.exposure.value))
            self.mmc.snapImage()
            img = np.asarray(self.mmc.getImage())
        except Exception as e:
            self._say(f"Snap failed: {e}", "err")
            return

        mn, mx = float(img.min()), float(img.max())
        mean, std = float(img.mean()), float(img.std())
        try:
            depth = self.mmc.getImageBitDepth()
        except Exception:
            depth = 16
        full = (1 << depth) - 1

        note = ""
        if mx >= full:
            note = "  ← SATURATED, lower the exposure"
        elif std < 1.0:
            note = "  ← no variance; check illumination and light path"

        self.img_stats.object = (
            f"min {mn:.0f}  max {mx:.0f}  mean {mean:.1f}  "
            f"std {std:.1f}{note}"
        )
        self._render(img)

    def _render(self, img: np.ndarray) -> None:
        """Push a frame into the existing glyph — no <img> swap."""
        try:
            a = _stretch(img)
            h, w = a.shape
            self._img_fig.x_range.end = float(w)
            self._img_fig.y_range.end = float(h)
            # Bokeh draws bottom-up; flip so the image is not mirrored.
            self._img_source.data = {"image": [a[::-1]]}
            for r in self._img_fig.renderers:
                for attr in ("dw", "dh"):
                    try:
                        setattr(r.glyph, attr,
                                float(w) if attr == "dw" else float(h))
                    except Exception:
                        pass
        except Exception as e:
            self._say(f"Could not render preview: {e}", "warn")

    # ------------------------------------------------------ positions

    def _set_positions_df(self, df) -> None:
        self.pos_table.value = df.reset_index(drop=True)

    def _on_capture(self, _event=None) -> None:
        if not self._require_connection():
            return
        try:
            x, y, z = self.motion.get_xyz()
        except Exception as e:
            self._say(f"Could not read the stage: {e}", "err")
            return

        import pandas as pd
        df = self.pos_table.value
        name = f"scene_{len(df):03d}"
        row = pd.DataFrame([{
            "name": name,
            "x_um": round(x, 3),
            "y_um": round(y, 3),
            "z_um": round(z, 3),
        }])
        # Concatenating onto an all-NA/empty frame is deprecated in
        # pandas and changes dtype inference, so start from the row.
        combined = row if df.empty else pd.concat(
            [df, row], ignore_index=True
        )
        self._set_positions_df(combined)
        self._say(f"Captured {name} at ({x:.1f}, {y:.1f}, {z:.1f}).",
                  "ok")

    def _selected_rows(self) -> list[int]:
        sel = self.pos_table.selection or []
        return list(sel)

    def _on_goto(self, _event=None) -> None:
        if not self._require_connection():
            return
        rows = self._selected_rows()
        if len(rows) != 1:
            self._say("Select exactly one position to go to.", "warn")
            return
        r = self.pos_table.value.iloc[rows[0]]
        try:
            self.motion.move_to(
                x=float(r["x_um"]), y=float(r["y_um"]),
                z=float(r["z_um"]),
            )
        except Exception as e:
            self._say(f"Move failed: {e}", "err")
            return
        self._update_position()
        self._say(f"Moved to {r['name']}.", "ok")

    def _on_remove(self, _event=None) -> None:
        rows = self._selected_rows()
        if not rows:
            self._say("Nothing selected.", "warn")
            return
        df = self.pos_table.value.drop(
            self.pos_table.value.index[rows]
        )
        self._set_positions_df(df)
        self.pos_table.selection = []
        self._say(f"Removed {len(rows)} position(s).", "ok")

    def positions_config(self, root: str | None = None,
                         log_dir_name: str = "embryo_tracking") -> dict:
        """The position list in the shape the backend expects.

        `xyz_um` is the tracking baseline — the backend requires it and
        must never read it off the stage, or existing drift would
        become the reference.

        Pass `root` to also fill in `log_dir`, the folder where the ROI
        selection dashboard writes tracking_RoIs.json and where
        TrackingRunner looks for it.
        """
        cfg = {}
        for _, r in self.pos_table.value.iterrows():
            name = str(r["name"])
            entry = {
                "xyz_um": (float(r["x_um"]), float(r["y_um"]),
                           float(r["z_um"]))
            }
            if root:
                entry["log_dir"] = os.path.join(root, name,
                                                log_dir_name)
            cfg[name] = entry
        return cfg

    def roi_status(self, root: str | None = None) -> dict[str, bool]:
        """Which captured positions already have ROIs drawn."""
        root = (root or self.outdir.value or "").strip()
        out = {}
        for name in self.positions_config():
            path = os.path.join(root, name, "embryo_tracking",
                                "tracking_RoIs.json")
            out[name] = bool(root) and os.path.exists(path)
        return out

    def _on_save_positions(self, _event=None) -> None:
        path = self.positions_file.value.strip()
        if not path:
            self._say("Give a filename first.", "warn")
            return
        try:
            with open(path, "w") as fh:
                json.dump(self.positions_config(), fh, indent=2)
        except Exception as e:
            self._say(f"Save failed: {e}", "err")
            return
        self._say(f"Saved {len(self.pos_table.value)} position(s) to "
                  f"{path}.", "ok")

    def _on_load_positions(self, _event=None) -> None:
        path = self.positions_file.value.strip()
        if not os.path.exists(path):
            self._say(f"No such file: {path}", "err")
            return
        try:
            with open(path) as fh:
                cfg = json.load(fh)
            import pandas as pd
            rows = []
            for name, entry in cfg.items():
                x, y, z = entry["xyz_um"]
                rows.append({"name": name, "x_um": x, "y_um": y,
                             "z_um": z})
            self._set_positions_df(pd.DataFrame(rows))
        except Exception as e:
            self._say(f"Load failed: {e}", "err")
            return
        self._say(f"Loaded {len(cfg)} position(s).", "ok")

    # ------------------------------------------------------- tracking

    def channels(self) -> list[dict]:
        """The channel list as plain dicts."""
        out = []
        df = self.chan_table.value
        for _, r in df.iterrows():
            out.append({
                "name": str(r["name"]),
                "reflector_pos": int(r.get("reflector_pos", 0) or 0),
                "led_nm": int(r.get("led_nm", 0) or 0),
                "led_percent": float(r.get("led_percent", 0.0) or 0.0),
                # Transmitted light: ON for brightfield, OFF for
                # fluorescence — otherwise brightfield floods the
                # sensor and the fluo image has a bright background.
                "tl": bool(r.get("tl", False)),
                # Invert before tracking: the tracker thresholds for a
                # BRIGHT object on a DARK background, which is true of
                # fluorescence but inverted in brightfield.
                "invert": bool(r.get("invert", False)),
                "exposure_ms": float(r.get("exposure_ms",
                                           DEFAULT_EXPOSURE_MS)),
                "z_stack": bool(r.get("z_stack", False)),
            })
        return out

    def _on_add_channel(self, _event=None) -> None:
        import pandas as pd
        df = self.chan_table.value
        row = pd.DataFrame([{
            "name": f"CH{len(df)}",
            "reflector_pos": 0,
            "led_nm": 0,
            "led_percent": 0.0,
            "tl": True,
            "invert": True,
            "exposure_ms": float(self.exposure.value),
            "z_stack": True,
        }])
        self.chan_table.value = (
            row if df.empty
            else pd.concat([df, row], ignore_index=True)
        )

    def _on_capture_channel(self, _event=None) -> None:
        """Save the CURRENT optics state as a new channel row."""
        if not self._require_connection():
            return
        try:
            refl = self.session.changer("reflector")
            pos = refl.position or 0
            cube = None
            for el in refl.elements():
                if el["current"]:
                    cube = el["name"]
                    break
        except Exception as e:
            self._say(f"Could not read the reflector: {e}", "err")
            return

        # Capture the live LED state as well as the cube.
        led_nm, led_pct = 0, 0.0
        try:
            for led in self.session.colibri().leds():
                if led.is_on:
                    led_nm = int(led.wavelength or 0)
                    led_pct = float(led.intensity or 0.0)
                    break
        except Exception as e:
            logger.info("could not read the LEDs: %s", e)

        import pandas as pd
        df = self.chan_table.value
        label = (cube or f"CH{len(df)}")
        # 'none' as a cube name means brightfield, so do not move the
        # reflector for that channel.
        is_bf = (cube or "").strip().lower() in ("none", "", "-")
        # A channel with an LED is fluorescence: transmitted light
        # off, and no inversion (background is already dark).
        is_fluo = bool(led_nm)
        row = pd.DataFrame([{
            "name": (f"{led_nm}nm" if led_nm else
                     ("BF" if is_bf else
                      label.replace(" ", "_")[:16])),
            "reflector_pos": 0 if is_bf else int(pos),
            "led_nm": led_nm,
            "led_percent": led_pct,
            "tl": not is_fluo,
            "invert": not is_fluo,
            "exposure_ms": float(self.exposure.value),
            "z_stack": is_fluo or not is_bf,
        }])
        self.chan_table.value = (
            row if df.empty
            else pd.concat([df, row], ignore_index=True)
        )
        self._say(
            f"Captured channel: reflector {pos} ({cube or 'unnamed'}), "
            f"LED {led_nm or 'off'}"
            f"{f' @ {led_pct:g}%' if led_nm else ''}, "
            f"{self.exposure.value:g} ms.", "ok",
        )

    def _set_tl(self, open_it: bool) -> None:
        """Open or close the transmitted-light shutter."""
        if not self._require_connection():
            return
        try:
            self.session.changer("tl_shutter").set_open(open_it)
        except Exception as e:
            self._say(f"Could not switch transmitted light: {e}", "err")
            return
        self._say(
            f"Transmitted light {'ON' if open_it else 'OFF'}.", "ok"
        )

    def _on_read_leds(self, _event=None) -> None:
        """Build a live on/off + intensity control per Colibri LED."""
        if not self._require_connection():
            return
        try:
            colibri = self.session.colibri()
            leds = colibri.leds()
        except Exception as e:
            self.led_note.object = f"❌ Could not read the LEDs: {e}"
            logger.exception("LED read failed")
            return

        rows = []
        for led in leds:
            lo, hi = led.intensity_limits
            toggle = pn.widgets.Toggle(
                name=f"{led.wavelength} nm",
                value=bool(led.is_on), width=110,
                button_type="success",
            )
            slider = pn.widgets.FloatSlider(
                name=f"{led.unit}", start=lo, end=hi,
                value=float(led.intensity or 0.0),
                step=1.0, width=330,
            )

            def on_toggle(event, _led=led):
                try:
                    _led.set_on(bool(event.new))
                    self._say(
                        f"LED {_led.wavelength} nm "
                        f"{'ON' if event.new else 'OFF'}.", "ok",
                    )
                except Exception as exc:
                    self._say(
                        f"LED {_led.wavelength} nm would not switch: "
                        f"{exc}", "err",
                    )

            def on_slide(event, _led=led):
                try:
                    _led.set_intensity(float(event.new))
                except Exception as exc:
                    self._say(
                        f"LED {_led.wavelength} nm intensity: {exc}",
                        "err",
                    )

            toggle.param.watch(on_toggle, "value")
            slider.param.watch(on_slide, "value")
            rows.append(pn.Row(toggle, slider))

        self.led_rows.objects = rows or [
            pn.pane.Markdown("_No LEDs reachable._", width=600)
        ]
        self.led_note.object = (
            "Switching one LED on does **not** switch the others off "
            "here — that only happens when a channel is applied. For a "
            "clean fluorescence image also turn the **transmitted "
            "light OFF**: leaving it on is what makes a fluo frame come "
            "back with a bright background.\n\n"
            "Adjust with **Live** running to see the effect "
            "immediately, then use *Capture light path → channel*."
        )
        self._say(f"Read {len(leds)} LED(s).", "ok")

    def _all_leds_off(self) -> None:
        if not self._require_connection():
            return
        try:
            self.session.colibri().all_off()
            self._say("All Colibri LEDs off.", "ok")
        except Exception as e:
            self._say(f"Could not switch the LEDs off: {e}", "err")

    def _on_del_channel(self, _event=None) -> None:
        rows = list(self.chan_table.selection or [])
        if not rows:
            self._say("Tick the channels to delete first.", "warn")
            return
        df = self.chan_table.value
        self.chan_table.value = df.drop(
            df.index[rows]
        ).reset_index(drop=True)
        self.chan_table.selection = []
        self._say(f"Deleted {len(rows)} channel(s).", "ok")

    def _apply_channel(self, channel: dict) -> None:
        """Put the hardware into a channel's configuration.

        Order matters: switch the LED FIRST, then the cube, then the
        exposure — so a bright LED is never shining through the wrong
        filter, and the exposure is already correct when the frame is
        taken.
        """
        led_nm = int(channel.get("led_nm", 0) or 0)
        try:
            colibri = self.session.colibri()
            if led_nm:
                colibri.select(
                    led_nm, float(channel.get("led_percent", 0.0) or 0.0)
                )
            else:
                # Brightfield: no excitation at all.
                colibri.all_off()
        except Exception as e:
            logger.warning("LED setup failed for channel %s: %s",
                           channel.get("name"), e)

        # Transmitted light. Leaving it on during fluorescence is what
        # makes a "fluo" frame come back with a BRIGHT background —
        # brightfield simply swamps the emission.
        want_tl = bool(channel.get("tl", not led_nm))
        try:
            self.session.changer("tl_shutter").set_open(want_tl)
        except Exception as e:
            logger.warning(
                "could not %s the transmitted-light shutter: %s",
                "open" if want_tl else "close", e,
            )

        pos = channel.get("reflector_pos") or 0
        if pos:
            # 0 means "leave the reflector alone" — brightfield.
            self.session.changer("reflector").set_position(int(pos))
        if self.mmc is not None:
            self.mmc.setExposure(float(channel["exposure_ms"]))

    def _on_snap_channels(self, _event=None) -> None:
        """Snap every channel in turn and show them side by side.

        Sequential rather than simultaneous: one camera, one filter
        cube, so channels are inherently acquired one after another —
        the same order the acquisition loop will use.
        """
        if self.mmc is None:
            self._say("Camera not available.", "warn")
            return
        if not self._require_connection():
            return
        chans = self.channels()
        if not chans:
            self._say("No channels defined.", "warn")
            return

        self._stop_live()
        panes, summary = [], []
        for ch in chans:
            try:
                self._apply_channel(ch)
                self.mmc.snapImage()
                img = np.asarray(self.mmc.getImage())
            except Exception as e:
                self._say(f"Channel {ch['name']} failed: {e}", "err")
                logger.exception("channel snap failed")
                continue
            mn, mx = float(img.min()), float(img.max())
            std = float(img.std())
            summary.append(
                f"{ch['name']}: refl={ch['reflector_pos'] or '-'} "
                f"exp={ch['exposure_ms']:g}ms "
                f"min={mn:.0f} max={mx:.0f} std={std:.1f}"
            )
            panes.append(pn.Column(
                pn.pane.Markdown(
                    f"**{ch['name']}**  \n"
                    f"<span style='font-size:0.75em'>refl "
                    f"{ch['reflector_pos'] or '-'} · "
                    f"{ch['exposure_ms']:g} ms · std {std:.0f}</span>",
                    width=260,
                ),
                pn.pane.PNG(_stretch_to_png(img), width=260),
            ))

        self.channel_previews.objects = panes
        self.img_stats.object = "\n".join(summary)
        self._say(f"Snapped {len(panes)} channel(s).", "ok")

    def _on_read_lightpath(self, _event=None) -> None:
        """Build a dropdown per light-path changer, from the hardware.

        Element NAMES may be offset by one: GetElement(MaxPosition)
        raises on every changer here, which points at a 0-based element
        array behind a 1-based Position. SetPosition takes the 1-based
        position and is unambiguous, so switching is reliable even
        while the labels are uncertain — the position number is shown
        alongside each name so a mismatch is visible.
        """
        if not self._require_connection():
            return
        try:
            devices = self.session.light_path()
        except Exception as e:
            self.lightpath_note.object = (
                f"❌ Could not read the light path: {e}"
            )
            logger.exception("light path read failed")
            return

        rows = []
        self._lightpath_widgets = {}
        for role in LIGHT_PATH_ROLES:
            dev = devices.get(role)
            if dev is None:
                continue
            try:
                elements = dev.elements()
                cur = dev.position
            except Exception as e:
                rows.append(pn.pane.Markdown(
                    f"**{role}** — unreadable ({e})", width=600))
                continue

            options, current_label = {}, None
            for el in elements:
                nm = el["name"] or f"<{el['error'] or 'unnamed'}>"
                label = f"{el['index']}: {nm}"
                options[label] = el["index"]
                if el["current"]:
                    current_label = label

            select = pn.widgets.Select(
                name=f"{role} ({dev.name})",
                options=list(options), value=current_label,
                width=420,
            )
            btn = pn.widgets.Button(name="Set", width=70)
            btn.on_click(
                lambda e, r=role, sel=select, opts=options:
                self._set_lightpath(r, opts.get(sel.value))
            )
            self._lightpath_widgets[role] = (select, options)
            rows.append(pn.Row(select, btn))

        if not rows:
            rows.append(pn.pane.Markdown(
                "_No light-path components reachable._", width=600))
        self.lightpath_rows.objects = rows
        self.lightpath_note.object = (
            "Light travels shutter → condenser → objective → reflector "
            "→ sideport. **The camera is on the LEFT sideport**, so a "
            "sideport on `100% R` gives it nothing. For brightfield the "
            "condenser wants a non-phase position, and the transmitted "
            "lamp must be on (the Colibri is fluorescence excitation "
            "only). Position numbers are authoritative; names may be "
            "offset by one."
        )
        self._say("Light path read.", "ok")

    def _set_lightpath(self, role: str, index) -> None:
        """Move one light-path changer, then re-read everything."""
        if index is None:
            self._say("Nothing selected.", "warn")
            return
        if not self._require_connection():
            return
        try:
            dev = self.session.changer(role)
            self._say(f"Setting {role} to position {index} …")
            dev.set_position(int(index))
        except Exception as e:
            self._say(f"Could not set {role}: {e}", "err")
            logger.exception("light path set failed for %s", role)
            return
        self._on_read_lightpath()
        self._say(f"{role} set to position {index}.", "ok")

    def _on_read_objective(self, _event=None) -> None:
        """Ask MTB which objective is in, and derive the pixel pitch.

        Wrong pixel pitch mis-scales every correction, so deriving it
        beats trusting a typed-in number. Zeiss objective names carry
        the magnification, which is the fallback if MTB's typed
        property is unavailable.
        """
        if not self._require_connection():
            return
        try:
            obj = self.session.objective()
            pos = obj.position
            slots = obj.slots()
        except Exception as e:
            self.objective_info.object = (
                f"❌ Could not read the nosepiece: {e}"
            )
            logger.exception("objective read failed")
            return

        cam = float(self.camera_pixel_um.value)
        adapter = float(self.adapter_mag.value)

        # Show every slot: the indexing convention is not fully settled
        # (GetElementCount()=6 but GetElement(6) is out of bounds), so
        # the operator confirms against the light path rather than
        # trusting one possibly off-by-one lookup.
        rows = ["| Position | Objective | Mag | NA | Pixel µm |",
                "|---|---|---|---|---|"]
        options = {}
        for s in slots:
            marker = " ◀ current" if s["index"] == pos else ""
            if s["error"]:
                rows.append(
                    f"| {s['index']}{marker} | _{s['error']}_ | | | |"
                )
                continue
            if s["empty"]:
                rows.append(f"| {s['index']}{marker} | _empty_ | | | |")
                continue
            mag = s["magnification"]
            pitch = (sample_pixel_size_um(cam, mag, adapter)
                     if mag else None)
            rows.append(
                f"| {s['index']}{marker} | `{s['name']}` | "
                f"{mag:g}× | {s['aperture'] or ''} | "
                f"{pitch:.4f} |" if pitch else
                f"| {s['index']}{marker} | `{s['name']}` | | | |"
            )
            if pitch:
                label = (f"Position {s['index']}: {s['name']} "
                         f"({mag:g}×, {pitch:.4f} µm)")
                options[label] = (s["index"], pitch)

        lines = ["\n".join(rows)]

        self.objective_choice.options = list(options)
        self._objective_slots = {k: v[0] for k, v in options.items()}
        self._objective_pitches = {k: v[1] for k, v in options.items()}

        current = next((s for s in slots if s["index"] == pos), None)
        if current and not current["empty"] and current["magnification"]:
            pitch = sample_pixel_size_um(
                cam, current["magnification"], adapter
            )
            self.pixel_xy.value = round(pitch, 4)
            lines.append(
                f"Current slot **{pos}**: "
                f"`{current['name']}` → pixel pitch "
                f"**{pitch:.4f} µm**. _Pixel size XY updated._"
            )
            for label, idx in self._objective_slots.items():
                if idx == pos:
                    self.objective_choice.value = label
                    break
        else:
            lines.append(
                f"⚠️ **Slot {pos} is empty** — no objective in the "
                f"light path, so the pixel pitch cannot be derived. "
                f"Pick an objective below and click **Rotate to it**; "
                f"the turret will turn and the pitch is then read back "
                f"from the hardware."
            )
            if options:
                first = list(options)[0]
                self.objective_choice.value = first

        self.objective_info.object = "\n\n".join(lines)

    def _on_apply_objective(self, _event=None) -> None:
        """Rotate the nosepiece to the chosen slot, then re-read.

        Rotating and re-reading is better than trusting a dropdown:
        the pixel pitch ends up derived from the objective actually in
        the light path.
        """
        if not self._require_connection():
            return
        choice = self.objective_choice.value
        slot = getattr(self, "_objective_slots", {}).get(choice)
        if slot is None:
            self._say("Read the objectives first, then pick one.",
                      "warn")
            return

        try:
            obj = self.session.objective()
            self._say(f"Rotating nosepiece to slot {slot} …")
            obj.set_position(int(slot))
        except Exception as e:
            self._say(f"Could not rotate the nosepiece: {e}", "err")
            logger.exception("nosepiece rotation failed")
            return

        # Re-read so the pitch comes from the hardware, not the choice.
        self._on_read_objective()

    @property
    def tracking_requested(self) -> bool:
        return bool(self.auto_track.value)

    def _refresh_roi_state(self) -> str:
        """Report which positions have ROIs, and return a summary."""
        state = self.roi_status()
        if not state:
            self.roi_state.object = (
                "_No positions captured yet._"
            )
            return ""
        ready = [n for n, ok in state.items() if ok]
        missing = [n for n, ok in state.items() if not ok]
        lines = [
            f"ROIs found for **{len(ready)}/{len(state)}** position(s)."
        ]
        if missing:
            lines.append(
                "Missing: " + ", ".join(f"`{n}`" for n in missing)
            )
            lines.append(
                "Acquire at least one timepoint first, then draw ROIs "
                "with the **Selection** dashboard "
                "(`panel serve interactive_tools/panel_app.py`) — it "
                "writes `<position>/embryo_tracking/tracking_RoIs.json`."
            )
        self.roi_state.object = "\n\n".join(lines)
        return "\n".join(lines)

    def _validate_run(self) -> str | None:
        """Return an error message, or None when good to go."""
        if not self.connected:
            return "Hardware is not connected."
        if self.mmc is None:
            return ("Camera is unavailable — tracking needs images. "
                    "Reconnect, and check ZEN is closed.")
        if MicroscopeInterface_MTB is None:
            return (f"Tracking backend unavailable: "
                    f"{_BACKEND_IMPORT_ERROR}")
        if not self.pos_table.value.shape[0]:
            return ("No positions defined. Jog to each embryo and use "
                    "*Capture current position*.")
        root = self.outdir.value.strip()
        if not root:
            return "Set an experiment root directory."
        # Missing ROIs are NOT an error: acquisition starts first and
        # the tracker attaches once they are drawn. Only guard against
        # overwriting a previous experiment's frames.
        if os.path.isdir(root) and os.listdir(root):
            existing = self.roi_status(root)
            if not any(existing.values()):
                return (f"{root} already exists and is not empty — pick "
                        f"a fresh folder so frames cannot collide.")
        return None

    def _on_run(self, _event=None) -> None:
        problem = self._validate_run()
        if problem:
            self.run_status.object = f"⚠️ {problem}"
            return

        root = self.outdir.value.strip()
        os.makedirs(root, exist_ok=True)

        chans = self.channels()
        if not chans:
            self.run_status.object = "⚠️ No channels defined."
            return
        first = chans[0]

        params = {
            "z_axis": self.z_axis.value,
            "exposure_ms": first["exposure_ms"],
            "channel": first["name"],
            # Every channel, so the backend can cycle them per
            # position. Each carries its own reflector cube, exposure
            # and whether it is a stack.
            "channels": chans,
            "apply_channel": self._apply_channel,
            "interval_s": float(self.interval_s.value),
            "settle_s": float(self.settle_s.value),
            "max_xy_um": float(self.max_xy.value),
            "max_z_um": float(self.max_z.value),
            "stop_after_tp": (int(self.n_timepoints.value)
                              if self.n_timepoints.value else None),
            # Hand over our session AND our camera: MTB allows one
            # Login per process, and PVCAM one open handle per camera.
            "session": self.session,
            "mmc": self.mmc,
            "camera_label": self._cam_label,
        }
        if self.zstack_on.value:
            params["z_stack"] = {
                "range_um": float(self.zstack_range.value),
                "step_um": float(self.zstack_step.value),
            }

        try:
            self._runner = MicroscopeInterface_MTB(
                self.positions_config(root), root, params
            )
        except Exception as e:
            self.run_status.object = f"❌ Could not start: {e}"
            return

        # The tracker is NOT built here. Acquisition must start first so
        # ROIs can be drawn on real frames; _run_loop attaches the
        # tracker once they appear.
        self._tracker = None

        # The ROI watcher inside TrackingRunner watches DIRECTORIES, so
        # they must exist before it starts. Create them now, which also
        # gives the Selection dashboard somewhere to save into.
        for name in self.positions_config():
            try:
                os.makedirs(os.path.join(root, name, "embryo_tracking"),
                            exist_ok=True)
            except Exception as e:
                logger.warning("could not create ROI dir for %s: %s",
                               name, e)

        # The camera is exclusive — live snapping must not interleave.
        self._stop_live()
        self._stop_flag.clear()
        self._run_thread = threading.Thread(
            target=self._run_loop, name="mtb-panel-run", daemon=True
        )
        self._run_thread.start()

        self.btn_run.disabled = True
        self.btn_stop.disabled = False
        self.run_status.object = (
            f"▶ Tracking {len(self.positions_config())} position(s) "
            f"into `{root}`"
        )
        self._append_log("run started")

    def _build_tracker(self, root: str):
        """Construct a TrackingRunner over our MTB backend.

        Position discovery is deliberately a MERGE of two sources:

          * get_pos_config() scans the experiment root for
            <position>/embryo_tracking/tracking_RoIs.json and supplies
            the ROI geometry and log_dir — so only positions with ROIs
            drawn are trackable, which is the existing convention.
          * our captured table supplies xyz_um, the stage baseline,
            which exists nowhere on disk.

        run_zeiss() is the right loop despite the name: it is the
        variant that owns its own timing and watches the ROI JSON files
        for changes, so ROIs can be redrawn mid-run.
        """
        config_path = os.path.join(
            _ROOT, "tracking_tools", "tracking_config.yaml"
        )
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        runner_config = config["tracking_runner"]
        roi_tracker_config = dict(config["roi_tracker"])
        roi_tracker_config["serverkit"] = bool(self.serverkit.value)
        position_tracker_config = {
            "pixel_size_xy": float(self.pixel_xy.value),
            "pixel_size_z": float(self.pixel_z.value),
            "tracking_2d": bool(self.tracking_2d.value),
        }
        log_dir_name = runner_config["log_dir_name"]

        # get_pos_config() globs <root>/*/<log_dir_name> and opens
        # tracking_RoIs.json UNCONDITIONALLY, so it raises
        # FileNotFoundError for any position whose ROI folder exists but
        # is still empty. We create those folders up front (the ROI
        # watcher needs them to exist), so ask per-position and only for
        # the ones that actually have a JSON.
        ready = [n for n, ok in self.roi_status(root).items() if ok]
        if not ready:
            raise RuntimeError(
                f"No positions with ROIs under {root}. Each position "
                f"folder needs {log_dir_name}/tracking_RoIs.json."
            )

        discovered = {}
        for name in ready:
            try:
                discovered.update(
                    get_pos_config(root, log_dir_name,
                                   position_name=name)
                )
            except Exception as e:
                logger.warning(
                    "could not load ROI config for %s: %s", name, e
                )

        captured = self.positions_config(root, log_dir_name)
        merged = {}
        for name, entry in discovered.items():
            if name not in captured:
                logger.warning(
                    "position %s has ROIs but no captured baseline — "
                    "skipping, the tracker cannot correct it", name
                )
                continue
            merged[name] = dict(entry)
            merged[name]["xyz_um"] = captured[name]["xyz_um"]

        if not merged:
            raise RuntimeError(
                "No position has BOTH a captured baseline and ROIs. "
                "Capture positions here, acquire, then draw ROIs."
            )

        untracked = [n for n in self.positions_config()
                     if n not in merged]
        if untracked:
            # run_zeiss() logs "No tracker for [pos] — skipping" and
            # keeps acquiring, so these still get imaged; they just are
            # not drift-corrected. The ROI watcher only watches folders
            # in positions_config, so ROIs drawn for them AFTER this
            # point need a restart to take effect.
            logger.warning(
                "acquiring but NOT tracking (no ROIs): %s",
                ", ".join(untracked),
            )
            self._push_log(
                f"NOT tracked (no ROIs): {', '.join(untracked)} — "
                f"they keep acquiring; restart to pick up ROIs added "
                f"for them later"
            )

        # Keep the backend's view consistent with what we will track.
        self._runner.positions_config = merged
        self._runner.pos_names = list(merged.keys())
        for name in merged:
            self._runner.refresh_filename(name)

        return TrackingRunner(
            positions_config=merged,
            microscope_interface=self._runner,
            dirpath=root,
            runner_params=runner_config,
            roi_tracker_params=roi_tracker_config,
            position_tracker_params=position_tracker_config,
        )

    def _run_loop(self) -> None:
        """Drain frames from the backend until stopped.

        Runs on a worker thread; the camera and stage are driven by the
        backend's own acquisition thread. Tracker integration is the
        next step — for now this consumes frames so the loop advances
        and the operator can watch the series build.
        """
        iface = self._runner
        root = self.outdir.value.strip()
        try:
            iface.connect()
            self._push_log("acquiring (open loop — no ROIs yet)")

            next_check = 0.0
            poll = max(1.0, float(self.roi_poll_s.value))

            # Open-loop phase: acquire and save while the operator draws
            # ROIs on the frames appearing on disk.
            while not self._stop_flag.is_set():
                if self.tracking_requested and time.monotonic() >= next_check:
                    next_check = time.monotonic() + poll
                    state = self.roi_status(root)
                    ready = [n for n, ok in state.items() if ok]
                    pending = [n for n, ok in state.items() if not ok]
                    if ready and not (self.require_all_rois.value
                                      and pending):
                        self._push_log(
                            f"ROIs found for {len(ready)}/{len(state)} "
                            f"position(s): {', '.join(ready)} — "
                            f"attaching tracker"
                        )
                        break
                    if ready and pending:
                        self._push_log(
                            f"waiting for ROIs on "
                            f"{', '.join(pending)} "
                            f"({len(ready)}/{len(state)} ready)"
                        )

                # (None, None, None) on timeout — the contract every
                # backend shares; stop_requested says whether more
                # frames are coming.
                img, tp, pos_name = iface.wait_for_image(timeout_ms=500)
                if img is None:
                    if iface.stop_requested:
                        return
                    continue
                self._push_log(
                    f"t={tp} {pos_name} "
                    f"mean={float(np.mean(img)):.0f} (open loop)"
                )

            if self._stop_flag.is_set():
                return

            # Closed-loop phase: hand consumption to TrackingRunner. The
            # acquisition thread keeps running untouched — connect() is
            # idempotent — so this is a change of consumer, not a
            # restart. Frames are already on disk, so losing one across
            # the handoff costs nothing.
            try:
                self._tracker = self._build_tracker(root)
            except Exception as e:
                logger.exception("tracker construction failed")
                self._push_log(f"ERROR building tracker: {e}")
                self._push_log("continuing open loop")
                while not self._stop_flag.is_set():
                    img, _, _ = iface.wait_for_image(timeout_ms=500)
                    if img is None and iface.stop_requested:
                        break
                return

            self._push_status("▶ Closed-loop tracking active")
            self._push_log("closed-loop tracking via TrackingRunner")
            self._tracker.run_zeiss()

        except Exception as e:
            logger.exception("run loop failed")
            self._push_log(f"ERROR {e}")
        finally:
            try:
                iface.disconnect()
            except Exception:
                pass
            self._push_done()

    def _push_log(self, msg: str) -> None:
        """Update the log from a worker thread, safely."""
        try:
            pn.state.execute(lambda: self._append_log(msg))
        except Exception:
            self._append_log(msg)

    def _push_status(self, msg: str) -> None:
        """Update the run status from a worker thread."""
        def apply():
            self.run_status.object = msg
        try:
            pn.state.execute(apply)
        except Exception:
            self.run_status.object = msg

    def _push_done(self) -> None:
        def finish():
            self.btn_run.disabled = False
            self.btn_stop.disabled = True
            self.run_status.object = "_Idle._"
            self._append_log("run finished")
            self._update_position()
        try:
            pn.state.execute(finish)
        except Exception:
            finish()

    def _on_stop(self, _event=None) -> None:
        self._stop_flag.set()
        if self._tracker is not None:
            # TrackingRunner polls this between frames.
            try:
                self._tracker.stop_requested = True
            except Exception as e:
                logger.warning("could not stop tracker: %s", e)
        if self._runner is not None:
            try:
                self._runner.stop()
            except Exception as e:
                logger.warning("stop complained: %s", e)
        self.run_status.object = "Stopping…"
        self._append_log("stop requested")

    # --------------------------------------------------------- layout

    def layout(self) -> pn.Row:
        """Tabs on the left, camera PINNED on the right.

        The image has to stay visible whatever you are doing — jogging
        the stage, setting the light path, editing channels — so it
        lives outside the tab set rather than inside any one tab.
        """
        camera_column = pn.Column(
            "### Camera",
            pn.Row(self.exposure, self.btn_snap),
            pn.Row(self.btn_live, self.live_fps),
            self.img_stats,
            self.preview,
            pn.layout.Divider(),
            self.pos_readout,
            width=620,
        )

        tab_stage = pn.Column(
            "### Stage",
            pn.Row(self.xy_step, self.z_step, self.btn_refresh),
            pn.Row(
                pn.Column(
                    pn.Row(pn.Spacer(width=70), self.btn_yp),
                    pn.Row(self.btn_xm, self.btn_xp),
                    pn.Row(pn.Spacer(width=70), self.btn_ym),
                ),
                pn.Spacer(width=30),
                pn.Column(self.btn_zp, self.btn_zm),
            ),
            pn.layout.Divider(),
            "### Positions",
            pn.Row(self.btn_capture, self.btn_goto),
            self.pos_table,
            pn.Row(self.btn_remove, self.btn_clear),
            pn.Row(self.positions_file, self.btn_save_pos,
                   self.btn_load_pos),
            width=620,
        )

        tab_optics = pn.Column(
            "### Objective",
            pn.Row(self.camera_pixel_um, self.adapter_mag,
                   self.btn_read_objective),
            self.objective_info,
            pn.Row(self.objective_choice, self.btn_apply_objective),
            pn.layout.Divider(),
            "### Light path — manual control",
            pn.pane.Markdown(
                "These **Set** buttons move the hardware **right now**. "
                "Use them to get an image looking the way you want. "
                "Once it does, go to *Channels* and press **Capture "
                "light path → channel** to save that state as a "
                "channel the acquisition loop will reproduce.\n\n"
                "_Reading never moves anything; it only reports where "
                "things already are._",
                width=600, styles={"font-size": "0.85em"},
            ),
            pn.Row(self.btn_read_lightpath),
            self.lightpath_rows,
            self.lightpath_note,
            pn.layout.Divider(),
            "### Illumination",
            pn.Row(self.btn_read_leds, self.btn_tl_on, self.btn_tl_off),
            self.led_rows,
            self.led_note,
            pn.layout.Divider(),
            self.hw_info,
            width=620,
        )

        tab_channels = pn.Column(
            "### Channels",
            pn.pane.Markdown(
                "A channel is a **saved light-path recipe**. The "
                "acquisition loop applies each one in turn at every "
                "position. Set the optics up by hand on the *Optics* "
                "tab, then capture it here.",
                width=600, styles={"font-size": "0.85em"},
            ),
            pn.Row(self.btn_capture_channel, self.btn_add_channel),
            pn.Row(self.btn_snap_channels),
            pn.pane.Markdown(
                "_With **Live** running you can change the reflector "
                "or LED and watch the effect immediately — that is the "
                "intended way to dial a channel in before capturing "
                "it._",
                width=600, styles={"font-size": "0.8em"},
            ),
            self.chan_table,
            self.chan_note,
            pn.layout.Divider(),
            pn.Row(self.btn_del_channel, self.btn_leds_off),
            self.led_info,
            pn.layout.Divider(),
            "#### Snap result per channel",
            self.channel_previews,
            width=620,
        )

        tab_acquire = pn.Column(
            "### Timing",
            pn.Row(self.interval_s, self.n_timepoints, self.settle_s),
            pn.layout.Divider(),
            "### Z-stack",
            pn.Row(self.zstack_on, self.zstack_range, self.zstack_step),
            pn.pane.Markdown(
                "Applies to channels with `z_stack` ticked. Keep the "
                "step above the actuator's resolution (~0.06 µm for the "
                "piezo) — finer steps do not move and silently "
                "duplicate planes.",
                width=600, styles={"font-size": "0.8em"},
            ),
            pn.layout.Divider(),
            "### Drift correction",
            pn.Row(self.pixel_xy, self.pixel_z),
            pn.Row(self.tracking_2d, self.serverkit),
            pn.Row(self.max_xy, self.max_z),
            width=620,
        )

        tab_run = pn.Column(
            "### Run",
            self.outdir,
            pn.pane.Markdown(
                "**Start** begins acquiring immediately. Draw ROIs in "
                "the Selection dashboard while it runs and the tracker "
                "attaches to the running loop — no restart.",
                width=600, styles={"font-size": "0.85em"},
            ),
            pn.Row(self.auto_track),
            pn.Row(self.require_all_rois),
            pn.Row(self.roi_poll_s, self.btn_check_rois),
            self.roi_state,
            pn.layout.Divider(),
            self.run_status,
            pn.Row(self.btn_run, self.btn_stop),
            self.run_log,
            width=620,
        )

        left = pn.Column(
            "# Axio Observer 7",
            pn.Row(self.btn_connect, self.z_axis),
            self.status,
            pn.Tabs(
                ("Stage", tab_stage),
                ("Optics", tab_optics),
                ("Channels", tab_channels),
                ("Acquisition", tab_acquire),
                ("Run", tab_run),
                dynamic=False,
            ),
            width=640,
        )

        return pn.Row(left, pn.Spacer(width=20), camera_column,
                      width=1300)


def app() -> pn.Column:
    return MTBSetupPanel().layout()


if __name__.startswith("bokeh") or __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app().servable(title="Axio Observer 7 — Setup")
