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

pn.extension("tabulator", notifications=True)

logger = logging.getLogger(__name__)

# Jog step sizes offered in the UI (um). The stage's own repeatability
# is ~1.9 um, so anything below ~2 um is not meaningfully commandable
# on XY; the piezo resolves far finer.
XY_STEPS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
Z_STEPS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

DEFAULT_EXPOSURE_MS = 5.0     # 5-20 ms is the good range; 50 saturates


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
        self._run_thread = None
        self._stop_flag = threading.Event()

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
        self.preview = pn.pane.PNG(None, width=600)
        self.img_stats = pn.pane.Markdown(
            "", width=600, styles={"font-size": "0.8em",
                                   "font-family": "monospace"},
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
        self.chan_table = pn.widgets.Tabulator(
            pd.DataFrame(
                [{"name": "BF", "exposure_ms": DEFAULT_EXPOSURE_MS}]
            ),
            height=140, width=380, show_index=False,
        )
        self.chan_note = pn.pane.Markdown(
            "⚠️ Channels currently vary **exposure only**. Colibri LED "
            "switching is not wired up yet: `IMTBFluorescenceLEDControl` "
            "exposes `NumberOfLEDs`, an `LED[]` array and "
            "`SetOperationMode`, but no brightness/on-off member — that "
            "needs one more API probe before it can be done correctly "
            "rather than guessed.",
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
        self.outdir = pn.widgets.TextInput(
            name="Experiment root", value="", width=560,
            placeholder=r"D:\Users\zeiss\data\my_experiment",
        )

        self.btn_run = pn.widgets.Button(
            name="▶ Run tracking", button_type="success", width=170
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
        try:
            from tracking_tools.microscope_interface.mtb import (
                MTBMotion, MTBSession,
            )
        except ImportError as e:
            self._say(f"Cannot import the MTB layer: {e}", "err")
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
        if not self.connected:
            return
        try:
            from tracking_tools.microscope_interface.mtb import MTBMotion
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
        try:
            self.preview.object = _stretch_to_png(img)
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

    def positions_config(self) -> dict:
        """The position list in the shape the backend expects."""
        cfg = {}
        for _, r in self.pos_table.value.iterrows():
            cfg[str(r["name"])] = {
                "xyz_um": (float(r["x_um"]), float(r["y_um"]),
                           float(r["z_um"]))
            }
        return cfg

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

    def _validate_run(self) -> str | None:
        """Return an error message, or None when good to go."""
        if not self.connected:
            return "Hardware is not connected."
        if self.mmc is None:
            return ("Camera is unavailable — tracking needs images. "
                    "Reconnect, and check ZEN is closed.")
        if not self.pos_table.value.shape[0]:
            return ("No positions defined. Jog to each embryo and use "
                    "*Capture current position*.")
        root = self.outdir.value.strip()
        if not root:
            return "Set an experiment root directory."
        if os.path.isdir(root) and os.listdir(root):
            return (f"{root} already exists and is not empty — pick a "
                    f"fresh folder so frames cannot collide.")
        return None

    def _on_run(self, _event=None) -> None:
        problem = self._validate_run()
        if problem:
            self.run_status.object = f"⚠️ {problem}"
            return

        from tracking_tools.microscope_interface.mtb_backend import (
            MicroscopeInterface_MTB,
        )

        root = self.outdir.value.strip()
        os.makedirs(root, exist_ok=True)

        channels = self.chan_table.value
        first = channels.iloc[0] if channels.shape[0] else None
        channel_name = str(first["name"]) if first is not None else "BF"
        exposure = (float(first["exposure_ms"]) if first is not None
                    else float(self.exposure.value))

        params = {
            "z_axis": self.z_axis.value,
            "exposure_ms": exposure,
            "channel": channel_name,
            "interval_s": float(self.interval_s.value),
            "settle_s": float(self.settle_s.value),
            "max_xy_um": float(self.max_xy.value),
            "max_z_um": float(self.max_z.value),
            "stop_after_tp": (int(self.n_timepoints.value)
                              if self.n_timepoints.value else None),
            # Hand over our session: one Login per process.
            "session": self.session,
        }
        if self.zstack_on.value:
            params["z_stack"] = {
                "range_um": float(self.zstack_range.value),
                "step_um": float(self.zstack_step.value),
            }

        try:
            self._runner = MicroscopeInterface_MTB(
                self.positions_config(), root, params
            )
        except Exception as e:
            self.run_status.object = f"❌ Could not start: {e}"
            return

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

    def _run_loop(self) -> None:
        """Drain frames from the backend until stopped.

        Runs on a worker thread; the camera and stage are driven by the
        backend's own acquisition thread. Tracker integration is the
        next step — for now this consumes frames so the loop advances
        and the operator can watch the series build.
        """
        iface = self._runner
        try:
            iface.connect()
            while not self._stop_flag.is_set():
                item = iface.wait_for_image(timeout_ms=500)
                if item is None:
                    thread = getattr(iface, "_thread", None)
                    if thread is None or not thread.is_alive():
                        break
                    continue
                img, tp, pos_name = item
                self._push_log(
                    f"t={tp} {pos_name} "
                    f"mean={float(np.mean(img)):.0f} "
                    f"drift={iface.get_cum_drift(pos_name)}"
                )
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
        if self._runner is not None:
            try:
                self._runner.stop()
            except Exception as e:
                logger.warning("stop complained: %s", e)
        self.run_status.object = "Stopping…"
        self._append_log("stop requested")

    # --------------------------------------------------------- layout

    def layout(self) -> pn.Column:
        connect_row = pn.Row(self.btn_connect, self.z_axis)

        jog = pn.Column(
            "### Stage",
            pn.Row(self.pos_readout, self.btn_refresh),
            pn.Row(self.xy_step, self.z_step),
            pn.Row(
                pn.Column(
                    pn.Row(pn.Spacer(width=70), self.btn_yp),
                    pn.Row(self.btn_xm, self.btn_xp),
                    pn.Row(pn.Spacer(width=70), self.btn_ym),
                ),
                pn.Spacer(width=30),
                pn.Column(self.btn_zp, self.btn_zm),
            ),
        )

        camera = pn.Column(
            "### Camera",
            pn.Row(self.exposure, self.btn_snap),
            self.img_stats,
            self.preview,
        )

        positions = pn.Column(
            "### Positions",
            pn.Row(self.btn_capture, self.btn_goto),
            self.pos_table,
            pn.Row(self.btn_remove, self.btn_clear),
            pn.Row(self.positions_file, self.btn_save_pos,
                   self.btn_load_pos),
        )

        channels = pn.Column(
            "### Channels",
            self.chan_table,
            self.chan_note,
        )

        acquisition = pn.Column(
            "### Acquisition",
            pn.Row(self.interval_s, self.n_timepoints, self.settle_s),
            pn.Row(self.zstack_on, self.zstack_range, self.zstack_step),
            pn.Row(self.max_xy, self.max_z),
            self.outdir,
            self.run_status,
            pn.Row(self.btn_run, self.btn_stop),
            self.run_log,
        )

        return pn.Column(
            "# Axio Observer 7 — Setup",
            pn.pane.Markdown(
                "Motion via Zeiss **MTB**, camera via **Micro-Manager "
                "/ PVCAM**. Close ZEN before connecting — it holds the "
                "camera exclusively.",
                width=700, styles={"font-size": "0.9em"},
            ),
            connect_row,
            self.status,
            self.hw_info,
            pn.layout.Divider(),
            pn.Row(jog, pn.Spacer(width=40), camera),
            pn.layout.Divider(),
            pn.Row(positions, pn.Spacer(width=40), channels),
            pn.layout.Divider(),
            acquisition,
            width=1250,
        )


def app() -> pn.Column:
    return MTBSetupPanel().layout()


if __name__.startswith("bokeh") or __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app().servable(title="Axio Observer 7 — Setup")
