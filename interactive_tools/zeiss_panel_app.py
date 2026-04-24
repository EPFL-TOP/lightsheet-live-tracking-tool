"""
Zeiss Tracking Panel App
========================
Standalone Panel / Bokeh application for running the live-tracking tool
against a Zeiss microscope via the ZEN API, or offline from a CZI file.

Launch with::

    panel serve interactive_tools/zeiss_panel_app.py --show --port 5022

The app has two tabs:
  * **Tracking** — configure parameters, start / stop the tracking loop,
    and monitor the log output.
  * **ROI Selection** — embedded Bokeh ROI editor (existing ``bokeh_selection``
    dashboard).  Changes saved here are picked up automatically by the running
    tracker via the ``watchdog`` file-watcher in ``TrackingRunner.run_zeiss()``.

Workflow (first time / new experiment)
--------------------------------------
1. Fill in the ZEN Connection fields (pre-loaded from ``zeiss_config.ini``).
2. Set *Data directory* and *Number of scenes*.
3. Click **Capture Preview Frame** — connects to ZEN, grabs one frame per
   scene, saves to ``<data_dir>/<scene_folder>/max_proj/t0000.tif``, then
   disconnects.
4. Switch to the *ROI Selection* tab, load the preview TIF for each position,
   draw the ROI(s) and click Save.
5. Back in *Tracking*, click **Run Tracking**.
"""


"""
Spell-check only. The changes are complete. Here's a summary of everything that was done:

Root cause: Your ZEN API Gateway is an older build of ZEN 3.13 (pre-Autumn 2025). The ExperimentStreamingService/MonitorAllExperiments RPC was only added in the Autumn 2025 release of ZEN 3.13. The API Python package you have (2025.10.1) matches that newer release, but the running server doesn't implement it yet.

What changed:

1. MicroscopeInterface.py — three fixes:

 Stop the reconnect loop on UNIMPLEMENTED: detects GRPCError with status 12 and exits immediately with a clear actionable error instead of retrying every 5 seconds forever
 CZI file-poll mode: new _poll_czi_thread method — if czi_watch_dir is set, connect() starts a polling thread instead of streaming; it watches the directory for a CZI file ZEN is writing, reads each newly-completed timepoint with pylibCZIrw, and feeds frames into the same queue as the streaming path
 czi_watch_dir / czi_poll_interval_s params added to __init__

2. zeiss_config.ini — new [czi_fallback] section documenting czi_watch_dir and czi_poll_interval_s

3. zeiss_panel_app.py — new widgets for CZI watch directory and poll interval, wired into _load_zeiss_config() and both zeiss_params dicts

To use the file-poll mode now:

1. In the Panel app, set CZI watch directory to the folder where ZEN saves images (e.g. C:\Users\...\Documents\Carl Zeiss\ZENCore\Documents\Images)
2. Start the ZEN experiment — ZEN will write a .czi file there
3. Click Run Tracking — the interface will find the CZI, poll it every 5 seconds for new complete timepoints, and track normally
When you eventually upgrade to ZEN 3.13 Autumn 2025, just leave the CZI watch directory blank and streaming will work directly.
"""

import configparser
import os
import sys
import queue
import logging
import threading
import yaml

import panel as pn
import numpy as np

# Make the repo root importable when running via `panel serve`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pn.extension(sizing_mode='stretch_width')

# ─── Log capture ─────────────────────────────────────────────────────────────

_log_queue: queue.Queue = queue.Queue()


class _QueueHandler(logging.Handler):
    """Forwards log records to a thread-safe queue for Panel to consume."""
    def emit(self, record):
        _log_queue.put(self.format(record))


_queue_handler = _QueueHandler()
_queue_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(name)s %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
))

# ─── Widgets — acquisition parameters ────────────────────────────────────────

w_pixel_xy = pn.widgets.FloatInput(
    name='Pixel size x,y (µm)', value=0.347, step=0.001, width=220
)
w_pixel_z = pn.widgets.FloatInput(
    name='Step size z (µm)', value=1.0, step=0.1, width=220
)
w_dirpath = pn.widgets.TextInput(
    name='Data directory',
    placeholder='/path/to/experiment',
    width=420,
)
w_serverkit = pn.widgets.Checkbox(name='Use serverkit', value=True)
w_simulated = pn.widgets.Checkbox(name='Simulated microscope (CZI)', value=False)

# ─── Widgets — ZEN connection ─────────────────────────────────────────────────

w_zen_address = pn.widgets.TextInput(
    name='ZEN Gateway address', value='localhost', width=220
)
w_zen_port = pn.widgets.IntInput(name='Port', value=5002, width=100)
w_zen_cert = pn.widgets.TextInput(
    name='TLS certificate path',
    placeholder='C:/ProgramData/Carl Zeiss/.../CA_Root_Certificate.pem',
    width=420,
)
w_zen_token = pn.widgets.PasswordInput(
    name='Control token',
    placeholder='Paste token from GlobalControlToken.txt',
    width=420,
)
w_zen_expname = pn.widgets.TextInput(
    name='Experiment name in ZEN', placeholder='MyTimeLapse', width=280
)
w_z_proj = pn.widgets.Select(
    name='Z projection', options=['max', 'central_slice'], value='max', width=180
)
w_zen_channel = pn.widgets.IntInput(
    name='Tracking channel (0-based)', value=0, width=160
)
w_max_xy = pn.widgets.FloatInput(
    name='Max XY shift (µm)', value=500.0, step=10.0, width=180
)
w_max_z = pn.widgets.FloatInput(
    name='Max Z shift (µm)', value=100.0, step=5.0, width=180
)
w_czi_watch_dir = pn.widgets.TextInput(
    name='CZI watch directory (leave blank to use gRPC streaming)',
    placeholder='C:/Users/.../Documents/Carl Zeiss/ZENCore/Documents/Images',
    width=560,
)
w_czi_poll_interval = pn.widgets.FloatInput(
    name='CZI poll interval (s)', value=5.0, step=1.0, width=160
)

# ─── Widgets — preview capture ────────────────────────────────────────────────

w_n_scenes = pn.widgets.IntInput(
    name='Number of scenes / positions', value=1, start=1, width=200,
)
w_preview_timeout = pn.widgets.IntInput(
    name='Preview wait timeout (minutes)', value=20, start=1, width=220,
)
btn_preview = pn.widgets.Button(
    name='Capture Preview Frame', button_type='primary', width=220
)
_preview_info = pn.pane.Markdown(
    '_Connect to ZEN, wait for the next acquired frame per scene (up to the '
    'timeout above), save as TIF, then disconnect._',
    width=420,
)

# ─── Widgets — CZI simulation ─────────────────────────────────────────────────

w_czi_path = pn.widgets.TextInput(
    name='CZI file path', placeholder='/path/to/file.czi', width=420
)
w_tp_delay = pn.widgets.IntInput(
    name='Inter-timepoint delay (ms)', value=2000, width=220
)
w_start_tp = pn.widgets.IntInput(
    name='Starting timepoint', value=0, width=160
)

# ─── Run / Stop ───────────────────────────────────────────────────────────────

btn_run  = pn.widgets.Button(name='Run Tracking', button_type='success', width=180)
btn_stop = pn.widgets.Button(name='Stop',         button_type='danger',  width=120)
btn_stop.disabled = True

w_log = pn.widgets.TextAreaInput(
    name='Log output', value='', height=320, disabled=True, width=780
)

# ─── Load zeiss_config.ini → pre-fill widgets ────────────────────────────────

def _load_zeiss_config():
    """Read zeiss_config.ini from the repo root and apply values to widgets."""
    config_path = os.path.join(_ROOT, 'zeiss_config.ini')
    if not os.path.exists(config_path):
        return
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    w_zen_address.value  = cfg.get('host',       'address',            fallback='localhost')
    w_zen_port.value     = cfg.getint('host',    'port',               fallback=5002)
    w_zen_cert.value     = cfg.get('cert',        'path',              fallback='')
    w_zen_token.value    = cfg.get('api',         'control_token',     fallback='')
    w_zen_expname.value  = cfg.get('experiment',  'name',              fallback='')
    w_zen_channel.value  = cfg.getint('experiment', 'tracking_channel', fallback=0)
    w_z_proj.value       = cfg.get('experiment',  'z_projection',      fallback='max')
    w_max_xy.value       = cfg.getfloat('bounds', 'max_xy_um',         fallback=500.0)
    w_max_z.value        = cfg.getfloat('bounds', 'max_z_um',          fallback=100.0)
    w_czi_watch_dir.value    = cfg.get('czi_fallback', 'czi_watch_dir',       fallback='')
    w_czi_poll_interval.value = cfg.getfloat('czi_fallback', 'czi_poll_interval_s', fallback=5.0)


_load_zeiss_config()

# ─── Section visibility ───────────────────────────────────────────────────────

zen_section = pn.Column(
    pn.pane.Markdown('**ZEN Connection**'),
    pn.Row(w_zen_address, w_zen_port),
    w_zen_cert,
    w_zen_token,
    pn.Row(w_zen_expname, w_z_proj),
    pn.Row(w_zen_channel, w_max_xy, w_max_z),
    pn.layout.Divider(),
    pn.pane.Markdown(
        '**Fallback: CZI file-poll mode** — use when ZEN API Gateway < Autumn 2025\n\n'
        'Set the directory where ZEN saves its CZI output file.  Leave blank to use gRPC streaming.'
    ),
    pn.Row(w_czi_watch_dir, w_czi_poll_interval),
    pn.layout.Divider(),
    pn.pane.Markdown('**Preview capture** — run before defining ROIs'),
    _preview_info,
    pn.Row(w_n_scenes, w_preview_timeout),
    btn_preview,
)
sim_section = pn.Column(
    pn.pane.Markdown('**Simulation (CZI file)**'),
    w_czi_path,
    pn.Row(w_tp_delay, w_start_tp),
)


@pn.depends(w_simulated, watch=True)
def _toggle_sections(simulated):
    zen_section.visible = not simulated
    sim_section.visible = simulated


_toggle_sections(w_simulated.value)   # apply initial state

# ─── Runner state ─────────────────────────────────────────────────────────────

_state: dict = {'runner': None, 'microscope': None, 'thread': None}


# ─── Preview capture ──────────────────────────────────────────────────────────

def _run_capture_preview():
    """
    Connect to ZEN, wait for one frame per scene/position, save each as
    ``<data_dir>/<position>/max_proj/t0000.tif``, then disconnect.

    If no sub-folders exist under *data_dir* yet, ``scene_000``,
    ``scene_001``, … are created automatically (one per *w_n_scenes*).
    The user can rename them afterwards — alphabetical order must match ZEN
    scene order.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if _queue_handler not in root_logger.handlers:
        root_logger.addHandler(_queue_handler)

    try:
        from tracking_tools.microscope_interface.MicroscopeInterface import (
            MicroscopeInterface_Zeiss,
        )

        dirpath = w_dirpath.value.strip()
        if not dirpath or not os.path.isdir(dirpath):
            logging.error(
                "Data directory does not exist. "
                "Create it (or set a valid path) before capturing a preview."
            )
            return

        # Discover existing sub-folders or create scene_NNN placeholders
        subdirs = sorted([
            d for d in os.listdir(dirpath)
            if os.path.isdir(os.path.join(dirpath, d)) and not d.startswith('.')
        ])
        if not subdirs:
            n = w_n_scenes.value
            for i in range(n):
                os.makedirs(os.path.join(dirpath, f'scene_{i:03d}'), exist_ok=True)
            subdirs = sorted([f'scene_{i:03d}' for i in range(n)])
            logging.info(f"Created {n} position folder(s): {subdirs}")

        # MicroscopeInterface_Zeiss only needs position names from positions_config
        positions_config = {name: {} for name in subdirs}
        logging.info(f"Capturing preview for: {subdirs}")

        zeiss_params = {
            'address':              w_zen_address.value.strip(),
            'port':                 w_zen_port.value,
            'cert_path':            w_zen_cert.value.strip(),
            'control_token':        w_zen_token.value,
            'experiment_name':      w_zen_expname.value.strip(),
            'z_projection':         w_z_proj.value,
            'tracking_channel':     w_zen_channel.value,
            'max_xy_um':            w_max_xy.value,
            'max_z_um':             w_max_z.value,
            'czi_watch_dir':        w_czi_watch_dir.value.strip(),
            'czi_poll_interval_s':  w_czi_poll_interval.value,
        }

        microscope = MicroscopeInterface_Zeiss(
            positions_config=positions_config,
            dirpath=dirpath,
            zeiss_params=zeiss_params,
        )
        microscope.connect()
        total_wait_s   = w_preview_timeout.value * 60   # minutes → seconds
        poll_ms        = 20_000                          # 20 s per poll (keep loop responsive)
        elapsed_s      = 0
        received: dict = {}   # position_name → (image, tp)

        logging.info(
            f"Connected to ZEN API — waiting up to {w_preview_timeout.value} min "
            f"for the first frame per scene. Make sure the ZEN experiment is running."
        )

        while len(received) < len(subdirs) and elapsed_s < total_wait_s:
            image, tp, pos = microscope.wait_for_image(timeout_ms=poll_ms)
            elapsed_s += poll_ms / 1000

            if image is None:
                remaining = int(total_wait_s - elapsed_s)
                if remaining > 0:
                    logging.info(
                        f"No frame yet — elapsed {int(elapsed_s)}s, "
                        f"{remaining}s remaining."
                    )
                continue

            if pos not in received:
                received[pos] = (image, tp)
                tif_path = os.path.join(dirpath, pos, f't{tp:04d}.tif')
                logging.info(f"  [{pos}] preview saved → {tif_path}")

        if len(received) < len(subdirs) and elapsed_s >= total_wait_s:
            logging.warning(
                f"Timeout reached ({w_preview_timeout.value} min) before all "
                f"scenes were captured. Got: {sorted(received.keys())}"
            )

        microscope.disconnect()

        if received:
            paths = '\n'.join(
                f'  {dirpath}/{pos}/max_proj/t{tp:04d}.tif'
                for pos, (_, tp) in sorted(received.items())
            )
            logging.info(
                f"Preview capture complete.\n{paths}\n\n"
                "Next: open the ROI Selection tab, load each preview TIF,\n"
                "draw ROI(s) and click Save to create tracking_RoIs.json."
            )
        else:
            logging.error("No frames received — preview failed.")

    except Exception as e:
        logging.error(f"Preview capture error: {e}", exc_info=True)
    finally:
        pn.state.execute(lambda: setattr(btn_preview, 'disabled', False))


def _on_preview(event):
    btn_preview.disabled = True
    threading.Thread(
        target=_run_capture_preview, daemon=True, name='PreviewThread'
    ).start()


btn_preview.on_click(_on_preview)

# ─── Tracking run ─────────────────────────────────────────────────────────────

def _run_tracking():
    """Executed in a daemon thread; drives the full tracking lifecycle."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if _queue_handler not in root_logger.handlers:
        root_logger.addHandler(_queue_handler)

    try:
        config_path = os.path.join(_ROOT, 'tracking_tools', 'tracking_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        roi_tracker_config               = config['roi_tracker']
        roi_tracker_config['serverkit']  = w_serverkit.value
        position_tracker_config = {
            'pixel_size_xy': w_pixel_xy.value,
            'pixel_size_z':  w_pixel_z.value,
        }
        runner_config = config['tracking_runner']

        from tracking_tools.tracking_runner.TrackingRunner import TrackingRunner
        from tracking_tools.microscope_interface.MicroscopeInterface import (
            MicroscopeInterface_Zeiss,
            SimulatedMicroscopeInterface_Zeiss,
        )
        from tracking_tools.utils.tracking_utils import get_pos_config

        dirpath       = w_dirpath.value.strip()
        log_dir_name  = runner_config['log_dir_name']
        position_config = get_pos_config(dirpath, log_dir_name)

        if not position_config:
            logging.error(
                f"No trackable positions found in '{dirpath}'.\n"
                "Each position folder must contain "
                "embryo_tracking/tracking_RoIs.json.\n"
                "Use 'Capture Preview Frame' to get images, then define ROIs "
                "in the ROI Selection tab."
            )
            return

        if w_simulated.value:
            microscope = SimulatedMicroscopeInterface_Zeiss(
                positions_config=position_config,
                dirpath=dirpath,
                czi_path=w_czi_path.value.strip(),
                inter_timepoint_delay_ms=w_tp_delay.value,
                starting_timepoint=w_start_tp.value,
                z_projection=w_z_proj.value,
                tracking_channel=w_zen_channel.value,
            )
        else:
            zeiss_params = {
                'address':          w_zen_address.value.strip(),
                'port':             w_zen_port.value,
                'cert_path':        w_zen_cert.value.strip(),
                'control_token':    w_zen_token.value,
                'experiment_name':  w_zen_expname.value.strip(),
                'z_projection':     w_z_proj.value,
                'tracking_channel': w_zen_channel.value,
                'max_xy_um':        w_max_xy.value,
                'max_z_um':         w_max_z.value,
            }
            microscope = MicroscopeInterface_Zeiss(
                positions_config=position_config,
                dirpath=dirpath,
                zeiss_params=zeiss_params,
            )

        _state['microscope'] = microscope

        runner = TrackingRunner(
            positions_config=position_config,
            microscope_interface=microscope,
            dirpath=dirpath,
            runner_params=runner_config,
            roi_tracker_params=roi_tracker_config,
            position_tracker_params=position_tracker_config,
        )
        _state['runner'] = runner
        runner.run_zeiss()

    except Exception as e:
        logging.error(f"Tracking error: {e}", exc_info=True)
    finally:
        pn.state.execute(_reset_buttons)


def _reset_buttons():
    btn_run.disabled  = False
    btn_stop.disabled = True


def _on_run(event):
    btn_run.disabled  = True
    btn_stop.disabled = False
    w_log.value = ''
    t = threading.Thread(target=_run_tracking, daemon=True, name='ZeissTrackingThread')
    _state['thread'] = t
    t.start()


def _on_stop(event):
    runner    = _state.get('runner')
    microscope = _state.get('microscope')
    if runner:
        runner.stop_requested = True
    if microscope:
        microscope.stop_requested = True
    btn_run.disabled  = False
    btn_stop.disabled = True


btn_run.on_click(_on_run)
btn_stop.on_click(_on_stop)

# ─── Periodic log refresh ─────────────────────────────────────────────────────

_MAX_LOG_LINES = 300


def _update_log():
    lines = []
    try:
        while True:
            lines.append(_log_queue.get_nowait())
    except queue.Empty:
        pass
    if lines:
        current_lines = (w_log.value or '').splitlines()
        all_lines = current_lines + lines
        w_log.value = '\n'.join(all_lines[-_MAX_LOG_LINES:]) + '\n'


pn.state.add_periodic_callback(_update_log, period=500)

# ─── Layout ───────────────────────────────────────────────────────────────────

tracking_tab = pn.Column(
    pn.pane.Markdown('## Zeiss Microscope Live Tracking'),
    pn.layout.Divider(),
    pn.Row(
        pn.Column(
            pn.pane.Markdown('**Acquisition parameters**'),
            w_pixel_xy, w_pixel_z,
            w_dirpath,
            w_serverkit,
            w_simulated,
        ),
        pn.Spacer(width=30),
        pn.Column(zen_section, sim_section),
    ),
    pn.layout.Divider(),
    pn.Row(btn_run, btn_stop),
    w_log,
    sizing_mode='stretch_width',
)

# ROI selection tab — reuse the existing Bokeh dashboard
try:
    from interactive_tools.bokeh_selection import make_layout as _roi_make_layout
    roi_tab = pn.panel(_roi_make_layout, sizing_mode='stretch_both')
except Exception as _e:
    roi_tab = pn.pane.Alert(
        f'Could not load ROI selection dashboard: {_e}',
        alert_type='warning',
    )

tabs = pn.Tabs(
    ('Tracking',      tracking_tab),
    ('ROI Selection', roi_tab),
    sizing_mode='stretch_both',
)

tabs.servable()
