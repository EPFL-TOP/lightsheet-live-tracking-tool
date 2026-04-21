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
"""
 
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
 
btn_run  = pn.widgets.Button(label='▶  Run Tracking', button_type='success', width=180)
btn_stop = pn.widgets.Button(label='■  Stop',         button_type='danger',  width=120)
btn_stop.disabled = True
 
w_log = pn.widgets.TextAreaInput(
    name='Log output', value='', height=320, disabled=True, width=780
)
 
# ─── Section visibility ───────────────────────────────────────────────────────
 
zen_section = pn.Column(
    pn.pane.Markdown('**ZEN Connection**'),
    pn.Row(w_zen_address, w_zen_port),
    w_zen_cert,
    w_zen_token,
    pn.Row(w_zen_expname, w_z_proj),
    pn.Row(w_zen_channel, w_max_xy, w_max_z),
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
                "embryo_tracking/tracking_RoIs.json."
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
        # Keep only the last _MAX_LOG_LINES to avoid unbounded growth
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