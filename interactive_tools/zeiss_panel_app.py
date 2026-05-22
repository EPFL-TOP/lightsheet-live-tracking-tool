"""
Zeiss Tracking Panel App
========================
Standalone Panel / Bokeh application driving the live-tracking tool.

Default workflow (file-watching, no microscope feedback)
--------------------------------------------------------
1. Configure ZEN to save per-Z-slice TIFs into a *source* folder, named like
   ``<exp>_S0000(P4)_T000000_Z0000_C00_M0000_ORG.tif``.
2. Set *Experiment root* (where 3-D stacks + tracking outputs are written)
   and *Position folders* (comma-separated, ordered by ZEN scene index).
3. Click **Start Ingest** — a background thread watches the source folder,
   groups files by (S, T, C), waits for all Z-slices, writes
   ``<root>/<pos>/t{T:04d}_C{C:02d}.tif`` (one 3-D stack per channel).
4. Open the **ROI Selection** tab, load any stack, draw + save the ROI.
5. Click **Run Tracking**.  A file-watching microscope interface picks up
   new stacks matching each position's saved filename pattern and runs the
   tracker — no microscope hardware contacted, no stage feedback.

Streaming mode and stage feedback to ZEN are kept under the "Advanced"
expander and are unused until explicitly enabled.

Launch with::

    panel serve interactive_tools/zeiss_panel_app.py --show --port 5022
"""

import asyncio
import configparser
import os
import ssl
import sys
import queue
import logging
import threading
import yaml

import panel as pn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pn.extension(sizing_mode='stretch_width')

# ─── Log capture ─────────────────────────────────────────────────────────────

_log_queue: queue.Queue = queue.Queue()


class _QueueHandler(logging.Handler):
    """Forwards log records to a thread-safe queue for Panel to consume.

    The queue is held as an instance attribute (not looked up via the
    module's globals) so that background threads still emitting logs
    during interpreter shutdown do not hit a NameError when the module's
    globals have been cleared.
    """
    def __init__(self, log_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = log_queue

    def emit(self, record):
        try:
            self._queue.put_nowait(self.format(record))
        except Exception:
            # Swallow errors during shutdown / when the formatter chokes
            # on a partially-torn-down logger
            pass


_queue_handler = _QueueHandler(_log_queue)
_queue_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(name)s %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
))


# ─── Acquisition / experiment widgets ───────────────────────────────────────

w_dirpath = pn.widgets.TextInput(
    name='Experiment root — base folder; ingest creates scene_NNN inside',
    placeholder='/path/to/experiment-name',
    width=560,
)
w_n_scenes = pn.widgets.IntInput(
    name='Number of scenes (auto-creates scene_NNN subfolders)',
    value=1, start=1, width=320
)
w_n_channels = pn.widgets.IntInput(
    name='Number of channels (used for ingest validation)',
    value=1, start=1, width=320
)
w_pixel_xy = pn.widgets.FloatInput(
    name='Pixel size x,y (µm)', value=0.347, step=0.001, width=200
)
w_pixel_z = pn.widgets.FloatInput(
    name='Z step (µm)', value=1.0, step=0.1, width=200
)
w_n_z = pn.widgets.IntInput(
    name='Number of Z-slices per stack', value=1, start=1, width=200
)
w_tracking_2d = pn.widgets.Checkbox(
    name='2-D tracking only (Z stays constant)',
    value=False,
)
w_serverkit = pn.widgets.Checkbox(name='Use serverkit (CoTracker server)', value=True)


# ─── ZEN ingest widgets ─────────────────────────────────────────────────────

w_zen_source = pn.widgets.TextInput(
    name='ZEN source folder (where ZEN writes per-Z TIFs)',
    placeholder='H:/.../experiment_folder',
    width=560,
)
w_ingest_poll = pn.widgets.FloatInput(
    name='Ingest poll interval (s)', value=2.0, step=0.5, width=180
)
btn_ingest_start = pn.widgets.Button(
    name='Start Ingest', button_type='primary', width=160
)
btn_ingest_stop = pn.widgets.Button(
    name='Stop Ingest', button_type='warning', width=160, disabled=True
)
ingest_info = pn.pane.Markdown(
    '_The ZEN source folder is a single flat folder where ZEN drops every '
    "TIF (e.g. `pos2_S0001(P3)_T000002_Z0005_C02_M0000_ORG.tif`). The "
    'ingest groups files by (S, T, C), waits for all Z-slices, and writes '
    '3-D stacks under `Experiment root / scene_NNN / t{T:04d}_C{C:02d}.tif` — '
    "one stack per channel.  All channels are saved so you can switch the "
    'tracking channel mid-experiment by re-saving the ROI on a different '
    'channel from the ROI Selection tab.  Tracking later runs only on '
    'scenes that contain an `embryo_tracking/` subfolder._',
    width=600,
)


# ─── ZEN streaming + stage feedback (advanced, default-collapsed) ───────────

w_use_streaming = pn.widgets.Checkbox(
    name='Use ZEN gRPC streaming instead of TIF ingest (advanced)', value=False,
)
w_use_feedback = pn.widgets.Checkbox(
    name='Send relative_move shifts back to ZEN (advanced — default OFF)',
    value=False,
)
w_zen_address = pn.widgets.TextInput(name='ZEN Gateway address', value='localhost', width=220)
w_zen_port = pn.widgets.IntInput(name='Port', value=5002, width=120)
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
w_z_proj = pn.widgets.Select(
    name='Z projection', options=['max', 'central_slice'], value='max', width=180
)
w_max_xy = pn.widgets.FloatInput(name='Max XY shift (µm)', value=500.0, step=10.0, width=180)
w_max_z = pn.widgets.FloatInput(name='Max Z shift (µm)', value=100.0, step=5.0, width=180)


# ─── ZEN experiment control (API-started) ──────────────────────────────────
#
# ZEN's API can only manipulate (Status / TilesService / etc.) experiments
# that the API itself started.  To do multi-position position-list updates
# we need ZEN to run the experiment through ExperimentService.start_experiment
# rather than the ZEN UI's Start button — that way we own the experiment_id
# and can call TilesService.add_positions per acquisition cycle.

w_exp_name_select = pn.widgets.Select(
    name='Experiment name in ZEN', options=[''], value='', width=320,
)
w_exp_output_name = pn.widgets.TextInput(
    name='Output filename (no .czi extension)',
    placeholder='track_run_001',
    width=320,
)
btn_refresh_exps = pn.widgets.Button(
    name='Refresh experiment list', button_type='light', width=200,
)
btn_start_exp = pn.widgets.Button(
    name='Start experiment via ZEN API', button_type='primary', width=240,
)
btn_stop_exp = pn.widgets.Button(
    name='Stop experiment', button_type='warning', width=160, disabled=True,
)
exp_status_md = pn.pane.Markdown('_No experiment started via API_', width=560)


# ─── Run / Stop ─────────────────────────────────────────────────────────────

btn_run  = pn.widgets.Button(name='Run Tracking', button_type='success', width=180)
btn_stop = pn.widgets.Button(name='Stop',         button_type='danger',  width=120)
btn_stop.disabled = True

w_log = pn.widgets.TextAreaInput(
    name='Log output', value='', height=320, disabled=True, width=780
)


# ─── Load zeiss_config.ini → pre-fill widgets ───────────────────────────────

def _load_zeiss_config():
    config_path = os.path.join(_ROOT, 'zeiss_config.ini')
    if not os.path.exists(config_path):
        return
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    w_zen_address.value  = cfg.get('host',         'address',          fallback='localhost')
    w_zen_port.value     = cfg.getint('host',      'port',             fallback=5002)
    w_zen_cert.value     = cfg.get('cert',         'path',             fallback='')
    w_zen_token.value    = cfg.get('api',          'control_token',    fallback='')
    w_z_proj.value       = cfg.get('experiment',   'z_projection',     fallback='max')
    w_max_xy.value       = cfg.getfloat('bounds',  'max_xy_um',        fallback=500.0)
    w_max_z.value        = cfg.getfloat('bounds',  'max_z_um',         fallback=100.0)


_load_zeiss_config()


# ─── State ──────────────────────────────────────────────────────────────────

_state: dict = {
    'runner': None,
    'microscope': None,
    'tracking_thread': None,
    'ingest': None,
    # ZEN-API-started experiment bookkeeping (Phase 1: just track the id;
    # Phase 2 will use it for TilesService position updates per cycle).
    'experiment_id':   None,
    'experiment_name': None,
}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _get_position_names():
    """Position folder names auto-derived from the scene count.

    Each ZEN scene S{i:04d} is mapped to ``scene_{i:03d}`` under the
    experiment root.  The user picks which subset to track by saving an ROI
    in only those scenes' ``embryo_tracking/`` folders — the tracking step
    discovers them via ``get_pos_config``.
    """
    n = max(1, int(w_n_scenes.value or 0))
    return [f'scene_{i:03d}' for i in range(n)]


def _ensure_root_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if _queue_handler not in root_logger.handlers:
        root_logger.addHandler(_queue_handler)


# ─── ZEN experiment control (API-started) ──────────────────────────────────

def _open_grpc_channel_sync():
    """Open a fresh grpclib channel using the panel's ZEN connection widgets.

    Returns ``(channel, metadata)``.  Caller must close the channel.
    SSL setup mirrors MicroscopeInterface_Zeiss / _Files: h2 ALPN is
    mandatory for the ZEN Gateway to negotiate HTTP/2.
    """
    import grpclib.client
    host  = w_zen_address.value.strip()
    port  = int(w_zen_port.value)
    cert  = w_zen_cert.value.strip()
    token = w_zen_token.value

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if cert and os.path.exists(cert):
        ctx.load_verify_locations(cafile=cert)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True
    else:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    ctx.set_alpn_protocols(["h2"])
    channel = grpclib.client.Channel(host=host, port=port, ssl=ctx)
    metadata = [("control-token", token)]
    return channel, metadata


async def _list_experiments_async():
    from zen_api.acquisition.v1beta import (
        ExperimentServiceStub,
        ExperimentServiceGetAvailableExperimentsRequest,
    )
    channel, metadata = _open_grpc_channel_sync()
    try:
        svc = ExperimentServiceStub(channel=channel, metadata=metadata)
        resp = await svc.get_available_experiments(
            ExperimentServiceGetAvailableExperimentsRequest()
        )
        return [e.name for e in resp.experiments]
    finally:
        channel.close()


async def _start_experiment_async(exp_name, output_name):
    from zen_api.acquisition.v1beta import (
        ExperimentServiceStub,
        ExperimentServiceLoadRequest,
        ExperimentServiceStartExperimentRequest,
    )
    channel, metadata = _open_grpc_channel_sync()
    try:
        svc = ExperimentServiceStub(channel=channel, metadata=metadata)
        loaded = await svc.load(
            ExperimentServiceLoadRequest(experiment_name=exp_name)
        )
        exp_id = loaded.experiment_id
        await svc.start_experiment(
            ExperimentServiceStartExperimentRequest(
                experiment_id=exp_id,
                output_name=output_name,
            )
        )
        return exp_id
    finally:
        channel.close()


async def _stop_experiment_async(exp_id):
    from zen_api.acquisition.v1beta import (
        ExperimentServiceStub,
        ExperimentServiceStopRequest,
    )
    channel, metadata = _open_grpc_channel_sync()
    try:
        svc = ExperimentServiceStub(channel=channel, metadata=metadata)
        await svc.stop(ExperimentServiceStopRequest(experiment_id=exp_id))
    finally:
        channel.close()


def _on_refresh_exps(event):
    _ensure_root_logger()
    try:
        names = asyncio.run(_list_experiments_async())
        if not names:
            w_exp_name_select.options = ['']
            w_exp_name_select.value = ''
            logging.warning("No experiments found on ZEN gateway.")
        else:
            w_exp_name_select.options = names
            if w_exp_name_select.value not in names:
                w_exp_name_select.value = names[0]
        logging.info(f"Found {len(names)} ZEN experiment(s): {names}")
    except Exception as e:
        logging.error(f"List experiments failed: {e}", exc_info=True)


def _on_start_exp(event):
    _ensure_root_logger()
    name = (w_exp_name_select.value or '').strip()
    out  = (w_exp_output_name.value or '').strip()
    if not name:
        logging.error("No experiment selected — click 'Refresh experiment list' first.")
        return
    if not out:
        out = f"track_{name}"
        logging.info(f"Output filename was empty, defaulting to '{out}'")
    try:
        exp_id = asyncio.run(_start_experiment_async(name, out))
        _state['experiment_id']   = exp_id
        _state['experiment_name'] = name
        exp_status_md.object = (
            f"**Running**: `{name}` → `{out}.czi`  \n"
            f"experiment_id: `{exp_id}`"
        )
        btn_start_exp.disabled = True
        btn_stop_exp.disabled  = False
        logging.info(
            f"Started ZEN experiment {name!r} → '{out}.czi'  id={exp_id}"
        )
    except Exception as e:
        logging.error(f"Start experiment failed: {e}", exc_info=True)


def _on_stop_exp(event):
    _ensure_root_logger()
    exp_id = _state.get('experiment_id')
    name   = _state.get('experiment_name') or '?'
    if not exp_id:
        logging.warning("No API-started experiment to stop.")
        return
    try:
        asyncio.run(_stop_experiment_async(exp_id))
        logging.info(f"Stopped ZEN experiment {name!r}  id={exp_id}")
    except Exception as e:
        logging.error(f"Stop experiment failed: {e}", exc_info=True)
    finally:
        _state['experiment_id']   = None
        _state['experiment_name'] = None
        exp_status_md.object = '_Stopped_'
        btn_start_exp.disabled = False
        btn_stop_exp.disabled  = True


btn_refresh_exps.on_click(_on_refresh_exps)
btn_start_exp.on_click(_on_start_exp)
btn_stop_exp.on_click(_on_stop_exp)


# ─── Ingest control ─────────────────────────────────────────────────────────

def _on_ingest_start(event):
    _ensure_root_logger()
    try:
        from tracking_tools.zen_ingest import ZenIngest

        source = (w_zen_source.value or '').strip()
        out_root = (w_dirpath.value or '').strip()
        if not source:
            logging.error("ZEN source folder is empty.")
            return
        if not out_root:
            logging.error("Experiment root is empty.")
            return

        position_names = _get_position_names()
        if not position_names:
            logging.error("Number of scenes must be at least 1.")
            return

        os.makedirs(out_root, exist_ok=True)
        for name in position_names:
            os.makedirs(os.path.join(out_root, name), exist_ok=True)

        ingest = ZenIngest(
            source_dir=source,
            out_root=out_root,
            position_names=position_names,
            n_z=w_n_z.value,
            n_channels=w_n_channels.value,
            poll_interval_s=w_ingest_poll.value,
        )
        ingest.start()
        _state['ingest'] = ingest
        btn_ingest_start.disabled = True
        btn_ingest_stop.disabled = False
    except Exception as e:
        logging.error(f"Could not start ingest: {e}", exc_info=True)


def _on_ingest_stop(event):
    ingest = _state.get('ingest')
    if ingest is not None:
        ingest.stop()
        _state['ingest'] = None
    btn_ingest_start.disabled = False
    btn_ingest_stop.disabled = True


btn_ingest_start.on_click(_on_ingest_start)
btn_ingest_stop.on_click(_on_ingest_stop)


# ─── Tracking run ───────────────────────────────────────────────────────────

def _run_tracking():
    """Executed in a daemon thread; drives the full tracking lifecycle."""
    _ensure_root_logger()

    try:
        config_path = os.path.join(_ROOT, 'tracking_tools', 'tracking_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        roi_tracker_config              = config['roi_tracker']
        roi_tracker_config['serverkit'] = w_serverkit.value
        position_tracker_config = {
            'pixel_size_xy': w_pixel_xy.value,
            'pixel_size_z':  w_pixel_z.value,
            # 2-D mode zeroes the Z shift inside PositionTracker.compute_shift_um
            'tracking_2d':   w_tracking_2d.value,
        }
        runner_config = config['tracking_runner']

        from tracking_tools.tracking_runner.TrackingRunner import TrackingRunner
        from tracking_tools.microscope_interface.MicroscopeInterface import (
            MicroscopeInterface_Files,
            MicroscopeInterface_Zeiss,
        )
        from tracking_tools.utils.tracking_utils import get_pos_config

        dirpath      = (w_dirpath.value or '').strip()
        log_dir_name = runner_config['log_dir_name']
        position_config = get_pos_config(dirpath, log_dir_name)

        if not position_config:
            logging.error(
                f"No trackable positions found in '{dirpath}'.\n"
                "Each position folder must contain "
                "embryo_tracking/tracking_RoIs.json.\n"
                "Define ROIs from the ROI Selection tab first."
            )
            return

        if w_use_streaming.value:
            # In streaming mode the tracking channel is read from each
            # position's tracking_RoIs.json filename suffix (the same way
            # the file-watcher derives it).  We pass channel=0 as a benign
            # default for the stream subscription itself.
            zeiss_params = {
                'address':         w_zen_address.value.strip(),
                'port':            w_zen_port.value,
                'cert_path':       w_zen_cert.value.strip(),
                'control_token':   w_zen_token.value,
                'z_projection':    w_z_proj.value,
                'tracking_channel': 0,
                'max_xy_um':       w_max_xy.value,
                'max_z_um':        w_max_z.value,
                'feedback_enabled': w_use_feedback.value,
            }
            microscope = MicroscopeInterface_Zeiss(
                positions_config=position_config,
                dirpath=dirpath,
                zeiss_params=zeiss_params,
            )
        else:
            # File-watching for INPUT.  When feedback is on, we also open a
            # gRPC channel to ZEN so the tracker-computed shifts are sent
            # back as StageService.move_to calls.
            file_params = {
                'poll_interval_s': w_ingest_poll.value,
                'zen_feedback':       w_use_feedback.value,
                'zen_address':        w_zen_address.value.strip(),
                'zen_port':           w_zen_port.value,
                'zen_cert_path':      w_zen_cert.value.strip(),
                'zen_control_token':  w_zen_token.value,
                'max_xy_um':          w_max_xy.value,
                'max_z_um':           w_max_z.value,
            }
            microscope = MicroscopeInterface_Files(
                positions_config=position_config,
                dirpath=dirpath,
                file_params=file_params,
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
    t = threading.Thread(target=_run_tracking, daemon=True, name='TrackingThread')
    _state['tracking_thread'] = t
    t.start()


def _on_stop(event):
    runner    = _state.get('runner')
    microscope = _state.get('microscope')
    if runner:
        runner.stop_requested = True
    if microscope:
        microscope.stop_requested = True
        try:
            microscope.stop()
        except Exception:
            pass
    btn_run.disabled  = False
    btn_stop.disabled = True


btn_run.on_click(_on_run)
btn_stop.on_click(_on_stop)


# ─── Periodic log refresh ───────────────────────────────────────────────────

_MAX_LOG_LINES = 300


def _update_log():
    lines = []
    try:
        while True:
            lines.append(_log_queue.get_nowait())
    except queue.Empty:
        pass
    if lines:
        current = (w_log.value or '').splitlines()
        all_lines = current + lines
        w_log.value = '\n'.join(all_lines[-_MAX_LOG_LINES:]) + '\n'


pn.state.add_periodic_callback(_update_log, period=500)


# ─── Layout ─────────────────────────────────────────────────────────────────

acquisition_section = pn.Column(
    pn.pane.Markdown('### Acquisition / experiment'),
    w_dirpath,
    pn.Row(w_n_scenes, w_n_channels),
    pn.Row(w_pixel_xy, w_pixel_z, w_n_z),
    w_tracking_2d,
    w_serverkit,
)

ingest_section = pn.Column(
    pn.pane.Markdown('### ZEN ingest (per-Z TIF → 3-D stacks)'),
    ingest_info,
    w_zen_source,
    pn.Row(w_ingest_poll, btn_ingest_start, btn_ingest_stop),
)

experiment_section = pn.Column(
    pn.pane.Markdown(
        '### ZEN experiment control (API-started)\n\n'
        '_Start the experiment from here instead of the ZEN UI button.  '
        'That way our tool owns the running `experiment_id` and can update '
        'stored scene positions per cycle via `TilesService.add_positions`.  '
        'Refresh the list, pick an experiment, type an output filename, '
        'then **Start experiment via ZEN API**.  Verify TIFs spit out in '
        'the source folder as expected before relying on this for tracking._'
    ),
    pn.Row(w_exp_name_select, btn_refresh_exps),
    w_exp_output_name,
    pn.Row(btn_start_exp, btn_stop_exp),
    exp_status_md,
)

advanced_section = pn.Card(
    pn.pane.Markdown(
        '_Streaming mode contacts the ZEN API Gateway directly via gRPC.  '
        'Stage feedback sends `relative_move` shifts back to the microscope.  '
        'Both default to **off** while we validate the file-watching pipeline._'
    ),
    w_use_streaming,
    w_use_feedback,
    pn.layout.Divider(),
    pn.pane.Markdown('**ZEN Gateway connection** (used only when streaming is on)'),
    pn.Row(w_zen_address, w_zen_port),
    w_zen_cert,
    w_zen_token,
    pn.Row(w_z_proj, w_max_xy, w_max_z),
    title='Advanced — ZEN streaming & stage feedback',
    collapsed=True,
)

tracking_tab = pn.Column(
    pn.pane.Markdown('## Live Tracking'),
    pn.layout.Divider(),
    acquisition_section,
    pn.layout.Divider(),
    experiment_section,
    pn.layout.Divider(),
    ingest_section,
    pn.layout.Divider(),
    advanced_section,
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

# Visualisation tab — reuse the existing Bokeh dashboard
try:
    from interactive_tools.bokeh_visualisation import make_layout as _vis_make_layout
    vis_tab = pn.panel(_vis_make_layout, sizing_mode='stretch_both')
except Exception as _e:
    vis_tab = pn.pane.Alert(
        f'Could not load tracking visualisation dashboard: {_e}',
        alert_type='warning',
    )

tabs = pn.Tabs(
    ('Tracking',       tracking_tab),
    ('ROI Selection',  roi_tab),
    ('Visualisation',  vis_tab),
    sizing_mode='stretch_both',
)

tabs.servable()
