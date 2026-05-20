import time
import os
import ssl
import math
import queue
import threading
import asyncio
import numpy as np
import tifffile
from ..logger.logger import init_logger

class MicroscopeInterface_LS1:
    def __init__(self, positions_config) :
        import pymcs

        self.positions_config = positions_config
        # Get the position names seperated from the settings
        self.position_names = [posSetting.rsplit("_", 1)[0] for posSetting in self.positions_config.keys()]
        # Make the position to PositionSettings and channel lookup table
        self.pos_to_PosSettings = {}
        self.pos_to_Channel = {}
        for pos_name in positions_config.keys() :
            position_settings_splitted = pos_name.rsplit("_", 1)
            self.pos_to_PosSettings[position_settings_splitted[0]] = pos_name
            self.pos_to_Channel[position_settings_splitted[0]] = self._channel_from_filename(
                positions_config[pos_name].get("filename", "")
            )

        self.PosSettings_to_pos = {v:k for k, v in self.pos_to_PosSettings.items()}
        self.microscope = pymcs.Microscope()
        self.connect()
        self.time_lapse_controller = pymcs.TimeLapseController(self.microscope)
        self.stage_xyz = pymcs.StageXYZ(self.microscope, "STAGE")
        self.logger = init_logger(self.__class__.__name__)
        self.stop_requested = False

    @staticmethod
    def _channel_from_filename(filename):
        """Extract the channel suffix from a saved Viventis-style filename.

        ``t0001_Channel 1.tif`` → ``'Channel 1'``.  Empty filenames return
        ``''`` so ``read_image`` can still build a path (and fail loudly).
        """
        if not filename:
            return ''
        return filename.replace(".tif", "").split("_", 1)[-1] if "_" in filename else ""

    def refresh_filename(self, position_name):
        """Re-derive the tracking channel for *position_name* after the ROI
        dashboard saved a new ``tracking_RoIs.json``.

        ``TrackingRunner.reinitialize_tracker`` has just merged the new
        config (including the new ``filename``) into ``positions_config``.
        Here we update the cached channel suffix so subsequent calls to
        ``read_image`` pull frames from the new channel.
        """
        cfg = self.positions_config.get(position_name, {})
        filename = cfg.get("filename", "")
        if not filename:
            return
        stripped = position_name.rsplit("_", 1)[0]
        new_channel = self._channel_from_filename(filename)
        old = self.pos_to_Channel.get(stripped)
        self.pos_to_Channel[stripped] = new_channel
        if old != new_channel:
            self.logger.info(
                f"[{position_name}] LS1 tracking channel: {old!r} → {new_channel!r}"
            )

    # Waits for a new image
    def wait_for_image(self, timeout_ms) :
        timeout = False
        while not self.stop_requested :
            
            if not timeout:
                self.logger.info(f"Waiting for the next timepoint and position")
            position_name, time_point, timeout = self.wait_for_pause(timeout_ms=timeout_ms)
            if timeout : 
                continue
            if position_name not in self.position_names :
                self.continue_from_pause()
                continue

            if self.stop_requested :
                return

            # Read image
            PosSetting = self.pos_to_PosSettings[position_name]
            channel = self.pos_to_Channel[position_name]

            image = self.read_image(PosSetting, channel, time_point)
            return image, time_point, PosSetting
        
    def read_image(self, PosSetting, channel, time_point) :

        # Get path
        image_path = os.path.join(self.positions_config[PosSetting]["images_dir"], f"t{time_point:04}_{channel}.tif")

        if not os.path.exists(image_path):
            self.logger.error(f"Missing image at{image_path}")
            return None
        try :
            image = tifffile.imread(str(image_path))
            self.logger.info(f"Read image {image_path}")
            self.logger.info(f"Image shape : {image.shape}")
            return image
        except Exception as e:
            self.logger.error(f'Cannot read {image_path}: {e}')
            return None


    def wait_for_pause(self, timeout_ms) :
        position_name, timepoint, timeout = self.time_lapse_controller.wait_for_pause(timeout_ms)
        return position_name, timepoint, timeout

    def pause_after_position(self) :
        self.time_lapse_controller.pause_after_position()

    def no_pause_after_position(self) :
        self.time_lapse_controller.no_pause_after_position()

    def continue_from_pause(self) :
        self.time_lapse_controller.continue_from_pause()

    def relative_move(self, position_name, shift_x, shift_y, shift_z) :
        # Get the position out of the posSetting name
        position_name = position_name.rsplit("_", 1)[0] # Name is PositionName_SettingsName
        try :
            pos = self.stage_xyz.position_get(position_name=position_name)
            if math.fabs(pos.position_x-shift_x)>25000:
                shift_x=0
            if math.fabs(pos.position_y+shift_y)>1500:
                shift_y=0
            if math.fabs(pos.position_z-shift_z)>1500:
                shift_z=0    
            self.stage_xyz.position_set(
                position_name=position_name,
                position_x=pos.position_x - shift_x,
                position_y=pos.position_y + shift_y,  # Y Axis is inverted for the microscope stage coordinates
                position_z=pos.position_z - shift_z,
            )
            
        except pymcs.MicroscopeException as e :
            self.logger.info(f"Error during stage move : {e}")


    def connect(self) :
        self.microscope.connect()

    def disconnect(self) :
        self.microscope.disconnect()

    def stop(self) :
        self.logger.info("Stop")
        self.stop_requested = True
        self.microscope.disconnect()


class SimulatedMicroscopeInterface_LS1 :
    def __init__(self, positions_config, starting_timepoint=0, max_timeout=8) :
        self.positions_config = positions_config
        # Get the position names seperated from the settings
        self.position_names = [posSetting.rsplit("_", 1)[0] for posSetting in self.positions_config.keys()]
        # Make the position to PositionSettings and channel lookup table
        self.pos_to_PosSettings = {}
        self.pos_to_Channel = {}
        for pos_name in positions_config.keys() :
            position_settings_splitted = pos_name.rsplit("_", 1)
            self.pos_to_PosSettings[position_settings_splitted[0]] = pos_name
            self.pos_to_Channel[position_settings_splitted[0]] = positions_config[pos_name]["filename"].replace(".tif","").split("_")[-1]
        
        self.nb_positions = len(self.position_names)
        self.current_position_index = 0
        self.timepoint = starting_timepoint
        self.timeout_count = 1
        # Send x timeouts before pausing
        self.max_timeout = max_timeout
        # Set default logger
        self.logger = init_logger(self.__class__.__name__)
        self.stop_requested = False


    # Waits for a new image
    def wait_for_image(self, timeout_ms) :
        timeout = False
        while not self.stop_requested :
            
            if not timeout:
                self.logger.info(f"Waiting for the next timepoint and position")
            position_name, time_point, timeout = self.wait_for_pause(timeout_ms=timeout_ms)
            if timeout : 
                continue
            if position_name not in self.position_names :
                self.continue_from_pause()
                continue

            if self.stop_requested :
                return

            # Read image
            PosSetting = self.pos_to_PosSettings[position_name]
            channel = self.pos_to_Channel[position_name]

            image = self.read_image(PosSetting, channel, time_point)
            return image, time_point, PosSetting
        
    def read_image(self, PosSetting, channel, time_point) :
        # Get path
        image_path = os.path.join(self.positions_config[PosSetting]["images_dir"], f"t{time_point:04}_{channel}.tif")

        if not os.path.exists(image_path):
            msg = f"Missing image at {image_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        try :
            image = tifffile.imread(str(image_path))
            self.logger.info(f"Read image {image_path}")
            self.logger.info(f"Image shape : {image.shape}")
            return image
        except Exception as e:
            msg = f"Cannot read {image_path}: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e


    # Simulates LS1 wait for pause function
    def wait_for_pause(self, timeout_ms) :
        time.sleep(timeout_ms/1000)
        # Send some timeouts before pausing
        self.timeout_count = (self.timeout_count + 1) % (self.max_timeout + 1)
        if self.timeout_count % self.max_timeout != 0 :
            self.logger.info("Sending timeout")
            return None, None, True
        # Go through positions in a round robin cycle
        current_pos = self.position_names[self.current_position_index]
        current_timepoint = self.timepoint
        # Update timepoint if after a full cycle
        if self.current_position_index == 0 :
            self.timepoint = self.timepoint + 1
        # Update position for the next call
        self.current_position_index = (self.current_position_index + 1) % self.nb_positions
        self.logger.info(f"Pausing for position [{current_pos}] at timepoint {current_timepoint}")
        return current_pos, current_timepoint, False


    def pause_after_position(self) :
        self.logger.info("Pause after position")
        return
    
    def no_pause_after_position(self) :
        self.logger.info("No pause after position")
        return
    
    def continue_from_pause(self) :
        self.logger.info("Continue from pause")
        return
    
    def relative_move(self, position_name, shift_x, shift_y, shift_z) :
        self.logger.info(f"Relative move :[{position_name}], x={shift_x}, y={shift_y}, z={shift_z}")
        return
    
    def connect(self) :
        self.logger.info("Connect")
        return
    
    def disconnect(self) :
        self.logger.info("Disconnect")
        return
    
    def stop(self) :
        self.logger.info("Stop")
        self.stop_requested = True

    
class SimulatedMicroscopeInterface_General :
    def __init__(self, positions_config, starting_timepoint=0, back_track=False) :
        self.positions_config = positions_config
        self.position_names = list(self.positions_config.keys())
        self.nb_positions = len(self.position_names)
        self.logger = init_logger(self.__class__.__name__)

        # Positions naming format

        self.nb_digits = {}
        self.suffixes = {}

        for pos_name, cfg in self.positions_config.items() :
            digits, suffix = self.detect_format(cfg["filename"])
            self.nb_digits[pos_name] = digits
            self.suffixes[pos_name] = suffix

            self.logger.info(f"Position [{pos_name}] format: digits={digits}, suffix='{suffix}'")

        # self.nb_digits = self.detect_format(self.positions_config[next(iter(self.positions_config))]["filename"])
        self.timepoint = starting_timepoint
        self.current_position_index = 0
        self.back_track = back_track

    def wait_for_image(self, timeout_ms=100) :
        time.sleep(timeout_ms/1000)
        position_name, timepoint = self.get_pos_timepoint()
        image = self.read_image(position_name, timepoint)
        return image, timepoint, position_name


    def read_image(self, position_name, timepoint) :
        digits = self.nb_digits[position_name]
        suffix = self.suffixes[position_name]
        
        filename = f"t{timepoint:0{digits}d}{suffix}.tif"
        image_dir = self.positions_config[position_name]["images_dir"]
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            msg = f"Missing image at {image_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        try :
            image = tifffile.imread(str(image_path))
            self.logger.info(f"Read image {image_path}")
            self.logger.info(f"Image shape : {image.shape}")
            return image
        except Exception as e:
            msg = f"Cannot read {image_path}: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

    def get_pos_timepoint(self) :
        # Go through positions in a round robin cycle
        current_pos = self.position_names[self.current_position_index]
        current_timepoint = self.timepoint
        # Update timepoint if after a full cycle
        if self.current_position_index == 0 :
            if self.back_track :
                self.timepoint = max(0, self.timepoint - 1)
            else :
                self.timepoint = self.timepoint + 1
        # Update position for the next call
        self.current_position_index = (self.current_position_index + 1) % self.nb_positions
        self.logger.info(f"Position [{current_pos}] at timepoint {current_timepoint}")
        return current_pos, current_timepoint


    def detect_format(self, filename) :
        import re
        match = re.match(r"t(\d+)(.*)\.tif$", filename)
        if not match:
            self.logger.info(f"Could not match a filename with format t(\\d+)(.*)\\.tif$, {filename}")
        digits = match.group(1)
        suffix = match.group(2)
        return len(digits), suffix
    
    def relative_move(self, position_name, shift_x, shift_y, shift_z) :
        self.logger.info(f"Relative move :[{position_name}], x={shift_x}, y={shift_y}, z={shift_z}")
        return
    



# ---------------------------------------------------------------------------
# Zeiss / ZEN API interfaces
# ---------------------------------------------------------------------------
 
class SimulatedMicroscopeInterface_Zeiss:
    """
    Offline simulation that reads a CZI file and mimics real ZEN acquisition.
 
    For each timepoint it iterates over all scenes (positions), computes a 2-D
    max-projection (or central slice) of the z-stack, saves the result as a
    TIF file under ``{dirpath}/{position_name}/max_proj/t{t:04d}.tif``, and
    enqueues the numpy array so ``wait_for_image`` can return it.
 
    An ``inter_timepoint_delay_ms`` pause is inserted between timepoints to
    let the tracking loop keep pace with a realistic acquisition cadence.
 
    Parameters
    ----------
    positions_config : dict
        Output of ``get_pos_config()`` – keys are position folder names.
    dirpath : str
        Experiment root directory.
    czi_path : str
        Path to the CZI file to replay.
    inter_timepoint_delay_ms : int
        Simulated wait between consecutive timepoints.
    z_projection : str
        ``'max'`` for max-intensity projection, ``'central_slice'`` to pick
        the middle z-plane.
    starting_timepoint : int
        First timepoint index to replay (inclusive).
    tracking_channel : int
        Zero-based channel index to extract from the CZI file.
    """
 
    def __init__(self, positions_config, dirpath, czi_path,
                 inter_timepoint_delay_ms=2000, z_projection='max',
                 starting_timepoint=0, tracking_channel=0):
        self.positions_config = positions_config
        self.position_names   = sorted(positions_config.keys())
        self.dirpath          = dirpath
        self.czi_path         = czi_path
        self.inter_timepoint_delay_ms = inter_timepoint_delay_ms
        self.z_projection     = z_projection
        self.starting_timepoint = starting_timepoint
        self.tracking_channel = tracking_channel
 
        self._queue      = queue.Queue()
        self._stop_event = threading.Event()
        self._thread     = None
        self.stop_requested = False
        self.logger = init_logger(self.__class__.__name__)
 
        # Alphabetical sort maps scene index 0 → first position folder, etc.
        self.scene_to_position = {i: name for i, name in enumerate(self.position_names)}
 
    # ------------------------------------------------------------------
    def connect(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._produce_frames, daemon=True, name='ZeissSimThread'
        )
        self._thread.start()
        self.logger.info(f"Simulated Zeiss interface started from {self.czi_path}")
 
    # ------------------------------------------------------------------
    def _produce_frames(self):
        try:
            from pylibCZIrw import czi as pyczi
        except ImportError:
            self.logger.error(
                "pylibCZIrw not installed. Run: pip install pylibCZIrw"
            )
            self._queue.put((None, None, None))
            return
 
        if not os.path.exists(self.czi_path):
            self.logger.error(f"CZI file not found: {self.czi_path}")
            self._queue.put((None, None, None))
            return
 
        try:
            with pyczi.open_czi(self.czi_path) as czidoc:
                bb = czidoc.total_bounding_box
                self.logger.info(f"CZI dimensions: {bb}")
 
                # Values are (start, size) tuples
                n_t = bb.get('T', (0, 1))[1]
                n_s = bb.get('S', (0, 1))[1]
                n_z = bb.get('Z', (0, 1))[1]
                n_positions = min(n_s, len(self.position_names))
                self.logger.info(
                    f"Replaying {n_t} timepoints, {n_positions} scenes, {n_z} z-slices"
                )
 
                for t in range(self.starting_timepoint, n_t):
                    if self._stop_event.is_set():
                        break
 
                    for s in range(n_positions):
                        if self._stop_event.is_set():
                            break
 
                        z_slices = []
                        for z in range(n_z):
                            raw = czidoc.read(
                                plane={'T': t, 'Z': z,
                                       'C': self.tracking_channel, 'S': s}
                            )
                            arr = np.asarray(raw)
                            # pylibCZIrw may return (H, W, 1) — squeeze to (H, W)
                            if arr.ndim == 3:
                                arr = arr[:, :, 0]
                            z_slices.append(arr)
 
                        if not z_slices:
                            continue
 
                        z_stack = np.stack(z_slices)
                        image = (np.max(z_stack, axis=0)
                                 if self.z_projection == 'max'
                                 else z_stack[len(z_stack) // 2])
 
                        position_name = self.scene_to_position.get(s)
                        if position_name is None:
                            continue
 
                        max_proj_dir = os.path.join(
                            self.dirpath, position_name, 'max_proj'
                        )
                        os.makedirs(max_proj_dir, exist_ok=True)
                        tif_path = os.path.join(max_proj_dir, f't{t:04d}.tif')
                        tifffile.imwrite(tif_path, image)
                        self.logger.info(f"[SIM] Saved {tif_path}")
 
                        self._queue.put((image, t, position_name))
 
                        # Small inter-position pause to avoid flooding the queue
                        self._stop_event.wait(timeout=0.05)
 
                    if not self._stop_event.is_set() and t < n_t - 1:
                        self.logger.info(
                            f"[SIM] Timepoint {t} complete — "
                            f"waiting {self.inter_timepoint_delay_ms} ms"
                        )
                        self._stop_event.wait(
                            timeout=self.inter_timepoint_delay_ms / 1000
                        )
 
        except Exception as e:
            self.logger.error(f"Error reading CZI file: {e}", exc_info=True)
        finally:
            # Signal end-of-simulation so run_zeiss() can exit cleanly
            self.stop_requested = True
            self._queue.put((None, None, None))

    # ------------------------------------------------------------------
    def wait_for_image(self, timeout_ms=1000):
        try:
            return self._queue.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None, None, None
 
    def relative_move(self, position_name, shift_x, shift_y, shift_z):
        self.logger.info(
            f"[SIM] Move [{position_name}]: "
            f"dx={shift_x:.2f} dy={shift_y:.2f} dz={shift_z:.2f} µm"
        )
 
    def pause_after_position(self):    pass
    def no_pause_after_position(self): pass
    def continue_from_pause(self):     pass
 
    def wait_for_pause(self, timeout_ms=1000):
        return self.wait_for_image(timeout_ms)
 
    def disconnect(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
 
    def stop(self):
        self.stop_requested = True
        self.disconnect()
 
 
# ---------------------------------------------------------------------------
 
class MicroscopeInterface_Zeiss:
    """
    Live interface to a Zeiss microscope via the ZEN API Gateway (gRPC).
 
    A background asyncio loop handles all network I/O:
      * ``ExperimentStreamingServiceStub`` streams frames (one per z-slice) in
        real time.  Z-slices for the same (scene, timepoint) are assembled in a
        buffer; when the stack is complete a max-projection (or central slice)
        is computed, saved as TIF, and enqueued.
      * Stage XY and focus (Z) moves are submitted to that same asyncio loop
        via ``asyncio.run_coroutine_threadsafe``, so ``relative_move`` blocks
        the caller until the move completes (safe to call from the tracking
        thread).
 
    Positions are mapped to ZEN scenes by alphabetical order of the position
    folder names: scene 0 → first folder, scene 1 → second folder, etc.
 
    Parameters
    ----------
    positions_config : dict
        Output of ``get_pos_config()``.
    dirpath : str
        Experiment root directory.
    zeiss_params : dict
        Keys: ``address``, ``port``, ``cert_path``, ``control_token``,
        ``experiment_name``, ``z_projection`` (``'max'``|``'central_slice'``),
        ``tracking_channel`` (int), ``max_xy_um`` (float), ``max_z_um`` (float).
 
    Notes
    -----
    The exact gRPC service / message names depend on the installed
    ``zen_api`` package version.  Verify against the examples shipped with
    ZEN API (OAD/ZEN-API/python_examples/).
    """
 
    def __init__(self, positions_config, dirpath, zeiss_params):
        self.positions_config = positions_config
        self.position_names   = sorted(positions_config.keys())
        self.dirpath          = dirpath
 
        self.address          = zeiss_params.get('address',          'localhost')
        self.port             = zeiss_params.get('port',             5002)
        self.cert_path        = zeiss_params.get('cert_path',        '')
        self.control_token    = zeiss_params.get('control_token',    '')
        self.experiment_name  = zeiss_params.get('experiment_name',  '')
        self.z_projection     = zeiss_params.get('z_projection',     'max')
        self.tracking_channel = zeiss_params.get('tracking_channel', 0)
        self.max_xy_um        = float(zeiss_params.get('max_xy_um',  500.0))
        self.max_z_um         = float(zeiss_params.get('max_z_um',   100.0))
        # Optional: directory where ZEN saves the CZI file.  When set the
        # interface polls the CZI file for new timepoints instead of using the
        # gRPC streaming API (required for ZEN API Gateway < Autumn 2025).
        self.czi_watch_dir    = zeiss_params.get('czi_watch_dir',    '')
        self.czi_poll_interval_s = float(zeiss_params.get('czi_poll_interval_s', 5.0))
        # Optional: directory where ZEN saves individual TIF files (one per
        # Z-slice / channel / timepoint).  Each TIF has a name like:
        #   <exp>_H(0)_S0000(P4)_T000000_Z0000_C00_M0000_ORG.tif
        # When set, watch the folder, group by (S, T) for the tracking
        # channel, build a max projection once a stack is complete and
        # enqueue it.  Most reliable mode for older ZEN API Gateway versions.
        self.tif_watch_dir    = zeiss_params.get('tif_watch_dir',    '')
        self.tif_poll_interval_s = float(zeiss_params.get('tif_poll_interval_s', 2.0))

        self._queue      = queue.Queue()
        self._stop_event = threading.Event()
        self._loop       = None
        self._thread     = None
        self._channel    = None   # gRPC channel, set inside async loop
        self._metadata   = None   # gRPC call metadata
        # Expected z-slices per stack; queried from ZEN experiment at connect()
        self._z_slice_count = 1
        self.stop_requested = False
 
        self.scene_to_position = {i: n for i, n in enumerate(self.position_names)}
        self.position_to_scene = {n: i for i, n in enumerate(self.position_names)}
 
        self.logger = init_logger(self.__class__.__name__)
 
    # ------------------------------------------------------------------
    def connect(self):
        self._stop_event.clear()
        if self.tif_watch_dir:
            self.logger.info(
                f"TIF folder-poll mode: watching {self.tif_watch_dir}"
            )
            self._thread = threading.Thread(
                target=self._poll_tif_thread,
                daemon=True,
                name='ZeissTifPollThread',
            )
        elif self.czi_watch_dir:
            self.logger.info(
                f"CZI file-poll mode: watching {self.czi_watch_dir} for new CZI files"
            )
            self._thread = threading.Thread(
                target=self._poll_czi_thread,
                daemon=True,
                name='ZeissCziPollThread',
            )
        else:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=lambda: self._loop.run_until_complete(self._async_main()),
                daemon=True,
                name='ZeissAsyncThread',
            )
            self.logger.info(
                f"Streaming mode: connecting to ZEN API Gateway at {self.address}:{self.port}"
            )
        self._thread.start()
 
    # ------------------------------------------------------------------
    async def _async_main(self):
        # The zen_api wheel uses betterproto + grpclib (NOT grpcio / grpc.aio).
        # betterproto stubs call channel.request() which is a grpclib method;
        # passing a grpc.aio.Channel here produces the 'has no attribute request'
        # AttributeError.  Use grpclib.client.Channel instead.
        import ssl
        try:
            import grpclib.client
        except ImportError:
            self.logger.error(
                "grpclib not installed — required by the zen_api wheel. "
                "Run: pip install grpclib"
            )
            self._queue.put((None, None, None))
            return

        # Replicate initialize_zenapi from zen_api_utils/misc.py exactly.
        # ssl.SSLContext(PROTOCOL_TLS_CLIENT) + set_alpn_protocols(["h2"]) is
        # required: without the h2 ALPN hint the TLS handshake does not
        # negotiate HTTP/2 and grpclib terminates the stream immediately.
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if self.cert_path and os.path.exists(self.cert_path):
            ssl_ctx.load_verify_locations(cafile=self.cert_path)
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            ssl_ctx.check_hostname = True
        else:
            self.logger.warning(
                "No cert_path provided — TLS certificate verification disabled. "
                "Set cert_path to the ZEN Gateway CA certificate."
            )
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
        # h2 ALPN is mandatory for gRPC-over-HTTP/2
        ssl_ctx.set_alpn_protocols(["h2"])

        # ZEN closes the gRPC stream between timepoints (when no acquisition is
        # happening).  We reconnect automatically so that we catch the next
        # available frame regardless of when the connection is opened relative
        # to the experiment schedule.
        retry_delay_s = 5

        while not self._stop_event.is_set():
            channel = grpclib.client.Channel(
                host=self.address, port=int(self.port), ssl=ssl_ctx
            )
            self._channel  = channel
            self._metadata = [("control-token", self.control_token)]
            try:
                self.logger.info(
                    f"Connecting to ZEN API Gateway at {self.address}:{self.port}"
                )
                await self._stream_images()
                # _stream_images returned cleanly (stop_event set inside loop)
                break
            except asyncio.CancelledError:
                # Event loop is shutting down — exit without logging an error
                break
            except Exception as e:
                if self._stop_event.is_set():
                    break
                # UNIMPLEMENTED (gRPC status 12): the ZEN API Gateway on this
                # machine does not support ExperimentStreamingService.
                # MonitorAllExperiments was added in ZEN 3.13 Autumn 2025.
                # Retrying would loop forever — give up immediately.
                try:
                    from grpclib.exceptions import GRPCError
                    from grpclib.const import Status as GrpcStatus
                    if isinstance(e, GRPCError) and e.status == GrpcStatus.UNIMPLEMENTED:
                        self.logger.error(
                            "ZEN API Gateway returned UNIMPLEMENTED for "
                            "ExperimentStreamingService/MonitorAllExperiments.\n"
                            "This RPC requires ZEN 3.13 Autumn 2025 or later.\n"
                            "Options:\n"
                            "  1. Update your ZEN API Gateway to the Autumn 2025 release.\n"
                            "  2. Use file-polling mode: set 'czi_watch_dir' in zeiss_params\n"
                            "     to the folder where ZEN saves its CZI output.\n"
                            f"  Raw error: {e}"
                        )
                        self._queue.put((None, None, None))
                        return
                except ImportError:
                    pass
                self.logger.info(
                    f"ZEN stream closed ({type(e).__name__}: {e}). "
                    f"ZEN may be between timepoints — reconnecting in {retry_delay_s}s"
                )
                try:
                    await asyncio.sleep(retry_delay_s)
                except asyncio.CancelledError:
                    break
            finally:
                channel.close()
                self._channel = None
 
    # ------------------------------------------------------------------
    async def _stream_images(self):
        """
        Subscribe to the ZEN experiment stream and assemble z-stacks.

        Class and method names match the zen_api wheel shipped with ZEN 3.x:
            zen_api.acquisition.v1beta
        Cross-check with OAD/ZEN-API/python_examples/zenapi_streaming.py.
        """
        try:
            from zen_api.acquisition.v1beta import (
                ExperimentStreamingServiceStub,
                ExperimentStreamingServiceMonitorAllExperimentsRequest,
            )
        except ImportError as e:
            self.logger.error(
                f"zen_api package not found or wrong import path: {e}\n"
                "Install the zen_api wheel from the ZEN API distribution."
            )
            self._queue.put((None, None, None))
            return

        # Metadata (control token) is passed to the stub constructor, not each RPC call
        stub = ExperimentStreamingServiceStub(
            channel=self._channel, metadata=self._metadata
        )
        # Buffer: {(scene_idx, timepoint_idx): {z_idx: 2-D array}}
        frame_buffer = {}

        # Do not filter by channel_index in the request — some ZEN versions
        # close the stream immediately when that parameter is set.  We filter
        # per-frame in Python using fp.c instead.
        self.logger.info("ZEN image stream active — waiting for experiment frames")
        async for response in stub.monitor_all_experiments(
            ExperimentStreamingServiceMonitorAllExperimentsRequest(
                enable_raw_data=False,
            )
        ):
            if self._stop_event.is_set():
                break

            fp        = response.frame_data.frame_position
            scene_idx = fp.s
            tp_idx    = fp.t
            z_idx     = fp.z

            if fp.c != self.tracking_channel:
                continue

            key = (scene_idx, tp_idx)
            if key not in frame_buffer:
                frame_buffer[key] = {}

            full_size = response.frame_data.frame_size
            arr = np.frombuffer(
                response.frame_data.pixel_data.raw_data, dtype=np.uint16
            ).reshape((full_size.height, full_size.width))
            frame_buffer[key][z_idx] = arr

            if len(frame_buffer[key]) >= self._z_slice_count:
                self._process_complete_stack(frame_buffer.pop(key), scene_idx, tp_idx)
                # Drop stale keys to prevent unbounded memory growth
                stale = [(s, t) for (s, t) in list(frame_buffer) if t < tp_idx - 2]
                for k in stale:
                    frame_buffer.pop(k, None)
 
    # ------------------------------------------------------------------
    def _poll_czi_thread(self):
        """
        Fallback for ZEN API Gateway versions that do not implement the
        ExperimentStreamingService (pre-Autumn 2025 builds).

        Watches ``self.czi_watch_dir`` for a CZI file written by ZEN, then
        polls it every ``self.czi_poll_interval_s`` seconds for new complete
        timepoints and enqueues them exactly as the streaming path would.

        A timepoint T is considered complete once T+1 appears in the file
        (i.e. we process up to n_t-2 on each poll).  This guarantees we
        never read a partially-written plane while still keeping the lag at
        most one inter-timepoint gap.
        """
        import glob as glob_module
        try:
            from pylibCZIrw import czi as pyczi
        except ImportError:
            self.logger.error(
                "pylibCZIrw is required for CZI file-poll mode. "
                "Install with:  pip install pylibCZIrw"
            )
            self._queue.put((None, None, None))
            return

        czi_path = None
        last_t   = {}   # scene_idx → last enqueued timepoint index

        while not self._stop_event.is_set():
            # Discover the CZI file. The user-supplied path may be either:
            #   - a direct CZI file path (use it as-is), or
            #   - a directory (pick the most recently modified *.czi inside).
            if czi_path is None or not os.path.exists(czi_path):
                if (os.path.isfile(self.czi_watch_dir)
                        and self.czi_watch_dir.lower().endswith('.czi')):
                    czi_path = self.czi_watch_dir
                    self.logger.info(f"[FILE-POLL] Using CZI: {czi_path}")
                elif os.path.isdir(self.czi_watch_dir):
                    candidates = glob_module.glob(
                        os.path.join(self.czi_watch_dir, "*.czi")
                    )
                    if candidates:
                        czi_path = max(candidates, key=os.path.getmtime)
                        self.logger.info(f"[FILE-POLL] Found CZI: {czi_path}")
                    else:
                        self.logger.info(
                            f"[FILE-POLL] No CZI file yet in {self.czi_watch_dir} — waiting"
                        )
                        self._stop_event.wait(timeout=self.czi_poll_interval_s)
                        continue
                else:
                    self.logger.info(
                        f"[FILE-POLL] Path does not exist yet: {self.czi_watch_dir} — waiting"
                    )
                    self._stop_event.wait(timeout=self.czi_poll_interval_s)
                    continue

            try:
                with pyczi.open_czi(czi_path) as czidoc:
                    bb = czidoc.total_bounding_box
                    n_t = bb.get('T', (0, 1))[1]
                    n_s = bb.get('S', (0, 1))[1]
                    n_z = bb.get('Z', (0, 1))[1]
                    n_positions = min(n_s, len(self.position_names))

                    # Process all timepoints that are guaranteed complete
                    # (everything except the last one still being written)
                    safe_max_t = max(0, n_t - 1)

                    for s in range(n_positions):
                        start_t = last_t.get(s, -1) + 1
                        for t in range(start_t, safe_max_t):
                            z_slices = []
                            for z in range(n_z):
                                raw = czidoc.read(
                                    plane={'T': t, 'Z': z,
                                           'C': self.tracking_channel, 'S': s}
                                )
                                arr = np.asarray(raw)
                                if arr.ndim == 3:
                                    arr = arr[:, :, 0]
                                z_slices.append(arr)

                            if not z_slices:
                                continue

                            self._process_complete_stack(
                                {z: z_slices[z] for z in range(len(z_slices))}, s, t
                            )
                            last_t[s] = t

            except Exception as e:
                self.logger.debug(f"[FILE-POLL] Error reading CZI: {e}")

            self._stop_event.wait(timeout=self.czi_poll_interval_s)

        self.logger.info("[FILE-POLL] File-poll thread stopped")

    # ------------------------------------------------------------------
    def _poll_tif_thread(self):
        """
        Watch a directory where ZEN saves individual TIFs for every plane.

        File pattern (one TIF per Z-slice per channel per timepoint):
            <name>_H(0)_S<scene>(P<position>)_T<tp>_Z<z>_C<ch>_M<m>_ORG.tif

        For each (scene, timepoint) on the configured tracking channel we
        collect every Z-slice that has been written to disk, then build the
        max projection (or central slice) and enqueue it.

        A timepoint T is treated as 'complete' once T+1 starts appearing on
        disk — the strict guard avoids reading half-written stacks.  When
        the experiment ends, the user can press the reload button on the
        dashboard to flush the very last timepoint.
        """
        import re
        import glob as glob_module

        tif_pattern = re.compile(
            r'_S(?P<S>\d+)\(P\d+\)_T(?P<T>\d+)_Z(?P<Z>\d+)_C(?P<C>\d+)_'
            r'M\d+_ORG\.tif$',
            re.IGNORECASE,
        )

        last_t = {}                  # scene_idx → last enqueued timepoint
        # (scene, tp) → {z: filepath} accumulator for the tracking channel
        stacks = {}

        while not self._stop_event.is_set():
            if not os.path.isdir(self.tif_watch_dir):
                self.logger.info(
                    f"[TIF-POLL] Directory not present yet: {self.tif_watch_dir} — waiting"
                )
                self._stop_event.wait(timeout=self.tif_poll_interval_s)
                continue

            # Scan all TIFs and group by (scene, tp) for the tracking channel
            tifs = glob_module.glob(os.path.join(self.tif_watch_dir, "*.tif"))
            highest_t_per_scene = {}
            for path in tifs:
                m = tif_pattern.search(os.path.basename(path))
                if not m:
                    continue
                s = int(m.group('S'))
                t = int(m.group('T'))
                z = int(m.group('Z'))
                c = int(m.group('C'))
                if c != self.tracking_channel:
                    continue
                highest_t_per_scene[s] = max(highest_t_per_scene.get(s, -1), t)
                stacks.setdefault((s, t), {})[z] = path

            # Process every (scene, tp) for which a strictly newer tp exists
            # on disk — guarantees the stack is complete.
            for (s, t), z_files in sorted(stacks.items()):
                if t <= last_t.get(s, -1):
                    continue
                if t >= highest_t_per_scene.get(s, -1):
                    continue
                try:
                    z_slices = {z: tifffile.imread(p) for z, p in z_files.items()}
                except Exception as e:
                    self.logger.debug(f"[TIF-POLL] Read error for S{s} T{t}: {e}")
                    continue
                self._process_complete_stack(z_slices, s, t)
                last_t[s] = t

            # Garbage-collect entries we've already processed
            stacks = {k: v for k, v in stacks.items()
                      if k[1] > last_t.get(k[0], -1)}

            self._stop_event.wait(timeout=self.tif_poll_interval_s)

        self.logger.info("[TIF-POLL] TIF-poll thread stopped")

    # ------------------------------------------------------------------
    def _process_complete_stack(self, z_slices_dict, scene_idx, tp_idx):
        z_stack = np.stack(
            [z_slices_dict[z] for z in sorted(z_slices_dict)]
        )
        image = (np.max(z_stack, axis=0)
                 if self.z_projection == 'max'
                 else z_stack[len(z_stack) // 2])
 
        position_name = self.scene_to_position.get(scene_idx)
        if position_name is None:
            self.logger.warning(f"No position mapped to scene index {scene_idx}")
            return
 
        # Save alongside raw acquisitions in the position root folder,
        # consistent with how LS1 stores images.  TrackingRunner writes its
        # own copy to embryo_tracking/max_proj/ after the tracker runs.
        pos_dir = os.path.join(self.dirpath, position_name)
        os.makedirs(pos_dir, exist_ok=True)
        tif_path = os.path.join(pos_dir, f't{tp_idx:04d}.tif')
        tifffile.imwrite(tif_path, image)
        self.logger.info(f"Saved {tif_path}")
        self._queue.put((image, tp_idx, position_name))
 
    # ------------------------------------------------------------------
    def wait_for_image(self, timeout_ms=1000):
        try:
            return self._queue.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None, None, None
 
    # ------------------------------------------------------------------
    def relative_move(self, position_name, shift_x, shift_y, shift_z):
        if self._loop is None or self._stop_event.is_set():
            return
 
        # Soft clamp — log and clamp rather than silently skip
        def _clamp(v, limit, axis):
            if abs(v) > limit:
                self.logger.warning(
                    f"{axis} shift {v:.1f} µm exceeds limit {limit:.0f} µm — clamped"
                )
                return math.copysign(limit, v)
            return v
 
        shift_x = _clamp(shift_x, self.max_xy_um, 'X')
        shift_y = _clamp(shift_y, self.max_xy_um, 'Y')
        shift_z = _clamp(shift_z, self.max_z_um,  'Z')
 
        future = asyncio.run_coroutine_threadsafe(
            self._async_relative_move(shift_x, shift_y, shift_z), self._loop
        )
        try:
            future.result(timeout=10)
        except Exception as e:
            self.logger.error(f"Stage move failed: {e}")
 
    # ------------------------------------------------------------------
    async def _async_relative_move(self, dx_um, dy_um, dz_um):
        """
        Query current stage/focus position then apply a relative offset.
 
        Import paths and field names (``x``, ``y``, ``position``) should be
        verified against the installed zen_api package.
        Cross-check with:
            OAD/ZEN-API/python_examples/zenapi_stage_LM.py
            OAD/ZEN-API/zen_api_utils/stage.py
        """
        try:
            from zen_api.lm.hardware.v2 import (
                StageServiceStub,
                StageServiceGetPositionRequest,
                StageServiceMoveToRequest,
            )
            from zen_api.focus.v2 import (
                FocusServiceStub,
                FocusServiceGetPositionRequest,
                FocusServiceMoveToPositionRequest,
            )
        except ImportError as e:
            self.logger.error(f"zen_api hardware stubs not found: {e}")
            return
 
        # zen_api stubs are generated by betterproto: metadata in constructor,
        # method names in snake_case, no per-call metadata argument.
        stage_stub = StageServiceStub(channel=self._channel, metadata=self._metadata)
        focus_stub = FocusServiceStub(channel=self._channel, metadata=self._metadata)

        if abs(dx_um) > 1e-3 or abs(dy_um) > 1e-3:
            pos = await stage_stub.get_position(StageServiceGetPositionRequest())
            await stage_stub.move_to(
                StageServiceMoveToRequest(
                    x=pos.x + dx_um * 1e-6,
                    y=pos.y + dy_um * 1e-6,
                )
            )

        if abs(dz_um) > 1e-3:
            z_pos = await focus_stub.get_position(FocusServiceGetPositionRequest())
            # Field name for Z may be .z or .position depending on proto version
            current_z = getattr(z_pos, 'z', None) or getattr(z_pos, 'position', 0.0)
            await focus_stub.move_to_position(
                FocusServiceMoveToPositionRequest(
                    position=current_z + dz_um * 1e-6
                )
            )
 
    # ------------------------------------------------------------------
    def pause_after_position(self):    pass
    def no_pause_after_position(self): pass
    def continue_from_pause(self):     pass
 
    def wait_for_pause(self, timeout_ms=1000):
        return self.wait_for_image(timeout_ms)
 
    # ------------------------------------------------------------------
    def disconnect(self):
        self._stop_event.set()
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
 
    def stop(self):
        self.stop_requested = True
        self.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# Pure-file microscope interface (no hardware feedback).  Watches per-position
# folders for new TIFs whose names follow each position's tracking_RoIs.json
# `filename` field (with the t{NNNN} digits incrementing).  Used for offline
# tracking and for online tracking when frames are written to disk by the
# acquisition software (Viventis LS1, ZEN with TIF export, etc.).
# Compatible with run_zeiss() in TrackingRunner since it uses the same queue
# and reinit-watcher mechanism.
# ─────────────────────────────────────────────────────────────────────────────
class MicroscopeInterface_Files:
    """
    File-watching microscope interface.

    For every position in ``positions_config`` the watcher monitors the
    ``images_dir`` for new files matching the pattern
    ``<prefix>{tp:04d}<suffix>.tif`` where the prefix and suffix are derived
    from the ``filename`` saved in that position's ``tracking_RoIs.json``.
    Channel changes mid-experiment are handled because the saved filename
    encodes the channel — when ROI is re-saved on a different channel, the
    runner reinitialises the tracker and calls ``refresh_filename`` on us.

    ``relative_move`` is a no-op stub — no microscope feedback is sent.  Use
    a hardware interface (e.g. ``MicroscopeInterface_Zeiss``) once feedback
    is enabled.
    """

    def __init__(self, positions_config, dirpath, file_params=None):
        import re
        self._re = re

        self.positions_config = positions_config
        self.dirpath = dirpath
        # Position names == positions_config keys (matches what TrackingRunner
        # uses as keys for trackers / state).
        self.position_names = list(positions_config.keys())

        file_params = file_params or {}
        self.poll_interval_s = float(file_params.get('poll_interval_s', 1.0))

        # ZEN gRPC feedback (optional).  When ``zen_feedback`` is True we
        # open a gRPC channel to the ZEN Gateway just so we can call
        # StageService.move_to / FocusService.move_to_position with each
        # tracker-computed shift.  Frames keep flowing from disk via the
        # poll loop — only the OUTPUT side talks to ZEN.
        self.zen_feedback     = bool(file_params.get('zen_feedback', False))
        self.zen_address      = file_params.get('zen_address', 'localhost')
        self.zen_port         = int(file_params.get('zen_port', 5002))
        self.zen_cert_path    = file_params.get('zen_cert_path', '')
        self.zen_control_token = file_params.get('zen_control_token', '')
        self.max_xy_um        = float(file_params.get('max_xy_um', 500.0))
        self.max_z_um         = float(file_params.get('max_z_um', 100.0))

        # asyncio loop + gRPC channel for stage commands.  Lazily created
        # in connect() when feedback is enabled.
        self._zen_loop    = None
        self._zen_thread  = None
        self._zen_channel = None
        self._zen_metadata = None

        # Per-position state
        self._patterns = {}     # pos_name → (prefix, suffix)
        self._next_tp  = {}     # pos_name → next timepoint we expect to see
        self._json_mtime = {}   # pos_name → last seen mtime of tracking_RoIs.json
        for pos_name in self.position_names:
            self._refresh_filename_unlocked(pos_name)

        self._queue      = queue.Queue()
        self._stop_event = threading.Event()
        self._thread     = None
        self.stop_requested = False
        self.logger = init_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _parse_pattern(self, filename):
        """
        Split filename around the t{NNNN} number.

        Returns ``(prefix, suffix, start_tp)`` where ``start_tp`` is the
        integer value of the digits found in the filename — used so the
        watcher begins reading at exactly the timepoint at which the ROI
        was defined.  Crucial after a mid-experiment channel switch: the
        new tracker must start from the timepoint of the new ROI, not from
        T=0 (the ROI may not be valid at earlier frames).

        Examples:
          't0001_C00.tif'        → ('t', '_C00.tif',  1)
          't0001_Channel 1.tif'  → ('t', '_Channel 1.tif', 1)   # Viventis
          't0005.tif'            → ('t', '.tif',      5)        # LS1
        """
        m = self._re.match(r'^(.*?)t(\d+)(.*\.tif)$', filename, self._re.IGNORECASE)
        if not m:
            return ('t', '.tif', 0)
        return (m.group(1) + 't', m.group(3), int(m.group(2)))

    def _refresh_filename_unlocked(self, pos_name):
        """Re-parse the filename pattern and reset the next-timepoint cursor."""
        cfg = self.positions_config.get(pos_name, {})
        filename = cfg.get('filename', '')
        prefix, suffix, start_tp = self._parse_pattern(filename)
        self._patterns[pos_name] = (prefix, suffix)
        self._next_tp[pos_name] = start_tp
        # Track the JSON mtime so we can detect external edits and re-read
        # without the runner having to call us explicitly.
        log_dir = cfg.get('log_dir')
        if log_dir:
            roi_path = os.path.join(log_dir, 'tracking_RoIs.json')
            if os.path.isfile(roi_path):
                self._json_mtime[pos_name] = os.path.getmtime(roi_path)

    def refresh_filename(self, pos_name):
        """Public hook called by TrackingRunner.reinitialize_tracker."""
        self._refresh_filename_unlocked(pos_name)
        prefix, suffix = self._patterns[pos_name]
        self.logger.info(
            f"[{pos_name}] tracking filename updated to "
            f"{prefix}{{N}}{suffix} — starting at T={self._next_tp[pos_name]}"
        )

    # ------------------------------------------------------------------
    def connect(self):
        self._stop_event.clear()
        self.stop_requested = False

        if self.zen_feedback:
            try:
                self._open_zen_channel()
            except Exception as e:
                self.logger.error(
                    f"Could not open ZEN feedback channel — continuing without "
                    f"stage moves: {e}"
                )
                self.zen_feedback = False

        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name='FileWatchThread',
        )
        self._thread.start()
        self.logger.info(
            f"File-watch mode active — polling every {self.poll_interval_s:.1f}s "
            f"for {len(self.position_names)} position(s) — "
            f"stage feedback: {'ON' if self.zen_feedback else 'OFF'}"
        )

    # ------------------------------------------------------------------
    def _open_zen_channel(self):
        """Start a private asyncio loop in a daemon thread and open a gRPC
        channel + metadata to ZEN.  Mirrors the SSL setup used by
        ``MicroscopeInterface_Zeiss._async_main`` (h2 ALPN is mandatory)."""
        try:
            import grpclib.client
        except ImportError as e:
            raise RuntimeError(
                "grpclib is required for ZEN feedback. "
                "Run: pip install grpclib"
            ) from e

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if self.zen_cert_path and os.path.exists(self.zen_cert_path):
            ssl_ctx.load_verify_locations(cafile=self.zen_cert_path)
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            ssl_ctx.check_hostname = True
        else:
            self.logger.warning(
                "No ZEN cert_path provided — TLS verification disabled."
            )
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
        ssl_ctx.set_alpn_protocols(["h2"])

        self._zen_loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(self._zen_loop)
            self._zen_loop.run_forever()

        self._zen_thread = threading.Thread(
            target=_run_loop, daemon=True, name='ZenFeedbackLoop'
        )
        self._zen_thread.start()

        async def _open():
            self._zen_channel = grpclib.client.Channel(
                host=self.zen_address, port=self.zen_port, ssl=ssl_ctx
            )
            self._zen_metadata = [("control-token", self.zen_control_token)]

        future = asyncio.run_coroutine_threadsafe(_open(), self._zen_loop)
        future.result(timeout=10)
        self.logger.info(
            f"ZEN feedback channel opened to {self.zen_address}:{self.zen_port}"
        )

    def _poll_loop(self):
        while not self._stop_event.is_set():
            for pos_name in self.position_names:
                self._maybe_reread_json(pos_name)
                self._scan_position(pos_name)
            self._stop_event.wait(timeout=self.poll_interval_s)

    def _maybe_reread_json(self, pos_name):
        """If tracking_RoIs.json mtime changed, re-parse the filename."""
        cfg = self.positions_config.get(pos_name, {})
        log_dir = cfg.get('log_dir')
        if not log_dir:
            return
        roi_path = os.path.join(log_dir, 'tracking_RoIs.json')
        if not os.path.isfile(roi_path):
            return
        try:
            mtime = os.path.getmtime(roi_path)
        except OSError:
            return
        if mtime != self._json_mtime.get(pos_name):
            try:
                import json
                with open(roi_path) as f:
                    new_cfg = json.load(f)
                if 'filename' in new_cfg:
                    cfg['filename'] = new_cfg['filename']
                    old_pattern = self._patterns.get(pos_name)
                    prefix, suffix, start_tp = self._parse_pattern(new_cfg['filename'])
                    self._patterns[pos_name] = (prefix, suffix)
                    # Reset the cursor so tracking begins at the timepoint the
                    # ROI was redefined on, not from where we happened to be.
                    self._next_tp[pos_name] = start_tp
                    if old_pattern != (prefix, suffix):
                        self.logger.info(
                            f"[{pos_name}] filename pattern changed → "
                            f"{prefix}{{N}}{suffix} — starting at T={start_tp}"
                        )
                self._json_mtime[pos_name] = mtime
            except Exception as e:
                self.logger.warning(f"[{pos_name}] could not re-read tracking_RoIs.json: {e}")

    def _scan_position(self, pos_name):
        cfg = self.positions_config.get(pos_name, {})
        images_dir = cfg.get('images_dir') or os.path.join(self.dirpath, pos_name)
        if not os.path.isdir(images_dir):
            return
        prefix, suffix = self._patterns[pos_name]
        # Drain timepoints sequentially starting from next_tp.  Files must
        # already exist on disk; we don't speculatively wait.
        next_tp = self._next_tp[pos_name]
        while not self._stop_event.is_set():
            target = f"{prefix}{next_tp:04d}{suffix}"
            target_path = os.path.join(images_dir, target)
            if not os.path.isfile(target_path):
                break
            try:
                image = tifffile.imread(target_path)
            except Exception as e:
                self.logger.warning(f"[{pos_name}] failed to read {target}: {e}")
                break
            self.logger.info(f"[{pos_name}] queued {target}")
            self._queue.put((image, next_tp, pos_name))
            next_tp += 1
        self._next_tp[pos_name] = next_tp

    # ------------------------------------------------------------------
    def wait_for_image(self, timeout_ms=1000):
        try:
            return self._queue.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None, None, None

    def relative_move(self, position_name, shift_x, shift_y, shift_z):
        """Forward a tracker-computed shift to ZEN when feedback is on.

        When ``zen_feedback`` is False this is a logging no-op so the
        tracker can be exercised against on-disk frames without touching
        the microscope.  When True we soft-clamp the shift to the
        configured limits and call ``StageService.move_to`` /
        ``FocusService.move_to_position`` on the ZEN gateway.

        Stage moves are global on the ZEN side (one shared XYZ stage),
        so ``position_name`` is only used for logging here.
        """
        if not self.zen_feedback or self._zen_loop is None:
            self.logger.debug(
                f"[{position_name}] (no feedback) shift_um xyz="
                f"({shift_x:.2f}, {shift_y:.2f}, {shift_z:.2f})"
            )
            return

        def _clamp(v, limit, axis):
            if abs(v) > limit:
                self.logger.warning(
                    f"[{position_name}] {axis} shift {v:.1f} µm exceeds "
                    f"limit {limit:.0f} µm — clamped"
                )
                return math.copysign(limit, v)
            return v

        shift_x = _clamp(shift_x, self.max_xy_um, 'X')
        shift_y = _clamp(shift_y, self.max_xy_um, 'Y')
        shift_z = _clamp(shift_z, self.max_z_um,  'Z')

        future = asyncio.run_coroutine_threadsafe(
            self._async_relative_move(shift_x, shift_y, shift_z),
            self._zen_loop,
        )
        try:
            future.result(timeout=10)
            self.logger.info(
                f"[{position_name}] sent relative move xyz="
                f"({shift_x:.2f}, {shift_y:.2f}, {shift_z:.2f}) µm to ZEN"
            )
        except Exception as e:
            self.logger.error(f"[{position_name}] stage move failed: {e}")

    async def _async_relative_move(self, dx_um, dy_um, dz_um):
        """Query current stage / focus, apply relative offset.  Mirrors
        ``MicroscopeInterface_Zeiss._async_relative_move`` — see that
        method for cross-references with the official ZEN API examples."""
        try:
            from zen_api.lm.hardware.v2 import (
                StageServiceStub,
                StageServiceGetPositionRequest,
                StageServiceMoveToRequest,
            )
            from zen_api.focus.v2 import (
                FocusServiceStub,
                FocusServiceGetPositionRequest,
                FocusServiceMoveToPositionRequest,
            )
        except ImportError as e:
            self.logger.error(f"zen_api hardware stubs not found: {e}")
            return

        stage_stub = StageServiceStub(channel=self._zen_channel, metadata=self._zen_metadata)
        focus_stub = FocusServiceStub(channel=self._zen_channel, metadata=self._zen_metadata)

        if abs(dx_um) > 1e-3 or abs(dy_um) > 1e-3:
            pos = await stage_stub.get_position(StageServiceGetPositionRequest())
            await stage_stub.move_to(
                StageServiceMoveToRequest(
                    x=pos.x + dx_um * 1e-6,
                    y=pos.y + dy_um * 1e-6,
                )
            )

        if abs(dz_um) > 1e-3:
            z_pos = await focus_stub.get_position(FocusServiceGetPositionRequest())
            current_z = getattr(z_pos, 'z', None) or getattr(z_pos, 'position', 0.0)
            await focus_stub.move_to_position(
                FocusServiceMoveToPositionRequest(
                    position=current_z + dz_um * 1e-6
                )
            )

    def disconnect(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        # Tear down the ZEN feedback loop + channel
        if self._zen_loop is not None:
            try:
                if self._zen_channel is not None:
                    async def _close():
                        self._zen_channel.close()
                    asyncio.run_coroutine_threadsafe(_close(), self._zen_loop).result(timeout=5)
            except Exception as e:
                self.logger.warning(f"Error closing ZEN channel: {e}")
            try:
                self._zen_loop.call_soon_threadsafe(self._zen_loop.stop)
            except Exception:
                pass
            if self._zen_thread is not None:
                self._zen_thread.join(timeout=5)
                self._zen_thread = None
            self._zen_loop = None
            self._zen_channel = None
            self._zen_metadata = None

    def stop(self):
        self.stop_requested = True
        self.disconnect()

