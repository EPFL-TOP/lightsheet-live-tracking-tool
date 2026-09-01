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
    def refresh_filename(self, pos_name):
        """No-op on the ZEN backend.

        ZEN cannot repoint the tracking channel mid-acquisition without
        tearing down and re-establishing the streaming subscription.
        Users who need mid-run channel switches should use the Files
        or Micromanager backend instead.
        """
        pass

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
        Query current stage / focus position then apply a relative offset.

        Module paths and field names match zen_api-2025.10.1:
          - Stage + Focus services both live in ``zen_api.lm.hardware.v2``.
          - FocusServiceGetPositionResponse exposes ``.value`` (meters);
            FocusServiceMoveToRequest accepts ``value=...`` in meters.
        Cross-check with:
            OAD/ZEN-API/python_examples/zenapi_stage_LM.py
            OAD/ZEN-API/python_examples/zenapi_zdrive.py
        """
        try:
            from zen_api.lm.hardware.v2 import (
                FocusServiceStub,
                FocusServiceGetPositionRequest,
                FocusServiceMoveToRequest,
            )
            try:
                from zen_api.hardware.v1 import (
                    StageServiceStub,
                    StageServiceGetStagePositionRequest as _StageGetReq,
                    StageServiceMoveToRequest,
                    StageAxis,
                    AxisIdentifier,
                )
                _stage_api = 'v1_per_axis'
            except ImportError:
                from zen_api.lm.hardware.v2 import (
                    StageServiceStub,
                    StageServiceGetPositionRequest as _StageGetReq,
                    StageServiceMoveToRequest,
                )
                StageAxis = None
                AxisIdentifier = None
                _stage_api = 'lm_v2_xy'
        except ImportError as e:
            self.logger.error(f"zen_api hardware stubs not found: {e}")
            return

        # zen_api stubs are generated by betterproto: metadata in constructor,
        # method names in snake_case, no per-call metadata argument.
        stage_stub = StageServiceStub(channel=self._channel, metadata=self._metadata)
        focus_stub = FocusServiceStub(channel=self._channel, metadata=self._metadata)

        if abs(dx_um) > 1e-3 or abs(dy_um) > 1e-3:
            if _stage_api == 'v1_per_axis':
                resp = await stage_stub.get_stage_position(_StageGetReq())
                cur_x = cur_y = 0.0
                for a in resp.axis_positions:
                    if a.axis == AxisIdentifier.X:
                        cur_x = a.position
                    elif a.axis == AxisIdentifier.Y:
                        cur_y = a.position
                await stage_stub.move_to(StageServiceMoveToRequest(
                    axis_to_move=[
                        StageAxis(axis=AxisIdentifier.X, position=cur_x + dx_um * 1e-6),
                        StageAxis(axis=AxisIdentifier.Y, position=cur_y + dy_um * 1e-6),
                    ]
                ))
            else:
                pos = await stage_stub.get_position(_StageGetReq())
                await stage_stub.move_to(
                    StageServiceMoveToRequest(
                        x=pos.x + dx_um * 1e-6,
                        y=pos.y + dy_um * 1e-6,
                    )
                )

        if abs(dz_um) > 1e-3:
            z_pos = await focus_stub.get_position(FocusServiceGetPositionRequest())
            await focus_stub.move_to(
                FocusServiceMoveToRequest(
                    value=z_pos.value + dz_um * 1e-6
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
        # ZEN-API-started experiment id (required for multi-scene feedback
        # via TilesService.add_positions).  Empty string ⇒ single-scene
        # mode using StageService.move_to.
        self.zen_experiment_id = file_params.get('zen_experiment_id', '')
        # Total scenes ZEN's running experiment exposes — needed so we can
        # provide a complete position list to TilesService.add_positions.
        # Defaults to len(positions_config) which counts only the scenes
        # the user is tracking; bump it via file_params['n_scenes'] when
        # not every scene has a tracking_RoIs.json.
        self.n_scenes = int(file_params.get('n_scenes',
                                            len(self.position_names)))
        # Multi-scene feedback uses TilesService instead of StageService.
        self.multi_scene_mode = (
            self.zen_feedback
            and self.n_scenes > 1
            and bool(self.zen_experiment_id)
        )

        # asyncio loop + gRPC channel for stage commands.  Lazily created
        # in connect() when feedback is enabled.
        self._zen_loop    = None
        self._zen_thread  = None
        self._zen_channel = None
        self._zen_metadata = None
        self._zen_status_task = None
        # Per-position cumulative drift in µm (for diagnostics + as the
        # source of truth for multi-scene TilesService updates).
        self._cum_drift = {pos: [0.0, 0.0, 0.0] for pos in self.position_names}
        # Serialise moves across positions so multiple in-flight RPCs
        # cannot interleave on the ZEN channel.
        self._move_lock = threading.Lock()
        # Multi-scene-mode bookkeeping (only populated when
        # ``multi_scene_mode`` is True).  ``_initial_pos_m[i]`` is the
        # uncorrected stage+focus position of scene i in METERS.
        #
        # Source of truth for these values:
        #   1. ``file_params['initial_positions_um']`` — if non-empty, the
        #      user explicitly provided baselines via the panel text area.
        #      Each entry is ``(x, y, z)`` in µm; we convert to metres.
        #   2. Auto-capture from ZEN ``StageService.GetPosition`` the first
        #      time ``register_on_status_changed`` reports a scene as
        #      acquiring.  This is racy (status events lag the stage), so
        #      we use it only for slots the user didn't fill in.
        # Until every slot is populated we won't apply corrections — that
        # avoids wiping ZEN's stored positions with bogus values.
        manual_um = file_params.get('initial_positions_um') or []
        self._initial_pos_m = [None] * self.n_scenes
        for i, p in enumerate(manual_um[:self.n_scenes]):
            if p is None:
                continue
            x_um, y_um, z_um = p
            self._initial_pos_m[i] = (
                float(x_um) * 1e-6,
                float(y_um) * 1e-6,
                float(z_um) * 1e-6,
            )
        self._pending_drift_um = [[0.0, 0.0, 0.0] for _ in range(self.n_scenes)]
        self._pos_update_needed = False
        self._last_acq_running  = False
        self._pos_state_lock    = threading.Lock()

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
        # Reset per-position cumulative drift — the new tracker tracks a
        # (possibly different) target on a (possibly different) channel,
        # so the previous drift no longer reflects anything meaningful.
        self._cum_drift[pos_name] = [0.0, 0.0, 0.0]
        prefix, suffix = self._patterns[pos_name]
        self.logger.info(
            f"[{pos_name}] tracking filename updated to "
            f"{prefix}{{N}}{suffix} — starting at T={self._next_tp[pos_name]}, "
            f"cumulative drift reset to 0"
        )

    # ------------------------------------------------------------------
    # LS1-contract compatibility stubs — the files backend has no
    # "pause between positions" concept, so these are no-ops.  They
    # exist so the runner's LS1 path (run_LS1) can dispatch to us
    # without AttributeError'ing.
    def wait_for_pause(self, timeout_ms=1000):
        """LS1 contract compatibility — files backend has no pause cycle."""
        return self.wait_for_image(timeout_ms)

    def pause_after_position(self):    pass
    def no_pause_after_position(self): pass
    def continue_from_pause(self):     pass

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
        # Explicit per-position summary at startup so a multi-scene run
        # makes obvious which positions are being watched and with which
        # filename pattern / starting timepoint.
        for pos_name in self.position_names:
            prefix, suffix = self._patterns.get(pos_name, ('t', '.tif'))
            self.logger.info(
                f"  [{pos_name}] pattern={prefix}{{N:04d}}{suffix} "
                f"start_T={self._next_tp.get(pos_name, 0)} "
                f"cumulative=(0.00, 0.00, 0.00) µm"
            )
        # Baseline status for multi-scene mode
        if self.multi_scene_mode:
            for i, p in enumerate(self._initial_pos_m):
                if p is None:
                    self.logger.info(
                        f"  scene {i}: baseline = <auto-discover from ZEN>"
                    )
                else:
                    self.logger.info(
                        f"  scene {i}: baseline = "
                        f"({p[0]*1e6:+.1f}, {p[1]*1e6:+.1f}, "
                        f"{p[2]*1e6:+.1f}) µm  [manual]"
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
            f"ZEN feedback channel opened to {self.zen_address}:{self.zen_port} "
            f"— mode: {'multi-scene (TilesService)' if self.multi_scene_mode else 'single-scene (StageService)'}"
        )

        # Start the long-running status monitor as its own scheduled
        # coroutine.  Using ``run_coroutine_threadsafe`` here (rather than
        # ``asyncio.create_task`` inside ``_open()``) is the bulletproof
        # pattern: it returns a concurrent.futures.Future that we keep a
        # strong reference to, the loop knows it's a top-level task, and
        # we can observe the task with ``.done()`` / ``.exception()``.
        if self.multi_scene_mode:
            self.logger.info(
                "Scheduling _status_monitor_loop on ZEN feedback loop"
            )
            self._zen_status_task = asyncio.run_coroutine_threadsafe(
                self._status_monitor_loop(), self._zen_loop
            )
            # Drain the result asynchronously so any startup error
            # (e.g. ImportError on zen_api submodules) is surfaced
            # rather than swallowed.
            def _on_status_done(fut):
                try:
                    fut.result()
                except Exception as e:
                    self.logger.error(
                        f"_status_monitor_loop exited with: {e}",
                        exc_info=True,
                    )
            self._zen_status_task.add_done_callback(_on_status_done)

    async def _status_monitor_loop(self):
        """Subscribe to ExperimentService.RegisterOnStatusChanged and apply
        ``TilesService`` position updates between cycles.  See body for
        the per-event trigger logic."""
        # Diagnostic prints (sys.stderr) — needed because we don't yet
        # know whether the loop is actually running this coroutine on
        # the user's setup; if the entry log line below never appears
        # we want a separate channel that bypasses the logger.
        import sys
        print("[status-monitor] coroutine ENTERED", file=sys.stderr, flush=True)
        try:
            try:
                from zen_api.acquisition.v1beta import (
                    ExperimentServiceStub,
                    ExperimentServiceRegisterOnStatusChangedRequest,
                )
                # Focus stays in lm/hardware/v2 in both 2025.10 and 2026.05.
                from zen_api.lm.hardware.v2 import (
                    FocusServiceStub,
                    FocusServiceGetPositionRequest,
                )
                # Stage moved between releases:
                #   2025.10:  zen_api.lm.hardware.v2.StageServiceStub
                #             + StageServiceGetPositionRequest → .x .y
                #   2026.05:  zen_api.hardware.v1.StageServiceStub
                #             + StageServiceGetStagePositionRequest →
                #             .axis_positions: List[StageAxis]
                # Detect which one is available.
                try:
                    from zen_api.hardware.v1 import (
                        StageServiceStub,
                        StageServiceGetStagePositionRequest as _StageGetReq,
                        AxisIdentifier as _AxisId,
                    )
                    _stage_api = 'v1_per_axis'
                except ImportError:
                    from zen_api.lm.hardware.v2 import (
                        StageServiceStub,
                        StageServiceGetPositionRequest as _StageGetReq,
                    )
                    _AxisId = None
                    _stage_api = 'lm_v2_xy'
            except ImportError as e:
                print(f"[status-monitor] IMPORT FAILED: {e}",
                      file=sys.stderr, flush=True)
                self.logger.error(f"zen_api stubs not found: {e}")
                return

            print("[status-monitor] imports OK", file=sys.stderr, flush=True)

            exp_stub   = ExperimentServiceStub(channel=self._zen_channel,
                                               metadata=self._zen_metadata)
            stage_stub = StageServiceStub(channel=self._zen_channel,
                                          metadata=self._zen_metadata)
            focus_stub = FocusServiceStub(channel=self._zen_channel,
                                          metadata=self._zen_metadata)
            print("[status-monitor] stubs constructed", file=sys.stderr, flush=True)

            self.logger.info(
                f"Status monitor subscribing to experiment {self.zen_experiment_id} "
                f"(watching {self.n_scenes} scene(s))"
            )
            print(
                f"[status-monitor] subscribing to experiment "
                f"{self.zen_experiment_id} ({self.n_scenes} scenes)",
                file=sys.stderr, flush=True,
            )
        except Exception as e:
            print(f"[status-monitor] FAILED before subscribe: {e!r}",
                  file=sys.stderr, flush=True)
            raise
        # Empirical observation from the smoke test: ZEN sends 2 events per
        # scene per cycle, with ``is_acquisition_running`` set to True the
        # whole time (it never flips to False between cycles — it just
        # stops sending events during the time-lapse interval).  So we
        # can't use ``acq=False`` as the trigger for "apply position
        # updates".  Instead we detect the *post-acquire* event of the
        # last scene in a cycle: scene_idx == n_scenes-1 *and* the same
        # scene index just reported a previous event (= second event in
        # a row, i.e. the image was just captured rather than ZEN having
        # just snapped to a new scene).  After that event ZEN goes idle
        # for the time-lapse interval, which is exactly when we want to
        # rewrite the position list.
        last_idx_logged = None
        prev_scene_idx  = -1
        import sys as _sys
        first_event = True
        try:
            async for resp in exp_stub.register_on_status_changed(
                ExperimentServiceRegisterOnStatusChangedRequest(
                    self.zen_experiment_id
                )
            ):
                s = resp.status
                idx  = int(getattr(s, 'scenes_index', 0))
                acq  = bool(s.is_acquisition_running)
                tp   = int(getattr(s, 'time_points_index', 0))
                imgs = int(getattr(s, 'images_acquired_index', 0))
                if first_event:
                    print(
                        f"[status-monitor] FIRST event received: "
                        f"tp={tp} scene={idx} acq={acq} imgs={imgs}",
                        file=_sys.stderr, flush=True,
                    )
                    first_event = False
                if (idx, acq) != last_idx_logged:
                    self.logger.info(
                        f"[status] tp={tp} scene_idx={idx} "
                        f"acq_running={acq} imgs={imgs}"
                    )
                    last_idx_logged = (idx, acq)

                if acq and 0 <= idx < self.n_scenes \
                        and self._initial_pos_m[idx] is None:
                    # Auto-discovery fallback for baselines the user
                    # didn't manually enter.  Racy because ZEN reports
                    # ``acq=True`` slightly before the stage settles;
                    # prefer the panel's manual baseline text area.
                    await asyncio.sleep(0.5)
                    try:
                        # Cross-version stage read
                        if _stage_api == 'v1_per_axis':
                            resp = await stage_stub.get_stage_position(
                                _StageGetReq()
                            )
                            px = py = 0.0
                            for a in resp.axis_positions:
                                if a.axis == _AxisId.X:
                                    px = a.position
                                elif a.axis == _AxisId.Y:
                                    py = a.position
                        else:
                            p = await stage_stub.get_position(_StageGetReq())
                            px, py = p.x, p.y
                        z = await focus_stub.get_position(
                            FocusServiceGetPositionRequest()
                        )
                        self._initial_pos_m[idx] = (px, py, z.value)
                        self.logger.info(
                            f"Captured baseline for scene {idx}: "
                            f"({px*1e6:+.1f}, {py*1e6:+.1f}, "
                            f"{z.value*1e6:+.1f}) µm  [auto, racy]"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to read stage for scene {idx}: {e}"
                        )

                # Trigger: end-of-cycle = last scene's post-acquire event.
                # Post-acquire is the *second* event in a row for the
                # same scene_idx (first event is the pre-acquire snap).
                if (idx == self.n_scenes - 1
                        and idx == prev_scene_idx):
                    self.logger.info(
                        f"[cycle end] tp={tp} last-scene post-acquire "
                        f"detected — applying position updates"
                    )
                    await self._apply_position_updates_async()

                prev_scene_idx = idx

                # Defensive: keep the old acq=False handler too in case
                # a future ZEN release does flip the flag between cycles.
                if not acq and self._last_acq_running:
                    await self._apply_position_updates_async()
                self._last_acq_running = acq
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.warning(f"Status monitor stopped: {e}")

    async def _apply_position_updates_async(self):
        """Push current (baseline + cum_drift) to ZEN via TilesService."""
        # Snapshot drift under lock; quickly bail if nothing to do.
        with self._pos_state_lock:
            if not self._pos_update_needed:
                return
            if any(p is None for p in self._initial_pos_m):
                # Still waiting on first cycle to capture all baselines
                return
            drift = [list(d) for d in self._pending_drift_um]
            self._pos_update_needed = False

        try:
            from zen_api.lm.acquisition.v1 import (
                TilesServiceStub,
                TilesServiceClearRequest,
                TilesServiceAddPositionsRequest,
                Position3D,
            )
        except ImportError as e:
            self.logger.error(f"zen_api.lm.acquisition stubs not found: {e}")
            return

        tiles = TilesServiceStub(channel=self._zen_channel,
                                 metadata=self._zen_metadata)
        new_positions = []
        for i in range(self.n_scenes):
            ox, oy, oz = self._initial_pos_m[i]
            dx, dy, dz = drift[i]
            new_positions.append(Position3D(
                x=ox + dx * 1e-6,
                y=oy + dy * 1e-6,
                z=oz + dz * 1e-6,
            ))

        try:
            await tiles.clear(TilesServiceClearRequest(
                experiment_id=self.zen_experiment_id
            ))
            await tiles.add_positions(TilesServiceAddPositionsRequest(
                experiment_id=self.zen_experiment_id,
                positions=new_positions,
            ))
        except Exception as e:
            self.logger.error(f"TilesService update failed: {e}")
            # Re-mark needed so we retry on the next idle window
            with self._pos_state_lock:
                self._pos_update_needed = True
            return

        self.logger.info(
            "Applied TilesService position update — "
            + ", ".join(
                f"scene {i}: ({p.x*1e6:+.1f}, {p.y*1e6:+.1f}, {p.z*1e6:+.1f}) µm"
                for i, p in enumerate(new_positions)
            )
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

        Three modes:
          - ``zen_feedback`` False → no-op (logging only).  Use this when
            replaying recorded TIFs offline.
          - ``multi_scene_mode`` True (n_scenes > 1, experiment_id set)
            → don't move the stage live.  Accumulate per-scene drift; the
            background status-monitor task applies it via
            ``TilesService.clear`` + ``add_positions`` whenever ZEN is
            between acquisitions.  This is the right path for tile /
            multi-position experiments because ZEN re-snaps the stage to
            each scene's stored position before every acquisition.
          - Otherwise → single-scene live stage moves via
            ``StageService.move_to`` + ``FocusService.move_to`` (the
            previous default).
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

        # Accumulate per-position drift (diagnostic copy used by the
        # cumulative-drift log line and end-of-run summary).
        drift = self._cum_drift.setdefault(position_name, [0.0, 0.0, 0.0])
        drift[0] += shift_x
        drift[1] += shift_y
        drift[2] += shift_z

        # ── Multi-scene mode ──────────────────────────────────────────────
        # Do NOT touch the stage live; the status monitor will apply the
        # accumulated drift via TilesService when ZEN is idle between
        # acquisitions.
        if self.multi_scene_mode:
            scene_idx = self._scene_index_for(position_name)
            if scene_idx is None:
                self.logger.warning(
                    f"[{position_name}] could not map to a scene index — "
                    "drift recorded but won't be pushed to ZEN."
                )
                return
            with self._pos_state_lock:
                self._pending_drift_um[scene_idx][0] += shift_x
                self._pending_drift_um[scene_idx][1] += shift_y
                self._pending_drift_um[scene_idx][2] += shift_z
                self._pos_update_needed = True
            self.logger.info(
                f"[{position_name}] scene_idx={scene_idx} queued drift "
                f"({shift_x:+.2f}, {shift_y:+.2f}, {shift_z:+.2f}) µm  "
                f"cumulative=({drift[0]:+.2f}, {drift[1]:+.2f}, "
                f"{drift[2]:+.2f}) µm"
            )
            return

        # ── Single-scene mode ─────────────────────────────────────────────
        with self._move_lock:
            future = asyncio.run_coroutine_threadsafe(
                self._async_relative_move(shift_x, shift_y, shift_z),
                self._zen_loop,
            )
            try:
                future.result(timeout=10)
                self.logger.info(
                    f"[{position_name}] move xyz=({shift_x:+.2f}, "
                    f"{shift_y:+.2f}, {shift_z:+.2f}) µm  "
                    f"cumulative=({drift[0]:+.2f}, {drift[1]:+.2f}, "
                    f"{drift[2]:+.2f}) µm"
                )
            except Exception as e:
                self.logger.error(f"[{position_name}] stage move failed: {e}")

    def _scene_index_for(self, position_name):
        """Map a tracker position name to the ZEN scene index.

        Convention from ``ZenIngest``: position folders are named
        ``scene_{i:03d}`` (zero-padded).  We parse the trailing integer.
        Falls back to ``self.position_names.index(position_name)`` for
        legacy / custom names.
        """
        m = self._re.match(r'^scene_(\d+)$', position_name)
        if m:
            return int(m.group(1))
        try:
            return self.position_names.index(position_name)
        except ValueError:
            return None

    async def _async_relative_move(self, dx_um, dy_um, dz_um):
        """Query current stage / focus and apply a relative offset.

        Module paths and field names match zen_api-2025.10.1:
          - StageService.{get_position, move_to} live in
            ``zen_api.lm.hardware.v2`` (lm = light microscopy).
          - FocusService also lives there (NOT ``zen_api.focus``).
          - FocusServiceGetPositionResponse has a single ``value`` field
            in meters (not ``z`` or ``position``).
          - FocusService method is ``move_to`` (not ``move_to_position``)
            and its request takes ``value=...``.
        Cross-check with
          OAD/ZEN-API/python_examples/zenapi_stage_LM.py
          OAD/ZEN-API/python_examples/zenapi_zdrive.py
        """
        try:
            from zen_api.lm.hardware.v2 import (
                FocusServiceStub,
                FocusServiceGetPositionRequest,
                FocusServiceMoveToRequest,
            )
            try:
                # 2026.05: per-axis Stage API in zen_api.hardware.v1
                from zen_api.hardware.v1 import (
                    StageServiceStub,
                    StageServiceGetStagePositionRequest as _StageGetReq,
                    StageServiceMoveToRequest,
                    StageAxis,
                    AxisIdentifier,
                )
                _stage_api = 'v1_per_axis'
            except ImportError:
                # 2025.10 and earlier: XY-keyed Stage API in lm/hardware/v2
                from zen_api.lm.hardware.v2 import (
                    StageServiceStub,
                    StageServiceGetPositionRequest as _StageGetReq,
                    StageServiceMoveToRequest,
                )
                StageAxis = None
                AxisIdentifier = None
                _stage_api = 'lm_v2_xy'
        except ImportError as e:
            self.logger.error(f"zen_api hardware stubs not found: {e}")
            return

        stage_stub = StageServiceStub(channel=self._zen_channel, metadata=self._zen_metadata)
        focus_stub = FocusServiceStub(channel=self._zen_channel, metadata=self._zen_metadata)

        if abs(dx_um) > 1e-3 or abs(dy_um) > 1e-3:
            if _stage_api == 'v1_per_axis':
                resp = await stage_stub.get_stage_position(_StageGetReq())
                cur_x = cur_y = 0.0
                for a in resp.axis_positions:
                    if a.axis == AxisIdentifier.X:
                        cur_x = a.position
                    elif a.axis == AxisIdentifier.Y:
                        cur_y = a.position
                await stage_stub.move_to(StageServiceMoveToRequest(
                    axis_to_move=[
                        StageAxis(axis=AxisIdentifier.X, position=cur_x + dx_um * 1e-6),
                        StageAxis(axis=AxisIdentifier.Y, position=cur_y + dy_um * 1e-6),
                    ]
                ))
            else:
                pos = await stage_stub.get_position(_StageGetReq())
                await stage_stub.move_to(
                    StageServiceMoveToRequest(
                        x=pos.x + dx_um * 1e-6,
                        y=pos.y + dy_um * 1e-6,
                    )
                )

        if abs(dz_um) > 1e-3:
            z_pos = await focus_stub.get_position(FocusServiceGetPositionRequest())
            await focus_stub.move_to(
                FocusServiceMoveToRequest(
                    value=z_pos.value + dz_um * 1e-6
                )
            )

    def disconnect(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        # Final per-position drift summary so the user can see at a glance
        # how much each scene drifted over the whole run.
        if any(any(v != 0.0 for v in d) for d in self._cum_drift.values()):
            self.logger.info("Final cumulative drift per position:")
            for pos_name, d in self._cum_drift.items():
                self.logger.info(
                    f"  [{pos_name}] cumulative=({d[0]:+.2f}, "
                    f"{d[1]:+.2f}, {d[2]:+.2f}) µm"
                )

        # Tear down the ZEN feedback loop + channel
        if self._zen_loop is not None:
            # Cancel the status monitor first so its async-for stream
            # exits cleanly before the channel is closed under it.
            if self._zen_status_task is not None:
                try:
                    self._zen_status_task.cancel()
                except Exception:
                    pass
                self._zen_status_task = None
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


# ─────────────────────────────────────────────────────────────────────────────
# Micro-Manager (pymmcore-plus) — FLAGSHIP backend going forward.
#
# This class is the successor to the ZEN interface for hardware feedback: it
# talks to any Micro-Manager-supported microscope through pymmcore-plus and
# drives an MDA (Multi-Dimensional Acquisition) sequence that we ourselves
# enqueue timepoint-by-timepoint.  Feedback loop:
#
#   1. connect() loads the MM system config and starts run_mda() with an
#      infinite iterator-backed event queue.
#   2. The MDA engine executes each MDAEvent on its own thread; when a frame
#      is captured, the ``frameReady`` signal fires (on that same MDA thread).
#   3. Our frameReady handler:
#        (a) optionally substitutes the DemoCamera image with a synthetic
#            source (Phase-A smoke test path),
#        (b) saves the frame to disk (t{NNNN}_{channel}.tif),
#        (c) pushes (image, tp, pos_name) onto ``self._image_queue`` — the
#            tracker-facing queue that ``wait_for_image`` reads.
#      When the LAST scene of a cycle has produced its frameReady, we
#      UNCONDITIONALLY enqueue MDAEvents for all scenes at the next
#      timepoint (guarded only by ``stop_after_tp``).  This is Bug 1
#      prevention: zero-drift 5-cycle runs must complete.
#   4. relative_move() runs on the tracker thread and only mutates
#      ``self._cum_drift`` under a lock — this is Bug 2 prevention.
#      _enqueue_timepoint reads that dict when it builds the NEXT event
#      for the scene, adding the offset to the baseline stored at __init__.
#   5. Baselines come from ``positions_config[pos_name]["xyz_um"]`` — never
#      from ``mmc.getXYPosition``.  This is Bug 3 prevention.
# ─────────────────────────────────────────────────────────────────────────────
class MicroscopeInterface_Micromanager:
    """Live Micro-Manager backend via pymmcore-plus.

    Public interface (matches ``MicroscopeInterface_LS1`` /
    ``MicroscopeInterface_Files``):

        wait_for_image(timeout_ms) -> (image, tp, pos_name) or (None, None, None)
        relative_move(pos_name, dx, dy, dz)
        connect(); disconnect(); stop()
        refresh_filename(pos_name)
        pause_after_position(); no_pause_after_position(); continue_from_pause()
    """

    def __init__(self, positions_config, dirpath, mm_params):
        # ── Local imports so ZEN / LS1 users don't need pymmcore-plus ──
        # We import here (not at module level) so importing this module
        # never fails for LS1 / ZEN / Files users who lack pymmcore-plus.
        try:
            import pymmcore_plus  # noqa: F401  (probe availability)
            from useq import MDAEvent
        except ImportError as e:
            raise ImportError(
                "pymmcore-plus and useq are required for "
                "MicroscopeInterface_Micromanager. Install with:\n"
                "    pip install pymmcore-plus useq-schema"
            ) from e
        self._MDAEvent = MDAEvent

        self.positions_config = positions_config
        self.dirpath = str(dirpath)
        self.position_names = list(positions_config.keys())

        # ── mm_params defaults ────────────────────────────────────────
        mm_params = mm_params or {}
        self.cfg_path         = mm_params.get('cfg_path', '')
        self.channel_group    = mm_params.get('channel_group', 'Channel')
        self.channel_preset   = mm_params.get('channel_preset', 'Brightfield')
        self.exposure_ms      = float(mm_params.get('exposure_ms', 100.0))
        self.z_stack          = mm_params.get('z_stack', None)
        self.interval_s       = float(mm_params.get('interval_s', 0.0))
        self.max_xy_um        = float(mm_params.get('max_xy_um', 500.0))
        self.max_z_um         = float(mm_params.get('max_z_um', 100.0))
        self.synthetic_source = mm_params.get('synthetic_source', None)
        self.stop_after_tp    = mm_params.get('stop_after_tp', None)

        self.logger = init_logger(self.__class__.__name__)

        # ── Bug 3 prevention: baselines from positions_config ─────────
        self._baseline_um = {}
        for pos_name in self.position_names:
            xyz = positions_config.get(pos_name, {}).get('xyz_um')
            if xyz is None:
                self.logger.warning(
                    f"[{pos_name}] positions_config missing 'xyz_um' — "
                    "defaulting to (0.0, 0.0, 0.0). This position will start "
                    "at MM's current stage location; correct by editing the "
                    "positions config."
                )
                self._baseline_um[pos_name] = (0.0, 0.0, 0.0)
            else:
                x, y, z = xyz
                self._baseline_um[pos_name] = (float(x), float(y), float(z))

        # ── Threading + queue plumbing ────────────────────────────────
        # _cum_drift is the tracker-owned adjustment applied on top of the
        # baseline every time we build the next MDAEvent for the scene.
        self._cum_drift = {p: [0.0, 0.0, 0.0] for p in self.position_names}
        self._drift_lock = threading.Lock()          # protects _cum_drift
        self._config_lock = threading.Lock()         # protects _baseline_um & channel_preset
        self._image_queue = queue.Queue()            # frameReady -> tracker
        self._mda_queue = queue.Queue()              # us -> MDA engine
        self._stop_event = threading.Event()
        self._mda_future = None                      # from run_mda()
        # Per-cycle bookkeeping (populated in frameReady on the MDA thread)
        self._current_tp = 0
        self._scenes_seen_this_tp = set()
        self._cycle_lock = threading.Lock()
        self.stop_requested = False

        # Filled in by connect()
        self.mmc = None

    # ------------------------------------------------------------------
    def _wait_for_device(self, device, timeout_s=5.0):
        """Poll ``mmc.deviceBusy`` with 10 ms sleeps, raise on timeout.

        Use this everywhere we'd otherwise call ``mmc.waitForDevice`` —
        the built-in version has no upper bound and can hang forever if
        a device gets wedged.
        """
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                if not self.mmc.deviceBusy(device):
                    return
            except Exception:
                # deviceBusy can raise on undefined-yet devices; treat as
                # transient and keep polling.
                pass
            time.sleep(0.01)
        raise TimeoutError(
            f"Device {device!r} still busy after {timeout_s:.2f}s"
        )

    # ------------------------------------------------------------------
    def connect(self):
        from pymmcore_plus import CMMCorePlus

        self._stop_event.clear()
        self.stop_requested = False
        self.mmc = CMMCorePlus.instance()

        if self.cfg_path:
            self.logger.info(f"Loading MM config: {self.cfg_path}")
            self.mmc.loadSystemConfiguration(self.cfg_path)
        else:
            self.logger.info("Loading MM demo config (no cfg_path given)")
            self.mmc.loadSystemConfiguration()

        # Device-interface-version assertion: log so drift is visible
        try:
            self.logger.info(f"MMCore API version:    {self.mmc.getAPIVersionInfo()}")
            self.logger.info(f"MMCore build version:  {self.mmc.getVersionInfo()}")
        except Exception as e:
            self.logger.warning(f"Could not read MMCore version info: {e}")

        # Configure channel + exposure
        with self._config_lock:
            _cp = self.channel_preset
        try:
            self.mmc.setConfig(self.channel_group, _cp)
        except Exception as e:
            self.logger.warning(
                f"Could not set channel {self.channel_group}/"
                f"{_cp}: {e} — using current setting"
            )
        self.mmc.setExposure(self.exposure_ms)

        # Wire frameReady BEFORE starting the MDA
        self.mmc.mda.events.frameReady.connect(self._on_frame_ready)

        # Prime the MDA queue with tp=0 for every scene
        self._current_tp = 0
        self._scenes_seen_this_tp = set()
        self._enqueue_timepoint(0)

        # Start MDA with a generator that drains _mda_queue.  When the
        # queue yields a sentinel (None), the generator returns and the
        # MDA engine finishes cleanly.
        def _event_iterator():
            while not self._stop_event.is_set():
                try:
                    ev = self._mda_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if ev is None:
                    return
                yield ev

        self._mda_future = self.mmc.run_mda(_event_iterator())
        self.logger.info(
            f"MDA started — {len(self.position_names)} scene(s), "
            f"interval={self.interval_s:.2f}s, "
            f"z_stack={'ON' if self.z_stack else 'OFF'}, "
            f"synthetic_source={'ON' if self.synthetic_source else 'OFF'}"
        )

    # ------------------------------------------------------------------
    def _enqueue_timepoint(self, tp):
        """Build one MDAEvent per scene at timepoint ``tp`` and drop them
        onto the MDA queue.  Runs on:
          - the tracker thread (once, from connect() at tp=0), and
          - the MDA thread (from _on_frame_ready when a cycle completes).
        Reads ``_cum_drift`` under lock — Bug 2 propagation site.
        """
        if self.stop_after_tp is not None and tp >= int(self.stop_after_tp):
            self.logger.info(
                f"stop_after_tp={self.stop_after_tp} reached — signalling "
                "MDA engine to finish"
            )
            self._mda_queue.put(None)
            return

        with self._drift_lock:
            drift_snapshot = {p: tuple(d) for p, d in self._cum_drift.items()}

        with self._config_lock:
            channel_preset_snapshot = self.channel_preset

        for scene_idx, pos_name in enumerate(self.position_names):
            with self._config_lock:
                baseline_snapshot = self._baseline_um[pos_name]
            bx, by, bz = baseline_snapshot
            dx, dy, dz = drift_snapshot.get(pos_name, (0.0, 0.0, 0.0))
            x_um = bx + dx
            y_um = by + dy
            z_um = bz + dz

            # Optional z-stack: enqueue one event per slice; we keep the
            # position + timepoint metadata identical so frameReady can
            # collapse them if desired.  For Phase A we treat the middle
            # slice as the tracker-visible frame (last-slice fires the
            # cycle-advance).  Simpler path: single-slice event.
            if self.z_stack:
                rng = float(self.z_stack.get('range_um', 0.0))
                step = float(self.z_stack.get('step_um', 1.0))
                n_slices = max(1, int(round(rng / step)) + 1)
                zs = np.linspace(z_um - rng / 2.0, z_um + rng / 2.0, n_slices)
            else:
                zs = [z_um]

            for z_idx, zv in enumerate(zs):
                # useq.MDAEvent metadata is user-defined — we stuff the
                # tracker's per-frame context in there so frameReady can
                # recover it without needing an external side channel.
                meta = {
                    'pos_name': pos_name,
                    'tp': tp,
                    'scene_idx': scene_idx,
                    'z_idx': z_idx,
                    'n_z': len(zs),
                    'is_last_scene': (scene_idx == len(self.position_names) - 1),
                }
                ev = self._MDAEvent(
                    index={'t': tp, 'p': scene_idx, 'z': z_idx},
                    x_pos=x_um,
                    y_pos=y_um,
                    z_pos=float(zv),
                    exposure=self.exposure_ms,
                    channel={'group': self.channel_group,
                             'config': channel_preset_snapshot},
                    min_start_time=(tp * self.interval_s
                                    if self.interval_s > 0 else None),
                    metadata=meta,
                )
                self._mda_queue.put(ev)

        self.logger.info(
            f"Enqueued tp={tp} for {len(self.position_names)} scene(s); "
            f"drift snapshot: "
            + ", ".join(f"{p}=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})"
                        for p, d in drift_snapshot.items())
        )

    # ------------------------------------------------------------------
    def _on_frame_ready(self, image, event, metadata=None):
        """Runs on the MDA engine thread.

        Signature is compatible with pymmcore-plus' ``frameReady`` signal,
        which emits ``(image, event, metadata)``; older versions emit
        only ``(image, event)`` — the ``metadata=None`` default handles
        both.
        """
        try:
            meta = getattr(event, 'metadata', None) or {}
            pos_name = meta.get('pos_name')
            tp = meta.get('tp', 0)
            z_idx = meta.get('z_idx', 0)
            n_z = meta.get('n_z', 1)
            is_last_scene = bool(meta.get('is_last_scene', False))

            if pos_name is None:
                # Not one of our events — ignore
                return

            # ── Synthetic substitution ──────────────────────────────
            if self.synthetic_source is not None:
                try:
                    synth = self.synthetic_source(
                        float(event.x_pos), float(event.y_pos),
                        float(event.z_pos), pos_name,
                    )
                    if synth is not None:
                        image = synth
                except Exception as e:
                    self.logger.warning(
                        f"synthetic_source raised for {pos_name} "
                        f"tp={tp}: {e} — falling back to camera image"
                    )

            # ── Save + enqueue only the "representative" slice ──────
            # For a z-stack we save every slice under a slice-suffixed
            # name, but only the middle slice is pushed to the tracker
            # (the tracker consumes 2-D frames).
            save_dir = os.path.join(self.dirpath, pos_name)
            os.makedirs(save_dir, exist_ok=True)
            with self._config_lock:
                channel_preset_snapshot = self.channel_preset
            if n_z > 1:
                tif_name = f"t{tp:04d}_{channel_preset_snapshot}_z{z_idx:03d}.tif"
            else:
                tif_name = f"t{tp:04d}_{channel_preset_snapshot}.tif"
            tif_path = os.path.join(save_dir, tif_name)
            try:
                tifffile.imwrite(tif_path, np.asarray(image))
            except Exception as e:
                self.logger.warning(f"Could not save {tif_path}: {e}")

            tracker_slice = (z_idx == n_z // 2)
            if tracker_slice:
                self._image_queue.put((np.asarray(image), tp, pos_name))
                self.logger.info(
                    f"[{pos_name}] tp={tp} frame queued -> tracker "
                    f"(saved {tif_name})"
                )

            # ── Bug 1 prevention: cycle advance ─────────────────────
            # When the LAST scene of a cycle has emitted its LAST z-slice,
            # UNCONDITIONALLY enqueue the next timepoint.  We do NOT gate
            # this on drift, so zero-drift runs complete.  We use a lock
            # to keep the "scenes_seen" bookkeeping consistent when
            # z-stack events for the same tp interleave.
            do_enqueue = False
            next_tp = tp + 1
            if is_last_scene and z_idx == n_z - 1:
                with self._cycle_lock:
                    self._scenes_seen_this_tp.add(pos_name)
                    if len(self._scenes_seen_this_tp) >= len(self.position_names):
                        self._scenes_seen_this_tp.clear()
                        self._current_tp = next_tp
                        do_enqueue = True
            elif not is_last_scene and z_idx == n_z - 1:
                with self._cycle_lock:
                    self._scenes_seen_this_tp.add(pos_name)

            if do_enqueue:
                # Enqueue outside cycle lock — Bug 1: UNCONDITIONAL.
                self._enqueue_timepoint(next_tp)

        except Exception as e:
            self.logger.error(f"frameReady handler crashed: {e}",
                              exc_info=True)

    # ------------------------------------------------------------------
    def wait_for_image(self, timeout_ms=1000):
        try:
            return self._image_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None, None, None

    # ------------------------------------------------------------------
    def wait_for_pause(self, timeout_ms=1000):
        """Alias for wait_for_image — LS1 contract compatibility."""
        return self.wait_for_image(timeout_ms)

    # ------------------------------------------------------------------
    def relative_move(self, position_name, shift_x, shift_y, shift_z):
        """Bug 2 prevention: mutate ``_cum_drift`` synchronously under a
        lock and return.  The updated value is applied by
        ``_enqueue_timepoint`` when the NEXT event for that scene is
        built.  We do NOT touch the stage directly — the MDA engine
        drives the stage via ``x_pos``/``y_pos``/``z_pos`` on each event.
        """
        # Soft clamp
        def _clamp(v, limit, axis):
            if abs(v) > limit:
                self.logger.warning(
                    f"[{position_name}] {axis} shift {v:.1f} µm "
                    f"exceeds limit {limit:.0f} µm — clamped"
                )
                return math.copysign(limit, v)
            return v

        sx = _clamp(float(shift_x), self.max_xy_um, 'X')
        sy = _clamp(float(shift_y), self.max_xy_um, 'Y')
        sz = _clamp(float(shift_z), self.max_z_um,  'Z')

        with self._drift_lock:
            d = self._cum_drift.setdefault(position_name, [0.0, 0.0, 0.0])
            d[0] += sx
            d[1] += sy
            d[2] += sz
            snapshot = tuple(d)

        self.logger.info(
            f"[{position_name}] cum_drift updated: shift=({sx:+.2f},"
            f"{sy:+.2f},{sz:+.2f}) µm  cumulative=({snapshot[0]:+.2f},"
            f"{snapshot[1]:+.2f},{snapshot[2]:+.2f}) µm"
        )

    # ------------------------------------------------------------------
    def refresh_filename(self, pos_name):
        """Called by TrackingRunner after the ROI dashboard rewrote the
        positions JSON.  Re-read the entry and update the channel /
        baseline if they changed.  Logs a warning when ``xyz_um`` moved
        mid-run (unusual but valid — e.g. the user manually re-centered)."""
        cfg = self.positions_config.get(pos_name, {})
        new_channel = cfg.get('channel_preset')
        if new_channel and new_channel != self.channel_preset:
            self.logger.info(
                f"[{pos_name}] channel preset updated: "
                f"{self.channel_preset!r} → {new_channel!r}"
            )
            with self._config_lock:
                self.channel_preset = new_channel

        new_xyz = cfg.get('xyz_um')
        if new_xyz is not None:
            new_tuple = (float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]))
            with self._config_lock:
                old_tuple = self._baseline_um.get(pos_name)
            if old_tuple is not None and new_tuple != old_tuple:
                self.logger.warning(
                    f"[{pos_name}] xyz_um baseline changed mid-run: "
                    f"{old_tuple} → {new_tuple} (unusual but honored)"
                )
                with self._config_lock:
                    self._baseline_um[pos_name] = new_tuple

    # ------------------------------------------------------------------
    # LS1-contract compatibility stubs — Micro-Manager doesn't expose a
    # "pause between positions" concept the way Viventis does, so these
    # are no-ops.  They exist so callers can treat any interface the
    # same.
    def pause_after_position(self):    pass
    def no_pause_after_position(self): pass
    def continue_from_pause(self):     pass

    # ------------------------------------------------------------------
    def disconnect(self):
        self._stop_event.set()
        # Poison the MDA queue so the event iterator returns and the
        # engine exits cleanly.
        try:
            self._mda_queue.put_nowait(None)
        except Exception:
            pass

        if self.mmc is not None:
            try:
                self.mmc.mda.events.frameReady.disconnect(self._on_frame_ready)
            except Exception:
                pass
            try:
                if self._mda_future is not None:
                    # pymmcore-plus MDA futures expose .cancel() / .join()
                    if hasattr(self._mda_future, 'cancel'):
                        self._mda_future.cancel()
                    if hasattr(self._mda_future, 'join'):
                        self._mda_future.join(timeout=5)
            except Exception as e:
                self.logger.warning(f"Error stopping MDA: {e}")

        # Final per-position drift summary — mirrors Files interface
        with self._drift_lock:
            drift_snapshot = dict(self._cum_drift)
        if any(any(v != 0.0 for v in d) for d in drift_snapshot.values()):
            self.logger.info("Final cumulative drift per position:")
            for pos_name, d in drift_snapshot.items():
                self.logger.info(
                    f"  [{pos_name}] cumulative=({d[0]:+.2f}, "
                    f"{d[1]:+.2f}, {d[2]:+.2f}) µm"
                )

    def stop(self):
        self.stop_requested = True
        self.disconnect()
