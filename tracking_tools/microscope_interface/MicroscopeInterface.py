import time
import os
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
            self.pos_to_Channel[position_settings_splitted[0]] = positions_config[pos_name]["filename"].replace(".tif","").split("_")[-1]

        self.PosSettings_to_pos = {v:k for k, v in self.pos_to_PosSettings.items()}
        self.microscope = pymcs.Microscope()
        self.connect()
        self.time_lapse_controller = pymcs.TimeLapseController(self.microscope)
        self.stage_xyz = pymcs.StageXYZ(self.microscope, "STAGE")
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
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=lambda: self._loop.run_until_complete(self._async_main()),
            daemon=True,
            name='ZeissAsyncThread',
        )
        self._thread.start()
        self.logger.info(
            f"Connecting to ZEN API Gateway at {self.address}:{self.port}"
        )
 
    # ------------------------------------------------------------------
    async def _async_main(self):
        asyncio.set_event_loop(self._loop)
        try:
            import grpc
            import grpc.aio
 
            if self.cert_path and os.path.exists(self.cert_path):
                with open(self.cert_path, 'rb') as f:
                    cert_data = f.read()
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=cert_data
                )
            else:
                # Fall back to system CA bundle; may need --insecure for
                # self-signed certs in lab environments.
                credentials = grpc.ssl_channel_credentials()
                self.logger.warning(
                    "No cert_path provided — using system CA bundle. "
                    "Set cert_path to the ZEN Gateway certificate if connection fails."
                )
 
            async with grpc.aio.secure_channel(
                f"{self.address}:{self.port}", credentials
            ) as channel:
                self._channel  = channel
                self._metadata = [("control-token", self.control_token)]
                self.logger.info("Connected to ZEN API Gateway")
                await self._stream_images()
 
        except ImportError:
            self.logger.error(
                "grpcio not installed.  Run: pip install grpcio"
            )
            self._queue.put((None, None, None))
        except Exception as e:
            self.logger.error(f"ZEN API error: {e}", exc_info=True)
            self._queue.put((None, None, None))
 
    # ------------------------------------------------------------------
    async def _stream_images(self):
        """
        Subscribe to the ZEN experiment stream and assemble z-stacks.
 
        The exact import path and request class may need adjustment depending
        on the installed zen_api package version.  Cross-check with:
            OAD/ZEN-API/python_examples/zenapi_streaming.py
        """
        try:
            from zen_api.streaming.v2 import (
                ExperimentStreamingServiceStub,
                MonitorAllExperimentsRequest,   # adjust name if needed
            )
        except ImportError as e:
            self.logger.error(
                f"zen_api package not found or wrong import path: {e}\n"
                "Install the zen_api wheel from the ZEN API distribution."
            )
            self._queue.put((None, None, None))
            return
 
        stub = ExperimentStreamingServiceStub(self._channel)
        # Buffer: {(scene_idx, timepoint_idx): {z_idx: 2-D array}}
        frame_buffer = {}
 
        self.logger.info("ZEN image stream active")
        async for frame in stub.MonitorAllExperiments(
            MonitorAllExperimentsRequest(), metadata=self._metadata
        ):
            if self._stop_event.is_set():
                break
 
            scene_idx = frame.scenes_index
            tp_idx    = frame.time_points_index
            z_idx     = frame.zstack_slices_index
            ch_idx    = frame.channels_index
 
            if ch_idx != self.tracking_channel:
                continue
 
            key = (scene_idx, tp_idx)
            if key not in frame_buffer:
                frame_buffer[key] = {}
 
            arr = np.asarray(frame.pixel_data)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            frame_buffer[key][z_idx] = arr
 
            if len(frame_buffer[key]) >= self._z_slice_count:
                self._process_complete_stack(frame_buffer.pop(key), scene_idx, tp_idx)
                # Drop stale keys to prevent unbounded memory growth
                stale = [(s, t) for (s, t) in list(frame_buffer) if t < tp_idx - 2]
                for k in stale:
                    frame_buffer.pop(k, None)
 
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
 
        max_proj_dir = os.path.join(self.dirpath, position_name, 'max_proj')
        os.makedirs(max_proj_dir, exist_ok=True)
        tif_path = os.path.join(max_proj_dir, f't{tp_idx:04d}.tif')
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
 
        stage_stub = StageServiceStub(self._channel)
        focus_stub = FocusServiceStub(self._channel)
 
        if abs(dx_um) > 1e-3 or abs(dy_um) > 1e-3:
            pos = await stage_stub.GetPosition(
                StageServiceGetPositionRequest(), metadata=self._metadata
            )
            await stage_stub.MoveTo(
                StageServiceMoveToRequest(
                    x=pos.x + dx_um * 1e-6,
                    y=pos.y + dy_um * 1e-6,
                ),
                metadata=self._metadata,
            )
 
        if abs(dz_um) > 1e-3:
            z_pos = await focus_stub.GetPosition(
                FocusServiceGetPositionRequest(), metadata=self._metadata
            )
            # Field name for Z may be .z or .position depending on proto version
            current_z = getattr(z_pos, 'z', None) or getattr(z_pos, 'position', 0.0)
            await focus_stub.MoveToPosition(
                FocusServiceMoveToPositionRequest(
                    position=current_z + dz_um * 1e-6
                ),
                metadata=self._metadata,
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


    
