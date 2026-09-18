import numpy as np
from pathlib import Path
import os
import json
import pickle
from ..logger.logger import init_logger
import tifffile
import dataclasses
from dataclasses import asdict
from ..tracker.status import TrackingState, ReturnStatus
from ..position_tracker.PositionTracker import PositionTrackerMultiROI


class TrackingRunner() :
    def __init__(
            self,
            positions_config,
            microscope_interface,
            dirpath,
            runner_params,
            roi_tracker_params,
            position_tracker_params,
    ) :
        """High level tracking

        Args:
            positions_config (dict): Positions and their parameters : {PositionName1:((center_point), (hws)), PositionName2....}
            microscope_interface (MicroscopeInterface, optional): MicroscopeInterface instance, makes the communication with the microscope. Defaults to None.
            log_dir (str, optional): Logs directory. Defaults to None.
            logging (bool, optional): _description_. Defaults to False.
        """
        self.microscope = microscope_interface
        self.tracker_class = PositionTrackerMultiROI
        self.timeout_ms = runner_params["timeout_ms"]
        self.positions_config = positions_config
        if self.positions_config == {} :
            raise ValueError(f"position_config must not be empty : {self.positions_config}")
        self.log_dir_name = runner_params["log_dir_name"]
        self.log = runner_params["log"]
        self.tracking_state_dict = {k:TrackingState.TRACKING_ON for k in self.positions_config.keys()}
        self.trackers = {}
        self.stop_requested = False
        # Tracks saved by each previous tracker instance (keyed by position).
        # Populated in reinitialize_tracker so that tracks.pkl accumulates
        # across ROI re-initializations.
        self._tracks_history = {}   # position_name → list of tracks objects
        self._reinit_epoch   = {}   # position_name → int (increments each reinit)
        self.dirpath = Path(dirpath)
        self.roi_tracker_params = roi_tracker_params
        self.position_tracker_params = position_tracker_params
                
        # Set default logger
        self.logger = init_logger(self.__class__.__name__)
        self.to_save = {}
        # Populated by run_zeiss(); positions whose ROI JSON was modified at runtime
        self._pending_reinit = set()
        self._roi_observer   = None
        # Positions we have already explained away as untracked.
        self._untracked_reported = set()

    # ------------------------------------
    # Tracking Loops
    # -----------------------------------
    
    def run_LS1(self) :
        self._pending_reinit = set()
        # Initialize trackers before main loop
        self.initialize_ready_trackers()

        # Enable pause after position after initialisation to avoid timing problems
        self._start_roi_watcher()
        self.microscope.pause_after_position()
        self.logger.info(f"Main tracking loop")

        
        while not self.stop_requested :
            # Wait for a new image
                # Reinitialize any trackers whose ROI file changed since last frame
            for pos in list(self._pending_reinit):
                self.reinitialize_tracker(pos)
            self._pending_reinit.clear()

            image, time_point, position_name = self.microscope.wait_for_image(timeout_ms=self.timeout_ms)

            # If tracker for position do not exist, skip
            if position_name not in self.trackers.keys() :
                self.microscope.continue_from_pause()
            else :
                if self.tracking_state_dict[position_name] != TrackingState.TRACKING_OFF :
                    self.track_and_correct(position_name, time_point, image)
                self.microscope.continue_from_pause()

        self.microscope.no_pause_after_position()
        self._stop_roi_watcher()
        self.microscope.disconnect()
        

    def run_general(self) :

        
        # Initialize trackers before main loop
        self.initialize_ready_trackers()
        
        self.logger.info(f"Main tracking loop")
        while not self.stop_requested :
            # Get new image
            image, time_point, position_name = self.microscope.wait_for_image(timeout_ms=self.timeout_ms)

            # If tracker for position do not exist, skip
            if position_name not in self.trackers.keys() :
                continue
            else :
                if self.tracking_state_dict[position_name] != TrackingState.TRACKING_OFF :
                    self.track_and_correct(position_name, time_point, image)

    def initialize_tracker(self, position_name, image=None, time_point=None) :
        PosSetting = position_name
        use_detection = self.positions_config[PosSetting]['detection']
        tracking_mode = self.positions_config[PosSetting]["tracking_mode"]
        scaling_factor = self.positions_config[PosSetting]["scaling_factor"]
        blur_factor = self.positions_config[PosSetting]["blur_factor"]
        grid_size = self.positions_config[PosSetting]["grid_size"]
        mask_kernel_size = self.positions_config[PosSetting]["mask_kernel_size"]


        # Initialize tracker
        rois = self.positions_config[PosSetting]['RoIs']
        # Chosen in the ROI selection dashboard and carried in
        # tracking_RoIs.json. 'otsu' keeps the brighter side (right for
        # fluorescence, wrong for transmitted light); 'texture'
        # segments by local variance and suits both.
        # roi_tracker_params is the dict that flows all the way down to
        # the tracker, so per-position overrides ride along in it rather
        # than as extra keyword arguments the intermediate layers would
        # have to know about.
        roi_tracker_params = dict(self.roi_tracker_params)
        roi_tracker_params["threshold_type"] = (
            self.positions_config[PosSetting].get(
                "threshold_type", "otsu"
            )
        )
        # The base tracker needs it too, so it can skip computing a Z
        # position that would only be discarded.
        roi_tracker_params["tracking_2d"] = bool(
            self.positions_config[PosSetting].get(
                "tracking_2d",
                self.position_tracker_params.get("tracking_2d", False),
            )
        )
        # Same pattern as threshold_type above: the ROI selection
        # dashboard's per-position choice overrides the run-wide
        # default. tracking_2d suppresses the Z CORRECTION; the
        # tracker still receives the whole stack, because XY and Z are
        # measured by different methods (CoTracker on a max
        # projection, centre of mass across planes) and flattening the
        # input would destroy Z tracking irreversibly.
        position_tracker_params = dict(self.position_tracker_params)
        position_tracker_params["tracking_2d"] = bool(
            self.positions_config[PosSetting].get(
                "tracking_2d",
                self.position_tracker_params.get("tracking_2d", False),
            )
        )
        tracker = self.tracker_class(
            first_frame=image,
            rois=rois,
            log=self.log,
            use_detection=use_detection,
            tracking_mode=tracking_mode,
            scaling_factor=scaling_factor,
            blur_factor=blur_factor,
            grid_size=grid_size,
            mask_kernel_size=mask_kernel_size,
            position_name=position_name,
            roi_tracker_params=roi_tracker_params,
            position_tracker_params=position_tracker_params,
        )
        self.trackers[position_name] = tracker

    #: Keys that only exist once tracking_RoIs.json has been merged
    #: into positions_config. Their absence means "imaged, not yet
    #: tracked" - a normal state, not an error.
    _ROI_KEYS = ("RoIs", "detection", "tracking_mode", "scaling_factor",
                 "blur_factor", "grid_size", "mask_kernel_size")

    def is_trackable(self, position_name) :
        """Whether this position has ROIs and can therefore be tracked.

        Positions WITHOUT ROIs are still imaged - you may want to
        record twenty positions and drift-correct three, or decide
        halfway through a run that a fourth is worth following.
        """
        cfg = self.positions_config.get(position_name) or {}
        return all(k in cfg for k in self._ROI_KEYS)

    def tracked_positions(self) :
        return sorted(self.trackers.keys())

    def untracked_positions(self) :
        return sorted(n for n in self.positions_config
                      if n not in self.trackers)

    def initialize_ready_trackers(self) :
        """Build trackers for whichever positions have ROIs so far.

        Called at startup. Positions that acquire an ROI file later are
        picked up by the watcher, so this is a starting point rather
        than a fixed set.
        """
        for position_name in list(self.positions_config) :
            if self.is_trackable(position_name) :
                self.initialize_tracker(position_name)
        tracked = self.tracked_positions()
        waiting = self.untracked_positions()
        self.logger.info(
            f"Tracking {len(tracked)} of {len(self.positions_config)} "
            f"position(s): {', '.join(tracked) if tracked else 'none'}"
        )
        if waiting :
            self.logger.info(
                f"Imaging but not tracking (no ROIs yet): "
                f"{', '.join(waiting)} - save ROIs for any of them and "
                f"they join the tracking loop without a restart"
            )

    def track_and_correct(self, position_name, time_point, image) :
        tracker = self.trackers[position_name]
        shift_um, tracking_state = tracker.compute_shift_um(image)
        
        # Update tracking state
        self.tracking_state_dict[position_name] = tracking_state

        if tracking_state != TrackingState.TRACKING_OFF :
            self.microscope.relative_move(position_name, shift_um.x, shift_um.y, shift_um.z)

            if self.log:
                # PosSetting = self.position_name_to_PosSetting[position_name] ######## CHANGED
                PosSetting = position_name
                log_dir = self.dirpath / PosSetting / self.log_dir_name
                self.logger.info(f"Saving logs in {log_dir}")
                shifts_px = tracker.get_shifts_px()
                shifts_um = tracker.get_shifts_um()
                current_roi = tracker.get_current_rois()
                current_shift_px = shifts_px[-1] if len(shifts_px) > 0 else None
                current_shift_um = shifts_um[-1] if len(shifts_um) > 0 else None
                if position_name not in self.to_save:
                    self.to_save[position_name] = {}
                if str(time_point) not in self.to_save[position_name]:
                    self.to_save[position_name][str(time_point)] = {}
                self.to_save[position_name][str(time_point)]['shift_px'] = self.make_json_serializable(current_shift_px)
                self.to_save[position_name][str(time_point)]['shift_um'] = self.make_json_serializable(current_shift_um)
                self.to_save[position_name][str(time_point)]['roi'] = self.make_json_serializable(current_roi)
                tracks = tracker.get_tracks()
                epoch  = self._reinit_epoch.get(position_name, 0)
                history = self._tracks_history.get(position_name, [])
                # Store a GLOBAL tracks_id: sum of all prior-epoch lengths plus
                # the local index within the current epoch.  This maps directly
                # into the flattened track list stored in tracks.pkl so the
                # visualiser can index without knowing epoch sizes.
                epoch_offset = sum(len(ep) for ep in history)
                self.to_save[position_name][str(time_point)]['tracks_id']      = self.make_json_serializable(epoch_offset + len(tracks) - 1)
                self.to_save[position_name][str(time_point)]['scaling_factor'] = self.make_json_serializable(tracker.scaling_factor)
                self.to_save[position_name][str(time_point)]['reinit_epoch']   = epoch
                with open(log_dir / f"logs.json", 'w') as file :
                    json.dump(self.to_save[position_name], file, indent=4)
                    file.close()
                # tracks.pkl holds a list of per-epoch track objects so that
                # history is preserved across ROI re-initializations.
                # Each entry corresponds to one tracker epoch (epoch index ==
                # list index).  The current epoch is always the last element.
                all_tracks = history + [tracks]
                with open(log_dir / f"tracks.pkl", 'wb') as file :
                    pickle.dump(all_tracks, file)
                    file.close()
                max_proj_dir = log_dir / "max_proj"
                max_proj_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(max_proj_dir / f"t{time_point:04}_max_proj.tif", tracker.get_current_frame())

    
    def make_json_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif dataclasses.is_dataclass(obj):
            return self.make_json_serializable(asdict(obj))
        elif isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.make_json_serializable(v) for v in obj)
        else:
            return obj

    def stop(self) :
        self.microscope.stop()
        # self.microscope.no_pause_after_position()
        self.stop_requested = True
        

    # -----------------------------------------------------------------------
    # Zeiss / ZEN API tracking loop
    # -----------------------------------------------------------------------
 
    def run_zeiss(self):
        """
        Main tracking loop for a Zeiss microscope controlled via the ZEN API
        (or its CZI-file simulation).
 
        Differences from ``run_LS1``:
        * No pause/resume cycle — the inter-timepoint gap in ZEN is long enough
          for the tracker to compute and apply its correction (Option A).
        * A ``watchdog`` file-watcher monitors every ``tracking_RoIs.json``; if
          the user edits an ROI in the Bokeh dashboard while the loop is running,
          the affected tracker is reinitialized at the start of the next iteration.
        * The microscope's ``connect()`` / ``disconnect()`` are called here so
          that the gRPC stream lifetime is tied to the tracking loop.
        """
        self._pending_reinit = set()
 
        self.initialize_ready_trackers()
 
        self._start_roi_watcher()
        self.microscope.connect()
        self.logger.info("Zeiss tracking loop started")
 
        try:
            while not self.stop_requested:
                # Reinitialize any trackers whose ROI file changed since last frame
                for pos in list(self._pending_reinit):
                    self.reinitialize_tracker(pos)
                self._pending_reinit.clear()
 
                image, time_point, position_name = self.microscope.wait_for_image(
                    timeout_ms=self.timeout_ms
                )
 
                if image is None:
                    if getattr(self.microscope, 'stop_requested', False):
                        self.logger.info("Microscope signalled end-of-simulation — stopping loop")
                        break
                    continue

                if position_name not in self.trackers:
                    # Normal: this position is being imaged but has no
                    # ROIs yet. Say so ONCE - every timepoint would
                    # flood the log on a 20-position run.
                    if position_name not in self._untracked_reported:
                        self._untracked_reported.add(position_name)
                        self.logger.info(
                            f"[{position_name}] imaged but not tracked "
                            f"(no ROIs yet) - save ROIs to start"
                        )
                    continue
 
                if self.tracking_state_dict.get(position_name) != TrackingState.TRACKING_OFF:
                    self.track_and_correct(position_name, time_point, image)
 
        finally:
            self._stop_roi_watcher()
            self.microscope.disconnect()
 
    # -----------------------------------------------------------------------
    # Runtime ROI update support
    # -----------------------------------------------------------------------
 
    def _start_roi_watcher(self):
        """
        Start a ``watchdog`` filesystem observer that watches every
        ``tracking_RoIs.json`` file listed in ``positions_config``.
 
        When a file is modified (e.g. the user saves new ROIs from the Bokeh
        dashboard), the corresponding position name is added to
        ``_pending_reinit`` so the main loop can reinitialize its tracker
        between frames.
 
        If ``watchdog`` is not installed, a warning is logged and the feature
        is silently disabled.
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            self.logger.warning(
                "watchdog not installed — runtime ROI updates disabled. "
                "Install with: pip install watchdog"
            )
            self._roi_observer = None
            return
 
        runner = self
 
        roi_file_to_position = {}
        watch_dirs = set()
        for pos_name, cfg in self.positions_config.items():
            log_dir = cfg.get('log_dir')
            if not log_dir:
                self.logger.warning(
                    f"[{pos_name}] has no log_dir - ROIs saved for it "
                    f"cannot be noticed at runtime"
                )
                continue
            roi_path = os.path.normpath(
                os.path.join(log_dir, 'tracking_RoIs.json')
            )
            roi_file_to_position[roi_path] = pos_name
            # The directory must EXIST before watchdog can watch it,
            # and for an untracked position it may not yet. Creating it
            # here is what lets ROIs drawn mid-run start tracking
            # without a restart - watching the parent instead would
            # fire on every acquired frame.
            watch_dir = os.path.dirname(roi_path)
            try:
                os.makedirs(watch_dir, exist_ok=True)
            except OSError as e:
                self.logger.warning(
                    f"could not create {watch_dir}: {e}"
                )
            watch_dirs.add(watch_dir)
 
        # Normalise keys to lowercase so Windows case differences don't break matching
        roi_file_to_position = {k.lower(): v for k, v in roi_file_to_position.items()}

        class _RoIChangeHandler(FileSystemEventHandler):
            def _handle(self, event):
                if event.is_directory:
                    return
                path = os.path.normpath(event.src_path).lower()
                if path in roi_file_to_position:
                    pos = roi_file_to_position[path]
                    if pos in runner.trackers:
                        runner.logger.info(
                            f"tracking_RoIs.json changed for [{pos}] — "
                            "tracker will be reinitialized"
                        )
                    else:
                        runner.logger.info(
                            f"ROIs saved for [{pos}] — it will JOIN "
                            f"the tracking loop"
                        )
                    runner._pending_reinit.add(pos)

            # On Windows json.dump can trigger on_created (atomic overwrite)
            # instead of on_modified, so handle both.
            def on_modified(self, event):
                self._handle(event)

            def on_created(self, event):
                self._handle(event)
 
        handler = _RoIChangeHandler()
        self._roi_observer = Observer()
        for watch_dir in watch_dirs:
            if os.path.isdir(watch_dir):
                self._roi_observer.schedule(handler, watch_dir, recursive=False)
 
        self._roi_observer.start()
        self.logger.info(
            f"ROI file watcher active for {len(watch_dirs)} director(y/ies)"
        )
 
    def _stop_roi_watcher(self):
        if self._roi_observer is not None:
            self._roi_observer.stop()
            self._roi_observer.join(timeout=3)
            self._roi_observer = None
 
    def reinitialize_tracker(self, position_name):
        """
        Reload ``tracking_RoIs.json`` for *position_name* and recreate its
        tracker with the updated ROI definitions.

        The outgoing tracker's accumulated tracks are moved to
        ``_tracks_history`` so that ``tracks.pkl`` continues to hold all
        epochs and ``logs.json`` entries reference the correct epoch index.

        Called automatically by ``run_zeiss`` / ``run_LS1`` when the
        file-watcher detects a change, but can also be called manually.
        """
        roi_path = os.path.join(
            self.positions_config[position_name]['log_dir'], 'tracking_RoIs.json'
        )
        try:
            with open(roi_path) as f:
                new_config = json.load(f)
            # Merge ALL keys (RoIs, filename, scaling_factor, blur_factor, ...)
            # so that channel changes saved through the ROI dashboard propagate
            # to the tracker and the microscope-interface file watcher.
            for k, v in new_config.items():
                self.positions_config[position_name][k] = v

            # Notify any file-watching microscope interface that the source
            # filename / channel may have changed.
            if hasattr(self.microscope, 'refresh_filename'):
                try:
                    self.microscope.refresh_filename(position_name)
                except Exception as e:
                    self.logger.warning(
                        f"microscope.refresh_filename failed for [{position_name}]: {e}"
                    )

            # Archive the current tracker's tracks before replacing
            # it. Absent for a position JOINING the loop for the first
            # time, which is not a reinitialization at all.
            joining = position_name not in self.trackers
            if not joining:
                old_tracks = self.trackers[position_name].get_tracks()
                self._tracks_history.setdefault(position_name, []).append(old_tracks)
                self._reinit_epoch[position_name] = (
                    self._reinit_epoch.get(position_name, 0) + 1
                )

            if not self.is_trackable(position_name):
                self.logger.warning(
                    f"[{position_name}] ROI file is missing keys "
                    f"{[k for k in self._ROI_KEYS if k not in self.positions_config[position_name]]}"
                    f" - not tracking it"
                )
                return

            self.initialize_tracker(position_name)
            self._untracked_reported.discard(position_name)
            if joining:
                self.logger.info(
                    f"[{position_name}] JOINED the tracking loop with "
                    f"{len(new_config['RoIs'])} ROI(s) - no restart "
                    f"needed"
                )
            else:
                self.logger.info(
                    f"Tracker for [{position_name}] reinitialized "
                    f"(epoch {self._reinit_epoch.get(position_name, 0)}) "
                    f"with {len(new_config['RoIs'])} ROI(s)"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to reinitialize tracker for [{position_name}]: {e}"
            )