import numpy as np
from pathlib import Path
import json
import pickle
from ..logger.logger import init_logger
import tifffile
import dataclasses
from dataclasses import asdict
from ..tracker.status import TrackingState, ReturnStatus


class TrackingRunner() :
    def __init__(
            self,
            positions_config,
            position_tracker,
            microscope_interface,
            image_reader,
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
        self.tracker_class = position_tracker
        self.timeout_ms = runner_params["timeout_ms"]
        self.positions_config = positions_config
        if self.positions_config == {} :
            raise ValueError(f"position_config must not be empty : {self.positions_config}")
        # Lookup table to get positions_config key from the position name
        self.position_name_to_PosSetting = {config["Position"]: name for name, config in self.positions_config.items()}
        self.log_dir_name = runner_params["log_dir_name"]
        self.log = runner_params["log"]
        self.scaling_factor = roi_tracker_params["scaling_factor"]
        self.position_names = [config['Position'] for config in positions_config.values()]
        self.tracking_state_dict = {k:TrackingState.TRACKING_ON for k in self.position_names}
        self.trackers = {}
        self.stop_requested = False
        self.dirpath = Path(dirpath)
        self.roi_tracker_params = roi_tracker_params
        self.position_tracker_params = position_tracker_params
        self.reader = image_reader(dirpath=self.dirpath, log=self.log)

        for config_name in positions_config.keys() :
            with open(self.dirpath / config_name / runner_params["log_dir_name"] / "tracking_parameters.json", 'w') as json_file:
                to_save = dict()
                to_save['scaling_factor'] = self.scaling_factor
                json.dump(to_save, json_file, indent=4)
                
        # Set default logger
        self.logger = init_logger("TrackingRunner")
        self.to_save = {}

    def run_LS1(self) :

        # Initialize trackers before main loop
        self.logger.info(f"Initializing trackers")
        for position_name in self.position_names :
            self.initialize_tracker(position_name)

        # Enable pause after position after initialisation to avoid timing problems
        self.microscope.pause_after_position()
        
        self.logger.info(f"Main tracking loop")
        timeout = False
        while not self.stop_requested :
            
            if self.log and not timeout:
                self.logger.info(f"Waiting for the next timepoint and position")
            position_name, time_point, timeout = self.microscope.wait_for_pause(timeout_ms=self.timeout_ms)
            if timeout : 
                continue
            if position_name not in self.position_names :
                self.microscope.continue_from_pause()
                continue
            
            if self.log :
                self.logger.info(f"Timepoint {time_point}, Position {position_name}")
            
            # Read image
            PosSetting = self.position_name_to_PosSetting[position_name]
            settings = self.positions_config[PosSetting]['Settings']
            channel = self.positions_config[PosSetting]['channel']
            image = self.reader.read_image(position_name, settings, channel, time_point)
            if image is None :
                self.microscope.continue_from_pause()
                self.stop()
                continue
            
            # If tracker for position do not exist, skip
            if position_name not in self.trackers.keys() :
                self.microscope.continue_from_pause()
            else :
                if self.tracking_state_dict[position_name] != TrackingState.TRACKING_OFF :
                    self.track_and_correct(position_name, time_point, image)
                self.microscope.continue_from_pause()

        self.microscope.no_pause_after_position()
        self.microscope.disconnect()

    def initialize_tracker(self, position_name, image=None, time_point=None) :
        PosSetting = self.position_name_to_PosSetting[position_name]
        use_detection = self.positions_config[PosSetting]['use_detection']
        # Append starting point
        if time_point :
            with open(self.dirpath / PosSetting / self.log_dir_name / "tracking_parameters.json", "r") as json_file:
                to_save = json.load(json_file)
            to_save['starting_time_point'] = time_point
            with open(self.dirpath / PosSetting / self.log_dir_name / "tracking_parameters.json", 'w') as json_file:
                json.dump(to_save, json_file, indent=4)
        # Initialize tracker
        rois = self.positions_config[PosSetting]['RoIs']
        tracker = self.tracker_class(
            first_frame=image,
            rois=rois,
            log=self.log,
            use_detection=use_detection,
            position_name=position_name,
            roi_tracker_params=self.roi_tracker_params,
            position_tracker_params=self.position_tracker_params,
        )
        self.trackers[position_name] = tracker

    def track_and_correct(self, position_name, time_point, image) :
        tracker = self.trackers[position_name]
        shift_um, tracking_state = tracker.compute_shift_um(image)
        # Update tracking state
        self.tracking_state_dict[position_name] = tracking_state

        if tracking_state != TrackingState.TRACKING_OFF :
            self.microscope.relative_move(position_name, shift_um.x, shift_um.y, shift_um.z)

            if self.log:
                PosSetting = self.position_name_to_PosSetting[position_name]
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
                self.to_save[position_name][str(time_point)]['tracks_id'] = self.make_json_serializable(len(tracks)-1)
                self.to_save[position_name][str(time_point)]['scaling_factor'] = self.make_json_serializable(self.scaling_factor)
                with open(log_dir / f"logs.json", 'w') as file :
                    json.dump(self.to_save[position_name], file, indent=4)
                    file.close()
                with open(log_dir / f"tracks.pkl", 'wb') as file :
                    pickle.dump(tracks, file)
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
        self.stop_requested = True
        # self.microscope.no_pause_after_position()