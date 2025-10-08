import numpy as np
from ..logger.logger import init_logger
from dataclasses import asdict
from ..utils.structures import ROI, Position3D, Shift3D
import copy
from ..tracker.status import TrackingState, ReturnStatus
from  ..utils.tracking_utils import compute_Fis, sweepLine2D, clamp, prioritized_intersection


class PositionTrackerSingleRoI_v2 :
    def __init__(
        self,
        first_frame,
        rois,
        log,
        use_detection,
        position_name,
        roi_tracker_params,
        position_tracker_params,
    ) :
        """Tracks a position in a streaming video, shift cpmputation and unit conversion

        Args:
            first_frame (np.ndarray): Initialization frame
            rois (list): list of ROIs to tracks. The current implementation only supports one ROI
            log (Bool): Log 
            use_detection (Bool): Use detection nad sensor fusion
            position_name (String): Position name
            roi_tracker_params (dict): Dict containing the parameters of the ROI Tracker
            position_tracker_params (dict): Dict contianing the parameters of the Position Tracker
        """

        # If roi is a list, take the first element
        if isinstance(rois, list) :
            self.roi = rois[0]

        self.scaling_factor = roi_tracker_params["scaling_factor"]
        self.position_name = position_name
        self.pixel_size_xy = position_tracker_params["pixel_size_xy"]
        self.pixel_size_z = position_tracker_params["pixel_size_z"]
        self.log = log

        if first_frame :
            self.shape = first_frame.shape
            self.positions = [Position3D(x=self.roi['x'], y=self.roi['y'], z=self.shape[0]//2)]
            self.ref_position = Position3D(x=self.roi['x'], y=self.roi['y'], z=self.shape[0]//2)
            self.tracking_state = self.base_tracker.tracking_state
        else :
            self.shape = None
            self.positions = None
            self.ref_position = None
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT

        # Initialize Tracker
        base_tracker_params = {k:v for k, v in roi_tracker_params.items() if k not in ["pixel_size_xy", "pixel_size_z"]}
        base_tracker_params = {
            "first_frame": first_frame,
            "roi": self.roi,
            "use_detection": use_detection,
            "log": log,
            **base_tracker_params
        }
        self.base_tracker = self.initialize_tracker(base_tracker_params)

        # Initialize tracking vars
        self.tracking_state_list = [self.tracking_state]
        self.ref_position_list = [copy.copy(self.ref_position)]
        self.shifts_px = []
        self.shifts_um = []

        # Set default logger
        self.logger = init_logger("PositionTrackerSingleRoI_v2")
        if self.log : 
            param_lines = [f"  {k}: {v}" for k, v in vars(self).items()
                    if not k.startswith('_') and k in ["shape", "scaling_factor", "position_name", "pixel_size_xy",
                                                       "pixel_size_z"]]
            param_block = "\n".join(param_lines)
            self.logger.info(f"Initialized a new position tracker for position: {self.position_name}\nParameters:\n{param_block}")


    def compute_shift_um(self, frame) :
        # if first frame initialize pos_tracker and base_tracker and return 0 shift
        if self.shape == None :
            self.shape = frame.shape
            self.positions = [Position3D(x=self.roi['x'], y=self.roi['y'], z=self.shape[0]//2)]
            self.ref_position = Position3D(x=self.roi['x'], y=self.roi['y'], z=self.shape[0]//2)
            new_positions, tracking_state = self.base_tracker.compute_new_position(frame)
            self.shifts_px.append(Shift3D(x=0, y=0, z=0))
            self.shifts_um.append(Shift3D(x=0, y=0, z=0))
            return Shift3D(x=0, y=0, z=0), TrackingState.WAIT_FOR_NEXT_TIME_POINT

        # Check for frame shape 
        if frame.shape != self.shape :
            self.logger.warning(f"Image shape is not {self.shape}, Skipping image")
            self.shifts_px.append(Shift3D(x=0, y=0, z=0))
            self.shifts_um.append(Shift3D(x=0, y=0, z=0))
            self.ref_position_list.append(copy.copy(self.ref_position))
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
            self.tracking_state_list.append(self.tracking_state)
            self.base_tracker.fill_placeholders()
            return Shift3D(x=0, y=0, z=0), self.tracking_state

        
        new_position, tracking_state = self.base_tracker.compute_new_position(frame)
        self.positions.append(new_position)

        # Update tracking state
        self.tracking_state = tracking_state
        self.tracking_state_list.append(self.tracking_state)

        # If Tracking state is not on, do not move the microscope, return 0 shifts
        if self.tracking_state == TrackingState.WAIT_FOR_NEXT_TIME_POINT :
            self.shifts_px.append(Shift3D(x=0, y=0, z=0))
            self.shifts_um.append(Shift3D(x=0, y=0, z=0))
            self.ref_position_list.append(copy.copy(self.ref_position))
            self.base_tracker.fill_placeholders()
            return Shift3D(x=0, y=0, z=0), self.tracking_state
        
        # Get and scale the ROI
        roi = copy.copy(self.base_tracker.rois_list[-1][0])
        roi = roi * 2 ** self.scaling_factor

        # Compute shift in pixels
        shift_px = self.compute_shift_px(new_position, roi)

        # Convert into um
        shift_x_um = self.convert_px_um(shift_px.x, self.pixel_size_xy, invert=False)
        shift_y_um = self.convert_px_um(shift_px.y, self.pixel_size_xy, invert=False)
        shift_z_um = self.convert_px_um(shift_px.z, self.pixel_size_z, invert=False)
        shift_um = Shift3D(x=shift_x_um, y=shift_y_um, z=shift_z_um)
        self.shifts_um.append(shift_um)

        if self.log :
            self.logger.info(f"[{self.position_name}] Real shift [um] shift (z, y, x): {shift_z_um, shift_y_um, shift_x_um}")
        return shift_um, self.tracking_state
       
    
    def compute_shift_px(self, new_position, roi):
        D, H, W = self.shape

        # Compute base shift
        ref_position = self.ref_position
        shift_px = Shift3D.from_positions(new_position, Position3D(x=ref_position.x, y=ref_position.y, z=D//2))

        # Compute ROI bounds after shift
        half_height = roi.height / 2
        half_width = roi.width / 2

        # Projected ROI center after shift
        projected_y = new_position.y - shift_px.y
        projected_x = new_position.x - shift_px.x
        
        # ROI corners in the image after applying shift
        min_y = projected_y - half_height
        max_y = projected_y + half_height
        min_x = projected_x - half_width
        max_x = projected_x + half_width

        # Apply correction if ROI exceeds image boundaries
        dy_correction = 0
        dx_correction = 0
        if min_y < 0:
            dy_correction = min_y
        elif max_y > H:
            dy_correction = max_y - H
        if min_x < 0:
            dx_correction = min_x
        elif max_x > W:
            dx_correction = max_x - W
        self.logger.info(f"[{self.position_name}] Correction needed: dx={dx_correction}, dy={dy_correction}")

        # Apply corrections to shift
        shift_px.y += dy_correction
        shift_px.x += dx_correction
        
        # Log and return
        self.shifts_px.append(shift_px)
        if self.log:
            self.logger.info(
                f"[{self.position_name}] Pixel shift: "
                f"{shift_px}"
            )
        return shift_px

    def initialize_tracker(self, params) :
        from ..tracker.BaseTracker import SingleRoIBaseTracker_v2
        base_tracker = SingleRoIBaseTracker_v2(**params)
        return base_tracker

    def get_tracks(self) :
        return self.base_tracker.tracks
    
    def get_shifts_um(self) :
        return self.shifts_um

    def get_shifts_px(self) :
        return self.shifts_px
    
    def get_current_frame(self) :
        return self.base_tracker.current_frame_proj
    
    def get_current_rois(self) :
        # Convert centerpoint, hws to anchor point, full size
        return self.base_tracker.rois_list[-1]

    @staticmethod
    def convert_px_um(shift_px, pixel_size, invert) :
        if invert :
            shift_um = -shift_px * pixel_size
        else :
            shift_um = shift_px * pixel_size
        return shift_um
    

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

class PositionTrackerMultiROI :
    def __init__(
        self,
        rois,
        first_frame,
        log,
        use_detection,
        position_name,
        roi_tracker_params,
        position_tracker_params,
        tracking_mode = "SingleROI",
    ) :
        """Tracks a position in a streaming video, shift cpmputation and unit conversion

        Args:
            first_frame (np.ndarray): Initialization frame
            rois (list): list of ROIs to tracks. The current implementation only supports one ROI
            log (Bool): Log 
            use_detection (Bool): Use detection nad sensor fusion
            position_name (String): Position name
            roi_tracker_params (dict): Dict containing the parameters of the ROI Tracker
            position_tracker_params (dict): Dict contianing the parameters of the Position Tracker
        """

        self.rois = rois
        self.scaling_factor = roi_tracker_params["scaling_factor"]
        self.position_name = position_name
        self.pixel_size_xy = position_tracker_params["pixel_size_xy"]
        self.pixel_size_z = position_tracker_params["pixel_size_z"]
        self.log = log

        if first_frame :
            self.shape = first_frame.shape
            self.positions = [Position3D(x=self.rois[0]['x'], y=self.rois[0]['y'], z=self.shape[0]//2)]
            self.ref_position = Position3D(x=self.rois[0]['x'], y=self.rois[0]['y'], z=self.shape[0]//2)
            self.tracking_state = self.base_tracker.tracking_state
        else :
            self.shape = None
            self.positions = None
            self.ref_position = None
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT

        self.mode = tracking_mode

        # Initialize Tracker
        # Use detection only if SingleROI mode selected, else set to false by default
        if self.mode != "SingleROI" :
            use_detection = False
        base_tracker_params = {k:v for k, v in roi_tracker_params.items() if k not in ["pixel_size_xy", "pixel_size_z"]}
        base_tracker_params = {
            "first_frame": first_frame,
            "rois": self.rois,
            "use_detection": use_detection,
            "log": log,
            **base_tracker_params
        }
        self.base_tracker = self.initialize_tracker(base_tracker_params)

        # Initialize tracking vars
        self.ref_position_list = [copy.copy(self.ref_position)]
        self.shifts_px = []
        self.shifts_um = []
        self.tracking_state_list = [self.tracking_state]


        # Set default logger
        self.logger = init_logger(self.__class__.__name__)
        if self.log : 
            param_lines = [f"  {k}: {v}" for k, v in vars(self).items()
                    if not k.startswith('_') and k in ["shape", "scaling_factor", "position_name", "pixel_size_xy",
                                                       "pixel_size_z", "mode"]]
            param_block = "\n".join(param_lines)
            self.logger.info(f"Initialized a new position tracker for position: {self.position_name}\nParameters:\n{param_block}")


    def compute_shift_um(self, frame) :
        # If frame is 2D, convert to 3D 
        if frame.ndim == 2 :
            self.logger.info("Converting 2D image to 3D")
            frame = frame[np.newaxis, ...]


        # if first frame initialize pos_tracker and base_tracker and return 0 shift
        if self.shape == None :
            self.shape = frame.shape
            self.positions = [Position3D(x=self.rois[0]['x'], y=self.rois[0]['y'], z=self.shape[0]//2)]
            self.ref_position = Position3D(x=self.rois[0]['x'], y=self.rois[0]['y'], z=self.shape[0]//2)
            new_positions, tracking_state = self.base_tracker.compute_new_positions(frame)
            self.shifts_px.append(Shift3D(x=0, y=0, z=0))
            self.shifts_um.append(Shift3D(x=0, y=0, z=0))
            return Shift3D(x=0, y=0, z=0), TrackingState.WAIT_FOR_NEXT_TIME_POINT

        # Check for frame shape 
        if frame.shape != self.shape :
            self.logger.warning(f"Image shape is not {self.shape}, Skipping image")
            self.shifts_px.append(Shift3D(x=0, y=0, z=0))
            self.shifts_um.append(Shift3D(x=0, y=0, z=0))
            self.ref_position_list.append(copy.copy(self.ref_position))
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
            self.tracking_state_list.append(self.tracking_state)
            self.base_tracker.fill_placeholders()
            return Shift3D(x=0, y=0, z=0), self.tracking_state
        

        
        new_positions, tracking_state = self.base_tracker.compute_new_positions(frame)
        new_position = new_positions[0] ### Only taking the first position
        self.positions.append(new_position)

        # Update tracking state
        self.tracking_state = tracking_state
        self.tracking_state_list.append(self.tracking_state)

        # If Tracking state is not on, do not move the microscope, return 0 shifts
        if self.tracking_state != TrackingState.TRACKING_ON :
            self.shifts_px.append(Shift3D(x=0, y=0, z=0))
            self.shifts_um.append(Shift3D(x=0, y=0, z=0))
            self.ref_position_list.append(copy.copy(self.ref_position))
            self.base_tracker.fill_placeholders()
            return Shift3D(x=0, y=0, z=0), self.tracking_state
    

        # Compute shift in pixels
        shift_px = self.compute_shift_px(new_position, self.base_tracker.rois_list)

        # Convert into um
        shift_x_um = self.convert_px_um(shift_px.x, self.pixel_size_xy, invert=False)
        shift_y_um = self.convert_px_um(shift_px.y, self.pixel_size_xy, invert=False)
        shift_z_um = self.convert_px_um(shift_px.z, self.pixel_size_z, invert=False)
        shift_um = Shift3D(x=shift_x_um, y=shift_y_um, z=shift_z_um)
        self.shifts_um.append(shift_um)

        if self.log :
            self.logger.info(f"[{self.position_name}] Real shift [um] shift (z, y, x): {shift_z_um, shift_y_um, shift_x_um}")
        return shift_um, self.tracking_state
       
    
    def compute_shift_px(self, new_position, rois):
        # Get and scale ROIs
        rois_list_scaled = []
        rois_list = copy.copy(rois[-1])
        for roi in rois_list :
            roi_scaled =  roi * 2 ** self.scaling_factor
            rois_list_scaled.append(roi_scaled)

        if self.mode == "SingleROI" :
            roi = rois_list_scaled[0]
            shift_px = self.shift_single_roi(new_position, roi)

        elif self.mode == "MultiROI_max_rois_non_weighted" :
            shift_px = self.shift_multi_roi_max_based(rois_list_scaled, weighted=False)

        elif self.mode == "MultiROI_max_rois_weighted" :
            shift_px = self.shift_multi_roi_max_based(rois_list_scaled, weighted=True)

        elif self.mode == "MultiROI_priority" :
            shift_px = self.shift_multi_roi_priority_based(rois_list_scaled)

        else :
            raise Exception(f"Tracking mode {self.mode} not supported.")
        
        # Log and return
        self.shifts_px.append(shift_px)
        if self.log:
            self.logger.info(
                f"[{self.position_name}] Pixel shift: "
                f"{shift_px}"
            )
        
        return shift_px
    
    def shift_single_roi(self, new_position, roi) :
        D, H, W = self.shape

        # Compute base shift
        ref_position = self.ref_position
        shift_px = Shift3D.from_positions(new_position, Position3D(x=ref_position.x, y=ref_position.y, z=D//2))

        # Compute ROI bounds after shift
        half_height = roi.height / 2
        half_width = roi.width / 2

        # Projected ROI center after shift
        projected_y = new_position.y - shift_px.y
        projected_x = new_position.x - shift_px.x
        
        # ROI corners in the image after applying shift
        min_y = projected_y - half_height
        max_y = projected_y + half_height
        min_x = projected_x - half_width
        max_x = projected_x + half_width

        # Apply correction if ROI exceeds image boundaries
        dy_correction = 0
        dx_correction = 0
        if min_y < 0:
            dy_correction = min_y
        elif max_y > H:
            dy_correction = max_y - H
        if min_x < 0:
            dx_correction = min_x
        elif max_x > W:
            dx_correction = max_x - W
        self.logger.info(f"[{self.position_name}] Correction needed: dx={dx_correction}, dy={dy_correction}")

        # Apply corrections to shift
        shift_px.y += dy_correction
        shift_px.x += dx_correction
        
        return shift_px
    
    def shift_multi_roi_max_based(self, rois, weighted=False) :
        D, H, W = self.shape
        rois_list = []
        for roi in rois :
            xmin = roi.x - roi.width / 2
            xmax = roi.x + roi.width / 2
            ymax = H - (roi.y - roi.height / 2)
            ymin = H - (roi.y + roi.height / 2)
            rois_list.append((xmin, xmax, ymin, ymax))

        # Create shift feasable sets of the ROIs
        Fis = compute_Fis(H, W, rois_list)
        # Weight the Fis (w = 1 if non weighted mode selected)
        Fis_weighted = []
        for i, (xmin, xmax, ymin, ymax) in enumerate(Fis) :
            if weighted :
                weight = len(Fis) - i
            else :
                weight = 1
            Fis_weighted.append((xmin, xmax, ymin, ymax, weight))
        # Compute the best shift region
        max_count, best_region = sweepLine2D(Fis_weighted)
        # Best shift as the closest point to 0 in the best region (lowest shift magnitude)
        best_shift_x = clamp(0, best_region[0], best_region[1])
        best_shift_y = clamp(0, best_region[2], best_region[3])
        best_shift_z = 0 # No shift in z until a better approach is determined

        shift_px = Shift3D(x=best_shift_x, y=best_shift_y, z=best_shift_z) 

        return shift_px
    
    def shift_multi_roi_priority_based(self, rois) :
        D, H, W = self.shape
        rois_list = []
        for roi in rois :
            xmin = roi.x - roi.width / 2
            xmax = roi.x + roi.width / 2
            ymax = H - (roi.y - roi.height / 2)
            ymin = H - (roi.y + roi.height / 2)
            rois_list.append((xmin, xmax, ymin, ymax)) 

        # Create shift feasable sets of the ROIs
        Fis = compute_Fis(H, W, rois_list)
        # Compute the best shift region
        best_region, selected = prioritized_intersection(Fis)
        # Best shift as the closest point to 0 in the best region (lowest shift magnitude)
        best_shift_x = clamp(0, best_region[0], best_region[1])
        best_shift_y = clamp(0, best_region[2], best_region[3])
        best_shift_z = 0 # No shift in z until a better approach is determined

        shift_px = Shift3D(x=best_shift_x, y=best_shift_y, z=best_shift_z)

        return shift_px

        

    def initialize_tracker(self, params) :
        from ..tracker.BaseTracker import MultiRoIBaseTracker
        base_tracker = MultiRoIBaseTracker(**params)
        return base_tracker

    def get_tracks(self) :
        return self.base_tracker.tracks
    
    def get_shifts_um(self) :
        return self.shifts_um

    def get_shifts_px(self) :
        return self.shifts_px
    
    def get_current_frame(self) :
        return self.base_tracker.current_frame_proj
    
    def get_current_rois(self) :
        # Convert centerpoint, hws to anchor point, full size
        return self.base_tracker.rois_list[-1]

    @staticmethod
    def convert_px_um(shift_px, pixel_size, invert) :
        if invert :
            shift_um = -shift_px * pixel_size
        else :
            shift_um = shift_px * pixel_size
        return shift_um