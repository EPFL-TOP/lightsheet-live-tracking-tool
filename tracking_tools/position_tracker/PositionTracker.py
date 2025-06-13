import numpy as np
from ..logger.logger import init_logger
from dataclasses import asdict
from ..utils.structures import ROI, Position3D, Shift3D
import copy
from ..tracker.status import TrackingState, ReturnStatus


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

        # If roi is a list, take the first element
        if isinstance(rois, list) :
            self.roi = rois[0]

        self.shape = first_frame.shape
        self.scaling_factor = roi_tracker_params["scaling_factor"]
        self.position_name = position_name
        self.pixel_size_xy = position_tracker_params["pixel_size_xy"]
        self.pixel_size_z = position_tracker_params["pixel_size_z"]
        self.log = log

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
        self.tracking_state = self.base_tracker.tracking_state
        self.tracking_state_list = [self.tracking_state]
        self.positions = [Position3D(x=self.roi['x'], y=self.roi['y'], z=self.shape[0]//2)]
        self.ref_position = Position3D(x=self.roi['x'], y=self.roi['y'], z=self.shape[0]//2)
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
        # Y Axis is inverted for the microscope stage coordinate
        shift_x_um = self.convert_px_um(shift_px.x, self.pixel_size_xy, invert=False)
        shift_y_um = self.convert_px_um(shift_px.y, self.pixel_size_xy, invert=True)
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
    

