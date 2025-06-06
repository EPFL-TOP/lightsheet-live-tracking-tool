import numpy as np
from ..utils.tracking_utils import *
from scipy.ndimage import center_of_mass
from ..logger.logger import init_logger
from ..utils.structures import ROI, Position2D, Position3D, Shift2D
import copy
from .status import ReturnStatus, TrackingState


class SingleRoIBaseTracker_v2 :
    def __init__(
            self,
            first_frame,
            roi,
            border,
            scaling_factor,
            serverkit,
            server_addresses,
            window_length,
            grid_size,
            num_points,
            base_kernel_size_xy,
            kernel_size_z,
            num_neighbours,
            enable_sanity_check,
            sanity_threshold,
            log,
            use_detection,
    ) :
        # Takes raw frames as input, handle processing (norm, scaling, projection), manage the sliding window
        # and manage the computation of the RoI position across frames
        self.border = border
        self.scaling_factor = scaling_factor
        self.window_length = window_length
        self.grid_size = grid_size
        self.num_neighbours = num_neighbours
        self.num_points = num_points
        self.kernel_size_xy = base_kernel_size_xy // (2**scaling_factor)
        self.kernel_size_z = kernel_size_z
        self.enable_sanity_check = enable_sanity_check
        self.sanity_threshold = sanity_threshold
        self.log = log
        self.current_frame = self._downsample(first_frame)
        self.shape = self.current_frame.shape
        self.current_frame_proj = np.max(self.current_frame, axis=0)
        self.use_detection = use_detection
        # Convert to ROI dataclass
        roi = ROI(**roi)
        # Downscale RoI
        self.roi_init = self._scale_roi(roi, down=True)
        self.center_point_init = self.roi_init.to_position3D(z=self.shape[0]//2)
        # Set default logger
        self.logger = init_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")
        param_lines = [f"  {k}: {v}" for k, v in vars(self).items()
                if not k.startswith('_') and k in ["scaling_factor", "window_length", "grid_size",
                                                    "shape", "kernel_size_xy", "kernel_size_z",
                                                    "use_detection"]]
        param_block = "\n".join(param_lines)
        self.logger.info(f"Initialized a new ROI tracker: \nParameters:\n{param_block}")
        self.count = 1
        # Initialize rolling window
        current_frame_proj = np.max(self.current_frame, axis=0)
        self.window_frames = [current_frame_proj]
        # Initialize queries
        self.queries_init = self._initialize_queries(current_frame_proj, self.roi_init)
        self.tracks = []
        # Initialize tracked points
        self.tracked_points = [copy.copy(self.center_point_init)]
        self.tracking_state = TrackingState.TRACKING_ON
        self.detected = False

        # Initialize model
        self.predictor = self.initialize_predictor(serverkit, server_addresses=server_addresses)
        if self.use_detection :
            self.detector = self.initialize_detector(serverkit, server_addresses=server_addresses)
        self.detection_freq = 1 # take one detection out of 5 to avoid noisy updates
        self.time_since_last_detection = 0 # to keep track of when should the detection be done

        # Initialize 2D RoI List
        self.rois_list = [[self.roi_init]]
        self.predicted_points = [self.roi_init.to_position2D()]
        self.detected_points = [self.roi_init.to_position2D()]
        self.motions_list = []

    def compute_new_position(self, frame) :
        if self.tracking_state != TrackingState.TRACKING_OFF :
            self.count += 1

            frame = self._downsample(frame)
            self.current_frame = frame
            frame_proj = np.max(frame, axis=0)
            self.current_frame_proj = frame_proj

            self.update_rolling_window(frame_proj)

            new_tracks, tracking_state = self.update_tracks()
            self.tracking_state = tracking_state

            if tracking_state == TrackingState.TRACKING_OFF :
                return Position3D.invalid(), self.tracking_state # Placeholder return
            
            self.tracks.append([new_tracks])
            last_tracked_point = self.tracked_points[-1]

            # Sanity check, still needs to decide what to do in case of failed check
            if self.enable_sanity_check :
                if not self.perform_sanity_check(new_tracks) :
                    self.logger.warning(f'Inconsistent tracking detected')

            if tracking_state == TrackingState.WAIT_FOR_NEXT_TIME_POINT :
                return Position3D.invalid(), self.tracking_state # Placeholder
            
            else :
                # position_yx_predicted = self.compute_predicted_position_yx(new_tracks, last_tracked_point.to_position2D(), num_neighbours=self.num_neighbours)
                position_yx_predicted = self.compute_predicted_position_yx_ls(new_tracks, last_tracked_point)
                last_roi = self._get_past_roi(index=-1)
                roi_predicted = ROI(
                    x=position_yx_predicted.x,
                    y=position_yx_predicted.y,
                    height=last_roi.height,
                    width=last_roi.width,
                    order=1
                )
                self.predicted_points.append(position_yx_predicted)
                if self.use_detection :
                    roi_detected, score, returnStatus = self.compute_detected_roi(self.current_frame_proj)
                    self.detected_points.append(roi_detected.to_position2D())
                else :
                    self.detected = False
                    roi_detected = ROI.invalid(order=1) # Placeholder
                    score = 0                           # Placeholder
                position_yx, new_roi = self.fuse_positions_yx(roi_predicted, roi_detected, score)
                self.rois_list.append([new_roi])
                position_z = self.compute_new_position_z(new_roi)
                new_position = position_yx.to_position3D(z=position_z)
            self.tracked_points.append(new_position)
            return self._scale_position(new_position, axis='xy', down=False), self.tracking_state
        else :
            return Position3D.invalid(), self.tracking_state # Placeholder return
    
    def update_rolling_window(self, frame_proj) :
        self.window_frames.append(frame_proj)
        # Slice the rolling window to always get the last [window_length] frames
        self.window_frames = self.window_frames[-self.window_length:]

    def update_tracks(self) :
        if self.count <= self.window_length :
            queries = self.queries_init
        else :
            # Redefine a region aroud the last tracking point, and add the center point as part of the grid
            roi = self._get_past_roi(index=self.count-self.window_length)
            queries = self._initialize_queries(self.window_frames[0], roi)

        normalized_video = self._normalize_percentile(self.window_frames) 
        pred_tracks_np, returnStatus = self.predictor._run_model(normalized_video, queries)
        tracking_state = self.tracking_state

        if returnStatus == ReturnStatus.SUCCESS :
            return pred_tracks_np, tracking_state
        
        elif returnStatus == ReturnStatus.SERVER_ERROR :
            self.logger.warning(f"Serverkit Tracker failed — Retrying at the next time point. {returnStatus}")
            tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
            return pred_tracks_np, tracking_state
        
        elif returnStatus == ReturnStatus.LOCAL_TRACKER_ERROR :
            self.logger.warning(f"Tracker failed, disabling this position. {returnStatus}")
            tracking_state = TrackingState.TRACKING_OFF
            return np.zeros_like(queries), tracking_state
    
    def compute_predicted_position_yx(self, tracks, last_tracked_point, num_neighbours) :
        # Link the tracked point to the closest point in previous frame
        current_points = tracks[-1]
        previous_points = tracks[-2]
        distances = np.linalg.norm(previous_points[:,::-1] - last_tracked_point.to_array('yx'), axis=1) # tracks are (x, y) so invert there coordinates
        neighbours_indices = np.argsort(distances, axis=0)[:num_neighbours]
        if num_neighbours == 1 :
            shift = current_points[neighbours_indices,::-1] - previous_points[neighbours_indices, ::-1]
        else :
            motions = current_points[neighbours_indices,::-1] - previous_points[neighbours_indices, ::-1]
            self.motions_list.append(motions)
            shift = np.mean(motions, axis=0)
        new_pos_yx = last_tracked_point + Shift2D(x=shift[1], y=shift[0])
        return new_pos_yx
    
    def compute_predicted_position_yx_ls(self, tracks, last_tracked_point) :
        current_points = tracks[-1][:,::-1]  # xy -> yx
        previous_points = tracks[-2][:,::-1]
        motions = current_points - previous_points # yx
        # Fit a model to the vector field using least square regression
        X = np.hstack([previous_points, np.ones((previous_points.shape[0], 1))]) # add intercept (y, x, 1)
        U = motions
        W, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        R = W[:2].T
        x = W[2]
        # Compute the new position
        last_tracked_point = last_tracked_point.to_array('yx') # yx
        new_pos_yx = last_tracked_point + R @ last_tracked_point + x # yx
        new_pos_yx = Position2D(x=new_pos_yx[1], y=new_pos_yx[0])
        return new_pos_yx
    
    def compute_detected_roi(self, frame) :
        prediction, returnStatus = self.detector._run_model(frame)
        scores = prediction['scores']
        if len(scores) > 0 :
            max_score_idx = np.argmax(prediction['scores'])
            score = prediction['scores'][max_score_idx]
            box = prediction['boxes'][max_score_idx]
            xmin, ymin, xmax, ymax = box
            x = xmin + (xmax-xmin)/2
            y = ymin + (ymax-ymin)/2
            height = ymax - ymin
            width = xmax - xmin
            self.detected = True
            return ROI(x=x, y=y, height=height, width=width, order=1), score, returnStatus
        else :
            self.detected = False
            score = 0.0
            return ROI.invalid(order=1), score , ReturnStatus.NO_OP # Placeholder

    
    def fuse_positions_yx(self, roi_predicted, roi_detected, score, containment_threshold=0.4) : ### TODO : Replace hard coded values by class parameters
        fusion_valid = False
        self.time_since_last_detection += 1

        # Check if detection is present, and valid
        if self.detected :
            # containement, size_ratio = self.compute_rois_matching_metrics(position_detected, hws_detected, position_predicted, last_roi['hws'])
            containement, size_ratio = self.compute_rois_matching_metrics(roi_predicted, roi_detected)
            self.logger.info(f"{score}, {containement}, {size_ratio}, {self.time_since_last_detection}")
            if (score > 0.9) and (containement > containment_threshold) and (size_ratio > 0.3) and (self.time_since_last_detection >= self.detection_freq):
                self.logger.info("oui")
                fusion_valid = True
                self.time_since_last_detection = 0
            
        if fusion_valid :
            position_detected = roi_detected.to_position2D()
            position_predicted = roi_predicted.to_position2D()
            fused_pos_yx = self.confidence_weighted_average(position_detected, position_predicted, containement, confidence=score)
            new_roi = ROI(
                x=fused_pos_yx.x,
                y=fused_pos_yx.y,
                height=roi_detected.height,
                width=roi_detected.width,
                order=1
            )
        else :
            fused_pos_yx = roi_predicted.to_position2D()
            new_roi = roi_predicted
        return fused_pos_yx, new_roi


    def compute_new_position_z(self, roi) :
        D, H, W = self.current_frame.shape
        position_yx = roi.to_position2D()
        hws = [roi.height / 2, roi.width / 2]
        center_point_z_tracking = position_yx.to_position3D(z=D//2).to_array(order='zyx')
        hws_z_tracking = np.array([D//2, *hws])
        frame_cropped = crop_image(self.current_frame, center_point_z_tracking, hws_z_tracking)
        frame_cropped_filtered = filter_image(frame_cropped, median_kernel=0, gaussian_kernel_xy=self.kernel_size_xy, gaussian_kernel_z=self.kernel_size_z)
        binary = threshold_image(frame_cropped_filtered)
        # Compute center of mass in z
        com_z = center_of_mass(binary)[0]
        return com_z

    def _initialize_queries(self, frame, roi) : 
        center_point = roi.to_position2D().to_array(order='yx')
        hws = [roi.height / 2, roi.width / 2]
        if self.border:
            points = generate_border_points_from_region(frame, center_point, hws, kernel_size=self.kernel_size_xy, num_points=self.num_points)
        else:
            points = generate_uniform_grid_in_region(frame, center_point, hws, grid_size=self.grid_size, gaussian_kernel=self.kernel_size_xy)
        if len(points) == 0:
            self.logger.warning(f"No query points generated for RoI.")
            points = np.empty((0, 2))  # Ensure it stays consistent
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
        return points
    
    def _normalize_percentile(self, video_batch) :
        q1, q99 = np.quantile(video_batch, [0.01, 0.99])
        value_range = q99 - q1
        normalized_video = np.clip((video_batch - q1) / value_range, 0, 1)
        normalized_video = (normalized_video * 255).astype(np.uint8)
        return normalized_video

    def _downsample(self, image) :
        image = image[:, ::2**self.scaling_factor, ::2**self.scaling_factor]
        return image
    
    def _scale_roi(self, roi, down) :
        factor = 1 / (2 ** self.scaling_factor) if down else 2 ** self.scaling_factor
        return copy.copy(roi) * factor
    
    def _scale_position(self, position, axis='xy', down=True):
        factor = 2 ** self.scaling_factor
        scale = (lambda v: v / factor) if down else (lambda v: v * factor)
        # Copy fields manually, scaling only specified axes
        kwargs = {}
        for field in position.__dataclass_fields__:
            value = getattr(position, field)
            if field in axis:
                value = scale(value)
            kwargs[field] = value
        return type(position)(**kwargs)
    
    def perform_sanity_check(self, tracks) :
        current_points = tracks[-1]
        previous_points = tracks[-2]
        return sanity_check_pairwise_matrix(previous_points, current_points, threshold=self.sanity_threshold)
    
    def initialize_predictor(self, serverkit, server_addresses) :
        if serverkit :
            from .predictor import TrackerSK
            predictor = TrackerSK(server_addresses=server_addresses)
            predictor.connect()
        else :
            from .predictor import Tracker
            predictor = Tracker()
        return predictor
    
    def initialize_detector(self, serverkit, server_addresses) :
        if serverkit :
            from .detector import DetectorSK
            detector = DetectorSK(server_addresses=server_addresses)
            detector.connect()
        else :
            from .detector import Detector
            detector = Detector(model_path="/home/pili/Desktop/fasterrcnn_model/tail_detection_model_v2_augmented_100ep.pth", device="cuda")
        return detector


    def compute_rois_matching_metrics(self, roi1, roi2):
        center1 = np.array([roi1.y, roi1.x])
        hws1 = np.array([roi1.height / 2, roi1.width / 2])
        center2 = np.array([roi2.y, roi2.x])
        hws2 = np.array([roi2.height / 2, roi2.width / 2])

        min1 = center1 - hws1
        max1 = center1 + hws1
        min2 = center2 - hws2
        max2 = center2 + hws2

        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        inter_dims = np.maximum(inter_max - inter_min, 0)

        intersection = np.prod(inter_dims)
        area1 = np.prod(max1 - min1)
        area2 = np.prod(max2 - min2)

        containment = intersection / min(area1, area2) if min(area1, area2) > 0 else 0
        size_ratio = min(area1, area2) / max(area1, area2)

        return containment, size_ratio

    def confidence_weighted_average(self, detection_pos, prediction_pos, containment, confidence, k=5.0, c0=0.4) :
        misalignement = 1 - containment
        alpha = 1 / (1 + np.exp(-k * (misalignement - c0)))
        fused_pos = alpha * detection_pos + (1 - alpha) * prediction_pos
        return fused_pos
    
    def _get_past_roi(self, index) :
        return copy.copy(self.rois_list[index][0])
    
    def fill_placeholders(self) :
        self.tracked_points.append(self.tracked_points[-1])
        self.predicted_points.append(Position2D.invalid())
        self.detected_points.append(Position2D.invalid())
        self.rois_list.append([self._get_past_roi(-1)])
        self.tracks.append([])
        self.logger.info("Filled placeholder data for tracker.")



def sanity_check_pairwise_matrix(previous_points, current_points, threshold=4.0) :

    from scipy.spatial.distance import pdist, squareform

    # Compute pairwise distance
    D_prev = squareform(pdist(previous_points))
    D_curr = squareform(pdist(current_points))

    # Compute error
    error = np.mean(np.abs(D_curr - D_prev))

    # Decision based on threshold
    return error < threshold
    