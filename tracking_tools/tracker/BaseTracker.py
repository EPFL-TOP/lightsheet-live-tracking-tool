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
            scaling_factor,
            serverkit,
            server_addresses,
            window_length,
            grid_size,
            base_kernel_size_xy,
            kernel_size_z,
            log,
            use_detection,
            containment_threshold,
            k,
            c0,
            size_ratio_threshold,
            score_threshold,
            model_path,
    ) :
        """ROI Tracker class. Uses CoTracker and Faster-RCNN to track an ROI in a streaming video.
        The compute_new_position() updates the tracker.

        Args:
            first_frame (np.ndarray): The initialization frame
            roi (dict): Dict conatining "x", "y", "width", "height, and "order" keys. The "order" key is not used, its goal is to extend the class to a multi ROI tracking.
            scaling_factor (int): The image will be downsampled by 2**scaling_factor in the X and Y dimensions.
            serverkit (Bool): Use remote inference via imaging-server-kit
            server_addresses (List): List of the server addresses that the serverkit interface will try to connect to (Tries to connect to the first one, then the second one etc...)
            window_length (int): Length of the sliding window.
            grid_size (int): Number of points in one dimension to create the grid (e.g. a grid size of 40 will generate a grid of 40 * 40 = 1600 uniformly distributed points)
            base_kernel_size_xy (int): Size of the full scale gaussian kernel in x and y for the filtering operations. Will be downscaled by the scaling factor
            kernel_size_z (int): Size of the gaussian kernel in z
            log (Bool): Print logs
            use_detection (Bool): Use the detection model the sensor fusion logic
            containment_threshold (float): Containment threshold for the detection model validation
            k (float): Steepness parameter of the sigmoid function for the sensor fusion weight computation
            c0 (float): Inflection point of the sigmoid function for the sensor fusion weight computation
            size_ratio_threshold (float): Size ratio threshold for the detection model validation
            score_threshold (float): Softmax score threshold for the detection model validation
            model_path (string): Detection model weights path. "default" for the default path (tracking_tools/weights/*.pth) or a custom path.
        """
        # Takes raw frames as input, handle processing (norm, scaling, projection), manage the sliding window
        # and manage the computation of the RoI position across frames
        self.scaling_factor = scaling_factor
        self.window_length = window_length
        self.grid_size = grid_size
        self.kernel_size_xy = base_kernel_size_xy // (2**scaling_factor)
        self.kernel_size_z = kernel_size_z
        self.log = log
        self.use_detection = use_detection
        self.containment_threshold = containment_threshold
        self.k = k
        self.c0 = c0
        self.size_ratio_threshold = size_ratio_threshold
        self.score_threshold = score_threshold
        self.model_path = model_path
        # Convert to ROI dataclass
        roi = ROI(**roi)
        # Downscale RoI
        self.roi_init = roi.scale(scaling_factor=self.scaling_factor, down=True)

        # Initialize rolling window, queries, tracked points
        if first_frame :
            # Initialize window
            self.current_frame = self._downsample(first_frame)
            self.shape = self.current_frame.shape
            self.center_point_init = self.roi_init.to_position3D(z=self.shape[0]//2)
            self.current_frame_proj = np.max(self.current_frame, axis=0)
            self.window_frames = [self.current_frame_proj]
            # Initialize queries
            self.queries_init = self._initialize_queries(self.current_frame_proj, self.roi_init)
            self.tracks = []
            # Initialize tracked points
            self.tracked_points = [copy.copy(self.center_point_init)]
            self.tracking_state = TrackingState.TRACKING_ON
            self.count = 1
        else :
            self.window_frames = []
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
            self.count = 0
       
        self.detected = False

        # Initialize model
        self.predictor = self.initialize_predictor(serverkit, server_addresses=server_addresses)
        if self.use_detection :
            self.detector = self.initialize_detector(serverkit, server_addresses=server_addresses)
        self.time_since_last_detection = 0 # to keep track of when should the detection be done

        # Initialize 2D RoI List
        self.rois_list = [[self.roi_init]]
        self.predicted_points = [self.roi_init.to_position2D()]
        self.detected_points = [self.roi_init.to_position2D()]

        # Set default logger
        self.logger = init_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")
        param_lines = [f"  {k}: {v}" for k, v in vars(self).items()
                if not k.startswith('_') and k in ["scaling_factor", "window_length", "grid_size",
                                                    "shape", "kernel_size_xy", "kernel_size_z",
                                                    "use_detection", "serverkit", "k", "c0", "containment_threshold",
                                                    "score_threshold", "size_ratio_threshold"]]
        param_block = "\n".join(param_lines)
        self.logger.info(f"Initialized a new ROI tracker: \nParameters:\n{param_block}")

    def compute_new_position(self, frame) :
        """Compute the new position of the ROI in the given image.
        Make use of the dataclasses defined in tracking_tools/utils/structures.

        Args:
            frame (np.ndarray): New frame

        Returns:
            tuple: Position3D of the tracked ROI (Full scale), trackingState
        """
        if self.tracking_state != TrackingState.TRACKING_OFF :
            self.count += 1

            # If first frame, process it and return 0 shift
            if self.window_frames == [] :
                # Initialize window
                self.current_frame = self._downsample(frame)
                self.shape = self.current_frame.shape
                self.center_point_init = self.roi_init.to_position3D(z=self.shape[0]//2)
                self.current_frame_proj = np.max(self.current_frame, axis=0)
                self.window_frames = [self.current_frame_proj]
                # Initialize queries
                self.queries_init = self._initialize_queries(self.current_frame_proj, self.roi_init)
                self.tracks = [np.empty((0, 2))]
                # Initialize tracked points
                self.tracked_points = [copy.copy(self.center_point_init)]
                self.tracking_state = TrackingState.TRACKING_ON
                return Position3D.invalid(), TrackingState.WAIT_FOR_NEXT_TIME_POINT # Placeholder return

            # Process input frame
            frame = self._downsample(frame)
            self.current_frame = frame
            frame_proj = np.max(frame, axis=0)
            self.current_frame_proj = frame_proj

            self.update_rolling_window(frame_proj)

            # Compute CoTracker3 tracks
            new_tracks, tracking_state = self.update_tracks()
            self.tracking_state = tracking_state

            if tracking_state == TrackingState.TRACKING_OFF :
                return Position3D.invalid(), self.tracking_state # Placeholder return
            
            self.tracks.append([new_tracks])
            last_tracked_point = self.tracked_points[-1]

            if tracking_state == TrackingState.WAIT_FOR_NEXT_TIME_POINT :
                return Position3D.invalid(), self.tracking_state # Placeholder
            
            else :
                # Compute predicted ROI
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
                # Compute the detected ROI
                if self.use_detection :
                    roi_detected, score, returnStatus = self.compute_detected_roi(self.current_frame_proj)
                    self.detected_points.append(roi_detected.to_position2D())
                else :
                    self.detected = False
                    roi_detected = ROI.invalid(order=1) # Placeholder
                    score = 0                           # Placeholder

                # Fuse positions and compute z position
                position_yx, new_roi = self.fuse_positions_yx(roi_predicted, roi_detected, score)
                self.rois_list.append([new_roi])

                position_z = self.compute_new_position_z(new_roi)

                new_position = position_yx.to_position3D(z=position_z)
            self.tracked_points.append(new_position)
            return new_position.scale(scaling_factor=self.scaling_factor, down=False, axes="xy"), self.tracking_state
        else :
            return Position3D.invalid(), self.tracking_state # Placeholder return
    
    def update_rolling_window(self, frame_proj) :
        self.window_frames.append(frame_proj)
        # Slice the rolling window to always get the last [window_length] frames
        self.window_frames = self.window_frames[-self.window_length:]

    def update_tracks(self) :
        """Generate new query points and run the CoTracker3 model

        Returns:
            tuple: Returns the CoTracker3 predicted tracks and tracking state (One of TRACKING_ON, TRACKING_OFF, WAIT_FOR_NEXT_TIME_POINT)
        """
        if self.count <= self.window_length :
            queries = self.queries_init
        else :
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
    
    
    def compute_predicted_position_yx_ls(self, tracks, last_tracked_point) :
        """Fit a linear regression model to the tracks and predict a position

        Args:
            tracks (_type_): _description_
            last_tracked_point (_type_): _description_

        Returns:
            _type_: _description_
        """
        current_points = tracks[-1][:,::-1]  # xy -> yx
        previous_points = tracks[-2][:,::-1] # xy -> yx
        motions = current_points - previous_points # yx
        # Fit a model to the vector field using least square regression
        X = np.hstack([previous_points, np.ones((previous_points.shape[0], 1))]) # add intercept [y, x, 1]
        U = motions
        W, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        R = W[:2].T
        x = W[2]
        # Compute the new position
        last_tracked_point = last_tracked_point.to_array('yx') # yx
        new_pos_yx = last_tracked_point + R @ last_tracked_point + x # yx, predicted point = point + predicted motion
        new_pos_yx = Position2D(x=new_pos_yx[1], y=new_pos_yx[0])
        return new_pos_yx
    
    def compute_detected_roi(self, frame) :
        """Run the Faster-RCNN model to predict an ROI in a frame 

        Args:
            frame (np.ndarray): Input frame (2D)

        Returns:
            Tuple: Detected ROI, softmax score of the detection, returnStatus of the model
            returnStatus one of SUCESS indicating a detected ROI or NO_OP indicating no detection.
        """
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

    
    def fuse_positions_yx(self, roi_predicted, roi_detected, score) :
        """Validate the detection and fuses the predicted and detected ROIs

        Args:
            roi_predicted (ROI): Prediction ROI, from CoTracker3
            roi_detected (ROI): Detction ROI, from Faster-RCNN
            score (float): softmax score of the detection

        Returns:
            tuple: Fused position xy, fuse ROI.
        """
        fusion_valid = False
        # Check if detection is present, and valid
        if self.detected :
            containement, size_ratio = self.compute_rois_matching_metrics(roi_predicted, roi_detected)
            self.logger.info(f"Detection validation : model score :{score}, containment: {containement}, size_ratio: {size_ratio}")
            # Validation conditions
            if (score > self.score_threshold) and (containement > self.containment_threshold) and (size_ratio > self.size_ratio_threshold):
                fusion_valid = True
                self.logger.info("Valid Detection")
            else :
                self.logger.info("Invalid Detection")
        
        if fusion_valid :
            position_detected = roi_detected.to_position2D()
            position_predicted = roi_predicted.to_position2D()
            fused_pos_yx = self.confidence_weighted_average(position_detected, position_predicted, containement)
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
        """Computes the position in z given an ROI, using the center of mass method

        Args:
            roi (ROI): ROI used to crop the image

        Returns:
            float: center of mass in z
        """
        # Build the bounding box (ROI in xy and full dimension in z)
        D, H, W = self.current_frame.shape
        position_yx = roi.to_position2D()
        hws = [roi.height / 2, roi.width / 2]
        center_point_z_tracking = position_yx.to_position3D(z=D//2).to_array(order='zyx')
        hws_z_tracking = np.array([D//2, *hws])

        # Crop
        frame_cropped = crop_image(self.current_frame, center_point_z_tracking, hws_z_tracking)

        # Filter (Gaussian blurring)
        frame_cropped_filtered = filter_image(frame_cropped, median_kernel=0, gaussian_kernel_xy=self.kernel_size_xy, gaussian_kernel_z=self.kernel_size_z)

        # Binarize
        binary = threshold_image(frame_cropped_filtered)

        # center of mass in z
        com_z = center_of_mass(binary)[0]
        return com_z

    def _initialize_queries(self, frame, roi) : 
        """Generate query points given a frame and an ROI

        Args:
            frame (np.ndarray): Input frame (2D)
            roi (ROI): Input ROI

        Returns:
            np.ndarray: Nx2 array of points coordinates
        """
        # Build bounding box
        center_point = roi.to_position2D().to_array(order='yx')
        hws = [roi.height / 2, roi.width / 2]

        points = generate_uniform_grid_in_region(frame, center_point, hws, grid_size=self.grid_size, gaussian_kernel=self.kernel_size_xy)

        if len(points) == 0:
            self.logger.warning(f"No query points generated for RoI.")
            points = np.empty((0, 2))  # Ensure it stays consistent
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT

        return points
    
    def _normalize_percentile(self, video_batch) :
        """Normalize a video using the 1st and 99th percentiles. Converts to uint8

        Args:
            video_batch (np.ndarray): video N*H*W

        Returns:
            np.nadarray: Normalized video
        """
        q1, q99 = np.quantile(video_batch, [0.01, 0.99])
        value_range = q99 - q1
        normalized_video = np.clip((video_batch - q1) / value_range, 0, 1)
        normalized_video = (normalized_video * 255).astype(np.uint8)
        return normalized_video

    def _downsample(self, image) :
        image = image[:, ::2**self.scaling_factor, ::2**self.scaling_factor]
        return image
    
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
            from pathlib import Path
            import os
            if self.model_path == "default" :
                weights_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "weights")))
                pth_files = list(weights_dir.glob("*.pth"))
                if not pth_files : 
                    raise FileNotFoundError(f"No .pth files found in {weights_dir}")
                weights_path = pth_files[0]
            else :
                weights_path = self.model_path
            detector = Detector(model_path=weights_path, device="cuda")
        return detector

    def compute_rois_matching_metrics(self, roi1, roi2):
        """Computes size ratio and containment of two ROIs

        Args:
            roi1 (ROI): _description_
            roi2 (ROI): _description_

        Returns:
            tuple: containment, size_ratio
        """
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

    def confidence_weighted_average(self, detection_pos, prediction_pos, containment) :
        """Compute the confidence weighted average.
        Compute alpha using a sigmoid function

        Args:
            detection_pos (Positon2D): Detection position
            prediction_pos (Position2D): Prediction position
            containment (float): ROI containment

        Returns:
            Position2D: The fused position
        """
        misalignement = 1 - containment
        alpha = 1 / (1 + np.exp(-self.k * (misalignement - self.c0)))
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




#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------


class MultiRoIBaseTracker :
    def __init__(
            self,
            first_frame,
            rois,
            scaling_factor,
            serverkit,
            server_addresses,
            window_length,
            grid_size,
            base_kernel_size_xy,
            kernel_size_z,
            log,
            use_detection,
            containment_threshold,
            k,
            c0,
            size_ratio_threshold,
            score_threshold,
            model_path,
    ) :
        # Takes raw frames as input, handle processing (norm, scaling, projection), manage the sliding window
        # and manage the computation of the RoI position across frames
        self.scaling_factor = scaling_factor
        self.window_length = window_length
        self.grid_size = grid_size
        self.kernel_size_xy = base_kernel_size_xy // (2**scaling_factor)
        self.kernel_size_z = kernel_size_z
        self.log = log
        self.use_detection = False ############# NO detection for multi roi
        self.containment_threshold = containment_threshold
        self.k = k
        self.c0 = c0
        self.size_ratio_threshold = size_ratio_threshold
        self.score_threshold = score_threshold
        self.model_path = model_path

        # Convert to ROI dataclass, downscale rois
        self.rois = [ROI(**roi).scale(scaling_factor=self.scaling_factor, down=True) for roi in rois]      

        if use_detection == True :
            self.logger.warning("Detection set to False, not supported for MultiROI")

        # Initialize rolling window, queries, tracked points
        if first_frame :
            # Initialize window
            self.current_frame = self._downsample(first_frame)
            self.shape = self.current_frame.shape
            self.current_frame_proj = np.max(self.current_frame, axis=0)
            self.window_frames = [self.current_frame_proj]
            # Initialize queries
            self.queries_init, self.queries_lengths_init = self._initialize_queries(self.current_frame_proj, self.rois)
            self.tracks = []
            # Initialize tracked points
            self.tracked_points = [[roi.to_position3D(z=self.shape[0]//2) for roi in self.rois]]
            self.tracking_state = TrackingState.TRACKING_ON
            self.count = 1
        else :
            self.window_frames = []
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
            self.count = 0

        # Initialize model
        self.predictor = self.initialize_predictor(serverkit, server_addresses=server_addresses)
        if self.use_detection :
            self.detector = self.initialize_detector(serverkit, server_addresses=server_addresses)
        self.time_since_last_detection = 0 # to keep track of when should the detection be done

        # Initialize 2D RoI List
        self.rois_list = [self.rois]
        self.predicted_points = [[roi.to_position2D() for roi in self.rois]]
        # self.detected_points = [self.roi_init.to_position2D()]  #### DETECTION NOT TRIVIAL FOR MULTI ROI
        self.detected = False

        # Set default logger
        self.logger = init_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")
        param_lines = [f"  {k}: {v}" for k, v in vars(self).items()
                if not k.startswith('_') and k in ["scaling_factor", "window_length", "grid_size",
                                                    "shape", "kernel_size_xy", "kernel_size_z",
                                                    "use_detection", "serverkit", "k", "c0", "containment_threshold",
                                                    "score_threshold", "size_ratio_threshold"]]
        param_block = "\n".join(param_lines)
        self.logger.info(f"Initialized a new ROI tracker: \nParameters:\n{param_block}")

    def compute_new_positions(self, frame) :
        """Compute the new position of the ROIs in the given image.
        Make use of the dataclasses defined in tracking_tools/utils/structures.

        Args:
            frame (np.ndarray): New frame

        Returns:
            tuple: Position3D of the tracked ROI (Full scale), trackingState
        """
        if self.tracking_state != TrackingState.TRACKING_OFF :
            self.count += 1

            # If first frame, process it and return 0 shift
            if self.window_frames == [] :
                # Initialize window
                self.current_frame = self._downsample(frame)
                self.shape = self.current_frame.shape
                self.current_frame_proj = np.max(self.current_frame, axis=0)
                self.window_frames = [self.current_frame_proj]
                # Initialize queries
                self.queries_init, self.queries_lengths_init = self._initialize_queries(self.current_frame_proj, self.rois)
                self.tracks = [np.empty((0, 2))]
                # Initialize tracked points
                self.tracked_points = [[roi.to_position3D(z=self.shape[0]//2) for roi in self.rois]]
                self.tracking_state = TrackingState.TRACKING_ON
                self.detected = False
                return Position3D.invalid(), TrackingState.WAIT_FOR_NEXT_TIME_POINT # Placeholder return

            # Process input frame
            frame = self._downsample(frame)
            self.current_frame = frame
            frame_proj = np.max(frame, axis=0)
            self.current_frame_proj = frame_proj

            self.update_rolling_window(frame_proj)

            # Compute CoTracker3 tracks
            new_tracks, queries_lengths, tracking_state = self.update_tracks()
            self.tracking_state = tracking_state

            # Retrieve the different rois tracks
            n_rois = len(queries_lengths)
            tracks_by_roi = [[] for _ in range(n_rois)]
            for tp in new_tracks:
                start = 0
                for roi_idx, length in enumerate(queries_lengths) :
                    stop = start + length
                    tracks_by_roi[roi_idx].append(tp[start:stop])
                    start = stop

            if tracking_state == TrackingState.TRACKING_OFF :
                return Position3D.invalid(), self.tracking_state # Placeholder return
            
            self.tracks.append(tracks_by_roi)

            if tracking_state == TrackingState.WAIT_FOR_NEXT_TIME_POINT :
                return Position3D.invalid(), self.tracking_state # Placeholder
            
            else :
                # Compute predicted ROIs
                predicted_points_by_roi = []
                new_rois = []
                new_positions = []
                for i, roi_tracks in enumerate(tracks_by_roi) :
                    last_tracked_point = self.tracked_points[-1][i]
                    position_yx_predicted = self.compute_predicted_position_yx_ls(roi_tracks, last_tracked_point)
                    last_roi = self._get_past_roi(index=-1)[i]
                    roi_predicted = ROI(
                        x=position_yx_predicted.x,
                        y=position_yx_predicted.y,
                        height=last_roi.height,
                        width=last_roi.width,
                        order=1
                    )
                    # self.predicted_points.append(position_yx_predicted) ###################
                    predicted_points_by_roi.append(position_yx_predicted)
                    # Compute the detected ROI
                    if self.use_detection :
                        roi_detected, score, returnStatus = self.compute_detected_roi(self.current_frame_proj)
                        self.detected_points.append(roi_detected.to_position2D())
                    else :
                        self.detected = False
                        roi_detected = ROI.invalid(order=1) # Placeholder
                        score = 0                           # Placeholder

                    # Fuse positions and compute z position
                    position_yx, new_roi = self.fuse_positions_yx(roi_predicted, roi_detected, score)
                    # self.rois_list.append([new_roi]) ######################
                    new_rois.append(new_roi)

                    position_z = self.compute_new_position_z(new_roi)

                    new_positions.append(position_yx.to_position3D(z=position_z))
                
                self.rois_list.append(new_rois)
                self.predicted_points.append(predicted_points_by_roi)

            self.tracked_points.append(new_positions)
            return [position.scale(scaling_factor=self.scaling_factor, down=False, axes="xy") for position in new_positions], self.tracking_state
        else :
            return Position3D.invalid(), self.tracking_state # Placeholder return

    def update_rolling_window(self, frame_proj) :
        self.window_frames.append(frame_proj)
        # Slice the rolling window to always get the last [window_length] frames
        self.window_frames = self.window_frames[-self.window_length:]

    def update_tracks(self) :
        """Generate new query points and run the CoTracker3 model

        Returns:
            tuple: Returns the CoTracker3 predicted tracks and tracking state (One of TRACKING_ON, TRACKING_OFF, WAIT_FOR_NEXT_TIME_POINT)
        """
        if self.count <= self.window_length :
            queries = self.queries_init
            queries_lengths = self.queries_lengths_init
        else :
            rois = self._get_past_roi(index=self.count-self.window_length)
            queries, queries_lengths = self._initialize_queries(self.window_frames[0], rois)

        normalized_video = self._normalize_percentile(self.window_frames) 
        pred_tracks_np, returnStatus = self.predictor._run_model(normalized_video, queries)
        tracking_state = self.tracking_state

        if returnStatus == ReturnStatus.SUCCESS :
            return pred_tracks_np, queries_lengths, tracking_state
        
        elif returnStatus == ReturnStatus.SERVER_ERROR :
            self.logger.warning(f"Serverkit Tracker failed — Retrying at the next time point. {returnStatus}")
            tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
            return pred_tracks_np, queries_lengths, tracking_state
        
        elif returnStatus == ReturnStatus.LOCAL_TRACKER_ERROR :
            self.logger.warning(f"Tracker failed, disabling this position. {returnStatus}")
            tracking_state = TrackingState.TRACKING_OFF
            return np.zeros_like(queries), queries_lengths, tracking_state
    
    
    def compute_predicted_position_yx_ls(self, tracks, last_tracked_point) :
        """Fit a linear regression model to the tracks and predict a position

        Args:
            tracks (_type_): _description_
            last_tracked_point (_type_): _description_

        Returns:
            _type_: _description_
        """
        current_points = tracks[-1][:,::-1]  # xy -> yx
        previous_points = tracks[-2][:,::-1] # xy -> yx
        motions = current_points - previous_points # yx
        # Fit a model to the vector field using least square regression
        X = np.hstack([previous_points, np.ones((previous_points.shape[0], 1))]) # add intercept [y, x, 1]
        U = motions
        W, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        R = W[:2].T
        x = W[2]
        # Compute the new position
        last_tracked_point = last_tracked_point.to_array('yx') # yx
        new_pos_yx = last_tracked_point + R @ last_tracked_point + x # yx, predicted point = point + predicted motion
        new_pos_yx = Position2D(x=new_pos_yx[1], y=new_pos_yx[0])
        return new_pos_yx
    
    def compute_detected_roi(self, frame) :
        """Run the Faster-RCNN model to predict an ROI in a frame 

        Args:
            frame (np.ndarray): Input frame (2D)

        Returns:
            Tuple: Detected ROI, softmax score of the detection, returnStatus of the model
            returnStatus one of SUCESS indicating a detected ROI or NO_OP indicating no detection.
        """
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

    
    def fuse_positions_yx(self, roi_predicted, roi_detected, score) :
        """Validate the detection and fuses the predicted and detected ROIs

        Args:
            roi_predicted (ROI): Prediction ROI, from CoTracker3
            roi_detected (ROI): Detction ROI, from Faster-RCNN
            score (float): softmax score of the detection

        Returns:
            tuple: Fused position xy, fuse ROI.
        """
        fusion_valid = False
        # Check if detection is present, and valid
        if self.detected :
            containement, size_ratio = self.compute_rois_matching_metrics(roi_predicted, roi_detected)
            self.logger.info(f"Detection validation : model score :{score}, containment: {containement}, size_ratio: {size_ratio}")
            # Validation conditions
            if (score > self.score_threshold) and (containement > self.containment_threshold) and (size_ratio > self.size_ratio_threshold):
                fusion_valid = True
                self.logger.info("Valid Detection")
            else :
                self.logger.info("Invalid Detection")
        
        if fusion_valid :
            position_detected = roi_detected.to_position2D()
            position_predicted = roi_predicted.to_position2D()
            fused_pos_yx = self.confidence_weighted_average(position_detected, position_predicted, containement)
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
        """Computes the position in z given an ROI, using the center of mass method

        Args:
            roi (ROI): ROI used to crop the image

        Returns:
            float: center of mass in z
        """
        # Build the bounding box (ROI in xy and full dimension in z)
        D, H, W = self.current_frame.shape
        position_yx = roi.to_position2D()
        hws = [roi.height / 2, roi.width / 2]
        center_point_z_tracking = position_yx.to_position3D(z=D//2).to_array(order='zyx')
        hws_z_tracking = np.array([D//2, *hws])

        # Crop
        frame_cropped = crop_image(self.current_frame, center_point_z_tracking, hws_z_tracking)

        # Filter (Gaussian blurring)
        frame_cropped_filtered = filter_image(frame_cropped, median_kernel=0, gaussian_kernel_xy=self.kernel_size_xy, gaussian_kernel_z=self.kernel_size_z)

        # Binarize
        binary = threshold_image(frame_cropped_filtered)

        # center of mass in z
        com_z = center_of_mass(binary)[0]
        return com_z

    def _initialize_queries(self, frame, rois) : 
        """Generate query points given a frame and a list of ROIs

        Args:
            frame (np.ndarray): Input frame (2D)
            list(rois) (ROI): Input ROIs

        Returns:
            np.ndarray: Nx2 array of points coordinates
        """

        center_points_list = []
        hws_list = []

        for roi in rois :
            center_points_list.append(roi.to_position2D().to_array(order='yx'))
            hws_list.append([roi.height / 2, roi.width / 2])

        points, queries_lengths = generate_uniform_grid_in_region_list(frame, center_points_list, hws_list, grid_size=self.grid_size, gaussian_kernel=self.kernel_size_xy)

        # Log warning for no points
        for i, roi in enumerate(rois) :
            length = queries_lengths[i]
            if length == 0:
                self.logger.warning(f"No query points generated for RoI {roi.order}.")

        if points.any() :
            return points, queries_lengths
        else :
            self.logger.warning("No query points generated for any ROIs")
            self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
            return np.empty((0, 2)), queries_lengths # No points from any rois

        # total_points = []
        # queries_lengths = []
        # # Build regions 
        # for roi in rois :
        #     # Build bounding box
        #     center_point = roi.to_position2D().to_array(order='yx')
        #     hws = [roi.height / 2, roi.width / 2]

        #     points = generate_uniform_grid_in_region(frame, center_point, hws, grid_size=self.grid_size, gaussian_kernel=self.kernel_size_xy)

        #     if len(points) == 0:
        #         self.logger.warning(f"No query points generated for RoI {roi.order}.")
        #         total_points.append(np.empty((0, 2)))
        #     else :
        #         total_points.append(points)
        #         queries_lengths.append(len(points))

        # if total_points:
        #     return np.vstack(total_points), queries_lengths
        # else :
        #     self.logger.warning("No query points generated for any ROIs")
        #     self.tracking_state = TrackingState.WAIT_FOR_NEXT_TIME_POINT
        #     return np.empty((0, 2)), queries_lengths # No points from any rois

    
    def _normalize_percentile(self, video_batch) :
        """Normalize a video using the 1st and 99th percentiles. Converts to uint8

        Args:
            video_batch (np.ndarray): video N*H*W

        Returns:
            np.nadarray: Normalized video
        """
        q1, q99 = np.quantile(video_batch, [0.01, 0.99])
        value_range = q99 - q1
        normalized_video = np.clip((video_batch - q1) / value_range, 0, 1)
        normalized_video = (normalized_video * 255).astype(np.uint8)
        return normalized_video

    def _downsample(self, image) :
        image = image[:, ::2**self.scaling_factor, ::2**self.scaling_factor]
        return image
    
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
            from pathlib import Path
            import os
            if self.model_path == "default" :
                weights_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "weights")))
                pth_files = list(weights_dir.glob("*.pth"))
                if not pth_files : 
                    raise FileNotFoundError(f"No .pth files found in {weights_dir}")
                weights_path = pth_files[0]
            else :
                weights_path = self.model_path
            detector = Detector(model_path=weights_path, device="cuda")
        return detector

    def compute_rois_matching_metrics(self, roi1, roi2):
        """Computes size ratio and containment of two ROIs

        Args:
            roi1 (ROI): _description_
            roi2 (ROI): _description_

        Returns:
            tuple: containment, size_ratio
        """
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

    def confidence_weighted_average(self, detection_pos, prediction_pos, containment) :
        """Compute the confidence weighted average.
        Compute alpha using a sigmoid function

        Args:
            detection_pos (Positon2D): Detection position
            prediction_pos (Position2D): Prediction position
            containment (float): ROI containment

        Returns:
            Position2D: The fused position
        """
        misalignement = 1 - containment
        alpha = 1 / (1 + np.exp(-self.k * (misalignement - self.c0)))
        fused_pos = alpha * detection_pos + (1 - alpha) * prediction_pos
        return fused_pos
    
    def _get_past_roi(self, index) :
        return copy.copy(self.rois_list[index])
    
    def fill_placeholders(self) :
        self.tracked_points.append(self.tracked_points[-1])
        self.predicted_points.append(Position2D.invalid())
        self.detected_points.append(Position2D.invalid())
        self.rois_list.append([self._get_past_roi(-1)])
        self.tracks.append([])
        self.logger.info("Filled placeholder data for tracker.")
    