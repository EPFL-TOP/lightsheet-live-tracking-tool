import torch
import torchvision
import numpy as np
from imaging_server_kit import Client
from .status import ReturnStatus
from ..logger.logger import init_logger

class Tracker :
    model = None # Shared across instances
    def __init__(self, *args, **kwargs) :
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = init_logger(self.__class__.__name__)
        if Tracker.model is None :
            print('Initialize model running on device: ', device)
            # self.model = CoTrackerPredictor(checkpoint='../co-tracker/checkpoints/scaled_offline.pth')
            Tracker.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
            Tracker.model = Tracker.model.to(device)
        self.model = Tracker.model
        self.device = device

    def _run_model(self, window_frames, queries) :
        self.logger.info("Running Cotracker3")
        video_chunk = self._prepare_video_chunk(window_frames)
        queries_tensor = torch.tensor(
            np.column_stack((np.zeros(queries.shape[0]), queries)),
            dtype=torch.float16,
            device=self.device
        )[None]
        pred_tracks, _ = self.model(video_chunk, queries=queries_tensor)
        return pred_tracks.cpu().numpy()[0], ReturnStatus.SUCCESS

    def _prepare_video_chunk(self, window_frames) :
        frames = np.asarray(window_frames.copy())
        if frames.ndim == 3 : # If not RGB (no channel dimension)
            frames = np.repeat(frames[..., np.newaxis], 3, axis=-1) # Convert to RGB by duplicating into 3 channels
        video_chunk = torch.tensor(
            np.stack(frames), device=self.device
            ).float().permute(0, 3, 1, 2)[None]
        return video_chunk
    

class TrackerSK:
    def __init__(self, server_addresses):
        self.logger = init_logger(self.__class__.__name__)
        self.server_addresses = server_addresses
        self.client = None
        
    def connect(self) :
        for address in self.server_addresses:
            try:
                self.logger.info(f"Trying to connect to server at {address}...")
                self.client = Client(address)  # Attempt connection
                self.logger.info(f"Connected to {address}")
                return ReturnStatus.SUCCESS
            except Exception as e:
                self.logger.warning(f"Failed to connect to {address}: {e}")

        self.logger.error(f"Could not connect to any of the specified servers : {self.server_addresses}")
        return ReturnStatus.SERVER_ERROR


    def _run_model(self, window_frames, queries):
        try :
            client_output = self.client.run_algorithm(
                algorithm="tracker",
                video=np.array(window_frames),
                query_points=queries
            )
            return client_output[0][0], ReturnStatus.SUCCESS
        
        except Exception as e :
            self.logger.error(f"Server error: {e}")
            fallback = self._fallback(queries)
            return fallback, ReturnStatus.SERVER_ERROR

    def _fallback(self, queries) :
        self.logger.warning("Using fallback: returning original query points.")
        return queries
    
