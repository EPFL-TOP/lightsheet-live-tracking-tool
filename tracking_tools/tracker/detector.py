import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2
from ..logger.logger import init_logger
from imaging_server_kit import Client
from .status import ReturnStatus

import numpy as np


class Detector :
    model = None # Shared across instances

    def __init__(
            self,
            model_path,
            # device,
            num_classes=2
    ) :
        self.logger = init_logger(self.__class__.__name__)
        if Detector.model is None :
            Detector.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT')
            in_features = Detector.model.roi_heads.box_predictor.cls_score.in_features
            Detector.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.logger.info(f'Initialize model running on device: {device},  with weights: {model_path}')

            checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))

            Detector.model.load_state_dict(checkpoint['model_state_dict'])
            Detector.model.to(device)
            Detector.model.eval()
        self.model = Detector.model
        self.device = device

    def _run_model(self, frame):
        self.logger.info("Running Faster-RCNN")
        if frame.ndim == 3 :
            frame = np.max(frame, axis=0)
        frame_pp = self._preprocess_frame(frame)
        frame_pp = frame_pp.to(self.device)
        predictions = self.model(frame_pp)[0]
        predictions_cpu = {k:v.detach().cpu().numpy() for k, v in predictions.items()}
        return predictions_cpu, ReturnStatus.SUCCESS
    
    def _preprocess_frame(self, frame) :
        transform = ToTensorNormalize()
        frame_pp = transform(frame)
        return frame_pp.unsqueeze(0) 
    

class DetectorSK:
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


    def _run_model(self, frame):
        try :
            client_output = self.client.run_algorithm(
                algorithm="detector",
                image=frame,
            )
            box_vertices = client_output[0][0]
            scores = client_output[0][1]['features']['probabilities']
            return self._convert_format(box_vertices, scores), ReturnStatus.SUCCESS
        
        except Exception as e :
            self.logger.error(f"Server error: {e}")
            fallback = self._fallback()
            return fallback, ReturnStatus.SERVER_ERROR

    def _fallback(self) :
        self.logger.warning("Using fallback: Returning invalid 0,0,0,0 box with score 0.")
        return {'boxes':[[0,0,0,0]], 'scores':[0]}
    
    def _convert_format(self, box_vertices, scores) :
        # Convert (N, 4, 2) vertices to {'boxes':(N, 4), 'scores':(N,)} xmin ymin xmax ymax
        xmins = np.min(box_vertices[:, :, 0], axis=1)
        ymins = np.min(box_vertices[:, :, 1], axis=1)
        xmaxs = np.max(box_vertices[:, :, 0], axis=1)
        ymaxs = np.max(box_vertices[:, :, 1], axis=1)
        boxes = np.stack([xmins, ymins, xmaxs, ymaxs], axis=1)
        output = {'boxes':boxes.tolist(), 'scores':scores.tolist()}
        return output

    


class ToTensorNormalize:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = transforms.functional.pil_to_tensor(image).float()
        image = (image - image.min()) / (image.max() - image.min())
        return image
    



