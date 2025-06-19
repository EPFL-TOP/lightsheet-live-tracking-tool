from typing import List
import numpy as np
import uvicorn
import torch
import numpy as np
# from cotracker.predictor import CoTrackerPredictor
import utils.tracker_utils as tracker_utils
import utils.detector_utils as detector_utils

from imaging_server_kit import algorithm_server, ImageUI, PointsUI
from imaging_server_kit import MultiAlgorithmServer
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2

from pathlib import Path



device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Running on device:', device)
# model = CoTrackerPredictor(checkpoint='../co-tracker/checkpoints/scaled_offline.pth')
model_tracker =  torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
model_tracker = model_tracker.to(device)

model_detector_path = Path(__file__).resolve().parent / "tail_detection_model_v2_augmented_100ep.pth"
model_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
in_features = model_detector.roi_heads.box_predictor.cls_score.in_features
model_detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
checkpoint = torch.load(model_detector_path, weights_only=True)
model_detector.load_state_dict(checkpoint['model_state_dict'])
model_detector.to(device)
model_detector.eval()


#___________________________________________________________________________________

@algorithm_server(
    algorithm_name="tracker",
    parameters={
        "video": ImageUI(),
        "query_points": PointsUI()
    },
)
def tracker_server(
    video: np.ndarray,
    query_points: np.ndarray,
):
    """Runs the algorithm."""
    video = np.repeat(video[..., np.newaxis], 3, axis=-1) # Convert to RGB
    pred_tracks, pred_visibility = tracker_utils.process_step_offline(model_tracker, video, query_points, device)
    # Convert pred_tracks (batch, video_length, num_points, coords_dim) into a numpy array with shape (video_length, num_points, coords_dim)
    # batch = 1, coords_dim = 2
    pred_tracks_np = pred_tracks.cpu().numpy()[0] # (video_length, num_points, 2)

    return_params = {
        "name": "Prediction result, shape = (video_length, num_points, 2)"
    }  # Add information about the result (optional)

    return [
        (pred_tracks_np, return_params, "image")
    ]  # Choose the right output type (`mask` for a segmentation mask)



#___________________________________________________________________________________
@algorithm_server(
    algorithm_name="detector",
    parameters={
        "image": ImageUI(title="Image", description="Input Grayscale Image."),
    },
)
def detector_server(
    image: np.ndarray,
) -> List[tuple]:
    """Runs the algorithm."""
    frame_pp = detector_utils.preprocess_frame(image)
    frame_pp = frame_pp.to(device)
    predictions = model_detector(frame_pp)[0]
    predictions_cpu = {k:v.detach().cpu().numpy() for k, v in predictions.items()}


    probabilities = np.array(predictions_cpu['scores']) # (N,)
    boxes = np.array(predictions_cpu['boxes'])
    boxes_corners = np.array([
        [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        for (xmin, ymin, xmax, ymax) in boxes
    ]) # (N, 4, 2)

    # import pdb; pdb.set_trace()

    return_params = {
        "name": "Prediction result",
        "features": {
            "probabilities": probabilities,
        }
    }

    return [
        (boxes_corners, return_params, "boxes")
    ]  # Choose the right output type (`mask` for a segmentation mask)


#___________________________________________________________________________________
server = MultiAlgorithmServer(
    server_name="multi-algo",
    algorithm_servers=[
      tracker_server,   # Implemented with @algorithm_server
      detector_server,  # Implemented with @algorithm_server
    ]
)


#___________________________________________________________________________________
if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
