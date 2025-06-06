import os, sys
import torch
import json
import tifffile
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ToTensorNormalize:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = transforms.functional.pil_to_tensor(image).float()
        image = (image - image.min()) / (image.max() - image.min())
        return image


def preprocess_image_pytorch(image_array):
    transform = ToTensorNormalize()
    image = transform(image_array)
    return image.unsqueeze(0) 


def visualize_predictions(image, predictions):
    fig, ax = plt.subplots(1)
    print(image.max(),  '  ', image.min())
    img = image#.cpu().numpy()
    ax.imshow(img, cmap='gray')
    for idx, box in enumerate(predictions[0]['boxes']):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min+(y_max-y_min)/2, f"{predictions[0]['scores'][idx].cpu().numpy() :.3f}", fontsize=10, color='white')
    

    plt.show()



class DetectModel:
    def __init__(self):
        self.model = None
        self.device = None

    def load_model_detect(self, model_path, num_classes, device_str):
        self.device = torch.device(device_str)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device).eval()

        #return model, device


    def get_predictions(self, image):
        with torch.no_grad():
            labels = self.model(image)
            print(labels)
            return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate object detection model on new images')
    parser.add_argument('model_path', type=str, help='Path to the trained model (.pth)')
    parser.add_argument('input_dir', type=str, help='Path to directory of .tif images')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes including background')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    #model = load_model_detect(args.model_path, args.num_classes, args.device)
    #model, device = load_model_detect(args.model_path, args.num_classes, args.device)
    detect_model = DetectModel()
    detect_model.load_model_detect(args.model_path, args.num_classes, args.device)

    image_dir = args.input_dir
    image_filenames = [os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
    image_filenames.sort()


    for img in image_filenames:
        image = tifffile.imread(img)


        image_pp = preprocess_image_pytorch(image).to(detect_model.device)
        labels = detect_model.get_predictions(image_pp)

        visualize_predictions(image, labels)




