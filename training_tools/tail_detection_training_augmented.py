import os
import torch
import json
import tifffile
import numpy as np
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import box_iou

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CellDataset(Dataset):
    def __init__(self, base_path, transforms=None):
        self.base_path = base_path
        self.transforms = transforms
        self.json_files = self._collect_json_files()

    def _collect_json_files(self):
        json_files = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        image_path = annotations['image']
        try:
            image = tifffile.imread(image_path)
        except FileNotFoundError:
            image_name = os.path.basename(image_path)
            image = tifffile.imread(os.path.join(self.base_path, image_name))

        RoIs = annotations['RoIs']
        h, w = image.shape[:2]

        # format boxes as Pascal VOC
        boxes = []
        for RoI in RoIs:
            x_c, y_c = RoI['x'], RoI['y']
            half_w, half_h = RoI['width']/2, RoI['height']/2
            x1 = max(0, x_c - half_w)
            y1 = max(0, y_c - half_h)
            x2 = min(w, x_c + half_w)
            y2 = min(h, y_c + half_h)
            boxes.append([x1, y1, x2, y2])

        labels = [1] * len(boxes)

        if self.transforms:
            augmented = self.transforms(image=image.astype(np.uint8), bboxes=boxes, labels=labels)
            image = augmented['image']

            # Albumentations may remove boxes if they move outside image
            if len(augmented['bboxes']) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(augmented['bboxes'], dtype=torch.float32)
                labels = torch.tensor(augmented['labels'], dtype=torch.int64)


            #boxes = torch.tensor(augmented['bboxes'], dtype=torch.float32)
            #labels = torch.tensor(augmented['labels'], dtype=torch.int64)
        else:
            image = ToTensorV2()(image=image.astype(np.uint8))['image']
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target


def get_transform(train=True):
    if train:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Affine(scale=(0.8, 1.2), p=0.5)
#            A.RandomBrightnessContrast(p=0.5),
            #A.OneOf([
                # Mild scaling: 0.8x – 1.2x
             #   A.Affine(scale=(0.8, 1.2), p=1.0),
                # Strong scaling: 1.8x – 2.2x
                #A.Affine(scale=(1.8, 2.2), p=1.0),
            #], p=0.6),

            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),

            A.Resize(512, 512),
            A.Normalize(mean=(0.0,), std=(1.0,)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.0,), std=(1.0,)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_model(model, data_loader, device):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, target in zip(outputs, targets):
                gt_boxes = target['boxes'].to(device)
                pred_boxes = pred['boxes']
                scores = pred['scores']

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    iou_scores.append(0.0)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                max_iou = ious.max().item()
                iou_scores.append(max_iou)

    avg_iou = np.mean(iou_scores)
    return avg_iou


# Paths
base_path = ''
if os.path.isdir(r'E:\tail_tracking\train'):
    base_path = r'E:\tail_tracking'
elif os.path.isdir('/mnt/c/Users/helsens/'):
    base_path = '/mnt/h/PROJECTS-03/clement/tail_tracking'

train_dataset = CellDataset(os.path.join(base_path, 'train'), transforms=get_transform(train=True))
val_dataset   = CellDataset(os.path.join(base_path, 'valid'), transforms=get_transform(train=False))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Model setup: Faster R-CNN ResNet50 FPN v2
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2  # 1 class + background
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Training setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=1e-4)
num_epochs = 100
model_save_path = 'tail_detection_model.pth'

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")

#    for images, targets in train_loader:
    for i, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        if i % 10 == 0:
            print(f'[{epoch + 1}, {i}] loss: {losses.item():.4f}')

    avg_loss = epoch_loss / len(train_loader)
    print(f"Avg Loss: {avg_loss:.4f}")

    avg_val_iou = evaluate_model(model, val_loader, device)
    print(f"Validation IoU: {avg_val_iou:.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, model_save_path)
    print(f"Saved checkpoint: {model_save_path} (epoch {epoch+1})")

# Inference & visualization functions (unchanged)
