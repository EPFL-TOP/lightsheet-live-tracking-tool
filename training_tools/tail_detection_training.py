import os
import torch
import json
import tifffile
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.optim as optim


import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CellDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path  # Base directory containing all experimental conditions
        self.transform = transform
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
            image_name = image_path.split("\\")[-1]
            image_path = os.path.join(self.base_path,image_name)
            image = tifffile.imread(image_path)

        RoIs = annotations["RoIs"]

        #‘xyxy’: boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right. This is the format that torchvision utilities expect.
        image_array = np.array(image, dtype=np.float32)  # Convert to float32 to avoid overflow
        boxes = []
        for RoI in RoIs:
            box=[int(RoI["x"]-RoI["width"]/2),
                 int(RoI["y"]-RoI["height"]/2),
                 int(RoI["x"]+RoI["width"]/2),
                 int(RoI["y"]+RoI["height"]/2)
                 ]
            boxes.append(box)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all cells belong to one class

        if self.transform:
            image = self.transform(image_array)
        else:
            image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add channel dimension

        target = {
            'boxes': boxes,
            'labels': labels
        }
        return image, target

class ToTensorNormalize:
    def __call__(self, image):
        # If input is a NumPy array
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        else:  # Handle PIL Image (if applicable)
            image = transforms.functional.pil_to_tensor(image).float()
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        return image

transform = transforms.Compose([
    ToTensorNormalize(),
    transforms.Resize((512, 512))
])


base_path = ''
if os.path.isdir(r'E:\tail_tracking\train'):
    base_path = r'E:\tail_tracking'
if os.path.isdir('/mnt/c/Users/helsens/'):
    base_path = '/mnt/h/PROJECTS-03/clement/tail_tracking'

train_path = os.path.join(base_path, 'train')
train_dataset = CellDataset(base_path=train_path, transform=transform)

val_path = os.path.join(base_path, 'valid')
val_dataset = CellDataset(base_path=val_path, transform=transform)

# Split dataset into training and validation sets
#train_size = int(0.8 * len(dataset))
#val_size = len(dataset) - train_size
batch_size = 4  # Adjust this based on your GPU memory
#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Print the number of samples in training and validation sets
print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2  # 1 class (cell) + background
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model_save_path = 'tail_detection_model.pth'
#model.load_state_dict(torch.load(model_save_path))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device: ',device)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=1e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f'Starting epoch {epoch + 1}/{num_epochs}')

    for i, (images, targets) in enumerate(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        if i % 10 == 0:
            print(f'[{epoch + 1}, {i}] loss: {losses.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }
    #torch.save(model.state_dict(), model_save_path)
    torch.save(checkpoint, model_save_path)


    print(f'Model saved to {model_save_path} after epoch {epoch + 1 }')




def predict(model, image_array):
    model.eval()
    transform = transforms.Compose([
        ToTensorNormalize(),
        transforms.Resize((512, 512))
    ])
    image = transform(image_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image)
        
    return prediction



def plot_predictions(image, prediction):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    
    
    for box in prediction[0]['boxes']:
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

model.eval()
for i, (images, targets) in enumerate(val_loader):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    with torch.no_grad():
        predictions = model(images)
    
    for j in range(len(images)):
        image = images[j].cpu().numpy().squeeze(0)
        plot_predictions(image, predictions)
        if j == 2:
            break
    break



