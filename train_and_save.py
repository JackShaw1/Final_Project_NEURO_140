import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import PIL.Image as Image
import os
import matplotlib.pyplot as plt

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DentalDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(annotation_file)
        self.class_to_idx = {'Implant': 1, 'Fillings': 2, 'Cavity': 3, 'Impacted Tooth': 4}
        self.data['image_path'] = self.data['filename'].apply(lambda x: os.path.join(self.root, x))
        self.data = self.data[self.data['image_path'].apply(os.path.exists)]
        self.grouped = self.data.groupby('filename')

    def __getitem__(self, idx):
        image_name = list(self.grouped.groups.keys())[idx]
        annotations = self.grouped.get_group(image_name)
        img_path = os.path.join(self.root, image_name)
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        for _, row in annotations.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[row['class']])
        if not boxes:
            return None, None
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.grouped)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet18', pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    model.to(device)
    return model

# Load the dataset
full_dataset = DentalDataset('YOUR_DATA_FOLDER_HERE', 'YOUR_ANNOTATION_FILE_HERE', transforms=F.to_tensor)

# Splitting the dataset
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Model and Optimizer
model = get_model(num_classes=5)  # 4 classes + 1 background
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 5

train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch+1}")
    total_train_loss = 0
    total_val_loss = 0

    i = 1
    # Training loop
    model.train()
    for images, targets in train_loader:
        print(i)
        i = i + 1
        images = [img for img in images if img is not None]
        targets = [t for t in targets if t is not None]
        if images and targets:
            images = list(map(lambda img: img.to(device), images))
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            optimizer.step()
            total_train_loss += losses.item()
    epoch_train_loss = total_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    with torch.no_grad():
        for images, targets in valid_loader:
            images = [img for img in images if img is not None]
            targets = [t for t in targets if t is not None]
            if images and targets:
                images = list(map(lambda img: img.to(device), images))
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets] 
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                total_val_loss += losses.item()

    epoch_val_loss = total_val_loss / len(valid_loader)
    valid_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1} completed, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")

# Save the entire model
torch.save(model, '/content/drive/MyDrive/dental_radiographs/full_model_final.pth')


plt.figure(figsize=(10, 5))
# Create an array with epoch numbers starting from 1
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.xticks(epochs)  # Setting x-ticks to show correct epoch numbers
plt.show()
