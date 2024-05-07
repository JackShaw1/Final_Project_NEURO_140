import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
def load_trained_model(num_classes):
    model = torch.load("YOUR_MODEL_PATH_HERE")
    return model

model = load_trained_model(5)  # 4 classes + 1 background
model.to(device)

# Define the dataset class
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = [os.path.join(root, img) for img in os.listdir(root) if img.endswith('.jpg') or img.endswith('.png')]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_path

    def __len__(self):
        return len(self.images)

# Load annotations
def load_annotations(path):
    annotations_df = pd.read_csv(path)
    annotations_dict = {}
    for _, row in annotations_df.iterrows():
        key = row['filename']
        if key not in annotations_dict:
            annotations_dict[key] = []
        annotations_dict[key].append({
            'class': row['class'],
            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        })
    return annotations_dict

annotations_dict = load_annotations('PATH_TO_YOUR_ANNOTATIONS')

# IoU calculation
def bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def predict_and_evaluate(model, dataset, annotations_dict, class_names, thresholds, iou_threshold=0.5):
    model.eval()
    results = []
    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        total_ground_truth = 0

        for img, path in dataset:
            filename = os.path.basename(path)
            if filename in annotations_dict:
                ground_truths = annotations_dict[filename]
                total_ground_truth += len(ground_truths)

                with torch.no_grad():
                    prediction = model([img.to(device)])[0]

                # Track which ground truths have been matched
                matched_ground_truths = [False] * len(ground_truths)

                for pred_idx in range(len(prediction['boxes'])):
                    pred_score = prediction['scores'][pred_idx].item()
                    if pred_score > threshold:
                        pred_bbox = prediction['boxes'][pred_idx].cpu().numpy().astype(int)
                        pred_class_id = prediction['labels'][pred_idx].item()
                        pred_class = class_names[pred_class_id]

                        for gt_idx, gt in enumerate(ground_truths):
                            if not matched_ground_truths[gt_idx]:  # Only consider unmatched ground truths
                                gt_class = gt['class']
                                gt_bbox = gt['bbox']

                                if pred_class == gt_class and bbox_iou(pred_bbox, gt_bbox) > iou_threshold:
                                    true_positives += 1
                                    matched_ground_truths[gt_idx] = True
                                    break

                        # Count unmatched predictions as false positives
                        for pred_idx in range(len(prediction['boxes'])):
                            pred_score = prediction['scores'][pred_idx].item()
                            if pred_score > threshold:
                                pred_bbox = prediction['boxes'][pred_idx].cpu().numpy().astype(int)
                                pred_class_id = prediction['labels'][pred_idx].item()
                                pred_class = class_names[pred_class_id]
                        
                                # Check if the prediction matches any of the ground truth annotations
                                matched = any(
                                    gt['class'] == pred_class and 
                                    bbox_iou(pred_bbox, gt['bbox']) > iou_threshold
                                    for gt_idx, gt in enumerate(ground_truths) if not matched_ground_truths[gt_idx]
                                )
                        
                                if not matched:
                                    false_positives += 1


        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        fdr = false_positives / (false_positives + true_positives) if (false_positives + true_positives) > 0 else 0
        results.append((threshold, recall, fdr))
        print(f'Threshold: {threshold}, Recall: {recall}, FDR: {fdr}, TP: {true_positives}, FP: {false_positives}, Total GT: {total_ground_truth}')

    return results





# Define class names mapping based on your model's training
class_names = {
    0: 'background',
    1: 'Implant',
    2: 'Fillings',
    3: 'Cavity',
    4: 'Impacted Tooth'
}

# Apply transformations
transforms = T.Compose([T.ToTensor()])

# Load dataset
prediction_dataset = SimpleDataset('PATH_TO_TESTING_DATASET', transforms=transforms)

# Calculate metrics over a range of thresholds
thresholds = [i * 0.1 for i in range(1, 10)]
evaluation_results = predict_and_evaluate(model, prediction_dataset, annotations_dict, class_names, thresholds)

# Plotting
thresholds, recalls, fdrs = zip(*evaluation_results)
print(recalls)
print(fdrs)
plt.figure(figsize=(10, 5))
plt.plot(thresholds, recalls, 'o-', label='Recall')
plt.plot(thresholds, fdrs, 'x-', label='False Discovery Rate')
plt.title('Recall and False Discovery Rate vs. Confidence Threshold')
plt.xlabel('Confidence Threshold')
plt.ylabel('Rate')
plt.yticks([0.2, 0.4, 0.6, 0.8]) 
plt.legend()
plt.grid(True)
plt.savefig('recall_fdr_plot.png')
plt.close()
