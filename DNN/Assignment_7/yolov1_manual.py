import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import yaml
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- 1. UTILITY FUNCTION ---
def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates intersection over union for bounding boxes.
    :param boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
    :param boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    :return: tensor: Intersection over union for all examples
    """
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

# --- 2. YOLOv1 ARCHITECTURE (STRICTLY FROM DIAGRAM) ---

# This configuration list defines the architecture from the diagram.
# Tuple format: ("C", kernel_size, num_filters, stride, padding)
# List format: [("C", ...), ("C", ...), num_repeats] for repeated blocks
# "M" represents a MaxPool layer: ("M", kernel_size, stride)
architecture_config = [
    ("C", 7, 64, 2, 3),
    ("M", 2, 2),
    ("C", 3, 192, 1, 1),
    ("M", 2, 2),
    ("C", 1, 128, 1, 0),
    ("C", 3, 256, 1, 1),
    ("C", 1, 256, 1, 0),
    ("C", 3, 512, 1, 1),
    ("M", 2, 2),
    [("C", 1, 256, 1, 0), ("C", 3, 512, 1, 1), 4], # Block repeated x4
    ("C", 1, 512, 1, 0),
    ("C", 3, 1024, 1, 1),
    ("M", 2, 2),
    [("C", 1, 512, 1, 0), ("C", 3, 1024, 1, 1), 2], # Block repeated x2
    ("C", 3, 1024, 1, 1),
    ("C", 3, 1024, 2, 1),
    ("C", 3, 1024, 1, 1),
    ("C", 3, 1024, 1, 1),
]

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                if x[0] == "C":
                    layers += [
                        nn.Conv2d(in_channels, x[2], kernel_size=x[1], stride=x[3], padding=x[4]),
                        nn.LeakyReLU(0.1, inplace=True),
                    ]
                    in_channels = x[2]
                elif x[0] == "M":
                    layers += [nn.MaxPool2d(kernel_size=x[1], stride=x[2])]
            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        nn.Conv2d(in_channels, conv1[2], kernel_size=conv1[1], stride=conv1[3], padding=conv1[4]),
                        nn.LeakyReLU(0.1, inplace=True),
                        nn.Conv2d(conv1[2], conv2[2], kernel_size=conv2[1], stride=conv2[3], padding=conv2[4]),
                        nn.LeakyReLU(0.1, inplace=True),
                    ]
                    in_channels = conv2[2]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)),
        )

# --- 3. YOLOv1 LOSS FUNCTION ---
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # Reshape predictions to be S x S x (C + B*5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes
        # The first predictor's box starts after the class probabilities (C)
        iou_b1 = intersection_over_union(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5])
        # The second predictor's box starts 5 positions after the first one
        iou_b2 = intersection_over_union(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Select the box with the highest IoU as responsible for the prediction
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., self.C:self.C+1] # Iobj_i in paper (1 if object exists in cell i)

        # === COORDINATE LOSS ===
        # This applies only to the 'responsible' predictor in cells where an object exists
        box_predictions = exists_box * (
            (best_box * predictions[..., self.C+6:self.C+10]) + ((1 - best_box) * predictions[..., self.C+1:self.C+5])
        )
        box_targets = exists_box * target[..., self.C+1:self.C+5]

        # Use sign-safe sqrt for width and height to penalize errors in small boxes more
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # === OBJECT LOSS (Confidence of box with object) ===
        pred_box = best_box * predictions[..., self.C+5:self.C+6] + (1 - best_box) * predictions[..., self.C:self.C+1]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C+1])
        )

        # === NO OBJECT LOSS (Confidence of boxes without objects) ===
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        # === CLASS LOSS ===
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        # === FINAL LOSS ===
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss / len(predictions)

# --- 4. CUSTOM DATASET LOADER ---
class WiderFaceDataset(Dataset):
    def __init__(self, yaml_file, split="train", S=7, B=2, C=1, transform=None):
        with open(yaml_file) as f:
            data_config = yaml.safe_load(f)

        self.data_root = data_config['path']
        self.split_dir = data_config[split]

        self.img_dir = os.path.join(self.data_root, self.split_dir, "images")
        self.label_dir = os.path.join(self.data_root, self.split_dir, "labels")
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        self.annotations = []
        for img_file in sorted(os.listdir(self.img_dir)):
            label_file = os.path.splitext(img_file)[0] + ".txt"
            if os.path.exists(os.path.join(self.label_dir, label_file)):
                self.annotations.append({"image": img_file, "label": label_file})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations[index]["image"])
        label_path = os.path.join(self.label_dir, self.annotations[index]["label"])

        image = Image.open(img_path).convert("RGB")

        # In YOLO format, labels are space-separated: class_id x_center y_center width height
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, w, h])

        if self.transform:
            image = self.transform(image)

        # Convert bounding boxes to target tensor format S x S x (C+5)
        # Note: In the target, there is only one bounding box per cell, so size is C+5
        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_label, x, y, w, h = box
            class_label = int(class_label)

            # Find which grid cell the center of the box belongs to
            i, j = int(self.S * y), int(self.S * x)
            # Find coordinates relative to the grid cell
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S # width and height relative to image size

            # If a cell is not already occupied (a limitation of YOLOv1)
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1 # Set confidence to 1
                label_matrix[i, j, class_label] = 1 # Set one-hot encoded class
                label_matrix[i, j, self.C+1:] = torch.tensor([x_cell, y_cell, w, h]) # Set coordinates

        return image, label_matrix

# --- 5. TRAINING SCRIPT ---
def train_fn(train_loader, model, optimizer, loss_fn, device):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    # Hyperparameters
    LEARNING_RATE = 5e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 20 # Training YOLOv1 from scratch is slow, needs many epochs
    YAML_FILE = "widerface.yaml"
    IMG_SIZE = 448
    S, B, C = 7, 2, 1 # Grid size, num boxes, num classes

    print(f"Training on device: {DEVICE}")

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Initialize model, loss, optimizer
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = YoloLoss(S=S, B=B, C=C)

    # Load datasets
    train_dataset = WiderFaceDataset(
        yaml_file=YAML_FILE,
        split="train",
        transform=transform,
        S=S, B=B, C=C
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True, # Drop last batch if it's smaller than BATCH_SIZE
        num_workers=2,
        pin_memory=True
    )

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"--- EPOCH {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, DEVICE)

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"yolov1_widerface_epoch_{epoch+1}.pth")

    print("Training finished.")
    torch.save(model.state_dict(), "yolov1_widerface_final.pth")

if __name__ == "__main__":
    main()