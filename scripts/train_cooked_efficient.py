
# ResqFood – Cooked Food Training
# EfficientNet-B0 (CPU Safe)
# Classes: normal / spoiled / unclear


import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np


#  Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#  Model Definition

class CookedFoodEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


#  Transforms
# 
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#  Dataset & Loaders

train_ds = datasets.ImageFolder(
    "dataset_cooked/train",
    transform=train_tfms
)

val_ds = datasets.ImageFolder(
    "dataset_cooked/val",
    transform=val_tfms
)

print("Class mapping:", train_ds.class_to_idx)
# MUST be {'normal': 0, 'spoiled': 1, 'unclear': 2}

train_loader = DataLoader(
    train_ds,
    batch_size=8,      
    shuffle=True,
    num_workers=0      
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


#  Loss Function

criterion = nn.CrossEntropyLoss()


# Model & Optimizer

model = CookedFoodEfficientNet().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


#  Training Function

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


#  Validation Function

def validate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            predicted = torch.argmax(logits, dim=1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    return accuracy_score(targets, preds)


#  Training Loop

EPOCHS = 5   # Enough for cooked food

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_acc = validate(model, val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Accuracy: {val_acc:.4f}"
    )


#  Save Model

torch.save(
    model.state_dict(),
    "resqfood_cooked_efficientnet.pt"
)

print(" Cooked food model training complete.")
