
# ResqFood – Food Freshness Training
# EfficientNet-B0 (CPU Friendly)
# Binary Classifier: fresh vs avoid


import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score


#  Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



#  Model Definition

class FoodFreshnessEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)



#  Transforms

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    "dataset_raw/train",
    transform=train_tfms
)

val_ds = datasets.ImageFolder(
    "dataset_raw/val",
    transform=val_tfms
)

print("Class mapping:", train_ds.class_to_idx)
# Must be {'avoid': 0, 'fresh': 1}

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0   # WINDOWS SAFE
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=0
)



#  Loss (Safety-Biased)

avoid_count = sum(1 for _, y in train_ds.samples if y == 0)
fresh_count = sum(1 for _, y in train_ds.samples if y == 1)

pos_weight = torch.tensor(
    [avoid_count / fresh_count],
    device=device
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



#  Model & Optimizer

model = FoodFreshnessEfficientNet().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)



#  Training Function

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        #  progress print every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

    return total_loss / len(loader)




#  Validation Function

def validate(model, loader, threshold=0.75):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            preds.extend((probs >= threshold).cpu().numpy())
            targets.extend(labels.numpy())

    return precision_score(targets, preds)



#  TRAINING LOOP

EPOCHS = 3   

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_precision = validate(model, val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Precision: {val_precision:.4f}"
    )



#  Save Model

torch.save(
    model.state_dict(),
    "resqfood_freshness_efficientnet.pt"
)

print("Training complete. Model saved.")

