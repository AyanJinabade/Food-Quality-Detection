
# ResqFood – EfficientNet Validation ONLY


import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Model Definition (same as training)

class FoodFreshnessEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=None   # IMPORTANT: no pretrained weights here
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)


# Load Model

model = FoodFreshnessEfficientNet().to(device)
model.load_state_dict(
    torch.load("resqfood_freshness_efficientnet.pt", map_location=device)
)
model.eval()


# Validation Transforms

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation Dataset

val_ds = datasets.ImageFolder(
    "dataset_raw/val",
    transform=val_tfms
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print("Class mapping:", val_ds.class_to_idx)


# Validation Function

def validate(model, loader, threshold=0.75):
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            preds.extend((probs >= threshold).cpu().numpy())
            targets.extend(labels.numpy())

    return precision_score(targets, preds)


# Run Validation

precision = validate(model, val_loader, threshold=0.75)
print(f"FINAL Validation Precision (fresh): {precision:.4f}")
