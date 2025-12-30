import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Model Definition (same as training)

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


# Load Model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "resqfood_cooked_efficientnet.pt")

model = CookedFoodEfficientNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# Class Mapping (FIXED)

IDX_TO_CLASS = {
    0: "normal",
    1: "spoiled",
    2: "unclear"
}


# Image Transforms

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Prediction Function

def predict(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = IDX_TO_CLASS[pred_idx.item()]
    confidence = round(conf.item(), 4)

    #  NGO-safe decision logic
    if label == "normal":
        decision = "CHECK_TIME_AND_STORAGE"
    else:
        decision = "REJECT_DONATION"

    return {
        "label": label,
        "confidence": confidence,
        "decision": decision
    }

#  Example Run (manual test)

if __name__ == "__main__":
    result = predict(
        r"C:\Users\ilham\Desktop\Ayan\resqfood-food-quality\dataset_cooked\Train\normal\1000117179.jpg"
    )
    print(result)
