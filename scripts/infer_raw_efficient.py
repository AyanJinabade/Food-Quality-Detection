
# ResqFood – Inference Script
# EfficientNet-B0


import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Model Definition (same as training)

class FoodFreshnessEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)


# Load trained model

model = FoodFreshnessEfficientNet().to(device)
model.load_state_dict(
    torch.load("resqfood_freshness_efficientnet.pt", map_location=device)
)
model.eval()


# Image transforms (NO augmentation)

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Prediction function

def predict(image_path, threshold=0.75):
    img = Image.open(image_path).convert("RGB")
    img = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()

    if prob >= threshold:
        return {
            "label": "fresh",
            "confidence": round(prob, 4),
            "decision": "ACCEPT_FOR_DONATION"
        }
    else:
        return {
            "label": "avoid",
            "confidence": round(prob, 4),
            "decision": "REJECT_DONATION"
        }


# Test run

if __name__ == "__main__":
    result = predict(r"C:\Users\ilham\Desktop\Ayan\resqfood-food-quality\dataset_raw\train\fresh\apple\a_f008.png")
    print(result)
