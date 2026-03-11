import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model Definition

class FoodFreshnessEfficientNet(nn.Module):
    def __init__(self):
        super(FoodFreshnessEfficientNet, self).__init__()

        self.model = models.efficientnet_b0(weights=None)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)



# Load Model
MODEL_PATH = "resqfood_freshness_efficientnet.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = FoodFreshnessEfficientNet().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully")


# Image Transform
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Prediction Function
def predict(image_path, threshold=0.75):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    img = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()

    if prob >= threshold:
        result = {
            "label": "fresh",
            "confidence": round(prob, 4),
            "decision": "ACCEPT_FOR_DONATION"
        }
    else:
        result = {
            "label": "avoid",
            "confidence": round(prob, 4),
            "decision": "REJECT_DONATION"
        }

    return result

# Test Run

if __name__ == "__main__":

    image_path = r"C:\Users\ilham\Desktop\Ayan\resqfood-food-quality\dataset_raw\train\fresh\apple\a_f008.png"

    result = predict(image_path)

    print("\nPrediction Result")
    print(result)
