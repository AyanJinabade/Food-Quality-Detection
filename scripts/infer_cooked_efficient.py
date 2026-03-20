import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CONF_THRESHOLD = 0.55
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "resqfood_cooked_efficientnet.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = CookedFoodEfficientNet().to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.eval()
print(" Model loaded successfully")

IDX_TO_CLASS = {
    0: "normal",
    1: "spoiled",
    2: "unclear"
}

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICT ----------------
def predict(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Invalid image: {e}")

    img = tfms(img).unsqueeze(0).to(device)

    with torch.inference_mode():   # faster than no_grad
        logits = model(img)
        probs = torch.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)

    label = IDX_TO_CLASS.get(pred_idx.item(), "unknown")
    confidence = float(round(confidence.item(), 4))

    if confidence < CONF_THRESHOLD:
        decision = "REJECT_UNCLEAR_IMAGE"
        reason = "Low confidence prediction"
        safe = False

    elif label == "spoiled":
        decision = "REJECT_SPOILED_FOOD"
        reason = "Detected spoiled food"
        safe = False

    elif label == "unclear":
        decision = "REJECT_UNCLEAR_IMAGE"
        reason = "Image unclear"
        safe = False

    elif label == "normal":
        decision = "ACCEPT_FOOD"
        reason = "Food appears safe"
        safe = True

    else:
        decision = "REJECT_UNKNOWN"
        reason = "Unknown classification"
        safe = False

    return {
        "label": label,
        "confidence": confidence,
        "decision": decision,
        "safe_for_donation": safe,
        "reason": reason
    }

if __name__ == "__main__":

    test_image = r"C:\Users\ilham\Desktop\Ayan\resqfood-food-quality\dataset_cooked\Train\normal\1000117179.jpg"

    result = predict(test_image)

    print(result)
