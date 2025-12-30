
# ResqFood – Unified FastAPI Backend
# Raw + Cooked Food Decision API

import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse


# 🔧 FIX PYTHON PATH (CRITICAL FOR WINDOWS + UVICORN)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from decision_engine import unified_decision


# App init

app = FastAPI(
    title="ResqFood Food Safety API",
    description="Image-based food donation decision system",
    version="1.0"
)


# Food categories

RAW_FOODS = {
    "apple", "banana", "tomato", "potato",
    "capsicum", "bitter_gourd"
}

COOKED_FOODS = {
    "bread", "rice", "boiled_egg", "chapati"
}


# Temp upload dir

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Health check

@app.get("/")
def health():
    return {"status": "ResqFood API is running"}


# Prediction endpoint

@app.post("/predict")
async def predict_food(
    image: UploadFile = File(...),
    food_type: str = Form(...),
    hours_since_cooked: int | None = Form(None),
    storage: str | None = Form(None)
):
    image_path = os.path.join(UPLOAD_DIR, image.filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    food_type = food_type.lower()

    try:
        
        if food_type in RAW_FOODS:
            result = unified_decision(
                image_path=image_path,
                food_type=food_type
            )

        
        elif food_type in COOKED_FOODS:
            if hours_since_cooked is None or storage is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "hours_since_cooked and storage are required for cooked food"
                    }
                )

            result = unified_decision(
                image_path=image_path,
                food_type=food_type,
                hours_since_cooked=hours_since_cooked,
                storage=storage.lower()
            )

        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported food type: {food_type}"}
            )

        return result

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
