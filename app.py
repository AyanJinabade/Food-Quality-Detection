from fastapi import FastAPI

app = FastAPI(
    title="ResqFood Food Safety API",
    description="API for predicting food freshness and safety for donation logistics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get("/")
def health_check():
    return {
        "status": "ResqFood API is running",
        "service": "Food Safety Prediction API"
    }
