from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(
    title="ResqFood Food Safety API",
    description="API for predicting food freshness and safety for donation logistics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "ResqFood Food Safety API",
        "timestamp": datetime.utcnow().isoformat()
    }
@app.get("/version", tags=["Info"])
def get_version():
    return {
        "version": "1.0.0",
        "app": "ResqFood"
    }
