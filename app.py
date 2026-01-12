from fastapi import FastAPI
app = FastAPI(
    title="ResqFood Food Safety API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get("/")
def health():
    return {"status": "ResqFood API is running"}
