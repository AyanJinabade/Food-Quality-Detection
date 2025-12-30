
# ResqFood – FastAPI Docker Image
# CPU-only, inference-only


FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only needed files
COPY scripts/ scripts/
COPY resqfood_freshness_efficientnet.pt .
COPY resqfood_cooked_efficientnet.pt .

EXPOSE 8000

CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
