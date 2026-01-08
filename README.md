# Food-Quality-Detection
Food waste and unsafe food donation is a major issue.
This system helps NGOs and food donors decide whether.
food is safe to donate using AI-based image analysis.

 WHAT I BUILT:
 FastAPI backend for food safety prediction.
 Image-based ML inference (EfficientNet).
 Dockerized production deployment.
 Public REST API with Swagger docs.

LIVE DEMO:
Live API: https://theoretical-amye-resqfood-af38cc63.koyeb.app
Swagger Docs: /docs

Tech Stack:
Python, FastAPI, PyTorch, Docker, Koyeb, EfficientNet

Challenges Solved
Cloud deployment failures due to image size limits
Reverse proxy issues with API docs
Health check failures on free-tier platforms
Docker image optimization for ML workloads
These were resolved by:
Project restructuring
Proper Docker configuration
Switching to a suitable cloud platform (Koyeb)

Future Improvements
Add authentication for NGO access
Improve model accuracy with more data
Add a simple web UI for image upload
Log predictions for monitoring
Proper Docker configuration

Switching to a suitable cloud platform (Koyeb)
