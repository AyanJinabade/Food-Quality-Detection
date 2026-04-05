Food Quality Detection System

AI-powered system to assess food safety and reduce unsafe food donations using image-based analysis.
Overview
Food waste and unsafe food donation are critical challenges. This project provides an intelligent solution that helps NGOs and food donors determine whether food is safe for consumption using deep learning.

The system leverages image-based machine learning to classify food quality and supports real-time decision-making through a deployed API.

Key Features
AI-based food quality detection using image inputs
Real-time prediction via REST API
High-performance model using EfficientNet
Fully Dockerized for scalable deployment
Interactive API documentation with Swagger

What I Built
Backend API using FastAPI
Deep learning model using PyTorch
Image classification using EfficientNet architecture
Containerized deployment using Docker
Cloud deployment on Koyeb

Live Demo
API Endpoint: https://theoretical-amye-resqfood-af38cc63.koyeb.app
Swagger Docs: /docs

Tech Stack
Python
FastAPI
PyTorch
Docker
Koyeb
EfficientNet

Challenges & Solutions
Deployment Challenges
Docker image too large for cloud platforms
Reverse proxy issues affecting API documentation
Health check failures on free-tier hosting

Solutions Implemented
Optimized Docker image size
Restructured project for efficient deployment
Configured proper health endpoints
Migrated deployment to a more suitable platform (Koyeb)

Impact
Enables NGOs to make faster and safer food collection decisions
Reduces risk of distributing unsafe food
Contributes to minimizing food waste through better filtering
