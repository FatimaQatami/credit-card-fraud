# Credit Card Fraud Detection

This project implements multiple machine learning models for credit card fraud detection.
The repository includes data preprocessing, feature engineering, model training, evaluation,
and a FastAPI-based inference API.

## Structure
- api/ : FastAPI application for model inference
- src/ : training and evaluation code for models
- models/ : model-related code (artifacts excluded due to size)

## API
Run locally:
uvicorn api.main:app --reload

Test:
http://127.0.0.1:8000/docs

## Notes
Trained model artifacts (.cbm, .joblib) are not included due to GitHub size limits.
They are available upon request.
