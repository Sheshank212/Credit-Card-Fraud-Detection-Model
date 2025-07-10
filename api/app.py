"""
Fraud Detection API
"""

# importing all the necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict
import shap
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Add src to path for logging config
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from logging_config import setup_logging

# Setup structured logging
fraud_logger = setup_logging("INFO")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection for credit card transactions",
    version="1.0.0",
)

# Global variables
models = {}
scaler = None
model_performance = {}
explainer = None
background_data = None

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "api_request_duration_seconds", "API request duration", ["method", "endpoint"]
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total", "Total fraud predictions", ["prediction", "model"]
)
MODEL_PREDICTION_TIME = Histogram(
    "model_prediction_duration_seconds", "Model prediction time", ["model"]
)
ACTIVE_MODELS = Gauge("active_models_count", "Number of loaded models")
MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy score", ["model"])
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total", "Total prediction errors", ["model", "error_type"]
)


class TransactionRequest(BaseModel):
    """Transaction data for fraud prediction"""

    V1: float = Field(..., description="PCA Feature V1")
    V2: float = Field(..., description="PCA Feature V2")
    V3: float = Field(..., description="PCA Feature V3")
    V4: float = Field(..., description="PCA Feature V4")
    V5: float = Field(..., description="PCA Feature V5")
    V6: float = Field(..., description="PCA Feature V6")
    V7: float = Field(..., description="PCA Feature V7")
    V8: float = Field(..., description="PCA Feature V8")
    V9: float = Field(..., description="PCA Feature V9")
    V10: float = Field(..., description="PCA Feature V10")
    V11: float = Field(..., description="PCA Feature V11")
    V12: float = Field(..., description="PCA Feature V12")
    V13: float = Field(..., description="PCA Feature V13")
    V14: float = Field(..., description="PCA Feature V14")
    V15: float = Field(..., description="PCA Feature V15")
    V16: float = Field(..., description="PCA Feature V16")
    V17: float = Field(..., description="PCA Feature V17")
    V18: float = Field(..., description="PCA Feature V18")
    V19: float = Field(..., description="PCA Feature V19")
    V20: float = Field(..., description="PCA Feature V20")
    V21: float = Field(..., description="PCA Feature V21")
    V22: float = Field(..., description="PCA Feature V22")
    V23: float = Field(..., description="PCA Feature V23")
    V24: float = Field(..., description="PCA Feature V24")
    V25: float = Field(..., description="PCA Feature V25")
    V26: float = Field(..., description="PCA Feature V26")
    V27: float = Field(..., description="PCA Feature V27")
    V28: float = Field(..., description="PCA Feature V28")
    Amount: float = Field(..., description="Transaction amount", ge=0)


class FraudPrediction(BaseModel):
    """Fraud prediction response"""

    is_fraud: bool
    fraud_probability: float
    risk_level: str
    algorithm_used: str
    prediction_time_ms: float
    recommendation: str


class FraudPredictionWithExplanation(BaseModel):
    """Fraud prediction with model explanation"""

    is_fraud: bool
    fraud_probability: float
    risk_level: str
    algorithm_used: str
    prediction_time_ms: float
    recommendation: str
    explanation: Dict[str, float]
    top_risk_factors: List[Dict[str, float]]


def load_models():
    """Load trained models and scaler"""
    global models, scaler, model_performance, explainer, background_data

    start_time = time.time()
    try:
        # Set working directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_dir)

        # Load models
        models["logistic_regression"] = joblib.load("models/logistic_regression.pkl")
        models["random_forest"] = joblib.load("models/random_forest.pkl")
        models["xgboost"] = joblib.load("models/xgboost.pkl")

        # Load scaler
        scaler = joblib.load("models/scaler.pkl")

        # Load actual model performance metrics
        try:
            import pandas as pd

            perf_df = pd.read_csv("models/model_performance.csv", index_col=0)
            model_performance = {
                "logistic_regression": {
                    "auc": round(perf_df.loc["logistic_regression", "AUC"], 3),
                    "precision": round(
                        perf_df.loc["logistic_regression", "Precision"], 3
                    ),
                    "recall": round(perf_df.loc["logistic_regression", "Recall"], 3),
                },
                "random_forest": {
                    "auc": round(perf_df.loc["random_forest", "AUC"], 3),
                    "precision": round(perf_df.loc["random_forest", "Precision"], 3),
                    "recall": round(perf_df.loc["random_forest", "Recall"], 3),
                },
                "xgboost": {
                    "auc": round(perf_df.loc["xgboost", "AUC"], 3),
                    "precision": round(perf_df.loc["xgboost", "Precision"], 3),
                    "recall": round(perf_df.loc["xgboost", "Recall"], 3),
                },
            }
        except Exception as e:
            logger.error(f"Failed to load performance metrics: {e}")
            # Fallback to current actual values
            model_performance = {
                "logistic_regression": {
                    "auc": 0.990,
                    "precision": 0.167,
                    "recall": 0.967,
                },
                "random_forest": {"auc": 0.999, "precision": 0.963, "recall": 0.867},
                "xgboost": {"auc": 0.995, "precision": 0.903, "recall": 0.933},
            }

        # Initialize SHAP explainer with XGBoost (best model)
        try:
            if "xgboost" in models:
                # Create small background dataset for SHAP
                background_data = np.random.normal(0, 1, (100, 30))  # 30 features
                explainer = shap.TreeExplainer(models["xgboost"])
                logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")

        load_time = time.time() - start_time
        fraud_logger.log_model_load(
            model_names=list(models.keys()), load_time_seconds=load_time
        )

        # Update Prometheus metrics
        ACTIVE_MODELS.set(len(models))
        for model_name, performance in model_performance.items():
            MODEL_ACCURACY.labels(model=model_name).set(performance["auc"])

        logger.info(f"Successfully loaded {len(models)} models")

    except Exception as e:
        fraud_logger.log_error(
            error_type="ModelLoadError",
            error_message=str(e),
            context={
                "models_requested": ["logistic_regression", "random_forest", "xgboost"]
            },
        )
        logger.error(f"Error loading models: {e}")
        raise


def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability >= 0.8:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    elif probability >= 0.3:
        return "LOW"
    else:
        return "VERY_LOW"


def get_recommendation(probability: float, amount: float) -> str:
    """Business recommendation based on risk assessment"""
    if probability >= 0.8:
        return "BLOCK_TRANSACTION - High fraud risk detected"
    elif probability >= 0.5:
        return "MANUAL_REVIEW - Additional verification required"
    elif probability >= 0.3:
        return "FLAG_FOR_MONITORING - Monitor account activity"
    else:
        return "APPROVE - Transaction appears legitimate"


@app.on_event("startup")
async def startup_event():
    """Load models on application startup"""
    load_models()


@app.get("/")
async def root():
    """API health check and information"""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
async def get_model_info():
    """Get information about available models"""
    return {
        "available_models": list(models.keys()),
        "model_performance": model_performance,
        "default_model": "xgboost",
    }


@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionRequest, model_name: str = "xgboost"):
    """
    Predict fraud probability for a credit card transaction

    Returns fraud probability, risk level, and business recommendation
    """
    start_time = time.time()

    try:
        # Validate model exists
        if model_name not in models:
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="400").inc()
            raise HTTPException(
                status_code=400, detail=f"Model '{model_name}' not available"
            )

        # Engineer features (Amount_log, Time_hour defaults)
        amount_log = np.log1p(transaction.Amount)
        time_hour = (
            12.0  # Default noon hour (would be extracted from timestamp in production)
        )

        # Combine all features
        raw_features = [getattr(transaction, f"V{i}") for i in range(1, 29)] + [
            amount_log,
            time_hour,
        ]

        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform([raw_features])
        else:
            raise HTTPException(status_code=500, detail="Scaler not loaded")

        # Make prediction
        model = models[model_name]
        fraud_probability = model.predict_proba(features_scaled)[0, 1]
        is_fraud = fraud_probability >= 0.5

        # Calculate response time
        prediction_time_ms = (time.time() - start_time) * 1000

        # Business logic
        risk_level = get_risk_level(fraud_probability)
        recommendation = get_recommendation(fraud_probability, transaction.Amount)

        # Log prediction for monitoring
        fraud_logger.log_prediction(
            model_name=model_name,
            fraud_probability=fraud_probability,
            prediction_time_ms=prediction_time_ms,
        )

        # Update Prometheus metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
        FRAUD_PREDICTIONS.labels(
            prediction="fraud" if is_fraud else "legitimate", model=model_name
        ).inc()

        logger.info(
            f"Prediction: {fraud_probability:.4f}, "
            f"Model: {model_name}, Time: {prediction_time_ms:.2f}ms"
        )

        return FraudPrediction(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_probability, 4),
            risk_level=risk_level,
            algorithm_used=model_name,
            prediction_time_ms=round(prediction_time_ms, 2),
            recommendation=recommendation,
        )

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        PREDICTION_ERRORS.labels(model=model_name, error_type="prediction_failed").inc()

        fraud_logger.log_error(
            error_type="PredictionError",
            error_message=str(e),
            context={"model_name": model_name, "endpoint": "/predict"},
        )

        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/explain", response_model=FraudPredictionWithExplanation)
async def predict_fraud_with_explanation(
    transaction: TransactionRequest, model_name: str = "xgboost"
):
    """
    Predict fraud probability with model explanation using SHAP values

    Returns fraud probability, risk level, business recommendation, and feature importance
    """
    start_time = time.time()

    try:
        # Validate model exists
        if model_name not in models:
            raise HTTPException(
                status_code=400, detail=f"Model '{model_name}' not available"
            )

        # Check if explainer is available
        if explainer is None or model_name != "xgboost":
            raise HTTPException(
                status_code=400,
                detail="Model explanation only available for XGBoost model",
            )

        # Prepare features
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Time_hour"]

        # Engineer features
        amount_log = np.log1p(transaction.Amount)
        time_hour = 12.0  # Default noon hour

        # Combine all features
        raw_features = [getattr(transaction, f"V{i}") for i in range(1, 29)] + [
            amount_log,
            time_hour,
        ]

        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform([raw_features])
        else:
            raise HTTPException(status_code=500, detail="Scaler not loaded")

        # Make prediction
        model = models[model_name]
        fraud_probability = model.predict_proba(features_scaled)[0, 1]
        is_fraud = fraud_probability >= 0.5

        # Calculate SHAP values for explanation
        shap_values = explainer.shap_values(features_scaled)

        # Get SHAP values for fraud class (class 1)
        if len(shap_values.shape) == 3:  # Multi-class case
            fraud_shap_values = shap_values[1][0]  # Class 1 (fraud)
        else:  # Binary case
            fraud_shap_values = shap_values[0]

        # Create explanation dictionary
        explanation = {}
        for i, feature_name in enumerate(feature_names):
            explanation[feature_name] = float(fraud_shap_values[i])

        # Get top risk factors (features with highest positive SHAP values)
        top_risk_factors = []
        sorted_features = sorted(
            explanation.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for feature, importance in sorted_features[:10]:  # Top 10 factors
            top_risk_factors.append(
                {
                    "feature": feature,
                    "importance": float(importance),
                    "impact": (
                        "increases_fraud_risk"
                        if importance > 0
                        else "decreases_fraud_risk"
                    ),
                }
            )

        # Calculate response time
        prediction_time_ms = (time.time() - start_time) * 1000

        # Business logic
        risk_level = get_risk_level(fraud_probability)
        recommendation = get_recommendation(fraud_probability, transaction.Amount)

        # Log prediction for monitoring
        logger.info(
            f"Explained Prediction: {fraud_probability:.4f}, "
            f"Model: {model_name}, Time: {prediction_time_ms:.2f}ms"
        )

        return FraudPredictionWithExplanation(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_probability, 4),
            risk_level=risk_level,
            algorithm_used=model_name,
            prediction_time_ms=round(prediction_time_ms, 2),
            recommendation=recommendation,
            explanation=explanation,
            top_risk_factors=top_risk_factors,
        )

    except Exception as e:
        logger.error(f"Explained prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Explained prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_fraud_batch(
    transactions: List[TransactionRequest], model_name: str = "xgboost"
):
    """
    Batch fraud prediction for multiple transactions
    """
    start_time = time.time()

    try:
        if model_name not in models:
            raise HTTPException(
                status_code=400, detail=f"Model '{model_name}' not available"
            )

        if len(transactions) > 1000:
            raise HTTPException(
                status_code=400, detail="Maximum 1000 transactions per batch"
            )

        results = []
        for i, transaction in enumerate(transactions):
            # Engineer features
            amount_log = np.log1p(transaction.Amount)
            time_hour = 12.0

            raw_features = [getattr(transaction, f"V{i}") for i in range(1, 29)] + [
                amount_log,
                time_hour,
            ]
            features_scaled = scaler.transform([raw_features])

            # Predict
            model = models[model_name]
            fraud_probability = model.predict_proba(features_scaled)[0, 1]

            results.append(
                {
                    "transaction_id": i,
                    "is_fraud": fraud_probability >= 0.5,
                    "fraud_probability": round(fraud_probability, 4),
                    "risk_level": get_risk_level(fraud_probability),
                    "recommendation": get_recommendation(
                        fraud_probability, transaction.Amount
                    ),
                }
            )

        processing_time_ms = (time.time() - start_time) * 1000
        fraud_count = sum(1 for r in results if r["is_fraud"])

        logger.info(
            f"Batch processed: {len(transactions)} transactions, {fraud_count} fraud detected"
        )

        return {
            "predictions": results,
            "algorithm_used": model_name,
            "total_transactions": len(transactions),
            "fraud_detected": fraud_count,
            "processing_time_ms": round(processing_time_ms, 2),
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Detailed system health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "scaler_loaded": scaler is not None,
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
