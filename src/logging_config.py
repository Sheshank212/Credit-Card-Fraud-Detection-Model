"""
Structured JSON logging configuration for fraud detection system
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import os


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def __init__(self, include_fields: Optional[list] = None):
        super().__init__()
        self.include_fields = include_fields or [
            "timestamp",
            "level",
            "logger",
            "message",
            "module",
            "function",
            "line",
        ]

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": os.getpid(),
            "thread_id": record.thread,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        # Filter fields if specified
        if self.include_fields:
            filtered_entry = {
                k: v
                for k, v in log_entry.items()
                if k in self.include_fields or k.startswith("custom_")
            }
            log_entry = filtered_entry

        return json.dumps(log_entry, default=str)


class FraudDetectionLogger:
    """Centralized logger for fraud detection system"""

    def __init__(self, name: str = "fraud_detection", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Setup console handler with JSON formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)

        # Setup file handler for persistent logging
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(f"{log_dir}/fraud_detection.log")
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

    def log_model_training(
        self, model_name: str, metrics: Dict[str, Any], duration: float
    ):
        """Log model training information"""
        self.logger.info(
            "Model training completed",
            extra={
                "event_type": "model_training",
                "model_name": model_name,
                "auc_score": metrics.get("AUC"),
                "precision": metrics.get("Precision"),
                "recall": metrics.get("Recall"),
                "f1_score": metrics.get("F1"),
                "net_benefit": metrics.get("Net_Benefit"),
                "training_duration_seconds": duration,
            },
        )

    def log_prediction(
        self,
        model_name: str,
        fraud_probability: float,
        prediction_time_ms: float,
        features: Optional[Dict] = None,
    ):
        """Log fraud prediction"""
        self.logger.info(
            "Fraud prediction made",
            extra={
                "event_type": "fraud_prediction",
                "model_name": model_name,
                "fraud_probability": fraud_probability,
                "is_fraud": fraud_probability >= 0.5,
                "prediction_time_ms": prediction_time_ms,
                "feature_count": len(features) if features else None,
            },
        )

    def log_batch_prediction(
        self,
        model_name: str,
        transaction_count: int,
        fraud_detected: int,
        processing_time_ms: float,
    ):
        """Log batch prediction results"""
        self.logger.info(
            "Batch prediction completed",
            extra={
                "event_type": "batch_prediction",
                "model_name": model_name,
                "transaction_count": transaction_count,
                "fraud_detected": fraud_detected,
                "fraud_rate": (
                    fraud_detected / transaction_count if transaction_count > 0 else 0
                ),
                "processing_time_ms": processing_time_ms,
                "avg_time_per_transaction_ms": (
                    processing_time_ms / transaction_count
                    if transaction_count > 0
                    else 0
                ),
            },
        )

    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        user_agent: Optional[str] = None,
    ):
        """Log API request"""
        self.logger.info(
            "API request processed",
            extra={
                "event_type": "api_request",
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time_ms": response_time_ms,
                "user_agent": user_agent,
            },
        )

    def log_model_load(self, model_names: list, load_time_seconds: float):
        """Log model loading"""
        self.logger.info(
            "Models loaded successfully",
            extra={
                "event_type": "model_load",
                "model_names": model_names,
                "model_count": len(model_names),
                "load_time_seconds": load_time_seconds,
            },
        )

    def log_error(
        self, error_type: str, error_message: str, context: Optional[Dict] = None
    ):
        """Log structured error information"""
        self.logger.error(
            f"Error occurred: {error_message}",
            extra={
                "event_type": "error",
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
            },
        )

    def log_performance_metric(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ):
        """Log performance metrics for monitoring"""
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                "event_type": "performance_metric",
                "metric_name": metric_name,
                "metric_value": value,
                "tags": tags or {},
            },
        )

    def log_data_processing(
        self, operation: str, record_count: int, processing_time_seconds: float
    ):
        """Log data processing operations"""
        self.logger.info(
            f"Data processing: {operation}",
            extra={
                "event_type": "data_processing",
                "operation": operation,
                "record_count": record_count,
                "processing_time_seconds": processing_time_seconds,
                "records_per_second": (
                    record_count / processing_time_seconds
                    if processing_time_seconds > 0
                    else 0
                ),
            },
        )


def setup_logging(
    log_level: str = "INFO", log_to_file: bool = True
) -> FraudDetectionLogger:
    """Setup and return configured logger"""
    return FraudDetectionLogger(log_level=log_level)


def get_logger(name: str = "fraud_detection") -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


# Example usage and testing
if __name__ == "__main__":
    # Test the logging configuration
    fraud_logger = setup_logging("DEBUG")

    # Test different log types
    fraud_logger.log_model_training(
        model_name="xgboost",
        metrics={
            "AUC": 0.9876,
            "Precision": 0.89,
            "Recall": 0.91,
            "F1": 0.90,
            "Net_Benefit": 1250.0,
        },
        duration=125.5,
    )

    fraud_logger.log_prediction(
        model_name="xgboost", fraud_probability=0.8523, prediction_time_ms=4.2
    )

    fraud_logger.log_batch_prediction(
        model_name="random_forest",
        transaction_count=500,
        fraud_detected=3,
        processing_time_ms=156.7,
    )

    fraud_logger.log_api_request(
        endpoint="/predict", method="POST", status_code=200, response_time_ms=8.9
    )

    fraud_logger.log_performance_metric(
        metric_name="model_accuracy",
        value=0.9876,
        tags={"model": "xgboost", "dataset": "validation"},
    )

    fraud_logger.log_error(
        error_type="ModelLoadError",
        error_message="Failed to load XGBoost model",
        context={"model_path": "/models/xgboost.pkl", "file_exists": False},
    )

    print("Logging test completed. Check logs/fraud_detection.log for output.")
