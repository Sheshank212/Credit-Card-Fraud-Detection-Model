"""
Unit tests for fraud detection system
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_detector import FraudDetectionSystem


class TestFraudDetectionSystem:
    """Test suite for FraudDetectionSystem class"""

    @pytest.fixture
    def fraud_detector(self):
        """Create a FraudDetectionSystem instance for testing"""
        return FraudDetectionSystem(random_state=42)

    @pytest.fixture
    def sample_data(self):
        """Create sample fraud detection data for testing"""
        np.random.seed(42)
        n_samples = 1000

        # Create sample data with required columns
        data = {
            "Time": np.random.uniform(0, 172800, n_samples),  # 2 days in seconds
            "Amount": np.random.exponential(50, n_samples),
            "Class": np.random.choice(
                [0, 1], n_samples, p=[0.99, 0.01]
            ),  # Imbalanced but with some fraud cases
        }

        # Add V1-V28 PCA features
        for i in range(1, 29):
            data[f"V{i}"] = np.random.normal(0, 1, n_samples)

        return pd.DataFrame(data)

    def test_initialization(self, fraud_detector):
        """Test FraudDetectionSystem initialization"""
        assert fraud_detector.random_state == 42
        assert fraud_detector.models == {}
        assert fraud_detector.scaler is None
        assert fraud_detector.performance_metrics == {}

    def test_load_data_full(self, fraud_detector, sample_data, tmp_path):
        """Test loading full dataset"""
        # Create temporary CSV file
        temp_file = tmp_path / "test_data.csv"
        sample_data.to_csv(temp_file, index=False)

        # Test loading
        df = fraud_detector.load_data(str(temp_file))

        assert len(df) == 1000
        assert "Class" in df.columns
        assert "Amount" in df.columns
        assert all(f"V{i}" in df.columns for i in range(1, 29))

    def test_load_data_sample(self, fraud_detector, sample_data, tmp_path):
        """Test loading sampled dataset"""
        # Create temporary CSV file
        temp_file = tmp_path / "test_data.csv"
        sample_data.to_csv(temp_file, index=False)

        # Test sampling
        df = fraud_detector.load_data(str(temp_file), sample_size=500)

        assert len(df) == 500
        assert "Class" in df.columns

    def test_preprocess_data(self, fraud_detector, sample_data):
        """Test data preprocessing pipeline"""
        X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(sample_data)

        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) + len(y_test) == len(sample_data)

        # Check feature engineering
        assert "Amount_log" in X_train.columns
        assert "Time_hour" in X_train.columns

        # Check scaling
        assert fraud_detector.scaler is not None

        # Check stratification (approximate)
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        original_fraud_rate = sample_data["Class"].mean()

        # Allow some tolerance for small samples
        assert abs(train_fraud_rate - original_fraud_rate) < 0.01
        assert abs(test_fraud_rate - original_fraud_rate) < 0.01

    def test_feature_engineering(self, fraud_detector, sample_data):
        """Test specific feature engineering functions"""
        # Create a small sample for testing
        test_sample = sample_data.head(10).copy()

        X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(test_sample)

        # Test Amount_log transformation
        original_amounts = test_sample["Amount"].values
        _ = np.log1p(original_amounts)  # Expected log transformation

        # Check if log transformation is reasonable
        assert "Amount_log" in X_train.columns
        assert all(X_train["Amount_log"] >= 0)  # log1p should be non-negative

        # Test Time_hour extraction
        assert "Time_hour" in X_train.columns
        assert all(0 <= hour < 24 for hour in X_train["Time_hour"])

    def test_train_models_structure(self, fraud_detector, sample_data):
        """Test model training structure and outputs"""
        X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(sample_data)

        # Train models
        models = fraud_detector.train_models(X_train, y_train)

        # Check all expected models are trained
        expected_models = ["logistic_regression", "random_forest", "xgboost"]
        assert all(model_name in models for model_name in expected_models)
        assert all(
            model_name in fraud_detector.models for model_name in expected_models
        )

        # Check models can make predictions
        for model_name, model in models.items():
            assert hasattr(model, "predict")
            assert hasattr(model, "predict_proba")

            # Test prediction shapes
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)

            assert len(predictions) == len(X_test)
            assert probabilities.shape == (len(X_test), 2)

    def test_evaluate_models_structure(self, fraud_detector, sample_data):
        """Test model evaluation structure and metrics"""
        X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(sample_data)
        fraud_detector.train_models(X_train, y_train)

        # Evaluate models
        results = fraud_detector.evaluate_models(X_test, y_test)

        # Check results structure
        expected_models = ["logistic_regression", "random_forest", "xgboost"]
        assert all(model_name in results for model_name in expected_models)

        # Check metrics for each model
        expected_metrics = [
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "Net_Benefit",
            "TP",
            "FP",
            "TN",
            "FN",
        ]
        for model_name, metrics in results.items():
            assert all(metric in metrics for metric in expected_metrics)

            # Check metric ranges
            assert 0 <= metrics["AUC"] <= 1
            assert 0 <= metrics["Precision"] <= 1
            assert 0 <= metrics["Recall"] <= 1
            assert 0 <= metrics["F1"] <= 1

            # Check confusion matrix values are non-negative integers
            confusion_metrics = ["TP", "FP", "TN", "FN"]
            assert all(
                isinstance(metrics[metric], (int, np.integer)) and metrics[metric] >= 0
                for metric in confusion_metrics
            )

    def test_business_metrics_calculation(self, fraud_detector, sample_data):
        """Test business impact calculations"""
        X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(sample_data)
        fraud_detector.train_models(X_train, y_train)
        results = fraud_detector.evaluate_models(X_test, y_test)

        # Test business logic for each model
        for model_name, metrics in results.items():
            tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]

            # Calculate expected net benefit
            avg_fraud_amount = 100
            investigation_cost = 50
            expected_money_saved = tp * avg_fraud_amount
            expected_total_cost = (tp + fp) * investigation_cost + fn * avg_fraud_amount
            expected_net_benefit = expected_money_saved - expected_total_cost

            assert metrics["Net_Benefit"] == expected_net_benefit

    @patch("joblib.dump")
    def test_save_models(self, mock_dump, fraud_detector, sample_data, tmp_path):
        """Test model saving functionality"""
        X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(sample_data)
        fraud_detector.train_models(X_train, y_train)
        fraud_detector.evaluate_models(X_test, y_test)

        # Test saving
        model_dir = str(tmp_path / "test_models")
        fraud_detector.save_models(model_dir)

        # Check that joblib.dump was called for each model + scaler
        expected_calls = 4  # 3 models + 1 scaler
        assert mock_dump.call_count == expected_calls

    def test_data_validation(self, fraud_detector):
        """Test data validation and error handling"""
        # Test with invalid data
        invalid_data = pd.DataFrame({"invalid_column": [1, 2, 3], "Class": [0, 1, 0]})

        # Should handle missing required columns gracefully
        with pytest.raises((KeyError, ValueError)):
            fraud_detector.preprocess_data(invalid_data)

    def test_reproducibility(self):
        """Test that results are reproducible with same random state"""
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame(
            {
                "Time": np.random.uniform(0, 172800, 100),
                "Amount": np.random.exponential(50, 100),
                "Class": np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            }
        )

        for i in range(1, 29):
            sample_data[f"V{i}"] = np.random.normal(0, 1, 100)

        # Train two systems with same random state
        system1 = FraudDetectionSystem(random_state=42)
        system2 = FraudDetectionSystem(random_state=42)

        # Process data
        X_train1, X_test1, y_train1, y_test1 = system1.preprocess_data(sample_data)
        X_train2, X_test2, y_train2, y_test2 = system2.preprocess_data(sample_data)

        # Results should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)


class TestBusinessLogic:
    """Test business logic and edge cases"""

    def test_edge_case_no_fraud(self):
        """Test handling of dataset with no fraud cases"""
        # Create data with no fraud
        data = pd.DataFrame(
            {
                "Time": [1000, 2000, 3000],
                "Amount": [100, 200, 300],
                "Class": [0, 0, 0],  # No fraud
            }
        )

        for i in range(1, 29):
            data[f"V{i}"] = [0.1, 0.2, 0.3]

        fraud_detector = FraudDetectionSystem(random_state=42)

        # Should handle gracefully (might raise error or return sensible defaults)
        with pytest.raises((ValueError, Exception)):
            fraud_detector.preprocess_data(data)

    def test_edge_case_all_fraud(self):
        """Test handling of dataset with all fraud cases"""
        # Create data with all fraud
        data = pd.DataFrame(
            {
                "Time": [1000, 2000, 3000],
                "Amount": [100, 200, 300],
                "Class": [1, 1, 1],  # All fraud
            }
        )

        for i in range(1, 29):
            data[f"V{i}"] = [0.1, 0.2, 0.3]

        fraud_detector = FraudDetectionSystem(random_state=42)

        # Should handle gracefully
        with pytest.raises((ValueError, Exception)):
            fraud_detector.preprocess_data(data)


if __name__ == "__main__":
    pytest.main([__file__])
