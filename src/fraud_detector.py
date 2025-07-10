"""
Credit Card Fraud Detection - Main Training Pipeline
"""

# importing all the necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import warnings
import time
from logging_config import setup_logging

warnings.filterwarnings("ignore")


class FraudDetectionSystem:
    """Complete fraud detection pipeline for credit card transactions"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.performance_metrics = {}
        self.logger = setup_logging("INFO")

    def load_data(self, file_path, sample_size=None):
        """Load and optionally sample the dataset"""
        print(f"Loading fraud detection dataset from: {file_path}")

        if sample_size:
            # For large datasets, sample for faster processing
            df = pd.read_csv(file_path, nrows=sample_size)
            print(f"Loaded sample of {sample_size} transactions")
        else:
            df = pd.read_csv(file_path)
            print(f"Loaded full dataset: {df.shape[0]} transactions")

        print(f"Fraud cases: {df['Class'].sum()}")
        print(f"Fraud percentage: {df['Class'].mean() * 100:.3f}%")

        return df

    def preprocess_data(self, df):
        """Preprocess and engineer features"""
        print("Preprocessing data...")

        # Feature engineering
        df = df.copy()
        df["Amount_log"] = np.log1p(df["Amount"])
        df["Time_hour"] = (df["Time"] / 3600) % 24

        # Select features (use PCA features + engineered features)
        feature_cols = [col for col in df.columns if col.startswith("V")] + [
            "Amount_log",
            "Time_hour",
        ]
        X = df[feature_cols]
        y = df["Class"]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training fraud rate: {y_train.mean() * 100:.3f}%")

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        """Train multiple fraud detection models"""
        print("Training fraud detection models...")
        start_time = time.time()

        # Calculate class weight for imbalanced data
        fraud_count = len(y_train[y_train == 1])
        normal_count = len(y_train[y_train == 0])

        if fraud_count == 0:
            print("Warning: No fraud cases in training data")
            class_weight = 1.0
        else:
            class_weight = normal_count / fraud_count

        # 1. Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(
            class_weight="balanced", random_state=self.random_state, max_iter=1000
        )
        lr.fit(X_train, y_train)
        self.models["logistic_regression"] = lr

        # 2. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.models["random_forest"] = rf

        # 3. XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            scale_pos_weight=class_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        self.models["xgboost"] = xgb_model

        total_time = time.time() - start_time
        print(" Model training completed!")

        # Log training completion
        self.logger.log_data_processing(
            operation="model_training",
            record_count=len(X_train),
            processing_time_seconds=total_time,
        )

        return self.models

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("Evaluating model performance...")

        results = {}

        for name, model in self.models.items():
            print(f"Evaluating {name}...")

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Business metrics
            avg_fraud_amount = 100
            investigation_cost = 50
            money_saved = tp * avg_fraud_amount
            total_cost = (tp + fp) * investigation_cost + fn * avg_fraud_amount
            net_benefit = money_saved - total_cost

            results[name] = {
                "AUC": auc_score,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Net_Benefit": net_benefit,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            }

            print(f"  AUC: {auc_score:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  Net Benefit: ${net_benefit:,.2f}")

        self.performance_metrics = results
        return results

    def save_models(self, model_dir="models"):
        """Save trained models and scaler"""
        os.makedirs(model_dir, exist_ok=True)

        print("Saving models...")
        for name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{name}.pkl")

        # Save scaler
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")

        # Save performance metrics
        if self.performance_metrics:
            results_df = pd.DataFrame(self.performance_metrics).T
            results_df.to_csv(f"{model_dir}/model_performance.csv")

        print(" Models saved successfully!")

    def create_visualizations(self, X_test, y_test, save_dir="results"):
        """Create performance visualizations"""
        os.makedirs(save_dir, exist_ok=True)

        print("Creating visualizations...")

        # ROC Curves
        plt.figure(figsize=(10, 6))

        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = self.performance_metrics[name]["AUC"]
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Fraud Detection Models")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/roc_curves.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Performance Comparison
        metrics = ["AUC", "Precision", "Recall", "F1"]
        model_names = list(self.performance_metrics.keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [self.performance_metrics[name][metric] for name in model_names]
            bars = axes[i].bar(
                model_names, values, color=["skyblue", "lightgreen", "salmon"]
            )
            axes[i].set_title(f"{metric} Comparison")
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Generate predictions for additional visualizations
        predictions = {}
        performance_data = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            predictions[name] = y_pred

            # Use EXACT same net benefit values from performance metrics (already calculated)
            net_benefit = self.performance_metrics[name]["Net_Benefit"]
            performance_data[name] = {"Net Benefit ($)": net_benefit}

        # Generate additional visualizations for portfolio
        self._generate_confusion_matrices(y_test, predictions, save_dir)
        self._generate_feature_importance_plot(save_dir)
        self._generate_business_impact_plot(performance_data, save_dir)

        print(" Visualizations saved!")

    def _generate_confusion_matrices(self, y_test, predictions, save_dir):
        """Generate confusion matrices for all models"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, (name, y_pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
            axes[idx].set_title(f'{name.replace("_", " ").title()} Confusion Matrix')
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def _generate_feature_importance_plot(self, save_dir):
        """Generate feature importance plot for XGBoost"""
        if "xgboost" not in self.models:
            return

        # Get feature importance from XGBoost
        feature_importance = self.models["xgboost"].feature_importances_
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Time_hour"]

        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        ).sort_values("importance", ascending=True)

        # Plot top 15 features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.tail(15)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Feature Importance")
        plt.title("Top 15 Feature Importance (XGBoost)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def _generate_business_impact_plot(self, performance_data, save_dir):
        """Generate business impact visualization"""
        models = list(performance_data.keys())
        net_benefits = [performance_data[model]["Net Benefit ($)"] for model in models]
        
        # Debug output to verify values
        print("Debug: Business Impact Values for Visualization:")
        for model, benefit in zip(models, net_benefits):
            print(f"  {model}: ${benefit:,.0f}")

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            models,
            net_benefits,
            color=["red" if x < 0 else "green" for x in net_benefits],
        )
        plt.title("Business Impact: Net Benefit by Model")
        plt.xlabel("Models")
        plt.ylabel("Net Benefit ($)")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, net_benefits):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (50 if value > 0 else -100),
                f"${value:,.0f}",
                ha="center",
                va="bottom" if value > 0 else "top",
            )

        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/business_impact.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


def main():
    """Main training pipeline"""
    print(" Credit Card Fraud Detection System")
    print("=" * 50)

    # Setup
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)

    # Initialize system
    fraud_detector = FraudDetectionSystem(random_state=42)

    # Load data (sample for faster processing)
    df = fraud_detector.load_data("data/creditcard 2.csv", sample_size=50000)

    # Preprocess
    X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(df)

    # Train models
    fraud_detector.train_models(X_train, y_train)

    # Evaluate
    fraud_detector.evaluate_models(X_test, y_test)

    # Save everything
    fraud_detector.save_models()
    fraud_detector.create_visualizations(X_test, y_test)

    # Show best model
    best_model = max(
        fraud_detector.performance_metrics.keys(),
        key=lambda x: fraud_detector.performance_metrics[x]["AUC"],
    )
    best_auc = fraud_detector.performance_metrics[best_model]["AUC"]

    print(f"\n Best Model: {best_model}")
    print(f" Best AUC: {best_auc:.4f}")
    print("\n Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
