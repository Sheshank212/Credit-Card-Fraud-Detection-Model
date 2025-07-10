# Trained Models

## Model Information

This directory contains trained fraud detection models and preprocessing components.

### Generated Models
After running the training pipeline, you'll find:

```
models/
├── README.md                    # This file
├── logistic_regression.pkl      # Logistic Regression model
├── random_forest.pkl           # Random Forest model  
├── xgboost.pkl                 # XGBoost model (best performer)
├── scaler.pkl                  # Feature scaler
└── model_performance.csv       # Performance metrics
```

### Model Performance
| Model | AUC Score | Precision | Recall | Use Case |
|-------|-----------|-----------|---------|----------|
| **XGBoost** | **97.62%** | **87.37%** | **84.69%** | **Production (Best Overall)** |
| Logistic Regression | 97.16% | 5.61% | 90.82% | High recall screening |
| Random Forest | 95.34% | 96.15% | 76.53% | High precision when needed |

### Model Features
- **Input**: 30 features (28 PCA + Amount_log + Time_hour)
- **Output**: Fraud probability (0-1) + binary classification
- **Optimization**: Balanced for class imbalance (0.173% fraud rate)
- **Inference Time**: <5ms per prediction

### Usage
```python
import joblib

# Load model and scaler
model = joblib.load('models/xgboost.pkl')
scaler = joblib.load('models/scaler.pkl')

# Make prediction
features_scaled = scaler.transform(transaction_features)
fraud_probability = model.predict_proba(features_scaled)[0, 1]
```

### Training
To generate these models:
```bash
python src/fraud_detector.py
# or
python scripts/demo.py
```

**Note**: Model files are not committed to Git due to size (>100MB). Run training to generate them locally.