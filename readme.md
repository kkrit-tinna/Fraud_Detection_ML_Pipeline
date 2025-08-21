# Fraud Detection ML Pipeline

A complete, production-ready machine learning pipeline for credit card fraud detection using scikit-learn, XGBoost, and FastAPI.

## Overview

This package provides an end-to-end solution for fraud detection, from feature engineering to model deployment. It trains and compares three different models (Logistic Regression, Random Forest, XGBoost) and automatically selects the best performer based on fraud detection effectiveness.

## Project Structure

```
project_root/
├── data/                           # Shared datasets
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── fraud_detection_pipeline/       # Main ML Pipeline Package
│   ├── src/                        # Core ML modules
│   │   ├── feature_engineering.py # Feature engineering pipeline
│   │   ├── logistic_regression.py # Logistic regression trainer
│   │   ├── random_forest.py      # Random forest trainer
│   │   └── xgboost_trainer.py     # XGBoost trainer
│   ├── evaluation/                 # Model evaluation and comparison
│   │   ├── model_evaluator.py     # Model comparison and analysis
│   │   └── run_evaluation.py      # Evaluation pipeline runner
│   ├── deployment/                 # Production API
│   │   ├── fraud_detection_api.py # FastAPI deployment
│   │   └── requirements_prod.txt  # Production dependencies
│   ├── scripts/                    # Entry point scripts
│   │   ├── train_and_evaluate.py # Run training pipeline
│   │   ├── deploy_api.py          # Deploy API
│   │   ├── main.py                # Original main script
│   │   └── fraud_detection_ml_testing.py  # Original testing script
│   ├── models/                     # Generated model artifacts (after training)
│   ├── results/                    # Training results and reports (after training)
│   ├── tests/                      # Unit tests
│   ├── requirements.txt            # Development dependencies
│   ├── summary.txt                 # Package documentation
│   └── README.md                   # This file
└── Torch_training/                 # PyTorch implementation (separate project)
    └── [torch training files]
```

## Quick Start

### 1. Install Dependencies

```bash
# For training and evaluation
pip install -r requirements.txt

# For production API only
pip install -r deployment/requirements_prod.txt
```

### Prepare Data

Ensure your datasets are in the data directory at the project root:
```bash
# Your project structure should look like:
project_root/
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
└── fraud_detection_pipeline/
    └── [package contents]
```

### 3. Train Models

```bash
cd fraud_detection_pipeline
python scripts/train_and_evaluate.py
```

This will:
- Extract advanced features (temporal, demographic, geographic)
- Train three models with hyperparameter tuning
- Compare performance and select best model
- Generate visualizations and reports
- Save model artifacts for deployment

### 4. Deploy API

```bash
# Make sure you're in the fraud_detection_pipeline directory
cd fraud_detection_pipeline
python scripts/deploy_api.py
```

The API will be available at:
- **Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **API Spec**: `http://localhost:8000/redoc`

## Model Selection

The pipeline automatically selects the best model based on:
- **40% Recall** - Minimize missed fraud (false negatives)
- **40% F1-Score** - Balance precision and recall
- **20% AUC** - Overall discriminative ability

This prioritizes fraud detection effectiveness over general accuracy.

## API Usage

### Health Check
```bash
curl http://localhost:8000/
```

### Predict Fraud
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amt": 150.00,
       "trans_hour": 14,
       "trans_month": 6,
       "age": 35,
       "city_pop": 50000,
       "is_weekend": 0,
       "is_night": 0
     }'
```

### Response Format
```json
{
  "is_fraud": false,
  "fraud_probability": 0.23,
  "risk_level": "LOW"
}
```

### Python Client Example
```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'amt': 500.00,
    'trans_hour': 2,
    'trans_month': 12,
    'age': 25,
    'city_pop': 100000,
    'is_weekend': 1,
    'is_night': 1
})

result = response.json()
print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.2%}")
print(f"Risk: {result['risk_level']}")
```

## Key Features

### Advanced Feature Engineering
- **Temporal Features**: Hour/month cyclical encoding, business hours detection
- **Geographic Features**: Distance calculations, population categorization  
- **Demographic Features**: Age groups, gender encoding
- **Risk Indicators**: Weekend/night flags, distant transactions

### Model Training
- **Hyperparameter Tuning**: Automated parameter optimization
- **Cross-Validation**: Stratified k-fold validation
- **Class Balancing**: SMOTE oversampling for imbalanced data
- **Feature Importance**: SHAP analysis for interpretability

### Production Deployment
- **FastAPI**: Modern, fast web framework
- **Automatic Docs**: Interactive API documentation
- **Model Versioning**: Persistent model artifacts
- **Error Handling**: Comprehensive error responses
- **Risk Categorization**: LOW/MEDIUM/HIGH risk levels

## Performance Metrics

After training, you'll get:
- **Model Comparison Table**: Accuracy, F1, AUC, Precision, Recall
- **ROC Curves**: Side-by-side performance visualization
- **Feature Rankings**: Top predictive features
- **Confusion Matrices**: Detailed classification breakdown

## Development

### Run Original Scripts
```bash
# Original monolithic script
python scripts/fraud_detection_ml_testing.py

# Current main script
python scripts/main.py
```

### Run Tests
```bash
python -m pytest tests/
```

### Custom Configuration
Modify training parameters in the respective trainer classes or create configuration files as needed.

## Requirements

### Training Environment
- Python 3.7+
- 8GB+ RAM recommended
- scikit-learn, XGBoost, pandas, matplotlib

### Production Environment  
- Python 3.7+
- 2GB+ RAM
- FastAPI, uvicorn, scikit-learn

## Model Performance Focus

This pipeline optimizes for **real-world fraud detection**:
- **Minimize False Negatives**: Reduce missed fraud cases
- **Balance Precision-Recall**: Maintain operational efficiency  
- **Financial Impact**: Focus on business value over accuracy
- **Reproducible Results**: Consistent performance across runs

## Production Deployment

The trained model is ready for:
- **REST API**: FastAPI server (included)
- **Batch Processing**: Load model and process CSV files
- **Real-time Inference**: Sub-100ms prediction latency
- **Containerization**: Docker-ready structure

## Support

For issues or questions:
1. Check the interactive API docs at `/docs`
2. Review the generated results in `results/`
3. Examine model artifacts in `models/`
4. Refer to `summary.txt` for detailed documentation

---

**Ready to detect fraud? Start with `python scripts/train_and_evaluate.py`** 