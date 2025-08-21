# Fraud Detection FastAPI Deployment
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class TransactionRequest(BaseModel):
    """Transaction data model for API requests"""
    amt: float
    trans_hour: int
    trans_month: int
    age: int
    city_pop: int
    is_weekend: int = 0
    is_night: int = 0
    is_business_hours: int = 0
    is_holiday_season: int = 0
    is_tax_season: int = 0
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    month_sin: float = 0.0
    month_cos: float = 0.0
    gender_encoded: int = 0
    is_distant_transaction: int = 0
    city_pop_category: int = 0
    age_group: int = 0
    category: int = 0

class PredictionResponse(BaseModel):
    """Fraud prediction response model"""
    is_fraud: bool
    fraud_probability: float
    risk_level: str

class FraudDetectionAPI:
    """Fraud Detection API class"""
    
    def __init__(self, model_path='../models', features_path='../models', scaler_path='../models'):
        """Initialize the fraud detection API"""
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.scaler = None
        
        # Load model artifacts
        self._load_model_artifacts()
    
    def _load_model_artifacts(self):
        """Load trained model, features, and scaler"""
        try:
            # Find the best model file
            model_files = [f for f in os.listdir(self.model_path) if f.startswith('best_model_')]
            if not model_files:
                raise FileNotFoundError("No trained model found. Please run training first.")
            
            model_file = model_files[0]  # Take the first (should be only one)
            model_full_path = os.path.join(self.model_path, model_file)
            
            # Load the trained model
            with open(model_full_path, 'rb') as f:
                self.model_trainer = pickle.load(f)
                self.model = self.model_trainer.model
            
            # Load feature names
            features_full_path = os.path.join(self.model_path, 'feature_names.pkl')
            with open(features_full_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load scaler if exists
            scaler_full_path = os.path.join(self.model_path, 'scaler.pkl')
            try:
                with open(scaler_full_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                self.scaler = None
                
            print(f"✓ Model loaded successfully from {model_full_path}")
            print(f"✓ Features loaded: {len(self.feature_names)} features")
            print(f"✓ Scaler loaded: {'Yes' if self.scaler else 'No'}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {str(e)}")
    
    def preprocess_transaction(self, transaction_data: Dict) -> np.ndarray:
        """Preprocess a single transaction for prediction"""
        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the features used in training and maintain order
        df = df[self.feature_names]
        
        # Scale if scaler exists (for Logistic Regression)
        if self.scaler is not None:
            df_scaled = self.scaler.transform(df)
            return df_scaled
        
        return df.values
    
    def predict_fraud(self, transaction_data: Dict) -> Dict:
        """Predict if a transaction is fraudulent"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Please check model files.")
        
        try:
            # Preprocess the data
            processed_data = self.preprocess_transaction(transaction_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0, 1]
            
            return {
                'is_fraud': bool(prediction),
                'fraud_probability': float(probability),
                'risk_level': self._get_risk_level(probability)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _get_risk_level(self, probability: float) -> str:
        """Categorize risk level based on probability"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Credit Card Fraud Detection using Machine Learning",
    version="1.0.0"
)

# Initialize API instance
try:
    fraud_api = FraudDetectionAPI()
except Exception as e:
    print(f"⚠️  Failed to initialize API: {str(e)}")
    fraud_api = None

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "Fraud Detection API is running",
        "status": "healthy",
        "model_loaded": fraud_api is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    """Predict fraud for a single transaction"""
    if fraud_api is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure model files are available."
        )
    
    # Convert Pydantic model to dictionary
    transaction_dict = transaction.dict()
    
    # Make prediction
    result = fraud_api.predict_fraud(transaction_dict)
    
    return PredictionResponse(**result)

@app.get("/model/info")
def get_model_info():
    """Get information about the loaded model"""
    if fraud_api is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(fraud_api.model).__name__,
        "num_features": len(fraud_api.feature_names),
        "features": fraud_api.feature_names[:10],  # First 10 features
        "scaler_available": fraud_api.scaler is not None
    }

if __name__ == "__main__":
    print("🚀 Starting Fraud Detection API...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)