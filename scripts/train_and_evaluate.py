# scripts/train_and_evaluate.py
"""
Entry point for training and evaluating fraud detection models
"""
import sys
import os

# Add evaluation module to path
sys.path.append('../evaluation')
from run_evaluation import main

if __name__ == "__main__":
    print("Starting Model Training and Evaluation...")
    main()

# ===== SEPARATOR =====

# scripts/deploy_api.py
"""
Entry point for deploying fraud detection API
"""
import sys
import os

# Add deployment module to path
sys.path.append('../deployment')

if __name__ == "__main__":
    print("Starting Fraud Detection API Deployment...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    
    # Import and run the FastAPI app
    from fraud_detection_api import app
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)