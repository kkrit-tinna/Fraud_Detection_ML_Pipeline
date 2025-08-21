# Main Fraud Detection Pipeline
# Model Evaluation and Deployment Script

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
sys.path.append('../src')
from feature_engineering import full_feature_engineering_pipeline, create_fraud_analysis_plots
from logistic_regression import LogisticRegressionTrainer
from random_forest import RandomForestTrainer
from xgboost_trainer import XGBoostTrainer

class FraudDetectionPipeline:
    """Complete fraud detection pipeline with model evaluation and deployment capabilities"""

    def __init__(self, train_file='../../data/fraudTrain.csv', test_file='../../data/fraudTest.csv'):
        self.train_file = train_file
        self.test_file = test_file
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def run_feature_engineering(self):
        """Execute feature engineering pipeline"""
        print("="*60)
        print("STEP 1: FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        self.data = full_feature_engineering_pipeline(self.train_file, self.test_file)
        self.feature_names = self.data['feature_names']
        
        print(f"Feature engineering completed successfully!")
        print(f"Total features: {len(self.feature_names)}")
        
        # Create visualization
        print("\nGenerating fraud analysis plots...")
        create_fraud_analysis_plots(self.data['train_enhanced'])
        
        return self.data
    
    def train_all_models(self):
        """Train all three models and collect results"""
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING AND EVALUATION")
        print("="*60)
        
        # Initialize trainers
        self.models = {
            'logistic_regression': LogisticRegressionTrainer(random_state=42),
            'random_forest': RandomForestTrainer(random_state=42),
            'xgboost': XGBoostTrainer(random_state=42)
        }
        
        # Train each model
        for model_name, trainer in self.models.items():
            print(f"\n{'-'*50}")
            print(f"TRAINING {model_name.upper().replace('_', ' ')}")
            print(f"{'-'*50}")
            
            try:
                results = trainer.train_full_pipeline(
                    self.data['X_train_balanced'],
                    self.data['y_train_balanced'],
                    self.data['X_test'],
                    self.data['y_test'],
                    self.feature_names
                )
                self.results[model_name] = results
                print(f"✓ {model_name.replace('_', ' ').title()} training completed successfully")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                continue
    
    def compare_models(self):
        """Compare all models and determine the best one"""
        print("\n" + "="*60)
        print("STEP 3: MODEL COMPARISON AND EVALUATION")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'F1 Score': results['f1'],
                'AUC': results['auc'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'False Negatives': results['false_negatives'],
                'False Positives': results['false_positives'],
                'True Positives': results['true_positives'],
                'True Negatives': results['true_negatives']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("="*80)
        print(comparison_df.round(4).to_string(index=False))
        
        # Determine best model based on recall rate and F1-score
        # Prioritize recall (fraud detection) and F1-score
        comparison_df['score'] = (comparison_df['Recall'] * 0.4 + 
                                comparison_df['F1 Score'] * 0.4 + 
                                comparison_df['AUC'] * 0.2)
        
        best_idx = comparison_df['score'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model'].lower().replace(' ', '_')
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 BEST MODEL: {comparison_df.loc[best_idx, 'Model']}")
        print(f"   - Composite Score: {comparison_df.loc[best_idx, 'score']:.4f}")
        print(f"   - Recall: {comparison_df.loc[best_idx, 'Recall']:.4f}")
        print(f"   - F1 Score: {comparison_df.loc[best_idx, 'F1 Score']:.4f}")
        print(f"   - False Negatives: {comparison_df.loc[best_idx, 'False Negatives']}")
        
        return comparison_df
    
    def plot_side_by_side_roc_curves(self):
        """Plot ROC curves for all models side by side"""
        print("\nGenerating side-by-side ROC curves...")
        
        plt.figure(figsize=(12, 8))
        colors = ['darkorange', 'green', 'red']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'fpr' in results and 'tpr' in results:
                plt.plot(results['fpr'], results['tpr'], 
                        color=colors[i], lw=2, 
                        label=f'{results["model_name"]} (AUC = {results["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison - All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_comparison_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def rank_features_by_best_model(self):
        """Rank features based on the best model"""
        print(f"\nRANKING FEATURES BASED ON BEST MODEL ({self.best_model_name.replace('_', ' ').title()}):")
        print("="*80)
        
        if self.best_model_name == 'random_forest':
            # Use feature importance for Random Forest
            results = self.results[self.best_model_name]
            if 'feature_importances' in results:
                importances = results['feature_importances']
                indices = results['feature_indices']
                
                feature_ranking = pd.DataFrame({
                    'Feature': [self.feature_names[i] for i in indices],
                    'Importance': importances[indices]
                }).head(20)
                
                print("TOP 20 FEATURES (Random Forest Importance):")
                print(feature_ranking.to_string(index=False))
                
                return feature_ranking
        
        elif self.best_model_name in ['logistic_regression', 'xgboost']:
            print("Feature importance available through SHAP analysis (already displayed above)")
            return None
        
        return None
    
    def save_models_and_results(self):
        """Save trained models and results for deployment"""
        print("\nSAVING MODELS AND RESULTS FOR DEPLOYMENT...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Save best model
        best_model_path = f'models/best_model_{self.best_model_name}.pkl'
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Best model saved: {best_model_path}")
        
        # Save feature names
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(list(self.feature_names), f)
        print("✓ Feature names saved")
        
        # Save scaler if exists (for Logistic Regression)
        if hasattr(self.best_model, 'scaler') and self.best_model.scaler is not None:
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.best_model.scaler, f)
            print("✓ Scaler saved")
        
        # Save results summary
        results_summary = {
            'best_model': self.best_model_name,
            'model_comparison': {name: {k: v for k, v in results.items() 
                               if k not in ['fpr', 'tpr', 'confusion_matrix', 'feature_importances', 'feature_indices']} 
                               for name, results in self.results.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open('results/model_results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print("✓ Results summary saved")
        
        return best_model_path
    
    def create_deployment_script(self):
        """Create a simple deployment script"""
        deployment_code = '''
# Fraud Detection Model Deployment Script
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class FraudDetectionAPI:
    def __init__(self, model_path='models/best_model_{}.pkl', 
                 features_path='models/feature_names.pkl',
                 scaler_path='models/scaler.pkl'):
        """Initialize the fraud detection API"""
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            self.model_trainer = pickle.load(f)
            self.model = self.model_trainer.model
        
        # Load feature names
        with open(features_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Load scaler if exists
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            self.scaler = None
    
    def preprocess_transaction(self, transaction_data):
        """Preprocess a single transaction for prediction"""
        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Apply the same feature engineering steps as training
        # (This would need to be customized based on your specific features)
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the features used in training
        df = df[self.feature_names]
        
        # Scale if scaler exists
        if self.scaler is not None:
            df_scaled = self.scaler.transform(df)
            return df_scaled
        
        return df.values
    
    def predict_fraud(self, transaction_data, return_probability=True):
        """Predict if a transaction is fraudulent"""
        # Preprocess the data
        processed_data = self.preprocess_transaction(transaction_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        
        if return_probability:
            probability = self.model.predict_proba(processed_data)[0, 1]
            return {{
                'is_fraud': bool(prediction),
                'fraud_probability': float(probability),
                'risk_level': self._get_risk_level(probability)
            }}
        
        return {{'is_fraud': bool(prediction)}}
    
    def _get_risk_level(self, probability):
        """Categorize risk level based on probability"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'

# Example usage:
if __name__ == "__main__":
    # Initialize the API
    api = FraudDetectionAPI()
    
    # Example transaction
    sample_transaction = {{
        'amt': 150.00,
        'trans_hour': 14,
        'trans_month': 6,
        'age': 35,
        'city_pop': 50000,
        'is_weekend': 0,
        'is_night': 0,
        # ... include all required features
    }}
    
    # Make prediction
    result = api.predict_fraud(sample_transaction)
    print(f"Prediction: {{result}}")
'''.format(self.best_model_name)
        
        with open('deploy_model.py', 'w') as f:
            f.write(deployment_code)
        
        print("✓ Deployment script created: deploy_model.py")
    
    def run_complete_pipeline(self):
        """Execute the complete fraud detection pipeline"""
        print("STARTING COMPLETE FRAUD DETECTION PIPELINE")
        print("="*80)
        
        try:
            # Step 1: Feature Engineering
            self.run_feature_engineering()
            
            # Step 2: Model Training
            self.train_all_models()
            
            # Step 3: Model Comparison
            comparison_df = self.compare_models()
            
            # Step 4: Visualizations
            self.plot_side_by_side_roc_curves()
            
            # Step 5: Feature Ranking
            self.rank_features_by_best_model()
            
            # Step 6: Save Models and Results
            model_path = self.save_models_and_results()
            
            # Step 7: Create Deployment Script
            self.create_deployment_script()
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"✓ Best Model: {self.best_model_name.replace('_', ' ').title()}")
            print(f"✓ Model saved at: {model_path}")
            print("✓ Deployment script created: deploy_model.py")
            print("✓ Results saved in: results/model_results_summary.json")
            print("\nREADY FOR PRODUCTION DEPLOYMENT! 🚢")
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(
        train_file='../../data/fraudTrain.csv',
        test_file='../../data/fraudTest.csv'
    )
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()