# Main Evaluation Pipeline Runner
import sys
import warnings
warnings.filterwarnings('ignore')

from model_evaluation import FraudDetectionEvaluator

def run_complete_evaluation_pipeline():
    """Execute the complete fraud detection evaluation pipeline"""
    print("STARTING COMPLETE FRAUD DETECTION EVALUATION PIPELINE")
    print("="*80)
    
    try:
        # Initialize evaluator
        evaluator = FraudDetectionEvaluator(
            train_file='../../data/fraudTrain.csv',
            test_file='../../data/fraudTest.csv'
        )
        
        # Step 1: Feature Engineering
        evaluator.run_feature_engineering()
        
        # Step 2: Model Training
        evaluator.train_all_models()
        
        # Step 3: Model Comparison
        comparison_df = evaluator.compare_models()
        
        # Step 4: Visualizations
        evaluator.plot_side_by_side_roc_curves()
        
        # Step 5: Feature Ranking
        evaluator.rank_features_by_best_model()
        
        # Step 6: Save Models and Results
        model_path = evaluator.save_models_and_results()
        
        print("\n" + "="*80)
        print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"✓ Best Model: {evaluator.best_model_name.replace('_', ' ').title()}")
        print(f"✓ Model saved at: {model_path}")
        print("✓ Results saved in: results/model_results_summary.json")
        print("\nREADY FOR DEPLOYMENT!")
        
        return evaluator
        
    except Exception as e:
        print(f"Evaluation pipeline failed with error: {str(e)}")
        raise

def main():
    """Main execution function"""
    evaluator = run_complete_evaluation_pipeline()
    return evaluator

if __name__ == "__main__":
    main()