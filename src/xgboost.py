# XGBoost Trainer Module
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score)
import xgboost as xg
from xgboost import XGBClassifier

# SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

class XGBoostTrainer:
    """XGBoost trainer with modular steps for hyperparameter tuning, training, evaluation, and visualization"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = []
        self.cv_auc_scores = []

    def tune_hyperparameters(self, X_train, y_train, n_iter=20, cv_folds=3):
        """Step 1: Tune hyperparameters using RandomizedSearchCV"""
        print("Tuning hyperparameters for XGBoost...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 5, 10],
            'scale_pos_weight': [(y_train == 0).sum() / (y_train == 1).sum()]
        }
        model = XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid, n_iter=n_iter, cv=cv_folds,
            scoring='f1', n_jobs=-1, random_state=self.random_state, verbose=1
        )
        random_search.fit(X_train, y_train)
        self.best_params = random_search.best_params_
        print(f"Best Parameters: {self.best_params}")
        return self.best_params

    def train_and_validate(self, X_train, y_train, k_folds=3):
        """Step 2: Train the model using k-fold cross-validation"""
        print(f"Training XGBoost with {k_folds}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        self.cv_scores = []
        self.cv_auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Use best parameters if available
            if self.best_params:
                self.model = XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0, **self.best_params)
            else:
                self.model = XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0)

            self.model.fit(X_fold_train, y_fold_train)

            # Validate on the validation set
            val_pred = self.model.predict(X_fold_val)
            val_prob = self.model.predict_proba(X_fold_val)[:, 1]

            fold_f1 = f1_score(y_fold_val, val_pred)
            fold_auc = roc_auc_score(y_fold_val, val_prob)
            self.cv_scores.append(fold_f1)
            self.cv_auc_scores.append(fold_auc)

            print(f"  Fold {fold + 1}: F1={fold_f1:.3f}, AUC={fold_auc:.3f}")

        print(f"\nAverage F1 Score: {np.mean(self.cv_scores):.3f}")
        print(f"Average AUC Score: {np.mean(self.cv_auc_scores):.3f}")

    def train_final_model(self, X_train, y_train):
        """Train final model on full training data"""
        print("Training final XGBoost model on full training data...")
        if self.best_params:
            self.model = XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0, **self.best_params)
        else:
            self.model = XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0)
        
        self.model.fit(X_train, y_train)
        print("Final model training complete.")

    def predict_on_test(self, X_test):
        """Step 3: Predict on test data"""
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        print("Predicting on test data...")
        test_pred = self.model.predict(X_test)
        test_prob = self.model.predict_proba(X_test)[:, 1]
        return test_pred, test_prob

    def calculate_metrics(self, y_test, test_pred, test_prob):
        """Step 4: Calculate TP, FP, FN, TN and success metrics"""
        print("Calculating metrics...")
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()

        # Core metrics
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_prob)
        test_precision = precision_score(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred)

        print(f"Metrics - Accuracy: {test_accuracy:.3f}, F1: {test_f1:.3f}, AUC: {test_auc:.3f}, "
              f"Precision: {test_precision:.3f}, Recall: {test_recall:.3f}")
        return {
            'accuracy': test_accuracy,
            'f1': test_f1,
            'auc': test_auc,
            'precision': test_precision,
            'recall': test_recall,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }

    def display_classification_report(self, y_test, test_pred):
        """Step 5: Display sklearn's classification report and confusion matrix"""
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, test_pred)
        print(cm)

    def plot_auc_curve(self, y_test, test_prob, title="XGBoost"):
        """Step 6: Plot the AUC curve"""
        print("Plotting AUC curve...")
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='red', lw=2, label=f'{title} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return fpr, tpr, roc_auc

    def plot_shap_feature_importance(self, X_test, feature_names, max_display=20):
        """Step 7: Plot SHAP feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping SHAP analysis.")
            return None

        try:
            print("Computing SHAP values...")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test[:200])

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, max_display=max_display, show=False)
            plt.title('SHAP Feature Importance - XGBoost')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"SHAP analysis failed: {e}")

    def train_full_pipeline(self, X_train_balanced, y_train_balanced, X_test, y_test, feature_names):
        """Complete training pipeline"""
        results = {}
        
        # Tune hyperparameters
        self.tune_hyperparameters(X_train_balanced, y_train_balanced)
        
        # Cross-validation
        self.train_and_validate(X_train_balanced.values, y_train_balanced.values, k_folds=5)
        results['cv_f1_mean'] = np.mean(self.cv_scores)
        results['cv_auc_mean'] = np.mean(self.cv_auc_scores)
        
        # Train final model
        self.train_final_model(X_train_balanced, y_train_balanced)
        
        # Predict on test
        test_pred, test_prob = self.predict_on_test(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, test_pred, test_prob)
        results.update(metrics)
        
        # Display results
        self.display_classification_report(y_test, test_pred)
        fpr, tpr, roc_auc = self.plot_auc_curve(y_test, test_prob)
        
        # Feature importance
        if SHAP_AVAILABLE:
            self.plot_shap_feature_importance(X_test, feature_names, max_display=20)
        
        results['fpr'] = fpr
        results['tpr'] = tpr
        results['roc_auc'] = roc_auc
        results['model_name'] = 'XGBoost'
        
        return results