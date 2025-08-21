# Logistic Regression Trainer Module
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score)
from joblib import Parallel, delayed

# SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

class LogisticRegressionTrainer:
    """Logistic Regression trainer with modular steps for hyperparameter tuning, k-fold validation, training, evaluation, and visualization"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = []
        self.cv_auc_scores = []
        self.results = None
        self.scaler = None

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def tune_hyperparameters(self, X_train, y_train, n_iter=10, cv_folds=3):
        """Step 1: Tune hyperparameters using RandomizedSearchCV with optimized settings"""
        print("Tuning hyperparameters for Logistic Regression...")
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced']
        }
        model = LogisticRegression(random_state=self.random_state, max_iter=100)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid, n_iter=n_iter, cv=cv_folds,
            scoring='f1', n_jobs=-1, random_state=self.random_state, verbose=1
        )
        random_search.fit(X_train, y_train)
        self.best_params = random_search.best_params_
        print(f"Best Parameters: {self.best_params}")
        return self.best_params

    def split_kfold(self, X_train, y_train, k_folds=3):
        """Step 2: Split training and evaluation set using k-fold"""
        print(f"Splitting data using {k_folds}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        return skf.split(X_train, y_train)

    def train_and_validate(self, X_train, y_train, train_idx, val_idx):
        """Step 3: Train the model on training data and validate on one remaining set"""
        print("Training Logistic Regression on training set and validating on validation set...")
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Use best parameters if available
        if self.best_params:
            self.model = LogisticRegression(random_state=self.random_state, max_iter=2000, **self.best_params)
        else:
            self.model = LogisticRegression(
                random_state=self.random_state, C=1.0, penalty='l2', solver='lbfgs',
                max_iter=2000, class_weight='balanced'
            )

        self.model.fit(X_fold_train, y_fold_train)

        # Validate on the validation set
        val_pred = self.model.predict(X_fold_val)
        val_prob = self.model.predict_proba(X_fold_val)[:, 1]

        # Calculate metrics
        fold_f1 = f1_score(y_fold_val, val_pred)
        fold_auc = roc_auc_score(y_fold_val, val_prob)
        self.cv_scores.append(fold_f1)
        self.cv_auc_scores.append(fold_auc)

        print(f"Validation Results - F1: {fold_f1:.3f}, AUC: {fold_auc:.3f}")
        return fold_f1, fold_auc

    def predict_on_test(self, X_test, X_train=None, y_train=None):
        """Step 4: Predict on test data"""
        if self.model is None:
            if self.best_params is None:
                raise ValueError("Model is not trained yet! Please train the model or provide training data.")
            if X_train is None or y_train is None:
                raise ValueError("Training data must be provided to train the model before predicting.")
            
            print("Training the model on the full training dataset...")
            self.model = LogisticRegression(random_state=self.random_state, max_iter=2000, **self.best_params)
            self.model.fit(X_train, y_train)
        
        print("Predicting on test data...")
        test_pred = self.model.predict(X_test)
        test_prob = self.model.predict_proba(X_test)[:, 1]
        return test_pred, test_prob

    def calculate_metrics(self, y_test, test_pred, test_prob):
        """Step 5: Calculate TP, FP, FN, TN and success metrics"""
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
        """Step 6: Display sklearn's classification report and confusion matrix"""
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, test_pred)
        print(cm)

    def plot_auc_curve(self, y_test, test_prob, title="Logistic Regression"):
        """Step 7: Plot the AUC curve"""
        print("Plotting AUC curve...")
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{title} (AUC = {roc_auc:.3f})')
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
        """Step 8: Plot SHAP feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping SHAP analysis.")
            return None

        try:
            print("Computing SHAP values...")
            explainer = shap.LinearExplainer(self.model, X_test[:100])
            shap_values = explainer.shap_values(X_test[:200])

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, max_display=max_display, show=False)
            plt.title('SHAP Feature Importance - Logistic Regression')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"SHAP analysis failed: {e}")

    def train_full_pipeline(self, X_train_balanced, y_train_balanced, X_test, y_test, feature_names):
        """Complete training pipeline"""
        results = {}
        
        # Scale features
        X_train_balanced_scaled, X_test_scaled = self.scale_features(X_train_balanced, X_test)
        
        # Tune hyperparameters
        self.tune_hyperparameters(X_train_balanced_scaled, y_train_balanced)
        
        # K-fold cross validation
        kfold_splits = list(self.split_kfold(X_train_balanced_scaled, y_train_balanced, k_folds=5))
        
        # Parallel training and validation
        cv_results = Parallel(n_jobs=-1)(
            delayed(self.train_and_validate)(
                X_train_balanced_scaled, y_train_balanced.values, train_idx, val_idx
            )
            for train_idx, val_idx in kfold_splits
        )
        
        f1_scores, auc_scores = zip(*cv_results)
        results['cv_f1_mean'] = np.mean(f1_scores)
        results['cv_auc_mean'] = np.mean(auc_scores)
        
        # Train final model
        self.model = LogisticRegression(random_state=self.random_state, max_iter=2000, **self.best_params)
        self.model.fit(X_train_balanced_scaled, y_train_balanced)
        
        # Predict on test
        test_pred, test_prob = self.predict_on_test(X_test_scaled, X_train_balanced_scaled, y_train_balanced)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, test_pred, test_prob)
        results.update(metrics)
        
        # Display results
        self.display_classification_report(y_test, test_pred)
        fpr, tpr, roc_auc = self.plot_auc_curve(y_test, test_prob)
        
        # Feature importance
        if SHAP_AVAILABLE:
            self.plot_shap_feature_importance(X_test_scaled, feature_names, max_display=20)
        
        results['fpr'] = fpr
        results['tpr'] = tpr
        results['roc_auc'] = roc_auc
        results['model_name'] = 'Logistic Regression'
        
        return results