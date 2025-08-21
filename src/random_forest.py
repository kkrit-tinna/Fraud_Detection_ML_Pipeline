# Random Forest Trainer Module
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score)

class RandomForestTrainer:
    """Random Forest trainer with modular steps for hyperparameter tuning, training, evaluation, and visualization"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def tune_hyperparameters(self, X_train, y_train, n_iter=20, cv_folds=3):
        """Step 1: Tune hyperparameters using RandomizedSearchCV"""
        print("Tuning hyperparameters for Random Forest...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5],
            'class_weight': ['balanced', None]
        }
        model = RandomForestClassifier(random_state=self.random_state)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid, n_iter=n_iter, cv=cv_folds,
            scoring='f1', n_jobs=-1, random_state=self.random_state, verbose=1
        )
        random_search.fit(X_train, y_train)
        self.best_params = random_search.best_params_

        # Apply caps on `n_estimators` and `max_depth`
        self.best_params['n_estimators'] = min(self.best_params.get('n_estimators', 100), 100)
        self.best_params['max_depth'] = min(self.best_params.get('max_depth', 10), 10)

        print(f"Best Parameters: {self.best_params}")
        return self.best_params

    def train(self, X_train, y_train):
        """Step 2: Train the model on the entire training dataset"""
        print("Training Random Forest on the full training dataset...")
        if self.best_params:
            self.model = RandomForestClassifier(random_state=self.random_state, **self.best_params)
        else:
            self.model = RandomForestClassifier(
                random_state=self.random_state, n_estimators=100, max_depth=10, class_weight='balanced'
            )
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def predict_on_test(self, X_test):
        """Step 3: Predict on test data"""
        if self.model is None:
            raise ValueError("Model is not trained yet! Please train the model before predicting.")
        
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

    def plot_auc_curve(self, y_test, test_prob, title="Random Forest"):
        """Step 6: Plot the AUC curve"""
        print("Plotting AUC curve...")
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='green', lw=2, label=f'{title} (AUC = {roc_auc:.3f})')
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

    def plot_feature_importance(self, feature_names, max_display=20):
        """Plot feature importance using Random Forest's built-in feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        try:
            print("Plotting feature importance...")
            # Get feature importances from the model
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:max_display]

            # Plot the feature importances
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_indices)), importances[top_indices], align="center", color="skyblue")
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.gca().invert_yaxis()
            plt.xlabel("Feature Importance")
            plt.title("Top Feature Importances - Random Forest")
            plt.tight_layout()
            plt.show()

            return importances, indices

        except Exception as e:
            print(f"Feature importance plotting failed: {e}")
            return None, None

    def train_full_pipeline(self, X_train_balanced, y_train_balanced, X_test, y_test, feature_names):
        """Complete training pipeline"""
        results = {}
        
        # Tune hyperparameters
        self.tune_hyperparameters(X_train_balanced, y_train_balanced)
        
        # Train model
        self.train(X_train_balanced, y_train_balanced)
        
        # Predict on test
        test_pred, test_prob = self.predict_on_test(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, test_pred, test_prob)
        results.update(metrics)
        
        # Display results
        self.display_classification_report(y_test, test_pred)
        fpr, tpr, roc_auc = self.plot_auc_curve(y_test, test_prob)
        
        # Feature importance
        importances, indices = self.plot_feature_importance(feature_names, max_display=20)
        
        results['fpr'] = fpr
        results['tpr'] = tpr
        results['roc_auc'] = roc_auc
        results['feature_importances'] = importances
        results['feature_indices'] = indices
        results['model_name'] = 'Random Forest'
        
        return results
