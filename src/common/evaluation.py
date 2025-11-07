"""
STEP 6: Model Evaluation
Evaluate model performance on validation and test sets
Include confusion matrix, metrics, feature importance, and cross-validation
"""

import numpy as np
import pandas as pd
import logging
import json
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Tuple

from .config import (
    CV_CONFIG, METRICS, DATA_OUTPUT_DIR,
    OUTPUT_FILES, LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to evaluate Random Forest model performance"""

    def __init__(self, model):
        """
        Initialize ModelEvaluator

        Args:
            model: Trained model
        """
        self.model = model
        self.val_metrics = {}
        self.test_metrics = {}
        self.cv_scores = {}
        self.feature_importance = None

    def evaluate_set(self, X: np.ndarray, y: np.ndarray,
                     set_name: str = "Test") -> Dict:
        """
        Evaluate model on a dataset

        Args:
            X: Feature matrix
            y: True labels
            set_name: Name of the dataset (for logging)

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating on {set_name} set...")

        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_pred_proba)

        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        metrics = {
            'confusion_matrix': cm.tolist(),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'specificity': float(specificity),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'n_samples': len(y),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }

        # Log results
        logger.info(f"\n{set_name} Set Results:")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN: {tn:4d}  FP: {fp:4d}")
        logger.info(f"    FN: {fn:4d}  TP: {tp:4d}")
        logger.info(f"\n  Metrics:")
        logger.info(f"    Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"    Precision:   {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"    Recall:      {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"    F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
        logger.info(f"    ROC-AUC:     {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        logger.info(f"    Specificity: {specificity:.4f} ({specificity*100:.2f}%)")

        return metrics

    def evaluate_validation(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Evaluate model on validation set

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary with validation metrics
        """
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SET EVALUATION")
        logger.info("="*70)

        self.val_metrics = self.evaluate_set(X_val, y_val, "Validation")

        return self.val_metrics

    def evaluate_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with test metrics
        """
        logger.info("\n" + "="*70)
        logger.info("TEST SET EVALUATION")
        logger.info("="*70)

        self.test_metrics = self.evaluate_set(X_test, y_test, "Test")

        # Classification report
        logger.info("\nClassification Report:")
        report = classification_report(
            y_test,
            self.test_metrics['y_pred'],
            target_names=['No Deforestation', 'Deforestation'],
            digits=4
        )
        logger.info("\n" + report)

        return self.test_metrics

    def calculate_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Calculate and rank feature importance

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        logger.info("\n" + "="*70)
        logger.info("FEATURE IMPORTANCE")
        logger.info("="*70)

        importances = self.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_normalized': importances / importances.sum() * 100
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)

        # Reorder columns
        importance_df = importance_df[['rank', 'feature', 'importance', 'importance_normalized']]

        self.feature_importance = importance_df

        # Log top features
        logger.info("\nTop 15 Most Important Features:")
        logger.info(f"{'Rank':<5} {'Feature':<30} {'Importance':<12} {'Normalized (%)':<15}")
        logger.info("-" * 70)
        for idx, row in importance_df.head(15).iterrows():
            logger.info(f"{row['rank']:<5} {row['feature']:<30} {row['importance']:<12.6f} {row['importance_normalized']:<15.2f}")

        return importance_df

    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform K-Fold Cross Validation

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Dictionary with cross-validation scores
        """
        logger.info("\n" + "="*70)
        logger.info("CROSS VALIDATION")
        logger.info("="*70)

        n_splits = CV_CONFIG['n_splits']
        logger.info(f"\nPerforming {n_splits}-Fold Cross Validation...")

        # Create StratifiedKFold
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=CV_CONFIG['shuffle'],
            random_state=CV_CONFIG['random_state']
        )

        # Calculate scores for each metric
        cv_results = {}

        for metric in METRICS:
            logger.info(f"\n  Calculating {metric}...")
            scores = cross_val_score(
                self.model, X, y,
                cv=skf,
                scoring=metric,
                n_jobs=-1
            )

            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max())
            }

            logger.info(f"    {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

        self.cv_scores = cv_results

        # Summary
        logger.info("\n" + "="*70)
        logger.info("Cross Validation Summary:")
        logger.info("="*70)
        for metric, result in cv_results.items():
            logger.info(f"  {metric:12s}: {result['mean']:.4f} ± {result['std']:.4f} "
                       f"(min: {result['min']:.4f}, max: {result['max']:.4f})")

        return cv_results

    def evaluate_all(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_names: list) -> Dict:
        """
        Perform all evaluations

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("="*70)

        # Evaluate validation set
        val_metrics = self.evaluate_validation(X_val, y_val)

        # Evaluate test set
        test_metrics = self.evaluate_test(X_test, y_test)

        # Calculate feature importance
        feature_importance = self.calculate_feature_importance(feature_names)

        # Cross validation
        cv_scores = self.cross_validate(X_train, y_train)

        logger.info("\n" + "="*70)
        logger.info("✓ MODEL EVALUATION COMPLETED")
        logger.info("="*70)

        return {
            'validation': val_metrics,
            'test': test_metrics,
            'feature_importance': feature_importance,
            'cross_validation': cv_scores
        }

    def save_results(self):
        """Save evaluation results to files"""
        logger.info("\n" + "="*70)
        logger.info("SAVING EVALUATION RESULTS")
        logger.info("="*70)

        # Save feature importance
        if self.feature_importance is not None:
            importance_path = OUTPUT_FILES['feature_importance']
            logger.info(f"\nSaving feature importance to: {importance_path}")
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"  ✓ Saved {len(self.feature_importance)} features")

        # Save evaluation metrics
        metrics_path = OUTPUT_FILES['evaluation_metrics']
        logger.info(f"\nSaving evaluation metrics to: {metrics_path}")

        metrics_data = {
            'validation': self.val_metrics,
            'test': self.test_metrics,
            'cross_validation': self.cv_scores
        }

        # Remove y_pred arrays to reduce file size (keep only metrics)
        if 'y_pred' in metrics_data['validation']:
            del metrics_data['validation']['y_pred']
            del metrics_data['validation']['y_pred_proba']
        if 'y_pred' in metrics_data['test']:
            del metrics_data['test']['y_pred']
            del metrics_data['test']['y_pred_proba']

        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"  ✓ Saved evaluation metrics")

        logger.info("="*70)

    def get_roc_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Get ROC curve data

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Tuple of (fpr, tpr, thresholds, auc)
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)

        return fpr, tpr, thresholds, auc

    def print_summary(self):
        """Print evaluation summary"""
        logger.info("\n" + "="*70)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*70)

        if self.val_metrics:
            logger.info("\nValidation Set:")
            logger.info(f"  - Accuracy:  {self.val_metrics['accuracy']:.4f}")
            logger.info(f"  - Precision: {self.val_metrics['precision']:.4f}")
            logger.info(f"  - Recall:    {self.val_metrics['recall']:.4f}")
            logger.info(f"  - F1-Score:  {self.val_metrics['f1_score']:.4f}")
            logger.info(f"  - ROC-AUC:   {self.val_metrics['roc_auc']:.4f}")

        if self.test_metrics:
            logger.info("\nTest Set:")
            logger.info(f"  - Accuracy:  {self.test_metrics['accuracy']:.4f}")
            logger.info(f"  - Precision: {self.test_metrics['precision']:.4f}")
            logger.info(f"  - Recall:    {self.test_metrics['recall']:.4f}")
            logger.info(f"  - F1-Score:  {self.test_metrics['f1_score']:.4f}")
            logger.info(f"  - ROC-AUC:   {self.test_metrics['roc_auc']:.4f}")

        if self.cv_scores:
            logger.info("\nCross Validation (Mean ± Std):")
            for metric, scores in self.cv_scores.items():
                logger.info(f"  - {metric:12s}: {scores['mean']:.4f} ± {scores['std']:.4f}")

        if self.feature_importance is not None:
            logger.info("\nTop 5 Features:")
            for idx, row in self.feature_importance.head(5).iterrows():
                logger.info(f"  {row['rank']}. {row['feature']}: {row['importance_normalized']:.2f}%")

        logger.info("="*70)


def main():
    """Main function to test model evaluation"""
    logger.info("Testing Step 6: Model Evaluation")

    # Import previous steps
    from step1_2_setup_and_load_data import DataLoader
    from step3_feature_engineering import FeatureEngineering
    from step4_extract_training_data import TrainingDataExtractor
    from step5_train_random_forest import RandomForestTrainer
    from config import FEATURE_NAMES

    # Load data
    loader = DataLoader()
    data = loader.load_all()

    # Engineer features
    engineer = FeatureEngineering()
    feature_stack, valid_mask = engineer.engineer_features(
        data['s2_before'],
        data['s2_after'],
        data['s1_before'],
        data['s1_after']
    )

    # Extract training data
    extractor = TrainingDataExtractor()
    training_df = extractor.extract_pixel_values(
        feature_stack,
        data['ground_truth'],
        data['metadata']['s2_before']['transform']
    )

    X = training_df[FEATURE_NAMES].values
    y = training_df['label'].values

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = extractor.split_data(X, y)

    # Train model
    trainer = RandomForestTrainer()
    model = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate_all(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        FEATURE_NAMES
    )

    # Save results
    evaluator.save_results()

    # Print summary
    evaluator.print_summary()

    return evaluator, results


if __name__ == "__main__":
    evaluator, results = main()
