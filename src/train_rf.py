"""
Training script for Random Forest baseline model
"""
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from .config import *
from .feature_extraction import extract_features_from_patches, get_feature_names
from .dataset import load_patches


def train_random_forest(patches_file, save_model=True, save_results=True):
    """
    Train Random Forest baseline model.

    Args:
        patches_file: Path to patches pickle file
        save_model: Whether to save trained model
        save_results: Whether to save results

    Returns:
        model: Trained Random Forest model
        results: Dictionary containing metrics and predictions
    """
    print("=" * 70)
    print("RANDOM FOREST BASELINE - TRAINING")
    print("=" * 70)

    # ============================================================
    # 1. Load patches
    # ============================================================
    print("\n[1/6] Loading patches...")
    patches, labels = load_patches(patches_file)
    print(f"Loaded {len(patches)} patches")
    print(f"  - No deforestation (0): {(labels == 0).sum()} ({(labels == 0).sum() / len(labels) * 100:.1f}%)")
    print(f"  - Deforestation (1): {(labels == 1).sum()} ({(labels == 1).sum() / len(labels) * 100:.1f}%)")

    # ============================================================
    # 2. Extract features
    # ============================================================
    print("\n[2/6] Extracting handcrafted features...")
    start_time = time.time()
    features = extract_features_from_patches(patches, verbose=True)
    extraction_time = time.time() - start_time
    print(f"Feature extraction took {extraction_time:.2f} seconds")

    # ============================================================
    # 3. Split data
    # ============================================================
    print("\n[3/6] Splitting data...")
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # Second split: val vs test
    val_size_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size_adjusted),
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"Data split:")
    print(f"  - Train: {len(X_train)} samples ({len(X_train) / len(features) * 100:.1f}%)")
    print(f"  - Val:   {len(X_val)} samples ({len(X_val) / len(features) * 100:.1f}%)")
    print(f"  - Test:  {len(X_test)} samples ({len(X_test) / len(features) * 100:.1f}%)")

    # ============================================================
    # 4. Train Random Forest
    # ============================================================
    print("\n[4/6] Training Random Forest...")
    print(f"Configuration:")
    print(f"  - n_estimators: {RF_N_ESTIMATORS}")
    print(f"  - max_depth: {RF_MAX_DEPTH}")
    print(f"  - min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  - n_jobs: {RF_N_JOBS}")

    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_SEED,
        verbose=1
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    # ============================================================
    # 5. Evaluate on all splits
    # ============================================================
    print("\n[5/6] Evaluating model...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Probabilities for AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics for each split
    def calculate_metrics(y_true, y_pred, y_proba, split_name):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        print(f"\n{split_name} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

        return metrics

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, "Train")
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba, "Validation")
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "Test")

    # Classification report
    print("\nTest Set - Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

    # ============================================================
    # 6. Save model and results
    # ============================================================
    print("\n[6/6] Saving model and results...")

    results = {
        'model_type': 'RandomForest',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'n_estimators': RF_N_ESTIMATORS,
            'max_depth': RF_MAX_DEPTH,
            'min_samples_split': RF_MIN_SAMPLES_SPLIT,
            'random_seed': RANDOM_SEED
        },
        'data_split': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        },
        'training_time': training_time,
        'feature_extraction_time': extraction_time,
        'num_features': features.shape[1],
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        'predictions': {
            'train': {'y_true': y_train.tolist(), 'y_pred': y_train_pred.tolist(), 'y_proba': y_train_proba.tolist()},
            'val': {'y_true': y_val.tolist(), 'y_pred': y_val_pred.tolist(), 'y_proba': y_val_proba.tolist()},
            'test': {'y_true': y_test.tolist(), 'y_pred': y_test_pred.tolist(), 'y_proba': y_test_proba.tolist()}
        }
    }

    # Save model
    if save_model:
        model_path = MODELS_DIR / "random_forest_baseline.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to: {model_path}")

    # Save results
    if save_results:
        results_path = RESULTS_DIR / "random_forest_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {results_path}")

    # ============================================================
    # 7. Visualizations
    # ============================================================
    print("\n[7/7] Creating visualizations...")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5))

    # 1. Confusion Matrix (Test Set)
    ax1 = plt.subplot(1, 4, 1)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 2. Feature Importance (Top 20)
    ax2 = plt.subplot(1, 4, 2)
    feature_names = get_feature_names()
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=8)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()

    # 3. ROC Curve
    ax3 = plt.subplot(1, 4, 3)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

    plt.plot(fpr_train, tpr_train, label=f'Train (AUC={train_metrics["auc"]:.3f})')
    plt.plot(fpr_val, tpr_val, label=f'Val (AUC={val_metrics["auc"]:.3f})')
    plt.plot(fpr_test, tpr_test, label=f'Test (AUC={test_metrics["auc"]:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(alpha=0.3)

    # 4. Metrics Comparison
    ax4 = plt.subplot(1, 4, 4)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    train_values = [train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1'], train_metrics['auc']]
    val_values = [val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1'], val_metrics['auc']]
    test_values = [test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1'], test_metrics['auc']]

    x = np.arange(len(metrics_names))
    width = 0.25
    plt.bar(x - width, train_values, width, label='Train', alpha=0.8)
    plt.bar(x, val_values, width, label='Val', alpha=0.8)
    plt.bar(x + width, test_values, width, label='Test', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Metrics Comparison')
    plt.xticks(x, metrics_names)
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "random_forest_evaluation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {fig_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("RANDOM FOREST TRAINING COMPLETED!")
    print("=" * 70)

    return model, results


if __name__ == "__main__":
    # Train Random Forest
    patches_file = PATCHES_DIR / "patches_64x64.pkl"

    if not patches_file.exists():
        print(f"Error: Patches file not found at {patches_file}")
        print("Please run preprocessing first to create patches.")
    else:
        model, results = train_random_forest(patches_file)
        print(f"\nFinal Test Accuracy: {results['metrics']['test']['accuracy']:.4f}")
        print(f"Final Test F1-Score: {results['metrics']['test']['f1']:.4f}")
