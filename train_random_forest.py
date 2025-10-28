"""
Train Random Forest model for deforestation detection

This script trains a Random Forest classifier on the patches dataset.
Unlike CNNs, Random Forest requires flattened features (1D vectors).

Usage:
    python train_random_forest.py

Output:
    - Trained model: checkpoints/random_forest_best.pkl
    - Training history: logs/random_forest_training.txt
"""
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from ml_models import RandomForestModel, load_patches_for_ml


def main():
    """Main training function"""

    print("=" * 80)
    print("RANDOM FOREST TRAINING FOR DEFORESTATION DETECTION")
    print("=" * 80)
    print()

    # Paths
    patches_dir = project_root / 'data' / 'patches'
    checkpoints_dir = project_root / 'checkpoints'
    logs_dir = project_root / 'logs'

    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading patches...")
    X_train, y_train = load_patches_for_ml(patches_dir, 'train')
    X_val, y_val = load_patches_for_ml(patches_dir, 'val')
    X_test, y_test = load_patches_for_ml(patches_dir, 'test')

    print()
    print("Dataset summary:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]} (14 bands × 128 × 128 pixels)")
    print()

    # Model configuration
    print("Model configuration:")
    model_config = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'random_state': 42,
        'n_jobs': -1  # Use all CPU cores
    }

    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print()

    # Create model
    model = RandomForestModel(**model_config)

    # Train model
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()

    start_time = datetime.now()
    metrics = model.train(X_train, y_train, X_val, y_val)
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print()
    print("Training completed!")
    print(f"Training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")
    print()

    # Print training metrics
    print("=" * 80)
    print("TRAINING METRICS")
    print("=" * 80)
    print()
    print("Train set:")
    print(f"  Accuracy:  {metrics['train_acc']:.4f}")
    print(f"  Precision: {metrics['train_precision']:.4f}")
    print(f"  Recall:    {metrics['train_recall']:.4f}")
    print(f"  F1 Score:  {metrics['train_f1']:.4f}")
    print(f"  AUC:       {metrics['train_auc']:.4f}")
    print()
    print("Validation set:")
    print(f"  Accuracy:  {metrics['val_acc']:.4f}")
    print(f"  Precision: {metrics['val_precision']:.4f}")
    print(f"  Recall:    {metrics['val_recall']:.4f}")
    print(f"  F1 Score:  {metrics['val_f1']:.4f}")
    print(f"  AUC:       {metrics['val_auc']:.4f}")
    print()

    # Evaluate on test set
    print("=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    print()

    test_metrics = model.evaluate(X_test, y_test)

    print("Test set:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print()

    # Feature importance
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    print()

    importance = model.get_feature_importance()
    print(f"Total features: {len(importance)}")
    print(f"Feature importance range: [{importance.min():.6f}, {importance.max():.6f}]")
    print(f"Mean importance: {importance.mean():.6f}")
    print()

    # Reshape importance to (14, 128, 128) for analysis
    # This shows which bands and spatial locations are most important
    importance_map = importance.reshape(128, 128, 14)
    band_importance = importance_map.mean(axis=(0, 1))  # Average importance per band

    print("Average importance by band:")
    band_names = [
        'Blue_2024', 'Green_2024', 'Red_2024', 'NIR_2024',
        'NDVI_2024', 'NBR_2024', 'NDMI_2024',
        'Blue_2025', 'Green_2025', 'Red_2025', 'NIR_2025',
        'NDVI_2025', 'NBR_2025', 'NDMI_2025'
    ]
    for i, (name, imp) in enumerate(zip(band_names, band_importance)):
        print(f"  Band {i:2d} ({name:12s}): {imp:.6f}")
    print()

    # Save model
    model_path = checkpoints_dir / 'random_forest_best.pkl'
    model.save(model_path)
    print()

    # Save training log
    log_path = logs_dir / 'random_forest_training.txt'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RANDOM FOREST TRAINING LOG\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Training date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)\n\n")

        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in model_config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("DATASET:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Train: {len(X_train)} samples\n")
        f.write(f"  Val: {len(X_val)} samples\n")
        f.write(f"  Test: {len(X_test)} samples\n")
        f.write(f"  Features: {X_train.shape[1]}\n\n")

        f.write("TRAINING METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write("Train set:\n")
        f.write(f"  Accuracy:  {metrics['train_acc']:.4f}\n")
        f.write(f"  Precision: {metrics['train_precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['train_recall']:.4f}\n")
        f.write(f"  F1 Score:  {metrics['train_f1']:.4f}\n")
        f.write(f"  AUC:       {metrics['train_auc']:.4f}\n\n")

        f.write("Validation set:\n")
        f.write(f"  Accuracy:  {metrics['val_acc']:.4f}\n")
        f.write(f"  Precision: {metrics['val_precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['val_recall']:.4f}\n")
        f.write(f"  F1 Score:  {metrics['val_f1']:.4f}\n")
        f.write(f"  AUC:       {metrics['val_auc']:.4f}\n\n")

        f.write("TEST METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:  {test_metrics['f1']:.4f}\n")
        f.write(f"  AUC:       {test_metrics['auc']:.4f}\n\n")

        f.write("FEATURE IMPORTANCE BY BAND:\n")
        f.write("-" * 80 + "\n")
        for i, (name, imp) in enumerate(zip(band_names, band_importance)):
            f.write(f"  Band {i:2d} ({name:12s}): {imp:.6f}\n")

    print(f"Training log saved to: {log_path}")
    print()

    # Summary
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  1. Model: {model_path}")
    print(f"  2. Training log: {log_path}")
    print()
    print("Next steps:")
    print("  - Compare with CNN models in notebook 04")
    print("  - Run full-image inference with Random Forest")
    print("  - Analyze feature importance for interpretability")
    print()


if __name__ == "__main__":
    main()
