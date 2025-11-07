"""
Training script for Simple CNN model with full optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from .config import *
from .models import SimpleCNN, count_parameters
from .dataset import load_patches, create_dataloaders
from .utils import set_seed


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, verbose=True, delta=0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            verbose: Print messages
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps=1):
    """Train for one epoch with gradient accumulation and AMP."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass with AMP
        with autocast(enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Scale loss for gradient accumulation

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Statistics
        running_loss += loss.item() * accumulation_steps * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Handle any remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            with autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs


def train_simple_cnn(patches_file, save_model=True, save_results=True):
    """
    Train Simple CNN model with full optimization.

    Args:
        patches_file: Path to patches pickle file
        save_model: Whether to save trained model
        save_results: Whether to save results

    Returns:
        model: Trained model
        results: Dictionary containing metrics and history
    """
    print("=" * 70)
    print("SIMPLE CNN - TRAINING")
    print("=" * 70)

    # Set random seeds
    set_seed(RANDOM_SEED)

    # Device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # ============================================================
    # 1. Load data and create dataloaders
    # ============================================================
    print("\n[1/7] Loading patches and creating dataloaders...")
    patches, labels = load_patches(patches_file)

    train_loader, val_loader, test_loader = create_dataloaders(
        patches, labels,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )

    print(f"\nDataLoader configuration:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Num workers: {NUM_WORKERS}")
    print(f"  - Pin memory: {PIN_MEMORY}")
    print(f"  - Prefetch factor: {PREFETCH_FACTOR}")
    print(f"  - Gradient accumulation steps: {ACCUMULATION_STEPS}")
    print(f"  - Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")

    # ============================================================
    # 2. Create model
    # ============================================================
    print("\n[2/7] Creating SimpleCNN model...")
    model = SimpleCNN(in_channels=TOTAL_INPUT_CHANNELS, num_classes=NUM_CLASSES)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters:")
    print(f"  - Total: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  - Trainable: {trainable_params:,}")

    # ============================================================
    # 3. Setup training components
    # ============================================================
    print("\n[3/7] Setting up training components...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler(enabled=USE_AMP)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

    print(f"Training configuration:")
    print(f"  - Optimizer: AdamW")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Weight decay: {WEIGHT_DECAY}")
    print(f"  - Loss function: CrossEntropyLoss")
    print(f"  - LR scheduler: ReduceLROnPlateau")
    print(f"  - Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  - Mixed precision (AMP): {USE_AMP}")
    print(f"  - Max epochs: {NUM_EPOCHS}")

    # ============================================================
    # 4. Training loop
    # ============================================================
    print("\n[4/7] Starting training...")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    best_model_path = MODELS_DIR / "simple_cnn_best.pth"

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ACCUMULATION_STEPS
        )

        # Validate
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_model_path)
            print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    # ============================================================
    # 5. Load best model and evaluate
    # ============================================================
    print("\n[5/7] Loading best model and evaluating...")

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # Evaluate on all splits
    def evaluate_split(loader, split_name):
        val_loss, val_acc, y_true, y_pred, y_proba = validate(model, loader, criterion, device)

        metrics = {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        print(f"\n{split_name} Metrics:")
        print(f"  Loss:      {metrics['loss']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

        return metrics, y_true, y_pred, y_proba

    train_metrics, y_train_true, y_train_pred, y_train_proba = evaluate_split(train_loader, "Train")
    val_metrics, y_val_true, y_val_pred, y_val_proba = evaluate_split(val_loader, "Validation")
    test_metrics, y_test_true, y_test_pred, y_test_proba = evaluate_split(test_loader, "Test")

    print("\nTest Set - Classification Report:")
    print(classification_report(y_test_true, y_test_pred, target_names=CLASS_NAMES))

    # ============================================================
    # 6. Save results
    # ============================================================
    print("\n[6/7] Saving results...")

    results = {
        'model_type': 'SimpleCNN',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS,
            'effective_batch_size': BATCH_SIZE * ACCUMULATION_STEPS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'num_epochs_trained': len(history['train_loss']),
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'use_amp': USE_AMP,
            'random_seed': RANDOM_SEED
        },
        'model_params': {
            'total': total_params,
            'trainable': trainable_params
        },
        'training_time': training_time,
        'best_epoch': checkpoint['epoch'] + 1,
        'history': history,
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        'predictions': {
            'train': {'y_true': y_train_true, 'y_pred': y_train_pred, 'y_proba': y_train_proba},
            'val': {'y_true': y_val_true, 'y_pred': y_val_pred, 'y_proba': y_val_proba},
            'test': {'y_true': y_test_true, 'y_pred': y_test_pred, 'y_proba': y_test_proba}
        }
    }

    if save_results:
        results_path = RESULTS_DIR / "simple_cnn_results.json"
        # Convert numpy arrays to lists for JSON serialization
        results_json = results.copy()
        for split in ['train', 'val', 'test']:
            for key in ['y_true', 'y_pred', 'y_proba']:
                results_json['predictions'][split][key] = [float(x) for x in results_json['predictions'][split][key]]

        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"✓ Results saved to: {results_path}")

    # ============================================================
    # 7. Create visualizations
    # ============================================================
    print("\n[7/7] Creating visualizations...")

    fig = plt.figure(figsize=(20, 10))

    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Accuracy curves
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. Learning rate
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(epochs, history['learning_rates'], marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(alpha=0.3)

    # 4. Confusion matrix (Test)
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(y_test_true, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 5. ROC curves
    ax5 = plt.subplot(2, 3, 5)
    from sklearn.metrics import roc_curve
    fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_proba)
    fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_proba)

    plt.plot(fpr_train, tpr_train, label=f'Train (AUC={train_metrics["auc"]:.3f})')
    plt.plot(fpr_val, tpr_val, label=f'Val (AUC={val_metrics["auc"]:.3f})')
    plt.plot(fpr_test, tpr_test, label=f'Test (AUC={test_metrics["auc"]:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(alpha=0.3)

    # 6. Metrics comparison
    ax6 = plt.subplot(2, 3, 6)
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
    fig_path = FIGURES_DIR / "simple_cnn_training.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {fig_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("SIMPLE CNN TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Best model saved to: {best_model_path}")
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1-Score: {test_metrics['f1']:.4f}")

    return model, results


if __name__ == "__main__":
    # Train Simple CNN
    patches_file = PATCHES_DIR / "patches_64x64.pkl"

    if not patches_file.exists():
        print(f"Error: Patches file not found at {patches_file}")
        print("Please run preprocessing first to create patches.")
    else:
        model, results = train_simple_cnn(patches_file)
