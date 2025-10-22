"""
Utility functions for training, evaluation, and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns


class Trainer:
    """Training manager with progress tracking"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        save_dir='models',
        model_name='model'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).long()  # Ensure labels are Long type

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)  # (B, 2, H, W)

            # For binary classification, we need to get per-pixel labels
            # Assuming labels are (B,) - expand to match output spatial dims
            if len(labels.shape) == 1:
                # labels is (B,) - this is for patch-level classification
                # We'll use the center pixel or average pooling
                outputs_pooled = outputs.mean(dim=(2, 3))  # (B, 2)
                loss = self.criterion(outputs_pooled, labels)

                # Predictions
                preds = outputs_pooled.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                # Pixel-wise labels
                loss = self.criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).long()  # Ensure labels are Long type

            outputs = self.model(images)

            if len(labels.shape) == 1:
                outputs_pooled = outputs.mean(dim=(2, 3))
                loss = self.criterion(outputs_pooled, labels)
                preds = outputs_pooled.argmax(dim=1)
            else:
                loss = self.criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.numel() if len(labels.shape) > 1 else labels.size(0)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, avg_acc, f1

    def train(self, epochs=50, early_stopping_patience=10):
        """
        Train for multiple epochs

        Args:
            epochs: Number of epochs
            early_stopping_patience: Stop if no improvement for N epochs

        Returns:
            history: Training history dict
        """
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING STARTED")
        print(f"{'='*80}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Save directory: {self.save_dir}")
        print(f"{'='*80}\n")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            print(f"\n📅 Epoch {epoch}/{epochs}")
            print("-" * 80)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_f1 = self.validate()

            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ Best model saved! (Val Acc: {val_acc*100:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ No improvement ({patience_counter}/{early_stopping_patience})")

            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⛔ Early stopping triggered at epoch {epoch}")
                print(f"Best epoch: {self.best_epoch} (Val Acc: {self.best_val_acc*100:.2f}%)")
                break

        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"Model saved to: {self.save_dir / 'best_model.pth'}")
        print(f"{'='*80}\n")

        return self.history

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }

        if is_best:
            save_path = self.save_dir / 'best_model.pth'
        else:
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, save_path)


def plot_training_history(history, save_path=None):
    """
    Plot training history

    Args:
        history: Dict with keys ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1']
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot([x*100 for x in history['train_acc']], label='Train Acc', marker='o')
    axes[0, 1].plot([x*100 for x in history['val_acc']], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='d', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning Rate
    axes[1, 1].plot(history['lr'], label='Learning Rate', marker='x', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")

    plt.show()


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set

    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        device: Device to use

    Returns:
        metrics: Dict with accuracy, F1, precision, recall
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    print("\n🧪 Evaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)

            if len(labels.shape) == 1:
                outputs_pooled = outputs.mean(dim=(2, 3))
                preds = outputs_pooled.argmax(dim=1)
            else:
                preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"{'='*80}\n")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return metrics


def sliding_window_inference(
    model,
    s1_t1_path,
    s1_t2_path,
    s2_t1_path,
    s2_t2_path,
    window_size=256,
    overlap=32,
    batch_size=16,
    device='cuda'
):
    """
    Perform sliding window inference on whole scene

    Args:
        model: Trained PyTorch model
        s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path: Paths to TIFF files
        window_size: Size of sliding window
        overlap: Overlap between windows
        batch_size: Batch size for inference
        device: Device to use

    Returns:
        probability_map: Numpy array (H, W) with probabilities [0, 1]
    """
    model.eval()
    model.to(device)

    # Open all files
    print("\n🗺️ Loading GeoTIFF files...")
    s1_t1 = rasterio.open(s1_t1_path)
    s1_t2 = rasterio.open(s1_t2_path)
    s2_t1 = rasterio.open(s2_t1_path)
    s2_t2 = rasterio.open(s2_t2_path)

    # Get dimensions
    height = s1_t1.height
    width = s1_t1.width
    print(f"Scene size: {width} × {height}")

    # Initialize output arrays
    probability_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)

    # Calculate step size
    step = window_size - overlap

    # Generate windows
    windows = []
    for y in range(0, height - window_size + 1, step):
        for x in range(0, width - window_size + 1, step):
            windows.append(Window(x, y, window_size, window_size))

    print(f"Total windows: {len(windows)}")
    print(f"Window size: {window_size}×{window_size}, Overlap: {overlap}, Step: {step}")

    # Process in batches
    print("\n🔄 Running inference...")
    for i in tqdm(range(0, len(windows), batch_size)):
        batch_windows = windows[i:i+batch_size]
        batch_images = []

        # Read patches
        for window in batch_windows:
            s1_t1_patch = s1_t1.read(window=window)
            s1_t2_patch = s1_t2.read(window=window)
            s2_t1_patch = s2_t1.read(window=window)
            s2_t2_patch = s2_t2.read(window=window)

            # Stack to 18 channels
            patch = np.concatenate([
                s1_t1_patch, s2_t1_patch,
                s1_t2_patch, s2_t2_patch
            ], axis=0)

            # Normalize
            patch = np.nan_to_num(patch, nan=0.0)
            batch_images.append(patch)

        # Convert to tensor
        batch_tensor = torch.from_numpy(np.array(batch_images)).float().to(device)

        # Predict
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1, :, :]  # Probability of class 1

        # Write to output map
        for j, window in enumerate(batch_windows):
            prob_patch = probs[j].cpu().numpy()
            probability_map[window.row_off:window.row_off+window.height,
                          window.col_off:window.col_off+window.width] += prob_patch
            count_map[window.row_off:window.row_off+window.height,
                     window.col_off:window.col_off+window.width] += 1

    # Average overlapping predictions
    probability_map = probability_map / np.maximum(count_map, 1)

    # Close files
    s1_t1.close()
    s1_t2.close()
    s2_t1.close()
    s2_t2.close()

    print("✅ Inference completed!\n")

    return probability_map


def save_geotiff(array, reference_tiff_path, output_path, dtype='float32'):
    """
    Save numpy array as GeoTIFF with georeferencing from reference

    Args:
        array: Numpy array (H, W)
        reference_tiff_path: Path to reference TIFF for georeferencing
        output_path: Output path
        dtype: Data type ('float32' or 'uint8')
    """
    with rasterio.open(reference_tiff_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype=dtype,
        count=1,
        compress='lzw'
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(dtype), 1)

    print(f"✅ Saved: {output_path}")


def visualize_predictions(image, prediction, title='Prediction'):
    """
    Visualize prediction on image

    Args:
        image: Input image (C, H, W) or (H, W, C)
        prediction: Binary prediction (H, W)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Input image (use first 3 channels as RGB)
    if image.shape[0] <= 3:
        img_rgb = np.transpose(image[:3], (1, 2, 0))
    else:
        img_rgb = np.transpose(image[:3], (1, 2, 0))

    axes[0].imshow(img_rgb)
    axes[0].set_title('Input (First 3 bands)')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(prediction, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
