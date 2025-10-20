"""
Enhanced utilities for Jupyter Notebooks with real-time visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display, clear_output
import time


class NotebookTrainer:
    """
    Enhanced Trainer with live visualization for Jupyter notebooks
    """

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
        self.save_dir = save_dir
        self.model_name = model_name

        from pathlib import Path
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

        # For live plotting
        self.fig = None
        self.axes = None

    def train_epoch(self, epoch):
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Create progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch} [Train]',
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device).long()  # Ensure labels are Long type

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Loss calculation
            if len(labels.shape) == 1:
                outputs_pooled = outputs.mean(dim=(2, 3))
                loss = self.criterion(outputs_pooled, labels)
                preds = outputs_pooled.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                loss = self.criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc*100:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, epoch):
        """Validate with progress bar"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch} [Val]  ',
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

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

        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, avg_acc, f1

    def plot_progress(self, epoch):
        """Live plotting of training progress"""
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle(f'Training Progress - {self.model_name}', fontsize=16, fontweight='bold')

        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        self.axes[0, 0].plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
        self.axes[0, 0].plot(epochs, self.history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
        self.axes[0, 0].set_xlabel('Epoch', fontsize=12)
        self.axes[0, 0].set_ylabel('Loss', fontsize=12)
        self.axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
        self.axes[0, 0].legend(fontsize=11)
        self.axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        self.axes[0, 1].plot(epochs, [x*100 for x in self.history['train_acc']], 'b-o', label='Train Acc', linewidth=2, markersize=6)
        self.axes[0, 1].plot(epochs, [x*100 for x in self.history['val_acc']], 'r-s', label='Val Acc', linewidth=2, markersize=6)
        self.axes[0, 1].set_xlabel('Epoch', fontsize=12)
        self.axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        self.axes[0, 1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
        self.axes[0, 1].legend(fontsize=11)
        self.axes[0, 1].grid(True, alpha=0.3)
        # Mark best epoch
        best_idx = self.best_epoch - 1 if self.best_epoch > 0 else 0
        if best_idx < len(self.history['val_acc']):
            self.axes[0, 1].scatter([self.best_epoch], [self.history['val_acc'][best_idx]*100],
                                   color='green', s=200, marker='*', zorder=5, label=f'Best (Epoch {self.best_epoch})')
            self.axes[0, 1].legend(fontsize=11)

        # F1 Score
        self.axes[1, 0].plot(epochs, self.history['val_f1'], 'g-d', label='Val F1', linewidth=2, markersize=6)
        self.axes[1, 0].set_xlabel('Epoch', fontsize=12)
        self.axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        self.axes[1, 0].set_title('F1 Score Curve', fontsize=14, fontweight='bold')
        self.axes[1, 0].legend(fontsize=11)
        self.axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        self.axes[1, 1].plot(epochs, self.history['lr'], 'm-x', label='Learning Rate', linewidth=2, markersize=8)
        self.axes[1, 1].set_xlabel('Epoch', fontsize=12)
        self.axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        self.axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        self.axes[1, 1].set_yscale('log')
        self.axes[1, 1].legend(fontsize=11)
        self.axes[1, 1].grid(True, alpha=0.3)

        self.fig.tight_layout()
        clear_output(wait=True)
        display(self.fig)

    def print_epoch_summary(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr):
        """Print formatted epoch summary"""
        print(f"\n{'='*90}")
        print(f"📊 Epoch {epoch} Summary")
        print(f"{'='*90}")
        print(f"  Train → Loss: {train_loss:.4f}  |  Acc: {train_acc*100:.2f}%")
        print(f"  Val   → Loss: {val_loss:.4f}  |  Acc: {val_acc*100:.2f}%  |  F1: {val_f1:.4f}")
        print(f"  LR: {lr:.6f}")
        print(f"{'='*90}")

    def train(self, epochs=50, early_stopping_patience=10, plot_every=1):
        """
        Train with live visualization

        Args:
            epochs: Number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            plot_every: Update plot every N epochs (default: 1)
        """
        print(f"\n{'='*90}")
        print(f"🚀 TRAINING STARTED - {self.model_name.upper()}")
        print(f"{'='*90}")
        print(f"Device: {self.device}")
        print(f"Total Epochs: {epochs}")
        print(f"Early Stopping Patience: {early_stopping_patience}")
        print(f"Save Directory: {self.save_dir}")
        print(f"{'='*90}\n")

        patience_counter = 0

        # Overall progress bar for epochs
        epoch_pbar = tqdm(
            range(1, epochs + 1),
            desc='Overall Progress',
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs'
        )

        for epoch in epoch_pbar:
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_f1 = self.validate(epoch)

            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            # Print summary
            self.print_epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc, val_f1, current_lr)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model! Val Acc: {val_acc*100:.2f}% (saved)")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")

            # Update plot
            if epoch % plot_every == 0:
                self.plot_progress(epoch)

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
                print(f"Best model: Epoch {self.best_epoch} with Val Acc: {self.best_val_acc*100:.2f}%")
                break

            # Update overall progress bar
            epoch_pbar.set_postfix({
                'best_acc': f'{self.best_val_acc*100:.2f}%',
                'patience': f'{patience_counter}/{early_stopping_patience}'
            })

        plt.ioff()

        print(f"\n{'='*90}")
        print(f"✅ TRAINING COMPLETED")
        print(f"{'='*90}")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"Model saved to: {self.save_dir / 'best_model.pth'}")
        print(f"{'='*90}\n")

        # Final plot
        self.plot_progress(epoch)
        plt.savefig(self.save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        print(f"✅ Training history plot saved: {self.save_dir / 'training_history.png'}")

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


def visualize_batch_with_progress(dataset, num_samples=8):
    """
    Visualize random samples from dataset with progress bar

    Args:
        dataset: PyTorch Dataset
        num_samples: Number of samples to visualize
    """
    import random

    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(2, num_samples//2, figsize=(20, 8))
    axes = axes.flatten()

    print(f"Loading {num_samples} random samples...")
    for i, idx in enumerate(tqdm(indices, desc='Loading samples', ncols=100)):
        img, label = dataset[idx]

        # Convert to numpy for visualization
        if torch.is_tensor(img):
            img = img.numpy()

        # Use first 3 channels
        img_vis = img[:3].transpose(1, 2, 0)
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)

        label_text = 'No Change' if label == 0 else 'Change'
        axes[i].imshow(img_vis)
        axes[i].set_title(f'Sample {idx}: {label_text}', fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    print(f"✅ Displayed {num_samples} samples")


def print_training_schedule(epochs, batch_size, total_samples, gpu_name=None):
    """
    Print estimated training schedule

    Args:
        epochs: Number of epochs
        batch_size: Batch size
        total_samples: Total training samples
        gpu_name: Name of GPU (optional)
    """
    batches_per_epoch = total_samples // batch_size
    total_batches = batches_per_epoch * epochs

    # Estimate time (rough estimates)
    seconds_per_batch = {
        'RTX 4090': 0.1,
        'RTX 3090': 0.15,
        'RTX 3080': 0.2,
        'A100': 0.08,
        'V100': 0.12,
        'default': 0.2
    }

    gpu_key = 'default'
    if gpu_name:
        for key in seconds_per_batch:
            if key in gpu_name:
                gpu_key = key
                break

    estimated_seconds = total_batches * seconds_per_batch[gpu_key]
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60

    print(f"\n{'='*80}")
    print(f"📅 TRAINING SCHEDULE ESTIMATE")
    print(f"{'='*80}")
    print(f"Total epochs: {epochs}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total batches: {total_batches:,}")
    print(f"\nEstimated time:")
    print(f"  Per epoch: ~{estimated_minutes/epochs:.1f} minutes")
    print(f"  Total: ~{estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    if gpu_name:
        print(f"\nGPU: {gpu_name}")
    print(f"{'='*80}\n")
