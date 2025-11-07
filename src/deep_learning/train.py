"""
Training module for 2D CNN
Includes training loop, validation, and model saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List
import logging
from pathlib import Path
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """PyTorch Dataset for patches"""

    def __init__(self, patches: np.ndarray, labels: np.ndarray):
        """
        Args:
            patches: (n_samples, patch_size, patch_size, n_features)
            labels: (n_samples,)
        """
        self.patches = torch.FloatTensor(patches)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


class CNNTrainer:
    """
    Trainer for CNN model
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: List[float] = None
    ):
        """
        Initialize trainer

        Args:
            model: CNN model
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            class_weights: Weights for imbalanced classes
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_state = None

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for patches, labels in train_loader:
            patches = patches.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(patches)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * patches.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(patches)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * patches.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the model

        Args:
            X_train: Training patches
            y_train: Training labels
            X_val: Validation patches
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dictionary
        """
        logger.info(f"\n{'='*70}")
        logger.info("STARTING CNN TRAINING")
        logger.info(f"{'='*70}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Device: {self.device}")
        logger.info(f"{'='*70}\n")

        # Create datasets and dataloaders
        train_dataset = PatchDataset(X_train, y_train)
        val_dataset = PatchDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # Training loop
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Log progress
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
                logger.info(f"  → New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"\n✓ Restored best model (Val Loss: {self.best_val_loss:.4f}, Val Acc: {self.best_val_acc:.2f}%)")

        logger.info(f"\n{'='*70}")
        logger.info("TRAINING COMPLETED")
        logger.info(f"{'='*70}\n")

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32) -> Dict:
        """
        Evaluate model on test set

        Args:
            X_test: Test patches
            y_test: Test labels
            batch_size: Batch size

        Returns:
            Dictionary with metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info("EVALUATING ON TEST SET")
        logger.info(f"{'='*70}")

        test_dataset = PatchDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for patches, labels in test_loader:
                patches = patches.to(self.device)
                outputs = self.model(patches)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

        logger.info(f"\nTest Set Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        logger.info(f"{'='*70}\n")

        return metrics

    def save_model(self, save_path: Path):
        """Save model checkpoint"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to: {save_path}")

    def load_model(self, load_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        logger.info(f"Model loaded from: {load_path}")

    def predict(self, X: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on new data

        Args:
            X: Input patches
            batch_size: Batch size

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        dataset = PatchDataset(X, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for patches, _ in loader:
                patches = patches.to(self.device)
                outputs = self.model(patches)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)
