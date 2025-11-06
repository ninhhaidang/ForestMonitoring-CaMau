"""
Visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import rasterio
from rasterio.plot import show


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    return plt.gcf()


def plot_roc_curve(y_true, y_scores, figsize=(8, 6)):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_training_history(history, figsize=(12, 4)):
    """
    Plot training history (loss and metrics)

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_patch(patch, title="Patch", figsize=(12, 4)):
    """
    Visualize a multi-channel patch

    Args:
        patch: Numpy array of shape (C, H, W)
        title: Plot title
        figsize: Figure size
    """
    num_channels = min(patch.shape[0], 6)  # Show max 6 channels
    fig, axes = plt.subplots(1, num_channels, figsize=figsize)

    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        axes[i].imshow(patch[i], cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_rgb_composite(s2_image, bands=[2, 1, 0], title="RGB Composite",
                            figsize=(10, 10), percentile=2):
    """
    Visualize RGB composite from Sentinel-2

    Args:
        s2_image: Sentinel-2 image array (C, H, W)
        bands: Band indices for R, G, B [default B4, B3, B2]
        title: Plot title
        figsize: Figure size
        percentile: Percentile for stretching
    """
    rgb = np.stack([s2_image[bands[0]], s2_image[bands[1]], s2_image[bands[2]]], axis=-1)

    # Percentile stretch for better visualization
    p2, p98 = np.percentile(rgb, (percentile, 100-percentile))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    return plt.gcf()


def plot_predictions_comparison(patches, true_labels, pred_labels,
                                num_samples=10, figsize=(15, 6)):
    """
    Plot comparison of true vs predicted labels

    Args:
        patches: Array of patches
        true_labels: True labels
        pred_labels: Predicted labels
        num_samples: Number of samples to show
        figsize: Figure size
    """
    # Randomly select samples
    indices = np.random.choice(len(patches), size=min(num_samples, len(patches)), replace=False)

    fig, axes = plt.subplots(2, num_samples, figsize=figsize)

    for i, idx in enumerate(indices):
        # Show first channel (arbitrary)
        axes[0, i].imshow(patches[idx][0], cmap='viridis')
        axes[0, i].set_title(f'True: {true_labels[idx]}')
        axes[0, i].axis('off')

        axes[1, i].imshow(patches[idx][0], cmap='viridis')
        axes[1, i].set_title(f'Pred: {pred_labels[idx]}')
        axes[1, i].axis('off')

    plt.tight_layout()
    return fig
