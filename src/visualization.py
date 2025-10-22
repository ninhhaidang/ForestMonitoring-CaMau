"""
Visualization and plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Union
import rasterio
from rasterio.plot import show


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_band(data: np.ndarray,
              title: str = "Band",
              cmap: str = 'RdYlGn',
              figsize: Tuple[int, int] = (10, 8),
              percentile_clip: Tuple[float, float] = (2, 98),
              save_path: Optional[Union[str, Path]] = None):
    """
    Plot a single band

    Args:
        data: 2D array
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        percentile_clip: Percentile range for display (vmin, vmax)
        save_path: Path to save figure (optional)

    Example:
        >>> from src import load_tiff, plot_band
        >>> data = load_tiff('S1_2024.tif', bands=[1])
        >>> plot_band(data, title='S1 VH 2024', cmap='viridis')
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Calculate display range
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, percentile_clip[0])
        vmax = np.percentile(valid_data, percentile_clip[1])
    else:
        vmin, vmax = data.min(), data.max()

    # Plot
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()


def plot_band_comparison(data_list: List[np.ndarray],
                        titles: List[str],
                        cmap: str = 'RdYlGn',
                        figsize: Tuple[int, int] = (16, 8),
                        suptitle: str = "Band Comparison",
                        save_path: Optional[Union[str, Path]] = None):
    """
    Plot multiple bands side by side for comparison

    Args:
        data_list: List of 2D arrays
        titles: List of titles for each band
        cmap: Colormap name
        figsize: Figure size
        suptitle: Main title
        save_path: Path to save figure

    Example:
        >>> from src import load_tiff, plot_band_comparison
        >>> data1 = load_tiff('S1_2024.tif', bands=[1])
        >>> data2 = load_tiff('S1_2025.tif', bands=[1])
        >>> plot_band_comparison([data1, data2],
        >>>                      titles=['2024', '2025'],
        >>>                      suptitle='S1 VH Comparison')
    """
    n_bands = len(data_list)
    ncols = min(4, n_bands)
    nrows = (n_bands + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_bands == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_bands > 1 else [axes]

    fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    for idx, (data, title) in enumerate(zip(data_list, titles)):
        ax = axes[idx]

        # Calculate display range
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = data.min(), data.max()

        # Plot
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()


def plot_statistics(stats_dict: dict,
                   metric: str = 'mean',
                   figsize: Tuple[int, int] = (15, 10),
                   save_path: Optional[Union[str, Path]] = None):
    """
    Plot statistics comparison across multiple files

    Args:
        stats_dict: Dictionary of {name: DataFrame} from get_tiff_stats
        metric: Metric to plot ('mean', 'std', 'nan_percent', etc.)
        figsize: Figure size
        save_path: Path to save figure

    Example:
        >>> from src import get_tiff_stats, plot_statistics
        >>> stats_s1_2024 = get_tiff_stats('S1_2024.tif')
        >>> stats_s1_2025 = get_tiff_stats('S1_2025.tif')
        >>> plot_statistics({'2024': stats_s1_2024, '2025': stats_s1_2025},
        >>>                 metric='mean')
    """
    n_files = len(stats_dict)
    ncols = 2
    nrows = (n_files + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_files == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f'{metric.upper()} Comparison', fontsize=16, fontweight='bold')

    for idx, (name, df) in enumerate(stats_dict.items()):
        ax = axes[idx]

        # Plot bar chart
        ax.bar(df['band'], df[metric], alpha=0.7, color=sns.color_palette()[idx % 10])

        if metric == 'nan_percent':
            ax.set_ylabel(f'{metric} (%)', fontsize=12)
        else:
            ax.set_ylabel(metric, fontsize=12)
            if metric in ['mean', 'std']:
                ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Band Number', fontsize=12)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df['band'])

    # Hide extra subplots
    for idx in range(n_files, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()


def plot_indices_comparison(stats_2024: pd.DataFrame,
                           stats_2025: pd.DataFrame,
                           indices_bands: List[int] = [5, 6, 7],
                           indices_names: List[str] = ['NDVI', 'NBR', 'NDMI'],
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[Union[str, Path]] = None):
    """
    Plot vegetation indices comparison between two years

    Args:
        stats_2024: Statistics DataFrame for 2024
        stats_2025: Statistics DataFrame for 2025
        indices_bands: Band numbers for indices (1-indexed)
        indices_names: Names of indices
        figsize: Figure size
        save_path: Path to save figure

    Example:
        >>> from src import get_tiff_stats, plot_indices_comparison
        >>> stats_2024 = get_tiff_stats('S2_2024.tif')
        >>> stats_2025 = get_tiff_stats('S2_2025.tif')
        >>> plot_indices_comparison(stats_2024, stats_2025)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Extract indices data
    indices_2024 = stats_2024[stats_2024['band'].isin(indices_bands)][['band', 'mean', 'std']].copy()
    indices_2025 = stats_2025[stats_2025['band'].isin(indices_bands)][['band', 'mean', 'std']].copy()

    x = np.arange(len(indices_names))
    width = 0.35

    # Plot bars
    ax.bar(x - width/2, indices_2024['mean'].values, width,
           label='2024', alpha=0.8, color='steelblue',
           yerr=indices_2024['std'].values, capsize=5)
    ax.bar(x + width/2, indices_2025['mean'].values, width,
           label='2025', alpha=0.8, color='coral',
           yerr=indices_2025['std'].values, capsize=5)

    # Styling
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Vegetation Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Value', fontsize=14, fontweight='bold')
    ax.set_title('Vegetation Indices: 2024 vs 2025', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(indices_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()

    # Print percentage change
    print("\n📊 VEGETATION INDICES CHANGE (2024 → 2025):")
    for idx, name in enumerate(indices_names):
        val_2024 = indices_2024['mean'].values[idx]
        val_2025 = indices_2025['mean'].values[idx]
        change_pct = ((val_2025 - val_2024) / abs(val_2024)) * 100 if val_2024 != 0 else 0
        print(f"  {name}: {val_2024:.3f} → {val_2025:.3f} ({change_pct:+.1f}%)")


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str] = ['No Deforestation', 'Deforestation'],
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[Union[str, Path]] = None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save figure

    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(y_true, y_pred)
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()


def plot_training_history(history: dict,
                         metrics: List[str] = ['loss', 'accuracy'],
                         figsize: Tuple[int, int] = (15, 5),
                         save_path: Optional[Union[str, Path]] = None):
    """
    Plot training history curves

    Args:
        history: Dictionary with training history
            Keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save figure

    Example:
        >>> history = {
        >>>     'train_loss': [...],
        >>>     'val_loss': [...],
        >>>     'train_acc': [...],
        >>>     'val_acc': [...]
        >>> }
        >>> plot_training_history(history, metrics=['loss', 'accuracy'])
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot train and val curves
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            ax.plot(history[train_key], label=f'Train {metric}', linewidth=2)
        if val_key in history:
            ax.plot(history[val_key], label=f'Val {metric}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()


def plot_patch_sample(patch: np.ndarray,
                     label: int,
                     band_indices: Optional[List[int]] = None,
                     band_names: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (16, 8),
                     save_path: Optional[Union[str, Path]] = None):
    """
    Plot sample patch showing multiple bands

    Args:
        patch: Patch array (H, W, C)
        label: Ground truth label
        band_indices: Indices of bands to plot (0-indexed)
        band_names: Names of bands
        figsize: Figure size
        save_path: Path to save figure

    Example:
        >>> patch = np.load('train_0001_label1.npy')
        >>> plot_patch_sample(patch, label=1,
        >>>                  band_indices=[0, 1, 6, 7, 8],
        >>>                  band_names=['S1_VH', 'S1_R', 'NDVI', 'NBR', 'NDMI'])
    """
    if band_indices is None:
        band_indices = [0, 1, 6, 7, 8, 15, 16, 17]  # Default: key bands

    if band_names is None:
        band_names = [f'Band {i}' for i in band_indices]

    n_bands = len(band_indices)
    ncols = 4
    nrows = (n_bands + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    label_name = 'No Deforestation' if label == 0 else 'Deforestation'
    fig.suptitle(f'Patch Sample - Label: {label} ({label_name})',
                 fontsize=16, fontweight='bold')

    for idx, (band_idx, band_name) in enumerate(zip(band_indices, band_names)):
        ax = axes[idx]

        band_data = patch[:, :, band_idx]

        # Plot
        im = ax.imshow(band_data, cmap='RdYlGn',
                      vmin=np.nanpercentile(band_data, 2),
                      vmax=np.nanpercentile(band_data, 98))
        ax.set_title(band_name, fontsize=11, fontweight='bold')
        ax.axis('off')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Testing visualization module...")

    # Create dummy data for testing
    dummy_data = np.random.randn(100, 100)

    print("\n✅ Testing plot_band...")
    plot_band(dummy_data, title="Test Band", cmap='viridis')

    print("\n✅ Visualization module tests completed")
