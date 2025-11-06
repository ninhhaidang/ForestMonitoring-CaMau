"""
Utility functions for data processing and visualization
"""
import numpy as np
import torch
import random
from pathlib import Path
import rasterio
from rasterio.windows import Window


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_geotiff(file_path, window=None):
    """
    Read GeoTIFF file

    Args:
        file_path: Path to GeoTIFF file
        window: Optional rasterio Window to read subset

    Returns:
        array: Numpy array of shape (bands, height, width)
        profile: Rasterio profile metadata
    """
    with rasterio.open(file_path) as src:
        if window:
            data = src.read(window=window)
        else:
            data = src.read()
        profile = src.profile
        transform = src.transform

    return data, profile, transform


def normalize_image(image, method="minmax"):
    """
    Normalize image

    Args:
        image: Numpy array
        method: "minmax" or "zscore"

    Returns:
        Normalized image
    """
    if method == "minmax":
        img_min = np.nanmin(image, axis=(1, 2), keepdims=True)
        img_max = np.nanmax(image, axis=(1, 2), keepdims=True)
        normalized = (image - img_min) / (img_max - img_min + 1e-8)
    elif method == "zscore":
        img_mean = np.nanmean(image, axis=(1, 2), keepdims=True)
        img_std = np.nanstd(image, axis=(1, 2), keepdims=True)
        normalized = (image - img_mean) / (img_std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Replace NaN with 0
    normalized = np.nan_to_num(normalized, nan=0.0)

    return normalized


def normalize_sentinel2(s2_image):
    """
    Normalize Sentinel-2 image with proper handling for different band types

    Args:
        s2_image: Numpy array of shape (7, H, W) with bands in order:
                  [B4, B8, B11, B12, NDVI, NBR, NDMI]

    Returns:
        Normalized Sentinel-2 image:
        - B4, B8, B11, B12 (bands 0-3): clipped to [0, 1]
        - NDVI, NBR, NDMI (bands 4-6): clipped to [-1, 1]
    """
    normalized = s2_image.copy()

    # Spectral bands (0-3): clip to [0, 1] (already in this range)
    normalized[0:4] = np.clip(normalized[0:4], 0, 1)

    # Spectral indices (4-6): clip to [-1, 1] (already in this range)
    normalized[4:7] = np.clip(normalized[4:7], -1, 1)

    # Replace NaN with 0
    normalized = np.nan_to_num(normalized, nan=0.0)

    return normalized


def pixel_to_coords(x, y, transform):
    """Convert pixel coordinates to geographic coordinates"""
    lon, lat = rasterio.transform.xy(transform, y, x)
    return lon, lat


def coords_to_pixel(lon, lat, transform):
    """Convert geographic coordinates to pixel coordinates"""
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    return col, row


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary'),
        "recall": recall_score(y_true, y_pred, average='binary'),
        "f1": f1_score(y_true, y_pred, average='binary'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    return metrics


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss
