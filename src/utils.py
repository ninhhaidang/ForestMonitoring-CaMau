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


def read_geotiff(file_path, window=None, clip_sentinel2=True):
    """
    Read GeoTIFF file and handle NoData values

    Args:
        file_path: Path to GeoTIFF file
        window: Optional rasterio Window to read subset
        clip_sentinel2: If True and file is Sentinel-2, clip outliers

    Returns:
        array: Numpy array of shape (bands, height, width)
               NoData values are converted to NaN
        profile: Rasterio profile metadata
        transform: Affine transform
    """
    with rasterio.open(file_path) as src:
        if window:
            data = src.read(window=window, masked=True)
        else:
            data = src.read(masked=True)

        profile = src.profile
        transform = src.transform

        # Convert masked array to regular array with NaN
        if hasattr(data, 'filled'):
            # If data is a masked array, convert masked values to NaN
            data = data.filled(np.nan)

        # Additionally handle NoData values explicitly
        # Some GeoTIFF files may have NoData = 0
        for i in range(data.shape[0]):
            nodata = src.nodatavals[i] if src.nodatavals else None
            if nodata is not None:
                data[i][data[i] == nodata] = np.nan

    # Clip Sentinel-2 outliers if requested
    if clip_sentinel2 and "S2_" in str(file_path):
        # Sentinel-2 file detected - clip band values to physical ranges
        if data.shape[0] == 7:  # 7 S2 bands
            # B4, B8, B11, B12 (bands 0-3): clip to [0, 1]
            for i in range(4):
                data[i] = np.clip(data[i], 0, 1)
            # NDVI, NBR, NDMI (bands 4-6): clip to [-1, 1]
            for i in range(4, 7):
                data[i] = np.clip(data[i], -1, 1)

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


def validate_sentinel2_ranges(s2_image):
    """
    Validate and clip Sentinel-2 band values to expected ranges

    NOTE: This function does NOT normalize/scale values. It only validates
    that values are within the expected physical ranges and clips outliers.

    Args:
        s2_image: Numpy array of shape (7, H, W) with bands in order:
                  [B4, B8, B11, B12, NDVI, NBR, NDMI]

    Returns:
        Validated Sentinel-2 image:
        - B4, B8, B11, B12 (bands 0-3): clipped to [0, 1] (reflectance range)
        - NDVI, NBR, NDMI (bands 4-6): clipped to [-1, 1] (index range)

    Expected input ranges (from GEE processing):
        - B4: [0.0001, ~0.45]
        - B8: [0.0001, ~0.55]
        - B11: [0.0068, ~0.51]
        - B12: [0.0052, ~0.96]
        - NDVI: [-1.0, 1.0]
        - NBR: [-1.0, ~0.93]
        - NDMI: [-1.0, ~0.81]
    """
    validated = s2_image.copy()

    # Spectral reflectance bands (0-3): clip to [0, 1]
    validated[0:4] = np.clip(validated[0:4], 0, 1)

    # Spectral indices (4-6): clip to [-1, 1]
    validated[4:7] = np.clip(validated[4:7], -1, 1)

    # Replace NaN with 0
    validated = np.nan_to_num(validated, nan=0.0)

    return validated


def pixel_to_coords(x, y, transform):
    """Convert pixel coordinates to geographic coordinates"""
    lon, lat = rasterio.transform.xy(transform, y, x)
    return lon, lat


def coords_to_pixel(lon, lat, transform):
    """Convert geographic coordinates to pixel coordinates"""
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    return col, row


def mask_raster_with_boundary(raster, transform, boundary_shapefile):
    """
    Mask raster data with boundary shapefile

    Args:
        raster: Numpy array of shape (bands, height, width)
        transform: Rasterio affine transform
        boundary_shapefile: Path to boundary shapefile

    Returns:
        masked_raster: Masked raster with values outside boundary set to NaN
        mask: Boolean mask (True for valid pixels, False for masked)
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask

    # Read boundary shapefile
    boundary_gdf = gpd.read_file(boundary_shapefile)

    # Get geometries
    geometries = boundary_gdf.geometry.values

    # Create mask (False = valid data, True = masked)
    mask_array = geometry_mask(
        geometries,
        out_shape=(raster.shape[1], raster.shape[2]),
        transform=transform,
        invert=False  # False = areas outside polygon are masked
    )

    # Invert to get (True = valid, False = masked)
    valid_mask = ~mask_array

    # Apply mask to raster
    masked_raster = raster.copy()
    for i in range(raster.shape[0]):
        masked_raster[i][~valid_mask] = np.nan

    return masked_raster, valid_mask


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
