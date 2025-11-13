"""
Temperature calibration and post-processing utilities for CNN predictions
"""

import numpy as np
import torch
from scipy.ndimage import generic_filter, median_filter
from sklearn.metrics import log_loss
import logging

logger = logging.getLogger(__name__)


def find_optimal_temperature(
    model: torch.nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = 'cuda',
    temperature_range: tuple = (1.0, 5.0),
    num_temps: int = 20
) -> float:
    """
    Find optimal temperature using validation set

    Args:
        model: Trained CNN model
        X_val: Validation patches (n_samples, n_features, H, W)
        y_val: Validation labels (n_samples,)
        device: 'cuda' or 'cpu'
        temperature_range: (min_temp, max_temp) to search
        num_temps: Number of temperatures to try

    Returns:
        Optimal temperature value
    """
    logger.info("\n" + "="*70)
    logger.info("TEMPERATURE CALIBRATION")
    logger.info("="*70)

    model.eval()
    device = torch.device(device)

    # Get model outputs on validation set
    X_val_tensor = torch.FloatTensor(X_val).to(device)

    with torch.no_grad():
        logits = model(X_val_tensor).cpu().numpy()

    # Try different temperatures
    temperatures = np.linspace(temperature_range[0], temperature_range[1], num_temps)
    losses = []

    for T in temperatures:
        # Apply temperature scaling
        scaled_logits = logits / T

        # Compute softmax probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Compute log loss (calibration metric)
        loss = log_loss(y_val, probs)
        losses.append(loss)

        logger.info(f"  T = {T:.2f} → Log Loss: {loss:.4f}")

    # Find best temperature
    best_idx = np.argmin(losses)
    best_temp = temperatures[best_idx]
    best_loss = losses[best_idx]

    logger.info(f"\n✓ Optimal Temperature: {best_temp:.2f} (Log Loss: {best_loss:.4f})")
    logger.info("="*70 + "\n")

    return best_temp


def majority_filter_2d(
    classification_map: np.ndarray,
    kernel_size: int = 3,
    nodata_value: int = 255
) -> np.ndarray:
    """
    Apply spatial majority filter to reduce salt-and-pepper noise

    Args:
        classification_map: Binary classification map (H, W)
        kernel_size: Size of neighborhood window (3, 5, 7, etc.)
        nodata_value: NoData value to ignore

    Returns:
        Filtered classification map
    """
    logger.info(f"Applying majority filter (kernel={kernel_size}x{kernel_size})...")

    def majority(values):
        # Ignore NoData
        valid_values = values[values != nodata_value]
        if len(valid_values) == 0:
            return nodata_value

        # Find majority class
        counts = np.bincount(valid_values.astype(int))
        return np.argmax(counts)

    filtered = generic_filter(
        classification_map,
        majority,
        size=kernel_size,
        mode='constant',
        cval=nodata_value
    )

    # Calculate changed pixels
    changed = np.sum((classification_map != filtered) & (classification_map != nodata_value))
    total_valid = np.sum(classification_map != nodata_value)

    logger.info(f"  Pixels changed: {changed:,} / {total_valid:,} ({changed/total_valid*100:.2f}%)")

    return filtered.astype(np.uint8)


def median_filter_2d(
    probability_map: np.ndarray,
    kernel_size: int = 3,
    nodata_value: float = -9999
) -> np.ndarray:
    """
    Apply spatial median filter to smooth probability map

    Args:
        probability_map: Probability map (H, W)
        kernel_size: Size of neighborhood window
        nodata_value: NoData value to ignore

    Returns:
        Smoothed probability map
    """
    logger.info(f"Applying median filter (kernel={kernel_size}x{kernel_size})...")

    # Create mask for valid data
    valid_mask = probability_map != nodata_value

    # Apply median filter only on valid data
    filtered = np.copy(probability_map)
    filtered[valid_mask] = median_filter(
        probability_map * valid_mask,
        size=kernel_size,
        mode='constant',
        cval=0
    )[valid_mask]

    # Restore NoData
    filtered[~valid_mask] = nodata_value

    logger.info(f"  Probability map smoothed")

    return filtered


def apply_probability_threshold(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    nodata_value: float = -9999
) -> np.ndarray:
    """
    Apply custom threshold to probability map

    Args:
        probability_map: Probability map (H, W)
        threshold: Classification threshold (default 0.5)
        nodata_value: NoData value to ignore

    Returns:
        Binary classification map
    """
    logger.info(f"Applying probability threshold: {threshold:.2f}")

    classification = np.zeros_like(probability_map, dtype=np.uint8)
    valid_mask = probability_map != nodata_value

    classification[valid_mask] = (probability_map[valid_mask] >= threshold).astype(np.uint8)
    classification[~valid_mask] = 255  # NoData

    # Statistics
    deforestation_pixels = np.sum(classification == 1)
    total_valid = np.sum(valid_mask)

    logger.info(f"  Deforestation: {deforestation_pixels:,} / {total_valid:,} ({deforestation_pixels/total_valid*100:.2f}%)")

    return classification


def calibration_summary(
    probability_map: np.ndarray,
    nodata_value: float = -9999
):
    """
    Print probability distribution summary

    Args:
        probability_map: Probability map (H, W)
        nodata_value: NoData value to ignore
    """
    valid_probs = probability_map[probability_map != nodata_value]

    logger.info("\nProbability Distribution:")
    logger.info(f"  Min:    {valid_probs.min():.4f}")
    logger.info(f"  Max:    {valid_probs.max():.4f}")
    logger.info(f"  Mean:   {valid_probs.mean():.4f}")
    logger.info(f"  Median: {np.median(valid_probs):.4f}")
    logger.info(f"  Std:    {valid_probs.std():.4f}")

    # Distribution bins
    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist, _ = np.histogram(valid_probs, bins=bins)

    logger.info("\n  Probability bins:")
    for i in range(len(bins)-1):
        pct = hist[i] / len(valid_probs) * 100
        logger.info(f"    {bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]:,} pixels ({pct:.2f}%)")
