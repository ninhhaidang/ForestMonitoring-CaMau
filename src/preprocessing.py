"""
Data preprocessing and patch extraction
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
import pickle

from .config import *
from .utils import read_geotiff, coords_to_pixel, normalize_image, validate_sentinel2_ranges, mask_raster_with_boundary


def load_ground_truth():
    """
    Load ground truth CSV file

    Returns:
        DataFrame with columns: id, label, x, y
    """
    df = pd.read_csv(GROUND_TRUTH_CSV)
    print(f"Loaded {len(df)} ground truth points")
    print(f"  - No deforestation (0): {(df['label'] == 0).sum()}")
    print(f"  - Deforestation (1): {(df['label'] == 1).sum()}")
    return df


def extract_patch_at_point(s1_2024, s1_2025, s2_2024, s2_2025,
                           x, y, transform, patch_size=64):
    """
    Extract patch centered at (x, y) from all imagery

    Args:
        s1_2024, s1_2025: Sentinel-1 arrays (bands, H, W)
        s2_2024, s2_2025: Sentinel-2 arrays (bands, H, W)
        x, y: Geographic coordinates (lon, lat)
        transform: Rasterio transform
        patch_size: Size of patch to extract

    Returns:
        Combined patch of shape (18, patch_size, patch_size) or None if invalid
        Channel order: [S2_2024 (7), S1_2024 (2), S2_2025 (7), S1_2025 (2)]
    """
    # Convert coordinates to pixel
    col, row = coords_to_pixel(x, y, transform)

    # Calculate patch boundaries
    half_size = patch_size // 2
    row_start = row - half_size
    col_start = col - half_size
    row_end = row_start + patch_size
    col_end = col_start + patch_size

    # Check boundaries
    if (row_start < 0 or col_start < 0 or
        row_end > s2_2024.shape[1] or col_end > s2_2024.shape[2]):
        return None

    try:
        # Extract patches
        s1_2024_patch = s1_2024[:, row_start:row_end, col_start:col_end]
        s1_2025_patch = s1_2025[:, row_start:row_end, col_start:col_end]
        s2_2024_patch = s2_2024[:, row_start:row_end, col_start:col_end]
        s2_2025_patch = s2_2025[:, row_start:row_end, col_start:col_end]

        # Check for valid data (no NaN or all zeros)
        if (np.any(np.isnan(s1_2024_patch)) or np.any(np.isnan(s1_2025_patch)) or
            np.any(np.isnan(s2_2024_patch)) or np.any(np.isnan(s2_2025_patch)) or
            np.all(s1_2024_patch == 0) or np.all(s1_2025_patch == 0) or
            np.all(s2_2024_patch == 0) or np.all(s2_2025_patch == 0)):
            return None

        # Combine: [S2_2024 (7 bands), S1_2024 (2 bands), S2_2025 (7 bands), S1_2025 (2 bands)]
        combined_patch = np.concatenate([
            s2_2024_patch,  # 7 bands
            s1_2024_patch,  # 2 bands (VV, VH)
            s2_2025_patch,  # 7 bands
            s1_2025_patch   # 2 bands (VV, VH)
        ], axis=0)  # Total: 18 bands

        return combined_patch

    except Exception as e:
        print(f"Error extracting patch at ({x}, {y}): {e}")
        return None


def create_patches_dataset(patch_size=64, output_dir=None, normalize=True):
    """
    Create patches dataset from all ground truth points

    Args:
        patch_size: Size of patches to extract
        output_dir: Directory to save patches
        normalize: Whether to normalize patches

    Returns:
        patches: List of patches (N, 16, H, W)
        labels: List of labels (N,)
    """
    if output_dir is None:
        output_dir = PATCHES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    df = load_ground_truth()

    # Load all imagery
    print("Loading Sentinel-2 imagery...")
    s2_2024, _, transform = read_geotiff(SENTINEL2_2024)
    s2_2025, _, _ = read_geotiff(SENTINEL2_2025)

    print("Loading Sentinel-1 imagery...")
    s1_2024, _, _ = read_geotiff(SENTINEL1_2024)
    s1_2025, _, _ = read_geotiff(SENTINEL1_2025)

    # Use both VV and VH bands from Sentinel-1
    s1_2024 = s1_2024[0:2, :, :]  # First 2 bands (VV, VH)
    s1_2025 = s1_2025[0:2, :, :]  # First 2 bands (VV, VH)

    print(f"S2_2024 shape: {s2_2024.shape}")
    print(f"S2_2025 shape: {s2_2025.shape}")
    print(f"S1_2024 shape: {s1_2024.shape}")
    print(f"S1_2025 shape: {s1_2025.shape}")

    # Apply forest boundary mask
    print("\nApplying forest boundary mask...")
    s2_2024, s2_mask = mask_raster_with_boundary(s2_2024, transform, FOREST_BOUNDARY)
    s2_2025, _ = mask_raster_with_boundary(s2_2025, transform, FOREST_BOUNDARY)
    s1_2024, _ = mask_raster_with_boundary(s1_2024, transform, FOREST_BOUNDARY)
    s1_2025, _ = mask_raster_with_boundary(s1_2025, transform, FOREST_BOUNDARY)
    print(f"Boundary mask applied - Valid pixels: {np.sum(s2_mask):,} / {s2_mask.size:,} ({np.sum(s2_mask)/s2_mask.size*100:.2f}%)")

    # Normalize imagery if requested
    if normalize:
        print("Validating and normalizing imagery...")
        # Sentinel-2: Validate ranges (no scaling, just clip outliers)
        # B4, B8, B11, B12: [0, 1], NDVI, NBR, NDMI: [-1, 1]
        s2_2024 = validate_sentinel2_ranges(s2_2024)
        s2_2025 = validate_sentinel2_ranges(s2_2025)
        # Sentinel-1: Use minmax normalization (dB values -> [0, 1])
        s1_2024 = normalize_image(s1_2024, method="minmax")
        s1_2025 = normalize_image(s1_2025, method="minmax")

    # Extract patches
    patches = []
    labels = []
    valid_indices = []

    print(f"Extracting {patch_size}x{patch_size} patches...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        x, y = row['x'], row['y']
        label = row['label']

        patch = extract_patch_at_point(
            s1_2024, s1_2025, s2_2024, s2_2025,
            x, y, transform, patch_size
        )

        if patch is not None:
            patches.append(patch)
            labels.append(label)
            valid_indices.append(idx)

    patches = np.array(patches, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"\nExtracted {len(patches)} valid patches out of {len(df)} points")
    print(f"  - No deforestation (0): {(labels == 0).sum()}")
    print(f"  - Deforestation (1): {(labels == 1).sum()}")
    print(f"Patches shape: {patches.shape}")

    # Save patches
    output_file = output_dir / f"patches_{patch_size}x{patch_size}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'patches': patches,
            'labels': labels,
            'valid_indices': valid_indices,
            'patch_size': patch_size
        }, f)
    print(f"Patches saved to {output_file}")

    return patches, labels
