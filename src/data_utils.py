"""
Data Utilities for Ca Mau Forest Change Detection
Author: Ninh Hai Dang (21021411)
Date: 2025-10-16

Functions for preprocessing Sentinel-1/2 data and creating training patches.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. LOADING DATA
# ============================================================================

def load_raster(raster_path: str, bands: Optional[List[int]] = None) -> Tuple[np.ndarray, dict]:
    """
    Load raster file (GeoTIFF) using rasterio.

    Args:
        raster_path: Path to GeoTIFF file
        bands: List of band indices to load (1-indexed). If None, load all bands.

    Returns:
        data: Numpy array of shape (height, width, n_bands)
        meta: Rasterio metadata dict
    """
    with rasterio.open(raster_path) as src:
        if bands is None:
            data = src.read()  # Shape: (n_bands, height, width)
        else:
            data = src.read(bands)  # Shape: (len(bands), height, width)

        # Transpose to (height, width, n_bands)
        data = np.transpose(data, (1, 2, 0))

        meta = src.meta.copy()
        meta.update({'transform': src.transform})

    return data, meta


def load_ground_truth(csv_path: str, shp_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load ground truth points from CSV or Shapefile.

    Args:
        csv_path: Path to CSV file
        shp_path: (Optional) Path to shapefile for spatial operations

    Returns:
        df: DataFrame with standardized columns [id, x, y, label]
    """
    df = pd.read_csv(csv_path)

    # Standardize column names (convert to lowercase)
    df.columns = df.columns.str.lower()

    # Check for coordinate columns
    if 'lon' in df.columns and 'lat' in df.columns:
        # Lat/Lon coordinates (rename for consistency)
        df = df.rename(columns={'lon': 'x', 'lat': 'y'})
    elif 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV must contain either (x, y) or (lon, lat) columns")

    # Check for label column
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column")

    # Check for id column
    if 'id' not in df.columns:
        # Create id if not exists
        df['id'] = range(len(df))

    if shp_path is not None:
        gdf = gpd.read_file(shp_path)
        # Merge with CSV if needed
        if 'geometry' in gdf.columns:
            df = pd.merge(df, gdf[['geometry']], left_index=True, right_index=True)

    return df


# ============================================================================
# 2. SPECTRAL INDICES COMPUTATION
# ============================================================================

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        red: Red band (B4 for Sentinel-2)
        nir: Near-Infrared band (B8 for Sentinel-2)

    Returns:
        ndvi: NDVI array with same shape as input
    """
    epsilon = 1e-8
    ndvi = (nir - red) / (nir + red + epsilon)
    return np.clip(ndvi, -1, 1)


def compute_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Burn Ratio (NBR).
    NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Args:
        nir: Near-Infrared band (B8 for Sentinel-2)
        swir2: Short-Wave Infrared 2 band (B12 for Sentinel-2)

    Returns:
        nbr: NBR array
    """
    epsilon = 1e-8
    nbr = (nir - swir2) / (nir + swir2 + epsilon)
    return np.clip(nbr, -1, 1)


def compute_ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Moisture Index (NDMI).
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Args:
        nir: Near-Infrared band (B8 for Sentinel-2)
        swir1: Short-Wave Infrared 1 band (B11 for Sentinel-2)

    Returns:
        ndmi: NDMI array
    """
    epsilon = 1e-8
    ndmi = (nir - swir1) / (nir + swir1 + epsilon)
    return np.clip(ndmi, -1, 1)


def compute_all_indices(s2_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute all spectral indices from Sentinel-2 data.

    Args:
        s2_data: Sentinel-2 array of shape (H, W, 4) with bands [B4, B8, B11, B12]

    Returns:
        indices: Dict with keys ['ndvi', 'nbr', 'ndmi']
    """
    b4_red = s2_data[:, :, 0]
    b8_nir = s2_data[:, :, 1]
    b11_swir1 = s2_data[:, :, 2]
    b12_swir2 = s2_data[:, :, 3]

    indices = {
        'ndvi': compute_ndvi(b4_red, b8_nir),
        'nbr': compute_nbr(b8_nir, b12_swir2),
        'ndmi': compute_ndmi(b8_nir, b11_swir1)
    }

    return indices


# ============================================================================
# 3. NORMALIZATION
# ============================================================================

def normalize_to_01(data: np.ndarray,
                    min_val: Optional[float] = None,
                    max_val: Optional[float] = None,
                    clip_percentile: bool = True) -> np.ndarray:
    """
    Normalize array to [0, 1] range, handling NaN values.

    Args:
        data: Input array
        min_val: Minimum value for normalization (if None, use data min)
        max_val: Maximum value for normalization (if None, use data max)
        clip_percentile: If True, use 2nd and 98th percentile to avoid outliers

    Returns:
        normalized: Array normalized to [0, 1], NaN values preserved
    """
    # Create mask for valid (non-NaN) values
    valid_mask = ~np.isnan(data)

    if not valid_mask.any():
        # All values are NaN
        return data

    valid_data = data[valid_mask]

    if clip_percentile:
        min_val = np.percentile(valid_data, 2) if min_val is None else min_val
        max_val = np.percentile(valid_data, 98) if max_val is None else max_val
    else:
        min_val = np.min(valid_data) if min_val is None else min_val
        max_val = np.max(valid_data) if max_val is None else max_val

    normalized = (data - min_val) / (max_val - min_val + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    # Preserve NaN values
    normalized[~valid_mask] = np.nan

    return normalized


def preprocess_sentinel2(s2_data: np.ndarray,
                         normalize: bool = True,
                         has_indices: bool = True) -> Dict[str, np.ndarray]:
    """
    Preprocess Sentinel-2 data: normalize bands and handle indices.

    Args:
        s2_data: Sentinel-2 array of shape (H, W, 7) with bands [B4, B8, B11, B12, NDVI, NBR, NDMI]
                 OR shape (H, W, 4) with bands [B4, B8, B11, B12] only
        normalize: Whether to normalize bands to [0, 1]
        has_indices: True if s2_data already contains NDVI, NBR, NDMI (bands 5-7)

    Returns:
        processed: Dict with keys ['bands', 'indices', 'all_channels']
                   - 'bands': Normalized S2 bands (H, W, 4)
                   - 'indices': Dict with NDVI, NBR, NDMI (H, W) each
                   - 'all_channels': Stacked array (H, W, 7) = [B4, B8, B11, B12, NDVI, NBR, NDMI]
    """
    if has_indices and s2_data.shape[2] >= 7:
        # Data already has indices (bands 5-7)
        bands_data = s2_data[:, :, :4]  # First 4 bands
        indices_data = s2_data[:, :, 4:7]  # Bands 5-7: NDVI, NBR, NDMI

        # Normalize bands (0-4)
        if normalize:
            bands_normalized = np.stack([
                normalize_to_01(bands_data[:, :, i]) for i in range(4)
            ], axis=-1)
        else:
            bands_normalized = bands_data

        # Normalize indices from [-1, 1] to [0, 1]
        indices_normalized = {
            'ndvi': (indices_data[:, :, 0] + 1) / 2,
            'nbr': (indices_data[:, :, 1] + 1) / 2,
            'ndmi': (indices_data[:, :, 2] + 1) / 2
        }

        # Stack all channels
        all_channels = np.concatenate([
            bands_normalized,
            indices_normalized['ndvi'][:, :, np.newaxis],
            indices_normalized['nbr'][:, :, np.newaxis],
            indices_normalized['ndmi'][:, :, np.newaxis]
        ], axis=-1)

    else:
        # Need to compute indices from raw bands
        # Normalize bands
        if normalize:
            s2_normalized = np.stack([
                normalize_to_01(s2_data[:, :, i]) for i in range(s2_data.shape[2])
            ], axis=-1)
        else:
            s2_normalized = s2_data

        # Compute indices
        indices = compute_all_indices(s2_data)

        # Normalize indices from [-1, 1] to [0, 1]
        indices_normalized = {
            k: (v + 1) / 2 for k, v in indices.items()
        }

        # Stack all channels: [B4, B8, B11, B12, NDVI, NBR, NDMI]
        all_channels = np.concatenate([
            s2_normalized,
            indices_normalized['ndvi'][:, :, np.newaxis],
            indices_normalized['nbr'][:, :, np.newaxis],
            indices_normalized['ndmi'][:, :, np.newaxis]
        ], axis=-1)

        bands_normalized = s2_normalized

    return {
        'bands': bands_normalized,
        'indices': indices_normalized,
        'all_channels': all_channels
    }


def preprocess_sentinel1(s1_data: np.ndarray,
                         normalize: bool = True) -> np.ndarray:
    """
    Preprocess Sentinel-1 SAR data: normalize to [0, 1].

    Args:
        s1_data: Sentinel-1 array of shape (H, W, 2) with [VH, Ratio]
        normalize: Whether to normalize to [0, 1]

    Returns:
        s1_normalized: Normalized S1 array (H, W, 2)
    """
    if normalize:
        s1_normalized = np.stack([
            normalize_to_01(s1_data[:, :, i]) for i in range(s1_data.shape[2])
        ], axis=-1)
    else:
        s1_normalized = s1_data

    return s1_normalized


# ============================================================================
# 4. PATCH EXTRACTION
# ============================================================================

def xy_to_pixel(x: float, y: float, transform) -> Tuple[int, int]:
    """
    Convert x/y coordinates (projected or geographic) to pixel row/col in raster.

    Args:
        x: X coordinate (longitude or easting)
        y: Y coordinate (latitude or northing)
        transform: Rasterio Affine transform

    Returns:
        row, col: Pixel coordinates (0-indexed)
    """
    col, row = ~transform * (x, y)
    return int(row), int(col)


def extract_patch(data: np.ndarray,
                  center_row: int,
                  center_col: int,
                  patch_size: int = 256,
                  min_valid_ratio: float = 0.5) -> Optional[np.ndarray]:
    """
    Extract a square patch centered at (center_row, center_col).

    Args:
        data: Input array of shape (H, W, C)
        center_row: Center row coordinate
        center_col: Center column coordinate
        patch_size: Size of square patch (default 256)
        min_valid_ratio: Minimum ratio of valid (non-NaN) pixels required (default 0.5)

    Returns:
        patch: Extracted patch of shape (patch_size, patch_size, C), or None if out of bounds or too many NaNs
    """
    half_size = patch_size // 2

    row_start = center_row - half_size
    row_end = center_row + half_size
    col_start = center_col - half_size
    col_end = center_col + half_size

    # Check bounds
    if (row_start < 0 or row_end > data.shape[0] or
        col_start < 0 or col_end > data.shape[1]):
        return None

    patch = data[row_start:row_end, col_start:col_end, :]

    # Verify patch size
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        return None

    # Check for too many NaN values
    # Check first channel as representative
    valid_pixels = ~np.isnan(patch[:, :, 0])
    valid_ratio = valid_pixels.sum() / (patch_size * patch_size)

    if valid_ratio < min_valid_ratio:
        # Too many NaN values, skip this patch
        return None

    # Fill remaining NaN values with 0 (or use other strategy)
    patch = np.nan_to_num(patch, nan=0.0)

    return patch


def create_binary_mask(label: int, patch_size: int = 256) -> np.ndarray:
    """
    Create a binary mask for change detection.

    Args:
        label: Ground truth label (0 = no change, 1 = change)
        patch_size: Size of mask

    Returns:
        mask: Binary mask of shape (patch_size, patch_size) with values {0, 255}
    """
    mask = np.full((patch_size, patch_size), fill_value=255 if label == 1 else 0, dtype=np.uint8)
    return mask


# ============================================================================
# 5. DATASET CREATION
# ============================================================================

def create_training_dataset(
    s2_t1_path: str,
    s2_t2_path: str,
    s1_t1_path: str,
    s1_t2_path: str,
    ground_truth_csv: str,
    output_dir: str,
    patch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Create complete training dataset from raw data.

    Args:
        s2_t1_path: Path to Sentinel-2 Time 1 GeoTIFF
        s2_t2_path: Path to Sentinel-2 Time 2 GeoTIFF
        s1_t1_path: Path to Sentinel-1 Time 1 GeoTIFF
        s1_t2_path: Path to Sentinel-1 Time 2 GeoTIFF
        ground_truth_csv: Path to ground truth CSV
        output_dir: Output directory for processed data
        patch_size: Size of extracted patches
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        random_seed: Random seed for reproducibility

    Returns:
        stats: Dict with number of samples in each split
    """
    np.random.seed(random_seed)

    print("=" * 80)
    print("CREATING TRAINING DATASET")
    print("=" * 80)

    # 1. Load data
    print("\n[1/6] Loading Sentinel-2 data...")
    s2_t1, meta_t1 = load_raster(s2_t1_path)
    s2_t2, meta_t2 = load_raster(s2_t2_path)
    print(f"  - S2 T1 shape: {s2_t1.shape}")
    print(f"  - S2 T2 shape: {s2_t2.shape}")

    print("\n[2/6] Loading Sentinel-1 data...")
    s1_t1, _ = load_raster(s1_t1_path)
    s1_t2, _ = load_raster(s1_t2_path)
    print(f"  - S1 T1 shape: {s1_t1.shape}")
    print(f"  - S1 T2 shape: {s1_t2.shape}")

    print("\n[3/6] Loading ground truth...")
    df = load_ground_truth(ground_truth_csv)
    print(f"  - Total points: {len(df)}")
    print(f"  - Class distribution:\n{df['label'].value_counts()}")

    # 2. Preprocess
    print("\n[4/6] Preprocessing Sentinel-2 (computing indices)...")
    s2_t1_processed = preprocess_sentinel2(s2_t1, normalize=True)
    s2_t2_processed = preprocess_sentinel2(s2_t2, normalize=True)
    print(f"  - S2 T1 channels: {s2_t1_processed['all_channels'].shape[-1]}")
    print(f"  - S2 T2 channels: {s2_t2_processed['all_channels'].shape[-1]}")

    print("\n[5/6] Preprocessing Sentinel-1...")
    s1_t1_processed = preprocess_sentinel1(s1_t1, normalize=True)
    s1_t2_processed = preprocess_sentinel1(s1_t2, normalize=True)

    # Stack S2 + S1 to get 9 channels per time step
    t1_all = np.concatenate([s2_t1_processed['all_channels'], s1_t1_processed], axis=-1)
    t2_all = np.concatenate([s2_t2_processed['all_channels'], s1_t2_processed], axis=-1)
    print(f"  - T1 total channels: {t1_all.shape[-1]}")
    print(f"  - T2 total channels: {t2_all.shape[-1]}")

    # 3. Extract patches
    print("\n[6/6] Extracting patches at ground truth locations...")
    transform = meta_t1['transform']

    valid_samples = []
    for idx, row in df.iterrows():
        x, y = row['x'], row['y']
        label = row['label']

        # Convert to pixel coordinates
        pixel_row, pixel_col = xy_to_pixel(x, y, transform)

        # Extract patches
        patch_t1 = extract_patch(t1_all, pixel_row, pixel_col, patch_size)
        patch_t2 = extract_patch(t2_all, pixel_row, pixel_col, patch_size)

        if patch_t1 is not None and patch_t2 is not None:
            mask = create_binary_mask(label, patch_size)
            valid_samples.append({
                'id': idx,
                'patch_t1': patch_t1,
                'patch_t2': patch_t2,
                'mask': mask,
                'label': label
            })

    print(f"  - Valid samples: {len(valid_samples)} / {len(df)}")

    # 4. Split train/val/test
    print("\n[7/7] Splitting into train/val/test...")
    n_samples = len(valid_samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    # Shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    print(f"  - Train: {len(train_indices)}")
    print(f"  - Val: {len(val_indices)}")
    print(f"  - Test: {len(test_indices)}")

    # 5. Save patches
    print("\n[8/8] Saving patches to disk...")
    output_path = Path(output_dir)

    for split_name, split_indices in splits.items():
        print(f"\n  Saving {split_name} split...")

        split_path = output_path / split_name
        (split_path / 'A').mkdir(parents=True, exist_ok=True)
        (split_path / 'B').mkdir(parents=True, exist_ok=True)
        (split_path / 'label').mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(split_indices, start=1):
            sample = valid_samples[idx]

            # Zero-padded filename
            filename = f"{i:04d}"

            # Save as GeoTIFF (float32 for images, uint8 for masks)
            with rasterio.open(
                split_path / 'A' / f"{filename}.tif",
                'w',
                driver='GTiff',
                height=patch_size,
                width=patch_size,
                count=sample['patch_t1'].shape[-1],
                dtype='float32',
                compress='lzw'
            ) as dst:
                for band_idx in range(sample['patch_t1'].shape[-1]):
                    dst.write(sample['patch_t1'][:, :, band_idx], band_idx + 1)

            with rasterio.open(
                split_path / 'B' / f"{filename}.tif",
                'w',
                driver='GTiff',
                height=patch_size,
                width=patch_size,
                count=sample['patch_t2'].shape[-1],
                dtype='float32',
                compress='lzw'
            ) as dst:
                for band_idx in range(sample['patch_t2'].shape[-1]):
                    dst.write(sample['patch_t2'][:, :, band_idx], band_idx + 1)

            # Save mask as PNG
            from PIL import Image
            mask_img = Image.fromarray(sample['mask'])
            mask_img.save(split_path / 'label' / f"{filename}.png")

        print(f"     Saved {len(split_indices)} samples to {split_path}")

    print("\n" + "=" * 80)
    print("DATASET CREATION COMPLETED!")
    print("=" * 80)

    return {
        'train': len(train_indices),
        'val': len(val_indices),
        'test': len(test_indices),
        'total': n_samples
    }


# ============================================================================
# 6. VERIFICATION
# ============================================================================

def verify_dataset(data_root: str) -> Dict[str, any]:
    """
    Verify created dataset structure and integrity.

    Args:
        data_root: Root directory of processed data

    Returns:
        report: Dict with verification results
    """
    data_path = Path(data_root)
    report = {}

    for split in ['train', 'val', 'test']:
        split_path = data_path / split

        if not split_path.exists():
            report[split] = {'exists': False}
            continue

        # Count files
        n_a = len(list((split_path / 'A').glob('*.tif')))
        n_b = len(list((split_path / 'B').glob('*.tif')))
        n_label = len(list((split_path / 'label').glob('*.png')))

        # Check consistency
        consistent = (n_a == n_b == n_label)

        report[split] = {
            'exists': True,
            'n_samples': n_a,
            'consistent': consistent,
            'details': f"A:{n_a}, B:{n_b}, label:{n_label}"
        }

    return report
