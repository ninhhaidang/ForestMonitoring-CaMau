"""
Data preprocessing and patch extraction functions
"""
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
from scipy import interpolate
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def normalize_band(data: np.ndarray,
                   method: str = 'standardize',
                   clip_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Normalize band data

    Args:
        data: Input array
        method: Normalization method
            - 'standardize': (x - mean) / std
            - 'minmax': (x - min) / (max - min)
            - 'clip': Clip to range then scale to [0, 1]
        clip_range: Range to clip data (min, max) for 'clip' method

    Returns:
        Normalized array

    Example:
        >>> # Standardize
        >>> norm_data = normalize_band(data, method='standardize')

        >>> # Min-max to [0, 1]
        >>> norm_data = normalize_band(data, method='minmax')

        >>> # Clip and scale
        >>> norm_data = normalize_band(data, method='clip', clip_range=(0, 1))
    """
    # Remove NaN for calculation
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]

    if len(valid_data) == 0:
        return data  # All NaN, return as is

    result = data.copy()

    if method == 'standardize':
        mean = valid_data.mean()
        std = valid_data.std()
        if std > 0:
            result[valid_mask] = (data[valid_mask] - mean) / std

    elif method == 'minmax':
        vmin = valid_data.min()
        vmax = valid_data.max()
        if vmax > vmin:
            result[valid_mask] = (data[valid_mask] - vmin) / (vmax - vmin)

    elif method == 'clip':
        if clip_range is None:
            raise ValueError("clip_range must be provided for 'clip' method")
        vmin, vmax = clip_range
        result = np.clip(data, vmin, vmax)
        result[valid_mask] = (result[valid_mask] - vmin) / (vmax - vmin)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return result


def handle_nan(data: np.ndarray,
               method: str = 'interpolate',
               fill_value: float = 0.0) -> np.ndarray:
    """
    Handle NaN values in array

    Args:
        data: Input array (2D or 3D)
        method: Method to handle NaN
            - 'interpolate': Interpolate from neighbors (2D only)
            - 'fill': Fill with constant value
            - 'mean': Fill with mean of valid pixels
            - 'median': Fill with median of valid pixels
        fill_value: Value to fill (for 'fill' method)

    Returns:
        Array with NaN handled

    Example:
        >>> # Interpolate NaN values
        >>> clean_data = handle_nan(data, method='interpolate')

        >>> # Fill with zeros
        >>> clean_data = handle_nan(data, method='fill', fill_value=0)

        >>> # Fill with mean
        >>> clean_data = handle_nan(data, method='mean')
    """
    result = data.copy()

    if method == 'fill':
        result[np.isnan(data)] = fill_value

    elif method == 'mean':
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            result[np.isnan(data)] = valid_data.mean()

    elif method == 'median':
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            result[np.isnan(data)] = np.median(valid_data)

    elif method == 'interpolate':
        if data.ndim != 2:
            raise ValueError("Interpolation only supports 2D arrays")

        # Get valid pixel indices
        valid_mask = ~np.isnan(data)
        if valid_mask.sum() == 0:
            return result  # All NaN

        y_valid, x_valid = np.where(valid_mask)
        values_valid = data[valid_mask]

        # Get NaN pixel indices
        y_nan, x_nan = np.where(~valid_mask)

        if len(y_nan) > 0:
            # Interpolate
            interpolated = interpolate.griddata(
                (y_valid, x_valid),
                values_valid,
                (y_nan, x_nan),
                method='nearest'
            )
            result[y_nan, x_nan] = interpolated

    else:
        raise ValueError(f"Unknown NaN handling method: {method}")

    return result


def extract_patch(data: np.ndarray,
                 center_y: int,
                 center_x: int,
                 patch_size: int) -> Optional[np.ndarray]:
    """
    Extract a patch from array centered at given coordinates

    Args:
        data: Input array (H, W) or (H, W, C)
        center_y: Center pixel Y coordinate
        center_x: Center pixel X coordinate
        patch_size: Size of square patch

    Returns:
        Patch array of shape (patch_size, patch_size) or (patch_size, patch_size, C)
        Returns None if patch is out of bounds

    Example:
        >>> patch = extract_patch(data, center_y=1000, center_x=2000, patch_size=128)
        >>> print(patch.shape)  # (128, 128, C)
    """
    half = patch_size // 2

    # Calculate bounds
    y_start = center_y - half
    y_end = center_y + half
    x_start = center_x - half
    x_end = center_x + half

    # Check if patch is within image bounds
    if data.ndim == 2:
        h, w = data.shape
    else:
        h, w, _ = data.shape

    if y_start < 0 or y_end > h or x_start < 0 or x_end > w:
        return None

    # Extract patch
    if data.ndim == 2:
        patch = data[y_start:y_end, x_start:x_end]
    else:
        patch = data[y_start:y_end, x_start:x_end, :]

    # Verify patch size
    if data.ndim == 2:
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            return None
    else:
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            return None

    return patch


def create_patches_dataset(
    s1_2024_path: Union[str, Path],
    s1_2025_path: Union[str, Path],
    s2_2024_path: Union[str, Path],
    s2_2025_path: Union[str, Path],
    ground_truth_csv: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int = 128,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    normalize: bool = True,
    handle_nan_method: str = 'fill',
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Create patches dataset from TIFF files and ground truth

    This is the main function to create the full dataset

    Args:
        s1_2024_path: Path to Sentinel-1 2024 file
        s1_2025_path: Path to Sentinel-1 2025 file
        s2_2024_path: Path to Sentinel-2 2024 file
        s2_2025_path: Path to Sentinel-2 2025 file
        ground_truth_csv: Path to CSV with ground truth labels
        output_dir: Directory to save patches
        patch_size: Size of square patches (default: 128)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        normalize: Apply normalization to bands
        handle_nan_method: Method to handle NaN values
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with counts: {'train': n_train, 'val': n_val, 'test': n_test}

    Example:
        >>> counts = create_patches_dataset(
        >>>     s1_2024_path='data/raw/sentinel1/S1_2024.tif',
        >>>     s1_2025_path='data/raw/sentinel1/S1_2025.tif',
        >>>     s2_2024_path='data/raw/sentinel2/S2_2024.tif',
        >>>     s2_2025_path='data/raw/sentinel2/S2_2025.tif',
        >>>     ground_truth_csv='data/raw/ground_truth/Training_Points_CSV.csv',
        >>>     output_dir='data/patches',
        >>>     patch_size=128
        >>> )
        >>> print(f"Created {counts['train']} train, {counts['val']} val, {counts['test']} test patches")
    """
    from .utils import load_tiff, load_ground_truth, coord_to_pixel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(exist_ok=True)

    # Load ground truth
    print("Loading ground truth...")
    gt_df = load_ground_truth(ground_truth_csv)
    print(f"Loaded {len(gt_df)} ground truth points")

    # Split into train/val/test
    print(f"Splitting dataset: {train_ratio}/{val_ratio}/{test_ratio}...")
    train_val_df, test_df = train_test_split(
        gt_df, test_size=test_ratio, stratify=gt_df['label'], random_state=random_seed
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio/(train_ratio+val_ratio),
        stratify=train_val_df['label'], random_state=random_seed
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load TIFF files
    print("\nLoading TIFF files...")
    # Skip Sentinel-1 - use only Sentinel-2 data

    print("  - Sentinel-2 2024...")
    with rasterio.open(s2_2024_path) as src:
        transform_s2 = src.transform
        s2_2024 = src.read()  # (7, H, W)

    print("  - Sentinel-2 2025...")
    with rasterio.open(s2_2025_path) as src:
        s2_2025 = src.read()  # (7, H, W)

    # Stack only Sentinel-2 bands: (14, H, W) = 7 S2 (2024) + 7 S2 (2025)
    print("\nStacking 14 bands (Sentinel-2 only)...")
    all_bands = np.concatenate([s2_2024, s2_2025], axis=0)
    print(f"Stacked shape: {all_bands.shape}")

    # Transpose to (H, W, 14)
    all_bands = np.transpose(all_bands, (1, 2, 0))
    print(f"Transposed shape: {all_bands.shape}")

    # Process each split
    counts = {}
    for split, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} set ({len(df)} samples)...")
        print(f"{'='*80}")

        saved_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split.upper()}", unit="patch"):
            geo_x, geo_y = row['x'], row['y']
            label = int(row['label'])

            # Convert to pixel coordinates
            pixel_x, pixel_y = coord_to_pixel(geo_x, geo_y, transform_s2)

            # Extract patch
            patch = extract_patch(all_bands, pixel_y, pixel_x, patch_size)

            if patch is None:
                tqdm.write(f"  ⚠️ Skipping point {row['id']}: out of bounds")
                continue

            # Handle NaN
            for c in range(patch.shape[2]):
                if np.isnan(patch[:, :, c]).any():
                    patch[:, :, c] = handle_nan(
                        patch[:, :, c],
                        method=handle_nan_method
                    )

            # Normalize
            if normalize:
                for c in range(patch.shape[2]):
                    # Different normalization for different band types
                    if c in [0, 1, 9, 10]:  # S1 bands
                        patch[:, :, c] = normalize_band(
                            patch[:, :, c],
                            method='standardize'
                        )
                    elif c in [2, 3, 4, 5, 11, 12, 13, 14]:  # S2 reflectance
                        patch[:, :, c] = normalize_band(
                            patch[:, :, c],
                            method='clip',
                            clip_range=(0, 1)
                        )
                    else:  # S2 indices (6,7,8,15,16,17)
                        # Scale from [-1, 1] to [0, 1]
                        patch[:, :, c] = (patch[:, :, c] + 1) / 2

            # Save patch
            filename = f"{split}_{int(row['id']):04d}_label{label}.npy"
            save_path = output_dir / split / filename
            np.save(save_path, patch.astype(np.float32))

            saved_count += 1

        counts[split] = saved_count
        print(f"✅ {split.upper()}: Saved {saved_count}/{len(df)} patches")

    # Save summary
    summary_path = output_dir / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PATCHES DATASET SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Patch size: {patch_size}×{patch_size}\n")
        f.write(f"Channels: 18\n")
        f.write(f"Normalized: {normalize}\n")
        f.write(f"NaN handling: {handle_nan_method}\n\n")
        f.write(f"Train: {counts['train']}\n")
        f.write(f"Val: {counts['val']}\n")
        f.write(f"Test: {counts['test']}\n")
        f.write(f"Total: {sum(counts.values())}\n")

    print(f"\n✅ Summary saved to: {summary_path}")
    print("\n" + "="*80)
    print("DATASET CREATION COMPLETED")
    print("="*80)

    return counts


def augment_patch(patch: np.ndarray,
                 augmentation: str) -> np.ndarray:
    """
    Apply data augmentation to patch

    Args:
        patch: Input patch (H, W, C)
        augmentation: Type of augmentation
            - 'rotate_90': Rotate 90 degrees
            - 'rotate_180': Rotate 180 degrees
            - 'rotate_270': Rotate 270 degrees
            - 'flip_h': Horizontal flip
            - 'flip_v': Vertical flip
            - 'noise': Add Gaussian noise

    Returns:
        Augmented patch

    Example:
        >>> aug_patch = augment_patch(patch, 'rotate_90')
        >>> aug_patch = augment_patch(patch, 'flip_h')
    """
    result = patch.copy()

    if augmentation == 'rotate_90':
        result = np.rot90(result, k=1, axes=(0, 1))
    elif augmentation == 'rotate_180':
        result = np.rot90(result, k=2, axes=(0, 1))
    elif augmentation == 'rotate_270':
        result = np.rot90(result, k=3, axes=(0, 1))
    elif augmentation == 'flip_h':
        result = np.flip(result, axis=1)
    elif augmentation == 'flip_v':
        result = np.flip(result, axis=0)
    elif augmentation == 'noise':
        noise = np.random.normal(0, 0.01, result.shape)
        result = result + noise
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    return result.copy()


if __name__ == "__main__":
    # Test functions
    print("Testing preprocessing module...")

    # Test normalize_band
    test_data = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    print("\nTest normalize_band:")
    print("Original:", test_data)
    print("Standardized:", normalize_band(test_data, method='standardize'))
    print("MinMax:", normalize_band(test_data, method='minmax'))

    # Test handle_nan
    print("\nTest handle_nan:")
    print("Original:", test_data)
    print("Fill with 0:", handle_nan(test_data, method='fill', fill_value=0))
    print("Fill with mean:", handle_nan(test_data, method='mean'))

    print("\n✅ Preprocessing module tests completed")
