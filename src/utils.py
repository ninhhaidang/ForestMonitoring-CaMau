"""
Utility functions for data loading and metadata checking
"""
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


def load_tiff(filepath: Union[str, Path],
              bands: Optional[list] = None,
              window: Optional[rasterio.windows.Window] = None) -> np.ndarray:
    """
    Load TIFF file and return as numpy array

    Args:
        filepath: Path to TIFF file
        bands: List of band indices to load (1-indexed). If None, load all bands
        window: Rasterio window to read subset of data

    Returns:
        numpy array of shape (height, width, bands) or (height, width) for single band

    Example:
        >>> data = load_tiff('S1_2024.tif')
        >>> print(data.shape)  # (11261, 7970, 2)

        >>> # Load specific bands
        >>> data = load_tiff('S2_2024.tif', bands=[1, 2, 3])
        >>> print(data.shape)  # (11261, 7970, 3)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with rasterio.open(filepath) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))

        if len(bands) == 1:
            data = src.read(bands[0], window=window)
        else:
            data = src.read(bands, window=window)
            # Transpose to (H, W, C) format
            data = np.transpose(data, (1, 2, 0))

    return data


def check_tiff_metadata(filepath: Union[str, Path],
                       verbose: bool = False) -> Dict:
    """
    Check metadata of TIFF file

    Args:
        filepath: Path to TIFF file
        verbose: Print metadata to console

    Returns:
        Dictionary with metadata information

    Example:
        >>> meta = check_tiff_metadata('S1_2024.tif', verbose=True)
        >>> print(meta['bands'], meta['width'], meta['height'])
    """
    filepath = Path(filepath)
    metadata = {}

    with rasterio.open(filepath) as src:
        metadata['name'] = filepath.name
        metadata['bands'] = src.count
        metadata['width'] = src.width
        metadata['height'] = src.height
        metadata['dtype'] = str(src.dtypes[0])
        metadata['crs'] = str(src.crs)
        metadata['transform'] = src.transform
        metadata['bounds'] = src.bounds
        metadata['pixel_size_x'] = src.transform[0]
        metadata['pixel_size_y'] = abs(src.transform[4])
        metadata['nodata'] = src.nodatavals

        # Calculate memory size
        bytes_per_pixel = {
            'float32': 4, 'float64': 8,
            'uint8': 1, 'int16': 2, 'uint16': 2
        }.get(str(src.dtypes[0]), 4)

        total_pixels = src.width * src.height * src.count
        metadata['memory_mb'] = (total_pixels * bytes_per_pixel) / (1024**2)

        if verbose:
            print(f"\n{'='*80}")
            print(f"FILE: {metadata['name']}")
            print(f"{'='*80}")
            print(f"\n📊 BASIC INFO:")
            print(f"  - Bands: {metadata['bands']}")
            print(f"  - Size: {metadata['width']} × {metadata['height']} pixels")
            print(f"  - Data type: {metadata['dtype']}")
            print(f"  - CRS: {metadata['crs']}")
            print(f"  - Pixel size: {metadata['pixel_size_x']:.2f} × {metadata['pixel_size_y']:.2f} meters")
            print(f"  - Memory: {metadata['memory_mb']:.2f} MB")

    return metadata


def get_tiff_stats(filepath: Union[str, Path],
                   sample_size: int = 1000,
                   from_center: bool = True) -> pd.DataFrame:
    """
    Get statistics for all bands in TIFF file

    Args:
        filepath: Path to TIFF file
        sample_size: Size of sample window (pixels)
        from_center: Sample from center if True, else from top-left

    Returns:
        DataFrame with statistics for each band

    Example:
        >>> stats = get_tiff_stats('S2_2024.tif', sample_size=1000)
        >>> print(stats)
    """
    filepath = Path(filepath)
    band_stats = []

    with rasterio.open(filepath) as src:
        # Define sampling window
        if from_center:
            center_x = src.width // 2
            center_y = src.height // 2
            half = sample_size // 2
            window = rasterio.windows.Window(
                center_x - half,
                center_y - half,
                sample_size,
                sample_size
            )
        else:
            window = rasterio.windows.Window(0, 0, sample_size, sample_size)

        # Calculate stats for each band
        for i in range(1, src.count + 1):
            band_data = src.read(i, window=window)

            nan_count = np.isnan(band_data).sum()
            inf_count = np.isinf(band_data).sum()
            valid_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]

            stats = {
                'file': filepath.name,
                'band': i,
                'total_pixels': band_data.size,
                'nan_count': nan_count,
                'nan_percent': 100 * nan_count / band_data.size,
                'inf_count': inf_count,
                'valid_count': len(valid_data),
            }

            if len(valid_data) > 0:
                stats.update({
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'median': float(np.median(valid_data)),
                    'q25': float(np.percentile(valid_data, 25)),
                    'q75': float(np.percentile(valid_data, 75)),
                })
            else:
                stats.update({
                    'min': np.nan, 'max': np.nan, 'mean': np.nan,
                    'std': np.nan, 'median': np.nan, 'q25': np.nan, 'q75': np.nan
                })

            band_stats.append(stats)

    return pd.DataFrame(band_stats)


def load_ground_truth(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load ground truth labels from CSV file

    Args:
        csv_path: Path to CSV file with columns: id, label, x, y

    Returns:
        DataFrame with ground truth data

    Example:
        >>> gt = load_ground_truth('Training_Points_CSV.csv')
        >>> print(gt.head())
        >>> print(f"Total points: {len(gt)}")
        >>> print(f"Class 0: {(gt['label']==0).sum()}, Class 1: {(gt['label']==1).sum()}")
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    required_cols = ['id', 'label', 'x', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate labels (should be 0 or 1)
    unique_labels = df['label'].unique()
    if not all(label in [0, 1] for label in unique_labels):
        raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")

    return df


def pixel_to_coord(pixel_x: int, pixel_y: int,
                   transform: rasterio.Affine) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates

    Args:
        pixel_x: Pixel column (0-indexed)
        pixel_y: Pixel row (0-indexed)
        transform: Rasterio transform object

    Returns:
        Tuple of (geo_x, geo_y) coordinates

    Example:
        >>> with rasterio.open('image.tif') as src:
        >>>     geo_x, geo_y = pixel_to_coord(100, 200, src.transform)
    """
    geo_x, geo_y = transform * (pixel_x + 0.5, pixel_y + 0.5)
    return geo_x, geo_y


def coord_to_pixel(geo_x: float, geo_y: float,
                   transform: rasterio.Affine) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates

    Args:
        geo_x: Geographic X coordinate (e.g., UTM easting)
        geo_y: Geographic Y coordinate (e.g., UTM northing)
        transform: Rasterio transform object

    Returns:
        Tuple of (pixel_x, pixel_y) coordinates (0-indexed)

    Example:
        >>> with rasterio.open('image.tif') as src:
        >>>     px, py = coord_to_pixel(500000, 1000000, src.transform)
    """
    inv_transform = ~transform
    pixel_x, pixel_y = inv_transform * (geo_x, geo_y)
    return int(pixel_x), int(pixel_y)


def get_sample_window(src: rasterio.DatasetReader,
                     center_x: int,
                     center_y: int,
                     window_size: int) -> rasterio.windows.Window:
    """
    Create a window centered at given pixel coordinates

    Args:
        src: Rasterio dataset reader
        center_x: Center pixel X coordinate
        center_y: Center pixel Y coordinate
        window_size: Size of window (square)

    Returns:
        Rasterio Window object

    Example:
        >>> with rasterio.open('image.tif') as src:
        >>>     window = get_sample_window(src, 1000, 2000, 128)
        >>>     data = src.read(window=window)
    """
    half = window_size // 2

    # Clamp to image bounds
    col_off = max(0, center_x - half)
    row_off = max(0, center_y - half)

    width = min(window_size, src.width - col_off)
    height = min(window_size, src.height - row_off)

    return rasterio.windows.Window(col_off, row_off, width, height)


def print_dataset_summary(gt_df: pd.DataFrame):
    """
    Print summary statistics of ground truth dataset

    Args:
        gt_df: Ground truth DataFrame

    Example:
        >>> gt = load_ground_truth('Training_Points_CSV.csv')
        >>> print_dataset_summary(gt)
    """
    print("\n" + "="*80)
    print("GROUND TRUTH DATASET SUMMARY")
    print("="*80)

    print(f"\n📊 OVERVIEW:")
    print(f"  - Total samples: {len(gt_df):,}")

    print(f"\n🏷️ CLASS DISTRIBUTION:")
    for label in sorted(gt_df['label'].unique()):
        count = (gt_df['label'] == label).sum()
        pct = 100 * count / len(gt_df)
        label_name = "No deforestation" if label == 0 else "Deforestation"
        print(f"  - Class {label} ({label_name}): {count:,} ({pct:.1f}%)")

    print(f"\n🗺️ SPATIAL EXTENT:")
    print(f"  - X range: {gt_df['x'].min():.2f} - {gt_df['x'].max():.2f}")
    print(f"  - Y range: {gt_df['y'].min():.2f} - {gt_df['y'].max():.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Test functions
    print("Testing utils module...")

    # Example paths (adjust as needed)
    base_dir = Path("../data/raw")
    s1_file = base_dir / "sentinel1" / "S1_2024_02_04_matched_S2_2024_01_30.tif"

    if s1_file.exists():
        print("\n✅ Testing check_tiff_metadata:")
        meta = check_tiff_metadata(s1_file, verbose=True)

        print("\n✅ Testing get_tiff_stats:")
        stats = get_tiff_stats(s1_file, sample_size=500)
        print(stats)

        print("\n✅ Testing load_tiff:")
        data = load_tiff(s1_file, bands=[1])
        print(f"Loaded data shape: {data.shape}")
    else:
        print(f"⚠️ Test file not found: {s1_file}")
