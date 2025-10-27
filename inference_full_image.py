"""
Full-Image Deforestation Probability Map Generation

This script applies the best trained model (Shallow U-Net) to the entire
study area to generate a deforestation probability map.

Input:
    - 4 TIFF files (S1 2024/2025, S2 2024/2025)
    - Best model checkpoint: checkpoints/shallow_unet_best.pth

Output:
    - outputs/deforestation_probability_map.tif (GeoTIFF)
    - outputs/deforestation_binary_map.tif (Binary: 0/1)
    - outputs/deforestation_statistics.txt
    - figures/full_map_visualization.png

Process:
    - Sliding window inference with 128x128 patches
    - Overlap: 64 pixels (50% overlap to reduce edge artifacts)
    - Average overlapping predictions
    - Save as GeoTIFF with original CRS and transform

Expected time: 10-15 minutes (GPU) or 30-60 minutes (CPU)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import Affine
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import get_model
from preprocessing import normalize_band, handle_nan


def load_full_image_stack(
    s1_2024_path: Path,
    s1_2025_path: Path,
    s2_2024_path: Path,
    s2_2025_path: Path,
    normalize: bool = True
) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """
    Load and stack all 4 TIFF files into 16-channel array (VH only from S1)

    Returns:
        data: (H, W, 16) array
        transform: Rasterio affine transform
        crs: Coordinate reference system
    """
    print(" Loading TIFF files...")

    # Load S1 2024 (VH only - band 1)
    with rasterio.open(s1_2024_path) as src:
        s1_2024_vh = src.read(1)  # Read only band 1 (VH): (H, W)
        s1_2024_vh = np.expand_dims(s1_2024_vh, axis=0)  # (1, H, W)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    # Load S1 2025 (VH only - band 1)
    with rasterio.open(s1_2025_path) as src:
        s1_2025_vh = src.read(1)  # Read only band 1 (VH): (H, W)
        s1_2025_vh = np.expand_dims(s1_2025_vh, axis=0)  # (1, H, W)

    # Load S2 2024
    with rasterio.open(s2_2024_path) as src:
        s2_2024 = src.read()  # (7, H, W)

    # Load S2 2025
    with rasterio.open(s2_2025_path) as src:
        s2_2025 = src.read()  # (7, H, W)

    # Stack: (16, H, W) = 1 VH (2024) + 1 VH (2025) + 7 S2 (2024) + 7 S2 (2025)
    all_bands = np.concatenate([s1_2024_vh, s1_2025_vh, s2_2024, s2_2025], axis=0)

    # Transpose to (H, W, 16)
    all_bands = np.transpose(all_bands, (1, 2, 0))

    print(f" Loaded: {all_bands.shape} ({all_bands.dtype})")
    print(f"   Transform: {transform}")
    print(f"   CRS: {crs}")

    # Handle NaN and normalize
    if normalize:
        print(" Processing bands...")
        for c in tqdm(range(16), desc="Normalize", unit="band"):
            # Handle NaN
            if np.isnan(all_bands[:, :, c]).any():
                all_bands[:, :, c] = handle_nan(all_bands[:, :, c], method='fill')

            # Normalize (same as training)
            # Channel mapping: 0=S1_VH_2024, 1=S1_VH_2025, 2-8=S2_2024, 9-15=S2_2025
            if c in [0, 1]:  # S1 VH bands
                all_bands[:, :, c] = normalize_band(all_bands[:, :, c], method='standardize')
            elif c in [2, 3, 4, 5, 9, 10, 11, 12]:  # S2 reflectance
                all_bands[:, :, c] = normalize_band(all_bands[:, :, c], method='clip', clip_range=(0, 1))
            else:  # S2 indices (6,7,8,13,14,15)
                all_bands[:, :, c] = (all_bands[:, :, c] + 1) / 2

    return all_bands, transform, crs


def sliding_window_inference(
    model: nn.Module,
    image: np.ndarray,
    window_size: int = 128,
    stride: int = 64,
    batch_size: int = 16,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Sliding window inference with overlap and averaging

    Args:
        model: Trained model
        image: (H, W, C) input image
        window_size: Patch size (default: 128)
        stride: Step size (default: 64, 50% overlap)
        batch_size: Number of patches per batch
        device: 'cuda' or 'cpu'

    Returns:
        probability_map: (H, W) probability map [0, 1]
    """
    model.eval()
    model = model.to(device)

    h, w, c = image.shape

    # Initialize output and count arrays
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.int32)

    # Calculate number of windows
    n_rows = (h - window_size) // stride + 1
    n_cols = (w - window_size) // stride + 1
    total_windows = n_rows * n_cols

    print(f"\n Sliding window inference:")
    print(f"   Image size: {h}  {w}")
    print(f"   Window size: {window_size}  {window_size}")
    print(f"   Stride: {stride} (overlap: {window_size - stride})")
    print(f"   Total windows: {total_windows:,}")
    print(f"   Batches: {(total_windows + batch_size - 1) // batch_size}")

    # Collect patches
    patches = []
    positions = []

    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride

            # Extract patch
            patch = image[y:y+window_size, x:x+window_size, :]

            # Skip if patch is incomplete (edge cases)
            if patch.shape[0] != window_size or patch.shape[1] != window_size:
                continue

            patches.append(patch)
            positions.append((y, x))

    # Process in batches
    n_patches = len(patches)
    n_batches = (n_patches + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Inference", unit="batch"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_patches)

            # Prepare batch
            batch_patches = patches[start_idx:end_idx]
            batch_positions = positions[start_idx:end_idx]

            # Convert to torch tensor: (B, C, H, W)
            batch_tensor = np.array(batch_patches)  # (B, H, W, C)
            batch_tensor = np.transpose(batch_tensor, (0, 3, 1, 2))  # (B, C, H, W)
            batch_tensor = torch.from_numpy(batch_tensor).float().to(device)

            # Forward pass
            outputs = model(batch_tensor)  # (B, 1, H, W) - logits

            # Apply sigmoid and average over spatial dimensions
            probs = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)
            probs = probs.cpu().numpy()

            # Add to probability map
            for i, (y, x) in enumerate(batch_positions):
                prob_map[y:y+window_size, x:x+window_size] += probs[i]
                count_map[y:y+window_size, x:x+window_size] += 1

    # Average overlapping predictions
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)

    print(f" Inference complete!")
    print(f"   Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    print(f"   Mean probability: {prob_map.mean():.4f}")

    return prob_map


def save_geotiff(
    output_path: Path,
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    dtype: str = 'float32',
    nodata: Optional[float] = None
):
    """Save array as GeoTIFF"""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)

    print(f" Saved: {output_path}")


def calculate_statistics(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    pixel_size_m: float = 10.0
) -> dict:
    """Calculate deforestation statistics"""

    # Binary map
    binary_map = (prob_map > threshold).astype(np.uint8)

    # Count pixels
    total_pixels = prob_map.size
    deforestation_pixels = binary_map.sum()
    no_deforestation_pixels = total_pixels - deforestation_pixels

    # Calculate area (km)
    pixel_area_m2 = pixel_size_m * pixel_size_m
    total_area_km2 = total_pixels * pixel_area_m2 / 1e6
    deforestation_area_km2 = deforestation_pixels * pixel_area_m2 / 1e6

    # Percentage
    deforestation_percentage = 100 * deforestation_pixels / total_pixels

    stats = {
        'threshold': threshold,
        'total_pixels': total_pixels,
        'deforestation_pixels': deforestation_pixels,
        'no_deforestation_pixels': no_deforestation_pixels,
        'total_area_km2': total_area_km2,
        'deforestation_area_km2': deforestation_area_km2,
        'deforestation_percentage': deforestation_percentage,
        'mean_probability': prob_map.mean(),
        'std_probability': prob_map.std(),
        'min_probability': prob_map.min(),
        'max_probability': prob_map.max()
    }

    return stats, binary_map


def visualize_probability_map(
    prob_map: np.ndarray,
    output_path: Path,
    title: str = "Deforestation Probability Map"
):
    """Create visualization of probability map"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Probability map
    ax = axes[0]
    im = ax.imshow(prob_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_title(f'{title}\n(Probability: 0=No Deforestation, 1=Deforestation)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Deforestation Probability', fontsize=12, fontweight='bold')

    # Plot 2: Binary map (threshold = 0.5)
    ax = axes[1]
    binary = (prob_map > 0.5).astype(float)
    im = ax.imshow(binary, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_title('Binary Deforestation Map\n(Threshold: 0.5)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                       ticks=[0, 1])
    cbar.ax.set_yticklabels(['No Deforestation', 'Deforestation'])

    # Statistics text
    defor_pct = 100 * binary.mean()
    fig.text(0.5, 0.02,
             f'Study Area: {prob_map.shape[1]}{prob_map.shape[0]} pixels | '
             f'Deforestation: {defor_pct:.2f}% of area',
             ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved visualization: {output_path}")
    plt.close()


def main():
    """Main inference pipeline"""

    print("="*80)
    print("FULL-IMAGE DEFORESTATION PROBABILITY MAP GENERATION")
    print("="*80)

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'raw'
    checkpoints_dir = base_dir / 'checkpoints'
    outputs_dir = base_dir / 'outputs'
    figures_dir = base_dir / 'figures'

    outputs_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Input files
    files = {
        'S1_2024': data_dir / 'sentinel1' / 'S1_2024_02_04_matched_S2_2024_01_30.tif',
        'S1_2025': data_dir / 'sentinel1' / 'S1_2025_02_22_matched_S2_2025_02_28.tif',
        'S2_2024': data_dir / 'sentinel2' / 'S2_2024_01_30.tif',
        'S2_2025': data_dir / 'sentinel2' / 'S2_2025_02_28.tif',
    }

    # Check files exist
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Load best model (Shallow U-Net)
    model_path = checkpoints_dir / 'shallow_unet_best.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"\n Loading model: {model_path.name}")
    model = get_model('shallow_unet', in_channels=18)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" Model loaded (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']*100:.2f}%)")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Load full image stack
    print("\n" + "="*80)
    image, transform, crs = load_full_image_stack(
        files['S1_2024'],
        files['S1_2025'],
        files['S2_2024'],
        files['S2_2025'],
        normalize=True
    )

    # Run inference
    print("\n" + "="*80)
    prob_map = sliding_window_inference(
        model=model,
        image=image,
        window_size=128,
        stride=64,  # 50% overlap
        batch_size=32 if device == 'cuda' else 8,
        device=device
    )

    # Save probability map
    print("\n" + "="*80)
    print(" Saving outputs...")

    prob_map_path = outputs_dir / 'deforestation_probability_map.tif'
    save_geotiff(prob_map_path, prob_map, transform, crs, dtype='float32')

    # Calculate statistics and save binary map
    stats, binary_map = calculate_statistics(prob_map, threshold=0.5, pixel_size_m=10.0)

    binary_map_path = outputs_dir / 'deforestation_binary_map.tif'
    save_geotiff(binary_map_path, binary_map, transform, crs, dtype='uint8')

    # Save statistics
    stats_path = outputs_dir / 'deforestation_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEFORESTATION STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: Shallow U-Net (Best Model)\n")
        f.write(f"Threshold: {stats['threshold']}\n")
        f.write(f"Pixel size: 10m  10m\n\n")

        f.write("AREA STATISTICS:\n")
        f.write(f"  Total area: {stats['total_area_km2']:.2f} km\n")
        f.write(f"  Deforestation area: {stats['deforestation_area_km2']:.2f} km\n")
        f.write(f"  Deforestation percentage: {stats['deforestation_percentage']:.2f}%\n\n")

        f.write("PIXEL COUNTS:\n")
        f.write(f"  Total pixels: {stats['total_pixels']:,}\n")
        f.write(f"  Deforestation pixels: {stats['deforestation_pixels']:,}\n")
        f.write(f"  No deforestation pixels: {stats['no_deforestation_pixels']:,}\n\n")

        f.write("PROBABILITY STATISTICS:\n")
        f.write(f"  Mean: {stats['mean_probability']:.4f}\n")
        f.write(f"  Std: {stats['std_probability']:.4f}\n")
        f.write(f"  Min: {stats['min_probability']:.4f}\n")
        f.write(f"  Max: {stats['max_probability']:.4f}\n")

    print(f" Saved statistics: {stats_path}")

    # Print statistics
    print("\n" + "="*80)
    print("DEFORESTATION STATISTICS")
    print("="*80)
    print(f"\n Area Statistics:")
    print(f"   Total area: {stats['total_area_km2']:.2f} km")
    print(f"   Deforestation area: {stats['deforestation_area_km2']:.2f} km")
    print(f"   Deforestation: {stats['deforestation_percentage']:.2f}% of total area")

    # Create visualization
    print("\n" + "="*80)
    print(" Creating visualization...")
    viz_path = figures_dir / 'full_map_visualization.png'
    visualize_probability_map(prob_map, viz_path)

    print("\n" + "="*80)
    print(" INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nOutput files:")
    print(f"  1. Probability map: {prob_map_path}")
    print(f"  2. Binary map: {binary_map_path}")
    print(f"  3. Statistics: {stats_path}")
    print(f"  4. Visualization: {viz_path}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
