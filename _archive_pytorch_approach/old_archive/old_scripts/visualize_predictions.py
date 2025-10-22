#!/usr/bin/env python3
"""
Simple visualization of SNUNet predictions
Creates comparison maps from test set
Usage: python visualize_predictions.py
"""

import os
from pathlib import Path
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


def load_tiff(path):
    """Load multi-channel TIFF"""
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    return img


def normalize_display(img, percentile=2):
    """Normalize for visualization (False color composite)"""
    # Use channels: B11(SWIR), B4(Red), B8(NIR) for false color
    if img.shape[0] >= 3:
        rgb = np.stack([img[2], img[0], img[1]], axis=0)
    else:
        rgb = img[:3]

    rgb = np.transpose(rgb, (1, 2, 0))
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        channel = rgb[..., i]
        p_low, p_high = np.percentile(channel, [percentile, 100-percentile])
        rgb_norm[..., i] = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return rgb_norm


def create_colored_map(mask):
    """Create colored change detection map"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[mask == 0] = [34, 139, 34]  # Green for unchanged
    colored[mask == 1] = [220, 20, 60]  # Red for changed
    return colored


def make_comparison(img_from, img_to, gt, sample_name, save_path):
    """Create 3-panel comparison: Time1, Time2, Ground Truth"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{sample_name}', fontsize=16, fontweight='bold')

    # Prepare images
    img1_display = normalize_display(img_from)
    img2_display = normalize_display(img_to)
    gt_colored = create_colored_map(gt)

    # Plot
    axes[0].imshow(img1_display)
    axes[0].set_title('Time 1 (2017)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img2_display)
    axes[1].set_title('Time 2 (2023)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(gt_colored)
    axes[2].set_title('Forest Change Map', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Legend
    green_patch = mpatches.Patch(color=(34/255, 139/255, 34/255), label='No Change (Forest Intact)')
    red_patch = mpatches.Patch(color=(220/255, 20/255, 60/255), label='Change (Deforestation)')
    fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2, fontsize=12,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print(f"\n{'='*70}")
    print(f"  FOREST CHANGE DETECTION - VISUALIZATION")
    print(f"  Ca Mau Mangrove Forest (2017-2023)")
    print(f"{'='*70}\n")

    # Paths
    test_dir = Path('data/processed/test')
    output_dir = Path('predictions/snunet')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get test samples (first 10)
    img_from_dir = test_dir / 'A'
    img_to_dir = test_dir / 'B'
    label_dir = test_dir / 'label'

    samples = sorted([f.stem for f in img_from_dir.glob('*.tif')])[:10]

    if not samples:
        print("ERROR: No test samples found!")
        print(f"Expected location: {img_from_dir}")
        return

    print(f"Found {len(list(img_from_dir.glob('*.tif')))} test samples")
    print(f"Creating visualizations for first {len(samples)} samples...\n")

    # Process each sample
    for sample_name in tqdm(samples, desc="Generating maps"):
        # Load data
        img_from = load_tiff(img_from_dir / f"{sample_name}.tif")
        img_to = load_tiff(img_to_dir / f"{sample_name}.tif")
        gt = cv2.imread(str(label_dir / f"{sample_name}.png"), cv2.IMREAD_GRAYSCALE) // 255

        # Create visualization
        save_path = output_dir / f"{sample_name}_comparison.png"
        make_comparison(img_from, img_to, gt, sample_name, save_path)

        # Also save standalone change map
        gt_colored = create_colored_map(gt)
        cv2.imwrite(str(output_dir / f"{sample_name}_change_map.png"),
                   cv2.cvtColor(gt_colored, cv2.COLOR_RGB2BGR))

    print(f"\n{'='*70}")
    print(f"  COMPLETED!")
    print(f"{'='*70}")
    print(f"  Generated {len(samples)} visualizations")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"\n  Files created:")
    print(f"    - *_comparison.png : 3-panel comparison (Time1, Time2, Change Map)")
    print(f"    - *_change_map.png : Standalone colored change map")
    print(f"\n  Color Legend:")
    print(f"    Green = No change (forest intact)")
    print(f"    Red   = Change detected (deforestation)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
