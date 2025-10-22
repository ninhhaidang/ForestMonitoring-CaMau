#!/usr/bin/env python3
"""
Create classification maps from model predictions
Usage: python create_maps_from_predictions.py
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
    """Normalize for RGB display (False color composite)"""
    # SWIR-Red-NIR composite
    if img.shape[0] >= 3:
        rgb = np.stack([img[2], img[0], img[1]], axis=0)  # B11, B4, B8
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
    """Create colored classification map"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # Green for unchanged (forest intact)
    colored[mask == 0] = [34, 139, 34]

    # Red for changed (deforestation)
    colored[mask == 1] = [220, 20, 60]

    return colored


def create_full_visualization(img_from, img_to, prediction, sample_name, save_path):
    """Create 3-panel visualization: Before | After | Prediction Map"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{sample_name} - Forest Change Detection (Model Prediction)',
                 fontsize=16, fontweight='bold')

    # Prepare images
    img1_display = normalize_display(img_from)
    img2_display = normalize_display(img_to)
    pred_colored = create_colored_map(prediction)

    # Plot
    axes[0].imshow(img1_display)
    axes[0].set_title('Time 1 (2017)\nFalse Color Composite', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img2_display)
    axes[1].set_title('Time 2 (2023)\nFalse Color Composite', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pred_colored)
    axes[2].set_title('Change Detection Map\n(SNUNet-CD Prediction)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # Legend
    green_patch = mpatches.Patch(color=(34/255, 139/255, 34/255),
                                 label='No Change (Forest Intact)')
    red_patch = mpatches.Patch(color=(220/255, 20/255, 60/255),
                               label='Change (Deforestation)')
    fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2,
              fontsize=12, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_overlay_map(img, prediction, sample_name, save_path, alpha=0.5):
    """Create overlay visualization on Time 2 image"""

    img_display = normalize_display(img)
    img_display = (img_display * 255).astype(np.uint8)

    # Create red overlay for changed areas
    overlay = img_display.copy()
    changed_mask = (prediction == 1)

    # Apply red tint to changed areas
    overlay[changed_mask, 0] = np.clip(overlay[changed_mask, 0] * (1-alpha) + 220 * alpha, 0, 255)  # R
    overlay[changed_mask, 1] = np.clip(overlay[changed_mask, 1] * (1-alpha) + 20 * alpha, 0, 255)   # G
    overlay[changed_mask, 2] = np.clip(overlay[changed_mask, 2] * (1-alpha) + 60 * alpha, 0, 255)   # B

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle(f'{sample_name} - Deforestation Overlay', fontsize=16, fontweight='bold')

    ax.imshow(overlay)
    ax.set_title(f'Time 2 (2023) + Change Overlay\nRed areas = Predicted deforestation',
                fontsize=13, fontweight='bold')
    ax.axis('off')

    red_patch = mpatches.Patch(color=(220/255, 20/255, 60/255, alpha),
                               label=f'Deforestation (α={alpha})')
    fig.legend(handles=[red_patch], loc='lower center', fontsize=12,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print(f"\n{'='*70}")
    print(f"  CREATING CLASSIFICATION MAPS FROM MODEL PREDICTIONS")
    print(f"  Model: SNUNet-CD")
    print(f"{'='*70}\n")

    # Paths
    test_dir = Path('data/processed/test')
    pred_dir = Path('predictions/snunet_predictions')
    output_dir = Path('predictions/snunet_maps')
    output_dir.mkdir(parents=True, exist_ok=True)

    img_from_dir = test_dir / 'A'
    img_to_dir = test_dir / 'B'

    # Get all predictions
    pred_files = sorted(pred_dir.glob('*.png'))
    total = len(pred_files)

    if total == 0:
        print("ERROR: No predictions found!")
        print(f"Expected location: {pred_dir}")
        return

    print(f"Found {total} predictions")
    print(f"Creating classification maps...\n")

    # Statistics
    stats = {
        'total_pixels': 0,
        'unchanged_pixels': 0,
        'changed_pixels': 0
    }

    # Process each prediction
    for pred_file in tqdm(pred_files, desc="Creating maps"):
        sample_name = pred_file.stem

        # Load prediction (0=unchanged, 255=changed)
        pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE) // 255

        # Load satellite images
        img_from_path = img_from_dir / f"{sample_name}.tif"
        img_to_path = img_to_dir / f"{sample_name}.tif"

        if not img_from_path.exists() or not img_to_path.exists():
            print(f"Skipping {sample_name}: Missing TIFF files")
            continue

        img_from = load_tiff(img_from_path)
        img_to = load_tiff(img_to_path)

        # Update statistics
        stats['total_pixels'] += pred_mask.size
        stats['unchanged_pixels'] += np.sum(pred_mask == 0)
        stats['changed_pixels'] += np.sum(pred_mask == 1)

        # 1. Full visualization (3-panel)
        save_path = output_dir / f"{sample_name}_prediction.png"
        create_full_visualization(img_from, img_to, pred_mask, sample_name, save_path)

        # 2. Standalone classification map
        pred_colored = create_colored_map(pred_mask)
        cv2.imwrite(str(output_dir / f"{sample_name}_map.png"),
                   cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))

        # 3. Overlay on Time 2
        save_path = output_dir / f"{sample_name}_overlay.png"
        create_overlay_map(img_to, pred_mask, sample_name, save_path, alpha=0.5)

    # Print statistics
    unchanged_pct = (stats['unchanged_pixels'] / stats['total_pixels']) * 100
    changed_pct = (stats['changed_pixels'] / stats['total_pixels']) * 100

    print(f"\n{'='*70}")
    print(f"  COMPLETED!")
    print(f"{'='*70}")
    print(f"  Total samples: {total}")
    print(f"  Output directory: {output_dir.absolute()}\n")
    print(f"  Files per sample:")
    print(f"    *_prediction.png : 3-panel comparison")
    print(f"    *_map.png        : Standalone classification map")
    print(f"    *_overlay.png    : Overlay on Time 2 image\n")
    print(f"  Classification Statistics:")
    print(f"    Total pixels: {stats['total_pixels']:,}")
    print(f"    Unchanged (Forest intact): {stats['unchanged_pixels']:,} ({unchanged_pct:.2f}%)")
    print(f"    Changed (Deforestation):   {stats['changed_pixels']:,} ({changed_pct:.2f}%)")
    print(f"\n  Color Legend:")
    print(f"    Green (34, 139, 34)  = No change / Forest intact")
    print(f"    Red (220, 20, 60)    = Change detected / Deforestation")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
