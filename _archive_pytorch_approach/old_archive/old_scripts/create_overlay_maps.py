#!/usr/bin/env python3
"""
Create overlay visualization - change detection overlay on satellite images
Usage: python create_overlay_maps.py
"""

import os
from pathlib import Path
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def load_tiff(path):
    """Load multi-channel TIFF"""
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    return img


def normalize_display(img, percentile=2):
    """Normalize for RGB display"""
    # False color: SWIR-Red-NIR (B11-B4-B8)
    if img.shape[0] >= 3:
        rgb = np.stack([img[2], img[0], img[1]], axis=0)
    else:
        rgb = img[:3]

    rgb = np.transpose(rgb, (1, 2, 0))
    rgb_norm = np.zeros_like(rgb, dtype=np.uint8)

    for i in range(3):
        channel = rgb[..., i]
        p_low, p_high = np.percentile(channel, [percentile, 100-percentile])
        channel_norm = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
        rgb_norm[..., i] = (channel_norm * 255).astype(np.uint8)

    return rgb_norm


def create_overlay(img_rgb, change_mask, alpha=0.5):
    """
    Overlay change mask on image with transparency

    Args:
        img_rgb: RGB image (H, W, 3) uint8
        change_mask: Binary mask (H, W) where 1=changed
        alpha: Transparency (0=transparent, 1=opaque)

    Returns:
        Overlaid image
    """
    overlay = img_rgb.copy()

    # Create red overlay for changed areas
    red_overlay = np.zeros_like(overlay)
    red_overlay[change_mask == 1] = [220, 20, 60]  # Crimson red

    # Blend
    result = img_rgb.copy()
    mask_3ch = np.stack([change_mask, change_mask, change_mask], axis=-1)
    result = np.where(mask_3ch == 1,
                      cv2.addWeighted(img_rgb, 1-alpha, red_overlay, alpha, 0),
                      img_rgb)

    return result.astype(np.uint8)


def create_side_by_side_overlay(img_before, img_after, change_mask, sample_name, save_path, alpha=0.5):
    """Create side-by-side visualization with overlay on After image"""

    # Normalize images
    before_rgb = normalize_display(img_before)
    after_rgb = normalize_display(img_after)

    # Create overlay on After image
    after_overlay = create_overlay(after_rgb, change_mask, alpha)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'{sample_name} - Deforestation Detection', fontsize=16, fontweight='bold')

    # Before
    axes[0].imshow(before_rgb)
    axes[0].set_title('Before (2017)\nFalse Color Composite', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # After with overlay
    axes[1].imshow(after_overlay)
    axes[1].set_title(f'After (2023) with Change Overlay\nRed = Deforestation (α={alpha})',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Legend
    red_patch = mpatches.Patch(color=(220/255, 20/255, 60/255, alpha),
                               label=f'Deforestation (α={alpha})')
    fig.legend(handles=[red_patch], loc='lower center', fontsize=12,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_triple_overlay(img_before, img_after, change_mask, sample_name, save_path, alpha=0.5):
    """Create 3-panel: Before | After | After+Overlay"""

    before_rgb = normalize_display(img_before)
    after_rgb = normalize_display(img_after)
    after_overlay = create_overlay(after_rgb, change_mask, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f'{sample_name} - Change Detection Overlay', fontsize=16, fontweight='bold')

    axes[0].imshow(before_rgb)
    axes[0].set_title('Time 1 (2017)', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(after_rgb)
    axes[1].set_title('Time 2 (2023)', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(after_overlay)
    axes[2].set_title(f'Time 2 + Change Overlay\nRed = Deforestation', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    red_patch = mpatches.Patch(color=(220/255, 20/255, 60/255, alpha),
                               label=f'Deforestation overlay (α={alpha})')
    fig.legend(handles=[red_patch], loc='lower center', fontsize=12,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_change_highlight(img_before, img_after, change_mask, sample_name, save_path):
    """Create change highlight visualization with both images"""

    before_rgb = normalize_display(img_before)
    after_rgb = normalize_display(img_after)

    # Create blended image showing change
    h, w = change_mask.shape
    highlight = np.zeros((h, w, 3), dtype=np.uint8)

    # Unchanged areas: show After image in grayscale
    unchanged_mask = (change_mask == 0)
    gray_after = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY)
    for c in range(3):
        highlight[:, :, c][unchanged_mask] = gray_after[unchanged_mask]

    # Changed areas: show After image in full color + red tint
    changed_mask = (change_mask == 1)
    red_tint = after_rgb.copy()
    red_tint[changed_mask, 0] = np.clip(red_tint[changed_mask, 0] * 1.3 + 50, 0, 255)  # Boost red
    red_tint[changed_mask, 1] = red_tint[changed_mask, 1] * 0.7  # Reduce green
    red_tint[changed_mask, 2] = red_tint[changed_mask, 2] * 0.7  # Reduce blue
    highlight[changed_mask] = red_tint[changed_mask]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'{sample_name} - Change Highlight Analysis', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(before_rgb)
    axes[0, 0].set_title('Before (2017)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(after_rgb)
    axes[0, 1].set_title('After (2023)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(highlight)
    axes[1, 0].set_title('Change Highlight\nGray=Unchanged, Color+Red=Changed',
                        fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')

    # Change map
    change_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    change_rgb[change_mask == 0] = [34, 139, 34]
    change_rgb[change_mask == 1] = [220, 20, 60]
    axes[1, 1].imshow(change_rgb)
    axes[1, 1].set_title('Change Classification', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')

    green_patch = mpatches.Patch(color=(34/255, 139/255, 34/255), label='Unchanged')
    red_patch = mpatches.Patch(color=(220/255, 20/255, 60/255), label='Deforestation')
    fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print(f"\n{'='*70}")
    print(f"  OVERLAY VISUALIZATION - Forest Change Detection")
    print(f"  Overlaying change detection on satellite imagery")
    print(f"{'='*70}\n")

    # Paths
    test_dir = Path('data/processed/test')
    output_dir = Path('predictions/snunet_overlay')
    output_dir.mkdir(parents=True, exist_ok=True)

    img_from_dir = test_dir / 'A'
    img_to_dir = test_dir / 'B'
    label_dir = test_dir / 'label'

    # Get samples
    samples = sorted([f.stem for f in img_from_dir.glob('*.tif')])[:10]

    if not samples:
        print("ERROR: No test samples found!")
        return

    print(f"Processing {len(samples)} samples with 3 overlay styles:\n")
    print("  1. Side-by-side: Before | After+Overlay")
    print("  2. Triple view: Before | After | After+Overlay")
    print("  3. Change highlight: 4-panel analysis\n")

    alpha = 0.5  # Transparency for overlay

    # Process each sample
    for sample_name in tqdm(samples, desc="Creating overlays"):
        # Load data
        img_from = load_tiff(img_from_dir / f"{sample_name}.tif")
        img_to = load_tiff(img_to_dir / f"{sample_name}.tif")
        change_mask = cv2.imread(str(label_dir / f"{sample_name}.png"),
                                cv2.IMREAD_GRAYSCALE) // 255

        # 1. Side-by-side overlay
        save_path = output_dir / f"{sample_name}_sidebyside.png"
        create_side_by_side_overlay(img_from, img_to, change_mask, sample_name, save_path, alpha)

        # 2. Triple overlay
        save_path = output_dir / f"{sample_name}_triple.png"
        create_triple_overlay(img_from, img_to, change_mask, sample_name, save_path, alpha)

        # 3. Change highlight
        save_path = output_dir / f"{sample_name}_highlight.png"
        create_change_highlight(img_from, img_to, change_mask, sample_name, save_path)

    print(f"\n{'='*70}")
    print(f"  COMPLETED!")
    print(f"{'='*70}")
    print(f"  Generated {len(samples) * 3} overlay visualizations")
    print(f"  Output: {output_dir.absolute()}\n")
    print(f"  File types:")
    print(f"    *_sidebyside.png : Before | After+Overlay (2 panels)")
    print(f"    *_triple.png     : Before | After | After+Overlay (3 panels)")
    print(f"    *_highlight.png  : 4-panel change highlight analysis")
    print(f"\n  Overlay settings:")
    print(f"    Alpha (transparency): {alpha}")
    print(f"    Color for deforestation: Red (220, 20, 60)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
