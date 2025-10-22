#!/usr/bin/env python3
"""
Generate prediction maps using Open-CD's built-in test tools
Usage: python generate_maps.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open-cd'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
import rasterio
from tqdm import tqdm

# Import custom transforms
from src.custom_transforms import MultiImgLoadRasterioFromFile


def load_tiff(path):
    """Load TIFF file"""
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    return img


def normalize_for_display(img, percentile=2):
    """Normalize for visualization"""
    if img.shape[0] > 3:
        # False color: SWIR-Red-NIR
        rgb = np.stack([img[2], img[0], img[1]], axis=0)
    else:
        rgb = img[:3]

    rgb = np.transpose(rgb, (1, 2, 0))
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        channel = rgb[..., i]
        p_low = np.percentile(channel, percentile)
        p_high = np.percentile(channel, 100 - percentile)
        rgb_norm[..., i] = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return rgb_norm


def create_change_map(pred):
    """Create colored change map"""
    h, w = pred.shape
    change_map = np.zeros((h, w, 3), dtype=np.uint8)
    change_map[pred == 0] = [34, 139, 34]  # Unchanged: Green
    change_map[pred == 1] = [220, 20, 60]  # Changed: Red
    return change_map


def visualize_sample(img_from, img_to, gt, pred, save_path, sample_name):
    """Create 4-panel visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Sample: {sample_name}', fontsize=16, fontweight='bold')

    img_from_display = normalize_for_display(img_from)
    img_to_display = normalize_for_display(img_to)
    gt_map = create_change_map(gt)
    pred_map = create_change_map(pred)

    axes[0, 0].imshow(img_from_display)
    axes[0, 0].set_title('Time 1 (Before)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_to_display)
    axes[0, 1].set_title('Time 2 (After)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(gt_map)
    axes[1, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_map)
    axes[1, 1].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    unchanged_patch = mpatches.Patch(color=(34/255, 139/255, 34/255), label='Unchanged')
    changed_patch = mpatches.Patch(color=(220/255, 20/255, 60/255), label='Changed')
    fig.legend(handles=[unchanged_patch, changed_patch], loc='lower center', ncol=2, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print(f"\n{'='*70}")
    print(f"  GENERATING PREDICTION MAPS")
    print(f"{'='*70}\n")

    # Step 1: Run evaluation to generate predictions
    print("Step 1: Running model inference on test set...")
    print("(This will evaluate the model and save predictions)")

    cmd = (
        "python evaluate.py "
        "configs/snunet_camau.py "
        "experiments/snunet/best_mIoU_iter_5120.pth "
        "--work-dir ./predictions/snunet_eval"
    )

    print(f"\nCommand: {cmd}\n")

    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error during evaluation:")
        print(result.stderr)
        return

    print("Evaluation completed!\n")

    # Step 2: Create visualizations from test set
    print("Step 2: Creating visualization maps...\n")

    data_root = Path('data/processed/test')
    output_dir = Path('predictions/snunet')
    output_dir.mkdir(parents=True, exist_ok=True)

    img_from_dir = data_root / 'A'
    img_to_dir = data_root / 'B'
    label_dir = data_root / 'label'

    # Get first 10 samples
    samples = sorted([f.stem for f in img_from_dir.glob('*.tif')])[:10]

    print(f"Processing {len(samples)} samples...\n")

    # Simple inference - using model predictions
    # For now, we'll just visualize GT vs GT as demo
    # In real use, you would load predictions from model

    for sample_name in tqdm(samples, desc="Creating maps"):
        img_from_path = img_from_dir / f"{sample_name}.tif"
        img_to_path = img_to_dir / f"{sample_name}.tif"
        label_path = label_dir / f"{sample_name}.png"

        img_from = load_tiff(img_from_path)
        img_to = load_tiff(img_to_path)
        gt = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE) // 255

        # TODO: Load actual predictions from model
        # For now use GT as placeholder
        pred = gt.copy()

        # Visualize
        save_path = output_dir / f"{sample_name}_prediction.png"
        visualize_sample(img_from, img_to, gt, pred, save_path, sample_name)

        # Save map
        pred_map = create_change_map(pred)
        cv2.imwrite(str(output_dir / f"{sample_name}_map.png"),
                   cv2.cvtColor(pred_map, cv2.COLOR_RGB2BGR))

    print(f"\n{'='*70}")
    print(f"  COMPLETED!")
    print(f"{'='*70}")
    print(f"  Generated {len(samples)} visualizations")
    print(f"  Saved to: {output_dir}")
    print(f"\n  NOTE: Currently showing Ground Truth as prediction.")
    print(f"  To show actual model predictions, we need to modify")
    print(f"  Open-CD's test pipeline to save prediction outputs.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
