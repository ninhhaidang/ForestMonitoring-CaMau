#!/usr/bin/env python3
"""
Create prediction maps for SNUNet-CD
Usage: python create_prediction_maps.py
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
import torch

from mmengine.config import Config
import torch.nn.functional as F

# Import custom transforms and model
from src.custom_transforms import MultiImgLoadRasterioFromFile


def load_tiff(path):
    """Load TIFF file using rasterio"""
    with rasterio.open(path) as src:
        img = src.read()  # (C, H, W)
    return img.astype(np.float32)


def normalize_image(img):
    """Normalize each channel to [0, 1]"""
    img_norm = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        channel = img[i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            img_norm[i] = (channel - min_val) / (max_val - min_val)
        else:
            img_norm[i] = channel
    return img_norm


def normalize_for_display(img, percentile=2):
    """Normalize image for display"""
    # Use first 3 channels for RGB display
    if img.shape[0] > 3:
        # Use SWIR-Red-NIR for false color
        # Channels: 0=B4(Red), 1=B8(NIR), 2=B11(SWIR1)
        rgb = np.stack([img[2], img[0], img[1]], axis=0)
    else:
        rgb = img[:3]

    # Transpose to (H, W, C)
    rgb = np.transpose(rgb, (1, 2, 0))

    # Normalize each channel
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        channel = rgb[..., i]
        p_low = np.percentile(channel, percentile)
        p_high = np.percentile(channel, 100 - percentile)
        channel_norm = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
        rgb_norm[..., i] = channel_norm

    return rgb_norm


def create_change_map(pred):
    """Create colored change map"""
    h, w = pred.shape
    change_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Unchanged: Forest Green
    change_map[pred == 0] = [34, 139, 34]

    # Changed: Crimson Red
    change_map[pred == 1] = [220, 20, 60]

    return change_map


def visualize_sample(img_from, img_to, gt, pred, save_path, sample_name):
    """Create 4-panel visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Sample: {sample_name}', fontsize=16, fontweight='bold')

    # Prepare images for display
    img_from_display = normalize_for_display(img_from)
    img_to_display = normalize_for_display(img_to)
    gt_map = create_change_map(gt)
    pred_map = create_change_map(pred)

    # Plot
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

    # Legend
    unchanged_patch = mpatches.Patch(color=(34/255, 139/255, 34/255), label='Unchanged')
    changed_patch = mpatches.Patch(color=(220/255, 20/255, 60/255), label='Changed')
    fig.legend(handles=[unchanged_patch, changed_patch], loc='lower center',
               ncol=2, fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print(f"\n{'='*70}")
    print(f"  CREATING PREDICTION MAPS")
    print(f"{'='*70}\n")

    # Paths
    config_path = 'configs/snunet_camau.py'
    checkpoint_path = 'experiments/snunet/best_mIoU_iter_5120.pth'
    data_root = Path('data/processed')
    output_dir = Path('predictions/snunet')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    print("Loading model...")
    cfg = Config.fromfile(config_path)

    # Import opencd models to register custom modules
    import opencd.models

    # Build model using registry
    from mmseg.registry import MODELS
    model = MODELS.build(cfg.model)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Get test samples
    test_dir = data_root / 'test'
    img_from_dir = test_dir / 'A'
    img_to_dir = test_dir / 'B'
    label_dir = test_dir / 'label'

    # Get all samples
    samples = sorted([f.stem for f in img_from_dir.glob('*.tif')])[:10]  # First 10 samples
    print(f"\nProcessing {len(samples)} samples...\n")

    # Process each sample
    for sample_name in tqdm(samples, desc="Creating maps"):
        # Load images
        img_from_path = img_from_dir / f"{sample_name}.tif"
        img_to_path = img_to_dir / f"{sample_name}.tif"
        label_path = label_dir / f"{sample_name}.png"

        img_from = load_tiff(img_from_path)  # (C, H, W)
        img_to = load_tiff(img_to_path)
        gt = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE) // 255  # (H, W)

        # Normalize
        img_from_norm = normalize_image(img_from)
        img_to_norm = normalize_image(img_to)

        # Concatenate for model input
        img_concat = np.concatenate([img_from_norm, img_to_norm], axis=0)  # (2*C, H, W)
        img_tensor = torch.from_numpy(img_concat).unsqueeze(0).to(device)  # (1, 2*C, H, W)

        # Predict
        with torch.no_grad():
            output = model(img_tensor, mode='predict')
            if isinstance(output, list):
                pred_logits = output[0]
            else:
                pred_logits = output

            # Get prediction
            if pred_logits.shape[1] == 2:  # 2 classes
                pred = pred_logits.argmax(dim=1).cpu().numpy().squeeze()
            else:  # Binary
                pred = (pred_logits > 0).cpu().numpy().squeeze().astype(np.uint8)

        # Visualize
        save_path = output_dir / f"{sample_name}_prediction.png"
        visualize_sample(img_from, img_to, gt, pred, save_path, sample_name)

        # Save individual map
        pred_map = create_change_map(pred)
        cv2.imwrite(str(output_dir / f"{sample_name}_map.png"),
                   cv2.cvtColor(pred_map, cv2.COLOR_RGB2BGR))

    print(f"\n{'='*70}")
    print(f"  COMPLETED!")
    print(f"{'='*70}")
    print(f"  Generated {len(samples)} prediction maps")
    print(f"  Saved to: {output_dir}")
    print(f"  Files:")
    print(f"    - *_prediction.png: 4-panel comparison")
    print(f"    - *_map.png: Colored map only")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
