#!/usr/bin/env python3
"""
Create prediction maps and visualizations for SNUNet-CD
Usage: python predict_and_visualize.py configs/snunet_camau.py experiments/snunet/best_mIoU_iter_5120.pth
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open-cd'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
import torch

# Import custom transforms
from src.custom_transforms import MultiImgLoadRasterioFromFile


def parse_args():
    parser = argparse.ArgumentParser(description='Generate prediction maps')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--output-dir',
        default='./predictions',
        help='Directory to save predictions')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to visualize (default: 10, use -1 for all)')
    args = parser.parse_args()
    return args


def normalize_for_display(img, percentile=2):
    """Normalize image for display using percentile clipping"""
    if img.ndim == 3 and img.shape[0] > 3:
        # For multi-channel, use first 3 channels (assume RGB-like)
        img = img[:3]

    # Move channels to last dimension
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    # Normalize each channel
    img_norm = np.zeros_like(img, dtype=np.float32)
    for i in range(min(3, img.shape[-1])):
        channel = img[..., i]
        p_low = np.percentile(channel, percentile)
        p_high = np.percentile(channel, 100 - percentile)
        channel_norm = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
        img_norm[..., i] = channel_norm

    return img_norm


def create_change_map(pred, gt=None):
    """Create colored change detection map"""
    # Create RGB image
    h, w = pred.shape
    change_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Unchanged: Green (0, 255, 0)
    change_map[pred == 0] = [34, 139, 34]  # Forest green

    # Changed: Red (255, 0, 0)
    change_map[pred == 1] = [220, 20, 60]  # Crimson

    return change_map


def visualize_sample(img_from, img_to, gt, pred, save_path, sample_name):
    """Visualize one sample with 4 panels"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Sample: {sample_name}', fontsize=16, fontweight='bold')

    # Normalize images for display (use RGB bands if available)
    # Assume channels are: B4, B8, B11, B12, NDVI, NBR, NDMI, VH, Ratio
    # Use B4(Red), B8(NIR), B11(SWIR) as RGB composite
    if img_from.shape[0] >= 3:
        # False color: NIR-R-G (B8-B4-B2) or similar
        # We have B4(0), B8(1), B11(2), B12(3)
        # Use B8(NIR), B4(Red), B11(SWIR) for false color
        rgb_from = np.stack([img_from[2], img_from[0], img_from[1]], axis=0)  # SWIR, R, NIR
        rgb_to = np.stack([img_to[2], img_to[0], img_to[1]], axis=0)
    else:
        rgb_from = img_from[:3]
        rgb_to = img_to[:3]

    img_from_display = normalize_for_display(rgb_from)
    img_to_display = normalize_for_display(rgb_to)

    # Time 1
    axes[0, 0].imshow(img_from_display)
    axes[0, 0].set_title('Time 1 (Before)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Time 2
    axes[0, 1].imshow(img_to_display)
    axes[0, 1].set_title('Time 2 (After)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Ground Truth
    gt_map = create_change_map(gt)
    axes[1, 0].imshow(gt_map)
    axes[1, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Prediction
    pred_map = create_change_map(pred)
    axes[1, 1].imshow(pred_map)
    axes[1, 1].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Add legend
    unchanged_patch = mpatches.Patch(color=(34/255, 139/255, 34/255), label='Unchanged (No deforestation)')
    changed_patch = mpatches.Patch(color=(220/255, 20/255, 60/255), label='Changed (Deforestation)')
    fig.legend(handles=[unchanged_patch, changed_patch], loc='lower center',
               ncol=2, fontsize=12, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  GENERATING PREDICTION MAPS")
    print(f"{'='*70}")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output dir: {output_dir}")
    print(f"  Samples to visualize: {args.num_samples if args.num_samples > 0 else 'ALL'}")
    print(f"{'='*70}\n")

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(output_dir)
    cfg.load_from = args.checkpoint

    # Build runner
    runner = Runner.from_cfg(cfg)

    # Load model
    runner.load_checkpoint(args.checkpoint)
    runner.model.eval()

    # Get test dataloader
    test_dataloader = runner.test_dataloader

    # Determine number of samples
    total_samples = len(test_dataloader.dataset)
    num_viz = total_samples if args.num_samples == -1 else min(args.num_samples, total_samples)

    print(f"Processing {num_viz} / {total_samples} test samples...\n")

    # Process samples
    device = next(runner.model.parameters()).device

    viz_count = 0
    for idx, data_batch in enumerate(tqdm(test_dataloader, desc="Generating maps")):
        if viz_count >= num_viz:
            break

        # Get batch data
        inputs = data_batch['inputs']
        data_samples = data_batch['data_samples']

        # Move to device
        inputs = [inp.to(device) for inp in inputs]

        # Inference
        with torch.no_grad():
            predictions = runner.model.test_step(data_batch)

        # Process each sample in batch
        batch_size = len(data_samples)
        for i in range(batch_size):
            if viz_count >= num_viz:
                break

            # Get data
            img_from = inputs[0][i].cpu().numpy()  # (C, H, W)
            img_to = inputs[1][i].cpu().numpy()

            # Get ground truth
            gt = data_samples[i].gt_sem_seg.data.cpu().numpy().squeeze()  # (H, W)

            # Get prediction
            pred = predictions[i].pred_sem_seg.data.cpu().numpy().squeeze()  # (H, W)

            # Get sample name
            img_path = data_samples[i].img_path
            sample_name = Path(img_path).stem if isinstance(img_path, str) else f"sample_{idx}_{i}"

            # Visualize
            save_path = output_dir / f"{sample_name}_prediction.png"
            visualize_sample(img_from, img_to, gt, pred, save_path, sample_name)

            # Also save individual maps
            pred_map = create_change_map(pred)
            cv2.imwrite(str(output_dir / f"{sample_name}_map.png"),
                       cv2.cvtColor(pred_map, cv2.COLOR_RGB2BGR))

            viz_count += 1

    print(f"\n{'='*70}")
    print(f"  COMPLETED!")
    print(f"{'='*70}")
    print(f"  Generated {viz_count} prediction visualizations")
    print(f"  Saved to: {output_dir}")
    print(f"  Files:")
    print(f"    - *_prediction.png: 4-panel comparison (Time1, Time2, GT, Pred)")
    print(f"    - *_map.png: Colored prediction map only")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
