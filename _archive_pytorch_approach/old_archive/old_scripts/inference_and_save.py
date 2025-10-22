#!/usr/bin/env python3
"""
Run inference and save prediction masks
Usage: python inference_and_save.py configs/snunet_camau.py experiments/snunet/best_mIoU_iter_5120.pth
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open-cd'))

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import Runner
import torch

# Import custom transforms
from src.custom_transforms import MultiImgLoadRasterioFromFile


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference and save predictions')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--output-dir',
        default='./predictions/snunet_predictions',
        help='Directory to save prediction masks')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  RUNNING INFERENCE - SNUNet-CD")
    print(f"{'='*70}")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(output_dir)
    cfg.load_from = args.checkpoint

    # Build runner
    print("Loading model...")
    runner = Runner.from_cfg(cfg)

    # Load checkpoint
    runner.load_checkpoint(args.checkpoint)
    runner.model.eval()

    device = next(runner.model.parameters()).device
    print(f"Model loaded on {device}\n")

    # Get test dataloader
    test_dataloader = runner.test_dataloader
    total_samples = len(test_dataloader.dataset)

    print(f"Running inference on {total_samples} test samples...\n")

    # Run inference
    predictions_saved = 0

    for idx, data_batch in enumerate(tqdm(test_dataloader, desc="Inference")):
        # Get inputs
        inputs = data_batch['inputs']
        data_samples = data_batch['data_samples']

        # Move to device
        inputs = [inp.to(device) for inp in inputs]

        # Inference
        with torch.no_grad():
            results = runner.model.test_step(data_batch)

        # Save predictions
        for i, result in enumerate(results):
            # Get prediction
            pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()  # (H, W)

            # Get sample info
            data_sample = data_samples[i]
            img_path = data_sample.img_path

            # Extract sample name
            if isinstance(img_path, str):
                sample_name = Path(img_path).stem
            else:
                sample_name = f"sample_{idx:04d}_{i}"

            # Save as PNG (0=unchanged, 255=changed)
            pred_png = (pred_mask * 255).astype(np.uint8)
            save_path = output_dir / f"{sample_name}.png"
            cv2.imwrite(str(save_path), pred_png)

            predictions_saved += 1

    print(f"\n{'='*70}")
    print(f"  INFERENCE COMPLETED!")
    print(f"{'='*70}")
    print(f"  Total predictions saved: {predictions_saved}")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Format: PNG (0=unchanged, 255=changed)")
    print(f"{'='*70}\n")

    return predictions_saved


if __name__ == '__main__':
    main()
