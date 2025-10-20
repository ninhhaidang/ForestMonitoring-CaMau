#!/usr/bin/env python3
"""
Evaluate SNUNet-CD on test set
Usage: python evaluate.py configs/snunet_camau.py experiments/snunet/best_mIoU_iter_5120.pth
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open-cd'))

from mmengine.config import Config
from mmengine.runner import Runner

# Import custom transforms to register them
from src.custom_transforms import MultiImgLoadRasterioFromFile


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--work-dir',
        default='./evaluation_results',
        help='Directory to save evaluation results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Override test config
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    # Use test dataloader
    cfg.test_dataloader = cfg.test_dataloader
    cfg.test_evaluator = cfg.test_evaluator

    # Build runner
    runner = Runner.from_cfg(cfg)

    # Run evaluation
    print(f"\n{'='*70}")
    print(f"  EVALUATING MODEL ON TEST SET")
    print(f"{'='*70}")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test samples: 129")
    print(f"{'='*70}\n")

    metrics = runner.test()

    # Print results
    print(f"\n{'='*70}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*70}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*70}\n")

    return metrics


if __name__ == '__main__':
    main()
