#!/usr/bin/env python3
"""Rename predictions to match TIFF file names"""

from pathlib import Path

pred_dir = Path('predictions/snunet_predictions')

count = 0
for pred_file in sorted(pred_dir.glob('sample_*.png')):
    # Extract number from "sample_0000_0.png"
    name = pred_file.stem  # "sample_0000_0"
    parts = name.split('_')
    if len(parts) >= 2:
        idx = int(parts[1])  # 0, 1, 2, ...
        new_idx = idx + 1     # 1, 2, 3, ...
        new_name = f"{new_idx:04d}.png"  # "0001.png", "0002.png", ...

        new_path = pred_file.parent / new_name
        pred_file.rename(new_path)
        count += 1

print(f"Renamed {count} files")
