"""
Custom training script for Ca Mau Forest Change Detection

This script registers custom transforms before running Open-CD training.

Usage:
    python train_camau.py configs/tinycdv2_camau.py
"""
import sys
import os

# Add src to path and register custom transforms
sys.path.insert(0, '.')
from src.custom_transforms import MultiImgLoadRasterioFromFile

# Now run Open-CD training script
sys.path.insert(0, 'open-cd')
from tools.train import main

if __name__ == '__main__':
    main()
