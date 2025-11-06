"""
Configuration file for the deforestation detection project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PATCHES_DIR = DATA_DIR / "patches"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data paths
GROUND_TRUTH_CSV = RAW_DATA_DIR / "ground_truth" / "Training_Points_CSV.csv"
SENTINEL1_2024 = RAW_DATA_DIR / "sentinel-1" / "S1_2024_02_04_matched_S2_2024_01_30.tif"
SENTINEL1_2025 = RAW_DATA_DIR / "sentinel-1" / "S1_2025_02_22_matched_S2_2025_02_28.tif"
SENTINEL2_2024 = RAW_DATA_DIR / "sentinel-2" / "S2_2024_01_30.tif"
SENTINEL2_2025 = RAW_DATA_DIR / "sentinel-2" / "S2_2025_02_28.tif"

# Model parameters
PATCH_SIZE = 64  # Size of patches to extract (64x64 or 128x128)
BATCH_SIZE = 4  # Small batch size for GTX 1060 6GB
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = "cuda"  # or "cpu"

# Random seed for reproducibility
RANDOM_SEED = 42

# Sentinel-2 band configuration
S2_BANDS = ["B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]  # 7 bands
S2_NUM_BANDS = len(S2_BANDS)

# Sentinel-1 band configuration
S1_BANDS = ["VH"]  # Only VH band
S1_NUM_BANDS = len(S1_BANDS)

# Total input channels: 2 time periods × (7 S2 bands + 1 S1 band) = 16 channels
TOTAL_INPUT_CHANNELS = 2 * (S2_NUM_BANDS + S1_NUM_BANDS)

# Class labels
CLASS_NAMES = ["No Deforestation", "Deforestation"]
NUM_CLASSES = 2
