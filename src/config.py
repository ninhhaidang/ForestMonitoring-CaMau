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
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [PATCHES_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data paths
GROUND_TRUTH_CSV = RAW_DATA_DIR / "ground_truth" / "Training_Points_CSV.csv"
SENTINEL1_2024 = RAW_DATA_DIR / "sentinel-1" / "S1_2024_02_04_matched_S2_2024_01_30.tif"
SENTINEL1_2025 = RAW_DATA_DIR / "sentinel-1" / "S1_2025_02_22_matched_S2_2025_02_28.tif"
SENTINEL2_2024 = RAW_DATA_DIR / "sentinel-2" / "S2_2024_01_30.tif"
SENTINEL2_2025 = RAW_DATA_DIR / "sentinel-2" / "S2_2025_02_28.tif"
FOREST_BOUNDARY = RAW_DATA_DIR / "boundary" / "forest_boundary.shp"

# Model parameters
PATCH_SIZE = 64  # Size of patches to extract (64x64)

# ============================================================
# Phase 1: Baseline Models Configuration
# ============================================================

# CNN Configuration - Optimized for GTX 1060 6GB + 64GB RAM
BATCH_SIZE = 24  # Optimized for Simple CNN (~1.2M params) with 6GB VRAM
NUM_WORKERS = 4  # Enough for cached dataset in RAM
PREFETCH_FACTOR = 2  # Prefetch 2 batches per worker
PIN_MEMORY = True  # Pin memory for faster GPU transfer
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Gradient accumulation to simulate larger batch size
ACCUMULATION_STEPS = 2  # Effective batch size = 24 * 2 = 48

# Learning rate and training
LEARNING_RATE = 1e-3  # Start learning rate (will use scheduler)
NUM_EPOCHS = 100  # Max epochs (early stopping will likely stop earlier)
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs
WEIGHT_DECAY = 1e-4  # L2 regularization

# Mixed precision training (saves ~40% VRAM)
USE_AMP = True  # Automatic Mixed Precision
TORCH_CUDNN_BENCHMARK = True  # Optimize convolution algorithms

# Random Forest Configuration
RF_N_ESTIMATORS = 500  # Number of trees
RF_MAX_DEPTH = 20  # Max depth of trees
RF_MIN_SAMPLES_SPLIT = 10  # Min samples to split
RF_N_JOBS = -1  # Use all CPU cores

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
S1_BANDS = ["VV", "VH"]  # VV and VH polarization
S1_NUM_BANDS = len(S1_BANDS)

# Total input channels: 2 time periods × (7 S2 bands + 2 S1 bands) = 18 channels
TOTAL_INPUT_CHANNELS = 2 * (S2_NUM_BANDS + S1_NUM_BANDS)

# Class labels
CLASS_NAMES = ["No Deforestation", "Deforestation"]
NUM_CLASSES = 2
