"""
Configuration file for Random Forest Deforestation Detection
Cà Mau Province - Sentinel-1 & Sentinel-2
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
# config.py is now at: src/config.py
# So PROJECT_ROOT is 1 level up from src/
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ground Truth
GROUND_TRUTH_DIR = RAW_DATA_DIR / "samples"
GROUND_TRUTH_CSV = GROUND_TRUTH_DIR / "4labels.csv"

# Sentinel-2 (Optical)
S2_DIR = RAW_DATA_DIR / "sentinel-2"
S2_BEFORE = S2_DIR / "S2_2024_01_30.tif"
S2_AFTER = S2_DIR / "S2_2025_02_28.tif"

# Sentinel-1 (SAR)
S1_DIR = RAW_DATA_DIR / "sentinel-1"
S1_BEFORE = S1_DIR / "S1_2024_02_04_matched_S2_2024_01_30.tif"
S1_AFTER = S1_DIR / "S1_2025_02_22_matched_S2_2025_02_28.tif"

# Boundary
BOUNDARY_DIR = RAW_DATA_DIR / "boundary"
BOUNDARY_SHP = BOUNDARY_DIR / "forest_boundary.shp"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
RASTERS_DIR = RESULTS_DIR / "rasters"
MODELS_DIR = RESULTS_DIR / "models"
DATA_OUTPUT_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Sentinel-2 bands configuration (7 bands)
S2_BANDS = {
    'names': ['B4', 'B8', 'B11', 'B12', 'NDVI', 'NBR', 'NDMI'],
    'count': 7,
    'resolution': 10  # meters
}

# Sentinel-1 bands configuration (2 bands)
S1_BANDS = {
    'names': ['VV', 'VH'],
    'count': 2,
    'resolution': 10  # meters
}

# Total features
TOTAL_FEATURES = (S2_BANDS['count'] * 3) + (S1_BANDS['count'] * 3)  # 27 features
# S2: 7 before + 7 after + 7 delta = 21
# S1: 2 before + 2 after + 2 delta = 6

# Feature names (27 total)
FEATURE_NAMES = (
    # Sentinel-2 Before (7)
    [f"S2_before_{band}" for band in S2_BANDS['names']] +
    # Sentinel-2 After (7)
    [f"S2_after_{band}" for band in S2_BANDS['names']] +
    # Sentinel-2 Delta (7)
    [f"S2_delta_{band}" for band in S2_BANDS['names']] +
    # Sentinel-1 Before (2)
    [f"S1_before_{band}" for band in S1_BANDS['names']] +
    # Sentinel-1 After (2)
    [f"S1_after_{band}" for band in S1_BANDS['names']] +
    # Sentinel-1 Delta (2)
    [f"S1_delta_{band}" for band in S1_BANDS['names']]
)

# Ground Truth configuration
# Note: total_points and class distribution are read dynamically from CSV
GROUND_TRUTH_CONFIG = {
    'label_column': 'label',
    'x_column': 'x',
    'y_column': 'y',
    'id_column': 'id',
    'crs': 'EPSG:32648'  # UTM Zone 48N
}

# ============================================================================
# RANDOM FOREST CONFIGURATION
# ============================================================================

RF_PARAMS = {
    'n_estimators': 100,           # Number of trees
    'max_features': 'sqrt',        # sqrt(27) ≈ 5 features per split
    'max_depth': None,             # Unlimited depth
    'min_samples_split': 2,        # Minimum samples to split a node
    'min_samples_leaf': 1,         # Minimum samples in leaf node
    'bootstrap': True,             # Bootstrap samples
    'oob_score': True,             # Out-of-Bag score
    'class_weight': 'balanced',    # Handle imbalanced data
    'n_jobs': -1,                  # Use all CPU cores
    'random_state': 42,            # Reproducibility
    'verbose': 1                   # Show progress
}

# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================

TRAIN_TEST_SPLIT = {
    'train_size': 0.70,      # 70% (~900 points)
    'val_size': 0.15,        # 15% (~193 points)
    'test_size': 0.15,       # 15% (~192 points)
    'random_state': 42,      # Reproducibility
    'stratify': True         # Maintain class distribution
}

# ============================================================================
# CROSS VALIDATION CONFIGURATION
# ============================================================================

CV_CONFIG = {
    'n_splits': 5,           # 5-Fold Cross Validation
    'shuffle': True,         # Shuffle before splitting
    'random_state': 42       # Reproducibility
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VIZ_CONFIG = {
    'dpi': 300,                    # Plot resolution
    'figsize': (12, 8),            # Default figure size
    'cmap_probability': 'RdYlGn_r',  # Colormap for probability
    'cmap_binary': 'RdYlGn',       # Colormap for binary
    'save_format': 'png'           # Save format
}

# ============================================================================
# DEEP LEARNING CONFIGURATION
# ============================================================================

DL_CONFIG = {
    # Model architecture
    'model_type': 'standard',          # 'standard' or 'deeper'
    'patch_size': 3,                    # 3x3 patches
    'n_features': TOTAL_FEATURES,       # 27 features
    'n_classes': 4,                     # Multi-class: 0=Forest Stable (1→1), 1=Deforestation (1→0), 2=Non-forest (0→0), 3=Reforestation (0→1)
    'dropout_rate': 0.5,                # Dropout for regularization

    # Training parameters
    'epochs': 50,                       # Maximum epochs
    'batch_size': 32,                   # Batch size
    'learning_rate': 0.001,             # Initial learning rate
    'weight_decay': 1e-4,               # L2 regularization
    'early_stopping_patience': 10,      # Early stopping patience

    # Data split (spatial-aware)
    'cluster_distance': 50.0,           # Clustering threshold (meters)
    'train_size': 0.70,                 # 70% for training
    'val_size': 0.15,                   # 15% for validation
    'test_size': 0.15,                  # 15% for testing
    'stratify_by_class': True,          # Maintain class balance

    # Normalization
    'normalize_method': 'standardize',  # 'standardize' or 'minmax'

    # Device
    'device': 'cuda',                   # 'cuda' or 'cpu'

    # Prediction
    'pred_batch_size': 8000,            # Batch size for full raster prediction
    'pred_stride': 1,                   # Stride for sliding window (1 = dense, 2 = balanced)

    # Class weights (for imbalanced data)
    'use_class_weights': True,
    'class_weights': [1.0, 1.0, 1.0, 1.0]    # Will be computed from data if needed (4 classes)
}

# Output files for Deep Learning
DL_OUTPUT_FILES = {
    # Rasters
    'classification_raster': RASTERS_DIR / 'cnn_classification.tif',
    'probability_raster': RASTERS_DIR / 'cnn_probability.tif',

    # Models
    'trained_model': MODELS_DIR / 'cnn_model.pth',

    # Data
    'training_patches': DATA_OUTPUT_DIR / 'cnn_training_patches.npz',
    'evaluation_metrics': DATA_OUTPUT_DIR / 'cnn_evaluation_metrics.json',
    'training_history': DATA_OUTPUT_DIR / 'cnn_training_history.json',

    # Plots
    'training_curves': PLOTS_DIR / 'cnn_training_curves.png',
    'confusion_matrix': PLOTS_DIR / 'cnn_confusion_matrix.png',
    'roc_curve': PLOTS_DIR / 'cnn_roc_curve.png',
    'classification_map': PLOTS_DIR / 'cnn_classification_map.png',
    'comparison_rf_cnn': PLOTS_DIR / 'comparison_rf_vs_cnn.png'
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# ============================================================================
# OUTPUT FILES
# ============================================================================

OUTPUT_FILES = {
    # Rasters
    'classification_raster': RASTERS_DIR / 'rf_classification.tif',
    'probability_raster': RASTERS_DIR / 'rf_probability.tif',

    # Models
    'trained_model': MODELS_DIR / 'rf_model.pkl',

    # Data
    'training_data': DATA_OUTPUT_DIR / 'rf_training_data.csv',
    'feature_importance': DATA_OUTPUT_DIR / 'rf_feature_importance.csv',
    'evaluation_metrics': DATA_OUTPUT_DIR / 'rf_evaluation_metrics.json',

    # Plots
    'confusion_matrices': PLOTS_DIR / 'rf_confusion_matrices.png',
    'roc_curve': PLOTS_DIR / 'rf_roc_curve.png',
    'feature_importance_plot': PLOTS_DIR / 'rf_feature_importance.png',
    'classification_maps': PLOTS_DIR / 'rf_classification_maps.png',
    'cv_scores': PLOTS_DIR / 'rf_cv_scores.png'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_directories():
    """Create all output directories if they don't exist"""
    directories = [
        RESULTS_DIR,
        RASTERS_DIR,
        MODELS_DIR,
        DATA_OUTPUT_DIR,
        PLOTS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("[OK] Output directories created successfully")

def verify_input_files():
    """Verify that all required input files exist"""
    required_files = {
        'Ground Truth CSV': GROUND_TRUTH_CSV,
        'Sentinel-2 Before': S2_BEFORE,
        'Sentinel-2 After': S2_AFTER,
        'Sentinel-1 Before': S1_BEFORE,
        'Sentinel-1 After': S1_AFTER,
        'Boundary Shapefile': BOUNDARY_SHP
    }

    missing_files = []
    for name, path in required_files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("[WARNING] Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("[OK] All required input files exist")
        return True

def print_config_summary(ground_truth_df=None):
    """
    Print configuration summary

    Args:
        ground_truth_df: Optional DataFrame with ground truth data.
                        If provided, will show actual point counts from data.
    """
    print("\n" + "="*70)
    print("RANDOM FOREST DEFORESTATION DETECTION - CONFIGURATION")
    print("="*70)
    print(f"\nDATA CONFIGURATION:")
    print(f"  - Total Features: {TOTAL_FEATURES}")
    print(f"  - Sentinel-2 Bands: {S2_BANDS['count']} bands")
    print(f"  - Sentinel-1 Bands: {S1_BANDS['count']} bands")

    # Read ground truth stats dynamically if DataFrame is provided
    if ground_truth_df is not None:
        total_points = len(ground_truth_df)
        class_counts = ground_truth_df[GROUND_TRUTH_CONFIG['label_column']].value_counts().sort_index()
        print(f"  - Ground Truth Points: {total_points}")
        if 0 in class_counts.index:
            print(f"  - Class 0 (No Deforestation): {class_counts[0]}")
        if 1 in class_counts.index:
            print(f"  - Class 1 (Deforestation): {class_counts[1]}")
    else:
        print(f"  - Ground Truth Points: (load data to see stats)")

    print(f"\nRANDOM FOREST CONFIGURATION:")
    print(f"  - Number of Trees: {RF_PARAMS['n_estimators']}")
    print(f"  - Max Features: {RF_PARAMS['max_features']}")
    print(f"  - Class Weight: {RF_PARAMS['class_weight']}")
    print(f"  - OOB Score: {RF_PARAMS['oob_score']}")

    print(f"\nDATA SPLIT:")
    print(f"  - Train: {TRAIN_TEST_SPLIT['train_size']*100:.0f}%")
    print(f"  - Validation: {TRAIN_TEST_SPLIT['val_size']*100:.0f}%")
    print(f"  - Test: {TRAIN_TEST_SPLIT['test_size']*100:.0f}%")

    print(f"\nCROSS VALIDATION:")
    print(f"  - K-Folds: {CV_CONFIG['n_splits']}")

    print(f"\nOUTPUT DIRECTORIES:")
    print(f"  - Rasters: {RASTERS_DIR}")
    print(f"  - Models: {MODELS_DIR}")
    print(f"  - Data: {DATA_OUTPUT_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_config_summary()
    create_output_directories()
    verify_input_files()
