"""
Ca Mau Deforestation Detection - Source Package

Modules:
    - utils: Utility functions for data loading and metadata checking
    - preprocessing: Data preprocessing and patch extraction
    - visualization: Plotting and visualization functions
    - models: CNN architectures for change detection
    - dataset: PyTorch Dataset classes
    - train: Training utilities
    - evaluate: Evaluation metrics
    - predict: Inference functions
"""

__version__ = "0.1.0"
__author__ = "Ninh Hải Đăng"

# Import key functions for easy access
from .utils import (
    load_tiff,
    check_tiff_metadata,
    get_tiff_stats,
    load_ground_truth
)

from .preprocessing import (
    normalize_band,
    handle_nan,
    extract_patch,
    create_patches_dataset
)

from .visualization import (
    plot_band,
    plot_band_comparison,
    plot_statistics,
    plot_confusion_matrix
)

__all__ = [
    # Utils
    'load_tiff',
    'check_tiff_metadata',
    'get_tiff_stats',
    'load_ground_truth',

    # Preprocessing
    'normalize_band',
    'handle_nan',
    'extract_patch',
    'create_patches_dataset',

    # Visualization
    'plot_band',
    'plot_band_comparison',
    'plot_statistics',
    'plot_confusion_matrix',
]
