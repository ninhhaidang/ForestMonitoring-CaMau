"""
Deforestation Detection Pipeline
Cà Mau Province - Sentinel-1 & Sentinel-2

CNN-based approach for deforestation detection
"""

__version__ = '2.0.0'
__author__ = 'Deforestation Detection Team'

# Main modules
from .core.data_loader import DataLoader
from .core.feature_extraction import FeatureExtraction
from .core.visualization import Visualizer

# CNN
from .models.cnn.architecture import DeforestationCNN, create_model
from .models.cnn.trainer import CNNTrainer
from .models.cnn.predictor import RasterPredictor as CNNPredictor
from .models.cnn.patch_extractor import PatchExtractor
from .models.cnn.random_split import RandomKFoldWithFixedTest

__all__ = [
    # Core
    'DataLoader',
    'FeatureExtraction',
    'Visualizer',
    # CNN
    'DeforestationCNN',
    'create_model',
    'CNNTrainer',
    'CNNPredictor',
    'PatchExtractor',
    'RandomKFoldWithFixedTest',
]
