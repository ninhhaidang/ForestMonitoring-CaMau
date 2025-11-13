"""
Deforestation Detection Pipeline
Cà Mau Province - Sentinel-1 & Sentinel-2

Multi-model approach: Random Forest & CNN
"""

__version__ = '2.0.0'
__author__ = 'Deforestation Detection Team'

# Main modules
from .core.data_loader import DataLoader
from .core.feature_extraction import FeatureExtraction
from .core.evaluation import ModelEvaluator
from .core.visualization import Visualizer

# Random Forest
from .models.rf.trainer import RandomForestTrainer
from .models.rf.predictor import RasterPredictor as RFPredictor

# CNN
from .models.cnn.architecture import DeforestationCNN, create_model
from .models.cnn.trainer import CNNTrainer
from .models.cnn.predictor import RasterPredictor as CNNPredictor
from .models.cnn.patch_extractor import PatchExtractor
from .models.cnn.spatial_split import SpatialSplitter

__all__ = [
    # Core
    'DataLoader',
    'FeatureExtraction',
    'ModelEvaluator',
    'Visualizer',
    # Random Forest
    'RandomForestTrainer',
    'RFPredictor',
    # CNN
    'DeforestationCNN',
    'create_model',
    'CNNTrainer',
    'CNNPredictor',
    'PatchExtractor',
    'SpatialSplitter',
]
