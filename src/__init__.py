"""
Random Forest Deforestation Detection Pipeline
Cà Mau Province - Sentinel-1 & Sentinel-2

This package contains the complete pipeline for deforestation detection using Random Forest.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# Import main classes for easy access
from .step1_2_setup_and_load_data import DataLoader
from .step3_feature_engineering import FeatureEngineering
from .step4_extract_training_data import TrainingDataExtractor
from .step5_train_random_forest import RandomForestTrainer
from .step6_model_evaluation import ModelEvaluator
from .step7_predict_full_raster import RasterPredictor
from .step8_vectorization import Vectorizer
from .step9_visualization import Visualizer

__all__ = [
    'DataLoader',
    'FeatureEngineering',
    'TrainingDataExtractor',
    'RandomForestTrainer',
    'ModelEvaluator',
    'RasterPredictor',
    'Vectorizer',
    'Visualizer'
]
