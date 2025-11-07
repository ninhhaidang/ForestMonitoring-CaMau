"""
STEP 5: Train Random Forest Model
Train Random Forest classifier with configured parameters
"""

import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
from typing import Dict, Tuple

from config import (
    RF_PARAMS, MODELS_DIR, OUTPUT_FILES,
    LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """Class to train Random Forest model"""

    def __init__(self, params: Dict = None):
        """
        Initialize RandomForestTrainer

        Args:
            params: Random Forest parameters (default: from config)
        """
        self.params = params if params is not None else RF_PARAMS
        self.model = None
        self.training_time = None
        self.oob_score = None

    def create_model(self) -> RandomForestClassifier:
        """
        Create Random Forest model with configured parameters

        Returns:
            RandomForestClassifier instance
        """
        logger.info("\n" + "="*70)
        logger.info("CREATING RANDOM FOREST MODEL")
        logger.info("="*70)

        logger.info("\nModel parameters:")
        for param, value in self.params.items():
            logger.info(f"  - {param}: {value}")

        self.model = RandomForestClassifier(**self.params)

        logger.info("\n✓ Random Forest model created")
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> RandomForestClassifier:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Trained model
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 5: TRAIN RANDOM FOREST MODEL")
        logger.info("="*70)

        # Create model if not exists
        if self.model is None:
            self.create_model()

        # Training info
        logger.info(f"\nTraining data:")
        logger.info(f"  - Training samples: {len(X_train)}")
        logger.info(f"  - Features: {X_train.shape[1]}")
        logger.info(f"  - Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

        if X_val is not None:
            logger.info(f"\nValidation data:")
            logger.info(f"  - Validation samples: {len(X_val)}")
            logger.info(f"  - Class distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")

        # Train model
        logger.info(f"\n🌲 Training Random Forest with {self.params['n_estimators']} trees...")
        logger.info("This may take a few minutes...\n")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()

        self.training_time = end_time - start_time

        logger.info(f"\n✓ Training completed in {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")

        # Get OOB score if available
        if self.params.get('oob_score', False):
            self.oob_score = self.model.oob_score_
            logger.info(f"  ✓ Out-of-Bag Score: {self.oob_score:.4f} ({self.oob_score*100:.2f}%)")

        # Evaluate on training set
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        logger.info(f"  ✓ Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            logger.info(f"  ✓ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

        # Model info
        logger.info(f"\nModel information:")
        logger.info(f"  - Number of trees: {self.model.n_estimators}")
        logger.info(f"  - Number of features: {self.model.n_features_in_}")
        logger.info(f"  - Number of classes: {self.model.n_classes_}")
        logger.info(f"  - Classes: {self.model.classes_}")

        logger.info("\n" + "="*70)
        logger.info("✓ MODEL TRAINING COMPLETED")
        logger.info("="*70)

        return self.model

    def save_model(self, output_path=None):
        """
        Save trained model to disk

        Args:
            output_path: Path to save model (default: from config)
        """
        if self.model is None:
            logger.warning("No model to save")
            return

        if output_path is None:
            output_path = OUTPUT_FILES['trained_model']

        logger.info(f"\nSaving model to: {output_path}")

        # Create directory if not exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"  ✓ Model saved successfully")
        logger.info(f"  ✓ File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    def load_model(self, model_path=None):
        """
        Load trained model from disk

        Args:
            model_path: Path to load model (default: from config)

        Returns:
            Loaded model
        """
        if model_path is None:
            model_path = OUTPUT_FILES['trained_model']

        logger.info(f"\nLoading model from: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"  ✓ Model loaded successfully")
        logger.info(f"  - Number of trees: {self.model.n_estimators}")
        logger.info(f"  - Number of features: {self.model.n_features_in_}")

        return self.model

    def get_feature_importance(self, feature_names: list = None) -> Dict:
        """
        Get feature importance from trained model

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary with feature importance
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return {}

        importances = self.model.feature_importances_

        # Create dictionary
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        feature_importance = {
            'features': feature_names,
            'importance': importances.tolist(),
            'importance_normalized': (importances / importances.sum() * 100).tolist()
        }

        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        feature_importance['sorted_features'] = [feature_names[i] for i in sorted_indices]
        feature_importance['sorted_importance'] = importances[sorted_indices].tolist()
        feature_importance['sorted_importance_normalized'] = (importances[sorted_indices] / importances.sum() * 100).tolist()

        return feature_importance

    def get_training_summary(self) -> Dict:
        """
        Get summary of training process

        Returns:
            Dictionary with training summary
        """
        if self.model is None:
            return {}

        summary = {
            'n_estimators': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'classes': self.model.classes_.tolist(),
            'training_time_seconds': self.training_time,
            'training_time_minutes': self.training_time / 60 if self.training_time else None,
            'oob_score': self.oob_score,
            'parameters': self.params
        }

        return summary

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(X)


def main():
    """Main function to test model training"""
    logger.info("Testing Step 5: Train Random Forest")

    # Import previous steps
    from step1_2_setup_and_load_data import DataLoader
    from step3_feature_engineering import FeatureEngineering
    from step4_extract_training_data import TrainingDataExtractor

    # Load data
    loader = DataLoader()
    data = loader.load_all()

    # Engineer features
    engineer = FeatureEngineering()
    feature_stack, valid_mask = engineer.engineer_features(
        data['s2_before'],
        data['s2_after'],
        data['s1_before'],
        data['s1_after']
    )

    # Extract training data
    from config import FEATURE_NAMES
    extractor = TrainingDataExtractor()
    training_df = extractor.extract_pixel_values(
        feature_stack,
        data['ground_truth'],
        data['metadata']['s2_before']['transform']
    )

    X = training_df[FEATURE_NAMES].values
    y = training_df['label'].values

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = extractor.split_data(X, y)

    # Train model
    trainer = RandomForestTrainer()
    model = trainer.train(X_train, y_train, X_val, y_val)

    # Get feature importance
    feature_importance = trainer.get_feature_importance(FEATURE_NAMES)
    logger.info("\nTop 10 Most Important Features:")
    for i in range(min(10, len(feature_importance['sorted_features']))):
        feature = feature_importance['sorted_features'][i]
        importance = feature_importance['sorted_importance_normalized'][i]
        logger.info(f"  {i+1}. {feature:30s}: {importance:6.2f}%")

    # Save model
    trainer.save_model()

    # Get training summary
    summary = trainer.get_training_summary()
    logger.info(f"\nTraining Summary:")
    logger.info(f"  - Trees: {summary['n_estimators']}")
    logger.info(f"  - Features: {summary['n_features']}")
    logger.info(f"  - Training time: {summary['training_time_minutes']:.2f} minutes")
    logger.info(f"  - OOB Score: {summary['oob_score']:.4f}" if summary['oob_score'] else "  - OOB Score: N/A")

    return trainer, model


if __name__ == "__main__":
    trainer, model = main()
