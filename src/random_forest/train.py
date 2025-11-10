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

from common.config import (
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
    from common.feature_extraction import FeatureExtraction
    from step4_extract_training_data import TrainingDataExtractor

    # Load data
    loader = DataLoader()
    data = loader.load_all()

    # Extract features
    feature_extractor = FeatureExtraction()
    feature_stack, valid_mask = feature_extractor.extract_features(
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
"""
STEP 4: Extract Training Data
Extract feature values at ground truth point locations and split into train/val/test sets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import rasterio
from rasterio.transform import rowcol
import logging
from typing import Tuple, Dict

from common.config import (
    GROUND_TRUTH_CONFIG, TRAIN_TEST_SPLIT,
    FEATURE_NAMES, DATA_OUTPUT_DIR,
    LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class TrainingDataExtractor:
    """Class to extract training data from features at ground truth locations"""

    def __init__(self):
        """Initialize TrainingDataExtractor"""
        self.training_df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def extract_pixel_values(self, feature_stack: np.ndarray,
                            ground_truth: pd.DataFrame,
                            transform) -> pd.DataFrame:
        """
        Extract feature values at ground truth point locations

        Args:
            feature_stack: Feature array (n_features, height, width)
            ground_truth: DataFrame with ground truth points
            transform: Rasterio transform for coordinate conversion

        Returns:
            DataFrame with extracted features and labels
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: EXTRACT TRAINING DATA")
        logger.info("="*70)

        n_features = feature_stack.shape[0]
        n_points = len(ground_truth)

        logger.info(f"\nExtracting features at {n_points} ground truth points...")
        logger.info(f"  - Number of features: {n_features}")

        # Prepare data containers
        extracted_features = []
        extracted_labels = []
        valid_indices = []
        skipped_points = 0

        # Extract features for each point
        for idx, row in ground_truth.iterrows():
            # Get coordinates
            x = row[GROUND_TRUTH_CONFIG['x_column']]
            y = row[GROUND_TRUTH_CONFIG['y_column']]
            label = row[GROUND_TRUTH_CONFIG['label_column']]

            try:
                # Convert geographic coordinates to pixel coordinates
                py, px = rowcol(transform, x, y)

                # Check if pixel is within bounds
                if 0 <= py < feature_stack.shape[1] and 0 <= px < feature_stack.shape[2]:
                    # Extract feature values
                    pixel_features = feature_stack[:, py, px]

                    # Check for NoData values
                    if not np.isnan(pixel_features).any() and not (pixel_features == 0).all():
                        extracted_features.append(pixel_features)
                        extracted_labels.append(label)
                        valid_indices.append(idx)
                    else:
                        skipped_points += 1
                else:
                    skipped_points += 1

            except Exception as e:
                logger.warning(f"  ⚠ Error extracting point {idx}: {str(e)}")
                skipped_points += 1

        # Convert to arrays
        X = np.array(extracted_features)
        y = np.array(extracted_labels)

        logger.info(f"\n✓ Extraction completed:")
        logger.info(f"  - Valid points: {len(X)}")
        logger.info(f"  - Skipped points: {skipped_points}")
        logger.info(f"  - Feature matrix shape: {X.shape}")

        # Create DataFrame
        self.training_df = pd.DataFrame(X, columns=FEATURE_NAMES)
        self.training_df['label'] = y
        self.training_df['original_index'] = valid_indices

        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        logger.info(f"\n✓ Class distribution:")
        logger.info(f"  - Class 0 (No deforestation): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y)*100:.1f}%)")
        logger.info(f"  - Class 1 (Deforestation): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y)*100:.1f}%)")

        return self.training_df

    def check_data_quality(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Check quality of extracted training data

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Dictionary with quality metrics
        """
        logger.info("\nChecking data quality...")

        quality_metrics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_missing_values': np.isnan(X).sum(),
            'missing_percentage': np.isnan(X).sum() / X.size * 100,
            'n_infinite_values': np.isinf(X).sum(),
            'class_balance': dict(pd.Series(y).value_counts())
        }

        logger.info(f"  ✓ Total samples: {quality_metrics['n_samples']}")
        logger.info(f"  ✓ Total features: {quality_metrics['n_features']}")
        logger.info(f"  ✓ Missing values: {quality_metrics['n_missing_values']} ({quality_metrics['missing_percentage']:.2f}%)")
        logger.info(f"  ✓ Infinite values: {quality_metrics['n_infinite_values']}")

        # Check for features with zero variance
        zero_variance_features = []
        for i, feature_name in enumerate(FEATURE_NAMES):
            if X[:, i].std() == 0:
                zero_variance_features.append(feature_name)

        if zero_variance_features:
            logger.warning(f"  ⚠ Features with zero variance: {zero_variance_features}")
            quality_metrics['zero_variance_features'] = zero_variance_features
        else:
            logger.info(f"  ✓ No features with zero variance")

        return quality_metrics

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train, validation, and test sets

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("\n" + "="*70)
        logger.info("SPLITTING DATA")
        logger.info("="*70)

        train_size = TRAIN_TEST_SPLIT['train_size']
        val_size = TRAIN_TEST_SPLIT['val_size']
        test_size = TRAIN_TEST_SPLIT['test_size']
        random_state = TRAIN_TEST_SPLIT['random_state']

        logger.info(f"\nSplit configuration:")
        logger.info(f"  - Train: {train_size*100:.0f}%")
        logger.info(f"  - Validation: {val_size*100:.0f}%")
        logger.info(f"  - Test: {test_size*100:.0f}%")
        logger.info(f"  - Stratified: {TRAIN_TEST_SPLIT['stratify']}")
        logger.info(f"  - Random state: {random_state}")

        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test, idx_temp, self.test_indices = train_test_split(
            X, y, np.arange(len(X)),
            test_size=test_size,
            random_state=random_state,
            stratify=y if TRAIN_TEST_SPLIT['stratify'] else None
        )

        # Second split: separate train and validation
        val_size_adjusted = val_size / (train_size + val_size)
        self.X_train, self.X_val, self.y_train, self.y_val, self.train_indices, self.val_indices = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp if TRAIN_TEST_SPLIT['stratify'] else None
        )

        # Print split statistics
        logger.info(f"\n✓ Data split completed:")
        logger.info(f"\nTraining set:")
        logger.info(f"  - Samples: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.1f}%)")
        logger.info(f"  - Shape: {self.X_train.shape}")
        train_class_counts = pd.Series(self.y_train).value_counts()
        logger.info(f"  - Class 0: {train_class_counts.get(0, 0)} ({train_class_counts.get(0, 0)/len(self.y_train)*100:.1f}%)")
        logger.info(f"  - Class 1: {train_class_counts.get(1, 0)} ({train_class_counts.get(1, 0)/len(self.y_train)*100:.1f}%)")

        logger.info(f"\nValidation set:")
        logger.info(f"  - Samples: {len(self.X_val)} ({len(self.X_val)/len(X)*100:.1f}%)")
        logger.info(f"  - Shape: {self.X_val.shape}")
        val_class_counts = pd.Series(self.y_val).value_counts()
        logger.info(f"  - Class 0: {val_class_counts.get(0, 0)} ({val_class_counts.get(0, 0)/len(self.y_val)*100:.1f}%)")
        logger.info(f"  - Class 1: {val_class_counts.get(1, 0)} ({val_class_counts.get(1, 0)/len(self.y_val)*100:.1f}%)")

        logger.info(f"\nTest set:")
        logger.info(f"  - Samples: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.1f}%)")
        logger.info(f"  - Shape: {self.X_test.shape}")
        test_class_counts = pd.Series(self.y_test).value_counts()
        logger.info(f"  - Class 0: {test_class_counts.get(0, 0)} ({test_class_counts.get(0, 0)/len(self.y_test)*100:.1f}%)")
        logger.info(f"  - Class 1: {test_class_counts.get(1, 0)} ({test_class_counts.get(1, 0)/len(self.y_test)*100:.1f}%)")

        logger.info("="*70)

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def save_training_data(self, output_path=None):
        """
        Save training data to CSV

        Args:
            output_path: Path to save CSV file (default: from config)
        """
        if self.training_df is None:
            logger.warning("No training data to save")
            return

        if output_path is None:
            output_path = DATA_OUTPUT_DIR / 'training_data.csv'

        logger.info(f"\nSaving training data to: {output_path}")
        self.training_df.to_csv(output_path, index=False)
        logger.info(f"  ✓ Saved {len(self.training_df)} samples")

    def get_summary(self) -> Dict:
        """
        Get summary of training data extraction

        Returns:
            Dictionary with summary information
        """
        if self.training_df is None:
            return {}

        summary = {
            'total_samples': len(self.training_df),
            'n_features': len(FEATURE_NAMES),
            'feature_names': FEATURE_NAMES,
            'class_distribution': dict(self.training_df['label'].value_counts()),
        }

        if self.X_train is not None:
            summary['train_samples'] = len(self.X_train)
            summary['val_samples'] = len(self.X_val)
            summary['test_samples'] = len(self.X_test)
            summary['train_class_dist'] = dict(pd.Series(self.y_train).value_counts())
            summary['val_class_dist'] = dict(pd.Series(self.y_val).value_counts())
            summary['test_class_dist'] = dict(pd.Series(self.y_test).value_counts())

        return summary


def main():
    """Main function to test training data extraction"""
    logger.info("Testing Step 4: Extract Training Data")

    # Import previous steps
    from step1_2_setup_and_load_data import DataLoader
    from common.feature_extraction import FeatureExtraction

    # Load data
    loader = DataLoader()
    data = loader.load_all()

    # Extract features
    feature_extractor = FeatureExtraction()
    feature_stack, valid_mask = feature_extractor.extract_features(
        data['s2_before'],
        data['s2_after'],
        data['s1_before'],
        data['s1_after']
    )

    # Extract training data
    extractor = TrainingDataExtractor()
    training_df = extractor.extract_pixel_values(
        feature_stack,
        data['ground_truth'],
        data['metadata']['s2_before']['transform']
    )

    # Get features and labels
    X = training_df[FEATURE_NAMES].values
    y = training_df['label'].values

    # Check data quality
    quality_metrics = extractor.check_data_quality(X, y)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = extractor.split_data(X, y)

    # Save training data
    extractor.save_training_data()

    # Print summary
    summary = extractor.get_summary()
    logger.info(f"\nExtraction Summary:")
    logger.info(f"  - Total samples: {summary['total_samples']}")
    logger.info(f"  - Train samples: {summary.get('train_samples', 'N/A')}")
    logger.info(f"  - Val samples: {summary.get('val_samples', 'N/A')}")
    logger.info(f"  - Test samples: {summary.get('test_samples', 'N/A')}")

    return extractor


if __name__ == "__main__":
    extractor = main()
