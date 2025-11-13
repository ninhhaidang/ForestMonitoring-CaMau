"""
STEP 7: Predict on Full Raster
Apply trained model to predict deforestation across entire study area
Generate binary classification and probability maps
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
import logging
import time
from typing import Tuple

from config import (
    OUTPUT_FILES, LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class RasterPredictor:
    """Class to predict on full raster images"""

    def __init__(self, model):
        """
        Initialize RasterPredictor

        Args:
            model: Trained model
        """
        self.model = model
        self.multiclass_map = None  # 3-class map (0, 1, 2)
        self.classification_map = None  # Binary map (0, 1)
        self.probability_map = None  # Binary probability

    def predict_raster(self, feature_stack: np.ndarray,
                      valid_mask: np.ndarray,
                      batch_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on full raster using batching for memory efficiency

        Args:
            feature_stack: Feature array (n_features, height, width)
            valid_mask: Boolean mask of valid pixels
            batch_size: Number of pixels to predict at once

        Returns:
            Tuple of (classification_map, probability_map)
            - classification_map: (height, width) BINARY map (0=No Deforestation, 1=Deforestation)
            - probability_map: (height, width) BINARY probability of deforestation
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 7: PREDICT ON FULL RASTER")
        logger.info("="*70)

        n_features, height, width = feature_stack.shape
        total_pixels = height * width
        valid_pixels = np.sum(valid_mask)

        logger.info(f"\nRaster information:")
        logger.info(f"  - Dimensions: {height} x {width}")
        logger.info(f"  - Total pixels: {total_pixels:,}")
        logger.info(f"  - Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")
        logger.info(f"  - Features: {n_features}")
        logger.info(f"  - Batch size: {batch_size:,}")

        # Reshape feature stack to (H*W, n_features)
        logger.info("\nReshaping feature stack...")
        features_2d = feature_stack.reshape(n_features, -1).T  # (H*W, n_features)
        logger.info(f"  ✓ Reshaped to: {features_2d.shape}")

        # Initialize output arrays
        multiclass_flat = np.full(total_pixels, 255, dtype=np.uint8)  # 255 for NoData
        classification_flat = np.full(total_pixels, 255, dtype=np.uint8)  # 255 for NoData (binary)
        probability_flat = np.full(total_pixels, -9999.0, dtype=np.float32)  # -9999 for NoData

        # Get indices of valid pixels
        valid_indices = np.where(valid_mask.ravel())[0]
        logger.info(f"\n  - Valid indices: {len(valid_indices):,}")

        # Predict in batches
        logger.info(f"\nPredicting in batches of {batch_size:,}...")
        n_batches = int(np.ceil(len(valid_indices) / batch_size))
        logger.info(f"  - Total batches: {n_batches}")

        start_time = time.time()

        for i in range(n_batches):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(valid_indices))
            batch_indices = valid_indices[start_idx:end_idx]

            # Get batch features
            batch_features = features_2d[batch_indices]

            # Multiclass predictions (3 classes: 0, 1, 2)
            multiclass_pred = self.model.predict(batch_features)
            multiclass_proba = self.model.predict_proba(batch_features)

            # Binary predictions: Class 1 (Deforestation) vs Rest (Class 0 + Class 2)
            # Binary probability = P(Class 1)
            binary_proba = multiclass_proba[:, 1]  # Probability of deforestation
            binary_pred = (binary_proba > 0.5).astype(np.uint8)  # Threshold at 0.5

            # Store predictions
            multiclass_flat[batch_indices] = multiclass_pred
            classification_flat[batch_indices] = binary_pred
            probability_flat[batch_indices] = binary_proba

            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                progress = (i + 1) / n_batches * 100
                logger.info(f"  - Batch {i+1}/{n_batches} ({progress:.1f}%)")

        end_time = time.time()
        prediction_time = end_time - start_time

        logger.info(f"\n✓ Prediction completed in {prediction_time:.2f} seconds ({prediction_time/60:.2f} minutes)")

        # Reshape back to 2D
        logger.info("\nReshaping predictions back to 2D...")
        self.multiclass_map = multiclass_flat.reshape(height, width)
        self.classification_map = classification_flat.reshape(height, width)
        self.probability_map = probability_flat.reshape(height, width)

        # Statistics
        logger.info("\nPrediction statistics:")

        # Multiclass map
        mc_unique, mc_counts = np.unique(
            self.multiclass_map[valid_mask],
            return_counts=True
        )
        logger.info("\n  Multiclass Predictions:")
        for cls, count in zip(mc_unique, mc_counts):
            class_name = {0: 'Forest Stable', 1: 'Deforestation', 2: 'Non-forest'}.get(cls, f'Class {cls}')
            logger.info(f"    Class {cls} ({class_name}): {count:,} pixels ({count/valid_pixels*100:.2f}%)")

        # Binary classification map
        binary_unique, binary_counts = np.unique(
            self.classification_map[valid_mask],
            return_counts=True
        )
        logger.info("\n  Binary Classification:")
        for cls, count in zip(binary_unique, binary_counts):
            class_name = {0: 'No Deforestation', 1: 'Deforestation'}.get(cls, f'Class {cls}')
            logger.info(f"    Class {cls} ({class_name}): {count:,} pixels ({count/valid_pixels*100:.2f}%)")

        # Probability map
        prob_values = self.probability_map[valid_mask]
        logger.info("\n  Probability Map:")
        logger.info(f"    Min:  {prob_values.min():.4f}")
        logger.info(f"    Max:  {prob_values.max():.4f}")
        logger.info(f"    Mean: {prob_values.mean():.4f}")
        logger.info(f"    Std:  {prob_values.std():.4f}")

        logger.info("\n" + "="*70)
        logger.info("✓ RASTER PREDICTION COMPLETED")
        logger.info("="*70)

        return self.classification_map, self.probability_map

    def save_raster(self, output_path, data: np.ndarray,
                   metadata: dict, dtype=None, nodata=None):
        """
        Save raster to file

        Args:
            output_path: Path to save raster
            data: Raster data (2D array)
            metadata: Rasterio metadata
            dtype: Data type for output
            nodata: NoData value
        """
        logger.info(f"\nSaving raster to: {output_path}")

        # Ensure data is 2D
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)  # Add band dimension

        # Update metadata
        output_meta = metadata.copy()
        output_meta.update({
            'count': 1,
            'dtype': dtype if dtype else data.dtype,
            'nodata': nodata
        })

        # Save raster
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(data)

        logger.info(f"  ✓ Raster saved successfully")
        logger.info(f"  - Shape: {data.shape}")
        logger.info(f"  - Dtype: {output_meta['dtype']}")
        logger.info(f"  - NoData: {output_meta['nodata']}")

    def save_classification_raster(self, metadata: dict, output_path=None, save_multiclass=False, multiclass_path=None):
        """
        Save binary classification raster (and optionally multiclass)

        Args:
            metadata: Rasterio metadata
            output_path: Path to save binary map (default: from config)
            save_multiclass: Whether to save multiclass map
            multiclass_path: Path to save multiclass map
        """
        if self.classification_map is None:
            logger.warning("No classification map to save")
            return

        if output_path is None:
            output_path = OUTPUT_FILES['classification_raster']

        logger.info("\n" + "="*70)
        logger.info("SAVING CLASSIFICATION RASTERS")
        logger.info("="*70)

        # Save binary classification
        logger.info("\nBinary classification:")
        self.save_raster(
            output_path,
            self.classification_map,
            metadata,
            dtype='uint8',
            nodata=255
        )

        # Save multiclass if requested
        if save_multiclass and self.multiclass_map is not None:
            if multiclass_path is None:
                multiclass_path = str(output_path).replace('.tif', '_multiclass.tif')
            logger.info("\nMulticlass predictions:")
            self.save_raster(
                multiclass_path,
                self.multiclass_map,
                metadata,
                dtype='uint8',
                nodata=255
            )

        logger.info("="*70)

    def save_probability_raster(self, metadata: dict, output_path=None):
        """
        Save probability raster

        Args:
            metadata: Rasterio metadata
            output_path: Path to save (default: from config)
        """
        if self.probability_map is None:
            logger.warning("No probability map to save")
            return

        if output_path is None:
            output_path = OUTPUT_FILES['probability_raster']

        logger.info("\n" + "="*70)
        logger.info("SAVING PROBABILITY RASTER")
        logger.info("="*70)

        self.save_raster(
            output_path,
            self.probability_map,
            metadata,
            dtype='float32',
            nodata=-9999.0
        )

        logger.info("="*70)

    def save_all_rasters(self, metadata: dict):
        """
        Save both classification and probability rasters

        Args:
            metadata: Rasterio metadata
        """
        self.save_classification_raster(metadata)
        self.save_probability_raster(metadata)

        logger.info("\n✓ All rasters saved successfully")

    def get_prediction_summary(self) -> dict:
        """
        Get summary of predictions

        Returns:
            Dictionary with prediction summary
        """
        if self.classification_map is None:
            return {}

        # Get valid pixels (not NoData)
        valid_mask = self.classification_map != 255

        # Classification statistics
        unique_classes, class_counts = np.unique(
            self.classification_map[valid_mask],
            return_counts=True
        )

        class_stats = {}
        for cls, count in zip(unique_classes, class_counts):
            class_stats[int(cls)] = {
                'count': int(count),
                'percentage': float(count / np.sum(valid_mask) * 100)
            }

        # Probability statistics
        prob_values = self.probability_map[valid_mask]
        prob_stats = {
            'min': float(prob_values.min()),
            'max': float(prob_values.max()),
            'mean': float(prob_values.mean()),
            'std': float(prob_values.std()),
            'median': float(np.median(prob_values))
        }

        summary = {
            'total_pixels': int(self.classification_map.size),
            'valid_pixels': int(np.sum(valid_mask)),
            'classification_stats': class_stats,
            'probability_stats': prob_stats
        }

        return summary


def main():
    """Main function to test raster prediction"""
    logger.info("Testing Step 7: Predict Full Raster")

    # Import previous steps
    from step1_2_setup_and_load_data import DataLoader
    from core.feature_extraction import FeatureExtraction
    from step4_extract_training_data import TrainingDataExtractor
    from step5_train_random_forest import RandomForestTrainer
    from config import FEATURE_NAMES

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

    # Load or train model
    logger.info("\nLoading/training model...")
    trainer = RandomForestTrainer()

    # Check if model exists
    if OUTPUT_FILES['trained_model'].exists():
        logger.info("  - Loading existing model...")
        model = trainer.load_model()
    else:
        logger.info("  - Training new model...")
        # Extract training data
        extractor = TrainingDataExtractor()
        training_df = extractor.extract_pixel_values(
            feature_stack,
            data['ground_truth'],
            data['metadata']['s2_before']['transform']
        )
        X = training_df[FEATURE_NAMES].values
        y = training_df['label'].values
        X_train, X_val, X_test, y_train, y_val, y_test = extractor.split_data(X, y)

        # Train
        model = trainer.train(X_train, y_train, X_val, y_val)
        trainer.save_model()

    # Predict on full raster
    predictor = RasterPredictor(model)
    classification_map, probability_map = predictor.predict_raster(
        feature_stack,
        valid_mask,
        batch_size=10000
    )

    # Save rasters
    predictor.save_all_rasters(data['metadata']['s2_before'])

    # Print summary
    summary = predictor.get_prediction_summary()
    logger.info("\nPrediction Summary:")
    logger.info(f"  - Total pixels: {summary['total_pixels']:,}")
    logger.info(f"  - Valid pixels: {summary['valid_pixels']:,}")
    logger.info("\n  Classification:")
    for cls, stats in summary['classification_stats'].items():
        logger.info(f"    Class {cls}: {stats['count']:,} pixels ({stats['percentage']:.2f}%)")
    logger.info("\n  Probability:")
    logger.info(f"    Mean: {summary['probability_stats']['mean']:.4f}")
    logger.info(f"    Std:  {summary['probability_stats']['std']:.4f}")

    return predictor, classification_map, probability_map


if __name__ == "__main__":
    predictor, classification_map, probability_map = main()
