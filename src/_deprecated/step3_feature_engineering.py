"""
STEP 3: Feature Engineering
Create 27 features from Sentinel-1 and Sentinel-2 data:
- S2: 7 before + 7 after + 7 delta = 21 features
- S1: 2 before + 2 after + 2 delta = 6 features
"""

import numpy as np
import logging
from typing import Tuple, Dict

from config import (
    S1_BANDS, S2_BANDS, FEATURE_NAMES,
    LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Class to handle feature engineering from satellite imagery"""

    def __init__(self):
        """Initialize FeatureEngineering"""
        self.feature_stack = None
        self.feature_names = FEATURE_NAMES
        self.n_features = len(FEATURE_NAMES)
        self.valid_mask = None

    @staticmethod
    def calculate_delta(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Calculate temporal change (delta) between before and after

        Args:
            before: Before image array (bands, height, width)
            after: After image array (bands, height, width)

        Returns:
            Delta array (after - before)
        """
        return after - before

    @staticmethod
    def create_valid_mask(s2_before: np.ndarray, s2_after: np.ndarray,
                          s1_before: np.ndarray, s1_after: np.ndarray,
                          nodata_value: float = 0) -> np.ndarray:
        """
        Create mask for valid pixels (no NoData in any band/time)

        Args:
            s2_before: Sentinel-2 before array
            s2_after: Sentinel-2 after array
            s1_before: Sentinel-1 before array
            s1_after: Sentinel-1 after array
            nodata_value: NoData value to mask

        Returns:
            Boolean mask (True = valid, False = NoData)
        """
        # Check for NoData in each dataset
        valid_s2_before = ~np.isnan(s2_before).any(axis=0)
        valid_s2_after = ~np.isnan(s2_after).any(axis=0)
        valid_s1_before = ~np.isnan(s1_before).any(axis=0)
        valid_s1_after = ~np.isnan(s1_after).any(axis=0)

        # Also check for zero values if used as NoData
        if nodata_value == 0:
            valid_s2_before &= (s2_before != 0).all(axis=0)
            valid_s2_after &= (s2_after != 0).all(axis=0)
            valid_s1_before &= (s1_before != 0).all(axis=0)
            valid_s1_after &= (s1_after != 0).all(axis=0)

        # Combine all masks
        valid_mask = valid_s2_before & valid_s2_after & valid_s1_before & valid_s1_after

        return valid_mask

    def engineer_features(self, s2_before: np.ndarray, s2_after: np.ndarray,
                          s1_before: np.ndarray, s1_after: np.ndarray,
                          nodata_value: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer all 27 features from satellite data

        Args:
            s2_before: Sentinel-2 before (7, H, W)
            s2_after: Sentinel-2 after (7, H, W)
            s1_before: Sentinel-1 before (2, H, W)
            s1_after: Sentinel-1 after (2, H, W)
            nodata_value: NoData value

        Returns:
            Tuple of (feature_stack, valid_mask)
            - feature_stack: (27, H, W) array of all features
            - valid_mask: (H, W) boolean mask
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("="*70)

        # Validate input shapes
        assert s2_before.shape[0] == S2_BANDS['count'], \
            f"Expected {S2_BANDS['count']} S2 bands, got {s2_before.shape[0]}"
        assert s1_before.shape[0] == S1_BANDS['count'], \
            f"Expected {S1_BANDS['count']} S1 bands, got {s1_before.shape[0]}"

        height, width = s2_before.shape[1], s2_before.shape[2]
        logger.info(f"\nInput dimensions: {height} x {width}")

        # Create valid mask
        logger.info("\nCreating valid pixel mask...")
        self.valid_mask = self.create_valid_mask(
            s2_before, s2_after, s1_before, s1_after, nodata_value
        )
        valid_pixels = np.sum(self.valid_mask)
        total_pixels = self.valid_mask.size
        logger.info(f"  ✓ Valid pixels: {valid_pixels:,} / {total_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")

        # Initialize feature stack
        feature_list = []

        # ============================================================
        # SENTINEL-2 FEATURES (21 features)
        # ============================================================
        logger.info("\nEngineering Sentinel-2 features...")

        # S2 Before (7 features)
        logger.info("  - Adding S2 Before bands (7 features)")
        for i, band_name in enumerate(S2_BANDS['names']):
            feature_list.append(s2_before[i])

        # S2 After (7 features)
        logger.info("  - Adding S2 After bands (7 features)")
        for i, band_name in enumerate(S2_BANDS['names']):
            feature_list.append(s2_after[i])

        # S2 Delta (7 features)
        logger.info("  - Calculating S2 Delta (7 features)")
        s2_delta = self.calculate_delta(s2_before, s2_after)
        for i, band_name in enumerate(S2_BANDS['names']):
            feature_list.append(s2_delta[i])

        logger.info(f"  ✓ Total S2 features: {len(feature_list)}")

        # ============================================================
        # SENTINEL-1 FEATURES (6 features)
        # ============================================================
        logger.info("\nEngineering Sentinel-1 features...")

        # S1 Before (2 features)
        logger.info("  - Adding S1 Before bands (2 features)")
        for i, band_name in enumerate(S1_BANDS['names']):
            feature_list.append(s1_before[i])

        # S1 After (2 features)
        logger.info("  - Adding S1 After bands (2 features)")
        for i, band_name in enumerate(S1_BANDS['names']):
            feature_list.append(s1_after[i])

        # S1 Delta (2 features)
        logger.info("  - Calculating S1 Delta (2 features)")
        s1_delta = self.calculate_delta(s1_before, s1_after)
        for i, band_name in enumerate(S1_BANDS['names']):
            feature_list.append(s1_delta[i])

        logger.info(f"  ✓ Total S1 features: 6")

        # ============================================================
        # STACK ALL FEATURES
        # ============================================================
        logger.info("\nStacking all features...")
        self.feature_stack = np.stack(feature_list, axis=0)  # Shape: (27, H, W)

        logger.info(f"  ✓ Feature stack shape: {self.feature_stack.shape}")
        logger.info(f"  ✓ Total features: {self.feature_stack.shape[0]}")
        assert self.feature_stack.shape[0] == self.n_features, \
            f"Expected {self.n_features} features, got {self.feature_stack.shape[0]}"

        # ============================================================
        # FEATURE STATISTICS
        # ============================================================
        logger.info("\nFeature statistics:")
        for i, feature_name in enumerate(self.feature_names):
            feature_data = self.feature_stack[i][self.valid_mask]
            if len(feature_data) > 0:
                logger.info(f"  {feature_name:30s}: "
                           f"min={feature_data.min():8.3f}, "
                           f"max={feature_data.max():8.3f}, "
                           f"mean={feature_data.mean():8.3f}, "
                           f"std={feature_data.std():8.3f}")

        logger.info("\n" + "="*70)
        logger.info("✓ FEATURE ENGINEERING COMPLETED")
        logger.info("="*70)
        logger.info(f"  - Total features: {self.n_features}")
        logger.info(f"  - Feature shape: {self.feature_stack.shape}")
        logger.info(f"  - Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")
        logger.info("="*70)

        return self.feature_stack, self.valid_mask

    def get_feature_summary(self) -> Dict:
        """
        Get summary of engineered features

        Returns:
            Dictionary with feature summary
        """
        if self.feature_stack is None:
            logger.warning("Features not engineered yet")
            return {}

        summary = {
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'feature_shape': self.feature_stack.shape,
            'valid_pixels': np.sum(self.valid_mask),
            'total_pixels': self.valid_mask.size,
            'valid_percentage': np.sum(self.valid_mask) / self.valid_mask.size * 100
        }

        # Add statistics for each feature
        feature_stats = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_data = self.feature_stack[i][self.valid_mask]
            if len(feature_data) > 0:
                feature_stats[feature_name] = {
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std())
                }

        summary['feature_stats'] = feature_stats

        return summary

    def reshape_for_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape feature stack for model prediction

        Returns:
            Tuple of (reshaped_features, valid_indices)
            - reshaped_features: (n_valid_pixels, n_features)
            - valid_indices: indices of valid pixels for reconstruction
        """
        if self.feature_stack is None:
            raise ValueError("Features not engineered yet")

        # Get dimensions
        n_features, height, width = self.feature_stack.shape

        # Reshape to (H*W, n_features)
        features_2d = self.feature_stack.reshape(n_features, -1).T

        # Extract only valid pixels
        valid_features = features_2d[self.valid_mask.ravel()]

        # Get indices of valid pixels
        valid_indices = np.where(self.valid_mask.ravel())[0]

        logger.info(f"\nReshaped features for prediction:")
        logger.info(f"  - Original shape: {self.feature_stack.shape}")
        logger.info(f"  - Reshaped shape: {features_2d.shape}")
        logger.info(f"  - Valid pixels shape: {valid_features.shape}")

        return valid_features, valid_indices


def main():
    """Main function to test feature engineering"""
    logger.info("Testing Step 3: Feature Engineering")

    # Import data loader
    from step1_2_setup_and_load_data import DataLoader

    # Load data
    loader = DataLoader()
    data = loader.load_all()

    # Create feature engineer
    engineer = FeatureEngineering()

    # Engineer features
    feature_stack, valid_mask = engineer.engineer_features(
        data['s2_before'],
        data['s2_after'],
        data['s1_before'],
        data['s1_after']
    )

    # Print summary
    summary = engineer.get_feature_summary()
    logger.info(f"\nFeature Summary:")
    logger.info(f"  - Total features: {summary['n_features']}")
    logger.info(f"  - Feature shape: {summary['feature_shape']}")
    logger.info(f"  - Valid pixels: {summary['valid_pixels']:,}")
    logger.info(f"  - Valid percentage: {summary['valid_percentage']:.2f}%")

    return feature_stack, valid_mask, engineer


if __name__ == "__main__":
    feature_stack, valid_mask, engineer = main()
