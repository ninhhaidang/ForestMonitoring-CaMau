"""
STEP 1 & 2: Setup & Configuration + Load Data
Load Sentinel-1, Sentinel-2, Ground Truth, and Boundary data
"""

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional

from .config import (
    S1_BEFORE, S1_AFTER, S2_BEFORE, S2_AFTER,
    GROUND_TRUTH_CSV, BOUNDARY_SHP,
    S1_BANDS, S2_BANDS, GROUND_TRUTH_CONFIG,
    LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class to handle loading all input data"""

    def __init__(self):
        """Initialize DataLoader"""
        self.s2_before = None
        self.s2_after = None
        self.s1_before = None
        self.s1_after = None
        self.ground_truth = None
        self.boundary = None
        self.metadata = {}

    def load_raster(self, raster_path: Path, name: str) -> Tuple[np.ndarray, dict]:
        """
        Load a raster file and return data array and metadata

        Args:
            raster_path: Path to raster file
            name: Name for logging

        Returns:
            Tuple of (data array, metadata dict)
        """
        logger.info(f"Loading {name}: {raster_path}")

        try:
            with rasterio.open(raster_path) as src:
                # Read all bands
                data = src.read()  # Shape: (bands, height, width)

                # Get metadata
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0] if src.count > 0 else 'float32',  # Get dtype of first band
                    'nodata': src.nodata,
                    'bounds': src.bounds
                }

                logger.info(f"  [OK] Shape: {data.shape}")
                logger.info(f"  [OK] Bands: {metadata['count']}")
                logger.info(f"  [OK] CRS: {metadata['crs']}")
                logger.info(f"  [OK] NoData: {metadata['nodata']}")

                return data, metadata

        except Exception as e:
            logger.error(f"  [FAIL] Failed to load {name}: {str(e)}")
            raise

    def load_sentinel2(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Sentinel-2 before and after images

        Returns:
            Tuple of (before_data, after_data)
        """
        logger.info("\n" + "="*70)
        logger.info("LOADING SENTINEL-2 DATA (OPTICAL)")
        logger.info("="*70)

        # Load before image
        self.s2_before, s2_before_meta = self.load_raster(
            S2_BEFORE, "Sentinel-2 Before (2024-01-30)"
        )
        self.metadata['s2_before'] = s2_before_meta

        # Load after image
        self.s2_after, s2_after_meta = self.load_raster(
            S2_AFTER, "Sentinel-2 After (2025-02-28)"
        )
        self.metadata['s2_after'] = s2_after_meta

        # Validate
        assert self.s2_before.shape == self.s2_after.shape, \
            "Sentinel-2 before and after images must have same shape"
        assert self.s2_before.shape[0] == S2_BANDS['count'], \
            f"Expected {S2_BANDS['count']} bands, got {self.s2_before.shape[0]}"

        logger.info(f"\n[OK] Sentinel-2 data loaded successfully")
        logger.info(f"  - Expected bands: {S2_BANDS['names']}")

        return self.s2_before, self.s2_after

    def load_sentinel1(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Sentinel-1 before and after images

        Returns:
            Tuple of (before_data, after_data)
        """
        logger.info("\n" + "="*70)
        logger.info("LOADING SENTINEL-1 DATA (SAR)")
        logger.info("="*70)

        # Load before image
        self.s1_before, s1_before_meta = self.load_raster(
            S1_BEFORE, "Sentinel-1 Before (2024-02-04)"
        )
        self.metadata['s1_before'] = s1_before_meta

        # Load after image
        self.s1_after, s1_after_meta = self.load_raster(
            S1_AFTER, "Sentinel-1 After (2025-02-22)"
        )
        self.metadata['s1_after'] = s1_after_meta

        # Validate
        assert self.s1_before.shape == self.s1_after.shape, \
            "Sentinel-1 before and after images must have same shape"
        assert self.s1_before.shape[0] == S1_BANDS['count'], \
            f"Expected {S1_BANDS['count']} bands, got {self.s1_before.shape[0]}"

        logger.info(f"\n[OK] Sentinel-1 data loaded successfully")
        logger.info(f"  - Expected bands: {S1_BANDS['names']}")

        return self.s1_before, self.s1_after

    def load_ground_truth(self) -> pd.DataFrame:
        """
        Load ground truth training points from CSV

        Returns:
            DataFrame with ground truth points
        """
        logger.info("\n" + "="*70)
        logger.info("LOADING GROUND TRUTH POINTS")
        logger.info("="*70)

        try:
            # Load CSV
            logger.info(f"Loading: {GROUND_TRUTH_CSV}")
            self.ground_truth = pd.read_csv(GROUND_TRUTH_CSV)

            # Validate columns
            required_cols = [
                GROUND_TRUTH_CONFIG['id_column'],
                GROUND_TRUTH_CONFIG['label_column'],
                GROUND_TRUTH_CONFIG['x_column'],
                GROUND_TRUTH_CONFIG['y_column']
            ]

            missing_cols = [col for col in required_cols if col not in self.ground_truth.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get statistics
            total_points = len(self.ground_truth)
            class_counts = self.ground_truth[GROUND_TRUTH_CONFIG['label_column']].value_counts()

            logger.info(f"\n[OK] Ground truth loaded successfully")
            logger.info(f"  - Total points: {total_points}")
            logger.info(f"  - Class 0 (No deforestation): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total_points*100:.1f}%)")
            logger.info(f"  - Class 1 (Deforestation): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total_points*100:.1f}%)")
            logger.info(f"  - Columns: {list(self.ground_truth.columns)}")

            return self.ground_truth

        except Exception as e:
            logger.error(f"[FAIL] Failed to load ground truth: {str(e)}")
            raise

    def load_boundary(self) -> gpd.GeoDataFrame:
        """
        Load boundary shapefile

        Returns:
            GeoDataFrame with boundary
        """
        logger.info("\n" + "="*70)
        logger.info("LOADING BOUNDARY SHAPEFILE")
        logger.info("="*70)

        try:
            # Load shapefile
            logger.info(f"Loading: {BOUNDARY_SHP}")
            self.boundary = gpd.read_file(BOUNDARY_SHP)

            logger.info(f"\n[OK] Boundary loaded successfully")
            logger.info(f"  - CRS: {self.boundary.crs}")
            logger.info(f"  - Features: {len(self.boundary)}")
            logger.info(f"  - Bounds: {self.boundary.total_bounds}")

            return self.boundary

        except Exception as e:
            logger.error(f"[FAIL] Failed to load boundary: {str(e)}")
            raise

    def load_all(self) -> Dict:
        """
        Load all data at once

        Returns:
            Dictionary containing all loaded data
        """
        logger.info("\n" + "="*70)
        logger.info("LOADING ALL DATA")
        logger.info("="*70)

        # Load all data
        self.load_sentinel2()
        self.load_sentinel1()
        self.load_ground_truth()
        self.load_boundary()

        logger.info("\n" + "="*70)
        logger.info("[SUCCESS] ALL DATA LOADED SUCCESSFULLY")
        logger.info("="*70)

        return {
            's2_before': self.s2_before,
            's2_after': self.s2_after,
            's1_before': self.s1_before,
            's1_after': self.s1_after,
            'ground_truth': self.ground_truth,
            'boundary': self.boundary,
            'metadata': self.metadata
        }

    def get_spatial_info(self) -> Dict:
        """
        Get spatial information summary

        Returns:
            Dictionary with spatial information
        """
        if not self.metadata:
            logger.warning("No data loaded yet")
            return {}

        # Use Sentinel-2 as reference
        s2_meta = self.metadata.get('s2_before', {})

        spatial_info = {
            'crs': str(s2_meta.get('crs', 'Unknown')),
            'width': s2_meta.get('width', 0),
            'height': s2_meta.get('height', 0),
            'bounds': s2_meta.get('bounds', None),
            'transform': s2_meta.get('transform', None),
            'resolution': S2_BANDS['resolution']
        }

        return spatial_info

    def print_summary(self):
        """Print summary of loaded data"""
        logger.info("\n" + "="*70)
        logger.info("DATA SUMMARY")
        logger.info("="*70)

        if self.s2_before is not None:
            logger.info(f"\nSentinel-2:")
            logger.info(f"  - Shape: {self.s2_before.shape}")
            logger.info(f"  - Bands: {S2_BANDS['names']}")
            logger.info(f"  - Resolution: {S2_BANDS['resolution']}m")

        if self.s1_before is not None:
            logger.info(f"\nSentinel-1:")
            logger.info(f"  - Shape: {self.s1_before.shape}")
            logger.info(f"  - Bands: {S1_BANDS['names']}")
            logger.info(f"  - Resolution: {S1_BANDS['resolution']}m")

        if self.ground_truth is not None:
            logger.info(f"\nGround Truth:")
            logger.info(f"  - Total points: {len(self.ground_truth)}")
            class_counts = self.ground_truth[GROUND_TRUTH_CONFIG['label_column']].value_counts()
            logger.info(f"  - Class distribution: {dict(class_counts)}")

        if self.boundary is not None:
            logger.info(f"\nBoundary:")
            logger.info(f"  - Features: {len(self.boundary)}")
            logger.info(f"  - CRS: {self.boundary.crs}")

        spatial_info = self.get_spatial_info()
        if spatial_info:
            logger.info(f"\nSpatial Information:")
            logger.info(f"  - CRS: {spatial_info['crs']}")
            logger.info(f"  - Dimensions: {spatial_info['width']} x {spatial_info['height']}")
            logger.info(f"  - Resolution: {spatial_info['resolution']}m")

        logger.info("="*70)


def main():
    """Main function to test data loading"""
    logger.info("Testing Step 1 & 2: Setup and Load Data")

    # Create data loader
    loader = DataLoader()

    # Load all data
    data = loader.load_all()

    # Print summary
    loader.print_summary()

    return data, loader


if __name__ == "__main__":
    data, loader = main()
