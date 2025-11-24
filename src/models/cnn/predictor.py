"""
Full raster prediction module for CNN
Predict deforestation for entire study area
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import logging
from pathlib import Path
import rasterio
from rasterio.transform import Affine

from .patch_extractor import extract_patches_for_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RasterPredictor:
    """
    Predict deforestation on full raster using trained CNN
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        patch_size: int = 3,
        batch_size: int = 1000
    ):
        """
        Initialize RasterPredictor

        Args:
            model: Trained CNN model
            device: 'cuda' or 'cpu'
            patch_size: Size of patches
            batch_size: Batch size for prediction
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.multiclass_map = None  # 4-class map (0, 1, 2, 3)

        logger.info(f"RasterPredictor initialized on device: {self.device}")

    def predict_raster(
        self,
        feature_stack: np.ndarray,
        valid_mask: np.ndarray,
        stride: int = 1,
        normalize: bool = True,
        normalization_stats: dict = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Predict on full raster

        Args:
            feature_stack: Feature array (n_features, height, width)
            valid_mask: Valid pixel mask (height, width)
            stride: Stride for sliding window (1 = dense prediction)
            normalize: Whether to normalize patches
            normalization_stats: Pre-computed normalization statistics (mean, std)
            temperature: Temperature scaling for calibration (>1 = softer probabilities)

        Returns:
            multiclass_map: (height, width) 4-class map (0=Forest Stable, 1=Deforestation, 2=Non-forest, 3=Reforestation)
        """
        logger.info(f"\n{'='*70}")
        logger.info("PREDICTING FULL RASTER WITH CNN")
        logger.info(f"{'='*70}")

        n_features, height, width = feature_stack.shape
        logger.info(f"Raster shape: {height} x {width}")
        logger.info(f"Patch size: {self.patch_size}x{self.patch_size}")
        logger.info(f"Stride: {stride}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Temperature: {temperature} {'(calibrated)' if temperature > 1.0 else '(normal)'}")

        # Initialize output map
        self.multiclass_map = np.zeros((height, width), dtype=np.uint8)  # 4-class predictions

        # Extract patches using sliding window
        logger.info("\nExtracting patches...")
        patches, coordinates = extract_patches_for_prediction(
            feature_stack,
            patch_size=self.patch_size,
            stride=stride,
            valid_mask=valid_mask
        )

        if len(patches) == 0:
            logger.warning("No valid patches extracted!")
            return self.multiclass_map

        # Normalize if requested
        if normalize:
            logger.info("Normalizing patches...")
            if normalization_stats is None:
                raise ValueError(
                    "normalization_stats is required when normalize=True. "
                    "Please provide the normalization statistics computed from training data. "
                    "Computing statistics from test/prediction data would cause data leakage."
                )

            # Use provided normalization stats (from training)
            mean = np.array(normalization_stats['mean']).reshape(1, 1, 1, -1)
            std = np.array(normalization_stats['std']).reshape(1, 1, 1, -1)
            logger.info("Using training normalization statistics")
            patches = (patches - mean) / (std + 1e-8)

        # Predict in batches (OPTIMIZED for GPU utilization)
        logger.info(f"\nPredicting {len(patches):,} patches...")

        # Pre-allocate output array for predictions only
        all_multiclass_preds = np.empty(len(patches), dtype=np.uint8)  # 4-class predictions

        dataset = PatchDataset(patches)

        # OPTIMIZATION: Increase num_workers for parallel data loading
        # This keeps GPU fed while CPU prepares next batch
        num_workers = 2 if self.device.type == 'cuda' else 0

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=num_workers > 0  # Keep workers alive
        )

        batch_start_idx = 0
        total_batches = len(loader)
        log_interval = max(1, total_batches // 10)  # Log 10 times during prediction

        with torch.no_grad():
            for batch_idx, batch_patches in enumerate(loader):
                batch_patches = batch_patches.to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(batch_patches)

                # Apply temperature scaling for calibration
                if temperature != 1.0:
                    scaled_outputs = outputs / temperature
                    probs = torch.softmax(scaled_outputs, dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)

                # Multiclass predictions (4 classes: 0=Forest Stable, 1=Deforestation, 2=Non-forest, 3=Reforestation)
                _, multiclass_preds = probs.max(1)

                # Convert to numpy efficiently
                batch_size = len(batch_patches)
                all_multiclass_preds[batch_start_idx:batch_start_idx + batch_size] = multiclass_preds.cpu().numpy()

                batch_start_idx += batch_size

                # Log progress
                if (batch_idx + 1) % log_interval == 0:
                    progress = (batch_idx + 1) / total_batches * 100
                    logger.info(f"  Progress: {batch_idx + 1}/{total_batches} batches ({progress:.1f}%)")

        # Fill output map
        logger.info("Filling output map...")
        for idx, (row, col) in enumerate(coordinates):
            self.multiclass_map[row, col] = all_multiclass_preds[idx]

        # Apply valid mask - set invalid areas to NoData
        logger.info("Applying valid mask...")
        self.multiclass_map[~valid_mask] = 255  # 255 = NoData for uint8

        # Statistics - Multiclass
        total_valid_pixels = np.sum(valid_mask)
        mc_class_0 = np.sum(self.multiclass_map == 0)  # Forest Stable
        mc_class_1 = np.sum(self.multiclass_map == 1)  # Deforestation
        mc_class_2 = np.sum(self.multiclass_map == 2)  # Non-forest
        mc_class_3 = np.sum(self.multiclass_map == 3)  # Reforestation

        logger.info(f"\n{'='*70}")
        logger.info("PREDICTION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total valid pixels: {total_valid_pixels:,}")
        logger.info(f"\n4-Class Predictions:")
        logger.info(f"  Class 0 (Forest Stable): {mc_class_0:,} ({mc_class_0/total_valid_pixels*100:.2f}%)")
        logger.info(f"  Class 1 (Deforestation): {mc_class_1:,} ({mc_class_1/total_valid_pixels*100:.2f}%)")
        logger.info(f"  Class 2 (Non-forest): {mc_class_2:,} ({mc_class_2/total_valid_pixels*100:.2f}%)")
        logger.info(f"  Class 3 (Reforestation): {mc_class_3:,} ({mc_class_3/total_valid_pixels*100:.2f}%)")
        logger.info(f"{'='*70}\n")

        return self.multiclass_map

    def save_rasters(
        self,
        reference_metadata: dict,
        multiclass_path: Path = None
    ):
        """
        Save multiclass raster

        Args:
            reference_metadata: Metadata from reference raster (transform, crs, etc.)
            multiclass_path: Path to save multiclass raster (4 classes)
        """
        logger.info("\nSaving output rasters...")

        # Multiclass raster
        if multiclass_path is not None:
            with rasterio.open(
                multiclass_path,
                'w',
                driver='GTiff',
                height=self.multiclass_map.shape[0],
                width=self.multiclass_map.shape[1],
                count=1,
                dtype=np.uint8,
                crs=reference_metadata['crs'],
                transform=reference_metadata['transform'],
                compress='lzw',
                nodata=255
            ) as dst:
                dst.write(self.multiclass_map, 1)
                dst.set_band_description(1, 'Multiclass (0=Forest Stable, 1=Deforestation, 2=Non-forest, 3=Reforestation, 255=NoData)')

            logger.info(f"  Multiclass raster saved: {multiclass_path}")


class PatchDataset(Dataset):
    """Simple dataset for patches without labels"""

    def __init__(self, patches: np.ndarray):
        self.patches = torch.FloatTensor(patches)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]
