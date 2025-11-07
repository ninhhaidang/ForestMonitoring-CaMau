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

        self.classification_map = None
        self.probability_map = None

        logger.info(f"RasterPredictor initialized on device: {self.device}")

    def predict_raster(
        self,
        feature_stack: np.ndarray,
        valid_mask: np.ndarray,
        stride: int = 1,
        normalize: bool = True,
        normalization_stats: dict = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            Tuple of (classification_map, probability_map)
            - classification_map: (height, width) binary map
            - probability_map: (height, width) probability map
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

        # Initialize output maps
        self.classification_map = np.zeros((height, width), dtype=np.uint8)
        self.probability_map = np.zeros((height, width), dtype=np.float32)

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
            return self.classification_map, self.probability_map

        # Normalize if requested
        if normalize:
            logger.info("Normalizing patches...")
            if normalization_stats is not None:
                # Use provided normalization stats (from training)
                mean = normalization_stats['mean']
                std = normalization_stats['std']
                logger.info("Using training normalization statistics")
            else:
                # Compute from current patches
                mean = patches.mean(axis=(0, 1, 2), keepdims=True)
                std = patches.std(axis=(0, 1, 2), keepdims=True)
                logger.info("Computing normalization from prediction patches")
            patches = (patches - mean) / (std + 1e-8)

        # Predict in batches
        logger.info(f"\nPredicting {len(patches)} patches...")

        all_predictions = []
        all_probabilities = []

        dataset = PatchDataset(patches)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        with torch.no_grad():
            for batch_patches in loader:
                batch_patches = batch_patches.to(self.device)

                # Forward pass
                outputs = self.model(batch_patches)

                # Apply temperature scaling for calibration
                if temperature != 1.0:
                    scaled_outputs = outputs / temperature
                    probs = torch.softmax(scaled_outputs, dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)

                _, preds = probs.max(1)

                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Fill output maps
        logger.info("Filling output maps...")
        for idx, (row, col) in enumerate(coordinates):
            self.classification_map[row, col] = all_predictions[idx]
            self.probability_map[row, col] = all_probabilities[idx]

        # Apply valid mask - set invalid areas to NoData
        logger.info("Applying valid mask...")
        self.classification_map[~valid_mask] = 255  # 255 = NoData for uint8
        self.probability_map[~valid_mask] = -9999  # NoData for float32

        # Statistics
        deforestation_pixels = np.sum(self.classification_map == 1)
        total_valid_pixels = np.sum(valid_mask)
        deforestation_percentage = deforestation_pixels / total_valid_pixels * 100

        logger.info(f"\n{'='*70}")
        logger.info("PREDICTION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total valid pixels: {total_valid_pixels:,}")
        logger.info(f"Deforestation pixels: {deforestation_pixels:,} ({deforestation_percentage:.2f}%)")
        logger.info(f"No deforestation pixels: {total_valid_pixels - deforestation_pixels:,} ({100-deforestation_percentage:.2f}%)")
        logger.info(f"{'='*70}\n")

        return self.classification_map, self.probability_map

    def save_rasters(
        self,
        classification_path: Path,
        probability_path: Path,
        reference_metadata: dict
    ):
        """
        Save classification and probability rasters

        Args:
            classification_path: Path to save classification raster
            probability_path: Path to save probability raster
            reference_metadata: Metadata from reference raster (transform, crs, etc.)
        """
        logger.info("\nSaving output rasters...")

        # Classification raster (uint8) with NoData
        with rasterio.open(
            classification_path,
            'w',
            driver='GTiff',
            height=self.classification_map.shape[0],
            width=self.classification_map.shape[1],
            count=1,
            dtype=np.uint8,
            crs=reference_metadata['crs'],
            transform=reference_metadata['transform'],
            compress='lzw',
            nodata=255  # Set NoData value
        ) as dst:
            dst.write(self.classification_map, 1)
            dst.set_band_description(1, 'Deforestation Classification (0=No, 1=Yes, 255=NoData)')

        logger.info(f"  Classification raster saved: {classification_path}")

        # Probability raster (float32) with NoData
        with rasterio.open(
            probability_path,
            'w',
            driver='GTiff',
            height=self.probability_map.shape[0],
            width=self.probability_map.shape[1],
            count=1,
            dtype=np.float32,
            crs=reference_metadata['crs'],
            transform=reference_metadata['transform'],
            compress='lzw',
            nodata=-9999  # Set NoData value
        ) as dst:
            dst.write(self.probability_map, 1)
            dst.set_band_description(1, 'Deforestation Probability (0.0-1.0, -9999=NoData)')

        logger.info(f"  Probability raster saved: {probability_path}")


class PatchDataset(Dataset):
    """Simple dataset for patches without labels"""

    def __init__(self, patches: np.ndarray):
        self.patches = torch.FloatTensor(patches)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]
