"""
Custom transforms for loading 9-channel TIFF files
Author: Ninh Hai Dang (21021411)
Date: 2025-10-17
"""
import numpy as np
import rasterio
from mmcv.transforms import BaseTransform
from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiImgLoadRasterioFromFile(BaseTransform):
    """
    Load multi-channel TIFF images using rasterio instead of OpenCV

    This is needed because OpenCV's TIFF decoder only supports up to 4 channels,
    but our data has 9 channels per time step.

    Args:
        to_float32 (bool): Whether to convert to float32. Default: True.
    """

    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def transform(self, results: dict) -> dict:
        """Load images using rasterio"""

        # Get image paths
        # Open-CD datasets provide img_path as a list: [path_from, path_to]
        img_paths = results.get('img_path', None)

        if img_paths is None:
            raise ValueError("img_path not found in results")

        if not isinstance(img_paths, (list, tuple)) or len(img_paths) != 2:
            raise ValueError(f"img_path should be a list of 2 paths, got {img_paths}")

        img_path_from, img_path_to = img_paths

        # Load image 1 (Time 1)
        try:
            with rasterio.open(img_path_from) as src:
                img_from = src.read()  # Shape: (C, H, W)
        except Exception as e:
            raise RuntimeError(f"Failed to load {img_path_from}: {e}")

        # Load image 2 (Time 2)
        try:
            with rasterio.open(img_path_to) as src:
                img_to = src.read()  # Shape: (C, H, W)
        except Exception as e:
            raise RuntimeError(f"Failed to load {img_path_to}: {e}")

        # Convert to (H, W, C) for Open-CD
        img_from = np.transpose(img_from, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        img_to = np.transpose(img_to, (1, 2, 0))

        # Convert to float32 if needed
        if self.to_float32:
            img_from = img_from.astype(np.float32)
            img_to = img_to.astype(np.float32)

        # Store results
        imgs = [img_from, img_to]
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]  # (H, W)
        results['ori_shape'] = imgs[0].shape[:2]

        # Set default values
        num_channels = imgs[0].shape[2] if len(imgs[0].shape) == 3 else 1
        results['num_channels'] = num_channels
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32})'
        return repr_str
