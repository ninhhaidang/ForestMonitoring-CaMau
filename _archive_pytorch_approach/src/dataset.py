"""
PyTorch Dataset for Forest Change Detection
Loads 18-channel patches (9 bands × 2 timestamps) from GeoTIFF files
"""

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ForestChangeDataset(Dataset):
    """
    Dataset for forest change detection using multi-sensor data

    Input: 4 GeoTIFF files + CSV with coordinates
    Output: 18-channel tensor (B, 18, H, W)
    """

    def __init__(
        self,
        csv_path,
        s1_t1_path,
        s1_t2_path,
        s2_t1_path,
        s2_t2_path,
        patch_size=256,
        transform=None,
        mode='train'
    ):
        """
        Args:
            csv_path: Path to CSV with columns [id, label, x, y]
            s1_t1_path: Sentinel-1 Time 1 (2 bands: VH, Ratio)
            s1_t2_path: Sentinel-1 Time 2 (2 bands: VH, Ratio)
            s2_t1_path: Sentinel-2 Time 1 (7 bands)
            s2_t2_path: Sentinel-2 Time 2 (7 bands)
            patch_size: Size of extracted patches (default: 256)
            transform: Albumentations transform
            mode: 'train', 'val', or 'test'
        """
        self.df = pd.read_csv(csv_path)
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode

        # Store paths (don't open files here for multiprocessing compatibility)
        self.s1_t1_path = str(s1_t1_path)
        self.s1_t2_path = str(s1_t2_path)
        self.s2_t1_path = str(s2_t1_path)
        self.s2_t2_path = str(s2_t2_path)

        print(f"Dataset initialized: {len(self.df)} samples")
        print(f"Patch size: {patch_size}×{patch_size}")
        print(f"Total channels: 18 (9 per timestamp)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (18, H, W) - 18-channel patch
            label: int - 0 (no change) or 1 (change)
        """
        row = self.df.iloc[idx]
        x, y = row['x'], row['y']
        label = int(row['label'])  # Ensure label is int

        # Open files (for multiprocessing compatibility)
        with rasterio.open(self.s1_t1_path) as s1_t1, \
             rasterio.open(self.s1_t2_path) as s1_t2, \
             rasterio.open(self.s2_t1_path) as s2_t1, \
             rasterio.open(self.s2_t2_path) as s2_t2:

            # Convert geographic coordinates to pixel coordinates
            # Assuming all images have the same geotransform
            py, px = s1_t1.index(x, y)

            # Calculate window bounds
            half_size = self.patch_size // 2
            window = rasterio.windows.Window(
                px - half_size,
                py - half_size,
                self.patch_size,
                self.patch_size
            )

            # Read patches from all 4 files
            s1_t1_patch = s1_t1.read(window=window)  # (2, H, W)
            s1_t2_patch = s1_t2.read(window=window)  # (2, H, W)
            s2_t1_patch = s2_t1.read(window=window)  # (7, H, W)
            s2_t2_patch = s2_t2.read(window=window)  # (7, H, W)

        # Concatenate: Time 1 (9 bands) + Time 2 (9 bands) = 18 bands
        patch = np.concatenate([
            s1_t1_patch,  # 2 bands
            s2_t1_patch,  # 7 bands
            s1_t2_patch,  # 2 bands
            s2_t2_patch   # 7 bands
        ], axis=0)  # (18, H, W)

        # Handle NaN and Inf values
        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

        # Normalize (0-1 range)
        patch = self._normalize(patch)

        # Convert to (H, W, C) for albumentations
        patch = np.transpose(patch, (1, 2, 0))  # (H, W, 18)

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=patch)
            patch = augmented['image']
        else:
            # Convert to tensor
            patch = torch.from_numpy(patch).permute(2, 0, 1).float()

        return patch, label

    def _normalize(self, patch):
        """
        Normalize each band to [0, 1] range
        """
        # Clip extreme values (percentile-based)
        for i in range(patch.shape[0]):
            band = patch[i]
            p_low, p_high = np.percentile(band[band != 0], [2, 98])
            patch[i] = np.clip(band, p_low, p_high)

            # Min-max normalization
            band_min = patch[i].min()
            band_max = patch[i].max()
            if band_max > band_min:
                patch[i] = (patch[i] - band_min) / (band_max - band_min)

        return patch


def get_transforms(mode='train'):
    """
    Get albumentations transforms for different modes

    Args:
        mode: 'train', 'val', or 'test'

    Returns:
        albumentations.Compose
    """
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])


def create_data_loaders(
    csv_path,
    s1_t1_path,
    s1_t2_path,
    s2_t1_path,
    s2_t2_path,
    batch_size=16,
    num_workers=4,
    patch_size=256,
    train_split=0.8,
    val_split=0.1
):
    """
    Create train/val/test dataloaders from CSV

    Args:
        csv_path: Path to CSV with all points
        s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path: Paths to TIFF files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        patch_size: Size of patches
        train_split: Fraction for training (default: 0.8)
        val_split: Fraction for validation (default: 0.1)

    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split

    # Load CSV
    df = pd.read_csv(csv_path)

    # Split: train (80%), val (10%), test (10%)
    train_df, test_df = train_test_split(
        df,
        test_size=(1 - train_split),
        stratify=df['label'],
        random_state=42
    )

    val_df, test_df = train_test_split(
        test_df,
        test_size=0.5,
        stratify=test_df['label'],
        random_state=42
    )

    # Save splits to CSV
    output_dir = Path(csv_path).parent
    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'val_split.csv', index=False)
    test_df.to_csv(output_dir / 'test_split.csv', index=False)

    print(f"\n📊 Data splits:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    # Create datasets
    train_dataset = ForestChangeDataset(
        output_dir / 'train_split.csv',
        s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path,
        patch_size=patch_size,
        transform=get_transforms('train'),
        mode='train'
    )

    val_dataset = ForestChangeDataset(
        output_dir / 'val_split.csv',
        s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path,
        patch_size=patch_size,
        transform=get_transforms('val'),
        mode='val'
    )

    test_dataset = ForestChangeDataset(
        output_dir / 'test_split.csv',
        s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path,
        patch_size=patch_size,
        transform=get_transforms('test'),
        mode='test'
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
