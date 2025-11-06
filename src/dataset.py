"""
PyTorch Dataset for deforestation detection
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
from pathlib import Path

from .config import *


class DeforestationDataset(Dataset):
    """PyTorch Dataset for deforestation patches"""

    def __init__(self, patches, labels, transform=None):
        """
        Args:
            patches: Numpy array of shape (N, C, H, W)
            labels: Numpy array of shape (N,)
            transform: Optional transforms (albumentations)
        """
        self.patches = patches
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]  # (C, H, W)
        label = self.labels[idx]

        # Convert to torch tensors
        patch = torch.from_numpy(patch).float()
        label = torch.tensor(label, dtype=torch.long)

        # Apply transforms if provided (for augmentation)
        if self.transform:
            # Albumentations expects (H, W, C) format
            patch_np = patch.permute(1, 2, 0).numpy()
            augmented = self.transform(image=patch_np)
            patch = torch.from_numpy(augmented['image']).permute(2, 0, 1).float()

        return patch, label


def load_patches(patches_file):
    """
    Load patches from pickle file

    Args:
        patches_file: Path to patches pickle file

    Returns:
        patches, labels
    """
    with open(patches_file, 'rb') as f:
        data = pickle.load(f)

    patches = data['patches']
    labels = data['labels']

    print(f"Loaded {len(patches)} patches from {patches_file}")
    print(f"  - Shape: {patches.shape}")
    print(f"  - Labels: {np.unique(labels, return_counts=True)}")

    return patches, labels


def create_dataloaders(patches, labels, batch_size=4, num_workers=4,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       random_seed=42):
    """
    Create train, validation, and test dataloaders

    Args:
        patches: Numpy array of patches
        labels: Numpy array of labels
        batch_size: Batch size
        num_workers: Number of workers for data loading
        train_ratio, val_ratio, test_ratio: Split ratios
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create dataset
    dataset = DeforestationDataset(patches, labels)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f"\nDataset split:")
    print(f"  - Train: {train_size} ({train_ratio*100:.0f}%)")
    print(f"  - Val: {val_size} ({val_ratio*100:.0f}%)")
    print(f"  - Test: {test_size} ({test_ratio*100:.0f}%)")

    # Split dataset
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
